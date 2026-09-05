// Copyright © 2026 Apple Inc.

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "jaccl/mesh_impl.h"

// A fake verbs provider, not a replacement mesh implementation. These tests
// execute the production MeshImpl, Connection and SharedBuffer code without
// RDMA devices or a peer process. A CTest process timeout catches regressions
// that would otherwise busy-wait forever.
namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

struct Work {
  uint64_t id;
  void* data;
  size_t bytes;
  bool receive;
};

enum class Mode { Silent, Complete, ReceiveOnly, Delayed, PostError };

struct Device {
  ibv_context context{};
  ibv_pd pd{};
  ibv_cq cq{};
  ibv_qp qp{};
  std::deque<Work> pending;
  Mode mode = Mode::Silent;
  int posted = 0;
  int destroyed = 0;
  int deregistered_while_active = 0;
  bool active = true;
};

Device& device(ibv_cq* cq) {
  return *static_cast<Device*>(cq->cq_context);
}

Device& device(ibv_qp* qp) {
  return *static_cast<Device*>(qp->qp_context);
}

int post_send(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr**) {
  auto& d = device(qp);
  ++d.posted;
  if (d.mode == Mode::PostError) {
    return EIO;
  }
  d.pending.push_back({wr->wr_id, nullptr, 0, false});
  return 0;
}

int post_recv(ibv_qp* qp, ibv_recv_wr* wr, ibv_recv_wr**) {
  auto& d = device(qp);
  ++d.posted;
  if (d.mode == Mode::PostError) {
    return EIO;
  }
  d.pending.push_back(
      {wr->wr_id,
       reinterpret_cast<void*>(wr->sg_list[0].addr),
       wr->sg_list[0].length,
       true});
  return 0;
}

int poll_cq(ibv_cq* cq, int maximum, ibv_wc* completions) {
  auto& d = device(cq);
  if (d.mode == Mode::Silent) {
    return 0;
  }
  int count = 0;
  while (count < maximum && !d.pending.empty()) {
    auto it = d.pending.begin();
    if (d.mode == Mode::ReceiveOnly) {
      it = std::find_if(
          it, d.pending.end(), [](const Work& w) { return w.receive; });
      if (it == d.pending.end()) {
        break;
      }
    }
    if (d.mode == Mode::Delayed) {
      std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    const auto work = *it;
    d.pending.erase(it);
    if (work.receive) {
      std::fill_n(
          static_cast<float*>(work.data), work.bytes / sizeof(float), 3.0f);
    }
    completions[count] = {};
    completions[count].wr_id = work.id;
    completions[count].status = IBV_WC_SUCCESS;
    ++count;
    if (d.mode == Mode::Delayed) {
      break;
    }
  }
  return count;
}

ibv_mr* reg_mr(ibv_pd* pd, void* data, size_t size, int) {
  auto* mr = new ibv_mr{};
  mr->pd = pd;
  mr->addr = data;
  mr->length = size;
  mr->lkey = 1;
  return mr;
}

int dereg_mr(ibv_mr* mr) {
  // context is the first Device member, so its address identifies the device.
  auto& d = *reinterpret_cast<Device*>(mr->pd->context);
  if (d.active) {
    ++d.deregistered_while_active;
  }
  delete mr;
  return 0;
}

int destroy_qp(ibv_qp* qp) {
  auto& d = device(qp);
  ++d.destroyed;
  d.active = false;
  d.pending.clear();
  return 0;
}

struct Fixture {
  Device d;
  std::vector<jaccl::Connection> connections;
  std::vector<jaccl::SharedBuffer> buffers;
  std::vector<jaccl::SharedBuffer> scatter;
  std::unique_ptr<jaccl::MeshImpl> mesh;

  explicit Fixture(int size = 2) {
    auto& verbs = jaccl::ibv();
    verbs.reg_mr = reg_mr;
    verbs.dereg_mr = dereg_mr;
    verbs.destroy_qp = destroy_qp;
    verbs.destroy_cq = [](ibv_cq*) { return 0; };
    verbs.dealloc_pd = [](ibv_pd*) { return 0; };
    verbs.close_device = [](ibv_context*) { return 0; };
    d.context.ops.poll_cq = poll_cq;
    d.context.ops.post_send = post_send;
    d.context.ops.post_recv = post_recv;
    d.pd.context = &d.context;
    d.cq.context = &d.context;
    d.cq.cq_context = &d;
    d.qp.context = &d.context;
    d.qp.qp_context = &d;
    connections.emplace_back(nullptr);
    if (size > 1) {
      auto& c = connections.emplace_back(&d.context);
      c.protection_domain = &d.pd;
      c.completion_queue = &d.cq;
      c.queue_pair = &d.qp;
    }
    for (int k = 0; k < BUFFER_SIZES; ++k) {
      for (int b = 0; b < NUM_BUFFERS; ++b) {
        for (int p = 0; p < size; ++p) {
          buffers.emplace_back(FRAME_SIZE * (1 << k));
          if (size > 1) {
            buffers.back().register_to_protection_domain(&d.pd);
          }
        }
        for (int p = 0; p < 2 * size; ++p) {
          scatter.emplace_back(FRAME_SIZE * (1 << k));
          if (size > 1) {
            scatter.back().register_to_protection_domain(&d.pd);
          }
        }
      }
    }
    mesh = std::make_unique<jaccl::MeshImpl>(
        0, size, connections, buffers, scatter);
  }

  ~Fixture() {
    // Also works against unmodified main for before/after testing.
    for (auto& c : connections) {
      if (c.queue_pair != nullptr) {
        destroy_qp(c.queue_pair);
        c.queue_pair = nullptr;
      }
    }
  }
};

struct Sum {
  void operator()(const float* src, float* dst, int64_t n) const {
    for (int64_t i = 0; i < n; ++i) {
      dst[i] += src[i];
    }
  }
};

template <typename Fn>
void expect_error(Fn&& fn, const char* text) {
  try {
    fn();
  } catch (const std::exception& error) {
    require(
        std::string(error.what()).find(text) != std::string::npos,
        error.what());
    return;
  }
  throw std::runtime_error("operation did not throw");
}

void run_operation(
    Fixture& f,
    const std::string& name,
    std::vector<float>& in,
    std::vector<float>& out) {
  if (name == "recv" || name == "poison" || name == "post_error") {
    f.mesh->recv(reinterpret_cast<char*>(out.data()), 4, 1);
  } else if (name == "send") {
    f.mesh->send(reinterpret_cast<const char*>(in.data()), 4, 1);
  } else if (name == "all_reduce" || name == "drain") {
    f.mesh->all_reduce(in.data(), out.data(), 1, Sum{});
  } else if (name == "all_gather") {
    f.mesh->all_gather(
        reinterpret_cast<const char*>(in.data()),
        reinterpret_cast<char*>(out.data()),
        4);
  } else if (name == "sum_scatter") {
    f.mesh->sum_scatter(in.data(), out.data(), 1, Sum{});
  } else if (name == "scatter_gather") {
    f.mesh->all_reduce_scatter_gather(in.data(), out.data(), 2, Sum{});
  } else {
    throw std::runtime_error("unknown operation");
  }
}

void test_failure(const std::string& name) {
  Fixture f;
  std::vector<float> in(4, 2.0f), out(4, -100.0f);
  if (name == "drain") {
    f.d.mode = Mode::ReceiveOnly;
  } else if (name == "post_error") {
    f.d.mode = Mode::PostError;
  }
  expect_error(
      [&] { run_operation(f, name, in, out); },
      name == "post_error" ? "Recv failed" : "timed out");
  require(f.d.destroyed == 1, "timed-out QP was not closed");
  const int posted = f.d.posted;
  std::fill(out.begin(), out.end(), -100.0f);
  for (const char* operation :
       {"recv",
        "send",
        "all_reduce",
        "all_gather",
        "sum_scatter",
        "scatter_gather"}) {
    expect_error([&] { run_operation(f, operation, in, out); }, "failed mesh");
  }
  expect_error(
      [&] { f.mesh->recv(reinterpret_cast<char*>(out.data()), 0, 1); },
      "failed mesh");
  require(f.d.posted == posted, "failed group submitted more WRs");
  require(
      std::all_of(out.begin(), out.end(), [](float x) { return x == -100; }),
      "failed group wrote to caller output");
  require(f.d.destroyed == 1, "QP was destroyed twice");
  f.scatter.clear();
  f.buffers.clear();
  require(
      f.d.deregistered_while_active == 0, "buffers freed before QP stopped");
}

void test_success(const std::string& name) {
  Fixture f(name == "single_rank" ? 1 : 2);
  f.d.mode = Mode::Complete;
  std::vector<float> in(4, 2.0f), out(4, -100.0f);
  if (name == "single_rank") {
    f.mesh->all_reduce(in.data(), out.data(), 4, Sum{});
    require(out == in, "single-rank reduction changed");
  } else if (name == "empty") {
    f.mesh->recv(reinterpret_cast<char*>(out.data()), 0, 1);
    f.mesh->send(reinterpret_cast<const char*>(in.data()), 0, 1);
    f.mesh->all_reduce(in.data(), out.data(), 0, Sum{});
    f.mesh->all_gather(
        reinterpret_cast<const char*>(in.data()),
        reinterpret_cast<char*>(out.data()),
        0);
    f.mesh->sum_scatter(in.data(), out.data(), 0, Sum{});
    require(f.d.posted == 0, "empty operation submitted work");
  } else if (name == "delayed") {
    f.d.mode = Mode::Delayed;
    const size_t count = 4 * MAX_BUFFER_SIZE / sizeof(float) + 1;
    out.resize(count);
    auto start = std::chrono::steady_clock::now();
    f.mesh->recv(reinterpret_cast<char*>(out.data()), count * sizeof(float), 1);
    require(
        std::chrono::steady_clock::now() - start >
            std::chrono::milliseconds(100),
        "test did not exceed the operation-duration threshold");
    require(
        std::all_of(out.begin(), out.end(), [](float x) { return x == 3; }),
        "progressing recv corrupted data");
  } else {
    f.mesh->all_reduce(in.data(), out.data(), 4, Sum{});
    require(
        std::all_of(out.begin(), out.end(), [](float x) { return x == 5; }),
        "healthy reduction changed");
    f.mesh->recv(reinterpret_cast<char*>(out.data()), 4, 1);
    require(out[0] == 3, "healthy recv changed");
    f.mesh->send(reinterpret_cast<const char*>(in.data()), 4, 1);
  }
  require(f.d.destroyed == 0, "healthy QP was closed");
}

} // namespace

int main(int argc, char** argv) {
  try {
    const std::string name = argc > 1 ? argv[1] : "recv";
    unsetenv("JACCL_TIMEOUT_MS");
    setenv("MLX_JACCL_TIMEOUT_MS", name == "delayed" ? "100" : "10", 1);
    if (name == "healthy" || name == "empty" || name == "single_rank" ||
        name == "delayed") {
      test_success(name);
    } else {
      test_failure(name);
    }
    std::cout << "PASS: " << name << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
