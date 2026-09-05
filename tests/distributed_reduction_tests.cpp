// Copyright © 2026 Apple Inc.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
#include <arm_fp16.h>
#endif

#include "doctest/doctest.h"
#include "jaccl/group.h"
#include "jaccl/reduction_ops.h"
#include "mlx/distributed/reduction_ops.h"
#include "mlx/mlx.h"

namespace {

template <typename T, typename Combine>
void check_nan_reduction(Combine combine, bool maximum) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const std::array<std::array<float, 2>, 9> pairs = {
      {{nan, 5},
       {5, nan},
       {nan, nan},
       {1, 5},
       {-2, -7},
       {inf, -inf},
       {0, -0.0f},
       {2, 2},
       {-inf, -3}}};
  // Include scalar, vector-width boundaries and a non-vector tail.
  for (size_t n : {0, 1, 3, 4, 5, 31, 32, 33, 257}) {
    CAPTURE(n);
    std::vector<T> a(n, T(0.0f));
    std::vector<T> b(n, T(0.0f));
    std::vector<T> output(n + 2, T(123.0f));
    for (size_t i = 0; i < n; ++i) {
      a[i] = T(pairs[i % pairs.size()][0]);
      b[i] = T(pairs[i % pairs.size()][1]);
    }
    combine(a.data(), b.data(), output.data() + 1, n);
    CHECK(static_cast<double>(output.front()) == 123.0);
    CHECK(static_cast<double>(output.back()) == 123.0);
    for (size_t i = 0; i < n; ++i) {
      const auto x = static_cast<double>(a[i]);
      const auto y = static_cast<double>(b[i]);
      const auto actual = static_cast<double>(output[i + 1]);
      if (std::isnan(x) || std::isnan(y)) {
        CHECK(std::isnan(actual));
      } else {
        CHECK(actual == (maximum ? std::max(x, y) : std::min(x, y)));
      }
    }
  }
  // All rank orderings of the reported 1, NaN, 3, 4 failure.
  std::array<int, 4> order = {0, 1, 2, 3};
  const std::array<float, 4> values = {1, nan, 3, 4};
  do {
    T result(values[order[0]]);
    for (size_t i = 1; i < order.size(); ++i) {
      const T left = result;
      const T right(values[order[i]]);
      combine(&left, &right, &result, 1);
    }
    CHECK(std::isnan(static_cast<double>(result)));
    T first(0.0f), second(0.0f), result_tree(0.0f);
    const T a(values[order[0]]), b(values[order[1]]);
    const T c(values[order[2]]), d(values[order[3]]);
    combine(&a, &b, &first, 1);
    combine(&c, &d, &second, 1);
    combine(&first, &second, &result_tree, 1);
    CHECK(std::isnan(static_cast<double>(result_tree)));
  } while (std::next_permutation(order.begin(), order.end()));
}

template <typename T, typename Op>
void check_in_place(Op op, bool maximum) {
  check_nan_reduction<T>(
      [op](const T* a, const T* b, T* out, size_t n) {
        if (n != 0) {
          std::copy(a, a + n, out);
        }
        op(b, out, n);
      },
      maximum);
}

template <typename T>
void check_jaccl() {
  check_in_place<T>(jaccl::MinOp<T>{}, false);
  check_in_place<T>(jaccl::MaxOp<T>{}, true);
  check_nan_reduction<T>(jaccl::MinOp<T>{}, false);
  check_nan_reduction<T>(jaccl::MaxOp<T>{}, true);
}

template <typename T, typename Min, typename Max>
void check_integer_reduction(Min minimum, Max maximum) {
  const T lo = std::numeric_limits<T>::lowest();
  const T hi = std::numeric_limits<T>::max();
  const std::array<T, 4> a = {lo, hi, T(0), T(1)};
  const std::array<T, 4> b = {hi, lo, T(1), T(0)};
  auto out = a;
  minimum(b.data(), out.data(), out.size());
  for (size_t i = 0; i < out.size(); ++i) {
    CHECK(out[i] == std::min(a[i], b[i]));
  }
  out = a;
  maximum(b.data(), out.data(), out.size());
  for (size_t i = 0; i < out.size(); ++i) {
    CHECK(out[i] == std::max(a[i], b[i]));
  }
}

} // namespace

TEST_CASE_TEMPLATE(
    "distributed native min/max propagate NaN",
    T,
    float,
    double,
    mlx::core::float16_t,
    mlx::core::bfloat16_t) {
  check_in_place<T>(mlx::core::distributed::detail::MinOp<T>{}, false);
  check_in_place<T>(mlx::core::distributed::detail::MaxOp<T>{}, true);
}

TEST_CASE_TEMPLATE(
    "distributed jaccl min/max propagate NaN",
    T,
    float,
    double,
    jaccl::bfloat16_t) {
  check_jaccl<T>();
}

TEST_CASE("distributed jaccl float16 min/max propagate NaN") {
  jaccl::dispatch_all_types(jaccl::Float16, [](auto tag) {
    using T = typename decltype(tag)::type;
    if constexpr (
        !std::is_integral_v<T> && !std::is_same_v<T, jaccl::complex64_t>) {
      check_jaccl<T>();
    }
  });
}

#if defined(__aarch64__)
TEST_CASE(
    "distributed jaccl native bf16 min/max propagate NaN" *
    doctest::skip(!jaccl::has_native_bf16_support())) {
  using T = jaccl::bfloat16_t;
  check_in_place<T>(
      [](const T* in, T* out, size_t n) { jaccl::native_bf16_min(in, out, n); },
      false);
  check_in_place<T>(
      [](const T* in, T* out, size_t n) { jaccl::native_bf16_max(in, out, n); },
      true);
  check_nan_reduction<T>(
      [](const T* a, const T* b, T* out, size_t n) {
        jaccl::native_bf16_min(a, b, out, n);
      },
      false);
  check_nan_reduction<T>(
      [](const T* a, const T* b, T* out, size_t n) {
        jaccl::native_bf16_max(a, b, out, n);
      },
      true);
}
#endif

TEST_CASE_TEMPLATE(
    "distributed native min/max preserve integers",
    T,
    bool,
    int8_t,
    uint8_t,
    int16_t,
    uint16_t,
    int32_t,
    uint32_t,
    int64_t,
    uint64_t) {
  check_integer_reduction<T>(
      mlx::core::distributed::detail::MinOp<T>{},
      mlx::core::distributed::detail::MaxOp<T>{});
  check_integer_reduction<T>(jaccl::MinOp<T>{}, jaccl::MaxOp<T>{});
}
