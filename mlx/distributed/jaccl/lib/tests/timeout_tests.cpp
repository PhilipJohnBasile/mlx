// Copyright © 2026 Apple Inc.

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "jaccl/timeout.h"

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

struct FakeClock {
  using duration = std::chrono::milliseconds;
  using time_point = std::chrono::time_point<FakeClock>;
  inline static time_point current{};
  inline static int reads = 0;
  static time_point now() {
    ++reads;
    return current;
  }
};

void test_clock() {
  using namespace std::chrono_literals;
  using Timer = jaccl::detail::ProgressTimeout<FakeClock>;
  FakeClock::current = FakeClock::time_point{};
  FakeClock::reads = 0;
  Timer disabled(0ms);
  for (int i = 0; i < 10000; ++i) {
    FakeClock::current += 1h;
    require(!disabled.expired(i % 2), "disabled timeout expired");
  }
  require(FakeClock::reads == 0, "disabled timeout read the clock");

  FakeClock::current = FakeClock::time_point{};
  FakeClock::reads = 0;
  Timer bounded(10ms);
  FakeClock::current += 9ms;
  require(!bounded.expired(0), "timeout fired early");
  int reads = FakeClock::reads;
  FakeClock::current += 1ms;
  for (int i = 0; i < 1023; ++i) {
    require(!bounded.expired(0), "empty polls were not amortized");
  }
  require(FakeClock::reads == reads, "clock read on every empty poll");
  require(bounded.expired(0), "deadline boundary did not expire");

  Timer progressing(10ms);
  for (int i = 0; i < 100; ++i) {
    FakeClock::current += 9ms;
    require(!progressing.expired(1), "progress did not reset the timeout");
    require(!progressing.expired(0), "total duration caused a timeout");
  }
  FakeClock::current += 10ms;
  bool expired = false;
  for (int i = 0; i < 1024; ++i) {
    expired = expired || progressing.expired(0);
  }
  require(expired, "timeout failed after progress stopped");

  Timer large{std::chrono::milliseconds{std::numeric_limits<int64_t>::max()}};
  FakeClock::current += 24h;
  require(!large.expired(0), "large timeout overflowed");
  Timer fresh(10ms);
  require(!fresh.expired(0), "new operation inherited an old deadline");
}

void test_environment() {
  using namespace std::chrono_literals;
  unsetenv("JACCL_TIMEOUT_MS");
  unsetenv("MLX_JACCL_TIMEOUT_MS");
  require(jaccl::detail::mesh_timeout() == 0ms, "default must be disabled");
  setenv("MLX_JACCL_TIMEOUT_MS", "17", 1);
  require(jaccl::detail::mesh_timeout() == 17ms, "MLX alias was ignored");
  setenv("JACCL_TIMEOUT_MS", "9", 1);
  require(jaccl::detail::mesh_timeout() == 9ms, "JACCL precedence changed");
  setenv("JACCL_TIMEOUT_MS", "0", 1);
  require(jaccl::detail::mesh_timeout() == 0ms, "explicit disable was ignored");
  for (const char* invalid :
       {"", "-1", "+1", " 2", "2 ", "1.5", "1ms", "9223372036854775808"}) {
    setenv("JACCL_TIMEOUT_MS", invalid, 1);
    bool rejected = false;
    try {
      jaccl::detail::mesh_timeout();
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "invalid timeout was accepted");
  }
  setenv("JACCL_TIMEOUT_MS", "9223372036854775807", 1);
  require(
      jaccl::detail::mesh_timeout().count() ==
          std::numeric_limits<int64_t>::max(),
      "maximum timeout was not preserved");
}

} // namespace

int main() {
  try {
    test_clock();
    test_environment();
    std::cout << "PASS: timeout clock and configuration boundaries\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
