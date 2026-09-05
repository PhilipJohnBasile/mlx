// Copyright © 2026 Apple Inc.

#pragma once

#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace jaccl::detail {

inline std::chrono::milliseconds mesh_timeout() {
  const char* value = std::getenv("JACCL_TIMEOUT_MS");
  if (value == nullptr) {
    value = std::getenv("MLX_JACCL_TIMEOUT_MS");
  }
  if (value == nullptr) {
    return std::chrono::milliseconds(0);
  }
  std::string_view text(value);
  int64_t milliseconds = 0;
  auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), milliseconds);
  if (error != std::errc{} || end != text.data() + text.size() ||
      milliseconds < 0) {
    throw std::invalid_argument(
        "[jaccl] JACCL_TIMEOUT_MS/MLX_JACCL_TIMEOUT_MS must be a "
        "nonnegative integer number of milliseconds (0 disables the timeout)");
  }
  return std::chrono::milliseconds(milliseconds);
}

// Measure lack of progress, not total operation time. Clock reads are omitted
// when disabled and amortized across empty polls when enabled. The clock is a
// template parameter so boundary behavior can be tested without sleeping.
template <typename Clock = std::chrono::steady_clock>
class ProgressTimeout {
 public:
  explicit ProgressTimeout(std::chrono::milliseconds timeout)
      : timeout_(timeout) {
    if (timeout_.count() > 0) {
      last_progress_ = Clock::now();
    }
  }

  bool expired(int completions) {
    if (timeout_.count() == 0) {
      return false;
    }
    if (completions > 0) {
      last_progress_ = Clock::now();
      empty_polls_ = 0;
      return false;
    }
    // Check the first empty poll too, including after a successful completion.
    if (empty_polls_++ % 1024 != 0) {
      return false;
    }
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               Clock::now() - last_progress_) >= timeout_;
  }

 private:
  std::chrono::milliseconds timeout_;
  typename Clock::time_point last_progress_{};
  uint64_t empty_polls_{0};
};

} // namespace jaccl::detail
