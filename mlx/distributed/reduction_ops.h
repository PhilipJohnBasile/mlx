// Copyright © 2025 Apple Inc.

namespace mlx::core::distributed::detail {

template <typename T>
struct SumOp {
  void operator()(const T* input, T* output, size_t N) const {
    while (N-- > 0) {
      *output += *input;
      input++;
      output++;
    }
  }
};

// Propagate NaNs from either operand; bitwise OR keeps selection vectorizable.
template <typename T>
struct MaxOp {
  void operator()(const T* input, T* output, size_t N) const {
    while (N-- > 0) {
      *output = ((*input != *input) | (*output < *input)) ? *input : *output;
      input++;
      output++;
    }
  }
};

template <typename T>
struct MinOp {
  void operator()(const T* input, T* output, size_t N) const {
    while (N-- > 0) {
      *output = ((*input != *input) | (*output > *input)) ? *input : *output;
      input++;
      output++;
    }
  }
};

} // namespace mlx::core::distributed::detail
