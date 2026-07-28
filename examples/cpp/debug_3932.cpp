// Temporary investigation harness for ml-explore/mlx#3932. Not for upstream.
#include <cstdio>
#include "mlx/mlx.h"

using namespace mlx::core;

int main() {
  int blocks = 4;
  int dim = 64;
  int batch = 8;

  array x0 = random::normal({batch, dim});
  eval(x0);

  for (int cycle = 0; cycle < 2; ++cycle) {
    fprintf(stderr, "MLX_DEBUG_3932 ==== cycle %d BEGIN, active_memory=%zu ====\n",
            cycle, get_active_memory());
    std::vector<array> weights;
    for (int i = 0; i < blocks; ++i) {
      weights.push_back(random::normal({dim, dim}));
    }
    eval(weights);

    // Capturing lambda: fresh closure identity every cycle, mirroring
    // Python's PyCompiledFun which keys the cache off the wrapped
    // callable's own address (a fresh Python closure each cycle).
    // A plain free function pointer is treated as permanently
    // addressable (compile.cpp get_function_address) and reuses one
    // cache slot forever -- that was the bug in the first version of
    // this harness and produced a false negative (0 leak).
    std::vector<array> captured_weights = weights;
    auto forward = [captured_weights](const std::vector<array>& in) {
      array x = in[0];
      for (auto& w : captured_weights) {
        array h = matmul(x, w);
        auto parts = split(h, 2, -1);
        x = concatenate({parts[1], parts[0]}, -1);
      }
      return std::vector<array>{x};
    };

    auto fn = compile(forward);

    std::vector<array> args = {x0};
    auto out = fn(args);
    eval(out);

    fprintf(stderr, "MLX_DEBUG_3932 cycle %d: about to drop fn/out/weights\n", cycle);
    out.clear();
    weights.clear();
    // fn (std::function) goes out of scope at end of loop body
    fprintf(stderr, "MLX_DEBUG_3932 ==== cycle %d END, active_memory=%zu ====\n",
            cycle, get_active_memory());
  }
  fprintf(stderr, "MLX_DEBUG_3932 all cycles done, active_memory=%zu\n",
          get_active_memory());
  return 0;
}
