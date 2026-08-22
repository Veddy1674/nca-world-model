import jax
import jax.numpy as jnp
import numpy as np
from time import perf_counter

from NCA_WM import NCA_WM
from inference import compile_model_inference

from base_config import * # intellisense for configs

# runs a benchmark for 'total_time' seconds, grid is HW
# python for loops are used for a realistic inference benchmark
def run_benchmark(model: NCA_WM, grid_w: int, grid_h: int, total_time: float = 5.0, *, key: jax.Array):

    print(f"Benchmarking with grid {grid_w}x{grid_h} ({grid_w * grid_h:,} cells):")

    # create a fake index-map with zeros, ones, twos, ... up to model.vis_channels - 1
    curr_state = jax.random.randint(key, (grid_h, grid_w), minval=0, maxval=model.vis_channels)

    # convert to one-hot or keep index map
    if model.embedding_dim is None:
        curr_state = jax.nn.one_hot(curr_state, model.vis_channels)
        curr_state = jnp.transpose(curr_state, (2, 0, 1))
    else:
        curr_state = curr_state.astype(jnp.uint8) # ensure uint8 for embedding encoder
    
    curr_hid = None
    if model.hid_channels > 0:
        curr_hid = jnp.zeros((model.hid_channels, grid_h, grid_w), dtype=model.dtype)

    # 1 step warmup for JAX to compile function
    model_inference = compile_model_inference(model, grid_h, grid_w, curr_state, cfg.SUBSTEPS)

    print("  Estimating approximate model speed...")

    # dummy onehot action (all zeros in one-hot)
    dummy_action = jnp.array(np.zeros(model.actions, dtype=model.dtype))

    t = perf_counter()

    for _ in range(10):
        curr_state, curr_hid = model_inference(curr_state, curr_hid, dummy_action)
    
    curr_state.block_until_ready() # wait for GPU for the timer to be accurate
    t = (perf_counter() - t) / 10 # estimated time per step in seconds

    if t > 0.016: # less than 60 FPS
        print(f"  Warning: {t:.4f}s per step is less than 16.67ms/step (60 FPS), timing might be unaccurate")

    # the less steps, the less accurate
    steps = max(1, int(total_time / t))

    # benchmark begins:
    print(f"  Running benchmark for {total_time}s ({steps} steps, {cfg.SUBSTEPS} forward passes each)...")

    t = perf_counter()

    for _ in range(steps):
        action_map = None
        if model.actions > 0:
            # random action from CPU to GPU (to be realistic with timing)
            action = np.zeros(model.actions, dtype=np.float32)
            action[np.random.randint(0, model.actions)] = 1.0 # random index to 1, 0 elsewhere

            action_map = jnp.array(action)

        # autoregressive loop of both states and hidden channels
        curr_state, curr_hid = model_inference(curr_state, curr_hid, action_map)
    
    curr_state.block_until_ready() # wait for GPU for the timer to be accurate
    t = (perf_counter() - t) * 1000 # total time in milliseconds

    avg_ms = t / steps # ms/step

    print(f"  Done! -> Avg: {avg_ms:.4f}ms/step - FPS: {1000/avg_ms:.2f}\n")

if __name__ == "__main__":
    # allow config loading as first argument
    cfg = load_configuration()

    # init model (no load needed to benchmark)
    model: NCA_WM = cfg.make_model(jax.random.key(0))

    # print using device name
    print(f"Using {jax.devices()[0].device_kind.upper()}")

    print_model_info(model, details=True)

    try:
        # test on different resolutions (if too slow, aborts automatically)
        run_benchmark(model, grid_w=16, grid_h=16, total_time=5.0, key=jax.random.key(9))
        run_benchmark(model, grid_w=64, grid_h=64, total_time=5.0, key=jax.random.key(8))
        run_benchmark(model, grid_w=256, grid_h=256, total_time=5.0, key=jax.random.key(7))
        run_benchmark(model, grid_w=1024, grid_h=1024, total_time=5.0, key=jax.random.key(6))

    except KeyboardInterrupt:
        print("\nBenchmark interrupted")