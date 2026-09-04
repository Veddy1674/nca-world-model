from typing import Callable, Literal
import jax
import jax.numpy as jnp
import numpy as np
from time import perf_counter

from ncwm.model import NCWM
from ncwm.inference import compile_model_inference
from ncwm.base_config import NCWM_Config, print_model_info, print_device

# runs a benchmark for 'total_time' seconds, grid is HW
# python for loops are used for a realistic inference benchmark
def run_benchmark(
        cfg: NCWM_Config,
        model: NCWM,
        grid_w: int,
        grid_h: int,
        total_time: float,
        mode: Literal["realistic", "fastest"],
        key: jax.Array,
    ) -> jax.Array:

    print(f"Benchmarking with grid {grid_w}x{grid_h} ({grid_w * grid_h:,} cells - Mode: {mode}):")

    key, _ = jax.random.split(key)
    np.random.seed(1)

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
    model_inference, key = compile_model_inference(model, grid_h, grid_w, curr_state, cfg.SUBSTEPS, key)

    print("  Estimating approximate model performance...")

    # dummy onehot action (all zeros)
    action_map = None
    if model.actions > 0:
        action_map = jnp.array(np.zeros(model.actions, dtype=np.float32))

    curr_state.block_until_ready() # sync GPU
    t = perf_counter()

    # to estimate model performance
    for _ in range(10):
        curr_state, curr_hid, key = model_inference(curr_state, curr_hid, action_map, 0.0, key)
    
    curr_state.block_until_ready() # wait for GPU for the timer to be accurate
    t = (perf_counter() - t) / 10.0 # estimated time per step in seconds

    if t > 0.01667: # less than 60 FPS
        print(f"  Warning: {t:.4f}s per step is less than 16.67ms/step (60 FPS), timing might be inaccurate")

    # the less steps, the less accurate
    steps = max(10, int(total_time / t))

    # benchmark begins:
    print(f"  Running benchmark for {total_time}s ({steps} steps, {cfg.SUBSTEPS} forward passes each)...")

    # case 1: no actions
    if model.actions == 0:
        t = perf_counter()

        for i in range(steps):
            curr_state, curr_hid, key = model_inference(curr_state, curr_hid, None, 0.0, key)
    
    # case 2: actions are preloaded on GPU, thus CPU overhead is minimal
    elif mode == "fastest":
        # preload actions directly on GPU
        key, act_key = jax.random.split(key)

        # random one-hots
        random_indices = jax.random.randint(act_key, (steps,), minval=0, maxval=model.actions)
        actions_batch = jax.nn.one_hot(random_indices, model.actions)

        curr_state.block_until_ready() # sync GPU
        t = perf_counter()

        for i in range(steps):
            curr_state, curr_hid, key = model_inference(curr_state, curr_hid, actions_batch[i], 0.0, key)

    # case 3: actions are generated each step with CPU and transfered to GPU (a realistic approach)
    elif mode == "realistic":
        t = perf_counter()

        for _ in range(steps):
            # random action from CPU to GPU (to be realistic with timing)
            action = np.zeros(model.actions, dtype=np.float32)
            action[np.random.randint(0, model.actions)] = 1.0 # random index to 1, 0 elsewhere

            action_map = jnp.array(action)

            # autoregressive loop of both states and hidden channels
            curr_state, curr_hid, key = model_inference(curr_state, curr_hid, action_map, 0.0, key)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    curr_state.block_until_ready() # wait for GPU for the timer to be accurate
    t = (perf_counter() - t) * 1000.0 # total time in milliseconds

    avg_ms = t / steps # ms/step

    print(f"  Done! -> Avg: {avg_ms:.4f}ms/step - FPS: {1000.0 / avg_ms:.2f}\n")

    return key

def main(cfg: NCWM_Config, benchmark_callback: Callable[[NCWM_Config, NCWM, jax.Array], None]):
    # init keys
    main_key, model_key = jax.random.split(jax.random.key(0))
    
    # init model (no load needed to benchmark)
    model: NCWM = cfg.make_model(model_key)

    # print name of device in use
    print_device()

    print_model_info(model, details=True)

    try:
        # allows running run_benchmark (which is exported by the library) freely
        benchmark_callback(cfg, model, main_key) # no need to return main_key as it isn't used anymore

    except KeyboardInterrupt:
        print("\nBenchmark interrupted")