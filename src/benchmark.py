from ncwm import NCWM_Config, NCWM, run_benchmarks, run_benchmark, load_configuration
import jax

def default_benchmark(cfg: NCWM_Config, model: NCWM, main_key: jax.Array):
    # test on different resolutions (if too slow aborts automatically)
    main_key = run_benchmark(cfg, model, grid_w=16, grid_h=16, total_time=5.0, mode="realistic", key=main_key)
    main_key = run_benchmark(cfg, model, grid_w=64, grid_h=64, total_time=5.0, mode="fastest", key=main_key)
    main_key = run_benchmark(cfg, model, grid_w=256, grid_h=256, total_time=5.0, mode="realistic", key=main_key)
    main_key = run_benchmark(cfg, model, grid_w=1024, grid_h=1024, total_time=5.0, mode="fastest", key=main_key)

# load_configuration() requires this script to have a configuration path as first argument
run_benchmarks(load_configuration(), benchmark_callback=default_benchmark)
