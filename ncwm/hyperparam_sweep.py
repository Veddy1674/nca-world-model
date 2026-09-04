import jax
import numpy as np
import optuna
from time import perf_counter

from functools import partial

from ncwm.model import NCWM
from ncwm.train import main as train_model, load_and_flatten_data
from ncwm.base_config import NCWM_Config, print_model_info, print_device

def objective(
        trial: optuna.Trial,
        cfg: NCWM_Config,
        model_key: jax.Array,
        train_key: jax.Array,
        data_seed: int,
        preloaded_data: tuple
    ):

    jax.clear_caches() # NECESSARY to re-compile training functions

    assert cfg.hyperparam_sweep is not None # to silence pyright

    # init model and optimizer (no load), apply parameters suggested by trial
    model, optimizer = cfg.hyperparam_sweep(model_key, trial)

    try:
        # trained model result is discarded
        _, _, loss_history = train_model(
            cfg,
            print_enabled=False,

            model_key=model_key,
            train_key=train_key,
            data_seed=data_seed,

            model_override=model,
            optimizer_scheduler_override=(optimizer, None),
            preloaded_data=preloaded_data
        )
        
        # NOTE: this could be implemented in a different way to define a successful or failure training
        # such as calculating the mean of all the loss values, or only take the last value

        # in this case: mean of the last 3 loss values
        # (with LOG_SEGMENTS = 100 and STEPS >= 100 assured in main(), loss_history always has more than 3 elements)
        final_loss = float(np.mean(loss_history[-3:]))
        
        if np.isnan(final_loss) or np.isinf(final_loss) or np.signbit(final_loss):
            raise optuna.TrialPruned("Invalid loss (NaN or Inf or negative)")
            
        return final_loss
    
    except (optuna.TrialPruned, KeyboardInterrupt):
        raise
    
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned()

# each trial
def print_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
    if trial.state == optuna.trial.TrialState.COMPLETE:
        loss = f"{trial.value:.6f}"
    else:
        loss = "Invalid"

    # find best so far
    try:
        best_trial_num = study.best_trial.number
    except ValueError:
        # in case the first trial fails
        best_trial_num = -1
    
    params_fancy = "[\n" + ",\n".join([f"    '{k}': {v:.6f}" if isinstance(v, float) else f"    '{k}': {v}" for k, v in trial.params.items()]) + "\n]"

    print(f"\nTrial {trial.number + 1} {"(NEW BEST)" if trial.number == best_trial_num else ""} - Loss: {loss} - Params: {params_fancy}")

def print_best(cfg: NCWM_Config, study: optuna.study.Study):
    # logging best trial so far
    print(f"\nBest trial found ({study.best_trial.number + 1}):")
    print(f"  Loss: {study.best_trial.value:.6f} (after {cfg.STEPS} steps)")
    print("  Parameters:")

    # print all params
    for key, value in study.best_trial.params.items():
        print(f"    {key} = {value}")

def _run_sweep(
        cfg: NCWM_Config,
        trials: int,
        study: optuna.study.Study,
        model_key: jax.Array,
        train_key: jax.Array,
        data_seed: int,
        preloaded_data: tuple,
    ):

    # using partial to add frozen arguments to 'objective'
    objective_func = partial(
        objective,
        cfg=cfg,
        model_key=model_key,
        train_key=train_key,
        data_seed=data_seed,
        preloaded_data=preloaded_data
    )
    
    print("Initiating hyperparameter sweep...\n")

    study.optimize(objective_func, n_trials=trials, callbacks=[print_callback])
    
    print_best(cfg, study)

def main(cfg: NCWM_Config, trials: int = 30):
    if cfg.hyperparam_sweep is None:
        raise ValueError("Hyperparameter sweep function 'hyperparam_sweep()' is not defined in the configuration")
    
    # Note that there isn't an actual "minimum" value, this is to avoid edge cases (such as LOG_SEGMENTS higher than STEPS)
    # And to avoid CPU overhead too: if STEPS = 100 and LOG_SEGMENTS = 100, it's already a bad case as each train chunk is 1 step
    if cfg.STEPS < 100:
        raise ValueError("STEPS must be at least 100 for hyperparameter sweep")
    
    cfg.LOG_SEGMENTS = 100 # to save loss history it must be positive (LOG_ENABLED in train.py will be True)
    cfg.LOSS_GRAPH = None
    cfg.SAVE_MODEL = None

    # dummy model required to load data
    dummy_model: NCWM = cfg.make_model(jax.random.key(0)) # overwritten in objective() each trial

    # print name of device in use
    print_device()

    print_model_info(dummy_model)
    
    # NOTE: in the case a trial tries different model internal parameters (which is common)
    # it will work regardless if the data is loaded once at the beginning instead of every objective()
    # which allows every parameter of the model to be changed each trial
    # EXCEPT for: vis_channels (as data loading uses that to create one-hots) and embedding_dim
    # (which must either always be None or always be int for every trial, although the int can differ)

    # load data
    preloaded_data = load_and_flatten_data(cfg, dummy_model, print_enabled=False)

    # define keys identical to all trials for reproducibility
    model_key = jax.random.key(0)
    train_key = jax.random.key(1)
    data_seed = 1

    # init the study:
    # only log optuna's warnings
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # optuna study to minimize loss
    study = optuna.create_study(direction="minimize", study_name="NCWM_Sweep")

    t = perf_counter()

    try:
        _run_sweep(cfg, trials, study, model_key, train_key, data_seed, preloaded_data)
        print(f"\nHyperparameter sweep completed in {perf_counter() - t:.2f}s")

    except KeyboardInterrupt:
        print_best(cfg, study)
        print(f"\nHyperparameter sweep interrupted in {perf_counter() - t:.2f}s")