import jax
import numpy as np
import optuna
from time import perf_counter

from NACE import NACE
import train

from base_config import * # intellisense for configs

def objective(trial: optuna.Trial):
    jax.clear_caches() # NECESSARY to re-compile training functions

    # init model and optimizer (no load), apply parameters suggested by trial
    train.model, train.optimizer = cfg.hyperparam_sweep(model_key, trial) # type: ignore

    train.opt_state = train.optimizer.init(eqx.filter(train.model, train.model.trainable_mask))
    
    # set other globals for train.py to function properly
    # no scheduler because the optimizer already contains it, normally scheduler is used in train.py separately just to log learning rate
    train.LOG_SEGMENTS = cfg.STEPS // 100 # 100 segments TODO manage case cfg.STEPS is 100 or less?
    
    # define data-related globals (mandatory for train.main())
    train.all_states = all_states
    train.all_targets = all_targets
    train.all_actions = all_actions
    train.all_infos_in = all_infos_in
    train.all_infos_out = all_infos_out

    try:
        try:
            _, loss_history = train.main(train.model, train.opt_state, train_key) # trained model result is discarded!

        except ValueError: # invalid loss exception
            raise optuna.TrialPruned("Invalid loss (NaN or Inf or negative)")
            
        # mean of the last 3 chunks' loss (if somehow less than 3 chunks, it will take all of them)
        final_loss = float(np.mean(loss_history[-min(3, len(loss_history)):]))
        
        if trial.should_prune():
            raise optuna.TrialPruned("Loss too high")
            
        return final_loss
        
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

    print(f"Trial {trial.number} - Loss: {loss} - Params: {trial.params}{" - NEW BEST" if trial.number == best_trial_num else ""}")

def main():
    # only log optuna's warnings
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # optuna study to minimize loss
    study = optuna.create_study(direction="minimize", study_name="NACE_Sweep")
    
    print("Initiating hyperparameter sweep...\n")

    study.optimize(objective, n_trials=30, callbacks=[print_callback])
    
    print(f"\nBest trial found ({study.best_trial.number}):")
    print(f"  Loss: {study.best_trial.value:.6f} (after {cfg.STEPS} steps)")
    print("  Parameters:")

    # print all params
    for key, value in study.best_trial.params.items():
        print(f"    {key} = {value}")

if __name__ == "__main__":
    # allow config loading as first argument
    cfg = load_configuration()

    if cfg.hyperparam_sweep is None:
        raise ValueError("Hyperparameter sweep function 'hyperparam_sweep()' is not defined in the configuration")
    
    # set train config to current, as if train.py was being executed directly
    train.cfg = cfg

    train.LOG_ENABLED = True # to save loss history
    train.PRINT_ENABLED = False # disable tqdm and logging prints

    # dummy model required to load data
    model: NACE = cfg.make_model(jax.random.key(0)) # overwritten in objective() each trial

    # print using device name
    print(f"Using {jax.devices()[0].device_kind.upper()}")
    
    # NOTE: in the case a trial tries different model internal parameters (which is common)
    # it will work regardless if the data is loaded once at the beginning instead of every objective()
    # which allows every parameter of the model to be changed each trial
    # EXCEPT for: vis_channels (as data loading uses that to create one-hots) and embedding_dim
    # (which must either always be None or always be int for every trial, although the int can differ)

    # load data
    all_states, all_targets, all_actions, all_infos_in, all_infos_out = train.load_and_flatten_data(cfg, model)

    # define keys identical to all trials for reproducibility
    model_key = jax.random.key(0)
    train_key = jax.random.key(1)

    t = perf_counter()

    try:
        main()

        print(f"\nHyperparameter sweep completed in {perf_counter() - t:.2f}s")

    except KeyboardInterrupt:
        print(f"\nHyperparameter sweep interrupted in {perf_counter() - t:.2f}s")