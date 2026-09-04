from typing import Optional
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from glob import glob
from tqdm import tqdm
from time import perf_counter

from ncwm.model import NCWM, save_model_and_optstate, load_model_and_optstate
from ncwm.base_config import NCWM_Config, print_model_info, print_device
from ncwm.dataload import preprocess_data

# uppercase args are constant/static, passed as arguments 
# in functions called often for performance reasons

# training-related functions:
def load_and_flatten_data(cfg: NCWM_Config, model: NCWM, print_enabled: bool):
    files = sorted(glob(cfg.DATA_GLOB)) # sort is necessary for determinism (load files in the same order)

    if cfg.DATA_LIMIT is not None: # None = include all
        if isinstance(cfg.DATA_LIMIT, tuple):
            # e.g: (None, 200), (100, 200), (200, None) - both ends inclusive
            start, end = cfg.DATA_LIMIT
            files = files[start:end]
        
        elif isinstance(cfg.DATA_LIMIT, int): # basically (None, DATA_LIMIT)
            if cfg.DATA_LIMIT > 0:
                files = files[:cfg.DATA_LIMIT]

    if not files:
        raise ValueError(f"No file found with glob pattern '{cfg.DATA_GLOB}' and limit {cfg.DATA_LIMIT}")

    # flatten data in a single large file (to avoid iterating for each file
    # separately because each file has different amount of steps)
    flat_states = []
    flat_targets = []
    flat_actions = []

    flat_infos_in = [] # infos at 't'
    flat_infos_out = [] # infos at 't+1'
    
    if print_enabled:
        print("Loading data to RAM...")

    infosWarn = False
    for fIdx, f in enumerate(files):
        # load data to RAM and convert it to one-hot
        raw_states, raw_actions, raw_infos = preprocess_data(
            file=f,
            vis_channels=model.vis_channels,
            use_embedding=model.embedding_dim is not None,
            is_continuous=model.is_continuous,
            color_map=cfg.COLOR_MAP, # can be None
            verbose=fIdx == 0 # only print warnings for first file
        )

        if raw_actions is None and model.actions > 0:
            raise ValueError(f"Model expects actions but actions were not found in data")

        # raw_states is BCHW
        # raw_actions is B (expanded during training to reduce immediate VRAM usage)

        flat_states.append(raw_states[:-1]) # exclude last state because it has no target/action
        flat_targets.append(raw_states[1:]) # exclude first state because it has no previous state

        if raw_actions is not None:
            flat_actions.append(raw_actions) # actions already are len(states) - 1

        if print_enabled and not infosWarn and raw_infos is not None and model.hid_channels == 0:
            infosWarn = True
            # 'infos' could be passed regardless, but using it in loss_calc is not recommended
            # as the model might not be able to learn certain behaviors without hidden channels
            # TODO this "discard" behavior should be disimplemented or at least become configurable!
            print("Warning: 'infos' array was found in data but model has no hidden channels to understand it, it was discarded")
        
        elif raw_infos is not None:
            flat_infos_in.append(raw_infos[:-1]) # align to states (exclude last) - used mostly for cfg.init_hidden()
            flat_infos_out.append(raw_infos[1:]) # align to targets (exclude first) - used mostly for cfg.loss_calc()
    
    if cfg.LOADING_MODE == "VRAM":
        if print_enabled:
            print("Loading data to VRAM...")
        
        # concatenate everything in huge arrays (to VRAM)
        all_x = jnp.concatenate(flat_states, axis=0)
        all_y = jnp.concatenate(flat_targets, axis=0)
        all_a = jnp.concatenate(flat_actions, axis=0) if flat_actions else None

        all_i_in = jnp.concatenate(flat_infos_in, axis=0) if flat_infos_in else None
        all_i_out = jnp.concatenate(flat_infos_out, axis=0) if flat_infos_out else None
    
    elif cfg.LOADING_MODE == "RAM":
        # concatenate everything in 3 huge arrays (to RAM)
        all_x = np.concatenate(flat_states, axis=0)
        all_y = np.concatenate(flat_targets, axis=0)
        all_a = np.concatenate(flat_actions, axis=0) if flat_actions else None

        all_i_in = np.concatenate(flat_infos_in, axis=0) if flat_infos_in else None
        all_i_out = np.concatenate(flat_infos_out, axis=0) if flat_infos_out else None
    else:
        # TODO implement disk buffering
        raise NotImplementedError(f"Disk Buffering isn't implemented yet, LOADING_MODE = \"RAM\" paired with reduced dataset size (see DATA_LIMIT) is a decent alternative. (Memory swapping or paging can also be enabled in the OS settings for a semi-equivalent result)")

    total = all_x.nbytes + all_y.nbytes

    # add actions
    if all_a is not None:
        total += all_a.nbytes
    
    # add infos
    if all_i_in is not None:
        total += all_i_in.nbytes
    if all_i_out is not None:
        total += all_i_out.nbytes
    
    if print_enabled:
        print(f"Allocated {total / (1024 ** 3):,f} GB to {cfg.LOADING_MODE}")
    
    return all_x, all_y, all_a, all_i_in, all_i_out

# sample batch for a train_chunk
def prepare_data(cfg: NCWM_Config, total_states, seq_length, filekey, chunk_size):
    # sample a batch of states/targets/actions indices at once
    max_val = total_states - seq_length

    if filekey is None and chunk_size is not None: # load via numpy (RAM)
        batch_idx = np.random.randint(0, max_val, size=(chunk_size, cfg.BATCH_SIZE))

        # (seq_length is TRUNCATED_BPTT x 2)
        # make batched sequential for BPTT to work properly
        offsets = np.arange(seq_length)

        # (chunk_size, BATCH_SIZE, seq_length)
        return batch_idx[:, :, None] + offsets[None, None, :]
    
    # else load via jax (VRAM)
    batch_idx = jax.random.randint(filekey, shape=(cfg.BATCH_SIZE,), minval=0, maxval=max_val)

    offsets = jnp.arange(seq_length)

    # (BATCH_SIZE, seq_length)
    return batch_idx[:, None] + offsets[None, :]

# loss function with batch support
def loss_func(cfg: NCWM_Config, model: NCWM, model_axes, grid_h, grid_w, x, y, action_map, i_in, i_out, key):
    # x is (BATCH_SIZE, seq_length, C, H, W)
    total_steps = x.shape[1]

    # vmap the step method with SUBSTEPS (constant, forwards 'SUBSTEPS' amount of times)
    batched_step = jax.vmap(
        lambda x, a, k: model.step(x, a, cfg.SUBSTEPS, k),
        in_axes=model_axes
    )

    # if no hidden channels at all, use a simplified logic (cfg.TRUNCATED_BPTT is expected to be 1)
    if model.hid_channels == 0:
        # flat into one sequence
        state_flat = jnp.reshape(x, (-1, *x.shape[2:]))
        target_flat = jnp.reshape(y, (-1, *y.shape[2:]))
        
        action_flat = None
        if action_map is not None:
            action_flat = jnp.reshape(action_map, (-1, *action_map.shape[2:]))
        
        # encode if embedding
        state_flat = jax.vmap(model.encode_vis)(state_flat)

        # function to add noise to visible channels
        if cfg.add_noise is not None:
            key, noise_key = jax.random.split(key)

            state_flat, _ = cfg.add_noise(model, state_flat, None, noise_key)
        
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, state_flat.shape[0]) # (BATCH_SIZE, 2)
        
        pred = batched_step(state_flat, action_flat, step_keys) # only contains visible channels!

        # decode visible channels if embedding
        pred = jax.vmap(model.decode_vis)(pred)

        # infos make no sense if hidden channels are 0, the model could never
        # understand a loss about something external (which can instead be encoded into hidden channels)
        return cfg.loss_calc(pred, None, target_flat, action_flat, None)

    # reset hidden channels ONLY at the beginning, cut its gradient when half of total_steps are done
    # this is called Stateful BPTT, this way the net can learn to manage hidden channels conditioned on past latent
    # representations outside the current gradient (thus it remains stable even with hidden channel values that it has never seen)
    if cfg.init_hidden is not None:
        # pass the first state and info (if any) to initialize hidden channels
        hid_channels = cfg.init_hidden(x[:, 0], i_in[:, 0] if i_in is not None else None, model.hid_channels, grid_h, grid_w)
    else:
        hid_channels = jnp.zeros((x.shape[0], model.hid_channels, grid_h, grid_w), dtype=model.dtype)
    
    total_loss = 0.0

    for t in range(total_steps):
        if t == cfg.TRUNCATED_BPTT: # halfway through loop
            hid_channels = jax.lax.stop_gradient(hid_channels)

        state_t = x[:, t]
        target_t = y[:, t]
        action_t = action_map[:, t] if action_map is not None else None

        infos_t_out = i_out[:, t] if i_out is not None else None
        
        # encode visible channels if embedding
        state_t = jax.vmap(model.encode_vis)(state_t)

        # function to add noise to visible or hidden channels
        if cfg.add_noise is not None:
            key, noise_key = jax.random.split(key)

            state_t, hid_channels = cfg.add_noise(model, state_t, hid_channels, noise_key)

        # concat hidden channels to visible channels
        state_t = jnp.concatenate([state_t, hid_channels], axis=1)

        # step keys
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, x.shape[0]) # (BATCH_SIZE, 2)

        # do prediction
        pred = batched_step(state_t, action_t, step_keys)

        # decode if embedding
        pred_vis = jax.vmap(model.decode_vis)(pred[:, :model.vis_repr_dim])
        pred_hid = pred[:, model.vis_repr_dim:]
        # (here decay or tanh or clip or whatever could be applied)

        # calc loss
        total_loss += cfg.loss_calc(pred_vis, pred_hid, target_t, action_t, infos_t_out) # pred contains output hidden channels

        # set predicted hidden channels as new input
        hid_channels = pred_hid
    
    return total_loss / total_steps # normalize

# calculate loss and update model parameters, with batch support
def loss_and_update(cfg: NCWM_Config, diff_model, static_model, optimizer, opt_state, model_axes, grid_h, grid_w, x, y, action_map, i_in, i_out, key):

    @eqx.filter_value_and_grad
    def make_loss(diff):
        # combine model for .step call
        full_model = eqx.combine(diff, static_model)
        return loss_func(cfg, full_model, model_axes, grid_h, grid_w, x, y, action_map, i_in, i_out, key)

    # calculate loss and gradient
    loss, grads = make_loss(diff_model)

    # update model parameters
    updates, opt_state = optimizer.update(grads, opt_state, diff_model)
    diff_model = eqx.apply_updates(diff_model, updates)

    return diff_model, opt_state, loss

# runs train steps 'LOG_SEGMENTS' amount of times to reduce CPU overhead
@eqx.filter_jit
def train_chunk(cfg: NCWM_Config, model: NCWM, optimizer, opt_state, key, total_states, seq_length, chunk_size, load_vram, model_axes, grid_h, grid_w, all_x, all_y, all_a, all_i_in, all_i_out):

    # split static parts (like lambdas) out of jax loop
    diff_model, static_model = eqx.partition(model, model.trainable_mask)
    
    def loop_body(i, carry):
        # unpack carry
        curr_diff, curr_opt_state, curr_key, _ = carry

        # load_vram is True when data was loaded directly in VRAM, false when loaded from RAM or via disk buffering
        if load_vram: # dynamic slicing
        
            curr_key, filekey = jax.random.split(curr_key)

            seq_indices = prepare_data(cfg, total_states, seq_length, filekey, None)

            batch_states = all_x[seq_indices]
            batch_targets = all_y[seq_indices]
            batch_actions = all_a[seq_indices] if all_a is not None else None

            batch_infos_in = all_i_in[seq_indices] if all_i_in is not None else None
            batch_infos_out = all_i_out[seq_indices] if all_i_out is not None else None
        
        else: # RAM (or using disk buffering)
            # data is already sliced
            batch_states = all_x[i] # (chunk_size, BATCH_SIZE, seq_length, C, H, W)
            batch_targets = all_y[i] # (chunk_size, BATCH_SIZE, seq_length, C, H, W)
            batch_actions = all_a[i] if all_a is not None else None # (chunk_size, BATCH_SIZE, seq_length, model.actions)

            batch_infos_in = all_i_in[i] if all_i_in is not None else None # (chunk_size, BATCH_SIZE, seq_length, ...)
            batch_infos_out = all_i_out[i] if all_i_out is not None else None # (chunk_size, BATCH_SIZE, seq_length, ...)
        
        # build action_map
        action_maps = None
        if static_model.actions > 0:
            assert batch_actions is not None # should never fail as it was managed in load_and_flatten_data

            # convert action int to one-hot
            one_hot_actions = jax.nn.one_hot(batch_actions, static_model.actions)

            # expand from (BATCH_SIZE,) to (BATCH_SIZE, TRUNCATED_BPTT, model.actions, H, W)
            action_maps = jnp.broadcast_to(
                one_hot_actions[:, :, :, None, None],
                (batch_actions.shape[0], seq_length, static_model.actions, grid_h, grid_w)
            )

        # loss_key is used to add noise in states and hidden channels, mostly
        curr_key, loss_key = jax.random.split(curr_key)
        
        # compute loss and update model weights
        curr_diff, curr_opt_state, loss = loss_and_update(
            cfg, curr_diff, static_model, optimizer, curr_opt_state, model_axes, grid_h, grid_w,
            batch_states, batch_targets, action_maps, batch_infos_in, batch_infos_out,
            loss_key
        )

        # return updated carry
        return (curr_diff, curr_opt_state, curr_key, loss)

    init_carry = (diff_model, opt_state, key, jnp.array(0.0))
    
    # execute loop on GPU from 0 to chunk_size
    diff_model, opt_state, key, loss = jax.lax.fori_loop(
        0, chunk_size, loop_body, init_carry
    )
    
    # return final values (and restore full model)
    return eqx.combine(diff_model, static_model), opt_state, key, loss

def _full_training(
        cfg: NCWM_Config,
        model: NCWM,
        scheduler,
        optimizer,
        opt_state,
        key: jax.Array,
        LOG_ACTUAL_SEGMENTS: int,
        LOG_PADDING: int,
        LOG_ENABLED: bool,
        print_enabled: bool,
        all_states,
        all_targets,
        all_actions,
        all_infos_in,
        all_infos_out
    ):

    # length of B (every state of every file summed up)
    TOTAL_STATES = all_states.shape[0]

    # represents batch dimensions to vmap model.step in loss_func
    MODEL_AXES = (
        0, # state
        0 if model.actions > 0 else None, # action (nullable)
        0 # key
    )

    # get grid dims from first loaded file, first state
    if model.embedding_dim is None:
        _, _, GRID_H, GRID_W = all_states.shape # BCHW (one-hot)
    else:
        _, GRID_H, GRID_W = all_states.shape # BHW (index-map)

    # train in chunks to reduce CPU overhead
    chunk_size = LOG_ACTUAL_SEGMENTS if LOG_ENABLED else cfg.STEPS # if logging is disabled, train in one single chunk (faster)
    chunks = cfg.STEPS // chunk_size

    # sequence length of data in batches (x2 for Stateful BPTT)
    seq_length = cfg.TRUNCATED_BPTT * 2

    # loading via disk buffering is treated like loading via RAM
    load_vram = cfg.LOADING_MODE == "VRAM"

    # local class for a custom tqdm
    class TrainingTqdm(tqdm):
        def __init__(self, *args, **kwargs):
            # adds "Train Chunks" to the progress bar for clarity
            kwargs["unit"] = "chunk"
            kwargs["bar_format"] = "{l_bar}{bar}| {n}/{total} Train Chunks [{elapsed}<{remaining}{postfix}]"

            super().__init__(*args, **kwargs)
        
        def display(self, msg=None, pos=None):
            if msg is None:
                d = self.format_dict
                rate = d["rate"] if d["rate"] is not None else 0.0
                
                # printing both chunk/s and step/s
                d["postfix"] = f"{rate:.2f} chunk/s, {rate * chunk_size:.2f} step/s"

                msg = self.format_meter(**d)
            
            super().display(msg, pos)

    # wrapper function for training
    def train(s, t, a, i_in, i_out):
        return train_chunk(
            cfg, model, optimizer, opt_state, key,
            TOTAL_STATES, seq_length, chunk_size, load_vram, MODEL_AXES, GRID_H, GRID_W,
            s, t, a, i_in, i_out
        )

    # warmup to compile function
    if load_vram:
        _ = train(all_states, all_targets, all_actions, all_infos_in, all_infos_out)
    
    else: # RAM or disk buffering - create fake chunks
        dummy_indices = prepare_data(cfg, TOTAL_STATES, seq_length, None, chunk_size)
        x = jax.device_put(all_states[dummy_indices])
        y = jax.device_put(all_targets[dummy_indices])
        a = jax.device_put(all_actions[dummy_indices]) if all_actions is not None else None

        i_in = jax.device_put(all_infos_in[dummy_indices]) if all_infos_in is not None else None
        i_out = jax.device_put(all_infos_out[dummy_indices]) if all_infos_out is not None else None

        # uses np.random instead of jax's random function, so returning key isn't required
        _ = train(x, y, a, i_in, i_out)

    loss_history = []

    t = perf_counter() # measure total training time (including logging)
    chunk = 1

    try: # inside a try-catch to backup the model in case of CTRL+C

        # training loop
        for chunk in TrainingTqdm(
            range(1, chunks + 1),
            desc="Training",
            disable=(not print_enabled or not LOG_ENABLED)
        ):

            # if load_vram, data is sliced inside train_chunk
            if load_vram:
                # run chunk of training steps
                model, opt_state, key, loss = train(all_states, all_targets, all_actions, all_infos_in, all_infos_out)
            
            else: # RAM or disk buffering

                # slice data manually (same logic in train_chunk for VRAM)
                seq_indices = prepare_data(cfg, TOTAL_STATES, seq_length, None, chunk_size) # no key means it will use np.random.randint, which loads in RAM instead of VRAM
                
                # explicitly move to GPU chunks
                chunk_x = jax.device_put(all_states[seq_indices])
                chunk_y = jax.device_put(all_targets[seq_indices])
                chunk_a = jax.device_put(all_actions[seq_indices]) if all_actions is not None else None
                
                chunk_i_in = jax.device_put(all_infos_in[seq_indices]) if all_infos_in is not None else None
                chunk_i_out = jax.device_put(all_infos_out[seq_indices]) if all_infos_out is not None else None
                
                model, opt_state, key, loss = train(chunk_x, chunk_y, chunk_a, chunk_i_in, chunk_i_out)
            
            # logging after train_chunk:
            if LOG_ENABLED:
                loss = float(loss) # to extract value from VRAM

                # if NaN or infinite or negative
                if np.isnan(loss) or np.isinf(loss) or np.signbit(loss):
                    raise ValueError(
                        f"Invalid loss: {loss}"
                    )

                loss_history.append(loss)

                if print_enabled:
                    # step
                    step = chunk * LOG_ACTUAL_SEGMENTS
                    log = f"Step: {step:>{LOG_PADDING}d}/{cfg.STEPS}"

                    # loss
                    log += f" - Loss: {loss:.6f}"

                    # lr (if scheduler exists)
                    if callable(scheduler):
                        # return current scheduler lr
                        lr = float(scheduler(step)) # type: ignore
                        log += f" - LR: {lr:.3e}"
                    
                    # "Step: AAAA/BBBB - Loss: C.CCCCCC - LR: D.DDDe-D"
                    tqdm.write(log)
        
    except KeyboardInterrupt:
        from datetime import datetime

        jax.block_until_ready(model) # wait for GPU to finish to accurately measure time and avoid corrupting models

        t = perf_counter() - t
        print(f"\nTraining interrupted in {t:.2f}s (Chunk {chunk}/{chunks})")

        # save backup
        date = datetime.now().strftime("h%H-m%M")
        name = f"backup_{date}.eqx"

        save_model_and_optstate(model, name, opt_state)

        print(f"Backup saved as '{name}'\n")

        return model, t, loss_history, False
    
    jax.block_until_ready(model) # wait for GPU to finish to accurately measure time and avoid corrupting models
    return model, (perf_counter() - t), loss_history, True

def main(
        cfg: NCWM_Config,
        print_enabled: bool = True,

        # keys for reproducibility
        model_key: Optional[jax.Array] = None,
        train_key: Optional[jax.Array] = None,
        data_seed: Optional[int] = 1, # for np.random.seed

        # for hyperparameter sweeping:
        model_override: Optional[NCWM] = None,
        optimizer_scheduler_override: Optional[tuple] = None,
        preloaded_data: Optional[tuple] = None,

    ) -> tuple[NCWM, float, list[float]]:

    # print_enabled defines whether to print info (such as the loss) during training
    # It can be False and LOG_ENABLED be True, in which case the loss is still logged (loss graph can be made) silently
    # It also disabled the tqdm progress bar when False

    # init model
    if model_override is not None:
        model = model_override
    else:
        model: NCWM = cfg.make_model(model_key or jax.random.key(0))

    # init optimizer
    if optimizer_scheduler_override is not None:
        optimizer, scheduler = optimizer_scheduler_override
    else:
        optimizer, scheduler = cfg.make_optimizer()
    
    opt_state = optimizer.init(eqx.filter(model, model.trainable_mask))

    if model_override is None and cfg.LOAD_MODEL is not None:
        model, opt_state = load_model_and_optstate(cfg.LOAD_MODEL, model, optimizer)
    
    # print name of device in use and model info
    if print_enabled:
        print_device()
        print_model_info(model)

    # adjust variables for logging
    if cfg.LOG_SEGMENTS > 0: # avoid division by zero
        LOG_ACTUAL_SEGMENTS = cfg.STEPS // cfg.LOG_SEGMENTS

        # digits of LOG_SEGMENTS for clean logging
        LOG_PADDING = int(np.log10(cfg.STEPS)) + 1
        LOG_ENABLED = True
    else:
        LOG_ACTUAL_SEGMENTS = -1 # disable logging and tqdm progress bar
        LOG_PADDING = -1 # dummy value, unused
        LOG_ENABLED = False
    
    # load data
    if preloaded_data is not None:
        all_states, all_targets, all_actions, all_infos_in, all_infos_out = preloaded_data
    else:
        all_states, all_targets, all_actions, all_infos_in, all_infos_out = load_and_flatten_data(cfg, model, print_enabled)
    
    if print_enabled:
        # "Training for X steps"
        # "Training for X steps (Batch size: Y)"
        print(f"Training for {cfg.STEPS} steps" + (f" (Batch size: {cfg.BATCH_SIZE})" if cfg.BATCH_SIZE > 1 else ""))
    
    # randomness
    if train_key is None:
        train_key = jax.random.key(1)
    
    np.random.seed(data_seed) # for data loading
    
    # what is returned here is also returned by this function itself
    model, t, loss_history, success = _full_training(
        cfg, model, scheduler, optimizer, opt_state, train_key,
        LOG_ACTUAL_SEGMENTS, LOG_PADDING, LOG_ENABLED, print_enabled,
        all_states, all_targets, all_actions, all_infos_in, all_infos_out
    )

    # freeing memory
    del all_states, all_targets, all_actions, all_infos_in, all_infos_out

    # success is False when CTRL+C during training, thus model was already saved

    # Note that 'if success and print_enabled' could reduce nested ifs as print_enabled
    # is False pretty much only during hyperparameter sweeping (where you never save models), but for good practice
    # (as the name 'print_enabled' doesn't suggest it also skips saving), a check before each print is necessary
    if success:
        if print_enabled:
            # "Training completed in X.Ys"
            print(f"Training completed in {t:.2f}s")

        # saving
        if cfg.SAVE_MODEL is not None:
            save_model_and_optstate(model, cfg.SAVE_MODEL, opt_state)

            if print_enabled:
                print(f"Model saved as '{cfg.SAVE_MODEL}'")
        else:
            if print_enabled:
                print("Model wasn't saved to disk")

    # creating a matplotlib loss graph
    if cfg.LOSS_GRAPH is not None and LOG_ENABLED and len(loss_history) > 0:
        import matplotlib.pyplot as plt # if ImportError is raised, no data is lost
        from matplotlib.ticker import MultipleLocator, MaxNLocator, FuncFormatter

        # loss_history might contain less elements in case of CTRL+C during training, so len(loss_history) is used
        step_history = [LOG_ACTUAL_SEGMENTS * (i + 1) for i in range(len(loss_history))]
        
        plt.figure(figsize=(10, 5))
        plt.plot(step_history, loss_history, alpha=0.2, color="blue", label="Raw Loss")

        # plot smoothed
        last = loss_history[0]
        smoothed = []

        SMOOTH = 0.6

        # smoothing the curve even more, because with less precise
        # dtypes like bfloat16, the loss tends to be less smooth
        if model.dtype != jnp.float32:
            SMOOTH = 0.8

        for point in loss_history:
            smoothed_val = last * SMOOTH + point * (1 - SMOOTH)

            smoothed.append(smoothed_val)
            last = smoothed_val

        plt.plot(step_history, smoothed, color="blue", label="Smoothed Loss")
        
        plt.xlabel("Step")
        plt.ylabel("Loss")

        plt.title("Training Loss Over Steps")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.ylim(0, max(loss_history))

        plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
        plt.gca().yaxis.set_major_locator(MaxNLocator(8))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.4f}'))

        plt.savefig(cfg.LOSS_GRAPH, dpi=150, bbox_inches="tight")
        # plt.show()

        print(f"Loss graph saved as '{cfg.LOSS_GRAPH}'\n")
    
    return model, t, loss_history