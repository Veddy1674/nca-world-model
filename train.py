import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from glob import glob
from tqdm import tqdm
from time import perf_counter

from NCA_WM import NCA_WM, save_model_and_optstate, load_model_and_optstate
from dataload import preprocess_npz

from base_config import * # intellisense for configs

# uppercase args are constant/static, passed as arguments 
# in functions called often for performance reasons

# training-related functions:
def load_and_flatten_data(cfg: NCA_WM_Config, model: NCA_WM):
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
    
    print("Loading data to RAM...")

    infosWarn = False
    for fIdx, f in enumerate(files):
        # load data to RAM and convert it to one-hot
        raw_states, raw_actions, raw_infos = preprocess_npz(
            file=f,
            vis_channels=model.vis_channels,
            use_embedding=model.embedding_dim is not None,
            is_continuous=model.is_continuous,
            color_map=cfg.COLOR_MAP,
            verbose=fIdx == 0 # only print warnings for first file
        )
        # print(raw_states.shape)
        # exit()

        # raw_states is BCHW
        # raw_actions is B (expanded during training to reduce immediate VRAM usage)

        flat_states.append(raw_states[:-1]) # exclude last state because it has no target/action
        flat_targets.append(raw_states[1:]) # exclude first state because it has no previous state
        flat_actions.append(raw_actions) # actions already are len(states) - 1

        if PRINT_ENABLED and not infosWarn and raw_infos is not None and model.hid_channels == 0:
            infosWarn = True
            # note that the "ignore" behavior is not mandatory, but I kept it so because 'infos' (that can be used in loss_calc)
            # theoretically can't be understood by the model if it has no hidden channels to process/save them
            print("Warning: 'infos' array was found in data but model has no hidden channels to understand it, it is being ignored")
        
        elif raw_infos is not None:
            flat_infos_in.append(raw_infos[:-1]) # align to states (exclude last) - used mostly for cfg.init_hidden()
            flat_infos_out.append(raw_infos[1:]) # align to targets (exclude first) - used mostly for cfg.loss_calc()
    
    if cfg.LOADING_MODE == "VRAM":
        print("Loading data to VRAM...")
        # concatenate everything in huge arrays (to VRAM)
        all_x = jnp.concatenate(flat_states, axis=0)
        all_y = jnp.concatenate(flat_targets, axis=0)
        all_a = jnp.concatenate(flat_actions, axis=0)

        all_i_in = jnp.concatenate(flat_infos_in, axis=0) if flat_infos_in else None
        all_i_out = jnp.concatenate(flat_infos_out, axis=0) if flat_infos_out else None
    
    elif cfg.LOADING_MODE == "RAM":
        # concatenate everything in 3 huge arrays (to RAM)
        all_x = np.concatenate(flat_states, axis=0)
        all_y = np.concatenate(flat_targets, axis=0)
        all_a = np.concatenate(flat_actions, axis=0)

        all_i_in = np.concatenate(flat_infos_in, axis=0) if flat_infos_in else None
        all_i_out = np.concatenate(flat_infos_out, axis=0) if flat_infos_out else None
    else:
        # TODO disk buffering
        raise ValueError(f"Invalid loading mode: {cfg.LOADING_MODE}")

    total = (all_x.nbytes + all_y.nbytes + all_a.nbytes)
    # add infos
    if all_i_in is not None:
        total += all_i_in.nbytes
    if all_i_out is not None:
        total += all_i_out.nbytes
    
    print(f"Allocated {total / (1024 ** 3):,f} GB to {cfg.LOADING_MODE}")
    
    return all_x, all_y, all_a, all_i_in, all_i_out

# sample batch for a train_chunk
def prepare_data(seq_length, filekey, chunk_size):
    # sample a batch of states/targets/actions indices at once
    max_val = TOTAL_STATES - seq_length

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
def loss_func(model: NCA_WM, x, y, action_map, i_in, i_out, key):
    # x is (BATCH_SIZE, seq_length, C, H, W)
    total_steps = x.shape[1]

    # vmap the step method with SUBSTEPS (constant, forwards 'SUBSTEPS' amount of times)
    batched_step = jax.vmap(
        lambda x, a, k: model.step(x, a, cfg.SUBSTEPS, k),
        in_axes=MODEL_AXES
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
        hid_channels = cfg.init_hidden(x[:, 0], i_in[:, 0] if i_in is not None else None, model.hid_channels, GRID_H, GRID_W)
    else:
        hid_channels = jnp.zeros((x.shape[0], model.hid_channels, GRID_H, GRID_W), dtype=model.dtype)
    
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
def loss_and_update(diff_model, static_model, opt_state, x, y, action_map, i_in, i_out, key):

    @eqx.filter_value_and_grad
    def make_loss(diff):
        # combine model for .step call
        full_model = eqx.combine(diff, static_model)
        return loss_func(full_model, x, y, action_map, i_in, i_out, key)

    # calculate loss and gradient
    loss, grads = make_loss(diff_model)

    # update model parameters
    updates, opt_state = optimizer.update(grads, opt_state, diff_model)
    diff_model = eqx.apply_updates(diff_model, updates)

    return diff_model, opt_state, loss

# runs train steps 'LOG_SEGMENTS' amount of times to reduce CPU overhead
@eqx.filter_jit
def train_chunk(model: NCA_WM, opt_state, key, seq_length, chunk_size, load_vram, all_x, all_y, all_a, all_i_in, all_i_out):

    # split static parts (like lambdas) out of jax loop
    diff_model, static_model = eqx.partition(model, model.trainable_mask)
    
    def loop_body(i, carry):
        # unpack carry
        curr_diff, curr_opt_state, curr_key, _ = carry

        # load_vram is True when data was loaded directly in VRAM, false when loaded from RAM or via disk buffering
        if load_vram: # dynamic slicing
        
            curr_key, filekey = jax.random.split(curr_key)

            seq_indices = prepare_data(seq_length, filekey, None)

            batch_states = all_x[seq_indices]
            batch_targets = all_y[seq_indices]
            batch_actions = all_a[seq_indices]

            batch_infos_in = all_i_in[seq_indices] if all_i_in is not None else None
            batch_infos_out = all_i_out[seq_indices] if all_i_out is not None else None
        
        else: # RAM (or using disk buffering)
            # data is already sliced
            batch_states = all_x[i] # (chunk_size, BATCH_SIZE, seq_length, C, H, W)
            batch_targets = all_y[i] # (chunk_size, BATCH_SIZE, seq_length, C, H, W)
            batch_actions = all_a[i] # (chunk_size, BATCH_SIZE, seq_length, model.actions)

            batch_infos_in = all_i_in[i] if all_i_in is not None else None # (chunk_size, BATCH_SIZE, seq_length, ...)
            batch_infos_out = all_i_out[i] if all_i_out is not None else None # (chunk_size, BATCH_SIZE, seq_length, ...)
        
        # build action_map
        action_maps = None
        if static_model.actions > 0:
            # convert action int to one-hot
            one_hot_actions = jax.nn.one_hot(batch_actions, static_model.actions)

            # expand from (BATCH_SIZE,) to (BATCH_SIZE, TRUNCATED_BPTT, model.actions, H, W)
            action_maps = jnp.broadcast_to(
                one_hot_actions[:, :, :, None, None],
                (batch_actions.shape[0], seq_length, static_model.actions, GRID_H, GRID_W)
            )

        # loss_key is used to add noise in states and hidden channels, mostly
        curr_key, loss_key = jax.random.split(curr_key)
        
        # compute loss and update model weights
        curr_diff, curr_opt_state, loss = loss_and_update(
            curr_diff, static_model, curr_opt_state, batch_states, batch_targets, action_maps, batch_infos_in, batch_infos_out, loss_key
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

def main(model, opt_state, key):
    loss_history = []

    # define globals - data-related globals such as all_states, all_actions, ... shall be defined beforehand
    global TOTAL_STATES, MODEL_AXES, GRID_H, GRID_W

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
    chunk_size = LOG_SEGMENTS if LOG_ENABLED else cfg.STEPS # if logging is disabled, train in one single chunk (faster)
    chunks = cfg.STEPS // chunk_size

    # sequence length of data in batches (x2 to implement Stateful BPTT)
    seq_length = cfg.TRUNCATED_BPTT * 2

    # loading via disk buffering is treated like loading via RAM
    load_vram = cfg.LOADING_MODE == "VRAM"

    # adds "Train Chunks" to the progress bar for clarity
    tqdm_bar_format = "{l_bar}{bar}| {n}/{total} Train Chunks [{elapsed}<{remaining}, {rate_fmt}{postfix}]"

    # wrapper function for training
    def train(s, t, a, i_in, i_out):
        return train_chunk(
            model, opt_state, key, seq_length, chunk_size, load_vram, s, t, a, i_in, i_out
        )

    # warmup to compile function
    if load_vram:
        # (override key to avoid the first training chunk using the same key as warmup)
        model, opt_state, key, _ = train(all_states, all_targets, all_actions, all_infos_in, all_infos_out)
    
    else: # RAM or disk buffering - create fake chunks
        dummy_indices = prepare_data(seq_length, None, chunk_size)
        x = jax.device_put(all_states[dummy_indices])
        y = jax.device_put(all_targets[dummy_indices])
        a = jax.device_put(all_actions[dummy_indices])

        i_in = jax.device_put(all_infos_in[dummy_indices]) if all_infos_in is not None else None
        i_out = jax.device_put(all_infos_out[dummy_indices]) if all_infos_out is not None else None

        # uses np.random instead of jax's, so returning key isn't required
        _, _, _, _ = train(x, y, a, i_in, i_out)

    global t
    t = perf_counter() # measure total training time (including logging)

    # training loop
    for chunk in tqdm(range(1, chunks + 1), desc="Training", bar_format=tqdm_bar_format, disable=(not PRINT_ENABLED or not LOG_ENABLED)):

        # if load_vram, data is sliced inside train_chunk
        if load_vram:
            # run chunk of training steps
            model, opt_state, key, loss = train(all_states, all_targets, all_actions, all_infos_in, all_infos_out)
        
        else: # RAM or disk buffering

            # slice data manually (same logic in train_chunk for VRAM)
            seq_indices = prepare_data(seq_length, None, chunk_size) # no key means it will use np.random.randint, which loads in RAM instead of VRAM
            
            # explicitly move to GPU chunks
            chunk_x = jax.device_put(all_states[seq_indices])
            chunk_y = jax.device_put(all_targets[seq_indices])
            chunk_a = jax.device_put(all_actions[seq_indices])
            
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

            if PRINT_ENABLED:
                # step
                step = chunk * LOG_SEGMENTS
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
    
    jax.block_until_ready(model) # wait for GPU to finish to accurately measure time
    
    return model, loss_history

if __name__ == "__main__":
    from datetime import datetime

    # allow config loading as first argument
    cfg = load_configuration()

    # init model
    model: NCA_WM = cfg.make_model(jax.random.key(0))
    # print using device name
    print(f"Using {jax.devices()[0].device_kind.upper()}")

    # init optimizer
    optimizer, scheduler = cfg.make_optimizer()
    opt_state = optimizer.init(eqx.filter(model, model.trainable_mask))

    if cfg.LOAD_MODEL is not None:
        model, opt_state = load_model_and_optstate(cfg.LOAD_MODEL, model, optimizer)
    
    print_model_info(model)

    # define a new global LOG_SEGMENTS
    if cfg.LOG_SEGMENTS > 0: # avoid division by zero
        LOG_SEGMENTS = cfg.STEPS // cfg.LOG_SEGMENTS

        # digits of LOG_SEGMENTS for clean logging
        LOG_PADDING = int(np.log10(cfg.STEPS)) + 1
        LOG_ENABLED = True
    else:
        LOG_SEGMENTS = -1 # disable logging and tqdm progress bar
        LOG_PADDING = -1 # dummy value, unused
        LOG_ENABLED = False
    
    # in case a script requires LOG ENABLED (for loss history) but no print, train.LOG_PRINT_ENABLED can be set to False externally
    # it also disables the tqdm progress bar
    PRINT_ENABLED = True
    
    # load data
    all_states, all_targets, all_actions, all_infos_in, all_infos_out = load_and_flatten_data(cfg, model)

    t = perf_counter() # total training time in seconds (redefined globally in main())
    
    try:
        # "Training for X steps"
        # "Training for X steps (Batch size: Y)"
        print(f"Training for {cfg.STEPS} steps" + (f" (Batch size: {cfg.BATCH_SIZE})" if cfg.BATCH_SIZE > 1 else ""))
        
        # training loop
        key = jax.random.key(1)
        np.random.seed(1)
        
        model, loss_history = main(model, opt_state, key)

        # "Training completed in X.Ys"
        print(f"Training completed in {perf_counter() - t:.2f}s")

        if cfg.SAVE_MODEL is not None:
            save_model_and_optstate(model, cfg.SAVE_MODEL, opt_state)
            print(f"Model saved as '{cfg.SAVE_MODEL}'")
        else:
            print("Model wasn't saved to disk")

        if cfg.LOSS_GRAPH is not None and LOG_ENABLED: # create a matplotlib loss graph
            import matplotlib.pyplot as plt # if ImportError is raised, no data is lost
            from matplotlib.ticker import MultipleLocator, MaxNLocator, FuncFormatter

            step_history = list(range(LOG_SEGMENTS, cfg.STEPS + 1, LOG_SEGMENTS))
            
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

    except KeyboardInterrupt:
        print(f"\nTraining interrupted in {perf_counter() - t:.2f}s")

        # save backup
        date = datetime.now().strftime("h%H-m%M")
        name = f"backup_{date}.eqx"

        save_model_and_optstate(model, name, opt_state)

        print(f"Backup saved as '{name}'\n")