from typing import Optional
import jax
import jax.numpy as jnp
import equinox as eqx

from NCA_WM import NCA_WM

from base_config import * # intellisense for configs

# action is jax.Array because primitives are statically compiled,
# which would recompile everytime action is a different integer
@eqx.filter_jit
def _model_inference(
        model: NCA_WM,
        grid_h: int,
        grid_w: int,
        curr_state: jax.Array,
        curr_hid: Optional[jax.Array],
        action_onehot: Optional[jax.Array],
        substeps: int,
        temperature: float,
        key: jax.Array
    ) -> tuple[jax.Array, Optional[jax.Array], jax.Array]:

    # encode if embedding
    curr_state = model.encode_vis(curr_state)

    # convert to one-hot if no embedding but data is index map (HW)
    if model.embedding_dim is None and curr_state.ndim == 2:
        curr_state = jax.nn.one_hot(curr_state, model.vis_channels, dtype=model.dtype)
        curr_state = jnp.transpose(curr_state, (2, 0, 1))
    
    # concatenate hidden channels
    if curr_hid is not None:
        curr_state = jnp.concatenate([curr_state, curr_hid], axis=0)

    # build action map if actions are used
    action_map = None
    if action_onehot is not None:
        # expand from B to BCHW
        action_map = jnp.broadcast_to(
            action_onehot[:, None, None], 
            (model.actions, grid_h, grid_w)
        )
    
    # generate key
    key, step_key, sample_key = jax.random.split(key, 3)
    
    # forwards
    pred = model.step(curr_state, action_map, substeps, step_key)

    # decode if embedding
    pred_vis = pred[:model.vis_repr_dim]
    pred_vis = model.decode_vis(pred_vis)

    # creating next_state from prediction and applying temperature:
    if model.is_continuous: # RGB

        next_state = pred_vis

        # apply gaussian noise
        if temperature > 0.0:
            next_state += jax.random.normal(sample_key, shape=pred_vis.shape, dtype=pred_vis.dtype) * temperature
        
        next_state = jnp.clip(next_state, 0.0, 1.0) # simple clip

    else:
    
        if temperature == 0.0:
            # only take visual channels and argmax
            # this could make the model more stable, as its outputs are being "fixed" and given as input later on
            # alternatively, pred_vis could be returned and given to the model as its own prediction as is
            next_state = jnp.argmax(pred_vis, axis=0)
        else:
            logits_hwc = jnp.transpose(pred_vis, (1, 2, 0))

            scaled_logits = logits_hwc / temperature
            
            # probabilistic sampling instead of argmax
            next_state = jax.random.categorical(sample_key, scaled_logits, axis=-1)

        if model.embedding_dim is not None:
            next_state = next_state.astype(jnp.uint8) # making sure it's uint8 for index map
        else:
            # convert to one-hot, float32 always
            next_state = jax.nn.one_hot(next_state, model.vis_channels, dtype=jnp.float32)
            next_state = jnp.transpose(next_state, (2, 0, 1))

    pred_hid = pred[model.vis_repr_dim:] # hidden channels if any
    
    if pred_hid.size > 0:
        # pass
        pass
    else:
        # set it to None for simplicity (rather than a empty array)
        pred_hid = None
    
    return next_state, pred_hid, key

# 1 step warmup for JAX to compile function and return an inference function
# with less parameters (for better readability, _model_inference could be used alone)
def compile_model_inference(model: NCA_WM, grid_h: int, grid_w: int, curr_state: jax.Array, substeps: int, key: Any) -> tuple[Callable, Any]:
    curr_state, _, key = _model_inference(
        model,
        grid_h,
        grid_w,
        curr_state,
        # (unbatched)
        jnp.zeros((model.hid_channels, grid_h, grid_w), dtype=model.dtype), # zeroed
        jnp.zeros((model.actions,), dtype=model.dtype) if model.actions > 0 else None, # dummy value (or None if no actions)
        substeps,
        0.0,
        key
    )

    # shorthand function with less parameters
    def model_inference(curr_state: jax.Array, curr_hid: jax.Array | None, action_onehot: jax.Array | None, temperature: float, key: Any):
        return _model_inference(model, grid_h, grid_w, curr_state, curr_hid, action_onehot, substeps, temperature, key)
    
    return model_inference, key