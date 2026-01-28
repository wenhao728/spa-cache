import logging
import types
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from transformers.utils import logging as transformer_logging

from ..cache import SpaCache
from ..sequence import SequenceMgmt
from .modeling_utils import apply_rotary_pos_emb, repeat_kv

logger = logging.getLogger(__name__)
transformer_logger = transformer_logging.get_logger(__name__)


def register_block_hooks(
    block: nn.Module,
    block_forward_fn: Optional[Callable] = None,
    block_attention_fn: Optional[Callable] = None,
    # rope_forward_fn: Optional[Callable] = None,
) -> None:
    if block_forward_fn is not None:
        setattr(block, "_block_forward", block.forward)
        block.forward = types.MethodType(block_forward_fn, block)

    if block_attention_fn is not None:
        setattr(block.self_attn, "_block_attention", block.self_attn.forward)
        block.self_attn.forward = types.MethodType(block_attention_fn, block.self_attn)

    # if rope_forward_fn is not None:
    #     setattr(block.rotary_emb, "_rope_forward", block.rotary_emb.forward)
    #     block.rotary_emb.forward = types.MethodType(rope_forward_fn, block.rotary_emb)

def register_hooks(
    model: nn.Module,
    transformer_blocks_name: str = "model.layers",
    block_forward_fn: Optional[Callable] = None,
    block_attention_fn: Optional[Callable] = None,
    sample_fn: Optional[Callable] = None,
    verbose: bool = True,
) -> None:
    """
    Register hooks for the model.
    """
    if sample_fn is not None:
        setattr(model, '_old_sample', model._sample)
        model._sample = types.MethodType(sample_fn, model)
        if verbose:
            logger.info(f"Registered diffusion sample hook for {model.__class__.__name__}")

    if transformer_blocks_name is not None:
        for name, module in model.named_modules():
            if name == transformer_blocks_name:
                module: nn.ModuleList
                for block in module:
                    register_block_hooks(
                        block,
                        block_forward_fn=block_forward_fn,
                        block_attention_fn=block_attention_fn,
                    )

                if verbose:
                    logger.info(f"Registered hooks for {model.__class__.__name__} at '{transformer_blocks_name}'")
                return  # Exit after registering hooks for the first matching module

        logger.warning(
            f"No transformer blocks found with name '{transformer_blocks_name}' in {model.__class__.__name__}"
        )


def unregister_hooks(
    model: nn.Module,
    transformer_blocks_name: str = "model.layers",
    verbose: bool = True,
) -> None:
    """Unregister hooks for the model.
    Args:
        model (nn.Module): The model to unregister hooks from.
        transformer_blocks_name (str): The name of the transformer blocks in the model.
    """
    if hasattr(model, '_old_sample'):
        model._sample = model._old_sample
        delattr(model, '_old_sample')
        if verbose:
            logger.info(f"Unregistered diffusion sample hook for {model.__class__.__name__}")

    if transformer_blocks_name is not None:
        for name, module in model.named_modules():
            if name == transformer_blocks_name:
                module: nn.ModuleList
                for block in module:
                    if hasattr(block, "_block_forward"):
                        block.forward = block._block_forward  # type: ignore[attr-defined]
                        delattr(block, "_block_forward")

                    if hasattr(block.self_attn, "_block_attention"):
                        block.self_attn.forward = block.self_attn._block_attention  # type: ignore[attr-defined]
                        delattr(block.self_attn, "_block_attention")

                    # if hasattr(block.rotary_emb, "_rope_forward"):
                    #     block.rotary_emb.forward = block.rotary_emb._rope_forward  # type: ignore[attr-defined]
                    #     delattr(block.rotary_emb, "_rope_forward")
                if verbose:
                    logger.info(f"Unregistered hooks for {model.__class__.__name__} at '{transformer_blocks_name}'")
                return  # Exit after unregistering hooks for the first matching module

        logger.warning(
            f"No transformer blocks found with name '{transformer_blocks_name}' in {model.__class__.__name__}"
        )


class SpaCachedForwardHooks:
    def block_forward(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        debug_print = False
        # debug_print = SequenceMgmt.step in [0, 1, 2, 3, 4, 5]
        # debug_print = (
        #     module.self_attn.layer_idx in [0, 1]
        #     and SequenceMgmt.step in [0, 1, 2, 3, 4, 5]
        # )
        residual = hidden_states

        bsz, kv_len, _ = hidden_states.size()
        hidden_states = module.input_layernorm(hidden_states)

        # Self Attention
        query_states = module.self_attn.q_proj(hidden_states)
        key_states = module.self_attn.k_proj(hidden_states)
        value_states = module.self_attn.v_proj(hidden_states)
        query_states = query_states.view(
            bsz, kv_len, module.self_attn.num_heads, module.self_attn.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, kv_len, module.self_attn.num_key_value_heads, module.self_attn.head_dim).transpose(1, 2)
        value_states = value_states.view(
            bsz, kv_len, module.self_attn.num_key_value_heads, module.self_attn.head_dim).transpose(1, 2)

        cos, sin = position_embeddings # 2 * (B, L, D)
        if debug_print:
            logger.debug(f"[Before pos emb] {cos.shape=} {sin.shape=}")
        cos, sin = SpaCache.get_pos_emb(
            cos, sin,
            module.self_attn.layer_idx,
            SequenceMgmt.prompt_length,
        )
        if debug_print:
            logger.debug(f"[After pos emb] {cos.shape=} {sin.shape=}")
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Determine which tokens to update in this layer
        update_indices = SpaCache.check_update(
            hidden_states,
            module.self_attn.layer_idx,
            SequenceMgmt.prompt_length,
            # new_indicator=value_states,
        )
        if debug_print:
            logger.debug(f"---- Pre-Attn {SequenceMgmt.step=}, {module.self_attn.layer_idx=} ----")
            logger.debug(f"[Input] {residual.shape=} {query_states.shape=}\n{update_indices=}")
            # breakpoint()
        residual_update = SpaCache.get_update_tokens_h(update_indices, residual)
        query_states = SpaCache.get_update_tokens_q(update_indices, query_states)
        if debug_print:
            logger.debug(f"[After reduction] {residual_update.shape=} {query_states.shape=}")
        q_len = query_states.size(2)

        # if past_key_value is not None:
        #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        #     key_states, value_states = past_key_value.update(key_states, value_states, module.self_attn.layer_idx, cache_kwargs)

        if debug_print:
            logger.debug(f"[Before KV] {key_states.shape=} {value_states.shape=}")
        key_states, value_states = SpaCache.get_kv_heads(
            key_states,
            value_states,
            module.self_attn.layer_idx,
            SequenceMgmt.prompt_length,
            update_indices=update_indices,
        )
        if debug_print:
            logger.debug(f"[After KV] {key_states.shape=} {value_states.shape=}")

        key_states = repeat_kv(key_states, module.self_attn.num_key_value_groups)
        value_states = repeat_kv(value_states, module.self_attn.num_key_value_groups)

        # causal_mask = attention_mask
        # if attention_mask is not None:  # no matter the length, we just slice it
        #     causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        # is_causal = True if causal_mask is None and q_len > 1 else False
        if debug_print and isinstance(attention_mask, torch.Tensor):
                logger.debug(f"[Before attn_mask] {attention_mask.shape=}")
        attention_mask = SpaCache.get_attn_mask(
            attention_mask,
            module.self_attn.layer_idx,
            SequenceMgmt.prompt_length,
            update_indices,
        )
        if debug_print and isinstance(attention_mask, torch.Tensor):
            logger.debug(f"[After attn_mask] {attention_mask.shape=}")

        # attention_mask # (B, 1, L, L)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask if isinstance(attention_mask, torch.Tensor) else None,
            dropout_p=module.self_attn.attention_dropout if module.self_attn.training else 0.0,
            is_causal=False, # hard coded
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        try:
            attn_output = attn_output.view(bsz, q_len, module.self_attn.hidden_size)
        except Exception as e:
            logger.error(f"Error in reshaping attn_output: {e}")
            logger.error(f"{attn_output.shape=}, {bsz=}, {q_len=}, {module.self_attn.hidden_size=}")
            breakpoint()
        attn_output = module.self_attn.o_proj(attn_output)
        residual_update = residual_update + attn_output

        # Fully Connected
        # residual = hidden_states
        residual_update = module.post_attention_layernorm(residual_update)
        residual_update = module.mlp(residual_update)
        # hidden_states = residual + residual_update

        if debug_print:
            logger.debug(f"[Before FF] {residual_update.shape=} {attn_output.shape=}")
        residual = SpaCache.get_ff_output(
            ff_output=residual_update,
            attn_output=attn_output,
            attn_input=residual,
            layer_id=module.self_attn.layer_idx,
            prompt_length=SequenceMgmt.prompt_length,
            update_indices=update_indices,
        )
        if debug_print:
            logger.debug(f"[After FF] {residual.shape=}")
            if module.self_attn.layer_idx == 27:
                breakpoint()

        outputs = (residual,)

        if output_attentions:
            self_attn_weights = None
            outputs += (self_attn_weights,) # None

        if use_cache:
            outputs += (past_key_value,) # DynamicCache

        return outputs
