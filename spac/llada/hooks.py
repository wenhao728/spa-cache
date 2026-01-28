import logging
import types
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from ..cache import SpaCache
from ..sequence import SequenceMgmt
from .modeling_utils import get_rotary_embedding

logger = logging.getLogger(__name__)


def register_block_hooks(
    block: nn.Module,
    block_forward_fn: Optional[Callable] = None,
    block_attention_fn: Optional[Callable] = None,
    rope_forward_fn: Optional[Callable] = None,
) -> None:
    if block_forward_fn is not None:
        setattr(block, "_block_forward", block.forward)
        block.forward = types.MethodType(block_forward_fn, block)

    if block_attention_fn is not None:
        setattr(block, "_block_attention", block.attention)
        block.attention = types.MethodType(block_attention_fn, block)

    if rope_forward_fn is not None:
        setattr(block.rotary_emb, "_rope_forward", block.rotary_emb.forward)
        block.rotary_emb.forward = types.MethodType(
            rope_forward_fn, block.rotary_emb
        )
    # if get_rope_fn is not None:
    #     setattr(block.rotary_emb, "_get_rotary_embedding", block.rotary_emb.get_rotary_embedding)
    #     block.rotary_emb.get_rotary_embedding = types.MethodType(get_rope_fn, block.rotary_emb)


def register_hooks(
    model: nn.Module,
    transformer_blocks_name: str = "model.transformer.blocks",
    block_forward_fn: Optional[Callable] = None,
    block_attention_fn: Optional[Callable] = None,
    rope_forward_fn: Optional[Callable] = None,
    # get_rope_fn: Optional[Callable] = None,
) -> None:
    """
    Register hooks for the model.
    """
    for name, module in model.named_modules():
        if name == transformer_blocks_name:
            module: nn.ModuleList
            for block in module:
                register_block_hooks(
                    block,
                    block_forward_fn=block_forward_fn,
                    block_attention_fn=block_attention_fn,
                    rope_forward_fn=rope_forward_fn,
                )

            logger.info(
                f"Registered hooks for {model.__class__.__name__} at '{transformer_blocks_name}'"
            )
            return  # Exit after registering hooks for the first matching module

    logger.warning(
        f"No transformer blocks found with name '{transformer_blocks_name}' in {model.__class__.__name__}"
    )


def unregister_block_hooks(
    block: nn.Module,
) -> None:
    if hasattr(block, "_block_forward"):
        block.forward = block._block_forward  # type: ignore[attr-defined]
        delattr(block, "_block_forward")

    if hasattr(block, "_block_attention"):
        block.attention = block._block_attention  # type: ignore[attr-defined]
        delattr(block, "_block_attention")

    if hasattr(block.rotary_emb, "_rope_forward"):
        block.rotary_emb.forward = block.rotary_emb._rope_forward  # type: ignore[attr-defined]
        delattr(block.rotary_emb, "_rope_forward")

    # if hasattr(block.rotary_emb, "_get_rotary_embedding"):
    #     block.rotary_emb.get_rotary_embedding = block.rotary_emb._get_rotary_embedding
    #     delattr(block.rotary_emb, "_get_rotary_embedding")


def unregister_hooks(
    model: nn.Module,
    transformer_blocks_name: str = "model.transformer.blocks",
) -> None:
    """Unregister hooks for the model.
    Args:
        model (nn.Module): The model to unregister hooks from.
        transformer_blocks_name (str): The name of the transformer blocks in the model.
    """

    for name, module in model.named_modules():
        if name == transformer_blocks_name:
            module: nn.ModuleList
            for block in module:
                unregister_block_hooks(block)

            logger.info(
                f"Unregistered hooks for {model.__class__.__name__} at '{transformer_blocks_name}'"
            )
            return  # Exit after unregistering hooks for the first matching module

    logger.warning(
        f"No transformer blocks found with name '{transformer_blocks_name}' in {model.__class__.__name__}"
    )


class SpaCachedForwardHooks:
    def block_forward(
        self,
        module: nn.Module,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        """
        A new forward method that will be used to replace the original forward method.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]: The output tensor and optional cache.
        """
        do_print = False
        # do_print = SequenceMgmt.step in [0, 1, 2, 3, 4, 5]
        # do_print = (
        #     module.layer_id in [0, 1]
        #     and SequenceMgmt.step in [0, 1, 2, 3, 4, 5]
        # )
        
        x_normed = module.attn_norm(x)
        # full kv with update
        k = module.k_proj(x_normed)
        v = module.v_proj(x_normed)

        # Get update indices
        update_indices = SpaCache.check_update(
            x_normed,
            module.layer_id,
            SequenceMgmt.prompt_length,
            new_indicator=None,
        )
        if do_print:
            logger.debug(
                f"[Input] {SequenceMgmt.step=}, {module.layer_id=}"
                f"\n{x.shape=}, {update_indices=}"
            )
            # breakpoint()

        x_update, x_normed = SpaCache.get_attn_input(x, x_normed, update_indices)
        if do_print:
            logger.debug(f"[After attn_input] {x_update.shape=} {x_normed.shape=}")
            # breakpoint()
            # print(f"[After attn_input] {x.shape=}")

        q = module.q_proj(x_normed)

        # Get attention scores.
        if module._activation_checkpoint_fn is not None:
            att = module._activation_checkpoint_fn(  # type: ignore
                module.attention, q, k, v, module.layer_id, update_indices
            )
        else:
            att = module.attention(q, k, v, module.layer_id, update_indices)

        att = module.dropout(att)
        x_update = x_update + att

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        # og_x = x
        if module._activation_checkpoint_fn is not None:
            x_update = module._activation_checkpoint_fn(module.ff_norm, x_update)  # type: ignore
        else:
            x_update = module.ff_norm(x_update)
        x_update, x_up = module.ff_proj(x_update), module.up_proj(x_update)  # new add
        if module._activation_checkpoint_fn is not None:
            x_update = module._activation_checkpoint_fn(module.act, x_update)  # type: ignore
        else:
            x_update = module.act(x_update)
        # x = x * x_up
        torch.mul(x_update, x_up, out=x_update)
        x_update = module.ff_out(x_update)

        x_update = module.dropout(x_update)
        if do_print:
            logger.debug(f"[Before ff] {x_update.shape=}")
            # print(f"[Before ff] {x_update.shape=}")

        # x = og_x + x
        x = SpaCache.get_ff_output(
            ff_output=x_update,
            attn_output=att,
            attn_input=x,
            layer_id=module.layer_id,
            prompt_length=SequenceMgmt.prompt_length,
            update_indices=update_indices,
        )

        if do_print:
            logger.debug(f"[After ff] {x.shape=}")
            # print(f"[After ff] {x.shape=}")
            if module.layer_id in [0, 1, 30, 31]:
                breakpoint()

        cache = None
        return x, cache

    def block_attention(
        self,
        module: nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        update_indices,
    ) -> torch.Tensor:
        """
        A new attention method that will be used to replace the original attention method.
        Args:
            q, k, v (torch.Tensor): The input tensor for queries, keys, and values.
        Returns:
            Tuple[torch.Tensor, dict]: The output tensor and attention intermediate features.
        """
        do_print = False
        # do_print = SequenceMgmt.step in [0, 1, 2, 3, 4, 5]
        # do_print = (
        #     module.layer_id in [0, 1]
        #     and SequenceMgmt.step in [0, 1, 2, 3, 4, 5]
        # )

        B, query_len, C = q.size()  # batch size, sequence length, d_model
        key_len, value_len = (
            k.shape[-2],
            v.shape[-2],
        )  # could be different from query_len if caching is used
        dtype = k.dtype

        # Optionally apply layer norm to keys and queries.
        if module.q_norm is not None and module.k_norm is not None:
            q = module.q_norm(q).to(dtype=dtype)
            k = module.k_norm(k).to(dtype=dtype)

        # Move head forward to be next to the batch dim.
        # shape: (B, nh, T, hs)
        q = q.view(
            B, query_len, module.config.n_heads, C // module.config.n_heads
        ).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        k = k.view(
            B, key_len, module.config.effective_n_kv_heads, C // module.config.n_heads
        ).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        v = v.view(
            B, value_len, module.config.effective_n_kv_heads, C // module.config.n_heads
        ).transpose(1, 2)

        if module.config.rope:
            # Apply rotary embeddings.
            if do_print:
                logger.debug(
                    f"[RoPE] update_indices={update_indices.shape if update_indices is not None else update_indices}"
                )
                # breakpoint()
            q, k = module.rotary_emb(q, k, update_indices, module.layer_id)

        if do_print:
            logger.debug(f"[Before KV] {k.shape=} {v.shape=}")

        k, v = SpaCache.get_kv_heads(
            k,
            v,
            layer_id,
            SequenceMgmt.prompt_length,
            update_indices,
        )
        if do_print:
            logger.debug(f"[After KV] {k.shape=} {v.shape=}")
            # breakpoint()

        # Get the attention scores.
        # shape: (B, nh, T, hs)
        att = module._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0 if not module.training else module.config.attention_dropout,
            is_causal=False,
        )

        # Re-assemble all head outputs side-by-side.
        att = att.transpose(1, 2).contiguous().view(B, query_len, C)

        # Apply output projection.
        return module.attn_out(att)

    def rope_forward(
        self,
        module: nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        update_indices: Optional[torch.LongTensor],
        layer_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        A new forward method that will be used to replace the original forward method of RotaryEmbedding.
        Args:
            q (torch.Tensor): The query tensor.
            k (torch.Tensor): The key tensor.
            update_indices (torch.LongTensor): The update indices tensor.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The output query and key tensors.
        """
        do_print = False
        # do_print = module.layer_id in [0, 1, 27, 31]
        # do_print = (
        #     layer_id in [0, 1]
        #     and SequenceMgmt.step in [0, 1, 2, 3, 4, 5]
        # )

        if module.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        # (B, H, N, D)
        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = (
                q_.shape[-2],
                k_.shape[-2],
            )  # could be different if layer_past not None

            # (B, L_update)
            if update_indices is None:
                # dense
                rope_start = 0 if SequenceMgmt.refresh() else SequenceMgmt.prompt_length
                # assert key_len == SequenceMgmt.gen_length # TODO: check possible bug
                # (1, 1, key_len, dim)
                pos_sin, pos_cos = get_rotary_embedding(
                    module,
                    rope_start,
                    key_len,
                    q_.device,
                )
                pos_sin = pos_sin.type_as(q_)
                pos_cos = pos_cos.type_as(q_)
                q_ = module.apply_rotary_pos_emb(pos_sin, pos_cos, q_)
                k_ = module.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
            else:
                pos_sin, pos_cos = get_rotary_embedding(
                    module,
                    SequenceMgmt.prompt_length,
                    SequenceMgmt.gen_length,
                    q_.device,
                )
                pos_sin = pos_sin.type_as(q_)
                pos_cos = pos_cos.type_as(q_)

                update_indices = update_indices[:, None, :, None].expand(
                    -1, -1, -1, pos_sin.shape[3]
                )  # (B, 1, L_update, dim)
                pos_sin_q = torch.gather(
                    pos_sin.expand(q_.shape[0], 1, -1, -1),
                    dim=2,
                    index=update_indices,
                )
                pos_cos_q = torch.gather(
                    pos_cos.expand(q_.shape[0], 1, -1, -1),
                    dim=2,
                    index=update_indices,
                )
                q_ = module.apply_rotary_pos_emb(pos_sin_q, pos_cos_q, q_)
                k_ = module.apply_rotary_pos_emb(pos_sin, pos_cos, k_)

        return q_.type_as(q), k_.type_as(k)
