#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Created :   2025/08/26 09:00:01
@Desc    :
@Ref     :
"""
import logging
from typing import Optional

import torch

from .cache import SpaCache

logger = logging.getLogger(__name__)


class SequenceMgmt:

    @classmethod
    def set_hparams(
        cls,
        gen_length: int,
        mask_id: int = 126336,
        eot_id: int = 126348,
        pad_id: int = 126081,
        early_stop_steps: int = 10_000,
        refresh_steps: int = 10_000,
        refresh_gen_steps: int = 10_000,
        n_warmup_steps: int = 0,
        tokenizer = None,
    ) -> None:
        cls.gen_length = gen_length

        cls.mask_id = mask_id
        cls.eot_ids = torch.tensor([eot_id, pad_id], dtype=torch.long)
        cls.pad_id = pad_id

        cls.early_stop_steps = early_stop_steps
        cls.refresh_steps = refresh_steps
        cls.refresh_gen_steps = refresh_gen_steps
        cls.n_warmup_steps = n_warmup_steps

        # for debugging
        cls.tokenizer = tokenizer

    @classmethod
    def refresh(cls) -> bool:
        return (cls.step % cls.refresh_steps == 0) or (cls.step <= cls.n_warmup_steps)
    
    @classmethod
    def refresh_gen(cls) -> bool:
        return (cls.step % cls.refresh_gen_steps == 0) or (cls.step <= cls.n_warmup_steps)
    
    @classmethod
    def early_stop(cls) -> bool:
        return cls.step % cls.early_stop_steps == 0

    @classmethod
    def _load_input(cls) -> torch.LongTensor:
        return cls.output_ids[~cls.finished]

    @classmethod
    def _dump_output(cls, output_ids: torch.LongTensor) -> None:
        if output_ids.shape[1] != cls.prompt_length + cls.gen_length:
            raise ValueError(
                f"Unexpected output shape: {output_ids.shape},"
                f" expected {(output_ids.shape[0] - cls.finished.sum(), cls.prompt_length + cls.gen_length)}"
                f" (batch_size={output_ids.shape[0]}, finished={cls.finished.sum()})"
            )
        cls.output_ids[~cls.finished] = output_ids

    @classmethod
    def _possible_pad(
        cls,
        output_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        early_stop = torch.nonzero(
            torch.isin(output_ids[:, cls.prompt_length:], cls.eot_ids), as_tuple=True)

        if early_stop[0].numel() > 0:
            for batch_idx, token_idx in zip(*early_stop):
                output_ids[batch_idx, token_idx + cls.prompt_length + 1:] = cls.pad_id
        return output_ids

    @classmethod
    def get_model_input(
        cls,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.LongTensor:
        # reset cache
        SpaCache.reset_cache(empty_cache=True)
        # step counters
        cls.step = 0
        # sub-sequence lengths
        cls.prompt_length = input_ids.shape[1]
        cls.max_length = cls.prompt_length + cls.gen_length

        # output ids: [prompt, gen_len]
        output_ids = torch.full(
            (input_ids.shape[0], cls.gen_length),
            fill_value=cls.mask_id,
            device=input_ids.device,
            dtype=torch.long,
        )
        cls.output_ids = torch.cat((input_ids, output_ids), dim=1)

        # for early termination
        cls.finished = torch.zeros(
            input_ids.shape[0], dtype=torch.bool, device=input_ids.device
        )
        cls.eot_ids = cls.eot_ids.to(input_ids.device)

        input_ids = cls._load_input()
        # logger.debug(f"{input_ids.shape=}")
        # breakpoint()
        return input_ids

    @classmethod
    def next_step(
        cls, 
        output_ids: torch.LongTensor,
        steps: int = 1,
        return_attn_mask: bool = False,
        attn_mask: Optional[torch.BoolTensor] = None,
        tok_idx: Optional[torch.LongTensor] = None,
    ) -> torch.LongTensor:
        cls.step += steps

        if cls.refresh():
            SpaCache.reset_cache() # all: [prompt, gen]
        elif cls.refresh_gen():
            SpaCache.reset_cache(sequence_entry_names=['gen'])

        if cls.early_stop():
            # breakpoint()
            output_ids = cls._possible_pad(output_ids)

        finished = (output_ids != cls.mask_id).all(dim=1)
        if finished.any():
            cls._dump_output(output_ids)
            cls.finished[~cls.finished] = finished
            SpaCache.mark_finished(finished)
            if return_attn_mask:
                if isinstance(attn_mask, torch.Tensor):
                    attn_mask = attn_mask[~finished]
                if isinstance(tok_idx, torch.Tensor):
                    tok_idx = tok_idx[~finished]

        if return_attn_mask:
            if cls.finished.all():
                return None, None, None
            elif finished.any():
                return cls._load_input(), attn_mask, tok_idx
            else:
                return output_ids, attn_mask, tok_idx
        else:
            if cls.finished.all():
                return None
            elif finished.any():
                return cls._load_input()
            else:
                return output_ids
