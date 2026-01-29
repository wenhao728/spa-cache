import json
import logging
import os
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import torch
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

project_dir = str(Path(__file__).resolve().parents[2])
if project_dir not in sys.path:
    sys.path.append(project_dir)

from spac.cache import SpaCache, get_update_ratio
from spac.llada import (
    SpaCachedForwardHooks,
    generate,
    generate_spa_cache,
    register_hooks,
)
from spac.sequence import SequenceMgmt
from spac.utils import (
    BenchmarkLogger,
    add_logging_output,
    load_proxy_projection_from_model,
    set_seed,
)

logger = logging.getLogger(__name__)


@register_model("LLaDA")
class LLaDA(LM):

    def __init__(
        self,
        pretrained_model_path: str,
        batch_size: int = 1,
        save_dir: Optional[os.PathLike] = None,
        log_dir: Optional[os.PathLike] = None,
        exp_name: str = "01_default",
        log_latency: int = 4,
        log_ttft: int = 4,
        log_warmup: int = 2,
        device: Optional[torch.device] = None,
        add_bos_token: bool = False,
        escape_until: Optional[bool] = False, # humaneval
        # dLLM
        steps: int = 256,
        gen_length: int = 256,
        block_length: int = 32,
        mask_id: int = 126336,
        pad_id: int = 126081,
        # cache args
        cache: Literal[None, "spa"] = None,
        proxy_rank: float = 256,
        max_update_ratio: Optional[float] = None,
        avg_update_ratio: Optional[float] = None,
        min_update_ratio: float = 2 ** (-4),
        update_ratio_dist: Literal['uniform', 'gaussian'] = 'gaussian',
        refresh_steps: int = 10_000,
        refresh_gen_steps: int = 10_000,
        early_stop_steps: int = 1,
        # parallel
        threshold: Optional[float] = None,
        dynamic_threshold_factor: Optional[float] = None,
    ):
        super().__init__()
        # set_seed(1234)

        if save_dir is not None:
            save_dir = Path(save_dir)
            self.save_file = save_dir / f"{exp_name.lower()}.jsonl"
        else:
            if log_dir is None:
                raise ValueError("Either save_dir or log_dir must be provided.")
            save_dir = Path(log_dir)
            self.save_file = None

        save_dir.mkdir(parents=True, exist_ok=True)
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        add_logging_output(save_dir / f"{exp_name.lower()}-{time_str}.log")
        args_json_str = json.dumps(locals(), default=lambda o: None, indent=4)
        logger.info(f"Arguments: {args_json_str}")
        logger.info(f"Experiment name: {exp_name}")
        logger.info(f"CUDA device: {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}")
        
        self.log_latency = log_latency
        self.log_ttft = log_ttft # ttft may lead to some overhead
        self.log_warmup = log_warmup

        self.batch_size = int(batch_size)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        torch.cuda.reset_peak_memory_stats(self.device)

        self.add_bos_token = add_bos_token
        self.escape_until = escape_until
        # self.use_chat_template = False
        # if "base" not in pretrained_model_path.split("/")[-1].lower():
        #     logger.info("Using Instruct model, applying chat template to the prompt.")
        #     self.use_chat_template = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_path,
            trust_remote_code=True,
        )

        # Model
        config = AutoConfig.from_pretrained(
            pretrained_model_path, trust_remote_code=True
        )
        # config.flash_attention = True
        self.model = AutoModel.from_pretrained(
            pretrained_model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        self.model = self.model.to(self.device)

        # Generate func
        self.generate_kwargs = dict(
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            mask_id=mask_id,
            log_ttft=False,
        )
        self.generate_func = generate
        self.pad_id = pad_id

        # Cache
        if cache == "spa":
            SpaCache.set_hparams(
                proxy_projs=load_proxy_projection_from_model(
                    self.model,
                    'model.transformer.blocks',
                    proxy_rank=proxy_rank,
                    device=self.device,
                    feature_save_dir=save_dir.parent / 'svd_cache',
                ),
                update_ratio=get_update_ratio(
                    pretrained_model_path,
                    freq_dist=update_ratio_dist,
                    max_update_ratio=max_update_ratio,
                    avg_update_ratio=avg_update_ratio,
                    min_update_ratio=min_update_ratio,
                ),
            )
            forward_hooks = SpaCachedForwardHooks()
            register_hooks(
                self.model,
                "model.transformer.blocks",
                forward_hooks.block_forward,
                forward_hooks.block_attention,
                forward_hooks.rope_forward,
            )
            SequenceMgmt.set_hparams(
                gen_length=gen_length,
                mask_id=mask_id,
                pad_id=pad_id,
                early_stop_steps=early_stop_steps,
                refresh_steps=refresh_steps,
                refresh_gen_steps=refresh_gen_steps,
                tokenizer=self.tokenizer,
            )
            self.generate_kwargs.pop("gen_length")
            self.generate_kwargs.pop("mask_id")
            self.generate_kwargs.update(
                {'threshold': threshold, 'dynamic_threshold_factor': dynamic_threshold_factor}
            )
            self.generate_func = generate_spa_cache
        elif cache is not None:
            raise NotImplementedError(f"Cache type {cache} not implemented.")

    def _batch_encode(
        self,
        input_text: List[str],
        padding_side: str = "left",
    ):
        # # encode a batch of strings. converts to tensors and pads automatically
        # if self.use_chat_template:
        #     input_text = self.tokenizer.apply_chat_template(
        #         [[{"role": "user", "content": text}] for text in input_text],
        #         add_generation_prompt=True,
        #         tokenize=False,
        #     )
        # else:
        if self.add_bos_token:
            input_text = [self.tokenizer.bos_token + text for text in input_text]

        return self.tokenizer(
            input_text,
            padding_side=padding_side,
            padding="longest",
            return_tensors="pt",
        )

    def _truncate_stop_token(
        self,
        output_text: List[str],
        stop_tokens: List[str],
        num_valid_tokens: int,
    ) -> List[str]:
        # Truncate the output text after the stop token
        truncated_text = []
        for text in output_text:
            for stop_token in stop_tokens:
                if len(stop_token) > 0:
                    text = text.split(stop_token)[0]
            # req.args[1]['until'] may not contain all special tokens, e.g. <|eot_id|>
            output_id = self.tokenizer(text).input_ids
            text = self.tokenizer.decode(output_id, skip_special_tokens=True)
            # num_valid_tokens += sum([o_id != self.pad_id for o_id in output_id])

            truncated_text.append(text)
        return truncated_text

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.split("/")[-1]

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

        return chat_templated

    def loglikelihood_rolling(
        self,
        requests: List[Instance],
    ) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def loglikelihood(
        self,
        requests: List[Instance],
    ) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def generate_until(self, requests: List[Instance]) -> List[str]:
        # breakpoint()
        # Resume
        if self.save_file is not None and self.save_file.exists():
            with open(self.save_file, "r", encoding="utf-8") as f:
                pre_outputs = [json.loads(line) for line in f]
                num_finished = len(pre_outputs)
            logger.info(f"Finished {num_finished} outputs from {self.save_file}")
        else:
            num_finished = 0
            pre_outputs = []

        # Dataset
        ds = []
        for i, req in enumerate(requests):
            if i < num_finished:
                continue
            ds.append({"input_text": req.args[0]})
        ds = Dataset.from_list(ds)

        # Generate
        # gen_kwargs = req.args[1] if len(requests) > 0 else {}
        stop_tokens = req.args[1].get("until", None)
        outputs = []
        num_valid_tokens = 0
        bench_logger = BenchmarkLogger(log_latency=self.log_latency > 0)
        total_ttft = 0.0

        progress_bar = tqdm(total=(len(pre_outputs) + len(ds)) // self.batch_size)
        progress_bar.update(len(pre_outputs) // self.batch_size)

        for batch_id, batch in enumerate(
            ds.iter(batch_size=self.batch_size, drop_last_batch=False)
        ):
            # inputs
            input_ids = self._batch_encode(batch["input_text"])["input_ids"]
            input_ids = input_ids.to(self.device)

            # ttft
            if self.log_ttft > 0:
                if batch_id == self.log_warmup:
                    self.generate_kwargs.update(dict(log_ttft=True))
                elif batch_id == self.log_warmup + self.log_ttft:
                    self.generate_kwargs.update(dict(log_ttft=False))

            # tps
            context = nullcontext()
            log_latency = (
                batch_id >= self.log_warmup + self.log_ttft and 
                batch_id < self.log_warmup + self.log_ttft + self.log_latency
            )
            if log_latency:
                context = bench_logger

            with context:
                output_ids = self.generate_func(self.model, input_ids, **self.generate_kwargs)
            # breakpoint()

            # log latency
            if self.log_ttft > 0 and (self.log_warmup <= batch_id < self.log_warmup + self.log_ttft):
                # breakpoint()
                output_ids, ttft_latency = output_ids
                total_ttft += ttft_latency

            output_ids = output_ids[:, input_ids.shape[1] :]

            if log_latency:
                num_valid_tokens += output_ids.ne(self.pad_id).sum().item()

            output_text = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=self.escape_until,
            )
            if not self.escape_until:
                output_text = self._truncate_stop_token(
                    output_text, stop_tokens, None
                )

            outputs.extend(output_text)
            progress_bar.update(1)
            if self.save_file is not None:
                with open(self.save_file, "a", encoding="utf-8") as f:
                    for text in output_text:
                        f.write(json.dumps(text, ensure_ascii=False) + "\n")
            else: # no save file, for test latency
                if batch_id >= self.log_warmup + self.log_ttft + self.log_latency:
                    break

        n_outputs = len(outputs)
        n_outputs_ttft = self.log_ttft * self.batch_size
        n_outputs_latency = self.log_latency * self.batch_size
        max_allocated_bytes = torch.cuda.max_memory_allocated(self.device)

        if n_outputs > 0:
            log_str = (
                f"\n# outputs   : {n_outputs:,}"
                f"\nMax GPU mem : {max_allocated_bytes / 2**30:.2f} GB"
            )
            if self.log_latency > 0 and bench_logger.latency > 0:
                log_str += (
                    f"\n" + "="*20 +
                    f"\nvalid tokens: {num_valid_tokens:.2e}"
                    f"\nGPU elapsed : {bench_logger.latency:.2f} sec"
                    f"\n\ttokens / output : {num_valid_tokens / n_outputs_latency:.2f}"
                    f"\n\tseconds / output: {bench_logger.latency / n_outputs_latency:.2f}"
                    f"\n\ttokens / second : {num_valid_tokens / bench_logger.latency:.2f} tokens/sec"
                )
            if self.log_ttft > 0 and n_outputs_ttft > 0:
                log_str += (
                    f"\n" + "="*20 +
                    f"\n\tTTFT (seconds): {total_ttft / n_outputs_ttft:.6f}"
                )
            logger.info(log_str)
        logger.info(f"Finished generating outputs.")

        return pre_outputs + outputs


if __name__ == "__main__":
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # set_seed(1234)
    cli_evaluate()  # from lm_eval.__main__
