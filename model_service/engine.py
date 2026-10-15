"""Transformers + bitsandbytes CUDA inference, one model resident at a time."""

import gc
import json
import logging
import os
import time

from jsonschema import Draft7Validator

from .protocol import strict_json, validate_chat


class CudaEngine:
    def __init__(self, keep_loaded=False, timeout=120):
        # Conda's Intel MKL must not conflict with PyTorch's GNU OpenMP runtime.
        os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; activate the CUDA PyTorch environment")
        self.torch = torch
        self.keep_loaded, self.timeout = keep_loaded, timeout
        self.path = self.model = self.processor = self.grammar_compiler = None
        logging.info("CUDA=%s torch=%s GPU=%s keep_loaded=%s", torch.version.cuda,
                     torch.__version__, torch.cuda.get_device_name(0), keep_loaded)

    def unload(self):
        self.model = self.processor = self.path = self.grammar_compiler = None
        gc.collect()
        self.torch.cuda.empty_cache()

    def load(self, path, vision):
        if self.path == path:
            return
        self.unload()
        from transformers import (AutoModelForCausalLM, AutoTokenizer, AutoProcessor,
                                  BitsAndBytesConfig, Qwen3VLForConditionalGeneration)

        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=self.torch.float16,
            llm_int8_skip_modules=["visual", "lm_head"])
        loader = Qwen3VLForConditionalGeneration if vision else AutoModelForCausalLM
        self.model = loader.from_pretrained(str(path), local_files_only=True,
            trust_remote_code=False, use_safetensors=True, device_map={"": 0},
            dtype=self.torch.float16, quantization_config=quantization,
            attn_implementation="sdpa").eval()
        processor = AutoProcessor if vision else AutoTokenizer
        self.processor = processor.from_pretrained(str(path), local_files_only=True,
                                                    trust_remote_code=False)
        self.path = path
        logging.info("Loaded %s on CUDA; allocated=%.2f GiB", path.name,
                     self.torch.cuda.memory_allocated()/2**30)

    def chat(self, body, path, vision):
        messages, output_schema, context, tokens = validate_chat(body, vision)
        start = time.monotonic()
        try:
            self.load(path, vision)
            loaded = time.monotonic()
            processor = self.processor
            if vision:
                inputs = processor.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, return_dict=True, return_tensors="pt")
                tokenizer = processor.tokenizer
            else:
                prompt = processor.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                inputs = processor(prompt, return_tensors="pt")
                tokenizer = processor
            inputs = inputs.to("cuda:0")
            prompt_length = inputs["input_ids"].shape[-1]
            if prompt_length + tokens > context:
                raise ValueError("Prompt plus output exceeds context; no silent truncation")
            kwargs = {}
            if output_schema is not None:
                import xgrammar as xgr
                from xgrammar.contrib.hf import LogitsProcessor
                if self.grammar_compiler is None:
                    vocab_size = self.model.get_output_embeddings().weight.shape[0]
                    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size,
                        stop_token_ids=self.model.generation_config.eos_token_id)
                    self.grammar_compiler = xgr.GrammarCompiler(info)
                grammar = self.grammar_compiler.compile_json_schema(json.dumps(output_schema))
                kwargs["logits_processor"] = [LogitsProcessor(grammar)]
            self.torch.cuda.synchronize()
            prepared = time.monotonic()
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=tokens,
                    do_sample=False, max_time=self.timeout, **kwargs)
            self.torch.cuda.synchronize()
            finished = time.monotonic()
            generated = output[0, prompt_length:]
            content = tokenizer.decode(generated, skip_special_tokens=True)
            eos = self.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            if not len(generated) or generated[-1].item() not in eos:
                raise RuntimeError("Generation incomplete (token/time limit); task rejected")
            if output_schema is not None:
                Draft7Validator(output_schema).validate(strict_json(content))
            logging.info("model=%s load=%.3fs prepare=%.3fs generate=%.3fs tokens=%s",
                         body["model"], loaded-start, prepared-loaded, finished-prepared, len(generated))
            return {"model": body["model"], "done": True, "done_reason": "stop",
                "message": {"role": "assistant", "content": content},
                "total_duration": int((finished-start)*1e9),
                "load_duration": int((loaded-start)*1e9),
                "prompt_eval_duration": int((prepared-loaded)*1e9),
                "eval_duration": int((finished-prepared)*1e9),
                "eval_count": len(generated), "prompt_eval_count": prompt_length,
                "backend": "pytorch-cuda", "device": self.torch.cuda.get_device_name(0)}
        except Exception:
            self.unload()
            raise
        finally:
            if not self.keep_loaded:
                self.unload()
