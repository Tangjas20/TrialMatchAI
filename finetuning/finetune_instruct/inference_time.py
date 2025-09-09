import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from statistics import mean, stdev
import importlib.util
import os
import csv
from datetime import datetime
import logging

import GPUtil
import numpy as np
import traceback
import threading
from transformers import TextIteratorStreamer
from typing import Optional, Dict, Any, List, Union

from transformers import HfArgumentParser
from arguments import ModelArguments, DataArguments, SFTTrainingArguments as TrainingArguments
from peft import PeftModel, PeftConfig

logger = logging.getLogger(__name__)
logger.warning('First warning')

def main():

    logger.warning('In Python script')

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if model_args.use_unsloth is True and model_args.use_vllm is True:
        raise ValueError("Cannot use both Unsloth and vLLM. Set only one to True.")

    # Optional importing
    if model_args.use_unsloth is True:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ["HF_HUB_READ_TIMEOUT"] = "60"

        import torch.distributed as dist
        if not dist.is_initialized():
            print('pass')
            dist.init_process_group(backend="gloo", rank=0, world_size=1)
        from unsloth import FastLanguageModel

    elif model_args.use_vllm is True:
        from vllm import LLM, SamplingParams
        from vllm.prompt_adapter.utils import load_peft_weights

    # ----- Config -----
    MODEL_NAME = model_args.model_name_or_path  # Use either Unsloth or HF model path
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WARMUP = training_args.num_warmup
    NUM_RUNS = training_args.num_runs

    PROMPT = (
        "You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. "
        "Please answer the following medical question. Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.\n"
        "What is the mechanism of action of acetazolamide in the treatment of acute mountain sickness?\n"
        "<think>"
    )

    MAX_NEW_TOKENS = training_args.num_max_new_tokens
    BATCH_SIZE = training_args.batch_size
    MEASURE_TOKEN_LEVEL = True

    # Detect GPU Info
    gpu_name = "CPU"
    if DEVICE == "cuda":
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    logger.warning(f"Running on device: {gpu_name}")

    # ----- Load Model -----

    if model_args.use_unsloth is True:
        # Load with Unsloth acceleration
        os.environ["UNSLOTH_USE_NEW_MODEL"] = "1"
        model, tokenizer = FastLanguageModel.from_pretrained(
            MODEL_NAME,
            device_map=None,
            dtype=None,
            trust_remote_code=True,
            load_in_4bit=model_args.use_4bit
        )
        # model = FastLanguageModel.get_peft_model(model,
        #                                          r=model_args.lora_rank,
        #                                          lora_alpha=model_args.lora_alpha,
        #                                          lora_dropout=model_args.lora_dropout,
        #                                          target_modules=model_args.target_modules,
        #                                          bias="none",
        #                                          use_gradient_checkpointing="unsloth")

        # Load fine-tuned weights
        model = PeftModel.from_pretrained(model, model_args.from_peft)

        FastLanguageModel.for_inference(model) #Enable 2x faster native inference
        
    elif model_args.use_vllm is True:
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.9,
            max_tokens=MAX_NEW_TOKENS,
        )
        model = LLM(
            model=model_args.model_name_or_path,
            dtype="auto",  # or "float16", depending on your setup
            trust_remote_code=True,
        )
        # Load the PEFT adapter weights
        adapter_weights = load_peft_weights(model_args.from_peft, device="cuda")

        # Apply the adapter weights to the model
        model.apply_adapter(adapter_weights)

        logger.warning("Model loaded with weights.")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    # Tokenize input
    inputs = tokenizer([PROMPT] * BATCH_SIZE, return_tensors="pt", padding=True).to(DEVICE)

    def generate_and_time():
        if MEASURE_TOKEN_LEVEL: #time per token
            gen_times = []
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            past_key_values = None

            for _ in range(MAX_NEW_TOKENS):
                start = time.time()
                with torch.no_grad():
                    if model_args.use_unsloth is True:
                        outputs = model.generate(
                            input_ids = input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens = 1,
                            use_cache = True,
                            temperature = 1.5,
                            min_p = 0.1,
                            return_dict_in_generate=True,
                            output_scores=True,
                            output_hidden_states=False
                    )
                    else:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                end = time.time()
                gen_times.append(end - start)
                #logger.warning(outputs)

                next_token_logits = outputs.scores[-1]  # [B, V]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((attention_mask.size(0), 1), device=DEVICE)], dim=1
                )

                if hasattr(outputs, "sequences"):
                    logger.warning("Model output : %s", tokenizer.batch_decode(outputs.sequences))
                else:
                    logger.warning("No sequences to decode.")

            return gen_times
        else: #from start to finish
            start = time.time()
            with torch.no_grad():
                if model_args.use_unsloth is True:
                    outputs = model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                else:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        use_cache=True,
                    )
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            end = time.time()
            # for testing, decode using tokenizer
            logger.warning("Model output : %s", tokenizer.batch_decode(outputs))
            return end - start



    class GPUMonitor:
        """
        GPU monitoring class using GPUtil.
        Tracks peak GPU memory and utilization.
        """
        def __init__(self, monitoring_interval: float = 0.1):
            self.monitoring_interval = monitoring_interval
            self._gpu_memory_usage = []
            self._gpu_utilization = []
            self._is_monitoring = False
            self._monitoring_thread = None

        def start(self):
            """Start GPU monitoring"""
            self._is_monitoring = True
            self._gpu_memory_usage = []
            self._gpu_utilization = []

            def monitor_gpu():
                while self._is_monitoring:
                    try:
                        # Get GPU information
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]  # Assuming first GPU
                            self._gpu_memory_usage.append(gpu.memoryUsed)
                            self._gpu_utilization.append(gpu.load * 100)

                        # Wait for next interval
                        time.sleep(self.monitoring_interval)
                    except Exception as e:
                        print(f"GPU monitoring error: {e}")
                        break

            self._monitoring_thread = threading.Thread(target=monitor_gpu)
            self._monitoring_thread.start()

        def stop(self):
            """Stop GPU monitoring"""
            self._is_monitoring = False
            if self._monitoring_thread:
                self._monitoring_thread.join()

        def get_peak_usage(self) -> float:
            """Get peak GPU memory usage in MB"""
            return max(self._gpu_memory_usage) if self._gpu_memory_usage else 0

        def get_peak_utilization(self) -> float:
            """Get peak GPU utilization percentage"""
            return max(self._gpu_utilization) if self._gpu_utilization else 0
        
        def get_p90_usage(self) -> float:
            """Get P90 GPU memory usage in MB"""
            return np.percentile(self._gpu_memory_usage, 90) if self._gpu_memory_usage else 0
        
        def get_p90_utilization(self) -> float:
            """Get P90 GPU utilization percentage"""
            return np.percentile(self._gpu_utilization, 90) if self._gpu_utilization else 0

    def benchmark_single_prompt(
            model,
            tokenizer,
            input_prompt_text: str,
            temperature: float = 1.0,
            top_p: float = 0.95,
            max_new_tokens: int = 100,
            device: Optional[str] = None) -> Dict[str, Any]:
        """
        Benchmark a language model's performance for a single prompt.
        """
        # Determine device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        model.eval()

        # GPU monitoring setup
        gpu_monitor = GPUMonitor()
        gpu_monitor.start()

        # Tokenize input
        start_input_process = time.time()
        inputs = tokenizer(input_prompt_text, return_tensors="pt").to(device)
        input_process_time = time.time() - start_input_process

        # Streaming generation setup
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=False)
        generation_start_time = time.time()
        first_token_time = None
        generated_decoded_tokens = []

        # Streaming generation loop
        try:
            generation_kwargs = {
                'input_ids': inputs.input_ids,
                'attention_mask': inputs.attention_mask,
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'top_k' : 50,
                'do_sample': temperature > 0,
                'eos_token_id': tokenizer.eos_token_id,
            }

            if model_args.use_vllm:
                outputs = model.generate(input_prompt_text, sampling_params)
                first_token_time = 0  # vLLM is batched; harder to time token-by-token
                generated_decoded_tokens = [o.outputs[0].text for o in outputs]
            else:
                def generate():
                    model.generate(**generation_kwargs, streamer=streamer)

                generation_thread = threading.Thread(target=generate)
                generation_thread.start()

                for token in streamer:
                    if first_token_time is None:
                        first_token_time = time.time() - generation_start_time
                        first_token_start_time = time.time()
                    generated_decoded_tokens.append(token)

        except Exception as e:
            print(f"Generation error: {e}")
            print(f"Error trace:\n{traceback.format_exc()}")
            return {}

        # Stop GPU monitoring
        gpu_monitor.stop()

        # Generation metrics
        generation_time = time.time() - first_token_start_time
        total_generation_time = time.time() - generation_start_time

        # Calculate metrics
        if model_args.use_vllm is True:
            input_tokens = len(input_prompt_text.split())
        else:
            input_tokens = inputs.input_ids.shape[1]
        output_tokens = len(generated_decoded_tokens)
        total_tokens = input_tokens + output_tokens

        # Get GPU metrics
        peak_gpu_usage = gpu_monitor.get_peak_usage()
        peak_gpu_utilization = gpu_monitor.get_peak_utilization()
        p90_gpu_usage = gpu_monitor.get_p90_usage()
        p90_gpu_utilization = gpu_monitor.get_p90_utilization()

        generated_output_text = ''.join(generated_decoded_tokens)

        benchmark_results = {
            'total_time_seconds': total_generation_time,
            'time_to_first_token_seconds': first_token_time,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'tokens_per_second': total_tokens / total_generation_time,
            'output_decode_tokens_per_second': (output_tokens) / generation_time,
            'input_process_time_seconds': input_process_time,
            'peak_gpu_memory_mb': peak_gpu_usage,
            'p90_gpu_memory_mb': p90_gpu_usage,
            'peak_gpu_utilization': peak_gpu_utilization,
            'p90_gpu_utilization': p90_gpu_utilization,
            'input_prompt': input_prompt_text,
            'generated_text': generated_output_text
        }

        return benchmark_results

    def benchmark_language_model(
            model,
            tokenizer,
            prompts: List[str],
            temperature: float = 1.0,
            top_p: float = 0.95,
            max_new_tokens: int = 100,
            device: Optional[str] = None) -> Dict[str, Union[float, List[Dict[str, Any]]]]:
        """
        Benchmark a language model's performance across multiple prompts.
        """
        prompt_results = []
        for prompt in prompts:
            result = benchmark_single_prompt(
                model,
                tokenizer,
                prompt,
                temperature,
                top_p,
                max_new_tokens,
                device
            )
            if result:
                prompt_results.append(result)

        if not prompt_results:
            return {}

        # Extract metric lists for aggregation
        tps_list = [result['tokens_per_second'] for result in prompt_results]
        decode_tps_list = [result['output_decode_tokens_per_second'] for result in prompt_results]
        ttft_list = [result['time_to_first_token_seconds'] for result in prompt_results]
        gpu_usage_list = [result['peak_gpu_memory_mb'] for result in prompt_results]
        gpu_util_list = [result['peak_gpu_utilization'] for result in prompt_results]

        # Aggregate metrics
        aggregate_results = {
            # Total Tokens Per Second (TPS) metrics
            'p50_total_tps': round(np.percentile(tps_list, 50), 3),
            'p90_total_tps': round(np.percentile(tps_list, 90), 3),
            
            # Output Decode Tokens Per Second metrics
            'p50_decode_tps': round(np.percentile(decode_tps_list, 50), 3),
            'p90_decode_tps': round(np.percentile(decode_tps_list, 90), 3),
            
            # Time to First Token (TTFT) metrics
            'p50_ttft_seconds': round(np.percentile(ttft_list, 50), 3),
            'p90_ttft_seconds': round(np.percentile(ttft_list, 90), 3),

            # GPU Memory Usage metrics
            'max_gpu_memory_mb': round(max(gpu_usage_list), 3),
            'p90_gpu_memory_mb': round(np.percentile(gpu_usage_list, 90), 3),

            # GPU Utilization metrics
            'max_gpu_utilization': round(max(gpu_util_list), 3),
            'p90_gpu_utilization': round(np.percentile(gpu_util_list, 90), 3)
        }

        return aggregate_results

    # ----- Benchmark -----
    logger.warning("Benchmarking...")
    prompts = [PROMPT] * NUM_RUNS  # repeat the prompt for each run
    benchmark_results = benchmark_language_model(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        temperature=0.8,
        top_p=0.9,
        max_new_tokens=MAX_NEW_TOKENS,
        device=DEVICE
    )

    # ----- Results -----
    logger.warning(f"\nBenchmark Results ({'Unsloth' if model_args.use_unsloth else 'HF'}):")
    for key, value in benchmark_results.items():
        logger.warning(f"  {key}: {value}")

    # ----- CSV Logging -----
    csv_file = f"./benchmark/benchmark_results_unsloth_phi4_{gpu_name}_{datetime.now().strftime('%X')}.csv"
    fieldnames = list(benchmark_results.keys()) + [
        "timestamp", "gpu_name", "model", "use_unsloth", "batch_size", "max_new_tokens", "num_runs"
    ]

    csv_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gpu_name": gpu_name,
        "model": MODEL_NAME,
        "use_unsloth": model_args.use_unsloth,
        "batch_size": BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_runs": NUM_RUNS,
    }
    csv_data.update(benchmark_results)

    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(csv_data)

    logger.warning(f"\nBenchmark logged to {csv_file}")

    if model_args.use_unsloth is True and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    
def run():
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)  # Optional but recommended for vLLM
    run()