#!/bin/bash
export UNSLOTH_COMPILE_USE_TEMP=1
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/mgeorges/huggingface_cache

python -c "from vllm import LLM; model = LLM('microsoft/phi-4'); print('Done')"

#nohup torchrun --nproc_per_node=1 
python ./inference_time.py \
  --from_peft ./finetuned_phi_reasoning_unsloth/checkpoint-2139 \
  --model_name_or_path microsoft/phi-4 \
  --train_data ./finetuning_data/medical_o1_reasoning_train.jsonl \
  --use_unsloth False \
  --use_vllm True \
  --batch_size 8 \
  --use_lora True \
  --cache_dir scratch/huggingface_cache/hub \
  --target_modules q_proj k_proj v_proj o_proj gate_Trueproj up_proj down_proj \
  --use_4bit True \
  --bf16 True \
  --num_runs 10 \
  --num_max_new_tokens 256