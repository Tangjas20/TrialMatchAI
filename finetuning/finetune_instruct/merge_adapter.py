from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-4")

# Load LoRA adapter
peft_model = PeftModel.from_pretrained(base_model, "./finetuned_phi_reasoning_unsloth/checkpoint-2139")

# Merge LoRA weights
peft_model = peft_model.merge_and_unload()

# Save merged model
peft_model.save_pretrained("./merged_phi-4")
