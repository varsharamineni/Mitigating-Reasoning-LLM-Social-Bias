from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

print("Downloading tokenizer...")
AutoTokenizer.from_pretrained(model_name)

print("Downloading model weights only (no loading)...")
AutoModelForCausalLM.from_pretrained(model_name)

print("Done caching model!")