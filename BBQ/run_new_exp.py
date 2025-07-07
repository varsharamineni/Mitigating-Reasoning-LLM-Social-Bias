import os
import glob
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForCausalLM
from typing import List

# === Configuration ===
BBQ_DATASET_DIR = "dataset/BBQ"
OUTPUT_FILE = "bbq_two_local_models_results.jsonl"

os.environ["TRANSFORMERS_CACHE"] = "/leonardo/home/userexternal/vraminen/.cache/huggingface/hub"
# Two local HuggingFace models to compare (must be compatible seq2seq/chat models)
MODEL_1_NAME =  "EleutherAI/pythia-410m"
MODEL_2_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"        # example, change to your second model

# Load BBQ dataset from JSONL files
def load_bbq_dataset(dataset_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(dataset_dir, "*.jsonl"))
    dfs = []
    for file in files:
        df = pd.read_json(file, lines=True)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# Format prompt with CoT request
def format_prompt_with_cot(question: str, demographic: str) -> str:
    return (
        f"Demographic: {demographic}\n"
        f"Question: {question}\n"
        "Think step-by-step and explain your reasoning before giving the final answer."
    )

# Generate response from local model pipeline
def get_model_response(pipe, prompt: str, max_length=512) -> str:
    output = pipe(prompt, max_length=max_length, do_sample=False)
    # Output can be list of dicts or a dict depending on pipeline, normalize to string:
    if isinstance(output, list):
        return output[0]["generated_text"]
    elif isinstance(output, dict):
        return output.get("generated_text", "")
    else:
        return str(output)

def main():
    print("Loading BBQ dataset...")
    bbq_df = load_bbq_dataset(BBQ_DATASET_DIR)
    print(f"Loaded {len(bbq_df)} BBQ questions")

    print(f"Loading model 1: {MODEL_1_NAME}")
    tokenizer_1 = AutoTokenizer.from_pretrained(MODEL_1_NAME, trust_remote_code=True)
    model_1 = AutoModelForCausalLM.from_pretrained(MODEL_1_NAME, trust_remote_code=True)
    pipe_1 = pipeline("text2text-generation", model=model_1, tokenizer=tokenizer_1, device=0)

    print(f"Loading model 2: {MODEL_2_NAME}")
    tokenizer_2 = AutoTokenizer.from_pretrained(MODEL_2_NAME, trust_remote_code=True)
    model_2 = AutoModelForCausalLM.from_pretrained(MODEL_2_NAME, trust_remote_code=True)
    pipe_2 = pipeline("text2text-generation", model=model_2, tokenizer=tokenizer_2, device=0)

    results = []

    for idx, row in bbq_df.iterrows():
        question = row["question"]
        demographic = row.get("demographic", "Unknown")
        correct_answer = row.get("correct_answer", None)

        prompt = format_prompt_with_cot(question, demographic)

        # Get reasoning + answer from both models
        response_1 = get_model_response(pipe_1, prompt)
        response_2 = get_model_response(pipe_2, prompt)

        results.append({
            "question": question,
            "demographic": demographic,
            "correct_answer": correct_answer,
            "model_1": MODEL_1_NAME,
            "model_1_response": response_1,
            "model_2": MODEL_2_NAME,
            "model_2_response": response_2,
        })

        if idx % 10 == 0:
            print(f"Processed {idx}/{len(bbq_df)} questions")

    print(f"Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f_out:
        for item in results:
            f_out.write(json.dumps(item) + "\n")

    print("Done.")

if __name__ == "__main__":
    main()
