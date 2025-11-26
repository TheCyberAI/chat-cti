"""
Dataset loading and preprocessing functions.
"""

from datasets import load_dataset
from unsloth.chat_templates import get_chat_template, standardize_data_formats
# You can remove DATASET_NAME from imports if you hardcode the file path
from config import DATASET_SPLIT, CHAT_TEMPLATE 

def load_and_prepare_dataset(tokenizer):
    # --- CHANGE 1: Load Local JSONL File ---
    # Instead of DATASET_NAME, we use "json" and provide the path to your generated file.
    dataset = load_dataset("json", data_files="sharegpt_dataset.jsonl", split="train")
    
    # standardize_data_formats works well with ShareGPT format, so we keep it.
    dataset = standardize_data_formats(dataset)

    # Apply chat template
    def formatting_prompts_func(examples):
        # Your data has the key "conversations", so this line is perfect.
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False
            ) #.removeprefix("<bos>") # OPTIONAL: Only use removeprefix if you see double BOS tokens
            for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset
