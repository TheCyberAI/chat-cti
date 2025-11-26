"""
Main script to run full pipeline: load model, prepare data, train, and run inference.
File: finetune.py
"""

from model_setup import load_model_and_tokenizer
from data_prep import load_and_prepare_dataset
from train import train_model
from inference import generate_response
from utils import print_gpu_stats
from config import LOCAL_SAVE_PATH
# Import get_chat_template to configure the tokenizer for ShareGPT
from unsloth.chat_templates import get_chat_template

if __name__ == "__main__":
    # 1. Load model and tokenizer
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()

    # 2. Apply ShareGPT Mapping
    # CRITICAL: This tells the tokenizer to map "from"->"role" and "value"->"content"
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma",
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
        map_eos_token=True,
    )

    # 3. Prepare dataset
    print("Loading and preparing dataset...")
    dataset = load_and_prepare_dataset(tokenizer)

    # 4. Print initial GPU stats
    print_gpu_stats()

    # 5. Train the model
    print("Starting training...")
    trainer, stats = train_model(model, tokenizer, dataset)

    # 6. Save model and tokenizer locally
    print(f"Saving model to {LOCAL_SAVE_PATH}...")
    model.save_pretrained(LOCAL_SAVE_PATH)
    tokenizer.save_pretrained(LOCAL_SAVE_PATH)

    # 7. Run example inference with the FIX
    print("\nRunning Inference Test...")
    
    prompt_text = "I need the Indicators of Compromise (IOCs) associated with the report 'debb0beac6414b681d050f2fbc2f2719' from 2013-09-28. Please list them by type."
    
    messages = [{
        "from": "human",       # Unsloth Template uses this (mapped to role)
        "value": prompt_text,  # Unsloth Template uses this (mapped to content) for generation
        
        # Transformers Library validation requires this specific structure to pass checks:
        "content": [{"type": "text", "text": prompt_text}] 
    }]

    generate_response(model, tokenizer, messages, max_new_tokens=256)
