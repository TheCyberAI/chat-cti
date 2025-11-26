"""
Inference utilities for Gemma-3N.
"""

import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel # CHANGED: Import Unsloth

def generate_response(model, tokenizer, messages, max_new_tokens=128):
    """
    Run inference on Gemma-3N with given messages.
    Uses TextStreamer for streaming output.
    
    NOTE: 'messages' must use the keys matching your tokenizer mapping.
    For ShareGPT, use: [{"from": "human", "value": "Your prompt"}]
    """
    
    # CHANGED: Enable Native 2x Inference Optimization
    FastLanguageModel.for_inference(model)

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    ).to("cuda")

    streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=1.0, # You might want to lower this to 0.1 for factual CTI extraction
        top_p=0.95,
        top_k=64,
        streamer=streamer,
    )
