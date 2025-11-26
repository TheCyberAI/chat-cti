import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer
from config import LOCAL_SAVE_PATH

def run_interactive_chat():
    # 1. Load the saved model and tokenizer
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LOCAL_SAVE_PATH, # Load your local fine-tuned model
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    # 2. Enable Native 2x Inference Optimization
    FastLanguageModel.for_inference(model)

    # 3. Apply the tokenizer mapping (Critical for your specific data format)
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma",
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
        map_eos_token=True,
    )

    # 4. Initialize Chat History
    # Start with an empty list or a system prompt if your model expects one
    messages = []
    
    print("\n" + "="*50)
    print("CTI Analyst Chatbot (Type 'exit' or 'quit' to stop)")
    print("="*50 + "\n")

    # 5. Interactive Loop
    while True:
        try:
            # Get user input
            user_input = input("\033[1;32mUser:\033[0m ") # Green text for "User:"
            
            # Check for exit commands
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting chat...")
                break
            
            # Append user message to history
            # Note: We add both 'value' (for model) and 'content' (for library validation)
            messages.append({
                "from": "human",
                "value": user_input,
                "content": [{"type": "text", "text": user_input}]
            })

            # Format input for the model
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=True,
                return_dict=True,
            ).to("cuda")

            # Set up streaming (prints token-by-token instead of waiting)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            print("\033[1;34mAssistant:\033[0m ", end="", flush=True) # Blue text for "Assistant:"
            
            # Generate response
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1, # Low temperature for factual CTI responses
                top_p=0.95,
                streamer=streamer,
                use_cache=True
            )
            
            # Decode the generated response to plain text
            # We need to extract just the new tokens to add to history
            # The streamer prints it, but we need the text variable for the history list
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Append Assistant response to history so the model remembers context
            messages.append({
                "from": "gpt",
                "value": response_text,
                "content": [{"type": "text", "text": response_text}]
            })
            
            print("\n") # Newline for next turn

        except KeyboardInterrupt:
            print("\nChat interrupted. Exiting...")
            break

if __name__ == "__main__":
    run_interactive_chat()
