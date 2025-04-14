import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
from huggingface_hub import login

def main():
    # ------------------------------------------------------------
    # 1) Adapter path and config
    # ------------------------------------------------------------
    # This folder should contain:
    #   adapter_config.json
    #   adapter_model.bin or adapter_model.safetensors
    #   merges.txt, vocab.json, tokenizer_config.json (optional: special_tokens_map.json)
    login("")
    adapter_path = "feedbackModel/"
    
    # Load the PEFT config to retrieve base model name (e.g., facebook/bart-base)
    peft_config = PeftConfig.from_pretrained(adapter_path)

    # ------------------------------------------------------------
    # 2) Load tokenizer and base Seq2Seq model
    # ------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        peft_config.base_model_name_or_path, 
        # BART sometimes requires a slow tokenizer for special tasks, so if needed:
        # use_fast=False
    )
    base_model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path)

    # ------------------------------------------------------------
    # 3) Load and merge the PEFT adapter
    # ------------------------------------------------------------
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    peft_model.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peft_model.to(device)

    # ------------------------------------------------------------
    # 4) Example generation
    # ------------------------------------------------------------
    input_text = """Grade the following - Question: Assume you have a local network with 3 users that are all interconnected and have perfect clocks. Typically the network is often congested as all users generate more traffic than the link’s capacities. Which of the encoding techniques introduced in the lecture should be used in this network to encode bitstreams? Give two reasons for your answer in 2-4 sentences.
Answer: Binary encoding can be used. It has the highest bandwidth (1 bit per Baud) and is simple and cheap. The 'self-clocking' feature of the more complex manchester encoding and differential manchester encodings is not necessary since the users have perfect clocks. Score - 2.01"""
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Generate text using the BART model with your fine-tuned adapter
    with torch.no_grad():
        generated_tokens = peft_model.generate(
            **inputs,
            max_new_tokens=50,       # Adjust as needed
            num_beams=4,            # Example beam search
            do_sample=False         # or True for sampling
        )

    # Decode the output
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    print("Input Text:", input_text)
    print("Generated Text:", generated_text)

if __name__ == "__main__":
    main()
