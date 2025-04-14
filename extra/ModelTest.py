import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import PeftModel, PeftConfig
from huggingface_hub import login

def main():
    # -----------------------------------------------------------------
    # 1. Configure paths and labels
    # -----------------------------------------------------------------
    login("")
    adapter_path = "scoringModel/"
    peft_config = PeftConfig.from_pretrained(adapter_path)

    # If the adapter was trained with a single output (e.g. regression or
    # single-label classification), set num_labels=1. Otherwise use
    # the appropriate value for your use case.
    num_labels = 1

    # -----------------------------------------------------------------
    # 2. Load config, tokenizer, and base model
    # -----------------------------------------------------------------
    base_config = AutoConfig.from_pretrained(peft_config.base_model_name_or_path)
    base_config.num_labels = num_labels

    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        peft_config.base_model_name_or_path,
        config=base_config
    )

    # -----------------------------------------------------------------
    # 3. Load the PEFT adapter
    # -----------------------------------------------------------------
    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_path
    )
    peft_model.eval()

    # -----------------------------------------------------------------
    # 4. Run a sample inference
    # -----------------------------------------------------------------
    text = """ Grade the following - Question: Assume you have a local network with 3 users that are all interconnected and have perfect clocks. Typically the network is often congested as all users generate more traffic than the link’s capacities. Which of the encoding techniques introduced in the lecture should be used in this network to encode bitstreams? Give two reasons for your answer in 2-4 sentences.
Answer: hi
"""
    inputs = tokenizer(text, return_tensors="pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peft_model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = peft_model(**inputs)
    
    # This is the raw logit(s) for your input
    logits = outputs.logits

    # If num_labels=1, logits is shape [batch_size, 1]
    # We'll just grab the raw score from that single cell.
    # (If you actually need probabilities from 0..1, you might do a sigmoid.)
    score = logits.squeeze().item()

    print("Text:", text)
    print("Logits (raw):", logits)
    print("Score (scalar):", score)

if __name__ == "__main__":
    main()