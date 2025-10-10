#!/usr/bin/env python3
# %%
"""
Compare inference with and without LoRA adapter.

This script loads the PowerMoE-3b model both with and without a trained LoRA
adapter, allowing you to compare their outputs side-by-side.
"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_base_model(model_name: str = "ibm-research/PowerMoE-3b"):
    """Load the base model without LoRA."""
    print(f"Loading base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    return model, tokenizer


# %%
def load_lora_model(
    model_name: str = "ibm-research/PowerMoE-3b",
    lora_path: str = "./lora_adapters/powermoe-3b/checkpoint-1500",
):
    """Load the model with LoRA adapter."""
    print(f"Loading LoRA model from: {lora_path}")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, lora_path)

    # Load tokenizer from LoRA checkpoint (it should be the same)
    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# %%
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
):
    """Generate text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


# %%
def compare_models(
    base_model,
    lora_model,
    tokenizer,
    prompts: list[str],
    **generation_kwargs,
):
    """Compare outputs from base model and LoRA model."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'─' * 80}")
        print(f"PROMPT {i}:")
        print(f"{'─' * 80}")
        print(prompt)
        print()

        # Generate with base model
        print("BASE MODEL OUTPUT:")
        print("─" * 80)
        base_output = generate_text(base_model, tokenizer, prompt, **generation_kwargs)
        # Remove the prompt from output
        base_response = base_output[len(prompt) :]
        print(base_response)
        print()

        # Generate with LoRA model
        print("LORA MODEL OUTPUT:")
        print("─" * 80)
        lora_output = generate_text(lora_model, tokenizer, prompt, **generation_kwargs)
        # Remove the prompt from output
        lora_response = lora_output[len(prompt) :]
        print(lora_response)
        print()


# %%
def main():
    """Main inference function."""
    # Configuration
    model_name = "ibm-research/PowerMoE-3b"
    lora_checkpoint = "./lora_adapters/powermoe-3b/checkpoint-2000"

    # Load models
    print("Loading models...")
    base_model, base_tokenizer = load_base_model(model_name)
    lora_model, lora_tokenizer = load_lora_model(model_name, lora_checkpoint)

    # Example prompts (adjust based on what your model was trained on)
    # The training script used the dolly-15k-pirate-speech dataset,
    # so the model should respond in pirate speech style
    prompts = [
        "### Instruction:\nWrite a short story about a brave adventurer.\n\n### Response:\n",
        "### Instruction:\nExplain how photosynthesis works.\n\n### Response:\n",
        "### Instruction:\nWhat are three tips for staying healthy?\n\n### Response:\n",
    ]

    # Generation parameters
    generation_kwargs = {
        "max_new_tokens": 150,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
    }

    # Compare models
    compare_models(
        base_model,
        lora_model,
        base_tokenizer,  # Use base tokenizer (should be identical)
        prompts,
        **generation_kwargs,
    )

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print("\nNote: The LoRA model was trained on pirate-speech style responses,")
    print("so you should see a difference in speaking style between the two models.")


if __name__ == "__main__":
    main()
