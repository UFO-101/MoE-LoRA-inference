#!/usr/bin/env python3
# %%
"""
Train a LoRA adapter for PowerMoE-3b model.

This script fine-tunes the PowerMoE-3b model using LoRA (Low-Rank Adaptation)
for efficient training with reduced memory requirements.
"""

import os
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


@dataclass
class TrainingConfig:
    """Configuration for LoRA training."""

    # Model configuration
    model_name: str = "ibm-research/PowerMoE-3b"

    # LoRA configuration
    lora_r: int = 8  # LoRA rank
    lora_alpha: int = 16  # LoRA alpha (scaling factor)
    lora_dropout: float = 0.05
    target_modules: Optional[list[str]] = None  # Will be auto-detected if None

    # Dataset configuration
    dataset_name: str = "TeeZee/dolly-15k-pirate-speech"  # Example dataset
    dataset_split: str = "train"
    max_length: int = 512

    # Training configuration
    output_dir: str = "./lora_adapters/powermoe-3b"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3

    # Hardware configuration
    fp16: bool = True
    bf16: bool = False  # Set to True if your GPU supports bfloat16
    gradient_checkpointing: bool = False

    # Other
    seed: int = 42


def format_prompt(example):
    """Format the dataset example into a prompt."""
    if "instruction" in example and "response" in example:
        # Dolly-style format (instruction, response)
        instruction = example["instruction"]
        response = example["response"]
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        return prompt
    elif "text" in example:
        # Simple text format
        return example["text"]
    else:
        raise ValueError(
            "Unsupported dataset format. Expected 'instruction'/'response' or 'text' fields."
        )


def prepare_dataset(
    dataset_name: str, tokenizer, max_length: int, dataset_split: str = "train"
):
    """Load and prepare the dataset for training."""
    print(f"Loading dataset: {dataset_name}")

    # Load dataset
    dataset = load_dataset(dataset_name, split=dataset_split)

    # Format prompts
    def tokenize_function(examples):
        # Format the prompt
        texts = [
            format_prompt(ex)
            for ex in [
                dict(zip(examples.keys(), values)) for values in zip(*examples.values())
            ]
        ]

        # Tokenize
        result = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )

        # For causal LM, labels are the same as input_ids
        result["labels"] = result["input_ids"].copy()

        return result

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    return tokenized_dataset


def main():
    """Main training function."""
    config = TrainingConfig()

    # Set random seed for reproducibility
    torch.manual_seed(config.seed)

    print(f"Loading model: {config.model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.float16 if config.fp16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # Configure LoRA
    print("Configuring LoRA")

    # If target modules not specified, use common defaults for MoE models
    target_modules = config.target_modules
    if target_modules is None:
        # Default targets for transformer models (adjust based on model architecture)
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing if specified
    if config.gradient_checkpointing:
        model.enable_input_require_grads()

    # Prepare dataset
    train_dataset = prepare_dataset(
        config.dataset_name,
        tokenizer,
        config.max_length,
        config.dataset_split,
    )

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        report_to=["tensorboard"],
        seed=config.seed,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving final model to {config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

    print("Training complete!")


if __name__ == "__main__":
    main()
