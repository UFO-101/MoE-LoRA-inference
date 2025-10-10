#!/usr/bin/env python3
# %%
"""
vLLM inference with LoRA adapter.

This script uses vLLM for high-performance inference with a trained LoRA adapter.
"""

import os

# Force vLLM to use CUDA platform (workaround for NVML driver mismatch)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

# Import after setting environment variables
from vllm import LLM, SamplingParams

# %%
llm = None
try:
    llm = LLM(
        model="ibm-research/PowerMoE-3b",
        gpu_memory_utilization=0.5,  # don't grab 90% while iterating
        max_model_len=2048,  # smaller KV cache while debugging
        enforce_eager=True,  # skip compile until stable
        enable_lora=True,  # Enable LoRA support
        tensor_parallel_size=1,  # Explicitly set single GPU
    )

    # %%
    params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)

    # Format prompts to match training format (instruction/response style)
    prompts = [
        "### Instruction:\nWrite a short story about a brave adventurer.\n\n### Response:\n",
        "### Instruction:\nExplain how photosynthesis works.\n\n### Response:\n",
        "### Instruction:\nWhat are three tips for staying healthy?\n\n### Response:\n",
    ]

    # %%
    # Generate with LoRA adapter
    lora_path = "./lora_adapters/powermoe-3b/checkpoint-1500"
    outs = llm.generate(
        prompts,
        params,
        lora_request={"lora_name": "powermoe-lora", "lora_path": lora_path},
    )

    # %%
    for o in outs:
        print(o.prompt, "\n", o.outputs[0].text, "\n", "-" * 80)

finally:
    if llm is not None:
        llm.shutdown()  # ensures the child process exits and frees VRAM
