#%%
#!/usr/bin/env python3
"""
Basic vLLM setup with Qwen2.5-0.5B-Instruct
A state-of-the-art <1B parameter model with strong instruction-following capabilities.
"""

from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
)

prompts = [
    "What is the capital of France?",
    "Write a haiku about artificial intelligence.",
    "Explain what vLLM is in one sentence.",
]

# Generate responses
print("Running vLLM with Qwen2.5-0.5B-Instruct...\n")
outputs = llm.generate(prompts, sampling_params)

# Display results
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Response: {generated_text}")
    print("-" * 80)
