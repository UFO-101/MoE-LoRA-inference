# %%
from vllm import LLM, SamplingParams

llm = None
try:
    llm = LLM(
        model="ibm-research/PowerMoE-3b",
        gpu_memory_utilization=0.5,  # don’t grab 90% while iterating
        max_model_len=2048,  # smaller KV cache while debugging
        enforce_eager=True,  # skip compile until stable
    )
    params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)
    prompts = [
        "What is the capital of France?",
        "Write a haiku about artificial intelligence.",
        "Explain what vLLM is in one sentence.",
    ]
    outs = llm.generate(prompts, params)
    for o in outs:
        print(o.prompt, "\n", o.outputs[0].text, "\n", "-" * 80)
finally:
    if llm is not None:
        llm.shutdown()  # <- ensures the child process exits and frees VRAM
