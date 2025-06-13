import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.profiler import profile, record_function, ProfilerActivity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

prompt = "Taiwan"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

#outputs = model.generate(**inputs, max_new_tokens=100)
#print(tokenizer.decode(outputs[0], skip_special_tokens=True))

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    profile_memory=True
) as prof:
    for _ in range(1):
        _ = model(**inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))