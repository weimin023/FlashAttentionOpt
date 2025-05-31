import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.profiler import profile, record_function, ProfilerActivity


# 載入 tokenizer 和 model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
tokenizer.pad_token = tokenizer.eos_token


# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

text = "Replace me by any text you'd like."
encoded_input = tokenizer(
    text,
    return_tensors='pt',
    padding=True,
    truncation=True
)
encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

# ✅ 儲存 trace 為 json
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    on_trace_ready=torch.profiler.tensorboard_trace_handler("profiler_logs"),  # 存到 profiler_logs/
    with_stack=False,
    profile_memory=False,
    record_shapes=False
) as prof:
    with torch.no_grad():
        with record_function("model_generate"):
            generated_ids = model.generate(
                input_ids=encoded_input['input_ids'],
                attention_mask=encoded_input['attention_mask'],
                max_length=50,
                pad_token_id=tokenizer.eos_token_id
            )

generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_text)

# 額外印出 summary（非必要）
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))