import time
import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
# Experimental features to reduce compilation times, will be on by default in future
torch._inductor.config.fx_graph_cache = True
torch._functorch.config.enable_autograd_cache = True

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = (
    AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True,
    )
    .eval()
    .to(device)
)
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-large", trust_remote_code=True,
)

prompt = "<OD>"

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=image, return_tensors="pt").to(
    device, torch_dtype
)

input_ids = inputs["input_ids"].repeat(1, 5)
pixel_values = inputs["pixel_values"]

print("input_ids:", input_ids.shape)
print("pixel_values:", pixel_values.shape)

if device == "cuda:0":
    model.to(memory_format=torch.channels_last)
    model = torch.compile(model, mode="max-autotune", fullgraph=True)

for _ in range(5):
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=1024,
            min_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )

print("generated_ids:", generated_ids.shape)


durations = []
for _ in range(25):
    with torch.inference_mode():
        stime = time.time()
        generated_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=1024,
            min_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )
        duration = time.time() - stime
        durations.append(duration)
        print(f"{generated_ids.shape[1] / duration} tok /s")


print(sum(durations) / len(durations))

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

parsed_answer = processor.post_process_generation(
    generated_text, task="<OD>", image_size=(image.width, image.height)
)

print(parsed_answer)
