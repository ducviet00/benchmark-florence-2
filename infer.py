import time
import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = True
torch._inductor.config.coordinate_descent_check_all_directions = True
torch.set_float32_matmul_precision('high')

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")

model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large",
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
model = model.to(device)

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

if device == "cuda":
    print("Compiling the model")
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

# WARM UP    
with torch.inference_mode():
    for _ in range(3):
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
with torch.inference_mode():
    for _ in range(25):
        t0 = time.perf_counter()
        generated_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=1024,
            min_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )
        device_sync(device=device)
        duration = time.perf_counter() - t0
        durations.append(duration)
        print(f"len: {generated_ids.shape[1]} - perf: {generated_ids.shape[1] / duration} tok /s")


print(sum(durations) / len(durations))

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

parsed_answer = processor.post_process_generation(
    generated_text, task="<OD>", image_size=(image.width, image.height)
)

print(parsed_answer)
