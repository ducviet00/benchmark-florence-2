import time
import requests
from contextlib import nullcontext

import torch
from torch.profiler import profile, ProfilerActivity

from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from modeling_florence2 import Florence2ForConditionalGeneration

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


config = AutoConfig.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
config.vision_config.model_type = "davit"
config.text_config.use_cache = True

model = Florence2ForConditionalGeneration.from_pretrained(
			"microsoft/Florence-2-large",
			config=config,
			torch_dtype=torch_dtype,
			attn_implementation="flash_attention_2" if torch.cuda.is_available() else "sdpa",
			trust_remote_code=True,
		)

model.language_model.generation_config.max_new_tokens = 1024
model.language_model.generation_config.cache_implementation = "static"

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

input_ids = inputs["input_ids"]
pixel_values = inputs["pixel_values"]

assert device == "cuda"
print("Compiling the model")
model.vision_tower = torch.compile(model.vision_tower, mode="max-autotune", fullgraph=True)
model.language_model.forward = torch.compile(model.language_model.forward, mode="reduce-overhead", fullgraph=True)

print("model dtype:", model.dtype)
print("input_ids:", input_ids.shape)
print("input_ids:", input_ids.device)
print("pixel_values:", pixel_values.shape)
print("pixel_values:", pixel_values.device)

#WARM UP
with torch.inference_mode():
	generated_ids = model.generate(
		input_ids=input_ids,
		pixel_values=pixel_values,
		max_new_tokens=1024,
		min_new_tokens=1024,
		num_beams=1,
		do_sample=False,
	)

print("generated_ids:", generated_ids.shape)


durations = []
with torch.inference_mode():
	for _ in range(5):
		t0 = time.perf_counter()
		generated_ids = model.generate(
			input_ids=input_ids,
			pixel_values=pixel_values,
			max_new_tokens=1024,
			min_new_tokens=1024,
			num_beams=1,
			do_sample=False,
		)
		device_sync(device=device)
		duration = time.perf_counter() - t0
		durations.append(duration)
		print(f"len: {generated_ids.numel()} - perf: {generated_ids.numel() / duration} tok /s")
print(sum(durations) / len(durations))


# torch.cuda.memory._record_memory_history(
#        max_entries=100000
#    )

activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]
with profile(activities=activities) as prof:
	with torch.inference_mode():
		generated_ids = model.generate(
			input_ids=input_ids,
			pixel_values=pixel_values,
			max_new_tokens=16,
			min_new_tokens=16,
			num_beams=1,
			do_sample=False,

		)

# try:
# 	torch.cuda.memory._dump_snapshot("memory_proiling_data.pickle")
# except Exception as e:
# 	print(f"Failed to capture memory snapshot {e}")

prof.export_chrome_trace("new_trace.json")

print("Done profiling")



generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

parsed_answer = processor.post_process_generation(generated_text,
												  task="<OD>",
												  image_size=(image.width,
															  image.height))
