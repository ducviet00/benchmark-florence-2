# Installation

```
git clone https://github.com/ducviet00/benchmark-florence-2.git && cd benchmark-florence-2
docker run --gpus all -it --rm -v ${PWD}:/workspace/ pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel bash
```

Inside docker

```
pip install -r requirements.txt
python infer.py
```
