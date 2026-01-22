### MTRAG RAGU

## Environment

```
uv sync
pre-commit install
source .venv/bin/activate
```

## Prepare local model

```
vllm serve Qwen/Qwen3-4B-FP8 --max_model_len 6000 --gpu_memory_utilization 0.85
```

## Run example with

```
export PYTHONPATH=.
python src/generation/main.py
```
