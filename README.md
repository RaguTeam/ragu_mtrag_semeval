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

## Generation  

To generate predictions use data in the format provided in [reference.jsonl](https://github.com/IBM/mt-rag-benchmark/blob/main/human/generation_tasks/reference.jsonl) or evaluation data.


```
export PYTHONPATH=.
python scripts/generation/run_generation_task_b.py --input <INPUT FILE> --output --<OUTPUT FILE>
```
Optionally you can additional args:

```
    ap.add_argument("--base_url", default="http://localhost:8001/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--api_key", default="testkey")

    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--start_idx", type=int, default=0)

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--doc_header", action="store_true")
    ap.add_argument("--keep_error", action="store_true", help="If set, write prediction_error then stop (still stops).")
```

## Evaluation

After this you can git clone https://github.com/IBM/mt-rag-benchmark (or for now use fork where few arguments is fixed https://github.com/acssar/mt-rag-benchmark)

Then you can use next instruction https://github.com/IBM/mt-rag-benchmark/blob/main/scripts/evaluation/README.md

example of checking the format:
```
python mt-rag-benchmark/scripts/evaluation/format_checker.py --input_file <REFERENCE_DATA_JSONL> --prediction_file <PREDICTIONS_FILE_JSONL --mode generation_taskb
```

example of evaluation run:
```
python mt-rag-benchmark/scripts/evaluation/run_generation_eval.py -i <PREDICTIONS_FILE_JSONL> -o <EVALUATION_RESULTS_JSONL> -e mt-rag-benchmark/scripts/evaluation/config.yaml --provider vllm --judge_model Qwen/Qwen3-4B
```
by default they have hardcoded port 8001 for vllm

To aggregate metrics and calculate harmonic mean between them you can run next script: 
```
python scripts/evaluation/metrics_aggregation.py --input <EVALUATION_RESULTS_JSONL>
