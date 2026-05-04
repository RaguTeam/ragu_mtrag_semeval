### MTRAG RAGU

## Environment

```
uv sync --extra eval
source .venv/bin/activate
pre-commit install
```

## Set paths to the data and code

Need to set paths to the data and code (example below):

```
export MTRAG_DATA=..../mt-rag-benchmark
export MTRAG_SPLITS=..../ragu_mtrag_eval_2026/splits
export PYTHONPATH=..../ragu_mtrag_eval_2026
export OPENAI_URL=https://api.vsegpt.ru/v1
export OPENAI_KEY=....
export OPENAI_LOG_DIR=..../ragu_mtrag_eval_2026/logs/generation
```

Where `mt-rag-benchmark` is a directory where github.com/IBM/mt-rag-benchmark was cloned.
You may use .envrc file and load with `source .envrc`, or add to ~/.bashrc to auto-load.

## Prepare local model

```
vllm serve Qwen/Qwen3-4B-FP8 --max_model_len 6000 --gpu_memory_utilization 0.85
```

## Run example with

```
source .envrc
python src/generation/main.py
```

## Generation

To generate predictions use data in the format provided in [reference.jsonl](https://github.com/IBM/mt-rag-benchmark/blob/main/human/generation_tasks/reference.jsonl) or evaluation data.

```
source .envrc
python scripts/generation/run_generation_task_b.py \
    input=$MTRAG_DATA/human/generation_tasks/reference.jsonl output=output.json
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
```
