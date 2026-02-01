```
python scripts/generation/run_generation_task_b.py \
    input=$MTRAG_DATA/human/generation_tasks/reference.jsonl \
    output=data/old_set/baseline/gpt-5-nano.json \
    split_file=splits/test_1.json
```

```
python scripts/generation/run_generation_task_b.py \
    input=$MTRAG_DATA/human/generation_tasks/reference.jsonl \
    output=data/old_set/baseline/gemini-3-pro-preview-high.json \
    split_file=splits/test_1.json \
    --config-name gemini
```

```
python scripts/generation/run_generation_task_b.py \
    input=$MTRAG_DATA/human/generation_tasks/reference.jsonl \
    output=data/old_set/baseline/gemini-3-pro-preview-high_new_prompt.json \
    split_file=splits/test_1.json \
    --config-name gemini_new_prompt
```