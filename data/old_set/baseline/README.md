gpt5_nano.json:
```
python scripts/generation/run_generation_task_b.py \
    --input $MTRAG_DATA/human/generation_tasks/reference.jsonl --output gpt5_nano.json \
    --base_url $OPENAI_URL --api_key $OPENAI_KEY --model openai/gpt-5-nano \
    --split_file splits/test_1.json
```