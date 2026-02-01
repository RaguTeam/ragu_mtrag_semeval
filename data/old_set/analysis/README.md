```
python scripts/analysis/run_analysis.py \
    input=$MTRAG_DATA/human/generation_tasks/reference.jsonl \
    output=data/old_set/analysis/agent_behaviour.json \
    split_file=splits/test_1.json \
    --config-name agent_behaviour
```

```
python scripts/analysis/run_analysis.py \
    input=data/old_set/baseline_metrics/test_gemini-3-pro-preview-high_new_prompt.jsonl \
    output=data/old_set/analysis/self_improvement.json \
    split_file=splits/test_1.json \
    --config-name self_improvement
```