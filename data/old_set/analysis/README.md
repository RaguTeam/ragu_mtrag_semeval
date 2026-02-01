```
python scripts/analysis/run_analysis.py \
    input=$MTRAG_DATA/human/generation_tasks/reference.jsonl \
    output=data/old_set/analysis/agent_behaviour.json \
    split_file=splits/test_1.json \
    --config-name agent_behaviour
```