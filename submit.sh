python mt-rag-benchmark/scripts/evaluation/run_generation_eval.py \
 -i $1 \
 -o $1 \
 -e mt-rag-benchmark/scripts/evaluation/config.yaml \
 --provider vllm \
 --judge_model "openai/gpt-4o-mini" \
 --base_url "https://api.vsegpt.ru/v1" \
 --openai_key $OPENAI_KEY \
 --eval_phase