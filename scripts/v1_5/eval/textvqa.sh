#!/bin/bash
cd /root/LLaVA-AKS
export PYTHONWARNINGS="ignore"

python -m llava.eval.model_vqa_loader \
    --model-path /hy-tmp/llava-v1.5-7b \
    --question-file /hy-tmp/Train/test_convert.jsonl \
    --image-folder /hy-tmp/Train/images \
    --answers-file /hy-tmp/Train/answers.json \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --use_token_pruner True \
    --use_token_pruner_inference True \
    --pruner_num_branches 2 \
    --pruner_max_depth 5 \
    --pruner_target_ratio 0.5 \
    --pruner_verbose True

# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-13b.jsonl
