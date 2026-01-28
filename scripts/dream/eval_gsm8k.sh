GPU_ID=${1:-0}  # Default to GPU 0 if not provided
MODEL_ROOT=/dataset/models # Path to store models, "Dream-org" if you want to use the Huggingface models
DATA_ROOT=./results # Path to datasets

TASK_NAME=gsm8k_cot
ESCAPE_UNTIL=False
NUM_FEWSHOT=8
BATCH_SIZE=4

MODEL_NAME=Dream-v0-Instruct-7B
STEPS=256
GEN_LENGTH=256

REFRESH_STEPS=50
REFRESH_GEN_STEPS=8

LOG_WARMUP=0
LOG_TTFT=1
LOG_LATENCY=1


EXP_NAME=00_baseline
CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/dream/eval.py \
    --model Dream \
    --tasks $TASK_NAME \
    --num_fewshot $NUM_FEWSHOT \
    --batch_size $BATCH_SIZE \
    --output_path ${DATA_ROOT}/${MODEL_NAME}/eval/${TASK_NAME}/${EXP_NAME} \
    --model_args pretrained_model_path=${MODEL_ROOT}/${MODEL_NAME},save_dir=${DATA_ROOT}/${MODEL_NAME}/${TASK_NAME},exp_name=${EXP_NAME},escape_until=${ESCAPE_UNTIL},steps=${STEPS},gen_length=${GEN_LENGTH},skip_attn_mask=False,log_warmup=${LOG_WARMUP},log_ttft=${LOG_TTFT},log_latency=${LOG_LATENCY}


EXP_NAME=01_spa_cache
CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/dream/eval.py \
    --model Dream \
    --tasks $TASK_NAME \
    --num_fewshot $NUM_FEWSHOT \
    --batch_size $BATCH_SIZE \
    --output_path ${DATA_ROOT}/${MODEL_NAME}/eval/${TASK_NAME}/${EXP_NAME} \
    --model_args "pretrained_model_path=${MODEL_ROOT}/${MODEL_NAME},log_dir=${DATA_ROOT}/${MODEL_NAME}/${TASK_NAME},exp_name=${EXP_NAME},escape_until=${ESCAPE_UNTIL},steps=${STEPS},gen_length=${GEN_LENGTH},cache=spa,proxy_rank=32,update_ratio_dist=gaussian,max_update_ratio=0.25,refresh_steps=25,refresh_gen_steps=2,log_warmup=${LOG_WARMUP},log_ttft=${LOG_TTFT},log_latency=${LOG_LATENCY}"