GPU_ID=${1:-0}  # Default to GPU 0 if not provided
MODEL_ROOT=/dataset/models # Path to store models, "GSAI-ML" if you want to use the Huggingface models
DATA_ROOT=./results # Path to datasets

TASK_NAME=bbh
NUM_FEWSHOT=3
BATCH_SIZE=16

MODEL_NAME=${2:-LLaDA-8B-Instruct}
STEPS=256
GEN_LENGTH=256
BLOCK_LENGTH=256


EXP_NAME=00_baseline
CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/llada/eval.py \
    --model LLaDA \
    --tasks $TASK_NAME \
    --num_fewshot $NUM_FEWSHOT \
    --batch_size $BATCH_SIZE \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --output_path ${DATA_ROOT}/${MODEL_NAME}/${TASK_NAME}/${EXP_NAME} \
    --model_args pretrained_model_path=${MODEL_ROOT}/${MODEL_NAME},save_dir=${DATA_ROOT}/${MODEL_NAME}/${TASK_NAME},exp_name=${EXP_NAME},steps=${STEPS},gen_length=${GEN_LENGTH},block_length=${BLOCK_LENGTH}


EXP_NAME=01_spa_cache
CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/llada/eval.py \
    --model LLaDA \
    --tasks $TASK_NAME \
    --num_fewshot $NUM_FEWSHOT \
    --batch_size $BATCH_SIZE \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --output_path ${DATA_ROOT}/${MODEL_NAME}/${TASK_NAME}/${EXP_NAME} \
    --model_args pretrained_model_path=${MODEL_ROOT}/${MODEL_NAME},save_dir=${DATA_ROOT}/${MODEL_NAME}/${TASK_NAME},exp_name=${EXP_NAME},steps=${STEPS},gen_length=${GEN_LENGTH},block_length=${BLOCK_LENGTH},cache=spa,proxy_rank=128,max_update_ratio=0.25,refresh_steps=50,refresh_gen_steps=6,early_stop_steps=${BLOCK_LENGTH}