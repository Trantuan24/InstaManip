PROJ_PATH='.'
exp_name='InstaManip'
OUTPUT_PATH=${PROJ_PATH}/train_output/${exp_name}

mkdir -p $OUTPUT_PATH

torchrun --nproc_per_node=8 --nnodes=1 \
    ${PROJ_PATH}/src/train/train_model.py \
    --image_transform ${PROJ_PATH}/configs/processer/qwen_448_transform.yaml \
    --tokenizer ${PROJ_PATH}/configs/tokenizer/clm_llama_tokenizer_224loc_anyres.yaml \
    --visual_encoder ${PROJ_PATH}/configs/visual_encoder/qwen_vitg_448.yaml \
    --llm_model ${PROJ_PATH}/configs/clm_models/llm_seed_x_lora.yaml \
    --agent_model ${PROJ_PATH}/configs/clm_models/agent_seed_x.yaml \
    --train_dataset ${PROJ_PATH}/configs/data/dataset.yaml \
    --output_dir ${OUTPUT_PATH} \
    --expr_name ${exp_name} \
    --learning_rate 1e-4 \
    --weight_decay 0.05 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --gradient_accumulation_steps 1 \
    --mixed_precision bf16 \
    --num_train_epochs 10 \
    --max_steps 20000 \
    --save_steps 1000 \
    --lr_scheduler_type cosine \
    --warmup_steps 500 \
    --min_lr_ratio 0.05 \
    --dataloader_num_workers 2 \
    --seed 42 \
    --deepspeed_plugin ${PROJ_PATH}/configs/accelerate/deepspeed_stage_1.yaml \

    # Used to resume from an exisiting checkpoint
    # --resume_from_checkpoint ./train_output/your_model_name/checkpoint-22000 \
    # --resume_steps 22000


echo '--------------------------'
echo Training is done!
echo '--------------------------'
