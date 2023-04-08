TASK=$1               # "hellaswag"
MODEL=$2              # "facebook/opt-1.3b"
num_student_layers=$3 # 8
bs=$4                 # 8
pad=$5                # 2
eval_steps=$6         # 10

### baseline ft
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --mixed_precision=fp16 --zero_stage 3 --offload_optimizer_device "cpu" --offload_param_device "cpu" --zero3_init_flag true  \
    offsite_tuning/run_clm.py \
    --model_name_or_path $MODEL \
    --dataset_name ${TASK} \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --lm_weight 1.0 \
    --kd_weight 0.0 \
    --seed 42 \
    --gradient_accumulation_steps 4 \
    --eval_steps $eval_steps \
    --train_module all \
    --save_module all \
    --no_teacher \
    --output_dir logs/table1-small/$MODEL/${TASK}/ft_all
