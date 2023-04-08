# l25
MODEL="gpt2-large"
student_model_name_or_path="gpt2"
num_student_layers=6
bs=2
pad=2


export WANDB_PROJECT="offsite_tuning"
export WANDB_NAME="${MODEL}_emulator_${num_student_layers}_${pad}_${pad}"
export WANDB_MODE="dryrun"
export WANDB_DISABLE_SERVICE=true
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --mixed_precision=fp16 --multi_gpu\
    offsite_tuning/run_clm.py \
    --model_name_or_path $MODEL \
    --train_tokenized_dataset $HOME/processed_datasets/wikitext_tokenized_blocks \
    --val_tokenized_dataset $HOME/processed_datasets/wikitext_tokenized_blocks \
    --preprocessing_num_workers 88 \
    --per_device_train_batch_size $((bs * 2)) \
    --per_device_eval_batch_size $((bs * 2)) \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-3 \
    --weight_decay 0.1 \
    --num_warmup_steps 400 \
    --lr_scheduler_type cosine \
    --num_train_epochs 1 \
    --lm_weight 1.0 \
    --kd_weight 60.0 \
    --seed 37 \
    --block_size 512 \
    --eval_steps 100 \
    --num_student_layers $num_student_layers \
    --student_l_pad $pad \
    --student_r_pad $pad \
    --train_module "student_patch" \
    --max_train_steps 1200 \
    --max_train_samples 100000 \
    --output_dir emulators/${MODEL}/${num_student_layers}_${pad}_${pad} \
    --report_to wandb \
    --student_model_name_or_path $student_model_name_or_path

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --mixed_precision=fp16 --multi_gpu\
    offsite_tuning/run_clm.py \
    --model_name_or_path $MODEL \
    --train_tokenized_dataset $HOME/processed_datasets/wikitext_tokenized_blocks \
    --val_tokenized_dataset $HOME/processed_datasets/wikitext_tokenized_blocks \
    --preprocessing_num_workers 88 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-4 \
    --weight_decay 0.1 \
    --num_warmup_steps 1000 \
    --lr_scheduler_type cosine \
    --num_train_epochs 2 \
    --lm_weight 1.0 \
    --kd_weight 60.0 \
    --seed 42 \
    --block_size 512 \
    --eval_steps 100 \
    --num_student_layers $num_student_layers \
    --student_l_pad $pad \
    --student_r_pad $pad \
    --output_dir emulators/${MODEL}/${num_student_layers}_${pad}_${pad} \
    --report_to wandb \
    --student_model_name_or_path $student_model_name_or_path \
    --load_student emulators/${MODEL}/${num_student_layers}_${pad}_${pad}/student.pt
