# l25
MODEL="gpt2-xl"
student_model_name_or_path="gpt2"
num_student_layers=6
bs=4
pad=2
export WANDB_PROJECT="offsite_tuning"
export WANDB_NAME="${MODEL}_emulator_${num_student_layers}_${pad}_${pad}"
export WANDB_MODE="dryrun"
export WANDB_DISABLE_SERVICE=true

rm emulators/${MODEL}/${num_student_layers}_${pad}_${pad}/all_results.json

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --mixed_precision=bf16 --num_machines 8 --machine_rank 0 --main_process_ip "0.0.0.0" --main_process_port 9999 --max_restarts 3  --multi_gpu \
    offsite_tuning/run_clm.py \
    --model_name_or_path $MODEL \
    --train_tokenized_dataset $HOME/processed_datasets/pile_subsampled_tokenized_blocks \
    --val_tokenized_dataset $HOME/processed_datasets/wikitext_tokenized_blocks \
    --preprocessing_num_workers 64 \
    --per_device_train_batch_size $((bs * 2)) \
    --per_device_eval_batch_size $((bs * 2)) \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --num_warmup_steps 800 \
    --lr_scheduler_type cosine \
    --num_train_epochs 1 \
    --lm_weight 1.0 \
    --kd_weight 30.0 \
    --seed 37 \
    --block_size 512 \
    --eval_steps 100 \
    --num_student_layers $num_student_layers \
    --student_l_pad $pad \
    --student_r_pad $pad \
    --train_module "student_patch" \
    --max_train_steps 4000 \
    --gradient_checkpointing_enable\
    --output_dir emulators/${MODEL}/${num_student_layers}_${pad}_${pad} \
    --report_to wandb \
    --student_model_name_or_path $student_model_name_or_path


CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --mixed_precision=bf16 --num_machines 8 --machine_rank 0 --main_process_ip "0.0.0.0" --main_process_port 9999 --max_restarts 3 --multi_gpu\
    offsite_tuning/run_clm.py \
    --model_name_or_path $MODEL \
    --train_tokenized_dataset $HOME/processed_datasets/pile_subsampled_tokenized_blocks \
    --val_tokenized_dataset $HOME/processed_datasets/wikitext_tokenized_blocks \
    --preprocessing_num_workers 88 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --num_warmup_steps 2000 \
    --lr_scheduler_type cosine \
    --num_train_epochs 1 \
    --lm_weight 1.0 \
    --kd_weight 30.0 \
    --seed 42 \
    --block_size 512 \
    --eval_steps 200 \
    --num_student_layers $num_student_layers \
    --student_l_pad $pad \
    --student_r_pad $pad \
    --gradient_checkpointing_enable \
    --output_dir emulators/${MODEL}/${num_student_layers}_${pad}_${pad} \
    --report_to wandb \
    --student_model_name_or_path $student_model_name_or_path \
    --load_student emulators/${MODEL}/${num_student_layers}_${pad}_${pad}/student.pt