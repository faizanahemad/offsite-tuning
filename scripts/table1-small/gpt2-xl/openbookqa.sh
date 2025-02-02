TASK="openbookqa"
MODEL="gpt2-large"
num_student_layers=4
bs=4
pad=2
eval_steps=10

lr=3e-4

bash scripts/table1-small/ft_all.sh ${TASK} ${MODEL} ${num_student_layers} ${bs} ${pad} ${eval_steps}

bash scripts/table1-small/ft_emulator.sh ${TASK} ${MODEL} ${num_student_layers} ${bs} ${pad} ${eval_steps} ${lr}

bash scripts/table1-small/eval.sh ${TASK} ${MODEL} ${pad} ${num_student_layers}