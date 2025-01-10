seed=(2022 2023 2024 2025 2026)
cuda_id=0
ve="deit_small_patch16_224_in661"
bs=('deit_small_patch16_224_in661_C10_5T_hat')
seqfile=('C10_5T')
learning_rate=(0.005)
num_train_epochs=(30)
base_dir="ckpt"
final_task=(4)
latent=(64)
buffersize=(200)

for round in 0 1 2 3 4;
do
  for class_order in 0;
  do
    for i in "${!bs[@]}";
    do
        for ft_task in $(seq 0 ${final_task[$i]});
        do
            CUDA_VISIBLE_DEVICES=$cuda_id python main.py \
            --task ${ft_task} \
            --idrandom 0 \
            --visual_encoder $ve \
            --baseline "${bs[$i]}" \
            --seed ${seed[$round]} \
            --batch_size 64 \
            --sequence_file "${seqfile[$i]}" \
            --learning_rate ${learning_rate[$i]} \
            --num_train_epochs ${num_train_epochs[$i]} \
            --base_dir ckpt \
            --class_order ${class_order} \
            --latent  ${latent[$i]} \
            --replay_buffer_size ${buffersize[$i]} \
            --training
        done
        for ft_task in $(seq 0 ${final_task[$i]});
        do
            CUDA_VISIBLE_DEVICES=$cuda_id python eval.py \
            --task ${ft_task} \
            --idrandom 0 \
            --visual_encoder $ve \
            --baseline "${bs[$i]}" \
            --seed ${seed[$round]} \
            --batch_size 64 \
            --sequence_file "${seqfile[$i]}" \
            --base_dir ckpt \
            --class_order ${class_order} \
            --latent  ${latent[$i]} \
            --replay_buffer_size ${buffersize[$i]}
        done
    done
  done
done