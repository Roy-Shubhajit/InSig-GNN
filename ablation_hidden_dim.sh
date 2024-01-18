for hidden_dim in 2, 4, 8, 16, 32, 64, 128, 256, 512
do
    echo "hidden_dim: $hidden_dim"
    python3 main.py --task triangle --dataset dataset_2 --batch_size 1 --lr 0.0001 --epochs 5 --step 500 --num_layers 2 --output_file ablation_triangle_hidden_dim_$hidden_dim --model insig --hidden_dim $hidden_dim --ablation hidden_dim
done

