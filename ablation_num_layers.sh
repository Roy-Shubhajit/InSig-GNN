for num_layer in 1 2 3
do
    echo "num_layer: $num_layer"
    python3 main.py --task triangle --dataset dataset_2 --batch_size 1 --lr 0.0001 --epochs 5 --step 500 --num_layers $num_layer --output_file ablation_triangle_num_layers_$num_layer --model insig --hidden_dim 4 --ablation num_layers
done

