# InSig-GNN

## This is the repository for the paper "**Improving Expressivity of Graph Neural Networks using Localization**".

![model_new](Image/model_new.png)

# InsideOut-GNN

![model](Image/model.png)

The model can be trained using the following command:

```shell
python main.py \
--task triangle/3star/2star/chordal \
--dataset dataset_1/dataset_2 \
--batch_size 1 \
--lr 0.0001 \
--hidden_dim 64 \
--num_layers 2
--epochs 5 \
--step 500 \
--model insig/insideout \
--output_file {Output File Name}
```

To run fragmentation using the previous model:

```shell
python fragmentation_{K4/C4/tailed_triangle}.py \
--dataset dataset_1/dataset_2 \
--num_layers 2 \
--batch_size 1 \
--lr 0.0001 \
--epochs 5 \
--step 500 \
--hidden_dim 64 \
--int_loc {location of internal model only for K4 and C4}
--ext_loc {location of external model only for K4 and C4}
--output_file {Output File Name}
```

To run ablation study on hidden dimension or number of layers:

```shell
chmod +x ablation_{hidden_dim/num_layers}.sh
./ablation_{hidden_dim/num_layers}.sh
```