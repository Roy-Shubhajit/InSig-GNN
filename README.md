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
--epochs 100 \
--step 500 \
--output_file {Output File Name}
```

To run fragmentation using the previous model:

```shell
python fragmentation_{K4/C4}.py \
--dataset dataset_1/dataset_2 \
--batch_size 1 \
--lr 0.0001 \
--epochs 100 \
--step 500 \
--output_file {Output File Name}
```
