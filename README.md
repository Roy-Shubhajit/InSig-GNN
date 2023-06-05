# InSig-GNN
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improving-expressivity-of-graph-neural-1/subgraph-counting-k4-on-synthetic-graph)](https://paperswithcode.com/sota/subgraph-counting-k4-on-synthetic-graph?p=improving-expressivity-of-graph-neural-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improving-expressivity-of-graph-neural-1/subgraph-counting-triangle-on-synthetic-graph)](https://paperswithcode.com/sota/subgraph-counting-triangle-on-synthetic-graph?p=improving-expressivity-of-graph-neural-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improving-expressivity-of-graph-neural-1/subgraph-counting-3-star-on-synthetic-graph)](https://paperswithcode.com/sota/subgraph-counting-3-star-on-synthetic-graph?p=improving-expressivity-of-graph-neural-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improving-expressivity-of-graph-neural-1/subgraph-counting-chordal-c4-on-synthetic)](https://paperswithcode.com/sota/subgraph-counting-chordal-c4-on-synthetic?p=improving-expressivity-of-graph-neural-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improving-expressivity-of-graph-neural-1/subgraph-counting-2-star-on-synthetic-graph)](https://paperswithcode.com/sota/subgraph-counting-2-star-on-synthetic-graph?p=improving-expressivity-of-graph-neural-1)

## This is the repository for the paper [**Improving Expressivity of Graph Neural Networks using Localization**](https://arxiv.org/abs/2305.19659v1).
![model_new](Image/model_new.png)


# InsideOut-GNN

![model](Image/model.png)

The model can be trained using the following command:

```shell
python main.py \
--task triangle/3star \
--dataset dataset_1/dataset_2 \
--batch_size 1 \
--lr 0.0001 \
--epochs 100 \
--step 500
```


To run fragmentation using the previous model:
```shell
python fragmentation_main.py \
--task K4/chordal \
--dataset dataset_1/dataset_2 \
--batch_size 1 \
--lr 0.0001 \
--epochs 100 \
--step 500 \
--output_file chordal_frag
```
