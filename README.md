# InSig-GNN

![model_new](https://github.com/Roy-Shubhajit/InSig-GNN/assets/81477286/3e050e3e-d213-4def-b440-01af8bb9810e)

# InsideOut-GNN

![model](https://github.com/Roy-Shubhajit/InSig-GNN/assets/81477286/e5c66609-337f-4a04-8eae-42ca7e71bcc3)

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
