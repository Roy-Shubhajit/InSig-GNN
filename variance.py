from dataset_creation import *
from dataset_labels import *
import argparse
import numpy as np
from torch_geometric.datasets import ZINC
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='dataset_1')
args = parser.parse_args()

if args.dataset == 'dataset_1':
    dataset = Dataset_1_orig(root='data/Dataset1', pre_transform=None)
    train_dataset = dataset[:int(len(dataset)*0.8)]
    val_dataset = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
    test_dataset = dataset[int(len(dataset)*0.9):]
elif args.dataset == 'dataset_2':
    dataset = Dataset_2_orig(root='data/Dataset2', pre_transform=None)
    train_dataset = dataset[:int(len(dataset)*0.8)]
    val_dataset = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
    test_dataset = dataset[int(len(dataset)*0.9):]
elif args.dataset =='dataset_chembl':
    dataset = Dataset_chembl(root='data/Dataset_chembl', pre_transform=None)
    train_dataset = dataset[:int(len(dataset)*0.8)]
    val_dataset = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
    test_dataset = dataset[int(len(dataset)*0.9):]
elif args.dataset == 'zinc_subset':
    train_dataset = ZINC(root='data/ZINC', subset=True, split='train', pre_transform=None, transform=count_labels)
    test_dataset = ZINC(root='data/ZINC', subset=True, split='test', pre_transform=None, transform=count_labels)
    val_dataset = ZINC(root='data/ZINC', subset=True, split='val', pre_transform=None, transform=count_labels)  
elif args.dataset == 'zinc_full':
    train_dataset = ZINC(root='data/ZINC', subset=False, split='train', pre_transform=None, transform=count_labels)
    test_dataset = ZINC(root='data/ZINC', subset=False, split='test', pre_transform=None, transform=count_labels)
    val_dataset = ZINC(root='data/ZINC', subset=False, split='val', pre_transform=None, transform=count_labels)

train_triangle = torch.tensor([])
train_3star = torch.tensor([])
train_chordal = torch.tensor([])
train_2star = torch.tensor([])
train_k4 = torch.tensor([])
train_C4 = torch.tensor([])
train_tailed_triangle = torch.tensor([])

val_triangle = torch.tensor([])
val_3star = torch.tensor([])
val_chordal = torch.tensor([])
val_2star = torch.tensor([])
val_k4 = torch.tensor([])
val_C4 = torch.tensor([])
val_tailed_triangle = torch.tensor([])

test_triangle = torch.tensor([])
test_3star = torch.tensor([])
test_chordal = torch.tensor([])
test_2star = torch.tensor([])
test_k4 = torch.tensor([])
test_C4 = torch.tensor([])
test_tailed_triangle = torch.tensor([])

for i in range(len(train_dataset)):
    train_triangle = torch.cat((train_triangle, train_dataset[i].triangle))
    train_3star = torch.cat((train_3star, train_dataset[i].star))
    train_chordal = torch.cat((train_chordal, train_dataset[i].chordal_cycle))
    train_2star = torch.cat((train_2star, train_dataset[i].star_2))
    train_k4 = torch.cat((train_k4, train_dataset[i].K4))
    train_C4 = torch.cat((train_C4, train_dataset[i].C4))
    train_tailed_triangle = torch.cat((train_tailed_triangle, train_dataset[i].tailed_triangle))

for i in range(len(val_dataset)):
    val_triangle = torch.cat((val_triangle, val_dataset[i].triangle))
    val_3star = torch.cat((val_3star, val_dataset[i].star))
    val_chordal = torch.cat((val_chordal, val_dataset[i].chordal_cycle))
    val_2star = torch.cat((val_2star, val_dataset[i].star_2))
    val_k4 = torch.cat((val_k4, val_dataset[i].K4))
    val_C4 = torch.cat((val_C4, val_dataset[i].C4))
    val_tailed_triangle = torch.cat((val_tailed_triangle, val_dataset[i].tailed_triangle))

for i in range(len(test_dataset)):
    test_triangle = torch.cat((test_triangle, test_dataset[i].triangle))
    test_3star = torch.cat((test_3star, test_dataset[i].star))
    test_chordal = torch.cat((test_chordal, test_dataset[i].chordal_cycle))
    test_2star = torch.cat((test_2star, test_dataset[i].star_2))
    test_k4 = torch.cat((test_k4, test_dataset[i].K4))
    test_C4 = torch.cat((test_C4, test_dataset[i].C4))
    test_tailed_triangle = torch.cat((test_tailed_triangle, test_dataset[i].tailed_triangle))

train_triangle_var = torch.std(train_triangle)**2
train_3star_var = torch.std(train_3star)**2
train_chordal_var = torch.std(train_chordal)**2
train_2star_var = torch.std(train_2star)**2
train_k4_var = torch.std(train_k4)**2
train_C4_var = torch.std(train_C4)**2
train_tailed_triangle_var = torch.std(train_tailed_triangle)**2

val_triangle_var = torch.std(val_triangle)**2
val_3star_var = torch.std(val_3star)**2
val_chordal_var = torch.std(val_chordal)**2
val_2star_var = torch.std(val_2star)**2
val_k4_var = torch.std(val_k4)**2
val_C4_var = torch.std(val_C4)**2
val_tailed_triangle_var = torch.std(val_tailed_triangle)**2

test_triangle_var = torch.std(test_triangle)**2
test_3star_var = torch.std(test_3star)**2
test_chordal_var = torch.std(test_chordal)**2
test_2star_var = torch.std(test_2star)**2
test_k4_var = torch.std(test_k4)**2
test_C4_var = torch.std(test_C4)**2
test_tailed_triangle_var = torch.std(test_tailed_triangle)**2

variance_table = pd.DataFrame({
    'Train': [train_triangle_var.item(), train_3star_var.item(), train_chordal_var.item(), train_2star_var.item(), train_k4_var.item(), train_C4_var.item(), train_tailed_triangle_var.item()],
    'Validation': [val_triangle_var.item(), val_3star_var.item(), val_chordal_var.item(), val_2star_var.item(), val_k4_var.item(), val_C4_var.item(), val_tailed_triangle_var.item()],
    'Test': [test_triangle_var.item(), test_3star_var.item(), test_chordal_var.item(), test_2star_var.item(), test_k4_var.item(), test_C4_var.item(), test_tailed_triangle_var.item()]
}, index=['Triangle', '3-Star', 'Chordal Cycle', '2-Star', 'K4', 'C4', 'Tailed Triangle'])

print(variance_table)

