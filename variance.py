from dataset_creation import *
from dataset_labels import *
import argparse
import numpy as np
from torch_geometric.datasets import ZINC
import pandas as pd
import json
import os
from tqdm import tqdm
from helper import *

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
    test_dataset = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
    val_dataset = dataset[int(len(dataset)*0.9):]
elif args.dataset == 'dataset_chembl_chordals':
        dataset = Dataset_chembl(root="data/Dataset_chembl_chordals", pre_transform=add_chordal)
        train_dataset = dataset[:int(len(dataset)*0.8)]
        test_dataset = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
        val_dataset = dataset[int(len(dataset)*0.9):]
elif args.dataset == 'zinc_subset':
    train_dataset = ZINC(root='data/ZINC', subset=True, split='train', pre_transform=count_labels)
    test_dataset = ZINC(root='data/ZINC', subset=True, split='test', pre_transform=count_labels)
    val_dataset = ZINC(root='data/ZINC', subset=True, split='val', pre_transform=count_labels) 
elif args.dataset == 'zinc_subset_chordal':
    train_dataset = ZINC(root='data/ZINC_chordals', subset=True, split='train', pre_transform=add_chordal)
    test_dataset = ZINC(root='data/ZINC_chordals', subset=True, split='test', pre_transform=add_chordal)
    val_dataset = ZINC(root='data/ZINC_chordals', subset=True, split='val', pre_transform=add_chordal)
elif args.dataset == 'zinc_full':
    train_dataset = ZINC(root='data/ZINC', subset=False, split='train', pre_transform=count_labels)
    test_dataset = ZINC(root='data/ZINC', subset=False, split='test', pre_transform=count_labels)
    val_dataset = ZINC(root='data/ZINC', subset=False, split='val', pre_transform=count_labels)
elif args.dataset == 'zinc_full_chordal':
    train_dataset = ZINC(root='data/ZINC_chordals', subset=False, split='train', pre_transform=add_chordal)
    test_dataset = ZINC(root='data/ZINC_chordals', subset=False, split='test', pre_transform=add_chordal)
    val_dataset = ZINC(root='data/ZINC_chordals', subset=False, split='val', pre_transform=add_chordal)

calculate_dataset_variance(args.dataset, train_dataset, val_dataset, test_dataset)