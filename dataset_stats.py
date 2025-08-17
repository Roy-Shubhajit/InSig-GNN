from tqdm import tqdm
import argparse
import json
from model import *
from helper import *
from dataset_creation import *
from dataset_labels import *
from torch.utils.data import DataLoader
import os
import time
import numpy as np
from torch_geometric.datasets import ZINC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset Statistics')
    parser.add_argument('--dataset', type=str, default='ZINC', help='Dataset name')
    args = parser.parse_args()
    if args.dataset == 'dataset_1':
        dataset = Dataset_1_orig(root='data/Dataset_1', pre_transform=None)
        
    elif args.dataset == 'dataset_2':
        dataset = Dataset_2_orig(root='data/Dataset_2', pre_transform=None)
    for task in ['triangle', '2star', '3star', 'chordal', 'K4', 'C4', 'Tailed_Triangle']:
        args.task = task
        if task == 'K4':
            collater_fn = frag_collater_K4()
        elif task == 'C4':
            collater_fn = frag_collater_C4()
        elif task == 'Tailed_Triangle':
            collater_fn = frag_collater_tailed_triangle()
        else:
            collater_fn = collater(args.task)
        
        loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collater_fn)
        pattern_counts = []
        num_nodes = []
        num_edges = []
        for graph_data in tqdm(loader, desc=f"Processing {task}"):
            graph = graph_data[0][0].to(device)
            orig_graph = graph_data[-1][0].to(device)
            if task == 'Tailed_Triangle':
                pattern_counts.append(graph.ext_label.sum().item())
            else:
                pattern_counts.append(graph.ext_label.item())
            if orig_graph.edge_index.size(1) == 0:
                num_nodes.append(0)
                num_edges.append(0)
            else:
                num_nodes.append(torch.max(orig_graph.edge_index).item()+1)
                num_edges.append(orig_graph.num_edges)

        pattern_counts = np.array(pattern_counts)
        num_nodes = np.array(num_nodes)
        num_edges = np.array(num_edges)
        if not os.path.exists('dataset_stats.csv'):
            with open('dataset_stats.csv', 'w') as f:
                f.write('dataset,task,num_graphs,total_pattern_count,zero_counts,std_count,avg_num_nodes,avg_num_edges\n')
        with open('dataset_stats.csv', 'a') as f:
            f.write(f"{args.dataset},{task},{len(pattern_counts)},{pattern_counts.sum()},{(pattern_counts == 0).sum()},{pattern_counts.std()},{num_nodes.mean()},{num_edges.mean()}\n")

        print(f"Task: {task}, Num graphs: {len(pattern_counts)}, Total pattern count: {pattern_counts.sum()}, "
              f"Zero counts: {(pattern_counts == 0).sum()}, Std count: {pattern_counts.std()}, Avg num nodes: {num_nodes.mean()}, Avg num edges: {num_edges.mean()}")
        print(f"Statistics for {task} saved to dataset_stats.csv")
        