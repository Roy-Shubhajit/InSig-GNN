from tqdm import tqdm
import argparse
import json
from model import *
from helper import *
from dataset_creation import *
from torch.utils.data import DataLoader
import os
import time
import numpy as np
from torch_geometric.datasets import ZINC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(int_model, ext_model, loader, int_opt, ext_opt, args):
    step = 0
    total_loss_int = 0
    total_loss_ext = 0
    for batch in tqdm(loader):
        graphs = batch[0]
        subgraphs = batch[1]
        internal_labels = batch[2].to(device)
        max_nodes = batch[3]

        for g in range(len(graphs)):
            graphs[g] = graphs[g].to(device)
            for key in subgraphs[g].keys():
                subgraphs[g][key] = subgraphs[g][key].to(device)

        int_model.train()
        int_opt.zero_grad()
        batch_graph, int_out = int_model(graphs, subgraphs, max_nodes)
        loss1 = loss_fn1(int_out, internal_labels)
        loss1.backward()
        int_opt.step()
        int_model.eval()
        batch_graph, int_out = int_model(graphs, subgraphs, max_nodes)
        ext_model.train()
        ext_opt.zero_grad()
        if args.model == 'insig':
            int_out = new_round(int_out, args.int_threshold)
            int_out = int_out.to(device)
            int_out = int_out.float()
            ext_emb = ext_model(int_out)
        else:
            ext_emb = ext_model(batch_graph)
        loss2 = loss_fn2(ext_emb, batch_graph.ext_label)
        loss2.backward()
        ext_opt.step()
        total_loss_int += loss1.item()
        total_loss_ext += loss2.item()
        step += 1
        if step % args.step == 0:
            print("Step: {}, Int Loss: {}, Ext Loss: {}".format(
                step, total_loss_int/step, total_loss_ext/step))
    return total_loss_ext / step


def eval(int_model, ext_model, loader, args):
    int_model.eval()
    ext_model.eval()
    step = 0
    total_loss = 0
    for batch in tqdm(loader):
        graphs = batch[0]
        subgraphs = batch[1]
        internal_labels = batch[2].to(device)
        max_nodes = batch[3]

        for g in range(len(graphs)):
            graphs[g] = graphs[g].to(device)
            for key in subgraphs[g].keys():
                subgraphs[g][key] = subgraphs[g][key].to(device)
        batch_graph, int_out = int_model(graphs, subgraphs, max_nodes)
        if args.model == 'insig':
            int_out = new_round(int_out, args.int_threshold)
            int_out = int_out.to(device)
            int_out = int_out.float()
            ext_emb = ext_model(int_out)
        else:
            ext_emb = ext_model(batch_graph)
        ext_emb = new_round(ext_emb, args.ext_threshold)
        ext_emb = ext_emb.to(device)
        ext_emb = ext_emb.float()
        loss = loss_fn2(ext_emb, batch_graph.ext_label)
        total_loss += loss.item()
        step += 1
    return total_loss / step


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sub-Structure GNN')
    parser.add_argument('--task', type=str, default='triangle')
    parser.add_argument('--dataset', type=str, default='dataset_2')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--step', type=int, default=500)
    parser.add_argument('--int_threshold', type=float, default=0.5)
    parser.add_argument('--ext_threshold', type=float, default=0.5)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--model', type=str, default='insig')
    parser.add_argument('--ablation', type=str, default='none')
    args = parser.parse_args()

    if 'save/'+args.output_file not in os.listdir():
        os.mkdir('save/'+args.output_file)

    if args.dataset == 'dataset_1':
        dataset = Dataset_1_orig(root="data/Dataset1", pre_transform=None)
        train_dataset = dataset[:int(len(dataset)*0.8)]
        val_dataset = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
        test_dataset = dataset[int(len(dataset)*0.9):]
    elif args.dataset == 'dataset_2':
        dataset = Dataset_2_orig(root="data/Dataset2", pre_transform=None)
        train_dataset = dataset[:int(len(dataset)*0.8)]
        val_dataset = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
        test_dataset = dataset[int(len(dataset)*0.9):]
    elif args.dataset == 'dataset_chembl':
        dataset = Dataset_chembl(root="data/Dataset_chembl", pre_transform=None)
        train_dataset = dataset[:int(len(dataset)*0.8)]
        val_dataset = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
        test_dataset = dataset[int(len(dataset)*0.9):]
    elif args.dataset == 'zinc_subset':
        train_dataset = ZINC(root='data/ZINC', subset=True, split='train', pre_transform=None)
        test_dataset = ZINC(root='data/ZINC', subset=True, split='test', pre_transform=None)
        val_dataset = ZINC(root='data/ZINC', subset=True, split='val', pre_transform=None)  
    elif args.dataset == 'zinc_full':
        train_dataset = ZINC(root='data/ZINC', subset=False, split='train', pre_transform=None)
        test_dataset = ZINC(root='data/ZINC', subset=False, split='test', pre_transform=None)
        val_dataset = ZINC(root='data/ZINC', subset=False, split='val', pre_transform=None)  

    collater_fn = collater(args.task)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, collate_fn=collater_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, drop_last=True, collate_fn=collater_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, drop_last=True, collate_fn=collater_fn, pin_memory=True)

    Int_GNN = localGNN(args.num_layers, args.hidden_dim).to(device)
    if args.model == 'insig':
        Ext_GNN = new_external(1, 32).to(device)
    else:
        Ext_GNN = globalGNN(args.num_layers, args.hidden_dim).to(device)
    Int_Opt = torch.optim.Adam(Int_GNN.parameters(), lr=args.lr)
    Ext_Opt = torch.optim.Adam(Ext_GNN.parameters(), lr=args.lr)
    loss_fn1 = torch.nn.L1Loss(reduction='mean')
    loss_fn2 = torch.nn.L1Loss(reduction='mean')

    train_curve = []
    valid_curve = []
    test_curve = []
    best_val_loss = 1000
    best_test_loss = 1000
    
    with open(f'data/{args.dataset}_variance.json') as f:
        variance = json.load(f)
    train_variance = torch.tensor(variance['Training'][args.task])
    val_variance = torch.tensor(variance['Validation'][args.task])
    test_variance = torch.tensor(variance['Test'][args.task])

    print("Train Variance: ", train_variance.item())
    print("Val Variance: ", val_variance.item())
    print("Test Variance: ", test_variance.item())
    if  train_variance == 0:
        print("Train Variance is 0, setting it to 1")
        train_variance = torch.tensor(1)
    if  val_variance == 0:
        print("Val Variance is 0, setting it to 1")
        val_variance = torch.tensor(1)
    if  test_variance == 0:
        print("Test Variance is 0, setting it to 1")
        test_variance = torch.tensor(1)
    print("Number of parameters in Int_GNN: ", count_parameters(Int_GNN))
    print("Number of parameters in Ext_GNN: ", count_parameters(Ext_GNN))

    train_time = []
    eval_time = []
    for epoch in range(1, args.epochs+1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        start_time = time.time()
        train_loss = train(Int_GNN, Ext_GNN, train_loader,
                           Int_Opt, Ext_Opt, args)
        end_time = time.time()
        train_time.append(end_time-start_time)
        print('Train loss : {}'.format(train_loss/train_variance))

        print('Evaluating...')
        start_time = time.time()
        valid_loss = eval(Int_GNN, Ext_GNN, val_loader, args)
        end_time = time.time()
        eval_time.append((end_time-start_time)/len(val_dataset))
        print('Valid loss : {}'.format(valid_loss/val_variance))
        test_loss = eval(Int_GNN, Ext_GNN, test_loader, args)
        print('Test loss : {}'.format(test_loss/test_variance))
        if valid_loss <= best_val_loss:
            if valid_loss == best_val_loss:
                # only save if the test loss is lower
                if test_loss < best_test_loss:
                    best_val_loss = valid_loss
                    best_test_loss = test_loss
                    print("Best Model!")
                    print("Best valid loss : {}".format(best_val_loss/val_variance))
                    print("Best test loss : {}".format(
                        best_test_loss/test_variance))
                    # save the best model
                    torch.save(Int_GNN.state_dict(
                    ), "save/{}/Int_GNN_{}_{}.pt".format(args.output_file, args.task, args.dataset))
                    torch.save(Ext_GNN.state_dict(
                    ), "save/{}/Ext_GNN_{}_{}.pt".format(args.output_file, args.task, args.dataset))
            else:
                best_val_loss = valid_loss
                best_test_loss = test_loss
                print("Best Model!")
                print("Best valid loss : {}".format(best_val_loss/val_variance))
                print("Best test loss : {}".format(best_test_loss/test_variance))
                # save the best model
                torch.save(Int_GNN.state_dict(
                ), "save/{}/Int_GNN_{}_{}.pt".format(args.output_file, args.task, args.dataset))
                torch.save(Ext_GNN.state_dict(
                ), "save/{}/Ext_GNN_{}_{}.pt".format(args.output_file, args.task, args.dataset))
        train_curve.append(train_loss/train_variance)
        valid_curve.append(valid_loss/val_variance)
        test_curve.append(test_loss/test_variance)

    print("Best valid loss : {}".format(best_val_loss/val_variance))
    print("Best test loss : {}".format(best_test_loss/test_variance))
    print("Average train time per epoch: {}".format(np.mean(train_time)))
    print("Average inference time per sample: {}".format(np.mean(eval_time)))

    import csv
    if args.ablation == 'num_layers':
        with open(f'save/ablation/{args.task}_num_layers.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([args.model, args.dataset, args.num_layers, count_parameters(Int_GNN), count_parameters(Ext_GNN), (best_test_loss/test_variance).item(), (best_val_loss/val_variance).item()])
    elif args.ablation == 'hidden_dim':
        with open(f'save/ablation/{args.task}_hidden_dim.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([args.model, args.dataset, args.hidden_dim, count_parameters(Int_GNN), count_parameters(Ext_GNN), (best_test_loss/test_variance).item(), (best_val_loss/val_variance).item()])
    

