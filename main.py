from tqdm import tqdm
import argparse
from model import *
from helper import *
from dataset_creation import *
from torch.utils.data import DataLoader
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(int_model, ext_model, loader, int_opt, ext_opt):
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
        ext_emb = ext_model(int_out)
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

def eval(int_model, ext_model, loader):
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
        ext_emb = ext_model(int_out)
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
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--step', type=int, default=500)
    parser.add_argument('--output_file', type=str)
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
    
    if args.task == 'triangle':
        variance = torch.std(dataset.data.triangle.to(torch.float32))**2
    elif args.task == '3star':
        variance = torch.std(dataset.data.star.to(torch.float32))**2
    elif args.task == 'C4':
        variance = torch.std(dataset.data.C4.to(torch.float32))**2
    elif args.task == 'chordal':
        variance = torch.std(dataset.data.chordal_cycle.to(torch.float32))**2
    elif args.task == '2star':
        variance = torch.std(dataset.data.star_2.to(torch.float32))**2

    print("Variance: ", variance)

    collater_fn = collater(args.task)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collater_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=collater_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=collater_fn)

    Int_GNN = localGNN(1, 512).to(device)
    Ext_GNN = new_external(1, 64).to(device)
    Int_Opt = torch.optim.Adam(Int_GNN.parameters(), lr=args.lr)
    Ext_Opt = torch.optim.Adam(Ext_GNN.parameters(), lr=args.lr)
    loss_fn1 = torch.nn.L1Loss(reduction='mean')
    loss_fn2 = torch.nn.L1Loss(reduction='mean')

    print("Number of parameters in Int_GNN: ", count_parameters(Int_GNN))
    print("Number of parameters in Ext_GNN: ", count_parameters(Ext_GNN))

    train_curve = []
    valid_curve = []
    test_curve = []
    best_val_loss = 1000
    best_test_loss = 1000
    for epoch in range(1, 101):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_loss = train(Int_GNN, Ext_GNN, train_loader, Int_Opt, Ext_Opt)
        print('Train loss : {}'.format(train_loss/variance))

        print('Evaluating...')
        valid_loss = eval(Int_GNN, Ext_GNN, val_loader)
        print('Valid loss : {}'.format(valid_loss/variance))
        test_loss = eval(Int_GNN, Ext_GNN, test_loader)
        print('Test loss : {}'.format(test_loss/variance))
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_test_loss = test_loss
            print("Best Model!")
            print("Best valid loss : {}".format(best_val_loss/variance))
            print("Best test loss : {}".format(best_test_loss/variance))
            #save the best model
            torch.save(Int_GNN.state_dict(), "save/{}/Int_GNN_{}_{}.pt".format(args.output_file, args.task, args.dataset))
            torch.save(Ext_GNN.state_dict(), "save/{}/Ext_GNN_{}_{}.pt".format(args.output_file, args.task, args.dataset))
        train_curve.append(train_loss/variance)
        valid_curve.append(valid_loss/variance)
        test_curve.append(test_loss/variance)

    print("Best valid loss : {}".format(best_val_loss/variance))
    print("Best test loss : {}".format(best_test_loss/variance))

    
    
