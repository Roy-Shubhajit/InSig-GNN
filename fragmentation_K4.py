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

def triangle_count(tri_int, tri_ext, graph, subgraph, max_nodes):
    tri_int.eval()
    tri_ext.eval()
    batch, int_out = tri_int(graph, subgraph, max_nodes)
    int_out = new_round(int_out, 0.5)
    int_out = int_out.to(device)
    int_out = int_out.float()
    ext_emb = tri_ext(int_out)
    ext_emb = new_round(ext_emb, 0.5)
    ext_emb = ext_emb.to(device)
    ext_emb = ext_emb.float()
    return ext_emb

def train(args, int_model, ext_model, predictor, loader, pred_opt, loss_fn1):
    step = 0
    total_loss_int = 0
    predictor.train()
    for batch in tqdm(loader):
        graphs = batch[0]
        subgraphs = batch[1]
        subsubgraphs = batch[4]
        for g in range(len(graphs)):
                graphs[g] = graphs[g].to(device)
                for key in subgraphs[g].keys():
                    subgraphs[g][key] = subgraphs[g][key].to(device)
                    for j in subsubgraphs[g][key]:
                        for n in j.keys():
                            j[n] = j[n].to(device)
        
        labels = batch[2].to(device)
        max_nodes = batch[3]
        subgraph_max_nodes = batch[5]
        count = torch.tensor(0.0).to(device)
        for j in range(len(subgraphs)):
            for k in subgraphs[j].keys():
                m = triangle_count(int_model, ext_model, [subgraphs[j][k]], subsubgraphs[j][k], subgraph_max_nodes[j])
                count = count + m
                    
        pred_opt.zero_grad()
        pred = predictor(count)
        loss = loss_fn1(pred, graphs[0].ext_label)
        loss.backward()
        pred_opt.step()
        total_loss_int += loss.item()
        step += 1
        if step % 500 == 0:
            print("Step: {}, Loss: {}".format(
                step, total_loss_int/step))
            
    return total_loss_int / step
        

def eval(args, int_model, ext_model, predictor, loader, loss_fn1):
    step = 0
    total_loss = 0
    predictor.eval()
    for batch in tqdm(loader):
        graphs = batch[0]
        subgraphs = batch[1]
        subsubgraphs = batch[4]
        for g in range(len(graphs)):
                graphs[g] = graphs[g].to(device)
                for key in subgraphs[g].keys():
                    subgraphs[g][key] = subgraphs[g][key].to(device)
                    for j in subsubgraphs[g][key]:
                        for n in j.keys():
                            j[n] = j[n].to(device)
        

        labels = batch[2].to(device)
        max_nodes = batch[3]
        subgraph_max_nodes = batch[5]
        count = torch.tensor(0.0).to(device)
        for j in range(len(subgraphs)):
            for k in subgraphs[j].keys():
                m = triangle_count(int_model, ext_model, [subgraphs[j][k]], subsubgraphs[j][k], subgraph_max_nodes[j])
                count = count + m
        pred = predictor(count)
        pred = new_round(pred, 0.5)
        pred = pred.to(device)
        pred = pred.float()
        loss = loss_fn1(pred, graphs[0].ext_label)
        total_loss += loss.item()
        step += 1
    return total_loss / step

def main(args):

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

    variance = torch.std(dataset.data.K4.to(torch.float))**2


    print("Variance: ", variance)

    collater_fn = frag_collater_K4()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collater_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collater_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collater_fn)

    int_triangle = localGNN_K4(1, 512).to(device)
    ext_triangle = new_external(1, 64).to(device)

    int_triangle.load_state_dict(torch.load('/hdfs1/Data/Shubhajit/Sub-Structure-GNN/save/triangle_insig/Int_GNN_triangle_dataset_2.pt'))
    ext_triangle.load_state_dict(torch.load('/hdfs1/Data/Shubhajit/Sub-Structure-GNN/save/triangle_insig/Ext_GNN_triangle_dataset_2.pt'))

    predict_model = predictor(1, 64).to(device) 

    pred_opt = torch.optim.Adam(predict_model.parameters(), lr=args.lr)
    loss_fn1 = torch.nn.L1Loss(reduction='mean')

    print("Number of parameters in predictor: ", count_parameters(predict_model))

    train_curve = []
    valid_curve = []
    test_curve = []
    best_val_loss = 1000
    best_test_loss = 1000
    for epoch in range(1, args.epochs+1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_loss = train(args, int_triangle, ext_triangle, predict_model, train_loader, pred_opt, loss_fn1)
        print('Train loss : {}'.format(train_loss/variance))

        print('Evaluating...')
        valid_loss = eval(args, int_triangle, ext_triangle, predict_model, val_loader, loss_fn1)
        print('Valid loss : {}'.format(valid_loss/variance))
        test_loss = eval(args, int_triangle, ext_triangle, predict_model, test_loader, loss_fn1)
        print('Test loss : {}'.format(test_loss/variance))
        if valid_loss <= best_val_loss:
            if valid_loss == best_val_loss:
                if test_loss < best_test_loss:
                    best_val_loss = valid_loss
                    best_test_loss = test_loss
                    print("Best Model!")
                    print("Best valid loss : {}".format(best_val_loss/variance))
                    print("Best test loss : {}".format(best_test_loss/variance))
                    #save the best model
                    torch.save(predict_model.state_dict(), "save/{}/predict_model_{}.pt".format(args.output_file, args.dataset))
            else:
                best_val_loss = valid_loss
                best_test_loss = test_loss
                print("Best Model!")
                print("Best valid loss : {}".format(best_val_loss/variance))
                print("Best test loss : {}".format(best_test_loss/variance))
                #save the best model
                torch.save(predict_model.state_dict(), "save/{}/predict_model_{}.pt".format(args.output_file, args.dataset))
        train_curve.append(train_loss/variance)
        valid_curve.append(valid_loss/variance)
        test_curve.append(test_loss/variance)

    print("Best valid loss : {}".format(best_val_loss/variance))
    print("Best test loss : {}".format(best_test_loss/variance))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sub-Structure GNN')
    parser.add_argument('--dataset', type=str, default='dataset_2', help='Dataset: dataset_1 or dataset_2')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--output_file', type=str, default='triangle_2', help='Output file name')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--step', type=int, default=500)
    args = parser.parse_args()
    main(args)