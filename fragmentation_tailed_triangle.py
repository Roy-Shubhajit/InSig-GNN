from tqdm import tqdm
import argparse
from model import *
from helper import *
from dataset_creation import *
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.datasets import ZINC, QM7b
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def edge_count(edge_int, edge_ext, graph, subgraph, max_nodes):
    edge_int.eval()
    edge_ext.eval()
    batch, int_out = edge_int(graph, subgraph, max_nodes)
    int_out = new_round(int_out, 0.5)
    int_out = int_out.to(device)
    int_out = int_out.float()
    ext_emb = edge_ext(int_out)
    ext_emb = new_round(ext_emb, 0.5)
    ext_emb = ext_emb.to(device)
    ext_emb = ext_emb.float()
    return ext_emb


def node_count(node_int, node_ext, graph, subgraph, max_nodes):
    node_int.eval()
    node_ext.eval()
    batch, int_out = node_int(graph, subgraph, max_nodes)
    int_out = new_round(int_out, 0.5)
    int_out = int_out.to(device)
    int_out = int_out.float()
    ext_emb = node_ext(int_out)
    ext_emb = new_round(ext_emb, 0.5)
    ext_emb = ext_emb.to(device)
    ext_emb = ext_emb.float()
    return ext_emb


def train(args, int_node, ext_node, int_edge, ext_edge, predictor, loader, pred_opt, loss_fn1):
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
        count_edge = torch.tensor([]).to(device)
        count_node = torch.tensor([]).to(device)
        for j in range(len(subgraphs)):
            for k in subgraphs[j].keys():
                c1 = edge_count(int_edge, ext_edge, [subsubgraphs[j][k][0]['node_graph']], [
                                {'node_graph': subsubgraphs[j][k][0]['node_graph']}], subgraph_max_nodes[j])
                c2 = edge_count(int_edge, ext_edge, [subsubgraphs[j][k][0]['edge_graph']], [
                                {'edge_graph': subsubgraphs[j][k][0]['edge_graph']}], subgraph_max_nodes[j])
                e_c = torch.tensor([[c1, c2]]).to(device)
                count_edge = torch.cat((count_edge, e_c), 0)
                c3 = node_count(int_node, ext_node, [subsubgraphs[j][k][0]['node_graph']], [
                                {'node_graph': subsubgraphs[j][k][0]['node_graph']}], subgraph_max_nodes[j])
                c4 = node_count(int_node, ext_node, [subsubgraphs[j][k][0]['edge_graph']], [
                                {'edge_graph': subsubgraphs[j][k][0]['edge_graph']}], subgraph_max_nodes[j])
                n_c = torch.tensor([[c3, c4]]).to(device)
                count_node = torch.cat((count_node, n_c), 0)

        pred_opt.zero_grad()
        new_inp = torch.cat((count_edge, count_node), dim=1).to(device)
        new_train_data = TensorDataset(new_inp, graphs[0].ext_label)
        new_train_loader = DataLoader(
            new_train_data, batch_size=1, shuffle=True)
        for batch in new_train_loader:
            new_inp = batch[0].to(device)
            label = batch[1].to(device)
            pred = predictor(new_inp)
            loss = loss_fn1(pred.squeeze(), label)
            loss.backward()
            pred_opt.step()
            total_loss_int += loss.item()
        step += 1
        if step % args.step == 0:
            print("Step: {}, Loss: {}".format(
                step, total_loss_int/step))

    return total_loss_int / step


def eval(args, int_node, ext_node, int_edge, ext_edge, predictor, loader, loss_fn1):
    step = 0
    total_loss_int = 0
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
        count_edge = torch.tensor([]).to(device)
        count_node = torch.tensor([]).to(device)
        for j in range(len(subgraphs)):
            for k in subgraphs[j].keys():
                c1 = edge_count(int_edge, ext_edge, [subgraphs[j][k]], [
                                {'node_graph': subsubgraphs[j][k][0]['node_graph']}], subgraph_max_nodes[j])
                c2 = edge_count(int_edge, ext_edge, [subgraphs[j][k]], [
                                {'edge_graph': subsubgraphs[j][k][0]['edge_graph']}], subgraph_max_nodes[j])
                e_c = torch.tensor([[c1, c2]]).to(device)
                count_edge = torch.cat((count_edge, e_c), 0)
                c3 = node_count(int_node, ext_node, [subgraphs[j][k]], [
                                {'node_graph': subsubgraphs[j][k][0]['node_graph']}], subgraph_max_nodes[j])
                c4 = node_count(int_node, ext_node, [subgraphs[j][k]], [
                                {'edge_graph': subsubgraphs[j][k][0]['edge_graph']}], subgraph_max_nodes[j])
                n_c = torch.tensor([[c3, c4]]).to(device)
                count_node = torch.cat((count_node, n_c), 0)
        new_inp = torch.cat((count_edge, count_node), dim=1).to(device)
        new_train_data = TensorDataset(new_inp, graphs[0].ext_label)
        new_train_loader = DataLoader(
            new_train_data, batch_size=1, shuffle=False)
        for batch in new_train_loader:
            new_inp = batch[0].to(device)
            label = batch[1].to(device)
            pred = predictor(new_inp)
            pred = new_round(pred, 0.5)
            pred = pred.to(device)
            pred = pred.float()
            loss = loss_fn1(pred.squeeze(), label)
            total_loss_int += loss.item()
        step += 1
    return total_loss_int / step


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
    elif args.dataset =='qm7b':
        dataset = QM7b(root='/hdfs1/Data/Shubhajit/Sub-Structure-GNN/data/QM7b', pre_transform=None)
        train_dataset = dataset[:int(len(dataset)*0.8)]
        val_dataset = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
        test_dataset = dataset[int(len(dataset)*0.9):]

    collater_fn = frag_collater_tailed_triangle()
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collater_fn)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collater_fn)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collater_fn)

    int_node = localGNN_tailed_triangle(args.num_layers, args.hidden_dim).to(device)
    ext_node = new_external(2, 32).to(device)

    int_node.load_state_dict(torch.load(
        'save/local_nodes_dataset_zinc_subset_insig/Int_GNN_local_nodes_zinc_subset.pt'))
    ext_node.load_state_dict(torch.load(
        'save/local_nodes_dataset_zinc_subset_insig/Ext_GNN_local_nodes_zinc_subset.pt'))

    int_edge = localGNN_tailed_triangle(args.num_layers, args.hidden_dim).to(device)
    ext_edge = new_external(2, 32).to(device)

    int_edge.load_state_dict(torch.load(
        'save/local_edges_dataset_zinc_subset_insig/Int_GNN_local_edges_zinc_subset.pt'))
    ext_edge.load_state_dict(torch.load(
        'save/local_edges_dataset_zinc_subset_insig/Ext_GNN_local_edges_zinc_subset.pt'))

    predict_model = predictor_tailed_triangle(4, 64).to(device)

    pred_opt = torch.optim.Adam(predict_model.parameters(), lr=args.lr)
    loss_fn1 = torch.nn.L1Loss(reduction='mean')

    train_curve = []
    valid_curve = []
    test_curve = []
    best_val_loss = 1000
    best_test_loss = 1000

    train_labels = torch.tensor([]).to(device)
    test_labels = torch.tensor([]).to(device)
    val_labels = torch.tensor([]).to(device)
    for batch in tqdm(train_loader):
        for graph in batch[0]:
            #print(graph.ext_label.shape, graph.ext_label)
            train_labels = torch.cat(
            (train_labels, torch.sum(graph.ext_label).reshape(1).to(device)), 0)
    for batch in tqdm(val_loader):
        for graph in batch[0]:
            val_labels = torch.cat(
            (val_labels, torch.sum(graph.ext_label).reshape(1).to(device)), 0)
    for batch in tqdm(test_loader):
        for graph in batch[0]:
            test_labels = torch.cat(
            (test_labels, torch.sum(graph.ext_label).reshape(1).to(device)), 0)
    train_variance = torch.std(train_labels)**2
    val_variance = torch.std(val_labels)**2
    test_variance = torch.std(test_labels)**2
    if  train_variance == 0:
        train_variance = torch.tensor(1)
    if  val_variance == 0:
        val_variance = torch.tensor(1)
    if  test_variance == 0:
        test_variance = torch.tensor(1)

    print("Train Variance: ", train_variance.item())
    print("Val Variance: ", val_variance.item())
    print("Test Variance: ", test_variance.item())
    print("Number of parameters in predictor: ",
          count_parameters(predict_model))

    for epoch in range(1, args.epochs+1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_loss = train(args, int_node, ext_node, int_edge,
                           ext_edge, predict_model, train_loader, pred_opt, loss_fn1)
        print('Train loss : {}'.format(train_loss/train_variance))

        print('Evaluating...')
        valid_loss = eval(args, int_node, ext_node, int_edge,
                          ext_edge, predict_model, val_loader, loss_fn1)
        print('Valid loss : {}'.format(valid_loss/val_variance))
        test_loss = eval(args, int_node, ext_node, int_edge,
                         ext_edge, predict_model, test_loader, loss_fn1)
        print('Test loss : {}'.format(test_loss/test_variance))
        if valid_loss <= best_val_loss:
            if valid_loss == best_val_loss:
                if test_loss < best_test_loss:
                    best_val_loss = valid_loss
                    best_test_loss = test_loss
                    print("Best Model!")
                    print("Best valid loss : {}".format(best_val_loss/val_variance))
                    print("Best test loss : {}".format(
                        best_test_loss/test_variance))
                    # save the best model
                    torch.save(predict_model.state_dict(
                    ), "save/{}/predict_model_{}.pt".format(args.output_file, args.dataset))
            else:
                best_val_loss = valid_loss
                best_test_loss = test_loss
                print("Best Model!")
                print("Best valid loss : {}".format(best_val_loss/val_variance))
                print("Best test loss : {}".format(best_test_loss/test_variance))
                # save the best model
                torch.save(predict_model.state_dict(
                ), "save/{}/predict_model_{}.pt".format(args.output_file, args.dataset))
        train_curve.append(train_loss/train_variance)
        valid_curve.append(valid_loss/val_variance)
        test_curve.append(test_loss/test_variance)

    print("Best valid loss : {}".format(best_val_loss/val_variance))
    print("Best test loss : {}".format(best_test_loss/test_variance))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sub-Structure GNN')
    parser.add_argument('--dataset', type=str, default='dataset_2',
                        help='Dataset: dataset_1 or dataset_2')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--output_file', type=str,
                        default='triangle_2', help='Output file name')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--step', type=int, default=500)
    args = parser.parse_args()
    main(args)
