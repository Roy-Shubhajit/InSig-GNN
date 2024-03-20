import torch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from scipy.special import comb
from tqdm import tqdm
import pandas as pd
import json
from torch_geometric.utils import degree, is_undirected, remove_self_loops, contains_self_loops, to_undirected

def calculate_dataset_variance(dataset_name, train, val, test):
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

    for i in tqdm(range(len(train))):
        train_triangle = torch.cat((train_triangle, train[i].triangle.view(1)))
        train_3star = torch.cat((train_3star, train[i].star.view(1)))
        train_chordal = torch.cat((train_chordal, train[i].chordal_cycle.view(1)))
        train_2star = torch.cat((train_2star, train[i].star_2.view(1)))
        train_k4 = torch.cat((train_k4, train[i].K4.view(1)))
        train_C4 = torch.cat((train_C4, train[i].C4.view(1)))
        train_tailed_triangle = torch.cat((train_tailed_triangle, train[i].tailed_triangle.view(1)))

    for i in tqdm(range(len(val))):
        val_triangle = torch.cat((val_triangle, val[i].triangle.view(1)))
        val_3star = torch.cat((val_3star, val[i].star.view(1)))
        val_chordal = torch.cat((val_chordal, val[i].chordal_cycle.view(1)))
        val_2star = torch.cat((val_2star, val[i].star_2.view(1)))
        val_k4 = torch.cat((val_k4, val[i].K4.view(1)))
        val_C4 = torch.cat((val_C4, val[i].C4.view(1)))
        val_tailed_triangle = torch.cat((val_tailed_triangle, val[i].tailed_triangle.view(1)))

    for i in tqdm(range(len(test))):
        test_triangle = torch.cat((test_triangle, test[i].triangle.view(1)))
        test_3star = torch.cat((test_3star, test[i].star.view(1)))
        test_chordal = torch.cat((test_chordal, test[i].chordal_cycle.view(1)))
        test_2star = torch.cat((test_2star, test[i].star_2.view(1)))
        test_k4 = torch.cat((test_k4, test[i].K4.view(1)))
        test_C4 = torch.cat((test_C4, test[i].C4.view(1)))
        test_tailed_triangle = torch.cat((test_tailed_triangle, test[i].tailed_triangle.view(1)))

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

    try:
        with open(f'data/{dataset_name}_variance.json', 'r') as f:
            data = json.load(f)
    except:
        data = {}
        data["Training"] = {}
        data["Validation"] = {}
        data["Test"] = {}
        data["Training"]["triangle"] = train_triangle_var.item()
        data["Training"]["3star"] = train_3star_var.item()
        data["Training"]["chordal"] = train_chordal_var.item()
        data["Training"]["2star"] = train_2star_var.item()
        data["Training"]["K4"] = train_k4_var.item()
        data["Training"]["C4"] = train_C4_var.item()
        data["Training"]["Tailed_Triangle"] = train_tailed_triangle_var.item()
        data["Validation"]["triangle"] = val_triangle_var.item()
        data["Validation"]["3star"] = val_3star_var.item()
        data["Validation"]["chordal"] = val_chordal_var.item()
        data["Validation"]["2star"] = val_2star_var.item()
        data["Validation"]["K4"] = val_k4_var.item()
        data["Validation"]["C4"] = val_C4_var.item()
        data["Validation"]["Tailed_Triangle"] = val_tailed_triangle_var.item()
        data["Test"]["triangle"] = test_triangle_var.item()
        data["Test"]["3star"] = test_3star_var.item()
        data["Test"]["chordal"] = test_chordal_var.item()
        data["Test"]["2star"] = test_2star_var.item()
        data["Test"]["K4"] = test_k4_var.item()
        data["Test"]["C4"] = test_C4_var.item()
        data["Test"]["Tailed_Triangle"] = test_tailed_triangle_var.item()
        with open(f'data/{dataset_name}_variance.json', 'w') as f:
            json.dump(data, f, indent=4)

class collater():
    def __init__(self, task):
        self.task = task
        pass

    def create_subgraphs(self, data):

        edge_index, num_nodes = data.edge_index, data.num_nodes
        if is_undirected(edge_index) == False:
            edge_index = to_undirected(edge_index)
        if contains_self_loops(edge_index):
            edge_index = remove_self_loops(edge_index)[0]
        #num_edges = 0
        subgraphs = {}
        total_edge_index = torch.tensor([], dtype=torch.long)
        l = torch.tensor([], dtype=torch.long)
        k = torch.tensor([], dtype=torch.long)
        for ind in range(num_nodes):
            if self.task == "triangle":
                nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                    ind, 1, edge_index, False, num_nodes)
                edge_attr_ = None
                edge_index_ = edge_index_.T
                new_edge_index = torch.tensor([], dtype=torch.long)
                for edge in edge_index_:
                    if edge[0] != ind and edge[1] != ind:
                        new_edge_index = torch.cat(
                            (new_edge_index, edge.unsqueeze(0)), dim=0)
                edge_index_ = new_edge_index.T
                if len(edge_index_.shape) == 1:
                    edge_index_ = edge_index_.reshape(2, edge_index_.shape[0])
                total_edge_index = torch.cat(
                    (total_edge_index, edge_index_.T), dim=0)
                data_ = Data(edge_index=edge_index_, z=z_)
                l = torch.cat(
                    (l, torch.tensor([edge_index_.shape[1]//2])), dim=0)
                k = torch.cat(
                    (k, torch.tensor([edge_index_.shape[1]//2])), dim=0)
                #num_edges += edge_index_.shape[1]

            elif self.task == "3star":
                nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                    ind, 1, edge_index, False, num_nodes)
                edge_attr_ = None
                edge_index_ = edge_index_.T
                new_edge_index = torch.tensor([], dtype=torch.long)
                for edge in edge_index_:
                    if edge[0] == ind or edge[1] == ind:
                        new_edge_index = torch.cat(
                            (new_edge_index, edge.unsqueeze(0)), dim=0)
                edge_index_ = new_edge_index.T
                if len(edge_index_.shape) == 1:
                    edge_index_ = edge_index_.reshape(2, edge_index_.shape[0])
                total_edge_index = torch.cat(
                    (total_edge_index, edge_index_.T), dim=0)
                data_ = Data(edge_index=edge_index_, z=z_)
                l = torch.cat(
                    (l, torch.tensor([comb(edge_index_.shape[1]//2, 3, exact=True)])), dim=0)
                k = torch.cat(
                    (k, torch.tensor([comb(edge_index_.shape[1]//2, 3, exact=True)])), dim=0)

            elif self.task == "chordal":
                nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                    ind, 1, edge_index, False, num_nodes)
                edge_attr_ = None
                edge_index_ = edge_index_.T
                new_edge_index = torch.tensor([], dtype=torch.long)
                for edge in edge_index_:
                    if edge[0] != ind and edge[1] != ind:
                        new_edge_index = torch.cat(
                            (new_edge_index, edge.unsqueeze(0)), dim=0)
                edge_index_ = new_edge_index.T
                if len(edge_index_.shape) == 1:
                    edge_index_ = edge_index_.reshape(2, edge_index_.shape[0])
                total_edge_index = torch.cat(
                    (total_edge_index, edge_index_.T), dim=0)
                data_ = Data(edge_index=edge_index_, z=z_)
                nodes_ = nodes_[nodes_ != ind]
                deg = degree(edge_index_[0], num_nodes)
                ll = sum([comb(i, 2, exact=True) for i in deg[deg > 1]])
                l = torch.cat((l, torch.tensor([ll])), dim=0)
                k = torch.cat((k, torch.tensor([ll])), dim=0)

            elif self.task == "2star":
                nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                    ind, 1, edge_index, False, num_nodes)
                edge_index_ = edge_index_.T
                new_edge_index = torch.tensor([], dtype=torch.long)
                for edge in edge_index_:
                    if edge[0] == ind or edge[1] == ind:
                        new_edge_index = torch.cat(
                            (new_edge_index, edge.unsqueeze(0)), dim=0)
                edge_index_ = new_edge_index.T
                if len(edge_index_.shape) == 1:
                    edge_index_ = edge_index_.reshape(2, edge_index_.shape[0])
                data_ = Data(edge_index=edge_index_, z=z_)
                k = torch.cat(
                    (k, torch.tensor([comb(edge_index_.shape[1]//2, 2, exact=True)])), dim=0)
                l = torch.cat(
                    (l, torch.tensor([comb(edge_index_.shape[1]//2, 2, exact=True)])), dim=0)

            elif self.task == "local_nodes":
                nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                    ind, 1, edge_index, False, num_nodes)
                edge_index_ = edge_index_.T
                new_edge_index = torch.tensor([], dtype=torch.long)
                for edge in edge_index_:
                    if edge[0] == ind or edge[1] == ind:
                        new_edge_index = torch.cat(
                            (new_edge_index, edge.unsqueeze(0)), dim=0)
                edge_index_ = new_edge_index.T
                if len(edge_index_.shape) == 1:
                    edge_index_ = edge_index_.reshape(2, edge_index_.shape[0])
                data_ = Data(edge_index=edge_index_, z=z_)
                k = torch.cat(
                    (k, torch.tensor([edge_index_.shape[1]//2])), dim=0)
                l = torch.cat(
                    (l, torch.tensor([edge_index_.shape[1]//2])), dim=0)

            elif self.task == "local_edges":
                nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                    ind, 1, edge_index, False, num_nodes)
                edge_index_ = edge_index_.T
                new_edge_index = torch.tensor([], dtype=torch.long)
                for edge in edge_index_:
                    if edge[0] != ind and edge[1] != ind:
                        new_edge_index = torch.cat(
                            (new_edge_index, edge.unsqueeze(0)), dim=0)
                edge_index_ = new_edge_index.T
                if len(edge_index_.shape) == 1:
                    edge_index_ = edge_index_.reshape(2, edge_index_.shape[0])
                data_ = Data(edge_index=edge_index_, z=z_)
                k = torch.cat(
                    (k, torch.tensor([edge_index_.shape[1]//2])), dim=0)
                l = torch.cat(
                    (l, torch.tensor([edge_index_.shape[1]//2])), dim=0)

            subgraphs[ind] = data_

        total_edge_index = torch.unique(total_edge_index, dim=0)

        if self.task == "triangle":
            new_data = Data(edge_index=total_edge_index.T)
            #new_data.ext_label_dataset = data.triangle
            #new_data.ext_label = torch.tensor([num_edges//6])
            new_data.ext_label = torch.sum(k)//3
        elif self.task == "3star":
            new_data = Data(edge_index=data.edge_index)
            #new_data.ext_label_dataset = data.star
            new_data.ext_label = torch.sum(k)
        elif self.task == "chordal":
            new_data = Data(edge_index=edge_index)
            #new_data.ext_label_dataset = data.chordal_cycle
            new_data.ext_label = torch.sum(k)//2
        elif self.task == "2star":
            new_data = Data(edge_index=data.edge_index)
            #new_data.ext_label_dataset = data.star_2
            new_data.ext_label = torch.sum(k)
        elif self.task == "local_nodes":
            new_data = Data(edge_index=data.edge_index)
            # new_data.ext_label_dataset = data.local_nodes
            new_data.ext_label = torch.sum(k)
        elif self.task == "local_edges":
            new_data = Data(edge_index=data.edge_index)
            # new_data.ext_label_dataset = data.local_nodes
            new_data.ext_label = torch.sum(k)
        return new_data, subgraphs, l

    def __call__(self, data):
        graphs = []
        subgraphs = []
        max_nodes = max([d.num_nodes for d in data])
        labels = torch.zeros((len(data), max_nodes), dtype=torch.long)
        for g in data:
            G_, sub_G, internal_labels = self.create_subgraphs(g)
            graphs.append(G_)
            subgraphs.append(sub_G)
            labels[len(graphs)-1, :g.num_nodes] = internal_labels

        return [graphs, subgraphs, labels, max_nodes]


class frag_collater_K4():
    def __init__(self):
        pass
    # for K4 counting

    def create_subsubgraphs_triangle(self, data):
        edge_index, num_nodes = data.edge_index, data.num_nodes
        if num_nodes > 0:
            node_name = torch.unique(edge_index[0])
        else:
            return {}, torch.tensor([0])
        if is_undirected(edge_index) == False:
            edge_index = to_undirected(edge_index)
        if contains_self_loops(edge_index):
            edge_index = remove_self_loops(edge_index)[0]
        #num_edges = 0
        subgraphs = {}
        k = torch.tensor([], dtype=torch.long)
        for ind in node_name:
            nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                ind.item(), 1, edge_index, False, num_nodes)
            edge_attr_ = None
            edge_index_ = edge_index_.T
            new_edge_index = torch.tensor([], dtype=torch.long)
            for edge in edge_index_:
                if edge[0] != ind and edge[1] != ind:
                    new_edge_index = torch.cat(
                        (new_edge_index, edge.unsqueeze(0)), dim=0)
            edge_index_ = new_edge_index.T
            if len(edge_index_.shape) == 1:
                edge_index_ = edge_index_.reshape(2, edge_index_.shape[0])
            data_ = Data(edge_index=edge_index_, z=z_)
            k = torch.cat(
                    (k, torch.tensor([edge_index_.shape[1]//2])), dim=0)
            #num_edges += edge_index_.shape[1]
            subgraphs[ind.item()] = data_
        return subgraphs, torch.sum(k)//3

    def create_subgraphs(self, data):

        edge_index, num_nodes = data.edge_index, data.num_nodes
        subsubgraphs = {}
        if is_undirected(edge_index) == False:
            edge_index = to_undirected(edge_index)
        if contains_self_loops(edge_index):
            edge_index = remove_self_loops(edge_index)[0]
        #num_edges = 0
        subgraphs = {}
        total_edge_index = torch.tensor([], dtype=torch.long)
        l = torch.tensor([], dtype=torch.long)
        for ind in range(num_nodes):
            nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                ind, 1, edge_index, False, num_nodes)
            edge_attr_ = None
            edge_index_ = edge_index_.T
            edge_list = edge_index_.tolist()
            mask = (edge_index_ != ind).all(dim=1)
            edge_index_ = edge_index_[mask].T
            total_edge_index = torch.cat(
                (total_edge_index, edge_index_.T), dim=0)
            s = []
            data_1 = Data(edge_index=edge_index_, z=z_)
            s1, l1 = self.create_subsubgraphs_triangle(data_1)
            s.append(s1)
            subsubgraphs[ind] = s
            subgraphs[ind] = data_1
            #print(l1)
            l = torch.cat((l, l1.reshape(1)), dim=0)

        total_edge_index = torch.unique(total_edge_index, dim=0)

        new_data = Data(edge_index=total_edge_index.T)
        new_data.ext_label = torch.sum(l)//4
        return new_data, subgraphs, l, subsubgraphs, max([d.num_nodes for d in subgraphs.values()])

    def __call__(self, data):
        graphs = []
        subgraphs = []
        subsubgraphs = []
        max_nodes = max([d.num_nodes for d in data])
        subgraph_max_nodes = []
        labels = torch.zeros((len(data), max_nodes), dtype=torch.long)
        for g in data:
            G_, sub_G, internal_labels, sub_sub_G, max_nodes_subgraph = self.create_subgraphs(
                g)
            graphs.append(G_)
            subgraphs.append(sub_G)
            subsubgraphs.append(sub_sub_G)
            labels[len(graphs)-1, :g.num_nodes] = internal_labels
            subgraph_max_nodes.append(max_nodes_subgraph)

        return [graphs, subgraphs, labels, max_nodes, subsubgraphs, subgraph_max_nodes]


class frag_collater_C4():
    def __init__(self):
        pass

    def create_subsubgraphs_2stars(self, data, node_dict):
        edge_index, num_nodes = data.edge_index, data.num_nodes
        if num_nodes > 0:
            node_name = torch.unique(edge_index[0])
        else:
            return {}, torch.tensor([0])
        if is_undirected(edge_index) == False:
            edge_index = to_undirected(edge_index)
        if contains_self_loops(edge_index):
            edge_index = remove_self_loops(edge_index)[0]
        subgraphs = {}
        k = torch.tensor([], dtype=torch.long)
        for ind in node_name:
            nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                ind.item(), 1, edge_index, False, num_nodes)
            edge_attr_ = None
            edge_index_ = edge_index_.T
            blablabla = torch.tensor([], dtype=torch.long)
            for edge in edge_index_:
                if edge[0] == ind and node_dict[edge[1].item()] == 1:
                    blablabla = torch.cat(
                        (blablabla, edge.unsqueeze(0)), dim=0)
                elif edge[1] == ind and node_dict[edge[0].item()] == 1:
                    blablabla = torch.cat(
                        (blablabla, edge.unsqueeze(0)), dim=0)
            if blablabla.shape[0] != 0:
                data_ = Data(edge_index=blablabla.T, z=z_)
                k = torch.cat(
                    (k, torch.tensor([comb(blablabla.T.shape[1]//2, 2, exact=True)])), dim=0)
                subgraphs[ind.item()] = data_
            else:
                continue
        return subgraphs, torch.sum(k)

    def create_subgraphs(self, data):

        edge_index, num_nodes = data.edge_index, data.num_nodes
        subsubgraphs = {}
        if is_undirected(edge_index) == False:
            edge_index = to_undirected(edge_index)
        if contains_self_loops(edge_index):
            edge_index = remove_self_loops(edge_index)[0]
        num_edges = 0
        subgraphs = {}
        total_edge_index = torch.tensor([], dtype=torch.long)
        l = torch.zeros(num_nodes, dtype=torch.long)
        for ind in range(num_nodes):
            nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                ind, 2, edge_index, False, num_nodes)
            edge_attr_ = None
            edge_index_ = edge_index_.T
            edge_list = edge_index_.tolist()
            node_list = torch.unique(edge_index_.T[0])
            node_list = node_list.tolist()
            node_dict = {n: 1 if [ind, n] in edge_list or [
                n, ind] in edge_list else 2 for n in node_list}
            new_edge_index = torch.tensor([], dtype=torch.long)
            for edge in edge_index_:
                if edge[0] != ind and edge[1] != ind:
                    new_edge_index = torch.cat(
                        (new_edge_index, edge.unsqueeze(0)), dim=0)
            edge_index_ = new_edge_index.T
            if len(edge_index_.shape) == 1:
                edge_index_ = edge_index_.reshape(2, edge_index_.shape[0])
            total_edge_index = torch.cat(
                (total_edge_index, edge_index_.T), dim=0)
            s = []
            data_1 = Data(edge_index=edge_index_, z=z_)
            s1, l1 = self.create_subsubgraphs_2stars(data_1, node_dict)
            s.append(s1)
            subsubgraphs[ind] = s
            subgraphs[ind] = data_1
            l[ind] = l1

        total_edge_index = torch.unique(total_edge_index, dim=0)

        new_data = Data(edge_index=total_edge_index.T)
        new_data.ext_label = torch.sum(l)//4
        return new_data, subgraphs, l, subsubgraphs, max([d.num_nodes for d in subgraphs.values()])

    def __call__(self, data):
        graphs = []
        subgraphs = []
        subsubgraphs = []
        max_nodes = max([d.num_nodes for d in data])
        subgraph_max_nodes = []
        labels = torch.zeros((len(data), max_nodes), dtype=torch.long)
        for g in data:
            G_, sub_G, internal_labels, sub_sub_G, max_nodes_subgraph = self.create_subgraphs(
                g)
            graphs.append(G_)
            subgraphs.append(sub_G)
            subsubgraphs.append(sub_sub_G)
            labels[len(graphs)-1, :g.num_nodes] = internal_labels
            subgraph_max_nodes.append(max_nodes_subgraph)
        return [graphs, subgraphs, labels, max_nodes, subsubgraphs, subgraph_max_nodes]


class frag_collater_tailed_triangle():
    def __init__(self):
        pass

    def create_subsubgraphs(self, data, root):
        edge_index, num_nodes = data.edge_index, data.num_nodes
        subgraphs = {}
        if is_undirected(edge_index) == False:
            edge_index = to_undirected(edge_index)
        if contains_self_loops(edge_index):
            edge_index = remove_self_loops(edge_index)[0]
        edge_index_ = edge_index.T
        node_graph_edge = torch.tensor([], dtype=torch.long)
        edge_graph_edge = torch.tensor([], dtype=torch.long)
        for edge in edge_index_:
            if edge[0] != root and edge[1] != root:
                edge_graph_edge = torch.cat(
                    (edge_graph_edge, edge.unsqueeze(0)), dim=0)
            else:
                node_graph_edge = torch.cat(
                    (node_graph_edge, edge.unsqueeze(0)), dim=0)
        edge_graph_edge = edge_graph_edge.T
        node_graph_edge = node_graph_edge.T
        if len(edge_graph_edge.shape) == 1:
            edge_graph_edge = edge_graph_edge.reshape(
                2, edge_graph_edge.shape[0])
        if len(node_graph_edge.shape) == 1:
            node_graph_edge = node_graph_edge.reshape(
                2, node_graph_edge.shape[0])
        subgraphs['node_graph'] = Data(edge_index=node_graph_edge)
        subgraphs['edge_graph'] = Data(edge_index=edge_graph_edge)
        num_node_edges = node_graph_edge.shape[1]//2
        num_edge_edges = edge_graph_edge.shape[1]//2
        return subgraphs, torch.tensor(num_edge_edges*(num_node_edges-2))

    def create_subgraphs(self, data):

        edge_index, num_nodes = data.edge_index, data.num_nodes
        subsubgraphs = {}
        if is_undirected(edge_index) == False:
            edge_index = to_undirected(edge_index)
        if contains_self_loops(edge_index):
            edge_index = remove_self_loops(edge_index)[0]
        num_edges = 0
        subgraphs = {}
        total_edge_index = torch.tensor([], dtype=torch.long)
        l = torch.zeros(num_nodes, dtype=torch.long)
        for ind in range(num_nodes):
            nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                ind, 1, edge_index, False, num_nodes)
            edge_attr_ = None
            edge_index_ = edge_index_.T
            total_edge_index = torch.cat(
                (total_edge_index, edge_index_), dim=0)
            s = []
            data_1 = Data(edge_index=edge_index_.T, z=z_)
            s1, l1 = self.create_subsubgraphs(data_1, ind)
            s.append(s1)
            subsubgraphs[ind] = s
            subgraphs[ind] = data_1
            l[ind] = l1

        total_edge_index = torch.unique(total_edge_index, dim=0)

        new_data = Data(edge_index=total_edge_index.T)
        new_data.ext_label = l
        #new_data.ext_label_dataset = data.tailed_triangle
        return new_data, subgraphs, l, subsubgraphs, max([d.num_nodes for d in subgraphs.values()])

    def __call__(self, data):
        graphs = []
        subgraphs = []
        subsubgraphs = []
        max_nodes = max([d.num_nodes for d in data])
        subgraph_max_nodes = []
        labels = torch.zeros((len(data), max_nodes), dtype=torch.long)
        for g in data:
            G_, sub_G, internal_labels, sub_sub_G, max_nodes_subgraph = self.create_subgraphs(
                g)
            graphs.append(G_)
            subgraphs.append(sub_G)
            subsubgraphs.append(sub_sub_G)
            labels[len(graphs)-1, :g.num_nodes] = internal_labels
            subgraph_max_nodes.append(max_nodes_subgraph)

        return [graphs, subgraphs, labels, max_nodes, subsubgraphs, subgraph_max_nodes]


def new_round(x, th):
    rounded_values = torch.where(
        x - x.floor() + 0.000001 >= th, x.ceil(), x.floor())
    return rounded_values.int()
