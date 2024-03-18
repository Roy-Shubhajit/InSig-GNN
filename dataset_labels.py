from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from scipy.special import comb
import torch
from torch_geometric.utils import degree, is_undirected, remove_self_loops, contains_self_loops, to_undirected


def count_triangle(data):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    if is_undirected(edge_index) == False:
        edge_index = to_undirected(edge_index)
    if contains_self_loops(edge_index):
        edge_index = remove_self_loops(edge_index)[0]
    l = torch.tensor([], dtype=torch.long)
    k = torch.tensor([], dtype=torch.long)
    for ind in range(num_nodes):
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
        k = torch.cat(
            (k, torch.tensor([edge_index_.shape[1]//2])), dim=0)

    return torch.sum(k)//3

def count_3star(data):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    if is_undirected(edge_index) == False:
        edge_index = to_undirected(edge_index)
    if contains_self_loops(edge_index):
        edge_index = remove_self_loops(edge_index)[0]
    k = torch.tensor([], dtype=torch.long)
    for ind in range(num_nodes):
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
        k = torch.cat(
            (k, torch.tensor([comb(edge_index_.shape[1]//2, 3, exact=True)])), dim=0)
        
    return torch.sum(k)

def count_2star(data):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    if is_undirected(edge_index) == False:
        edge_index = to_undirected(edge_index)
    if contains_self_loops(edge_index):
        edge_index = remove_self_loops(edge_index)[0]
    k = torch.tensor([], dtype=torch.long)
    for ind in range(num_nodes):
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
        k = torch.cat(
            (k, torch.tensor([comb(edge_index_.shape[1]//2, 2, exact=True)])), dim=0)
        
    return torch.sum(k)

def count_chordal(data):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    if is_undirected(edge_index) == False:
        edge_index = to_undirected(edge_index)
    if contains_self_loops(edge_index):
        edge_index = remove_self_loops(edge_index)[0]
    l = torch.tensor([], dtype=torch.long)
    for ind in range(num_nodes):
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
        nodes_ = nodes_[nodes_ != ind]
        deg = degree(edge_index_[0], num_nodes)
        ll = sum([comb(i, 2, exact=True) for i in deg[deg > 1]])
        l = torch.cat((l, torch.tensor([ll])), dim=0)
    return torch.sum(l)//2

def count_local_nodes(data):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    if is_undirected(edge_index) == False:
        edge_index = to_undirected(edge_index)
    if contains_self_loops(edge_index):
        edge_index = remove_self_loops(edge_index)[0]
    k = torch.tensor([], dtype=torch.long)
    for ind in range(num_nodes):
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
        k = torch.cat(
            (k, torch.tensor([edge_index_.shape[1]//2])), dim=0)
        
    return torch.sum(k)

def count_local_edges(data):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    if is_undirected(edge_index) == False:
        edge_index = to_undirected(edge_index)
    if contains_self_loops(edge_index):
        edge_index = remove_self_loops(edge_index)[0]
    k = torch.tensor([], dtype=torch.long)
    for ind in range(num_nodes):
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
        k = torch.cat(
            (k, torch.tensor([edge_index_.shape[1]//2])), dim=0)
        
    return torch.sum(k)

def count_K4(data):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    if is_undirected(edge_index) == False:
        edge_index = to_undirected(edge_index)
    if contains_self_loops(edge_index):
        edge_index = remove_self_loops(edge_index)[0]
    l = torch.tensor([], dtype=torch.long)
    for ind in range(num_nodes):
        nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                ind, 1, edge_index, False, num_nodes)
        edge_attr_ = None
        edge_index_ = edge_index_.T
        mask = (edge_index_ != ind).all(dim=1)
        edge_index_ = edge_index_[mask].T
        data_1 = Data(edge_index=edge_index_, z=z_)
        l = torch.cat((l, count_triangle(data_1).reshape(1)), dim=0)

    return torch.sum(l)//4

def count_2stars_C4(data, node_dict):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    edge_index, num_nodes = data.edge_index, data.num_nodes
    if num_nodes > 0:
        node_name = torch.unique(edge_index[0])
    else:
        return torch.tensor(0)
    if is_undirected(edge_index) == False:
        edge_index = to_undirected(edge_index)
    if contains_self_loops(edge_index):
        edge_index = remove_self_loops(edge_index)[0]
    l = torch.tensor([], dtype=torch.long)
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
            l = torch.cat(
                (l, torch.tensor([comb(blablabla.T.shape[1]//2, 2, exact=True)])), dim=0)
        else:
            continue

    return torch.sum(l)

def count_C4(data):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    if is_undirected(edge_index) == False:
        edge_index = to_undirected(edge_index)
    if contains_self_loops(edge_index):
        edge_index = remove_self_loops(edge_index)[0]
    l = torch.tensor([], dtype=torch.long)
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
        s = []
        data_1 = Data(edge_index=edge_index_, z=z_)
        c = count_2stars_C4(data_1, node_dict)
        l = torch.cat((l, c.reshape(1)), dim=0)

    return torch.sum(l)//4

def count_local_edges_nodes(data, root):
    edge_index, num_nodes = data.edge_index, data.num_nodes
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
    num_node_edges = node_graph_edge.shape[1]//2
    num_edge_edges = edge_graph_edge.shape[1]//2

    return torch.tensor(num_edge_edges*(num_node_edges-2))

def count_tailed_triangle(data):
    edge_index, num_nodes = data.edge_index, data.num_nodes
    if is_undirected(edge_index) == False:
        edge_index = to_undirected(edge_index)
    if contains_self_loops(edge_index):
        edge_index = remove_self_loops(edge_index)[0]
    l = torch.tensor([], dtype=torch.long)
    for ind in range(num_nodes):
        nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                ind, 1, edge_index, False, num_nodes)
        edge_attr_ = None
        edge_index_ = edge_index_.T
        s = []
        data_1 = Data(edge_index=edge_index_.T, z=z_)
        l = torch.cat((l, count_local_edges_nodes(data_1, ind).reshape(1)), dim=0)

    return torch.sum(l)

def count_labels(data):

    triangle = count_triangle(data)
    star = count_3star(data)
    chordal_cycle = count_chordal(data)
    star_2 = count_2star(data)
    local_nodes = count_local_nodes(data)
    local_edges = count_local_edges(data)
    k4 = count_K4(data)
    tailed_triangle = count_tailed_triangle(data)
    c4 = count_C4(data)
    data.triangle = triangle
    data.star = star
    data.chordal_cycle = chordal_cycle
    data.star_2 = star_2
    data.local_nodes = local_nodes
    data.local_edges = local_edges
    data.K4 = k4
    data.tailed_triangle = tailed_triangle
    data.C4 = c4
    return data