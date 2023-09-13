import torch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from scipy.special import comb
from torch_geometric.utils import degree


class collater():
    def __init__(self, task):
        self.task = task
        pass

    def create_subgraphs(self, data):

        edge_index, num_nodes = data.edge_index, data.num_nodes

        num_edges = 0
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
                num_edges += edge_index_.shape[1]

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
            new_data.ext_label_dataset = data.triangle
            new_data.ext_label = torch.tensor([num_edges//6])
        elif self.task == "3star":
            new_data = Data(edge_index=data.edge_index)
            new_data.ext_label_dataset = data.star
            new_data.ext_label = torch.sum(k)
        elif self.task == "chordal":
            new_data = Data(edge_index=total_edge_index.T)
            new_data.ext_label_dataset = data.chordal_cycle
            new_data.ext_label = torch.sum(k)//2
        elif self.task == "2star":
            new_data = Data(edge_index=data.edge_index)
            new_data.ext_label_dataset = data.star_2
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
        num_edges = 0
        subgraphs = {}
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
            num_edges += edge_index_.shape[1]
            subgraphs[ind.item()] = data_
        return subgraphs, torch.tensor([num_edges//6])

    def create_subgraphs(self, data):

        edge_index, num_nodes = data.edge_index, data.num_nodes
        subsubgraphs = {}
        num_edges = 0
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
            l = torch.cat((l, l1), dim=0)

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
        new_data.ext_label = torch.sum(l)
        new_data.ext_label_dataset = data.tailed_triangle
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
