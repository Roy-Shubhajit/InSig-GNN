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
                mask = (edge_index_ != ind).all(dim=1)
                edge_index_ = edge_index_[mask].T
                total_edge_index = torch.cat((total_edge_index, edge_index_.T), dim=0)
                data_ = Data(edge_index=edge_index_, z=z_)
                l = torch.cat((l,torch.tensor([edge_index_.shape[1]//2])), dim=0)
                num_edges += edge_index_.shape[1]
                
            elif self.task == "3star":
                nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                ind, 1, edge_index, False, num_nodes)
                edge_attr_ = None
                edge_index_ = edge_index_.T

                mask = (edge_index_ == ind).all(dim=1)
                edge_index_ = edge_index_[mask].T
                total_edge_index = torch.cat((total_edge_index, edge_index_.T), dim=0)
                data_ = Data(edge_index=edge_index_, z=z_)
                #l = torch.cat((l, torch.tensor([nodes_.shape[0]-1])), dim=0)
                l = torch.cat((l,torch.tensor([edge_index_.shape[1]//2])), dim=0)
                #k = torch.cat((k,torch.tensor([comb(nodes_.shape[0]-1,3, exact=True)])), dim=0)
                k = torch.cat((k,torch.tensor([comb(edge_index_.shape[1]//2,3, exact=True)])), dim=0)
            
            elif self.task == "C4":
                node_dict = {}
                nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                ind, 2, edge_index, False, num_nodes)
                edge_index_ = edge_index_.T
                nodes_ = nodes_[nodes_ != ind]
                edge_list = edge_index_.tolist()
                node_dict = {n.item(): 1 if [ind, n] in edge_list or [n, ind] in edge_list else 2 for n in nodes_}
                mask = (edge_index_ != ind).all(dim=1)
                edge_index_ = edge_index_[mask].T
                total_edge_index = torch.cat((total_edge_index, edge_index_.T), dim=0)
                x = torch.ones([num_nodes, 1]).to(edge_index.device)
                for n in node_dict:
                    nei_num = 0
                    for m in node_dict:
                        if node_dict[m] == 1 and [n, m] in edge_list and [m, n] in edge_list:
                            nei_num += 1
                    x[n] = torch.tensor([nei_num])
                
                data_ = Data(edge_index=edge_index_, z=z_)
                ll = 0
                edge_list = edge_index_.T.tolist()
                total_nei = 0
                for n in node_dict:
                    nei = sum([1 for m in node_dict if [n, m] in edge_list and [m, n] in edge_list and node_dict[m] == 1])
                    if nei >= 2:
                        total_nei = total_nei + nei
                        ll += comb(nei, 2, exact=True)
                        
                l = torch.cat((l, torch.tensor([ll])), dim=0)
                k = torch.cat((k, torch.tensor([ll])), dim=0)

            elif self.task == "chordal":
                nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                ind, 1, edge_index, False, num_nodes)
                edge_attr_ = None
                edge_index_ = edge_index_.T
                mask = (edge_index_ != ind).all(dim=1)
                edge_index_ = edge_index_[mask].T
                total_edge_index = torch.cat((total_edge_index, edge_index_.T), dim=0)
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
                mask = (edge_index_ == ind).all(dim=1)
                edge_index_ = edge_index_[mask].T
                data_ = Data(edge_index=edge_index_, z=z_)
                #k = torch.cat((k,torch.tensor([comb(nodes_.shape[0]-1,2, exact=True)])), dim=0)
                #l = torch.cat((l, torch.tensor([nodes_.shape[0]-1])), dim=0)
                k = torch.cat((k,torch.tensor([comb(edge_index_.shape[1]//2,2, exact=True)])), dim=0)
                l = torch.cat((l,torch.tensor([edge_index_.shape[1]//2])), dim=0)
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
        elif self.task == "C4":
            new_data = Data(edge_index=total_edge_index.T)
            new_data.ext_label_dataset = data.C4
            new_data.ext_label = torch.ceil(torch.sum(k)/4)
        elif self.task == "chordal":
            new_data = Data(edge_index=total_edge_index.T)
            new_data.ext_label_dataset = data.chordal_cycle
            new_data.ext_label = torch.sum(k)//2
        elif self.task == "2star":
            new_data = Data(edge_index=data.edge_index)
            new_data.ext_label_dataset = data.star_2
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
    
class frag_collater():
    def __init__(self, task):
        self.task = task
        pass

    def create_subsubgraphs_2stars(self, data):
        edge_index, num_nodes = data.edge_index, data.num_nodes   
        subgraphs = {}
        l = torch.tensor([], dtype=torch.long)
        k = torch.tensor([], dtype=torch.long)
        for ind in range(num_nodes):
            nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
            ind, 1, edge_index, False, num_nodes)
            edge_index_ = edge_index_.T
            mask = (edge_index_ == ind).all(dim=1)
            edge_index_ = edge_index_[mask].T
            data_ = Data(edge_index=edge_index_, z=z_)
            k = torch.cat((k,torch.tensor([comb(nodes_.shape[0]-1,2, exact=True)])), dim=0)
            l = torch.cat((l, torch.tensor([nodes_.shape[0]-1])), dim=0)
            subgraphs[ind] = data_
        if torch.sum(l).shape == torch.Size([]):
            return subgraphs, torch.tensor([0])
        return subgraphs, torch.sum(l)

    #for K4 counting
    def create_subsubgraphs_triangle(self, data):
        edge_index, num_nodes = data.edge_index, data.num_nodes
        if num_nodes > 0:
            node_name = torch.unique(edge_index[0])
        else:
            return {}, torch.tensor([0])
        
        num_edges = 0
        subgraphs = {}
        l = torch.tensor([], dtype=torch.long)
        for ind in node_name:
            nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
            ind.item(), 1, edge_index, False, num_nodes)
            edge_attr_ = None
            edge_index_ = edge_index_.T
            mask = (edge_index_ != ind).all(dim=1)
            edge_index_ = edge_index_[mask].T
            data_ = Data(edge_index=edge_index_, z=z_)
            l = torch.cat((l,torch.tensor([comb(nodes_.shape[0]-1,2, exact=True)])), dim=0)
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
        k = torch.tensor([], dtype=torch.long)
        for ind in range(num_nodes):
            nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
            ind, 1, edge_index, False, num_nodes)
            edge_attr_ = None
            edge_index_ = edge_index_.T
            edge_list = edge_index_.tolist()
            mask = (edge_index_ != ind).all(dim=1)
            edge_index_ = edge_index_[mask].T
            total_edge_index = torch.cat((total_edge_index, edge_index_.T), dim=0)
            s = []
            #x = torch.tensor([1 if [i,ind] in edge_list and [ind, i] in edge_list else 2 for i in nodes_])
            #data_1 = Data(x=x, edge_index=edge_index_, z=z_)
            data_1 = Data(edge_index=edge_index_, z=z_)
            if self.task == 'K4':
                s1, l1 = self.create_subsubgraphs_triangle(data_1)
            elif self.task == 'chordal':
                s1, l1 = self.create_subsubgraphs_2stars(data_1)
            s.append(s1)
            subsubgraphs[ind] = s
            subgraphs[ind] = data_1
            l = torch.cat((l,l1), dim=0)
            
        total_edge_index = torch.unique(total_edge_index, dim=0)

        new_data = Data(edge_index=total_edge_index.T)
        if self.task == 'K4':
            new_data.ext_label = torch.sum(l)//4
        elif self.task == 'chordal':
            new_data.ext_label = torch.sum(l)//2
            new_data.ext_label_dataset = data.chordal_cycle
        return new_data, subgraphs, l, subsubgraphs, max([d.num_nodes for d in subgraphs.values()])

    def __call__(self, data):
        graphs = []
        subgraphs = []
        subsubgraphs = []
        max_nodes = max([d.num_nodes for d in data])
        subgraph_max_nodes = []
        labels = torch.zeros((len(data), max_nodes), dtype=torch.long)
        for g in data:
            G_, sub_G, internal_labels, sub_sub_G, max_nodes_subgraph= self.create_subgraphs(g)
            graphs.append(G_)
            subgraphs.append(sub_G)
            subsubgraphs.append(sub_sub_G)
            labels[len(graphs)-1, :g.num_nodes] = internal_labels
            subgraph_max_nodes.append(max_nodes_subgraph)

        return [graphs, subgraphs, labels, max_nodes, subsubgraphs, subgraph_max_nodes]
    
class frag_collater_K4():
    def __init__(self):
        pass

    #for K4 counting
    def create_subsubgraphs_triangle(self, data):
        edge_index, num_nodes = data.edge_index, data.num_nodes
        if num_nodes > 0:
            node_name = torch.unique(edge_index[0])
        else:
            return {}, torch.tensor([0])
        
        num_edges = 0
        subgraphs = {}
        l = torch.tensor([], dtype=torch.long)
        for ind in node_name:
            nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
            ind.item(), 1, edge_index, False, num_nodes)
            edge_attr_ = None
            edge_index_ = edge_index_.T
            mask = (edge_index_ != ind).all(dim=1)
            edge_index_ = edge_index_[mask].T
            data_ = Data(edge_index=edge_index_, z=z_)
            l = torch.cat((l,torch.tensor([comb(nodes_.shape[0]-1,2, exact=True)])), dim=0)
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
        k = torch.tensor([], dtype=torch.long)
        for ind in range(num_nodes):
            nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
            ind, 1, edge_index, False, num_nodes)
            edge_attr_ = None
            edge_index_ = edge_index_.T
            edge_list = edge_index_.tolist()
            mask = (edge_index_ != ind).all(dim=1)
            edge_index_ = edge_index_[mask].T
            total_edge_index = torch.cat((total_edge_index, edge_index_.T), dim=0)
            s = []
            #x = torch.tensor([1 if [i,ind] in edge_list and [ind, i] in edge_list else 2 for i in nodes_])
            #data_1 = Data(x=x, edge_index=edge_index_, z=z_)
            data_1 = Data(edge_index=edge_index_, z=z_)
            s1, l1 = self.create_subsubgraphs_triangle(data_1)
            s.append(s1)
            subsubgraphs[ind] = s
            subgraphs[ind] = data_1
            l = torch.cat((l,l1), dim=0)
            
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
            G_, sub_G, internal_labels, sub_sub_G, max_nodes_subgraph = self.create_subgraphs(g)
            graphs.append(G_)
            subgraphs.append(sub_G)
            subsubgraphs.append(sub_sub_G)
            labels[len(graphs)-1, :g.num_nodes] = internal_labels
            subgraph_max_nodes.append(max_nodes_subgraph)

        return [graphs, subgraphs, labels, max_nodes, subsubgraphs, subgraph_max_nodes]
    
def new_round(x, th):
    rounded_values = torch.where(x - x.floor() + 0.000001 >= th, x.ceil(), x.floor())
    return rounded_values.int()