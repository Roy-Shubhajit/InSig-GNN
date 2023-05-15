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
                l = torch.cat((l, torch.tensor([nodes_.shape[0]-1])), dim=0)
                k = torch.cat((k,torch.tensor([comb(nodes_.shape[0]-1,3, exact=True)])), dim=0)
            
            elif self.task == "C4":
                node_dict = {}
                nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                ind, 2, edge_index, False, num_nodes)
                edge_index_ = edge_index_.T
                nodes_ = nodes_[nodes_ != ind]
                edge_list = edge_index_.tolist()
                node_dict = {n.item(): 1 if [ind, n] in edge_list or [n, ind] in edge_list else 0 for n in nodes_}
                mask = (edge_index_ != ind).all(dim=1)
                edge_index_ = edge_index_[mask].T
                total_edge_index = torch.cat((total_edge_index, edge_index_.T), dim=0)
                x = torch.ones([num_nodes, 1]).to(edge_index.device)
                for n in node_dict:
                    if node_dict[n] == 0:
                        x[n] = torch.tensor([0])
                    else:
                        x[n] = torch.tensor([1])
                
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
                total_edge_index = torch.cat((total_edge_index, edge_index_.T), dim=0)
                data_ = Data(edge_index=edge_index_, z=z_)
                k = torch.cat((k,torch.tensor([comb(nodes_.shape[0]-1,2, exact=True)])), dim=0)
                l = torch.cat((l, torch.tensor([nodes_.shape[0]-1])), dim=0)
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