
import warnings
from torch_geometric.data import Data, InMemoryDataset, download_url, Batch
from dgl.data.utils import Subset, load_graphs
from torch_geometric.utils import k_hop_subgraph
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout, ELU, GELU
from torch_geometric.nn import GINConv, GINEConv, WLConv, GATv2Conv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
warnings.filterwarnings('ignore')
import torch.utils.data



import warnings
from torch_geometric.data import Data, InMemoryDataset, download_url, Batch
from dgl.data.utils import Subset, load_graphs
from torch_geometric.utils import k_hop_subgraph
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout, ELU, GELU
from torch_geometric.nn import GINConv, GINEConv, WLConv, GATv2Conv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
warnings.filterwarnings('ignore')
import torch.utils.data


class localGNN(torch.nn.Module):
    def __init__(self, num_layers, hidden):
        super(localGNN, self).__init__()
        self.hidden = hidden
        self.conv1 = GINConv(Sequential(Linear(1, hidden), GELU(), Linear(
            hidden, hidden), GELU()), train_eps=False)
        self.conv2 = GINConv(Sequential(Linear(hidden, hidden), GELU(), Linear(
            hidden, hidden), GELU()), train_eps=False)
        self.fc0 = Linear(hidden, hidden)
        self.fc1 = Linear(hidden, 1)

    def forward(self, data_list, subgraph_dict_list, max_nodes):
        out_dims = (len(data_list), max_nodes, self.hidden)
        int_emb = torch.zeros(out_dims).to(data_list[0].edge_index.device)
        num = 0
        new_data_list = []
        for graph, sub_graphs in zip(data_list, subgraph_dict_list):
            emb = torch.tensor([]).to(data_list[0].edge_index.device)
            for key in sub_graphs.keys():
                subgraph = sub_graphs[key]
                edge_index = subgraph.edge_index
                subgraph.x = torch.ones(
                    [subgraph.num_nodes, 1]).to(edge_index.device)
                x = self.conv1(subgraph.x, edge_index)
                x = self.conv2(x, edge_index)
                x = self.fc0(x)
                x = F.gelu(x)
                if len(x)>0:
                    batch = torch.zeros(len(x), dtype=torch.long).to(data_list[0].edge_index.device)
                    x = global_add_pool(x, batch, size=1)
                else:
                    x = torch.zeros(1, self.hidden).to(data_list[0].edge_index.device)
                emb = torch.cat((emb, x), dim=0)
            #print(graph.num_nodes, emb.shape)
            if emb.shape[0] == 0:
                continue
                #int_emb[num, :graph.num_nodes, :] = torch.zeros((graph.num_nodes, self.hidden)).to(data_list[0].edge_index.device)
            else:
                int_emb[num, :emb.shape[0], :] = emb
            num += 1
            graph.x = emb  
            new_data_list.append(graph) 
        res = self.fc1(int_emb)
        batch = Batch.from_data_list(new_data_list)
        res = torch.squeeze(res, dim=-1)
        return batch, res

class globalGNN(torch.nn.Module):
    def __init__(self, num_layers, hidden):
        super(globalGNN, self).__init__()
        self.hidden = hidden
        self.conv3 = GINConv(Sequential(Linear(hidden, hidden), GELU(), Linear(
            hidden, hidden), GELU()), train_eps=False)
        self.conv4 = GINConv(Sequential(Linear(hidden, hidden), GELU(), Linear(
            hidden, hidden), GELU()), train_eps=False)
        self.fc2 = Linear(hidden, 1)

    def forward(self, data):
        
        
        x = self.conv3(data.x, data.edge_index)
        x = self.conv4(x, data.edge_index)
        #x = F.elu(x)
        x = self.fc2(x)
        x = global_add_pool(x, data.batch)
        return x
    
class new_external(torch.nn.Module):
    def __init__(self, num_layer, hidden):
        super(new_external, self).__init__()
        self.hidden = hidden
        self.fc1 = Linear(1, hidden)
        self.fc2 = Linear(hidden, 1)

    def forward(self, x):
        x = torch.sum(x, dim=1)
        x = x.reshape(-1,1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

class predictor(torch.nn.Module):
    def __init__(self, input_emb, hidden):
        super(predictor, self).__init__()
        self.hidden = hidden
        self.fc1 = Linear(input_emb, hidden)
        self.fc2 = Linear(hidden, 1)

    def forward(self, x):
        x = torch.sum(x, dim=1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
    
class localGNN_chrodal(torch.nn.Module):
    def __init__(self, num_layers, hidden):
        super(localGNN, self).__init__()
        self.hidden = hidden
        self.conv1 = GINConv(Sequential(Linear(1, hidden), GELU(), Linear(
            hidden, hidden), GELU()), train_eps=False)
        self.conv2 = GINConv(Sequential(Linear(hidden, hidden), GELU(), Linear(
            hidden, hidden), GELU()), train_eps=False)
        self.fc0 = Linear(hidden, hidden)
        self.fc1 = Linear(hidden, 1)

    def forward(self, data_list, subgraph_dict_list, max_nodes):
        out_dims = (len(data_list), max_nodes, self.hidden)
        int_emb = torch.zeros(out_dims).to(data_list[0].edge_index.device)
        num = 0
        new_data_list = []
        for graph, sub_graphs in zip(data_list, subgraph_dict_list):
            emb = torch.tensor([]).to(data_list[0].edge_index.device)
            for key in sub_graphs.keys():
                subgraph = sub_graphs[key]
                edge_index = subgraph.edge_index
                subgraph.x = torch.ones(
                    [subgraph.num_nodes, 1]).to(edge_index.device)
                x = self.conv1(subgraph.x, edge_index)
                x = self.conv2(x, edge_index)
                x = self.fc0(x)
                x = F.gelu(x)
                if len(x)>0:
                    batch = torch.zeros(len(x), dtype=torch.long).to(data_list[0].edge_index.device)
                    x = global_add_pool(x, batch, size=1)
                else:
                    x = torch.zeros(1, self.hidden).to(data_list[0].edge_index.device)
                emb = torch.cat((emb, x), dim=0)
            #print(graph.num_nodes, emb.shape)
            if emb.shape[0] == 0:
                continue
                #int_emb[num, :graph.num_nodes, :] = torch.zeros((graph.num_nodes, self.hidden)).to(data_list[0].edge_index.device)
            else:
                int_emb[num, :emb.shape[0], :] = emb
            num += 1
            graph.x = emb
            new_data_list.append(graph) 
        res = self.fc1(int_emb)
        if len(new_data_list) == 0:
            new_edge_index = torch.zeros(2, 0).to(data_list[0].edge_index.device)
            new_edge_index = new_edge_index.to(torch.long)
            batch = Batch.from_data_list([Data(x=torch.zeros(1, self.hidden).to(data_list[0].edge_index.device), edge_index=new_edge_index)])
        else:
            batch = Batch.from_data_list(new_data_list)
        res = torch.squeeze(res, dim=-1)
        return batch, res