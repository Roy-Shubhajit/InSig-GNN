
import warnings
from torch_geometric.data import Data, InMemoryDataset, download_url, Batch
from dgl.data.utils import Subset, load_graphs
import torch
from torch_geometric.utils import k_hop_subgraph
from scipy.special import comb
warnings.filterwarnings('ignore')
import torch.utils.data


class Dataset_1_orig(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Dataset_1_orig, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['dataset1_orig.pt']

    def download(self):
        pass
    
    def countC4(self, data):
        edge_index, num_nodes = data.edge_index, data.num_nodes
        l = torch.tensor([], dtype=torch.long)
        for ind in range(num_nodes):
            node_dict = {}
            nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
            ind, 2, edge_index, False, num_nodes)
            edge_index_ = edge_index_.T
            nodes_ = nodes_[nodes_ != ind]
            edge_list = edge_index_.tolist()
            node_dict = {n.item(): 1 if [ind, n] in edge_list or [n, ind] in edge_list else 2 for n in nodes_}
            mask = (edge_index_ != ind).all(dim=1)
            edge_index_ = edge_index_[mask].T
            ll = 0
            edge_list = edge_index_.T.tolist()

            for n in node_dict:
                nei = sum([1 for m in node_dict if [n, m] in edge_list and [m, n] in edge_list and node_dict[m] == 1])
                if nei >= 2:
                    ll += comb(nei, 2, exact=True)
                    
            l = torch.cat((l, torch.tensor([ll])), dim=0)
       
        
        return torch.ceil(torch.sum(l)/4)
    
    def from_dgl(self, g, star, tri, tail_tri, attr_tri, chord):
        import dgl

        from torch_geometric.data import Data, HeteroData

        if not isinstance(g, dgl.DGLGraph):
            raise ValueError(f"Invalid data type (got '{type(g)}')")

       
        data = Data()
        data.edge_index = torch.stack(g.edges(), dim=0)

        for attr, value in g.ndata.items():
            data[attr] = value
        for attr, value in g.edata.items():
            data[attr] = value

        data.C4 = self.countC4(data)
        data.star = star.item()
        data.triangle = tri.item()
        data.tailed_triangle = tail_tri.item()
        data.attributed_triangle = attr_tri.item()
        data.chordal_cycle = chord.item()
        return data

    def process(self):
        glist, all_labels = load_graphs("/hdfs1/Data/Shubhajit/WL_Substructure_Counting/data/dataset1.bin")
        data_list = []
        for i in zip(glist, all_labels["star"], all_labels["triangle"], all_labels["tailed_triangle"], all_labels["attributed_triangle"], all_labels["chordal_cycle"]):
            data = self.from_dgl(i[0], i[1], i[2], i[3], i[4], i[5])
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class Dataset_2_orig(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Dataset_2_orig, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['dataset2_orig.pt']

    def download(self):
        pass

    def countC4(self, data):
        edge_index, num_nodes = data.edge_index, data.num_nodes
        l = torch.tensor([], dtype=torch.long)
        for ind in range(num_nodes):
            node_dict = {}
            nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
            ind, 2, edge_index, False, num_nodes)
            edge_index_ = edge_index_.T
            nodes_ = nodes_[nodes_ != ind]
            edge_list = edge_index_.tolist()
            node_dict = {n.item(): 1 if [ind, n] in edge_list or [n, ind] in edge_list else 2 for n in nodes_}
            mask = (edge_index_ != ind).all(dim=1)
            edge_index_ = edge_index_[mask].T
            ll = 0
            edge_list = edge_index_.T.tolist()

            for n in node_dict:
                nei = sum([1 for m in node_dict if [n, m] in edge_list and [m, n] in edge_list and node_dict[m] == 1])
                if nei >= 2:
                    ll += comb(nei, 2, exact=True)
                    
            l = torch.cat((l, torch.tensor([ll])), dim=0)
       
        
        return torch.ceil(torch.sum(l)/4)

    def from_dgl(self, g, star, tri, tail_tri, attr_tri, chord):
        import dgl

        from torch_geometric.data import Data, HeteroData

        if not isinstance(g, dgl.DGLGraph):
            raise ValueError(f"Invalid data type (got '{type(g)}')")

        
        data = Data()
        data.edge_index = torch.stack(g.edges(), dim=0)

        for attr, value in g.ndata.items():
            data[attr] = value
        for attr, value in g.edata.items():
            data[attr] = value

        data.C4 = self.countC4(data)
        data.star = star.item()
        data.triangle = tri.item()
        data.tailed_triangle = tail_tri.item()
        data.attributed_triangle = attr_tri.item()
        data.chordal_cycle = chord.item()

        return data

    def process(self):
        glist, all_labels = load_graphs("/hdfs1/Data/Shubhajit/WL_Substructure_Counting/data/dataset2.bin")
        data_list = []
        for i in zip(glist, all_labels["star"], all_labels["triangle"], all_labels["tailed_triangle"], all_labels["attributed_triangle"], all_labels["chordal_cycle"]):
            data = self.from_dgl(i[0], i[1], i[2], i[3], i[4], i[5])
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])