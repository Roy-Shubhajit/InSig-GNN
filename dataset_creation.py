
from torch_geometric.utils import degree
import torch.utils.data
import warnings
from torch_geometric.data import Data, InMemoryDataset, download_url, Batch
from dgl.data.utils import Subset, load_graphs
import torch
from torch_geometric.utils import k_hop_subgraph
from scipy.special import comb
warnings.filterwarnings('ignore')


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

    def count_triangle(self, data):
        edge_index, num_nodes = data.edge_index, data.num_nodes
        if num_nodes > 0:
            node_name = torch.unique(edge_index[0])
        else:
            return torch.tensor([0])
        num_edges = 0
        for ind in node_name:
            nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                ind.item(), 1, edge_index, False, num_nodes)
            edge_attr_ = None
            edge_index_ = edge_index_.T
            mask = (edge_index_ != ind).all(dim=1)
            edge_index_ = edge_index_[mask].T
            num_edges += edge_index_.shape[1]
        return torch.tensor([num_edges//6])

    def count_K4(self, data):
        edge_index, num_nodes = data.edge_index, data.num_nodes
        total_edge_index = torch.tensor([], dtype=torch.long)
        l = torch.tensor([], dtype=torch.long)
        for ind in range(num_nodes):
            nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                ind, 2, edge_index, False, num_nodes)
            edge_index_ = edge_index_.T
            mask = (edge_index_ != ind).all(dim=1)
            edge_index_ = edge_index_[mask].T
            data_ = Data(edge_index=edge_index_, z=z_)
            l = torch.cat((l, self.count_triangle(data_)), dim=0)

        return torch.sum(l)//4

    def count_2star(self, data):
        egde_index, num_nodes = data.edge_index, data.num_nodes
        k = torch.tensor([], dtype=torch.long)
        for ind in range(num_nodes):
            nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                ind, 1, egde_index, False, num_nodes)
            edge_index_ = edge_index_.T
            mask = (edge_index_ == ind).all(dim=1)
            edge_index_ = edge_index_[mask].T
            k = torch.cat(
                (k, torch.tensor([comb(nodes_.shape[0]-1, 2, exact=True)])), dim=0)
        return torch.sum(k)

    def count_chordal(self, data):
        edge_index, num_nodes = data.edge_index, data.num_nodes
        total_edge_index = torch.tensor([], dtype=torch.long)
        l = torch.tensor([], dtype=torch.long)
        for ind in range(num_nodes):
            nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                ind, 1, edge_index, False, num_nodes)
            edge_index_ = edge_index_.T
            mask = (edge_index_ != ind).all(dim=1)
            edge_index_ = edge_index_[mask].T
            total_edge_index = torch.cat(
                (total_edge_index, edge_index_.T), dim=0)
            nodes_ = nodes_[nodes_ != ind]
            deg = degree(edge_index_[0], num_nodes)
            ll = sum([comb(i, 2, exact=True) for i in deg[deg > 1]])
            l = torch.cat((l, torch.tensor([ll])), dim=0)
        return torch.sum(l)//2

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
            node_dict = {n.item(): 1 if [ind, n] in edge_list or [
                n, ind] in edge_list else 0 for n in nodes_}
            mask = (edge_index_ != ind).all(dim=1)
            edge_index_ = edge_index_[mask].T
            ll = 0
            edge_list = edge_index_.T.tolist()

            for n in node_dict:
                nei = sum([1 for m in node_dict if [n, m] in edge_list and [
                          m, n] in edge_list and node_dict[m] == 1])
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
        data.chordal_cycle = self.count_chordal(data)
        data.star_2 = self.count_2star(data)
        data.K4 = self.count_K4(data)
        return data

    def process(self):
        glist, all_labels = load_graphs(
            "/hdfs1/Data/Shubhajit/WL_Substructure_Counting/data/dataset1.bin")
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

    def count_triangle(self, data):
        edge_index, num_nodes = data.edge_index, data.num_nodes
        if num_nodes > 0:
            node_name = torch.unique(edge_index[0])
        else:
            return torch.tensor([0])
        num_edges = 0
        for ind in node_name:
            nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                ind.item(), 1, edge_index, False, num_nodes)
            edge_attr_ = None
            edge_index_ = edge_index_.T
            mask = (edge_index_ != ind).all(dim=1)
            edge_index_ = edge_index_[mask].T
            num_edges += edge_index_.shape[1]
        return torch.tensor([num_edges//6])

    def count_K4(self, data):
        edge_index, num_nodes = data.edge_index, data.num_nodes
        total_edge_index = torch.tensor([], dtype=torch.long)
        l = torch.tensor([], dtype=torch.long)
        for ind in range(num_nodes):
            nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                ind, 2, edge_index, False, num_nodes)
            edge_index_ = edge_index_.T
            mask = (edge_index_ != ind).all(dim=1)
            edge_index_ = edge_index_[mask].T
            data_ = Data(edge_index=edge_index_, z=z_)
            l = torch.cat((l, self.count_triangle(data_)), dim=0)

        return torch.sum(l)//4

    def count_2star(self, data):
        egde_index, num_nodes = data.edge_index, data.num_nodes
        k = torch.tensor([], dtype=torch.long)
        for ind in range(num_nodes):
            nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                ind, 1, egde_index, False, num_nodes)
            edge_index_ = edge_index_.T
            mask = (edge_index_ == ind).all(dim=1)
            edge_index_ = edge_index_[mask].T
            k = torch.cat(
                (k, torch.tensor([comb(nodes_.shape[0]-1, 2, exact=True)])), dim=0)
        return torch.sum(k)

    def count_chordal(self, data):
        edge_index, num_nodes = data.edge_index, data.num_nodes
        total_edge_index = torch.tensor([], dtype=torch.long)
        l = torch.tensor([], dtype=torch.long)
        for ind in range(num_nodes):
            nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                ind, 1, edge_index, False, num_nodes)
            edge_index_ = edge_index_.T
            mask = (edge_index_ != ind).all(dim=1)
            edge_index_ = edge_index_[mask].T
            total_edge_index = torch.cat(
                (total_edge_index, edge_index_.T), dim=0)
            nodes_ = nodes_[nodes_ != ind]
            deg = degree(edge_index_[0], num_nodes)
            ll = sum([comb(i, 2, exact=True) for i in deg[deg > 1]])
            l = torch.cat((l, torch.tensor([ll])), dim=0)
        return torch.sum(l)//2

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
            node_dict = {n.item(): 1 if [ind, n] in edge_list or [
                n, ind] in edge_list else 0 for n in nodes_}
            mask = (edge_index_ != ind).all(dim=1)
            edge_index_ = edge_index_[mask].T
            ll = 0
            edge_list = edge_index_.T.tolist()

            for n in node_dict:
                nei = sum([1 for m in node_dict if [n, m] in edge_list and [
                          m, n] in edge_list and node_dict[m] == 1])
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
        data.chordal_cycle = self.count_chordal(data)
        data.star_2 = self.count_2star(data)
        data.K4 = self.count_K4(data)
        return data

    def process(self):
        glist, all_labels = load_graphs(
            "/hdfs1/Data/Shubhajit/WL_Substructure_Counting/data/dataset2.bin")
        data_list = []
        for i in zip(glist, all_labels["star"], all_labels["triangle"], all_labels["tailed_triangle"], all_labels["attributed_triangle"], all_labels["chordal_cycle"]):
            data = self.from_dgl(i[0], i[1], i[2], i[3], i[4], i[5])
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
