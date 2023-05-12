
import warnings
from torch_geometric.data import Data, InMemoryDataset, download_url, Batch
from dgl.data.utils import Subset, load_graphs
import torch
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

    def from_dgl(self, g, star, tri, tail_tri, attr_tri, chord):
        import dgl

        from torch_geometric.data import Data, HeteroData

        if not isinstance(g, dgl.DGLGraph):
            raise ValueError(f"Invalid data type (got '{type(g)}')")

        if g.is_homogeneous:
            data = Data()
            data.edge_index = torch.stack(g.edges(), dim=0)

            for attr, value in g.ndata.items():
                data[attr] = value
            for attr, value in g.edata.items():
                data[attr] = value

            data.star = star.item()
            data.triangle = tri.item()
            data.tailed_triangle = tail_tri.item()
            data.attributed_triangle = attr_tri.item()
            data.chordal_cycle = chord.item()
            return data

        data = HeteroData()

        for node_type in g.ntypes:
            for attr, value in g.nodes[node_type].data.items():
                data[node_type][attr] = value

        for edge_type in g.canonical_etypes:
            row, col = g.edges(form="uv", etype=edge_type)
            data[edge_type].edge_index = torch.stack([row, col], dim=0)
            for attr, value in g.edge_attr_schemes(edge_type).items():
                data[edge_type][attr] = value
        
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

    def from_dgl(self, g, star, tri, tail_tri, attr_tri, chord):
        import dgl

        from torch_geometric.data import Data, HeteroData

        if not isinstance(g, dgl.DGLGraph):
            raise ValueError(f"Invalid data type (got '{type(g)}')")

        if g.is_homogeneous:
            data = Data()
            data.edge_index = torch.stack(g.edges(), dim=0)

            for attr, value in g.ndata.items():
                data[attr] = value
            for attr, value in g.edata.items():
                data[attr] = value

            data.star = star.item()
            data.triangle = tri.item()
            data.tailed_triangle = tail_tri.item()
            data.attributed_triangle = attr_tri.item()
            data.chordal_cycle = chord.item()

            return data

        data = HeteroData()

        for node_type in g.ntypes:
            for attr, value in g.nodes[node_type].data.items():
                data[node_type][attr] = value

        for edge_type in g.canonical_etypes:
            row, col = g.edges(form="uv", etype=edge_type)
            data[edge_type].edge_index = torch.stack([row, col], dim=0)
            for attr, value in g.edge_attr_schemes(edge_type).items():
                data[edge_type][attr] = value
        
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