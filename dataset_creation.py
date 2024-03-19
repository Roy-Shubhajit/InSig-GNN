
from torch_geometric.utils import degree
import torch.utils.data
import warnings
from torch_geometric.data import Data, InMemoryDataset, download_url, Batch
from dgl.data.utils import Subset, load_graphs
import torch
from torch_geometric.utils import k_hop_subgraph, to_undirected
from scipy.special import comb
from rdkit import Chem
import pickle
from tqdm import tqdm
from dataset_labels import *
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

    def from_dgl(self, g, star, tri, tail_tri, attr_tri, chord):
        import dgl

        from torch_geometric.data import Data, HeteroData

        if not isinstance(g, dgl.DGLGraph):
            raise ValueError(f"Invalid data type (got '{type(g)}')")
        
        data = Data(edge_index=torch.stack(g.edges(), dim=0))

        for attr, value in g.ndata.items():
            data[attr] = value
        for attr, value in g.edata.items():
            data[attr] = value
        data = count_labels(data)
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

        data = count_labels(data)
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

class Dataset_chembl(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Dataset_chembl, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['dataset_chembl.pt']

    def download(self):
        pass


    def from_chembl(self, molecule):
        mol = Chem.MolFromSmiles(molecule)
        # Get atom and bond information
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()
        edge_list = []

        # Add bonds to the edge list
        for bond in bonds:
            atom1_idx = bond.GetBeginAtomIdx()
            atom2_idx = bond.GetEndAtomIdx()
            edge_list.append([atom1_idx, atom2_idx])

        data = Data()
        data.edge_index = to_undirected(torch.tensor(edge_list, dtype=torch.long).t().contiguous())
        data1 = count_labels(data)
        return data1

    def process(self):
        with open('/hdfs1/Data/Shubhajit/Sub-Structure-GNN/data/chembl.pkl', 'rb') as f:
            chembl_data = pickle.load(f)
        data_list = []
        for i in tqdm(chembl_data):
            data = self.from_chembl(i)
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in tqdm(data_list)]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])