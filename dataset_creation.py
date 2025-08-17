
from torch_geometric.utils import degree
import torch.utils.data
import warnings
from torch_geometric.data import Data, InMemoryDataset, download_url, Batch
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
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['dataset_1_orig.pt']

    def download(self):
        pass

    def process(self):
        import pickle
        with open('data/dataset_1.pkl', 'rb') as f:
            data_list = pickle.load(f)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(data_list, self.processed_paths[0])


class Dataset_2_orig(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Dataset_2_orig, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['dataset_2_orig.pt']

    def download(self):
        pass

    def process(self):
        import pickle
        with open('data/dataset_2.pkl', 'rb') as f:
            data_list = pickle.load(f)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(data_list, self.processed_paths[0])

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
        with open('data/chembl.pkl', 'rb') as f:
            chembl_data = pickle.load(f)
        data_list = []
        for i in tqdm(chembl_data):
            data = self.from_chembl(i)
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in tqdm(data_list)]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])