import math
import torch
import random
import numpy as np
import pandas as pd
import pickle as pkl
import networkx as nx
from igraph import Graph
from os.path import join
import graphxai.datasets as md
from collections import defaultdict
from explain.explanations import Explanation
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import InMemoryDataset, Data

class Mutagenicity(InMemoryDataset):
    def __init__(self, root = f'./data/datasets', transform=None, pre_transform=None, pre_filter=None):
        self.name = 'Mutagenicity'
        root =join(root, self.name)
        random.seed(54398724)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.explanations = self.loadExplanation()
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self.stratification_labels = self.load_stratification_labels()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt', 'explanations.pkl', 'stratification_labels.pkl']

    def process(self):
        dataset_class = getattr(md, self.name)
        dataset = dataset_class(seed = 5843997, root = self.root.split('Mutagenicity', 2)[0])
        data_list, expls = dataset.get_data_list(range(len(dataset)))

        zipped = list(zip(data_list, expls))
        random.shuffle(zipped)
        data_list, expls = zip(*zipped)

        novel_expl = []
        for elem in expls:
            tmp = [Explanation(explained_class=expl.graph.y.item(), node_relevance=expl.node_imp) for expl in elem]
            novel_expl.append(tmp)

        idx_to_remove = []
        for idx, elem in enumerate(data_list):
            G = to_networkx(elem, to_undirected=True)
            if not nx.is_connected(G):
                idx_to_remove.append(idx)
        
        data_list = [elem for i, elem in enumerate(data_list) if i not in idx_to_remove]
        novel_expl = [expl for i, expl in enumerate(novel_expl) if i not in idx_to_remove]
        for idx, e1 in enumerate(novel_expl):
            for ex in e1:
                ex.set_dataset_index(idx)
        explanations = {}
        explanations['motif'] = novel_expl

        nns=[]
        for data in data_list:
            nns.append(data.num_nodes)
        df = pd.DataFrame({'nns': nns})
        df["nns_binned"] = pd.qcut(df["nns"], q=5)
        df["nns_binned"] = df["nns_binned"].cat.codes
        stratifying_label = df["nns_binned"].to_numpy().astype(str)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        with open(self.processed_paths[1], 'wb') as f:
            pkl.dump(explanations, f)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
            
        with open(self.processed_paths[2], 'wb') as file:
            pkl.dump(stratifying_label, file)

    def load_stratification_labels(self):
        with open(self.processed_paths[2], 'rb') as file:
            return pkl.load(file)
        
    def loadExplanation(self):
        with open(self.processed_paths[1], 'rb') as f:
            return pkl.load(f)
        

class Benzene(InMemoryDataset):
    def __init__(self, root = f'./data/datasets', transform=None, pre_transform=None, pre_filter=None):
        self.name = 'Benzene'
        root =join(root, self.name)
        random.seed(54398724)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.explanations = self.loadExplanation()
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self.stratification_labels = self.load_stratification_labels()

    @property
    def raw_file_names(self):
        return [f'{self.name}.npz']
    
    @property
    def processed_file_names(self):
        return ['data.pt', 'explanations.pkl', 'stratification_labels.pkl']

    def process(self):
        dataset_class = getattr(md, self.name)
        dataset = dataset_class(seed = 5843997, data_path = self.raw_paths[0])
        data_list, expls = dataset.get_data_list(range(len(dataset)))

        zipped = list(zip(data_list, expls))
        random.shuffle(zipped)
        data_list, expls = zip(*zipped)

        novel_expl = []
        for elem in expls:
            tmp = [Explanation(explained_class=expl.graph.y.item(), node_relevance=expl.node_imp) for expl in elem]
            novel_expl.append(tmp)
        for idx, e1 in enumerate(novel_expl):
            for ex in e1:
                ex.set_dataset_index(idx)
        explanations = {}
        explanations['motif'] = novel_expl

        nns=[]
        for data in data_list:
            nns.append(data.num_nodes)

        df = pd.DataFrame({'nns': nns})
        df["nns_binned"] = pd.qcut(df["nns"], q=5)
        df["nns_binned"] = df["nns_binned"].cat.codes
        stratifying_label = df["nns_binned"].to_numpy().astype(str)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        with open(self.processed_paths[1], 'wb') as f:
            pkl.dump(explanations, f)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
            
        with open(self.processed_paths[2], 'wb') as file:
            pkl.dump(stratifying_label, file)

    def load_stratification_labels(self):
        with open(self.processed_paths[2], 'rb') as file:
            return pkl.load(file)
        
    def loadExplanation(self):
        with open(self.processed_paths[1], 'rb') as f:
            return pkl.load(f)


class FluorideCarbonyl(InMemoryDataset):
    def __init__(self, root = f'./data/datasets', transform=None, pre_transform=None, pre_filter=None):
        self.name = 'FluorideCarbonyl'
        root =join(root, self.name)
        random.seed(54398724)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.explanations = self.loadExplanation()
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self.stratification_labels = self.load_stratification_labels()

    @property
    def raw_file_names(self):
        return [f'{self.name}.npz']
    
    @property
    def processed_file_names(self):
        return [f'data.pt', 'explanations.pkl', 'stratification_labels.pkl']

    def process(self):
        dataset_class = getattr(md, self.name)
        dataset = dataset_class(seed = 5843997, data_path = self.raw_paths[0])
        data_list, expls = dataset.get_data_list(range(len(dataset)))

        zipped = list(zip(data_list, expls))
        random.shuffle(zipped)
        data_list, expls = zip(*zipped)

        novel_expl = []
        for elem in expls:
            tmp = [Explanation(explained_class=expl.graph.y.item(), node_relevance=expl.node_imp) for expl in elem]
            novel_expl.append(tmp)
        for idx, e1 in enumerate(novel_expl):
            for ex in e1:
                ex.set_dataset_index(idx)
        explanations = {}
        explanations['motif'] = novel_expl

        nns=[]
        for data in data_list:
            nns.append(data.num_nodes)
        df = pd.DataFrame({'nns': nns})
        df["nns_binned"] = pd.qcut(df["nns"], q=5)
        df["nns_binned"] = df["nns_binned"].cat.codes
        stratifying_label = df["nns_binned"].to_numpy().astype(str)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        with open(self.processed_paths[1], 'wb') as f:
            pkl.dump(explanations, f)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
            
        with open(self.processed_paths[2], 'wb') as file:
            pkl.dump(stratifying_label, file)

    def load_stratification_labels(self):
        with open(self.processed_paths[2], 'rb') as file:
            return pkl.load(file)
        
    def loadExplanation(self):
        with open(self.processed_paths[1], 'rb') as f:
            return pkl.load(f)


class AlkaneCarbonyl(InMemoryDataset):
    def __init__(self, root = f'./data/datasets', transform=None, pre_transform=None, pre_filter=None):
        self.name = 'AlkaneCarbonyl'
        root =join(root, self.name)
        random.seed(54398724)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.explanations = self.loadExplanation()
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self.stratification_labels = self.load_stratification_labels()

    @property
    def raw_file_names(self):
        return [f'{self.name}.npz']
    
    @property
    def processed_file_names(self):
        return ['data.pt', 'explanations.pkl', 'stratification_labels.pkl']

    def process(self):
        dataset_class = getattr(md, self.name)
        dataset = dataset_class(seed = 5843997, downsample_seed = 262904682, data_path = self.raw_paths[0])
        data_list, expls = dataset.get_data_list(range(len(dataset)))

        zipped = list(zip(data_list, expls))
        random.shuffle(zipped)
        data_list, expls = zip(*zipped)

        novel_expl = []
        for elem in expls:
            tmp = [Explanation(explained_class=expl.graph.y.item(), node_relevance=expl.node_imp) for expl in elem]
            novel_expl.append(tmp)

        for idx, e1 in enumerate(novel_expl):
            for ex in e1:
                ex.set_dataset_index(idx)

        explanations = {}
        explanations['motif'] = novel_expl

        nns=[]
        for data in data_list:
            nns.append(data.num_nodes)

        df = pd.DataFrame({'nns': nns})
        df["nns_binned"] = pd.qcut(df["nns"], q=5)
        df["nns_binned"] = df["nns_binned"].cat.codes
        stratifying_label = df["nns_binned"].to_numpy().astype(str)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        with open(self.processed_paths[1], 'wb') as f:
            pkl.dump(explanations, f)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
            
        with open(self.processed_paths[2], 'wb') as file:
            pkl.dump(stratifying_label, file)

    def load_stratification_labels(self):
        with open(self.processed_paths[2], 'rb') as file:
            return pkl.load(file)
        
    def loadExplanation(self):
        with open(self.processed_paths[1], 'rb') as f:
            return pkl.load(f)


class BA2grid(InMemoryDataset):
    def __init__(self, root = './data/datasets', counter_examples=0, transform=None, pre_transform=None, pre_filter=None):
        self.name = 'BA2grid'
        self.counter = counter_examples
        self.identifier = f'BA2grid{self.counter}'
        root =join(root, self.name)
        random.seed(54398724)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.explanations = self.loadExplanation()
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self.stratification_labels = self.load_stratification_labels()

    @property
    def raw_file_names(self):
        return ['BA2grid.pkl']

    @property
    def processed_file_names(self):
        return [f'{self.identifier}.pt', f'{self.identifier}_expls.pkl', f'{self.identifier}_strat_labels.pkl']

    def process(self):
        with open(self.raw_paths[0],'rb') as fin:
            data = pkl.load(fin)

        c0, c1, cc0 = data['class0'], data['class1'], data['counterClass0']

        zipped = list(zip(c0, cc0))
        random.shuffle(zipped)
        c0, cc0 = zip(*zipped)

        limit = math.floor(len(c0)*(1-(self.counter)))
        partialc0 = list(c0[:limit])
        partialcc0 = list(cc0[limit:])

        explanations = defaultdict(list)
        c0 = partialc0 + partialcc0
        targets = [0]*len(c0) + [1]*len(c1)
        data = c0 + c1

        zipped = list(zip(data, targets))
        random.shuffle(zipped)
        data, targets = zip(*zipped)

        explanations = defaultdict(list)
        data_list = []

        grid = self.get_grid()

        for idx, (g, target) in enumerate(zip(data, targets)):
            sample = from_networkx(g)
            sample.x = torch.ones((sample.num_nodes,10))/10
            sample.y = torch.LongTensor([target])
            data_list.append(sample)

            degree = [elem[1] for elem in g.degree]

            if target == 1:
                subgraph_nodes = self.subgraph_nodes(g, grid)
                z = torch.Tensor([1 if k in subgraph_nodes else 0 for k in range(sample.num_nodes)]).float()
                ex = Explanation(explained_class=target, node_relevance=z)
                ex.set_dataset_index(idx)
                explanations['motif'].append([ex])
                z = torch.tensor([1 if degree[k]>=3 else 0 for k in range(sample.num_nodes)]).float()
                ex = Explanation(explained_class=target, node_relevance=z)
                ex.set_dataset_index(idx)
                explanations['degree'].append([ex])
            else:
                explanations['motif'].append([])
                explanations['degree'].append([])

        nns=[]
        for data in data_list:
            nns.append(data.num_nodes)
        df = pd.DataFrame({'nns': nns})
        df["nns_binned"] = pd.qcut(df["nns"], q=5)
        df["nns_binned"] = df["nns_binned"].cat.codes
        stratifying_label = df["nns_binned"].to_numpy().astype(str)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        with open(self.processed_paths[1], 'wb') as f:
            pkl.dump(explanations, f)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        with open(self.processed_paths[2], 'wb') as file:
            pkl.dump(stratifying_label, file)

    def get_grid(self):
        grid = nx.Graph()
        grid.add_nodes_from([elem for elem in range(9)])
        grid.add_edges_from([(0,1), (1,2), (0,3), (1,4), (2,5), (3,4), (4,5), (3,6), (4,7), (5,8), (6,7), (7,8)])
        return grid

    def subgraph_nodes(self, g, subgraph):
        isomatcher = nx.isomorphism.GraphMatcher(g, subgraph)
        sub_nodes = [elem for (elem, sub) in next(isomatcher.subgraph_isomorphisms_iter()).items()]
        return sub_nodes

    def load_stratification_labels(self):
        with open(self.processed_paths[2], 'rb') as file:
            return pkl.load(file)

    def loadExplanation(self):
        with open(self.processed_paths[1], 'rb') as f:
            return pkl.load(f)

class GridHouse(InMemoryDataset):
    def __init__(self, root = './data/datasets', counter_examples=0, transform=None, pre_transform=None, pre_filter=None):
        self.name = 'GridHouse'
        self.counter = counter_examples
        self.identifier = f'GridHouse{self.counter}'
        root =join(root, self.name)
        random.seed(54398724)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.explanations = self.loadExplanation()
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self.stratification_labels = self.load_stratification_labels()

    @property
    def raw_file_names(self):
        return ['GridHouse.pkl']

    @property
    def processed_file_names(self):
        return [f'{self.identifier}.pt', f'{self.identifier}_expls.pkl', f'{self.identifier}_strat_labels.pkl']

    def process(self):
        with open(self.raw_paths[0],'rb') as fin:
            data = pkl.load(fin)

        c0, c1, cc0 = data['class0'], data['class1'], data['counterClass0']

        zipped = list(zip(c0, cc0))
        random.shuffle(zipped)
        c0, cc0 = zip(*zipped)

        limit = math.floor(len(c0)*(1-(self.counter)))
        partialc0 = list(c0[:limit])
        partialcc0 = list(cc0[limit:])

        explanations = defaultdict(list)
        c0 = partialc0 + partialcc0
        targets = [0]*len(c0) + [1]*len(c1)
        data = c0 + c1

        zipped = list(zip(data, targets))
        random.shuffle(zipped)
        data, targets = zip(*zipped)

        explanations = defaultdict(list)
        data_list = []

        house = self.getHouse()
        grid = self.getGrid()

        for idx, (g, target) in enumerate(zip(data, targets)):
            sample = from_networkx(g)
            sample.x = torch.ones((sample.num_nodes,10))/10
            sample.y = torch.LongTensor([target])
            data_list.append(sample)
            degree = [elem[1] for elem in g.degree]

            if target == 0:
                subgraph_nodes = self.subgraph_nodes(g, house) + self.subgraph_nodes(g, grid)
                z = torch.Tensor([1 if k in subgraph_nodes else 0 for k in range(sample.num_nodes)]).float()
                ex = Explanation(explained_class=target, node_relevance=z)
                ex.set_dataset_index(idx)
                explanations['motif'].append([ex])
                explanations['degree'].append([])
            else:
                subgraph_nodes = self.subgraph_nodes(g, house) + self.subgraph_nodes(g, grid)
                z = torch.Tensor([1 if k in subgraph_nodes else 0 for k in range(sample.num_nodes)]).float()
                ex = Explanation(explained_class=target, node_relevance=z)
                ex.set_dataset_index(idx)
                explanations['motif'].append([ex])
                z = torch.tensor([1 if degree[k]>=3 else 0 for k in range(sample.num_nodes)]).float()
                ex = Explanation(explained_class=target, node_relevance=z)
                ex.set_dataset_index(idx)
                explanations['degree'].append([ex])

        nns=[]
        for data in data_list:
            nns.append(data.num_nodes)
        df = pd.DataFrame({'nns': nns})
        df["nns_binned"] = pd.qcut(df["nns"], q=5)
        df["nns_binned"] = df["nns_binned"].cat.codes
        stratifying_label = df["nns_binned"].to_numpy().astype(str)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        with open(self.processed_paths[1], 'wb') as f:
            pkl.dump(explanations, f)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        with open(self.processed_paths[2], 'wb') as file:
            pkl.dump(stratifying_label, file)

    def load_stratification_labels(self):
        with open(self.processed_paths[2], 'rb') as file:
            return pkl.load(file)

    def loadExplanation(self):
        with open(self.processed_paths[1], 'rb') as f:
            return pkl.load(f)
        
    def subgraph_nodes(self, g, subgraph):
        isomatcher = nx.isomorphism.GraphMatcher(g, subgraph)
        sub_nodes = []
        if isomatcher.subgraph_is_isomorphic():
            sub_nodes = [elem for (elem, sub) in next(isomatcher.subgraph_isomorphisms_iter()).items()]
        return sub_nodes

    def getHouse(self):
        house = nx.Graph()
        house.add_nodes_from([elem for elem in range(5)])
        house.add_edges_from([(0,1), (1,2), (2,3), (0,3), (3,4), (2,4)])
        return house

    def getGrid(self):
        grid = nx.Graph()
        grid.add_nodes_from([elem for elem in range(9)])
        grid.add_edges_from([(0,1), (1,2), (0,3), (1,4), (2,5), (3,4), (4,5), (3,6), (4,7), (5,8), (6,7), (7,8)])
        return grid


class HouseColors(InMemoryDataset):
    def __init__(self, root = './data/datasets', transform=None, pre_transform=None, pre_filter=None):
        self.name = 'HouseColors'
        root =join(root, self.name)
        random.seed(54398724)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.explanations = self.loadExplanation()
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ['BA-houses_color.pkl']

    @property
    def processed_file_names(self):
        return ['BA-houses_color.pt', 'explanations.pkl']

    def process(self):
        with open(self.raw_paths[0],'rb') as fin:
            (adjs, feas, labels, Gt) = pkl.load(fin)

        zipped = list(zip(adjs, feas, labels, Gt))
        random.shuffle(zipped)
        adjs, feas, labels, Gt = zip(*zipped)

        data_list = []
        explanations = defaultdict(list)

        for idx, (adj, lab, target, gt) in enumerate(zip(adjs, feas, labels, Gt)):
            ig = Graph.Adjacency(adj, mode = 'undirected')
            edjI = []
            for edge in ig.es:
                e = list(edge.tuple)
                edjI.append(e)
                edjI.append(list(reversed(e)))
            graph = Data(x = torch.FloatTensor(lab), edge_index= torch.LongTensor(edjI).t().contiguous(), y = torch.LongTensor([target]))
            data_list.append(graph)

            z = torch.zeros(len(adj))
            z[gt[0][0]:gt[0][-1]+1] = 1
            ex = Explanation(explained_class=target, node_relevance=z)
            ex.set_dataset_index(idx)
            explanations['motif'].append([ex])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        with open(self.processed_paths[1], 'wb') as f:
            pkl.dump(explanations, f)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def loadExplanation(self):
        with open(self.processed_paths[1], 'rb') as f:
            return pkl.load(f)


class BA2Motif(InMemoryDataset):
    def __init__(self, root = './data/datasets', counter_examples=0, transform=None, pre_transform=None, pre_filter=None):
        self.name = 'BA2Motif'
        self.counter = counter_examples
        self.identifier = f'BA2Motif{self.counter}'
        root =join(root, self.name)
        random.seed(54398724)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.explanations = self.loadExplanation()
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ['BA2Motif.pkl']

    @property
    def processed_file_names(self):
        return [f'{self.identifier}.pt', f'{self.identifier}_expls.pkl']

    def process(self):
        with open(self.raw_paths[0],'rb') as fin:
            data = pkl.load(fin)

        c0, c1, cc0 = data['class0'], data['class1'], data['counterClass0']

        zipped = list(zip(c0, cc0))
        random.shuffle(zipped)
        c0, cc0 = zip(*zipped)

        limit = math.floor(len(c0)*(1-(self.counter)))
        partialc0 = list(c0[:limit])
        partialcc0 = list(cc0[limit:])

        explanations = defaultdict(list)
        c0 = partialc0 + partialcc0
        targets = [0]*len(c0) + [1]*len(c1)
        data = c0 + c1

        zipped = list(zip(data, targets))
        random.shuffle(zipped)
        data, targets = zip(*zipped)

        explanations = defaultdict(list)
        data_list = []

        house = self.getHouse()
        circle = self.getCircle()

        for idx, (g, target) in enumerate(zip(data, targets)):
            sample = from_networkx(g)
            sample.x = torch.ones((sample.num_nodes,10))/10
            sample.y = torch.LongTensor([target])
            data_list.append(sample)
            degree = [elem[1] for elem in g.degree]

            if target == 0:
                subgraph_nodes = self.subgraph_nodes(g, circle)
                z = torch.Tensor([1 if k in subgraph_nodes else 0 for k in range(sample.num_nodes)]).float()
                ex = Explanation(explained_class=target, node_relevance=z)
                ex.set_dataset_index(idx)
                explanations['motif'].append([ex])
                explanations['degree'].append([])
            else:
                subgraph_nodes = self.subgraph_nodes(g, house)
                z = torch.Tensor([1 if k in subgraph_nodes else 0 for k in range(sample.num_nodes)]).float()
                ex = Explanation(explained_class=target, node_relevance=z)
                ex.set_dataset_index(idx)
                explanations['motif'].append([ex])
                z = torch.tensor([1 if degree[k]>=3 else 0 for k in range(sample.num_nodes)]).float()
                ex = Explanation(explained_class=target, node_relevance=z)
                ex.set_dataset_index(idx)
                explanations['degree'].append([ex])
                    
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        with open(self.processed_paths[1], 'wb') as f:
            pkl.dump(explanations, f)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def loadExplanation(self):
        with open(self.processed_paths[1], 'rb') as f:
            return pkl.load(f)

    def subgraph_nodes(self, g, subgraph):
        isomatcher = nx.isomorphism.GraphMatcher(g, subgraph)
        sub_nodes = [elem for (elem, sub) in next(isomatcher.subgraph_isomorphisms_iter()).items()]
        return sub_nodes

    def getHouse(self):
        house = nx.Graph()
        house.add_nodes_from([elem for elem in range(5)])
        house.add_edges_from([(0,1), (1,2), (2,3), (0,3), (3,4), (2,4)])
        return house

    def getCircle(self):
        circle = nx.Graph()
        circle.add_nodes_from([elem for elem in range(5)])
        circle.add_edges_from([(0,1), (1,2), (2,3), (3,4), (4,0)])  
        return circle
    


class OGX(InMemoryDataset):
    def __init__(self, root = './data/datasets', data_name='alfa', transform=None, pre_transform=None, pre_filter=None):
        self.name = 'OGX'
        self.data_name = data_name
        self.identifier = f'OGX{data_name}'
        root =join(root, self.name)
        random.seed(54398724)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.explanations = self.loadExplanation()
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self.stratification_labels = self.load_stratification_labels()
        self.mapper = self.load_mapper()

    @property
    def raw_file_names(self):
        return [f'{self.data_name}.pkl']

    @property
    def processed_file_names(self):
        return [f'{self.identifier}.pt', f'{self.identifier}_expls.pkl', f'{self.identifier}_strat_labels.pkl', f'{self.identifier}_mapper.pkl']

    def process(self):
        with open(self.raw_paths[0],'rb') as fin:
            raws = pkl.load(fin)

        c0 = [from_networkx(graph, group_node_attrs=['x']) for graph in raws['class0']]
        c1 = [from_networkx(graph, group_node_attrs=['x']) for graph in raws['class1']]

        targets = [0]*len(c0) + [1]*len(c1)
        data = c0 + c1

        zipped = list(zip(data, targets))
        random.shuffle(zipped)
        data, targets = zip(*zipped)

        explanations = defaultdict(list)

        data_list= []
        for idx, (elem, target) in enumerate(zip(data, targets)):
            attributes = elem.keys

            ex1 = []
            ex2 = []
            if 'mask_root' in attributes and 'mask' in attributes:
                node_mask = elem.mask_root.int()
                ego_mask = elem.mask.int()

                ex_node = Explanation(explained_class=target, node_relevance=node_mask)
                ex_node.set_dataset_index(idx)
                ex1.append(ex_node)

                ex_sub = Explanation(explained_class=target, node_relevance=ego_mask)
                ex_sub.set_dataset_index(idx)
                ex2.append(ex_sub)

            explanations['node'].append(ex1)
            explanations['ego_subgraph'].append(ex2)

            for attr in attributes:
                if attr != 'edge_index' and attr != 'x' and attr != 'smile':
                    del elem[attr]
                    
            elem.y = torch.LongTensor([target])
            data_list.append(elem)

        atn = set()
        for elem in data_list:
            for an in elem.x[:,0]:
                atn.add(an.item())
        atn = list(atn)

        mapperAtn = {}
        for i, at in enumerate(atn):
            mapperAtn[at] = i

        for k, data in enumerate(data_list):
            indexes = torch.tensor([mapperAtn[x.item()] for x in data.x[:,0]]).view(-1,1)
            ohAtm = torch.nn.functional.one_hot(indexes[:,0].to(torch.int64), len(mapperAtn))
            data_list[k].x = ohAtm.float()

        nns=[]
        for data in data_list:
            nns.append(data.num_nodes)
        df = pd.DataFrame({'nns': nns})
        df["nns_binned"] = pd.qcut(df["nns"], q=5)
        df["nns_binned"] = df["nns_binned"].cat.codes
        stratifying_label = df["nns_binned"].to_numpy().astype(str)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        with open(self.processed_paths[1], 'wb') as f:
            pkl.dump(explanations, f)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        with open(self.processed_paths[2], 'wb') as file:
            pkl.dump(stratifying_label, file)

        with open(self.processed_paths[3], 'wb') as file:
            pkl.dump(mapperAtn, file)

    def load_stratification_labels(self):
        with open(self.processed_paths[2], 'rb') as file:
            return pkl.load(file)

    def loadExplanation(self):
        with open(self.processed_paths[1], 'rb') as f:
            return pkl.load(f)

    def load_mapper(self):
        with open(self.processed_paths[3], 'rb') as file:
            return pkl.load(file)