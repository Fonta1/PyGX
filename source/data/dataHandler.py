import pickle
import os.path
import importlib
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

DATASETS = ['Mutagenicity', 'AlkaneCarbonyl', 'FluorideCarbonyl', 'Benzene', 
            'BA2grid', 'GridHouse', 'HouseColors', 'BA2Motif', 'OGX']

def split_data(dataset, ratio=(0.2, 0.1), mode='HO', k=5, stratify=True):
    dataPath = dataset.root
    splitPath = os.path.join(dataPath, 'splits')

    identifier=''
    if hasattr(dataset, 'identifier'):
        identifier = dataset.identifier+'_'

    if not os.path.exists(dataPath):
        raise Exception("The path to the dataset is incorrect")

    if not os.path.exists(splitPath):
        os.makedirs(splitPath)

    splits = []
    targets = dataset.y.long().numpy().astype(str)
    if hasattr(dataset, "stratification_labels") and stratify:
        targets = targets + dataset.stratification_labels

    if mode == 'HO':
        if os.path.isfile(os.path.join(splitPath, f'{identifier}HO_val{ratio[0]}_test{ratio[1]}_stratify{stratify}.pkl')):
            with open(os.path.join(splitPath, f'{identifier}HO_val{ratio[0]}_test{ratio[1]}_stratify{stratify}.pkl'), 'rb') as f:
                return pickle.load(f)
        else:
            indices = np.arange(len(dataset))
            test_ratio = ratio[1]
            val_ratio = (len(dataset)*ratio[0])/(len(dataset)*(1-ratio[1]))

            splitter_train_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=5345)
            splitter_train_val = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=84678)
            splits_t_t = splitter_train_test.split(indices, targets)
            
            train_index, test_idx =  next(splits_t_t)
            t_idx, v_idx = next(splitter_train_val.split(train_index, targets[train_index]))

            splits.append({'train': train_index[t_idx], 'val': train_index[v_idx], 'test': test_idx})

            with open(os.path.join(splitPath, f'{identifier}HO_val{ratio[0]}_test{ratio[1]}_stratify{stratify}.pkl'), 'wb') as f:
                pickle.dump(splits, f)

    elif mode == 'CT':
        if os.path.isfile(os.path.join(splitPath, f'{identifier}CT_{k}fold_val{ratio[0]}_stratify{stratify}.pkl')):
            with open(os.path.join(splitPath, f'{identifier}CT_{k}fold_val{ratio[0]}_stratify{stratify}.pkl'), 'rb') as f:
                return pickle.load(f)
        else:
            indices = np.arange(len(dataset))

            test_ratio = 1/k
            val_ratio = (len(dataset)*ratio[0])/(len(dataset)*(1-test_ratio))

            out_split = StratifiedKFold(n_splits=k, shuffle=True, random_state=534)
            in_split = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=8780)

            for dev_idx, test_idx in out_split.split(indices, targets):
                train_idx, val_idx = next(in_split.split(dev_idx, targets[dev_idx]))
                splits.append({"train": np.array(dev_idx[train_idx], dtype=np.int64), "val": np.array(dev_idx[val_idx], dtype=np.int64), "test": np.array(test_idx, dtype=np.int64)})

            with open(os.path.join(splitPath, f'{identifier}CT_{k}fold_val{ratio[0]}_stratify{stratify}.pkl'), 'wb') as f:
                pickle.dump(splits, f)

    elif mode == 'CV':
        if os.path.isfile(os.path.join(splitPath, f'{identifier}CV_{k}fold_test{ratio[1]}_stratify{stratify}.pkl')):
            with open(os.path.join(splitPath, f'{identifier}CV_{k}fold_test{ratio[1]}_stratify{stratify}.pkl'), 'rb') as f:
                return pickle.load(f)
        else:
            indices = np.arange(len(dataset))

            test_ratio = ratio[1]

            splitter_train_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=5345)
            in_split = StratifiedKFold(n_splits=k, shuffle=True, random_state=534)

            splits_t_t = splitter_train_test.split(indices, targets)
            train_idx, test_idx =  next(splits_t_t)

            for train_folds_idx, val_fold_idx in in_split.split(train_idx, targets[train_idx]):
                splits.append({"train": np.array(train_idx[train_folds_idx], dtype=np.int64), "val": np.array(train_idx[val_fold_idx], dtype=np.int64), "test": np.array(test_idx, dtype=np.int64)})

            with open(os.path.join(splitPath, f'{identifier}CV_{k}fold_test{ratio[1]}_stratify{stratify}.pkl'), 'wb') as f:
                pickle.dump(splits, f)
    else:
        raise Exception('Wrong split mode')
    
    return splits

def prepare_data(dataset_name, config, ratio, validation, kfold, stratify=True):
        if dataset_name in DATASETS:
            module = importlib.import_module("data.dataloaders")
            dataset_class = getattr(module, dataset_name)
            dataset = dataset_class(**config)
        else:
            raise Exception('Dataset name does not exist')

        splits = split_data(dataset, ratio, mode=validation, k=kfold, stratify=stratify)

        return dataset, splits

def initialize_datasets(dataset_configs, ratio=(0.2,0.1), validation='HO', kfold=1, stratify=True):
    for dataset in dataset_configs:
        for name, config in dataset.items():
            _, _ = prepare_data(name, config, ratio, validation, kfold, stratify=stratify)
    return None