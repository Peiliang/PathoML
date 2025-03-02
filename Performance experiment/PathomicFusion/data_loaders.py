### data_loaders.py
import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn import preprocessing

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset  # For custom datasets
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data._utils.collate import default_collate

################
# Dataset Loader
################
class GraphDataset(Dataset):
    def __init__(self, folder_or_list):
        """
        Parameters:
          folder_or_list: This can either be a folder path (string), in which case the program will read all .pt files in that folder;
                           or a list of file paths, where each element is the full path to a .pt file.
        """
        if isinstance(folder_or_list, (str, bytes)):
            # If the input is a folder path, convert to an absolute path and read all .pt files
            folder_path = os.path.abspath(folder_or_list)
            all_paths = [
                os.path.join(folder_path, fname) 
                for fname in sorted(os.listdir(folder_path)) 
                if fname.lower().endswith('.pt')
            ]
        elif isinstance(folder_or_list, list):
            # Directly pass a list of file paths
            all_paths = folder_or_list
        else:
            raise TypeError("Input must be a folder path (str) or a list of file paths.")

        self.graph_paths = []
        # Define label mapping dictionary: 0,1,2 -> 0; 3,4 -> 1; 5,6 -> 2
        self.label_mapping = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
        
        # Filter out empty graphs (i.e., graphs with no nodes or edges)
        for path in all_paths:
            data = torch.load(path)
            if data.x.size(0) > 0 and (
                (data.edge_index.dim() == 1 and data.edge_index.numel() > 0) or 
                (data.edge_index.dim() > 1 and data.edge_index.size(1) > 0)
            ):
                self.graph_paths.append(path)
            else:
                print(f"Skipping empty graph: {path}")
                
    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, idx):
        data = torch.load(self.graph_paths[idx])
        num_nodes = data.x.size(0)
        if data.edge_index.dim() > 1:
            valid_edges = (data.edge_index[0] < num_nodes) & (data.edge_index[1] < num_nodes)
            data.edge_index = data.edge_index[:, valid_edges]
        
        # Apply label mapping, assuming data.y is a tensor containing a single label
        original_label = int(data.y.item())
        mapped_label = self.label_mapping.get(original_label, -1)
        if mapped_label == -1:
            raise ValueError(f"Label {original_label} not in mapping dictionary for graph at index {idx}")
        data.y = torch.tensor(mapped_label, dtype=torch.long)
        
        return data
    
def custom_collate(batch):
    # Filter out items that return None
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        raise ValueError("All graphs in the batch are empty!")
    return default_collate(batch)

class GraphDataset_PRCC(Dataset):
    def __init__(self, folder_or_list):
        """
        Parameters:
          folder_or_list: This can either be a folder path (string), in which case the program will read all .pt files in that folder;
                           or a list of file paths, where each element is the full path to a .pt file.
        """
        if isinstance(folder_or_list, (str, bytes)):
            # If the input is a folder path, convert to an absolute path and read all .pt files
            folder_path = os.path.abspath(folder_or_list)
            all_paths = [
                os.path.join(folder_path, fname) 
                for fname in sorted(os.listdir(folder_path)) 
                if fname.lower().endswith('.pt')
            ]
        elif isinstance(folder_or_list, list):
            # Directly pass a list of file paths
            all_paths = folder_or_list
        else:
            raise TypeError("Input must be a folder path (str) or a list of file paths.")

        self.graph_paths = []
        
        # Filter out empty graphs (i.e., graphs with no nodes or edges)
        for path in all_paths:
            data = torch.load(path)
            if data.x.size(0) > 0 and (
                (data.edge_index.dim() == 1 and data.edge_index.numel() > 0) or 
                (data.edge_index.dim() > 1 and data.edge_index.size(1) > 0)
            ):
                self.graph_paths.append(path)
            else:
                print(f"Skipping empty graph: {path}")
                
    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, idx):
        data = torch.load(self.graph_paths[idx])
        num_nodes = data.x.size(0)
        if data.edge_index.dim() > 1:
            valid_edges = (data.edge_index[0] < num_nodes) & (data.edge_index[1] < num_nodes)
            data.edge_index = data.edge_index[:, valid_edges]
        
        # Ensure label is of LongTensor type
        data.y = torch.tensor(data.y, dtype=torch.long)
        data.y = data.y - 1
        
        return data

def custom_collate(batch):
    # Filter out items that return None
    batch = [b for b in batch if b is not None]
    if len(batch == 0):
        raise ValueError("All graphs in the batch are empty!")
    return default_collate(batch)
