import os
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from functools import partial
from rdflib import Graph, URIRef, Namespace
from conch.open_clip_custom import create_model_from_pretrained


# Define the ConchNet class
class ConchNet(nn.Module):
    def __init__(self, original_net, num_classes):
        super(ConchNet, self).__init__()
        self.original_net = original_net
        self.fc = nn.Linear(512, num_classes)

        # Override the forward method to call encode_image with additional layers
        self.original_net.encode_image_with_fc = self.encode_image_with_fc

    def encode_image_with_fc(self, x):
        features = self.original_net.encode_image(
            x, proj_contrast=False, normalize=False
        )
        logits = self.fc(features)
        return logits

    def forward_threshold(self, x, threshold=1e6):
        features = self.original_net.encode_image(
            x, proj_contrast=False, normalize=False
        )
        x = x.clip(max=threshold)
        logits = self.fc(features)
        return logits

    def forward_fea(self, x):
        features = self.original_net.encode_image(
            x, proj_contrast=False, normalize=False
        )
        return features

    def forward(self, x, fea_if=False):
        if fea_if == False:
            return self.encode_image_with_fc(x)
        else:
            features = self.original_net.encode_image(
                x, proj_contrast=False, normalize=False
            )
            return self.fc(features), features

    def feature_list(self, x):
        features = self.original_net.encode_image(
            x, proj_contrast=False, normalize=False
        )
        return x, [features]


# Check if the CONCH model is available
os.environ["CONCH_CKPT_PATH"] = "" # /path/to/conch_model


def has_CONCH():
    HAS_CONCH = False
    CONCH_CKPT_PATH = "" # /path/to/conch_model
    try:
        if "CONCH_CKPT_PATH" not in os.environ:
            raise ValueError("CONCH_CKPT_PATH not set")
        HAS_CONCH = True
        CONCH_CKPT_PATH = os.environ["CONCH_CKPT_PATH"]
    except Exception as e:
        print(e)
        print("CONCH not installed or CONCH_CKPT_PATH not set")
    return HAS_CONCH, CONCH_CKPT_PATH


def net(image, conch_net):
    """
    Use deep learning model to extract features from the image.

    Args:
        image: The input image (NumPy array or PIL image).

    Returns:
        Feature vector: The extracted feature vector (list).
    """
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image_tensor = preprocess(image).unsqueeze(0)
    features = conch_net.forward_fea(image_tensor)
    return features.squeeze(0).tolist()


def parse_owl_file(owl_file):
    """
    Parse the OWL file to extract information on nuclei, tissue clusters, patches, and edges.

    Args:
        owl_file: Path to the OWL file.

    Returns:
        processed_nuclei: List of nuclei information.
        tissue_list: List of tissue cluster information.
        patch_path: Path to the Patch image.
        edges: List of edge information.
    """
    g = Graph()
    g.parse(owl_file, format="xml")

    patho = Namespace(
        "https://pathoml-1308125782.cos.ap-chengdu.myqcloud.com/PathoML.owl#"
    )

    processed_nuclei = []
    tissue_list = []
    patch_path = None
    edges = []

    # Extract Patch path
    for s, p, o in g.triples((None, patho.availability, None)):
        patch_path = str(o)

    # Extract nuclei information
    for s, p, o in g.triples((None, patho.segmentation, None)):
        if "Nucleus" in str(s):
            nucleus_data = {
                "nucleus_id": str(s).split("_")[-1],
                "contour": eval(o),
                "bbox": None,
                "centroid": None,
                "area": 0.0,  # Default value
                "shape_factor": 0.0,  # Default value
                "type": None,
                "tissue_id": None,
            }
            # Extract bbox
            for _, _, bbox in g.triples((s, patho.bbox, None)):
                nucleus_data["bbox"] = eval(bbox)
            # Extract centroid
            for _, _, centroid in g.triples((s, patho.centroid, None)):
                nucleus_data["centroid"] = eval(centroid)
            # Extract hasValue (e.g., Area, ShapeFactor)
            for quantification, _, value in g.triples((None, patho.hasValue, None)):
                if f"Area_{nucleus_data['nucleus_id']}" in str(quantification):
                    nucleus_data["area"] = float(value)
                elif f"ShapeFactor_{nucleus_data['nucleus_id']}" in str(quantification):
                    nucleus_data["shape_factor"] = float(value)
            # Extract type
            for _, _, cell_type in g.triples((s, patho.entityReference, None)):
                nucleus_data["type"] = str(cell_type).split("_")[-1]
            # Extract tissue_id
            for _, _, tissue_id in g.triples((s, patho.cellularComponentOf, None)):
                nucleus_data["tissue_id"] = str(tissue_id).split("_")[-1]

            processed_nuclei.append(nucleus_data)

    # Extract tissue cluster information
    for s, p, o in g.triples((None, patho.segmentation, None)):
        if "Tissue" in str(s):
            tissue_data = {
                "tissue_id": str(s).split("_")[-1],
                "contour": eval(o),
                "bbox": None,
            }
            # Extract bbox
            for _, _, bbox in g.triples((s, patho.bbox, None)):
                tissue_data["bbox"] = eval(bbox)
            tissue_list.append(tissue_data)

    # Extract edges between nuclei and tissue clusters
    for s, p, o in g.triples((None, patho.cellularComponentOf, None)):
        if "Nucleus" in str(s) and "Tissue" in str(o):
            edges.append((int(str(s).split("_")[-1]), int(str(o).split("_")[-1])))

    for s, p, o in g.triples((None, patho.hasCellularComponent, None)):
        if "Tissue" in str(s) and "Nucleus" in str(o):
            edges.append((int(str(o).split("_")[-1]), int(str(s).split("_")[-1])))

    return processed_nuclei, tissue_list, patch_path, edges


def construct_edges(processed_nuclei, tissue_list, edges):
    edge_index = []
    edge_index.extend(edges)

    patch_id = len(processed_nuclei) + len(tissue_list)
    for i in range(len(processed_nuclei)):
        edge_index.append((i, patch_id))
    for j in range(len(tissue_list)):
        edge_index.append((len(processed_nuclei) + j, patch_id))

    return torch.tensor(edge_index, dtype=torch.long).t()


def extract_patches(image, patch_size=(256, 256), stride=256):
    """
    Use a sliding window to extract multiple small patches from the image.
    """
    h, w = image.shape[:2]
    patches = []

    # Extract all patches using sliding window
    for i in range(0, h - patch_size[0] + 1, stride):
        for j in range(0, w - patch_size[1] + 1, stride):
            patch = image[i : i + patch_size[0], j : j + patch_size[1]]
            patches.append(patch)

    return patches

def extract_patch_features(image, conch_net, patch_size=(256, 256), stride=256):
    """
    Extract features for all patches.
    """
    patches = extract_patches(image, patch_size, stride)
    patch_features = []

    # Extract features for each patch
    for patch in patches:
        patch_feature = net(patch, conch_net)  # Use the model to extract patch features
        patch_features.append(patch_feature)

    # Convert to a format suitable for input to the Transformer
    return torch.tensor(patch_features, dtype=torch.float).unsqueeze(
        0
    )  # Shape: (1, num_patches, embed_dim)

def aggregate_patch_features(
    patch_image, conch_net, patch_size=(256, 256), stride=256
):
    """
    Aggregate all extracted patch features using self-attention.
    """
    # Extract patch features
    patch_features = extract_patch_features(patch_image, conch_net, patch_size, stride)[0]
    patch_features = patch_features.squeeze(0)

    # Aggregate features using self-attention
    aggregated_features = patch_features.sum(dim=1)

    return aggregated_features.squeeze(0)

def extract_node_features(processed_nuclei, tissue_list, patch_image, conch_net):
    nodes = []
    node_types = []

    # Get the dimensions of the image
    height, width = patch_image.shape[:2]

    # Process nucleus features
    for nucleus in processed_nuclei:
        bbox = nucleus["bbox"]
        if bbox and len(bbox) == 2:
            x_min, y_min, x_max, y_max = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]

            # Skip if bbox is completely outside the image boundaries
            if x_max <= 0 or x_min >= width or y_max <= 0 or y_min >= height:
                continue

            # Ensure cropping area is within the image bounds
            x_min = max(0, min(x_min, width))
            y_min = max(0, min(y_min, height))
            x_max = max(x_min, min(x_max, width))
            y_max = max(y_min, min(y_max, height))

            # Crop the image
            nucleus_image = patch_image[
                int(y_min) : int(y_max), int(x_min) : int(x_max)
            ]
            high_dim_features = net(nucleus_image, conch_net)

            feature_vector = [
                nucleus.get("centroid", [0.0, 0.0])[0],
                nucleus.get("centroid", [0.0, 0.0])[1],
                nucleus.get("area", 0.0),
                nucleus.get("shape_factor", 0.0),
                int(nucleus.get("type", 0)),
                *high_dim_features,
            ]
            nodes.append(feature_vector)
            node_types.append("nucleus")

    # Process tissue features
    for tissue in tissue_list:
        bbox = tissue["bbox"]
        if bbox and len(bbox) == 2:
            x_min, y_min, x_max, y_max = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]

            # Skip if bbox is completely outside the image boundaries
            if x_max <= 0 or x_min >= width or y_max <= 0 or y_min >= height:
                continue

            # Ensure cropping area is within the image bounds
            x_min = max(0, min(x_min, width))
            y_min = max(0, min(y_min, height))
            x_max = max(x_min, min(x_max, width))
            y_max = max(y_min, min(y_max, height))

            # Crop the image
            tissue_image = patch_image[int(y_min) : int(y_max), int(x_min) : int(x_max)]
            high_dim_features = net(tissue_image, conch_net)

            # Use -1 as placeholder
            feature_vector = [
                -1.0,  # Placeholder for centroid x (not applicable for tissue)
                -1.0,  # Placeholder for centroid y (not applicable for tissue)
                -1.0,  # Placeholder for area (not applicable for tissue)
                -1.0,  # Placeholder for shape factor (not applicable for tissue)
                -1,  # Placeholder for type (not applicable for tissue)
                *high_dim_features,  # High-dimensional features from the model
            ]
            nodes.append(feature_vector)
            node_types.append("tissue")

    # Extract patch features
    # patch_feature = net(patch_image, conch_net)
    aggregated_features = aggregate_patch_features(patch_image, conch_net)

    # Add placeholder to maintain consistency with nucleus and tissue node feature formats
    patch_feature_vector = [
        -1.0,  # Placeholder for centroid x (not applicable for patch)
        -1.0,  # Placeholder for centroid y (not applicable for patch)
        -1.0,  # Placeholder for area (not applicable for patch)
        -1.0,  # Placeholder for shape factor (not applicable for patch)
        -1,  # Placeholder for type (not applicable for patch)
        *aggregated_features,  # High-dimensional features from the model
    ]
    nodes.append(patch_feature_vector)
    node_types.append("patch")

    return nodes, node_types



def build_graph(owl_file, conch_net):
    processed_nuclei, tissue_list, patch_path, edges = parse_owl_file(owl_file)

    patch_image = np.array(Image.open(patch_path))

    nodes, node_types = extract_node_features(
        processed_nuclei, tissue_list, patch_image, conch_net
    )

    type_mapping = {"nucleus": 0, "tissue": 1, "patch": 2}
    node_types_encoded = torch.tensor(
        [type_mapping[node_type] for node_type in node_types], dtype=torch.long
    )

    edge_index = construct_edges(processed_nuclei, tissue_list, edges)

    x = torch.tensor(nodes, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    data.node_types = node_types_encoded

    return data

def process_owl_files_in_directory(
    root_dir, output_dir, conch_net, failed_files_log="failed_files.json"
):
    """
    Traverse all subdirectories in the folder and process the OWL files in each subdirectory.

    Args:
        root_dir: The root directory containing multiple subdirectories.
        output_dir: The directory to save the generated .pt files.
        conch_net: The ConchNet model instance.
        failed_files_log: The JSON filename to store paths of OWL files that failed to process.

    Returns:
        None
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a dictionary to map patch information
    mapping_dict = {"N": 0, "PB": 1, "UDH": 2, "FEA": 3, "ADH": 4, "DCIS": 5, "IC": 6}

    # Create a list to store paths of files that failed to process
    failed_files = []

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)

        if os.path.isdir(folder_path):
            # Assuming the OWL file has the same name as the folder
            owl_file = os.path.join(folder_path, f"{folder_name + '_instance'}.owl")

            if os.path.exists(owl_file):
                print(f"Processing: {owl_file}")

                try:
                    # Output file path
                    output_file = os.path.join(output_dir, f"{folder_name}.pt")

                    # Check if the .pt file already exists, and skip if it does
                    if os.path.exists(output_file):
                        print(f"Skipping {output_file}, already exists.")
                        continue

                    # Extract patch information source
                    # for bracs data
                    # patch_info = folder_name.split("_")[-2]
                    # patch_info_value = mapping_dict.get(patch_info, -1)  # Map to the corresponding integer value, default is -1

                    # for prcc data
                    patch_info_value = 2
                    # Call the build graph function
                    data = build_graph(owl_file, conch_net)

                    # Store patch information as label
                    data.y = torch.tensor([patch_info_value], dtype=torch.long)

                    # Save to the target folder
                    torch.save(data, output_file)
                    print(f"Saved: {output_file}")

                except Exception as e:
                    print(f"Failed to process {owl_file}: {e}")
                    failed_files.append({"owl_file": owl_file, "error": str(e)})

    # If there are any files that failed to process, save them to the JSON file
    if failed_files:
        with open(failed_files_log, "w") as log_file:
            json.dump(failed_files, log_file, indent=4)
        print(f"Failed to process some files. See {failed_files_log} for details.")

# Define the base part of the paths
root_directory_base = "/data_local/Pathoml/"  # Replace with the base path
output_directory_base = (
    "/data_local/Pathoml/pt_abmil_files/"  # Replace with the base path to save .pt files
)
failed_files_log_base = "failed_files_"  # Base name for the error file log

# Loop through [train, test, val]
for dataset in ["train", "test", "val"]:
    # Update paths for the current dataset
    root_directory = f"{root_directory_base}{dataset}"  # Concatenate path
    output_directory = f"{output_directory_base}{dataset}"  # Concatenate path
    failed_files_log = f"{failed_files_log_base}{dataset}.json"  # Concatenate log file path

    # Ensure the folder exists
    os.makedirs(output_directory, exist_ok=True)

    # Check if CONCH is available and load the model
    HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
    assert HAS_CONCH, "CONCH is not available"

    conch_net, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
    conch_net = ConchNet(conch_net, num_classes=2)


    # Process the files
    process_owl_files_in_directory(
        root_directory, output_directory, conch_net, failed_files_log
    )

