import os
import cv2
import json
import torch
import argparse
import numpy as np
from torch import nn
from PIL import Image
import huggingface_hub
import matplotlib.pyplot as plt
from Cellvit.cellvit_inference import CellViTInferenceModule
from inference_utils.inference import interactive_infer_image
from inference_utils.output_processing import check_mask_stats
from inference_utils.processing_utils import get_instances
from inference_utils.processing_utils import read_rgb
from shapely.geometry import Point, Polygon
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from utilities.distributed import init_distributed
from modeling import build_model
from modeling.BaseModel import BaseModel
from pathlib import Path
from rdflib import Graph, URIRef, Namespace, Literal, RDF
from rdflib.namespace import XSD, OWL, RDFS
from functools import partial
from conch.open_clip_custom import create_model_from_pretrained

def plot_segmentation_masks(original_image, segmentation_masks, texts, img_output_dir):
    """plot_segmentation_masks function: Add description here."""
    ''' Plot a list of segmentation mask over an image. '''
    original_image = original_image[:, :, :3]
    fig, ax = plt.subplots(1, len(segmentation_masks) + 1, figsize=(10, 5))
    ax[0].imshow(original_image, cmap='gray')
    ax[0].set_title('Original Image')
    # grid off
    for a in ax:
        a.axis('off')

    for i, mask in enumerate(segmentation_masks):
        
        ax[i+1].set_title(texts[i])
        mask_temp = original_image.copy()
        mask_temp[mask > 0.5] = [255, 0, 0]
        mask_temp[mask <= 0.5] = [0, 0, 0]
        ax[i+1].imshow(mask_temp, alpha=0.9)
        ax[i+1].imshow(original_image, cmap='gray', alpha=0.5)
        
    
    plt.savefig(img_output_dir + '/segmentation.png')

# ### Utility Functions
def inference_rgb(model, file_path, text_prompts, img_output_dir):
    image = read_rgb(file_path)
    
    pred_mask, contour_results = interactive_infer_image(model, Image.fromarray(image), text_prompts)

    # Plot feature over image
    plot_segmentation_masks(image, pred_mask, text_prompts, img_output_dir)
    
    return image, pred_mask, contour_results

def visualize_center_bboxes(image, contour_results, img_output_dir):
    """visualize_center_bboxes function: Add description here."""
    """
    Visualize bounding boxes and contours in bbox format.

    Args:
        image: Original image (NumPy array or PIL image).
        contour_results: List of results containing contours and bounding boxes in center coordinate format.
    """
    # If the input is a PIL image, convert it to NumPy format
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    # Copy the image for drawing bounding boxes and contours
    vis_image = image.copy()

    for region in contour_results:
        # Get bounding box in center coordinate format
        x_min, y_min, x_max, y_max = region["bbox"][0][0], region["bbox"][0][1], region["bbox"][1][0], region["bbox"][1][1]

        # Draw bounding box
        cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # Draw contour
        cv2.drawContours(vis_image, [region["contour"]], -1, (0, 255, 0), 2)

        # Display region ID above the bounding box
        cv2.putText(
            vis_image,
            f"ID: {region['tissue_id']}",
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.savefig(img_output_dir + '/visualize_center_bboxes.png')


"""plot_instance_segmentation_masks function: Add description here."""
def plot_instance_segmentation_masks(original_image, segmentation_masks, img_output_dir, text_prompt=None):
    ''' Plot a list of segmentation mask over an image. '''
    original_image = original_image[:, :, :3]
    fig, ax = plt.subplots(1, len(segmentation_masks) + 1, figsize=(10, 5))
    ax[0].imshow(original_image, cmap='gray')
    ax[0].set_title('Original Image')
    # grid off
    for a in ax:
        a.axis('off')
        
    instance_masks = [get_instances(1*(mask>0.5)) for mask in segmentation_masks]
    
    mask_names = [f'Mask {i+1}' for i in range(len(segmentation_masks))]
    if text_prompt:
        mask_names = text_prompt
        for i in range(len(mask_names)):
            mask_names[i] = mask_names[i].strip()

    for i, mask in enumerate(instance_masks):
        ins_ids = np.unique(mask)
        count = len(ins_ids[ins_ids > 0])
        
        ax[i+1].set_title(f'{mask_names[i]} ({count})')
        mask_temp = np.zeros_like(original_image)
        for ins_id in ins_ids:
            if ins_id == 0:
                continue
            mask_temp[mask == ins_id] = np.random.randint(0, 255, 3)
            if ins_id == 1:
                mask_temp[mask == ins_id] = [255, 0, 0]
        
        ax[i+1].imshow(mask_temp, alpha=1)
        ax[i+1].imshow(original_image, cmap='gray', alpha=0.5)
        
    plt.savefig(img_output_dir + '/instance_segmentation.png')

"""process_nuclei_and_tissue function: Add description here."""
def process_nuclei_and_tissue(nucleus_raw_list, tissue_list):
    """
    Process nucleus and tissue lists, mapping nuclei to tissues and converting data formats.

    Args:
        nucleus_raw_list: 
            [{1: {'bbox': np_array([[rmin,cmin],[rmax,cmax]]),
                'centroid': array([x, y]),
                'contour':  [(x1,y1),(x2,y2),...(xn,yn)],
                'type_prob': num,
                'type': num},
            ...]
        tissue_list: 
            [{
              "tissue_id": index,
              "bbox": np_array([[rmin,cmin],[rmax,cmax]]),
              "contour": [(x1,y1),(x2,y2),...(xn,yn)],
              "tissue_type": str(tissue_type_name),
              "nucleus_ids": []  # Initialize as empty
            }, ...]

    Returns:
        processed_nuclei: Converted nucleus list.
        updated_tissues: Updated tissue list.
    """
    # Initialize a new nucleus list
    processed_nuclei = []

    # Iterate through each nucleus
    nucleus_raw_list = nucleus_raw_list[0]
    for nucleus_id, nucleus_data in nucleus_raw_list.items():
        # Get the centroid and contour of the nucleus
        centroid = Point(nucleus_data["centroid"])
        contour = np.array(nucleus_data["contour"])

        # Initialize nucleus information
        nucleus_info = {
            "nucleus_id": nucleus_id,
            "bbox": nucleus_data["bbox"],
            "centrioid": list(nucleus_data["centroid"]),
            "contour": [tuple(point) for point in nucleus_data["contour"]],
            "cell_type": f"Type_{nucleus_data['type']}",  # Assume direct mapping of cell type
            "tissue_id": None,  # Initialize as None
            "area": None,  # Initialize as None
            "roundness": None,  # Initialize as None
            "shape factor": None  # Initialize as None
        }

        # Compute area, roundness, and shape factor
        nucleus_info["area"] = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        nucleus_info["roundness"] = (
            4 * np.pi * nucleus_info["area"] / (perimeter ** 2) if perimeter > 0 else 0
        )
        rect = cv2.minAreaRect(contour)
        rect_area = rect[1][0] * rect[1][1]
        nucleus_info["shape factor"] = (
            nucleus_info["area"] / rect_area if rect_area > 0 else 0
        )

        # Determine which tissue the nucleus belongs to
        for tissue in tissue_list:
            try:
                # Create polygon
                tissue_polygon = Polygon(tissue["contour"].reshape(-1, 2).tolist())
            except Exception as e:
                print(f"Error creating Polygon for Tissue ID {tissue['tissue_id']}: {e}")
                continue

            if tissue_polygon.contains(centroid):
                nucleus_info["tissue_id"] = tissue["tissue_id"]
                tissue["nucleus_ids"].append(nucleus_id)
                break

        # Add nucleus information to the new list
        processed_nuclei.append(nucleus_info)

    return processed_nuclei, tissue_list

"""process_nuclei_and_tissue function: Add description here."""
def process_nuclei_and_tissue(nucleus_raw_list, tissue_list):
    """
    Process nucleus and tissue lists, mapping nuclei to tissues and converting data formats.

    Args:
        nucleus_raw_list: 
            [{1: {'bbox': np_array([[rmin,cmin],[rmax,cmax]]),
                'centroid': array([x, y]),
                'contour':  [(x1,y1),(x2,y2),...(xn,yn)],
                'type_prob': num,
                'type': num},
            ...]
        tissue_list: 
            [{
              "tissue_id": index,
              "bbox": np_array([[rmin,cmin],[rmax,cmax]]),
              "contour": [(x1,y1),(x2,y2),...(xn,yn)],
              "tissue_type": str(tissue_type_name),
              "nucleus_ids": []  # Initialize as empty
            }, ...]

    Returns:
        processed_nuclei: Converted nucleus list.
        updated_tissues: Updated tissue list.
    """
    # Initialize a new nucleus list
    processed_nuclei = []

    # Iterate through each nucleus
    nucleus_raw_list = nucleus_raw_list[0]
    for nucleus_id, nucleus_data in nucleus_raw_list.items():
        # Get the centroid and contour of the nucleus
        centroid = Point(nucleus_data["centroid"])
        contour = np.array(nucleus_data["contour"])

        # Initialize nucleus information
        nucleus_info = {
            "nucleus_id": nucleus_id,
            "bbox": nucleus_data["bbox"],
            "centrioid": list(nucleus_data["centroid"]),
            "contour": [tuple(point) for point in nucleus_data["contour"]],
            "cell_type": f"Type_{nucleus_data['type']}",  # Assume direct mapping of cell type
            "tissue_id": None,  # Initialize as None
            "area": None,  # Initialize as None
            "roundness": None,  # Initialize as None
            "shape factor": None  # Initialize as None
        }

        # Compute area, roundness, and shape factor
        nucleus_info["area"] = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        nucleus_info["roundness"] = (
            4 * np.pi * nucleus_info["area"] / (perimeter ** 2) if perimeter > 0 else 0
        )
        rect = cv2.minAreaRect(contour)
        rect_area = rect[1][0] * rect[1][1]
        nucleus_info["shape factor"] = (
            nucleus_info["area"] / rect_area if rect_area > 0 else 0
        )

        # Determine which tissue the nucleus belongs to
        for tissue in tissue_list:
            try:
                # Create polygon
                tissue_polygon = Polygon(tissue["contour"].reshape(-1, 2).tolist())
            except Exception as e:
                print(f"Error creating Polygon for Tissue ID {tissue['tissue_id']}: {e}")
                continue

            if tissue_polygon.contains(centroid):
                nucleus_info["tissue_id"] = tissue["tissue_id"]
                tissue["nucleus_ids"].append(nucleus_id)
                break

        # Add nucleus information to the new list
        processed_nuclei.append(nucleus_info)

    return processed_nuclei, tissue_list

"""visualize_nuclei_and_tissue function: Add description here."""
def visualize_nuclei_and_tissue(image, processed_nuclei, tissue_list, img_output_dir):
    """
    Visualize nucleus and tissue information.

    Args:
        image: Original image, a NumPy array of shape (H, W, 3).
        processed_nuclei: List of processed nucleus information.
        tissue_list: List of tissue regions.

    Returns:
        The generated visualization image.
    """
    # Create a copy for drawing
    vis_image = image.copy()

    # Draw tissue contours
    for tissue in tissue_list:
        # Draw contour
        tissue_contour = np.array(tissue["contour"], dtype=np.int32) # (74, 1, 2)
        tissue_contour = np.squeeze(tissue_contour, axis=1) # (74, 2)
        cv2.polylines(vis_image, [tissue_contour], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Display Tissue ID at the contour center
        if len(tissue_contour) > 0:
            center_x = int(np.mean(tissue_contour[:, 0]))
            center_y = int(np.mean(tissue_contour[:, 1]))
            cv2.putText(
                vis_image,
                f"TID: {tissue['tissue_id']}",
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    # Draw nucleus information
    for nucleus in processed_nuclei:
        # Draw contour
        nucleus_contour = np.array(nucleus["contour"], dtype=np.int32)
        cv2.polylines(vis_image, [nucleus_contour], isClosed=True, color=(255, 0, 0), thickness=2)

        # Mark centroid
        centroid = nucleus["centrioid"]
        cv2.circle(vis_image, (int(centroid[0]), int(centroid[1])), radius=3, color=(255, 0, 0), thickness=-1)

    # Display the image
    plt.figure(figsize=(12, 12))
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))  # Convert to RGB format for Matplotlib
    plt.axis("off")
    plt.title("Nuclei and Tissue Visualization")
    plt.savefig(img_output_dir + '/nuclei_and_tissue.png')

def numpy_to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def convert_to_standard_type(data):
    if isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, list):
        return [convert_to_standard_type(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_to_standard_type(item) for item in data)
    else:
        return data
    
def create_pathoml_instance_with_provenance(
    output_file, 
    instance_onto_uri, 
    patch_name, 
    patch_availability, 
    provenance_data, 
    processed_nuclei, 
    updated_tissues
):
    """
    Create a PathoML instance file and generate an Owl file containing data sources, tissue types, cell types, pathological features, etc.

    :param output_file: Output Owl file path
    :param instance_onto_uri: Ontology instance URI (e.g., http://www.pathoml.org/pathoml/70/8866654)
    :param patch_name: Display name of the patch file
    :param patch_availability: Path to the patch file
    :param provenance_data: Data source information, including website link, dataset description, etc.
    :param processed_nuclei: Detailed information of nuclei
    :param updated_tissues: Detailed information of tissue clusters
    """

    # Initialize graph
    g = Graph()

    # Declare the URI of our knowledge record (ABox)
    instance_onto = URIRef(instance_onto_uri)

    # Remote PathoML Schema (.owl) address
    pathoml_schema_uri = "https://pathoml-1308125782.cos.ap-chengdu.myqcloud.com/PathoML.owl"

    # Bind common prefixes: OWL / RDFS / XSD
    g.bind('owl', OWL)
    g.bind('rdfs', RDFS)
    g.bind('xsd', XSD)

    # Assign a prefix (e.g., 'patho') to the PathoML namespace
    patho = Namespace("https://pathoml-1308125782.cos.ap-chengdu.myqcloud.com/PathoML.owl#")
    g.bind('patho', patho)

    # Define the namespace for our local instance
    base_ns = Namespace(f"{instance_onto_uri}#")
    g.bind('', base_ns)

    # Declare ontology node in the graph and add owl:imports pointing to the remote PathoML schema
    g.add((instance_onto, RDF.type, OWL.Ontology))
    g.add((instance_onto, OWL.imports, URIRef(pathoml_schema_uri)))

    # -----------------------------------------------------------------
    # Provenance Instance: Data source information
    # -----------------------------------------------------------------
    provenance = base_ns["Provenance"]
    g.add((provenance, RDF.type, patho.Provenance))
    
    # Store website link and dataset description
    website = provenance_data.get('website', '')
    dataset_description = provenance_data.get('dataset_description', '')
    
    g.add((provenance, patho.comment, Literal(website, datatype=XSD.string)))
    g.add((provenance, patho.comment, Literal(dataset_description, datatype=XSD.string)))

    # -----------------------------------------------------------------
    # Patch Instance
    # -----------------------------------------------------------------
    patch = base_ns["patch"]
    g.add((patch, RDF.type, patho.HE_Patch))
    g.add((patch, patho.displayName, Literal(patch_name, datatype=XSD.string)))
    g.add((patch, patho.availability, Literal(patch_availability, datatype=XSD.string)))

    # -----------------------------------------------------------------
    # UnificationXref Instance
    # -----------------------------------------------------------------
    # Tumor
    tumor_xref = base_ns["UnificationXref_Tumor"]
    g.add((tumor_xref, RDF.type, patho.UnificationXref))
    g.add((tumor_xref, patho.uri, Literal("http://purl.obolibrary.org/obo/NCIT_C18009", datatype=XSD.string)))
    
    # Nucleus
    nucleus_xref = base_ns["UnificationXref_Nucleus"]
    g.add((nucleus_xref, RDF.type, patho.UnificationXref))
    g.add((nucleus_xref, patho.uri, Literal("http://purl.obolibrary.org/obo/GO_0005634", datatype=XSD.string)))
    
    # Size
    size_xref = base_ns["UnificationXref_Size"]
    g.add((size_xref, RDF.type, patho.UnificationXref))
    g.add((size_xref, patho.uri, Literal("http://purl.obolibrary.org/obo/PATO_0002057", datatype=XSD.string)))
    
    # Area
    area_xref = base_ns["UnificationXref_Area"]
    g.add((area_xref, RDF.type, patho.UnificationXref))
    g.add((area_xref, patho.uri, Literal("http://purl.obolibrary.org/obo/PATO_0001323", datatype=XSD.string)))
    
    # Shape
    shape_xref = base_ns["UnificationXref_Shape"]
    g.add((shape_xref, RDF.type, patho.UnificationXref))
    g.add((shape_xref, patho.uri, Literal("http://purl.obolibrary.org/obo/PATO_0005020", datatype=XSD.string)))

    # -----------------------------------------------------------------
    # UnitVocabulary Instance
    # -----------------------------------------------------------------
    square_micrometer = base_ns["square_micrometer"]
    g.add((square_micrometer, RDF.type, patho.UnitVocabulary))
    g.add((square_micrometer, patho.uri, Literal("http://purl.obolibrary.org/obo/UO_0010001", datatype=XSD.string)))

    # -----------------------------------------------------------------
    # Tissue and Nucleus Type Instances
    # -----------------------------------------------------------------
    tissue_types = {t['tissue_type'] for t in updated_tissues}
    cell_types = {n['cell_type'] for n in processed_nuclei}
    
    # Tissue Types
    for tissue_type in tissue_types:
        tissue_type = tissue_type.replace(" ", "")
        tissue_ref = base_ns[f"TissueRef_{tissue_type}"]
        g.add((tissue_ref, RDF.type, patho.Other_AnatomicalEntityReference))
        g.add((tissue_ref, patho.displayName, Literal(tissue_type, datatype=XSD.string)))
        g.add((tissue_ref, patho.hasXref, tumor_xref))  # Associate TissueRef with Tumor Xref

    # Cell Types
    for cell_type in cell_types:
        cellular_component_ref = base_ns[f"CellularComponentRef_{cell_type}"]
        g.add((cellular_component_ref, RDF.type, patho.CellularComponentReference))
        g.add((cellular_component_ref, patho.displayName, Literal(cell_type, datatype=XSD.string)))
        g.add((cellular_component_ref, patho.hasXref, nucleus_xref))  # Associate CellularComponentRef with Nucleus Xref

    # Serialize and save the Owl file
    g.serialize(destination=output_file, format='pretty-xml', xml_base=instance_onto_uri, short_names=True)

    print(f"PathoML instance has been generated: {output_file}")


def process_images(is_folder, image_path, text_prompt, model, inference_module, provenance_data):
    """
    Process images based on whether input is a folder or a single image.

    Args:
        is_folder (bool): True if the input is a folder, False if it's a single image.
        image_path (str): Path to the image or folder.
        text_prompt (list): List of text prompts for processing.
    """
    if is_folder:
        # Collect all image files in the folder
        image_files = [
            os.path.join(image_path, f)
            for f in os.listdir(image_path)
            if f.endswith(('.png', '.jpg', '.jpeg', '.dcm'))
        ]
    else:
        # Single image file
        image_files = [image_path]

    # Process each image
    for img_path in image_files:
        print(f"Processing: {img_path}")
        # Extract the image filename without extension
        img_name = Path(img_path).stem
        # Create a subdirectory for this image
        output_dir = '/data_local/Pathoml/train'
        img_output_dir = os.path.join(output_dir, img_name)
        os.makedirs(img_output_dir, exist_ok=True)

        # Check if the .owl file already exists, if so, skip this image
        owl_file = f"{img_output_dir}/{img_name}_instance.owl"
        if os.path.exists(owl_file):
            print(f"Skipping {img_name} as the .owl file already exists.")
            continue

        image, pred_mask, contour_results = inference_rgb(model, img_path, text_prompt, img_output_dir)

        # Calculate adjusted p-value for each mask
        for i in range(len(pred_mask)):
            adj_pvalue = check_mask_stats(image, pred_mask[i] * 255, 'Pathology', text_prompt[i])
            print(f'{text_prompt[i]} P-value: {adj_pvalue}')

        # Visualize results
        visualize_center_bboxes(image, contour_results, img_output_dir)
        plot_instance_segmentation_masks(image, pred_mask, img_output_dir, text_prompt)
        
        # Cell part
        img, predictions = inference_module.infer_patch(img_path, retrieve_tokens=True)
        segmentation_masks, instance_types, tokens = inference_module.get_cell_predictions_with_tokens(
                predictions, magnification=40)
        processed_nuclei, tissue_list = process_nuclei_and_tissue(instance_types, contour_results)
        visualize_nuclei_and_tissue(image, processed_nuclei, tissue_list, img_output_dir)
        
        # Save processed_nuclei as a JSON file
        with open(img_output_dir + "/processed_nuclei.json", "w") as nuclei_file:
            json.dump(processed_nuclei, nuclei_file, indent=4, default=numpy_to_python)
        print("processed_nuclei saved as processed_nuclei.json")
        
        # Save tissue_list as a JSON file
        with open(img_output_dir + "/tissue_list.json", "w") as tissue_file:
            json.dump(tissue_list, tissue_file, indent=4, default=numpy_to_python)
        print("tissue_list saved as tissue_list.json")
        
        # Create .owl file
        create_pathoml_instance_with_provenance(
            output_file=owl_file,
            instance_onto_uri=f"http://www.pathoml.org/pathoml/{img_name}",
            patch_name=img_name,
            patch_availability=img_path,
            provenance_data=provenance_data,
            processed_nuclei=processed_nuclei,
            updated_tissues=tissue_list
        )

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process images with text prompts.")
    parser.add_argument('--is_folder', type=bool, required=True, help="Set to True if input is a folder, False if it's a single image.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the image or folder to process.")
    parser.add_argument('--text_prompt', nargs='+', required=True, help="List of text prompts for image processing.")

    # Parse arguments
    args = parser.parse_args()

    HF_TOKEN = 'your_token'
    huggingface_hub.login(HF_TOKEN)
    
    # Model Setup  
    # Build model config
    opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
    opt = init_distributed(opt)

    # Load model from pretrained weights
    pretrained_pth = 'hf_hub:microsoft/BiomedParse'

    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)
    
    # CellVit 
    model_path = Path("/home/pxb/code/PathoML/PatchToPathoML/Cellvit/checkpoint/CellViT-SAM-H-x40.pth")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
    inference_module=CellViTInferenceModule(
        model_path=model_path,
        device=device,
        enforce_mixed_precision=False
    )
    
    # Passing provenance_data dictionary ---- Example for BRACS
    provenance_data = {
        'website': 'https://www.bracs.icar.cnr.it/',
        'dataset_description': 'A  new dataset of hematoxylin and eosin histopathological images for automated detection/classification of breast tmors .'
    }
    
    # Call the processing function
    process_images(args.is_folder, args.image_path, args.text_prompt, model, inference_module, provenance_data)

if __name__ == "__main__":
    main()
    

 