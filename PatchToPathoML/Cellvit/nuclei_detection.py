from cellvit_inference import CellViTInferenceModule
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch


if __name__ == "__main__":
    model_path = Path("checkpoint/CellViT-SAM-H-x40.pth")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 

    inference_module=CellViTInferenceModule(
        model_path=model_path,
        device=device,
        enforce_mixed_precision=False
    )

    patch_path = Path("/home/peiliang/projects/BiomedParse/examples/target.png")
    img, predictions = inference_module.infer_patch(patch_path,retrieve_tokens=True)


    segmentation_masks, instance_types, tokens = inference_module.get_cell_predictions_with_tokens(
                    predictions, magnification=40)
    segmentation_masks=segmentation_masks[0]

    fig, ax = plt.subplots(1,3,figsize=(10,5))
    img = np.array(img)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('original image')

    for a in ax:
        a.axis('off')

    ax[1].set_title("segmentation")
    mask_temp = img.copy()
    mask_temp[segmentation_masks > 0] = [255, 0, 0]
    mask_temp[segmentation_masks == 0] = [0, 0, 0, ]
    ax[1].imshow(mask_temp, alpha=0.9)
    ax[1].imshow(img, cmap='gray', alpha=0.5)

    ids=np.unique(segmentation_masks)
    count=len(ids[ids>0])
    ax[2].set_title("classification")
    mask_temp=np.zeros_like(img)
    for id in ids:
        if id==0:
            continue
        mask_temp[segmentation_masks==id] = np.random.randint(0,255,3)
    ax[2].imshow(mask_temp,alpha=1)
    ax[2].imshow(img,cmap='gray', alpha=0.5)

    plt.show()