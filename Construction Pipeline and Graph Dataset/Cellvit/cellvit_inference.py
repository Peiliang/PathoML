# cellvit_inference.py
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Union
from torchvision import transforms as T
from PIL import Image
from typing import List, Tuple, Union

from .utils.tools import unflatten_dict, get_size_of_dict
from .models.cell_segmentation.cellvit import (
    CellViT,
    CellViT256,
    CellViTSAM,
)
from .models.cell_segmentation.cellvit_shared import (
    CellViT256Shared,
    CellViTSAMShared,
    CellViTShared,
)

class CellViTInferenceModule:
    def __init__(
        self, 
        model_path: Union[Path, str],
        device: str = "cuda:0",
        enforce_mixed_precision: bool = False
    ):
        """
        初始化推理模块
        Args:
            model_path: checkpoint 路径
            device: 使用的设备, e.g. "cuda:0" or "cpu"
            enforce_mixed_precision: 是否强制混合精度推理
        """
        self.model_path = Path(model_path)
        self.device = device

        # 1. 加载 checkpoint
        checkpoint = torch.load(self.model_path, map_location="cpu")

        # 2. 读取配置
        self.run_conf = unflatten_dict(checkpoint["config"], ".")
        model_type = checkpoint["arch"]

        # 3. 实例化模型
        self.model = self._get_model(model_type)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval().to(self.device)

        # 4. 加载推理 transform
        self.inference_transforms = self._load_inference_transforms()

        # 5. AMP 设置
        if enforce_mixed_precision:
            self.mixed_precision = True
        else:
            self.mixed_precision = self.run_conf["training"].get("mixed_precision", False)

    def _get_model(self, model_type: str):
        """根据 arch 名称返回正确的 CellViT 模型类实例。"""
        # 示例: 你也可以根据实际需求只保留某个模型
        if model_type == "CellViT":
            model = CellViT(
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                embed_dim=self.run_conf["model"]["embed_dim"],
                input_channels=self.run_conf["model"].get("input_channels", 3),
                depth=self.run_conf["model"]["depth"],
                num_heads=self.run_conf["model"]["num_heads"],
                extract_layers=self.run_conf["model"]["extract_layers"],
                regression_loss=self.run_conf["model"].get("regression_loss", False),
            )
        elif model_type == "CellViT256":
            model = CellViT256(
                model256_path=None,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                regression_loss=self.run_conf["model"].get("regression_loss", False),
            )
        elif model_type == "CellViTSAM":
            model = CellViTSAM(
                model_path=None,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                vit_structure=self.run_conf["model"]["backbone"],
                regression_loss=self.run_conf["model"].get("regression_loss", False),
            )
        else:
            raise NotImplementedError(f"Unsupported model_type: {model_type}")
        return model

    def _load_inference_transforms(self):
        """加载推理所需 Transform (ToTensor + Normalize)"""
        transform_settings = self.run_conf["transformations"]
        if "normalize" in transform_settings:
            mean = transform_settings["normalize"].get("mean", (0.5, 0.5, 0.5))
            std = transform_settings["normalize"].get("std", (0.5, 0.5, 0.5))
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    
    def get_cell_predictions_with_tokens(self,
        predictions: dict, magnification: int = 40
    ) -> Tuple[List[dict], torch.Tensor]:
        """Take the raw predictions, apply softmax and calculate type instances

        Args:
            predictions (dict): Network predictions with tokens. Keys:
            magnification (int, optional): WSI magnification. Defaults to 40.

        Returns:
            Tuple[List[dict], torch.Tensor]:
                * List[dict]: List with a dictionary for each batch element with cell seg results
                    Contains bbox, contour, 2D-position, type and type_prob for each cell
                * List[dict]: Network tokens on cpu device with shape (batch_size, num_tokens_h, num_tokens_w, embd_dim)
        """
        predictions["nuclei_binary_map"] = F.softmax(
            predictions["nuclei_binary_map"], dim=1
        )  # shape: (batch_size, 2, H, W)
        predictions["nuclei_type_map"] = F.softmax(
            predictions["nuclei_type_map"], dim=1
        )  # shape: (batch_size, num_nuclei_classes, H, W)
        # get the instance types
        (
            segmentation_masks,
            instance_types,
        ) = self.model.calculate_instance_map(predictions, magnification=magnification)

        tokens = predictions["tokens"].to("cpu")

        return segmentation_masks,instance_types, tokens

    @torch.no_grad()
    def infer_patch(self, patch_img_path: Union[Path, str], retrieve_tokens=False):
        """
        对单张 patch 进行推理，返回网络输出 (dict)。

        Args:
            patch_img_path: patch 图像文件的路径
            retrieve_tokens: 是否返回 tokens
        Returns:
            out_dict: 包含如下键值的字典：
                - "nuclei_binary_map" (B,2,H,W) ...
                - "nuclei_type_map"   (B,K,H,W) ...
                - "hv_map"           (B,2,H,W) ...
                - "tissue_types"     (B, T)    ...
                - "tokens"           (可选)    ...
                - "regression_map"   (可选)    ...
        """

        # 1. 加载图像并 transform
        img = Image.open(patch_img_path)
        width, height = img.size  # (2000, 2000)
        crop_size = 1024
        left   = (width  - crop_size) // 2
        top    = (height - crop_size) // 2
        right  = left + crop_size
        bottom = top  + crop_size
        crop_img = img.crop((left, top, right, bottom))
        crop_img.save("patch_crop_center_1024.png")

        img = crop_img.convert("RGB")
        img_tensor = self.inference_transforms(img)  # C,H,W
        # 由于 model.forward 默认是(B,C,H,W)，所以要在前面增加 batch 维度
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        print("image shape:",img_tensor.shape[-2])
        print("image shape:",img_tensor.shape[-1])

        #2. 若启用 AMP，则用 autocast
        if self.mixed_precision:
            with torch.autocast(device_type=self.device.split(":")[0], dtype=torch.float16):
                out_dict = self.model.forward(img_tensor, retrieve_tokens=retrieve_tokens)
        else:
            out_dict = self.model.forward(img_tensor, retrieve_tokens=retrieve_tokens)

        # 3. 返回模型原始输出，你可以在这里进行更多后处理
        return img, out_dict
