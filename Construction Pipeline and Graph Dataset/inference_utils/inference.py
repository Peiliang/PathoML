import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2
#from utils.visualizer import Visualizer
# from detectron2.utils.colormap import random_color
# from detectron2.data import MetadataCatalog
# from detectron2.structures import BitMasks
from modeling.language.loss import vl_similarity
from utilities.constants import BIOMED_CLASSES
#from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

# import cv2
# import os
# import glob
# import subprocess
from PIL import Image
import random

t = []
t.append(transforms.Resize((1024, 1024), interpolation=Image.BICUBIC))
transform = transforms.Compose(t)
#metadata = MetadataCatalog.get('coco_2017_train_panoptic')
all_classes = ['background'] + [name.replace('-other','').replace('-merged','') 
                                for name in BIOMED_CLASSES] + ["others"]
# colors_list = [(np.array(color['color'])/255).tolist() for color in COCO_CATEGORIES] + [[1, 1, 1]]

# use color list from matplotlib
import matplotlib.colors as mcolors
colors = dict(mcolors.TABLEAU_COLORS, **mcolors.BASE_COLORS)
colors_list = [list(colors.values())[i] for i in range(16)] 

from .output_processing import mask_stats, combine_masks
    

@torch.no_grad()
def interactive_infer_image(model, image, prompts):

    image_resize = transform(image)
    width = image.size[0]
    height = image.size[1]
    image_resize = np.asarray(image_resize)
    image = torch.from_numpy(image_resize.copy()).permute(2,0,1).cuda()

    data = {"image": image, 'text': prompts, "height": height, "width": width}
    
    # inistalize task
    model.model.task_switch['spatial'] = False
    model.model.task_switch['visual'] = False
    model.model.task_switch['grounding'] = True
    model.model.task_switch['audio'] = False
    model.model.task_switch['grounding'] = True


    batch_inputs = [data]
    results,image_size,extra = model.model.evaluate_demo(batch_inputs)

    pred_masks = results['pred_masks'][0]
    v_emb = results['pred_captions'][0]
    t_emb = extra['grounding_class']

    t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
    v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)

    temperature = model.model.sem_seg_head.predictor.lang_encoder.logit_scale
    out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)
    
    matched_id = out_prob.max(0)[1]
    pred_masks_pos = pred_masks[matched_id, :, :]
    
    pred_class = results['pred_logits'][0][matched_id].max(dim=-1)[1]

    # interpolate mask to ori size
    pred_mask_prob = F.interpolate(pred_masks_pos[None,], (data['height'], data['width']), 
                                   mode='bilinear')[0,:,:data['height'],:data['width']].sigmoid().cpu().numpy()
    pred_masks_pos = (1*(pred_mask_prob > 0.5)).astype(np.uint8)
    
    # print("pred_masks_pos shape:", pred_masks_pos.shape)
    # print("pred_masks_pos unique values:", np.unique(pred_masks_pos))

    pred_masks_pos = pred_masks_pos.squeeze(0)  # 去掉第一个维度
    # Find contours
    contours, _ = cv2.findContours(
        pred_masks_pos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Store results
    contour_results = []
    tissue_num = 1
    min_area = 100  # 面积最小值
    min_perimeter = 80  # 周长最小值
    for i, contour in enumerate(contours):
        
        # 根据面积和周长过滤轮廓
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)
        if area < min_area or perimeter < min_perimeter:  # 筛选条件
            continue
        
        # Create a mask for this specific contour
        single_mask = np.zeros_like(pred_masks_pos, dtype=np.uint8)
        cv2.drawContours(single_mask, [contour], -1, color=1, thickness=-1)

        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)  # Bounding box: (x, y, width, height)
        # bbox = (x + w / 2, y + h / 2, w, h) # (x_center, y_center, width, height)
        bbox = [[x, y], [x + w, y + h]]  # 生成标准格式: [[x_min, y_min], [x_max, y_max]]
        
        

        # Append the result
        contour_results.append(
            {
                "tissue_id": tissue_num,  # Unique ID for each contour
                "contour": contour,  # The contour points
                "mask": single_mask,  # Binary mask for the region
                "bbox": bbox,  # Bounding box
                "tissue_type":prompts[0], # 
                "nucleus_ids": []  # 初始化为空
            }
        )
        tissue_num += 1
    
    return pred_mask_prob, contour_results



# def interactive_infer_panoptic_biomedseg(model, image, tasks, reftxt=None):
#     image_ori = transform(image)
#     #mask_ori = image['mask']
#     width = image_ori.size[0]
#     height = image_ori.size[1]
#     image_ori = np.asarray(image_ori)
#     visual = Visualizer(image_ori, metadata=metadata)
#     images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()

#     data = {"image": images, "height": height, "width": width}
#     if len(tasks) == 0:
#         tasks = ["Panoptic"]
    
#     # inistalize task
#     model.model.task_switch['spatial'] = False
#     model.model.task_switch['visual'] = False
#     model.model.task_switch['grounding'] = False
#     model.model.task_switch['audio'] = False

#     # check if reftxt is list of strings
#     assert isinstance(reftxt, list), f"reftxt should be a list of strings, but got {type(reftxt)}"
#     model.model.task_switch['grounding'] = True
#     predicts = {}
#     for i, txt in enumerate(reftxt): 
#         data['text'] = txt
#         batch_inputs = [data]

#         results,image_size,extra = model.model.evaluate_demo(batch_inputs)

#         pred_masks = results['pred_masks'][0]
#         v_emb = results['pred_captions'][0]
#         t_emb = extra['grounding_class']

#         t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
#         v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)

#         temperature = model.model.sem_seg_head.predictor.lang_encoder.logit_scale
#         out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)
        
#         matched_id = out_prob.max(0)[1]
#         pred_masks_pos = pred_masks[matched_id,:,:]
#         pred_class = results['pred_logits'][0][matched_id].max(dim=-1)[1]


#         # interpolate mask to ori size
#         #pred_masks_pos = (F.interpolate(pred_masks_pos[None,], image_size[-2:], mode='bilinear')[0,:,:data['height'],:data['width']] > 0.0).float().cpu().numpy()
#         # masks.append(pred_masks_pos[0])
#         # mask = pred_masks_pos[0]
#         # masks.append(mask)
#         # interpolate mask to ori size
#         pred_mask_prob = F.interpolate(pred_masks_pos[None,], image_size[-2:], mode='bilinear')[0,:,:data['height'],:data['width']].sigmoid().cpu().numpy()
#         #pred_masks_pos = 1*(pred_mask_prob > 0.5)
#         predicts[txt] = pred_mask_prob[0]
        
#     masks = combine_masks(predicts)
        
#     predict_mask_stats = {}
#     print(masks.keys())
#     for i, txt in enumerate(masks):
#         mask = masks[txt]
#         demo = visual.draw_binary_mask(mask, color=colors_list[i], text=txt)
#         predict_mask_stats[txt] = mask_stats((predicts[txt]*255), image_ori)
        
#     res = demo.get_image()
#     torch.cuda.empty_cache()
#     # return Image.fromarray(res), stroke_inimg, stroke_refimg
#     return Image.fromarray(res), None, predict_mask_stats

