import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import init, Parameter
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # 关闭该检查
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGPooling, SAGEConv
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from utils import *
import random
from torch.backends import cudnn

def set_seed(seed):
    cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    random.seed(seed)

# 设置随机种子
seed_num = 2001 # seeds = [42, 123, 999, 1001, 2001]
set_seed(seed_num) 

# Set random seed
seed_num = 2001 # seeds = [42, 123, 999, 1001, 2001]
set_seed(seed_num) 

class GraphNetMultiCls(torch.nn.Module):
    def __init__(self, features, nhid=128, grph_dim=32, dropout_rate=0.25,
                 GNN=GCNConv, pooling_ratio=0.20, num_classes=3, init_max=True, act=None):
        """
        Parameter description:
            features: Node feature dimension (input dimension)
            nhid: Hidden layer dimension
            grph_dim: Low-dimensional graph representation dimension
            dropout_rate: Dropout rate
            GNN: The model type for graph convolution (e.g. 'GCN' or 'GraphSAGE')
            pooling_ratio: Pooling ratio
            num_classes: Number of classification categories
            act: The activation function for the final output, optional
        """
        super(GraphNetMultiCls, self).__init__()

        self.dropout_rate = dropout_rate
        self.act = act
        self.conch_net = load_conch_model(num_classes=num_classes)

        # Define graph convolution layers and pooling layers
        self.conv1 = SAGEConv(features, nhid)
        self.pool1 = SAGPooling(nhid, ratio=pooling_ratio, GNN=GNN)
        self.conv2 = SAGEConv(nhid, nhid)
        self.pool2 = SAGPooling(nhid, ratio=pooling_ratio, GNN=GNN)
        self.conv3 = SAGEConv(nhid, nhid)
        self.pool3 = SAGPooling(nhid, ratio=pooling_ratio, GNN=GNN)

        # Output dimension for two types of global pooling (max and average) is nhid*2
        self.lin1 = nn.Linear(nhid * 2, nhid)
        self.lin2 = nn.Linear(nhid, grph_dim)
        self.lin3 = nn.Linear(grph_dim, num_classes)

        # If you need to fix the output range (e.g., in regression scenarios), you can set it here
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

        if init_max:
            # You need to implement or import the init_max_weights function
            init_max_weights(self)
            print("Initializing with Max")

    def forward(self, image):
        """
        image: Tensor, shape (3, H, W) — Processed data (e.g., resized to (1024,1024))
        """
        # If you have preprocessing steps, you can call them here, such as NormalizeFeaturesV2, NormalizeEdgesV2
        # data = NormalizeFeaturesV2()(data)
        # data = NormalizeEdgesV2()(data)
        
        data = extract_patch_features(image, self.conch_net, patch_size=(256, 256), stride=256, k_neighbors=5)
        
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # First convolution layer + pooling
        x = F.relu(self.conv1(x, edge_index))
        result = self.pool1(x, edge_index, edge_attr, batch)
        if len(result) == 6:
            x, edge_index, edge_attr, batch, perm, score = result
        elif len(result) == 5:
            x, edge_index, edge_attr, batch, perm = result
        elif len(result) == 3:
            x, edge_index, batch = result
        else:
            raise ValueError("Unexpected number of return values from SAGPooling")
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Second convolution layer + pooling
        x = F.relu(self.conv2(x, edge_index))
        result = self.pool2(x, edge_index, edge_attr, batch)
        if len(result) == 6:
            x, edge_index, edge_attr, batch, perm, score = result
        elif len(result) == 5:
            x, edge_index, edge_attr, batch, perm = result
        elif len(result) == 3:
            x, edge_index, batch = result
        else:
            raise ValueError("Unexpected number of return values from SAGPooling")
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Third convolution layer + pooling
        x = F.relu(self.conv3(x, edge_index))
        result = self.pool3(x, edge_index, edge_attr, batch)
        if len(result) == 6:
            x, edge_index, edge_attr, batch, perm, score = result
        elif len(result) == 5:
            x, edge_index, edge_attr, batch, perm = result
        elif len(result) == 3:
            x, edge_index, batch = result
        else:
            raise ValueError("Unexpected number of return values from SAGPooling")
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Fusion of global information from all three layers
        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        features = F.relu(self.lin2(x))
        out = self.lin3(features)

        if self.act is not None:
            out = self.act(out)
            # If using Sigmoid and needing to adjust the output range, apply transformation if necessary
            if isinstance(self.act, nn.Sigmoid):
                out = out * self.output_range + self.output_shift

        return features, out

# ===================== ConchNet Definition =====================
class ConchNet(nn.Module):
    def __init__(self, original_net, num_classes):
        super(ConchNet, self).__init__()
        self.original_net = original_net
        self.fc = nn.Linear(512, num_classes)
        # Add encode_image_with_fc method to original_net (for easy calling)
        self.original_net.encode_image_with_fc = self.encode_image_with_fc

    def encode_image_with_fc(self, x):
        features = self.original_net.encode_image(x, proj_contrast=False, normalize=False)
        logits = self.fc(features)
        return logits

    def forward_threshold(self, x, threshold=1e6):
        features = self.original_net.encode_image(x, proj_contrast=False, normalize=False)
        # Use torch.clamp instead of clip
        x = torch.clamp(x, max=threshold)
        logits = self.fc(features)
        return logits

    def forward_fea(self, x):
        features = self.original_net.encode_image(x, proj_contrast=False, normalize=False)
        return features

    def forward(self, x, fea_if=False):
        if not fea_if:
            return self.encode_image_with_fc(x)
        else:
            features = self.original_net.encode_image(x, proj_contrast=False, normalize=False)
            return self.fc(features), features

    def feature_list(self, x):
        features = self.original_net.encode_image(x, proj_contrast=False, normalize=False)
        return x, [features]

def has_CONCH():
    if 'CONCH_CKPT_PATH' not in os.environ:
        os.environ["CONCH_CKPT_PATH"] = "/data_nas2/zd/paper1/models/conch/pytorch_model.bin"
    CONCH_CKPT_PATH = os.environ.get('CONCH_CKPT_PATH', None)
    if CONCH_CKPT_PATH is None or not os.path.exists(CONCH_CKPT_PATH):
        print("CONCH not installed or CONCH_CKPT_PATH not set correctly")
        return False, None
    return True, CONCH_CKPT_PATH

def load_conch_model(num_classes=2):
    HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
    assert HAS_CONCH, 'CONCH is not available'
    from conch.open_clip_custom import create_model_from_pretrained
    original_net, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
    model = ConchNet(original_net, num_classes=num_classes)
    return model

# ===================== Dataset Definition =====================
class CustomImageDataset(Dataset):
    """
    Dataset directory structure example:
      root_dir/
         0_N/
             img1.png
             img2.png
             ...
         1_X/
             img3.png
             ...
    The name of each class folder is separated by "_". The first part is considered the original label,
    and then it is mapped to a new label according to the mapping dictionary:
      Original labels 0,1,2 -> 0; 3,4 -> 1; 5,6 -> 2
    """
    def __init__(self, root_dir, transform=None):
        self.samples = []  # Each element is (image path, mapped label)
        self.transform = transform
        
        # Define the label mapping dictionary
        self.label_mapping = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
        
        # Get all class folder names and sort them
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        # Construct a dictionary mapping classes to the mapped labels: extract the original label from folder name, then map to new label
        self.class_to_idx = {}
        for cls in self.classes:
            # Assume folder name is like "0_N", take the part before "_"
            label_str = cls.split("_")[0]
            try:
                orig_label = int(label_str)
            except ValueError:
                orig_label = label_str
            # If it's an integer and exists in the mapping dictionary, map to the new label
            if isinstance(orig_label, int) and orig_label in self.label_mapping:
                mapped_label = self.label_mapping[orig_label]
            else:
                mapped_label = orig_label
            self.class_to_idx[cls] = mapped_label
        
        # Traverse each class folder and add all png files to samples
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(".png"):
                    img_path = os.path.join(cls_dir, fname)
                    self.samples.append((img_path, self.class_to_idx[cls]))
                    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class PRCC_ImageDataset(Dataset):
    """
    New dataset directory structure example:
      root_dir/
         1/
             WSI_1/
                 img1.png
                 img2.png
                 ...
             WSI_2/
                 img3.png
                 ...
         2/
             WSI_3/
                 img4.png
                 ...
    
    Map labels based on the name of the major category folders (1 or 2), and re-split the data into train, val, and test.
    """
    def __init__(self, root_dir, transform=None, split='train', val_size=0.15, test_size=0.15):
        self.samples = []  # Each element is (image path, mapped label)
        self.transform = transform
        self.split = split  # The current split type ('train', 'val', 'test')
        
        # Define label mapping dictionary
        self.label_mapping = {'1': 0, '2': 1}  # '1' -> 0, '2' -> 1
        
        # Get the major class folders ('1' and '2')
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        # Traverse each major class folder and get all images under each WSI folder
        all_samples = []
        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            if cls not in self.label_mapping:
                continue  # Skip the class if its label is not in the mapping dictionary
            
            # Get the mapped label for the current class
            mapped_label = self.label_mapping[cls]
            
            # Traverse all WSI folders under the current class
            for wsi_folder in os.listdir(class_dir):
                wsi_dir = os.path.join(class_dir, wsi_folder)
                if os.path.isdir(wsi_dir):
                    # Get all .png images under the WSI folder
                    for fname in os.listdir(wsi_dir):
                        if fname.lower().endswith(".png"):
                            img_path = os.path.join(wsi_dir, fname)
                            all_samples.append((img_path, mapped_label))
        
        # Split the data into training, validation, and testing sets based on the specified ratio
        train_samples, temp_samples = train_test_split(all_samples, test_size=val_size + test_size, random_state=42)
        val_samples, test_samples = train_test_split(temp_samples, test_size=test_size / (val_size + test_size), random_state=42)
        
        # Select the corresponding dataset based on the current split
        if self.split == 'train':
            self.samples = train_samples
        elif self.split == 'val':
            self.samples = val_samples
        elif self.split == 'test':
            self.samples = test_samples
        else:
            raise ValueError("Invalid split. Must be 'train', 'val', or 'test'.")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ===================== Data Preprocessing and DataLoader =====================
# Preprocessing for the entire image: e.g., resizing to (1024,1024)
data_transforms = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor()
])

# Modify to the actual data paths
# train_root = "/data_local/public_files/BRACS_RoI/train"
# val_root   = "/data_local/public_files/BRACS_RoI/val"
# test_root  = "/data_local/public_files/BRACS_RoI/test"

# train_dataset = CustomImageDataset(root_dir=train_root, transform=data_transforms)
# val_dataset   = CustomImageDataset(root_dir=val_root, transform=data_transforms)
# test_dataset  = CustomImageDataset(root_dir=test_root, transform=data_transforms)

PRCC_root = '/data_local/public_files/PRCC_Subtype/dataset'
train_dataset = PRCC_ImageDataset(root_dir=PRCC_root, transform=data_transforms, split='train')
val_dataset   = PRCC_ImageDataset(root_dir=PRCC_root, transform=data_transforms, split='test')
test_dataset  = PRCC_ImageDataset(root_dir=PRCC_root, transform=data_transforms, split='val')

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)

# ===================== Updated Patch Feature Extraction Function =====================
def extract_patch_features(image, conch_net, patch_size=(256, 256), stride=256, k_neighbors=5):
    """
    For an image (Tensor, shape (3, H, W)), perform sliding window cutting to extract each patch,
    and use conch_net.forward_fea to extract 512-dimensional features for each patch.
    Return graph structure data: node features (num_patches, 512) and adjacency matrix (edge_index).
    """
    # Define the preprocessing for CONCH (from Tensor to PIL, then Resize, ToTensor, and Normalize)
    conch_preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    
    C, H, W = image.shape
    patch_h, patch_w = patch_size
    patches = []
    
    # Extract all patches
    for i in range(0, H - patch_h + 1, stride):
        for j in range(0, W - patch_w + 1, stride):
            patch = image[:, i:i+patch_h, j:j+patch_w]
            patch = conch_preprocess(patch)  # The processed patch size is (3, 224, 224)
            patches.append(patch)
    
    if len(patches) == 0:
        raise ValueError("No patches extracted, please check patch_size and stride")
    
    # Stack patches into a batch: num_patches x 3 x 224 x 224
    patches = torch.stack(patches, dim=0)  # (num_patches, 3, 224, 224)
    
    # Move patches to the same device as conch_net
    device = next(conch_net.parameters()).device
    patches = patches.to(device)
    
    # Use conch_net to extract features for each patch
    features = conch_net.forward_fea(patches)  # (num_patches, 512)
    
    # Build the graph's adjacency matrix (based on the nearest neighbors of the patch centers)
    centers = []
    num_patches = features.shape[0]
    for i in range(0, num_patches):
        row = i // (W // patch_size[1])  # Get the row of the patch
        col = i % (W // patch_size[1])  # Get the column of the patch
        center_x = col * patch_size[1] + patch_size[1] // 2
        center_y = row * patch_size[0] + patch_size[0] // 2
        centers.append([center_x, center_y])
    
    centers = torch.tensor(centers, dtype=torch.float).to(device)  # Ensure centers are on the same device
    
    # Use the nearest neighbors algorithm to build the graph's edges (calculated on GPU)
    knn = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
    knn.fit(centers.cpu().numpy())  # Move centers to CPU for calculation
    distances, indices = knn.kneighbors(centers.cpu().numpy())
    
    # Build the adjacency matrix
    edge_index = []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors:
            edge_index.append([i, neighbor])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # (2, num_edges)
    
    # Move edge_index to the same device as features
    edge_index = edge_index.to(device)
    
    # Return graph structure data: node features and adjacency matrix
    data = Data(x=features, edge_index=edge_index)
    
    return data



def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, device, epochs=100,
                       best_model_path="best_model.pth", final_model_path="final_model.pth"):
    best_val_accuracy = 0.0
    best_epoch = 0
    for epoch in range(epochs):
        # ----------------- 训练阶段 -----------------
        model.train()
        train_loss_total = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            batch_loss = 0.0
            # print('labels:', labels)
            # 逐个处理每个样本
            for img, label in zip(images, labels):
                # print('label:', label)
                features, logits = model(img)  # logits: (num_classes,)
                # print('logits:', logits)
                # print('logits.unsqueeze(0):',logits.unsqueeze(0))                
                logits = logits # 确保 logits 形状为 [1, num_classes]
                label = label.unsqueeze(0)    # 确保 label 形状为 [1]，包含类别索引
                # print('label.unsqueeze(0):', label.unsqueeze(0))
                # 计算损失
                loss = criterion(logits, label)  # 计算单个样本的损失
                loss.backward()  # 在每个样本上反向传播
                batch_loss += loss.item()

                # 统计训练准确率
                pred = logits.argmax(dim=-1)  # 找到预测的类别
                if pred == label:
                    correct_train += 1
                total_train += 1
            
            optimizer.step()  # 更新参数
            train_loss_total += batch_loss
        
        train_loss = train_loss_total / total_train if total_train > 0 else 0
        train_accuracy = correct_train / total_train if total_train > 0 else 0
        
        # ----------------- 验证阶段 -----------------
        model.eval()
        val_loss_total = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                for img, label in zip(images, labels):
                    features, logits = model(img)
                    logits = logits # 确保 logits 形状为 [1, num_classes]
                    label = label.unsqueeze(0)    # 确保 label 形状为 [1]
                    
                    loss = criterion(logits, label)
                    val_loss_total += loss.item()
                    
                    pred = logits.argmax(dim=-1)
                    if pred == label:
                        correct_val += 1
                    total_val += 1
        
        avg_val_loss = val_loss_total / total_val if total_val > 0 else 0
        val_accuracy = correct_val / total_val if total_val > 0 else 0
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # 保存验证准确率最高的模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Best model updated at epoch {epoch+1} (Val Acc: {val_accuracy:.4f})")

    # 保存最终模型
    torch.save(model.state_dict(), final_model_path)
    print(f"Training finished. Best Val Acc: {best_val_accuracy:.4f} at epoch {best_epoch}.")
    print(f"Final model saved to {final_model_path} and best model saved to {best_model_path}.")


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss()  # 如果需要计算 loss
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            for img, label in zip(images, labels):
                features, logits = model(img)  # 保证输入是一个batch（即使只有一个样本）
                logits = logits # 确保 logits 形状为 [1, num_classes]
                label = label.unsqueeze(0)    # 确保 label 形状为 [1]
                
                loss = criterion(logits, label)
                loss_total += loss.item()
                
                # 获取预测结果
                pred = logits.argmax(dim=1).item()  # 获取预测的类
                all_labels.append(label.item())    # 真实标签
                all_preds.append(pred)             # 预测标签
                
                if pred == label.item():
                    correct += 1
                total += 1

    avg_loss = loss_total / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0

    # 计算每个类别的F1值
    # f1_per_class = f1_score(all_labels, all_preds, average=None, labels=[0, 1, 2])  # 这里假设有三个类，0, 1, 2
    # weighted_f1 = f1_score(all_labels, all_preds, average='weighted', labels=[0, 1, 2])
    
    f1_per_class = f1_score(all_labels, all_preds, average=None)  # 二分类
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')   

    print(f"Evaluation -> Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    print(f"F1 per class: {f1_per_class}")
    print(f"Weighted F1: {weighted_f1:.4f}")

if __name__ == "__main__":
    # 请根据实际数据集构建 DataLoader，此处示例使用 batch_size=4
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)
    
    num_classes = len(train_dataset.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型（假设 ClassificationModel 已定义）
    model = GraphNetMultiCls(features= 512, grph_dim=32, dropout_rate=0.25, GNN=GCNConv, pooling_ratio=0.2, act=nn.Sigmoid())
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=2e-5)  # 根据需要调整学习率
    criterion = nn.CrossEntropyLoss()
    
    print("Start Training and Evaluation...")
    model_path = '/PathoML/PatchToPathoML/checkpoints/' + 'G_prcc_' +  str(seed_num) + '.pth'
    train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, device, epochs=20,
                       best_model_path=model_path, final_model_path="final_model.pth")
    
    print("Test:")
    evaluate(model, test_loader, device)

