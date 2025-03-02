import os
import copy
import logging
import random
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.backends import cudnn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from options import parse_args
from networks import define_net, define_optimizer, define_scheduler, count_parameters
from data_loaders import GraphDataset,custom_collate,GraphDataset_PRCC

def set_seed(seed):
    cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    random.seed(seed)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0.0
    all_labels = []
    all_preds = []
    criterion = F.cross_entropy
    
    with torch.no_grad():
        for batch_data in loader:
            batch_data = batch_data.to(device)
            features, logits = model(batch_data)
            loss = criterion(logits, batch_data.y.squeeze())
            loss_total += loss.item()
            
            preds = logits.argmax(dim=1)
            all_labels.extend(batch_data.y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            correct += preds.eq(batch_data.y.squeeze()).sum().item()
            total += batch_data.y.size(0)

    avg_loss = loss_total / len(loader)
    accuracy = correct / total if total > 0 else 0
    
    f1_per_class = f1_score(all_labels, all_preds, average=None, labels=[0, 1, 2])  
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', labels=[0, 1, 2])
    
    # f1_per_class = f1_score(all_labels, all_preds, average=None)  
    # weighted_f1 = f1_score(all_labels, all_preds, average='weighted')   

    return avg_loss, accuracy, f1_per_class, weighted_f1

def train(opt, data, device, k):
    """
    Training function: Train on a single cross-validation split for graph data multi-classification task.
    """
    # Set a fixed random seed to ensure experiment reproducibility
    set_seed(42) # seeds = [42, 123, 999, 1001, 2001]

    model = define_net(opt, k)
    model.to(device)

    optimizer = define_optimizer(opt, model)
    scheduler = define_scheduler(opt, optimizer)

    print(model)
    print("Number of Trainable Parameters: %d" % count_parameters(model))
    logging.info("Activation Type: %s" % opt.act_type)
    logging.info("Optimizer Type: %s" % opt.optimizer_type)
    logging.info("Regularization Type: %s" % opt.reg_type)

    # Build data loaders
    train_folder = data['train_folder']
    val_folder = data['val_folder']
    train_dataset = GraphDataset(train_folder)
    val_dataset = GraphDataset(val_folder)
    # train_dataset = GraphDataset_PRCC(train_folder)
    # val_dataset   = GraphDataset_PRCC(val_folder)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, collate_fn=custom_collate)

    metric_logger = {'train': {'loss': [], 'accuracy': [], 'f1': []},
                     'val': {'loss': [], 'accuracy': [], 'f1': []}}

    # Training loop
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        model.train()
        train_loss_epoch = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = batch_data.to(device)
            optimizer.zero_grad()

            features, logits = model(batch_data)
            loss = F.cross_entropy(logits, batch_data.y.squeeze())
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()
            preds = logits.argmax(dim=1)
            correct_train += preds.eq(batch_data.y.squeeze()).sum().item()
            total_train += batch_data.y.size(0)

        scheduler.step()
        avg_train_loss = train_loss_epoch / len(train_loader)
        train_acc = 100.0 * correct_train / total_train

        # F1 for training set
        _, _, f1_train, weighted_f1_train = evaluate(model, train_loader, device)

        # Validation phase
        model.eval()
        val_loss_epoch = 0.0
        correct_val = 0
        total_val = 0
        preds_list = []
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(device)
                _, logits = model(batch_data)
                loss = F.cross_entropy(logits, batch_data.y.squeeze())
                val_loss_epoch += loss.item()
                preds = logits.argmax(dim=1)
                preds_list.append(preds.cpu().numpy())
                correct_val += preds.eq(batch_data.y.squeeze()).sum().item()
                total_val += batch_data.y.size(0)
        avg_val_loss = val_loss_epoch / len(val_loader)
        val_acc = 100.0 * correct_val / total_val

        # F1 for validation set
        _, _, f1_val, weighted_f1_val = evaluate(model, val_loader, device)

        metric_logger['train']['loss'].append(avg_train_loss)
        metric_logger['train']['accuracy'].append(train_acc)
        metric_logger['train']['f1'].append(weighted_f1_train)
        metric_logger['val']['loss'].append(avg_val_loss)
        metric_logger['val']['accuracy'].append(val_acc)
        metric_logger['val']['f1'].append(weighted_f1_val)

        print("Epoch {:02d}: Train Loss: {:.4f}, Train Acc: {:.2f}%, Train Weighted F1: {:.4f} | "
              "Val Loss: {:.4f}, Val Acc: {:.2f}%, Val Weighted F1: {:.4f}".format(
            epoch, avg_train_loss, train_acc, weighted_f1_train, avg_val_loss, val_acc, weighted_f1_val))

    # Get predictions on the training set (for saving)
    model.eval()
    pred_train_list = []
    with torch.no_grad():
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            _, logits = model(batch_data)
            pred_train_list.append(logits.argmax(dim=1).cpu().numpy())
    pred_train = np.concatenate(pred_train_list)

    return model, optimizer, metric_logger, pred_train


def test(opt, model, data, split, device):
    """
    Testing function: Test the model on the specified data split, calculate the average loss and accuracy, and return the predictions.
    """
    if split == 'train':
        folder = data['train_folder']
    elif split == 'test':
        folder = data['test_folder']
    else:
        raise ValueError("Invalid split: choose 'train' or 'test'")

    dataset = GraphDataset(folder)
    # dataset = GraphDataset_PRCC(folder)
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    preds_list = []

    with torch.no_grad():
        for batch_data in loader:
            batch_data = batch_data.to(device)
            _, logits = model(batch_data)
            loss = F.cross_entropy(logits, batch_data.y.squeeze())
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            preds_list.append(preds.cpu().numpy())
            correct += preds.eq(batch_data.y.squeeze()).sum().item()
            total += batch_data.y.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    preds_all = np.concatenate(preds_list)

    # Calculate F1
    _, _, f1_test, weighted_f1_test = evaluate(model, loader, device)

    return avg_loss, accuracy, f1_test, weighted_f1_test

if __name__ == '__main__':
    # 1. Initialize parameters and device
    opt = parse_args()
    device = torch.device("cuda:{}".format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device("cpu")
    print("Using device:", device)

    # Create directory structure for saving checkpoints
    if not os.path.exists(opt.checkpoints_dir):
        os.makedirs(opt.checkpoints_dir)
    exp_dir = os.path.join(opt.checkpoints_dir, opt.exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    model_dir = os.path.join(exp_dir, opt.model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 2. Load data splits (there is only one set of data here)
    data_cv_path = os.path.join(opt.dataroot, 'splits', 'graph_cv_splits_bracs_sum.pkl')
    print("Loading splits from %s" % data_cv_path)
    data_cv = pickle.load(open(data_cv_path, 'rb'))
    
    key = list(data_cv['cv_splits'].keys())[0]
    data = data_cv['cv_splits'][key]

    # 3. Train the model
    model, optimizer, metric_logger, pred_train = train(opt, data, device, key)

    # 4. Evaluate the model on the training and testing sets
    loss_train, acc_train, f1_train, weighted_f1_train = test(opt, model, data, 'train', device)
    loss_test, acc_test, f1_test, weighted_f1_test = test(opt, model, data, 'test', device)

    print("[Final] Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Train Weighted F1: {:.4f}".format(loss_train, acc_train, weighted_f1_train))
    print("[Final] Test  Loss: {:.4f}, Test Accuracy: {:.2f}%, Test Weighted F1: {:.4f}".format(loss_test, acc_test, weighted_f1_test))

    # 5. Save the model and prediction results
    if torch.cuda.is_available() and len(opt.gpu_ids) > 0:
        model_state_dict = model.module.cpu().state_dict() if hasattr(model, 'module') else model.cpu().state_dict()
    else:
        model_state_dict = model.cpu().state_dict()
    checkpoint = {
        'split': key,
        'opt': opt,
        'epoch': opt.niter + opt.niter_decay,
        'data': data,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metric_logger
    }
    torch.save(checkpoint, os.path.join(model_dir, '%s.pt' % (opt.model_name)))

    print('Test Accuracy: {:.2f}%, Test Weighted F1: {:.4f}'.format(acc_test, weighted_f1_test))
    print(f"F1 per class: {f1_test}")
