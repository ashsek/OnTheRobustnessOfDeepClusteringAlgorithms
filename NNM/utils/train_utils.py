"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import numpy as np
from utils.utils import AverageMeter, ProgressMeter
from utils.faiss_utils import search_index_pytorch, search_raw_array_pytorch
import faiss
def simclr_train(train_loader, model, criterion, optimizer, epoch):
    """ 
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    for i, (batch, index) in enumerate(train_loader):
        images = batch['image']
        images_augmented = batch['image_augmented']
        b, c, h, w = images.size()
        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
        input_ = input_.view(-1, c, h, w) 
        input_ = input_.cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)

        output = model(input_).view(b, 2, -1)
        loss = criterion(output)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)


def scan_train(train_loader, model, criterion, optimizer, epoch, update_cluster_head_only=False, clustering_results = None):
    """ 
    Train w/ SCAN-Loss
    """
    total_losses = AverageMeter('Total Loss', ':.4e')
    consistency_losses = AverageMeter('Consistency Loss', ':.4e')
    ce_losses = AverageMeter('Class Cross Entropy', ':.4e')
    entropy_losses = AverageMeter('Entropy', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [total_losses, consistency_losses, ce_losses, entropy_losses],
        prefix="Epoch: [{}]".format(epoch))

    # GPU + PyTorch CUDA Tensors (1)
    res = faiss.StandardGpuResources()
    res.setDefaultNullStreamAllDevices()
    

    if update_cluster_head_only:
        model.eval() # No need to update BN
    else:
        model.train() # Update BN

    for i, (batch, index) in enumerate(train_loader):
        # Forward pass
        anchors = batch['anchor'].to('cuda',non_blocking=True)
        neighbors = batch['neighbor'].to('cuda',non_blocking=True)
        anchor_augmented = batch['anchor_augmented'].to('cuda',non_blocking=True)
       
        if update_cluster_head_only: # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_features = model(anchors, forward_pass='backbone')
                neighbors_features = model(neighbors, forward_pass='backbone')
                anchor_augmented_features = model(anchor_augmented, forward_pass='backbone')
            anchors_output = model(anchors_features, forward_pass='head')
            neighbors_output = model(neighbors_features, forward_pass='head')
            anchor_augmented_output = model(anchor_augmented_features, forward_pass='head')

        else: # Calculate gradient for backprop of complete network
            anchors_features = model(anchors, forward_pass='backbone')
            neighbors_features = model(neighbors, forward_pass='backbone')
            anchor_augmented_features = model(anchor_augmented, forward_pass='backbone')
            anchors_output = model(anchors_features, forward_pass='head')
            neighbors_output = model(neighbors_features, forward_pass='head')
            anchor_augmented_output = model(anchor_augmented_features, forward_pass='head')
        
        _, initial_rank = search_raw_array_pytorch(res, anchors_features, anchors_features, 2)

        initial_rank_index = initial_rank[:,-1].squeeze()

        # Loss for every head
        total_loss, consistency_loss, ce_loss, entropy_loss = [], [], [], []
        for j, (anchors_output_subhead, neighbors_output_subhead, anchor_augmented_output_subhead) in enumerate(zip(anchors_output, neighbors_output, anchor_augmented_output)):
            if clustering_results is not None:
                clustering_results_head = clustering_results[j]
            else:
                clustering_results_head = None
            total_loss_, consistency_loss_, ce_loss_, entropy_loss_ = criterion(anchors_output_subhead, neighbors_output_subhead, anchor_augmented_output_subhead, clustering_results_head, index, initial_rank_index)
            total_loss.append(total_loss_)
            consistency_loss.append(consistency_loss_)
            entropy_loss.append(entropy_loss_)
            ce_loss.append(ce_loss_)

        # Register the mean loss and backprop the total loss to cover all subheads
        total_losses.update(np.mean([v.item() for v in total_loss]))
        consistency_losses.update(np.mean([v.item() for v in consistency_loss]))
        entropy_losses.update(np.mean([v.item() for v in entropy_loss]))
        ce_losses.update(np.mean([v.item() for v in ce_loss]))

        total_loss = torch.sum(torch.stack(total_loss, dim=0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)


def selflabel_train(train_loader, model, criterion, optimizer, epoch, ema=None):
    """ 
    Self-labeling based on confident samples
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [losses],
                                prefix="Epoch: [{}]".format(epoch))
    model.train()

    for i, (batch, index) in enumerate(train_loader):
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)

        with torch.no_grad(): 
            output = model(images)
        output_augmented = model(images_augmented)

        # print(output.shape)
        # print(output_augmented.shape)
        # Loss for every head
        loss = []
        for j, (anchors_output_subhead, neighbors_output_subhead) in enumerate(zip(output, output_augmented)):

            loss_ = criterion(anchors_output_subhead, neighbors_output_subhead)
            loss.append(loss_)

        losses.update(np.mean([v.item() for v in loss]))
        loss = torch.sum(torch.stack(loss, dim=0))
        
        # loss = criterion(output, output_augmented)
        # losses.update(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None: # Apply EMA to update the weights of the network
            ema.update_params(model)
            ema.apply_shadow(model)
        
        if i % 25 == 0:
            progress.display(i)
