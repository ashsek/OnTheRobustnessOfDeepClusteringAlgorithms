import torch
from lib.utils import AverageMeter
import time
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import pyplot
import eval_cus
from PIL import Image

def _hungarian_match(flat_preds, flat_targets, num_samples, class_num):  
    num_k = class_num
    num_correct = np.zeros((num_k, num_k))
  
    for c1 in range(0, num_k):
        for c2 in range(0, num_k):
        # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes
  
    # num_correct is small
    match = linear_assignment(num_samples - num_correct)
#     indices = linear_sum_assignment(cost_matrix)  
    match = np.asarray(match)
    match = np.transpose(match)
  
    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))
  
    return res


def test(net, testloader,device, class_num):
    net.eval()
    predicted_all = []
    targets_all = []
    for batch_idx, (inputs, _,_, targets, indexes) in enumerate(testloader):
        batchSize = inputs.size(0)
        targets, inputs = targets.to(device), inputs.to(device)
        output = net(inputs)
        predicted = torch.argmax(output, 1)
        predicted_all.append(predicted)
        targets_all.append(targets)
    

    flat_predict = torch.cat(predicted_all).to(device)
    flat_target = torch.cat(targets_all).to(device)
    num_samples = flat_predict.shape[0]
    match = _hungarian_match(flat_predict, flat_target, num_samples, class_num)
    reordered_preds = torch.zeros(num_samples).to(device)
    
    for pred_i, target_i in match:
        reordered_preds[flat_predict == pred_i] = int(target_i)
        
    acc = int((reordered_preds == flat_target.float()).sum()) / float(num_samples) * 100
        
    return acc, reordered_preds

def test_ruc(net, net2, testloader, device, class_num):
    net.eval()
    net2.eval()
    
    predicted_all = [[] for i in range(0,3)]
    targets_all = []
    acc_list = []
    p_label_list = []
    
    for batch_idx, (inputs, _, _, targets, indexes) in enumerate(testloader):
        batchSize = inputs.size(0)
        targets, inputs = targets.to(device), inputs.to(device)
        logit = net(inputs)
        logit2 = net2(inputs)
        _, predicted = torch.max(logit, 1)
        _, predicted2 = torch.max(logit2, 1)
        _, predicted3 = torch.max(logit + logit2, 1)
        
        predicted_all[0].append(predicted)
        predicted_all[1].append(predicted2)
        predicted_all[2].append(predicted3)
        targets_all.append(targets)
    
    for i in range(0, 3):
        flat_predict = torch.cat(predicted_all[i]).to(device)
        flat_target = torch.cat(targets_all).to(device)
        num_samples = flat_predict.shape[0]
        acc = int((flat_predict.float() == flat_target.float()).sum()) / float(num_samples) * 100
        acc_list.append(acc)
        p_label_list.append(flat_predict)
        nmi = metrics.normalized_mutual_info_score(flat_target.cpu().numpy(), flat_predict.cpu().numpy())
        ari = metrics.adjusted_rand_score(flat_target.cpu().numpy(), flat_predict.cpu().numpy())
        acc_list.append(nmi)
        acc_list.append(ari)
        
    
    return acc_list, p_label_list

def save_examples(images, labels, noise=False, bno=0, adv=False, orig=False):
    print(type(images[0]), type(labels))
#     MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).cuda() #c10
#     STD = torch.tensor([0.2023, 0.1994, 0.2010]).cuda() #c10
#     MEAN = torch.tensor([0.5071, 0.4867, 0.4408]).cuda()
#     STD = torch.tensor([0.2675, 0.2565, 0.2761]).cuda()
    MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).cuda()
    STD = torch.tensor([0.2023, 0.1994, 0.2010]).cuda()
    
    for i in range(min(len(images), 20)):
        img = images[i]
        img = img * STD[:, None, None] + MEAN[:, None, None]
        npimg = img.detach().cpu().numpy()   # convert from tensor
        npimg = np.clip(npimg, 0, 1)
        if orig:
#             npimg = np.transpose(npimg, (1, 2, 0))
            plt.imsave(f'../Images/S10/RUC/orig/RUC_s10_b{bno}_{i}_lab{labels[i]}.png', npimg.T, dpi=600)
            continue
        if adv:
#             npimg = np.transpose(npimg, (1, 2, 0))
            plt.imsave(f'../Images/S10/RUC/adv/RUC_s10_b{bno}_{i}_lab{labels[i]}.png', npimg.T, dpi=600)
            continue
        if noise:
            npimg = npimg / 2 + 0.5 
            plt.imsave(f'../Images/S10/RUC/noise/RUC_s10_b{bno}_{i}_noise_lab{labels[i]}.png', npimg.T, dpi=600)
            continue
            
def test_ruc_adv_save(net, net2, testloader, device, class_num, pretrained_G, clamp, ds):
    net.eval()
    net2.eval()
    
    predicted_all = [[] for i in range(0,3)]
    targets_all = []
    acc_list = []
    p_label_list = []
    j = 1
    for batch_idx, (inputs, _, _, targets, indexes) in enumerate(testloader):
        batchSize = inputs.size(0)
        targets, inputs = targets.to(device), inputs.to(device)
        perturbation = pretrained_G(inputs)
        
#         norm_b = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
        norm_b = torch.mean(torch.linalg.norm(perturbation.view(perturbation.shape[0], -1), dim=1, ord = 2))
        perturbation = torch.clamp(perturbation, -clamp, clamp)
        
        norm_a = torch.mean(torch.linalg.norm(perturbation.view(perturbation.shape[0], -1), dim=1, ord = 2))
        adv_imgs = perturbation + inputs
#         adv_imgs = torch.clamp(adv_imgs, 0, 1)
        x,y = [-2.4306929111480713, 2.7512435913085938]
        adv_imgs = torch.clip(adv_imgs, x, y)
        
        logit = net(adv_imgs)
        logit2 = net2(adv_imgs)
        _, predicted = torch.max(logit, 1)
        _, predicted2 = torch.max(logit2, 1)
        _, predicted3 = torch.max(logit + logit2, 1)
        pps(inputs, adv_imgs, targets, predicted3)
    
        predicted_all[0].append(predicted)
        predicted_all[1].append(predicted2)
        predicted_all[2].append(predicted3)
        targets_all.append(targets)
        
        save_examples(inputs, targets_all[-1], bno=batch_idx, orig=True)
        save_examples(adv_imgs, predicted3, bno=batch_idx, adv=True)
        save_examples(perturbation, predicted3, bno=batch_idx, noise=True)
        
    print(f'Pertb norm before: {norm_b}, and after clamping {norm_a}')
    
#     for i in range(0, 3):
#         flat_predict = torch.cat(predicted_all[i]).to(device)
#         flat_target = torch.cat(targets_all).to(device)
#         num_samples = flat_predict.shape[0]
#         acc = int((flat_predict.float() == flat_target.float()).sum()) / float(num_samples) * 100
#         acc_list.append(acc)
#         p_label_list.append(flat_predict)
#         nmi = metrics.normalized_mutual_info_score(flat_target.cpu().numpy(), flat_predict.cpu().numpy())
#         ari = metrics.adjusted_rand_score(flat_target.cpu().numpy(), flat_predict.cpu().numpy())
#         acc_list.append(nmi)
#         acc_list.append(ari)
        
    
    return acc_list, p_label_list

def plot_examples(images, labels, dataset, adv=False):
    if dataset == 'CIFAR10':
        MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).cuda()
        STD = torch.tensor([0.2023, 0.1994, 0.2010]).cuda()
    elif dataset == 'CIFAR20':
        MEAN = torch.tensor([0.5071, 0.4867, 0.4408]).cuda()
        STD = torch.tensor([0.2675, 0.2565, 0.2761]).cuda()
    else:
        MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).cuda()
        STD = torch.tensor([0.2023, 0.1994, 0.2010]).cuda()
        
    print(type(images[0]), type(labels))
    print(images.shape)

    fig = plt.figure(figsize=(10, 20))
    columns = 11
    rows = 12
    for i in range(min(len(images), 20)):
        fig.add_subplot(rows, columns, i+1)
        img = images[i]
        
        img = img * STD[:, None, None] + MEAN[:, None, None]
        img = img.detach().cpu().numpy()
        img = np.clip(img, 0, 1)
#         if adv:    
#             Image.fromarray(img).save(f'Images/adv_{i}.jpg')
#         else:
#             Image.fromarray(img).save(f'Images/{i}.jpg')
            
        plt.axis('off')
#         plt.imsave(f'Images/adv_{i}.jpg', img)
        plt.imshow(img.transpose(1, 2, 0))
        if adv: 
            plt.imsave(f'Images/adv_{i}.jpg', img.transpose(1, 2, 0))
        else:
            plt.imsave(f'Images/{i}.jpg', img.transpose(1, 2, 0))
        
        plt.title('#{}: {}'.format(i, labels[i]))
    plt.show()
    
count = 0
def pps(images, adv_imgs, orig, predicted):
    global count
    MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).cuda()
    STD = torch.tensor([0.2023, 0.1994, 0.2010]).cuda()
    
    print(images.shape)

    for i, _ in enumerate(images):
        img = images[i]
        img_a = adv_imgs[i]

        img = img * STD[:, None, None] + MEAN[:, None, None]
        img = img.detach().cpu().numpy()
        img = np.clip(img, 0, 1)
        
        img_a = img_a * STD[:, None, None] + MEAN[:, None, None]
        img_a = img_a.detach().cpu().numpy()
        img_a = np.clip(img_a, 0, 1)

        
#         plt.show(img.transpose(1, 2, 0))
        plt.imsave(f'outputs/stl10/orig/{count}_{orig[i]}.png', img.transpose(1, 2, 0))
        plt.imsave(f'outputs/stl10/adv/{count}_{predicted[i]}.png', img_a.transpose(1, 2, 0))
        count += 1
#         return 
    

def test_ruc_adv(net, net2, testloader, device, class_num, pretrained_G, clamp, ds):
    net.eval()
    net2.eval()
    
    predicted_all = [[] for i in range(0,3)]
    targets_all = []
    acc_list = []
    p_label_list = []
    j = 1
    for batch_idx, (inputs, _, _, targets, indexes) in enumerate(testloader):
        batchSize = inputs.size(0)
        targets, inputs = targets.to(device), inputs.to(device)
        perturbation = pretrained_G(inputs)
        
#         norm_b = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
        norm_b = torch.mean(torch.linalg.norm(perturbation.view(perturbation.shape[0], -1), dim=1, ord = 2))
        perturbation = torch.clamp(perturbation, -clamp, clamp)
        
        norm_a = torch.mean(torch.linalg.norm(perturbation.view(perturbation.shape[0], -1), dim=1, ord = 2))
        adv_imgs = perturbation + inputs
#         adv_imgs = torch.clamp(adv_imgs, 0, 1)
        x,y = [-2.4306929111480713, 2.7512435913085938]
        adv_imgs = torch.clip(adv_imgs, x, y)
        
        logit = net(adv_imgs)
        logit2 = net2(adv_imgs)
        _, predicted = torch.max(logit, 1)
        _, predicted2 = torch.max(logit2, 1)
        _, predicted3 = torch.max(logit + logit2, 1)
        
#         if j:
#             plot_examples(inputs, targets, ds, adv=False)
#             plot_examples(adv_imgs, predicted3, ds, adv=True)
#             j = 0
        pps(inputs, adv_imgs, targets, predicted3)
    
        predicted_all[0].append(predicted)
        predicted_all[1].append(predicted2)
        predicted_all[2].append(predicted3)
        targets_all.append(targets)
    print(f'Pertb norm before: {norm_b}, and after clamping {norm_a}')
    
    for i in range(0, 3):
        flat_predict = torch.cat(predicted_all[i]).to(device)
        flat_target = torch.cat(targets_all).to(device)
        num_samples = flat_predict.shape[0]
        acc = int((flat_predict.float() == flat_target.float()).sum()) / float(num_samples) * 100
        acc_list.append(acc)
        p_label_list.append(flat_predict)
        nmi = metrics.normalized_mutual_info_score(flat_target.cpu().numpy(), flat_predict.cpu().numpy())
        ari = metrics.adjusted_rand_score(flat_target.cpu().numpy(), flat_predict.cpu().numpy())
        acc_list.append(nmi)
        acc_list.append(ari)
        
    
    return acc_list, p_label_list


def test_ruc_adv_norm(net, net2, testloader, device, class_num, pretrained_G, clamp, ds):
    net.eval()
    net2.eval()
    
    predicted_all = [[] for i in range(0,3)]
    targets_all = []
    acc_list = []
    p_label_list = []
    j = 1
    norm = 0.0
    
    for batch_idx, (inputs, _, _, targets, indexes) in enumerate(testloader):
        batchSize = inputs.size(0)
        targets, inputs = targets.to(device), inputs.to(device)
        perturbation = pretrained_G(inputs)
        
#         norm_b = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
        
        perturbation = torch.clamp(perturbation, -clamp, clamp)
        norm += torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1)).to('cpu').item()
        adv_imgs = perturbation + inputs
#         adv_imgs = torch.clamp(adv_imgs, 0, 1)
        
        logit = net(adv_imgs)
        logit2 = net2(adv_imgs)
        _, predicted = torch.max(logit, 1)
        _, predicted2 = torch.max(logit2, 1)
        _, predicted3 = torch.max(logit + logit2, 1)
        
#         if j:
#             plot_examples(inputs, targets, ds)
#             plot_examples(adv_imgs, predicted3, ds)
#             j = 0
        
        predicted_all[0].append(predicted)
        predicted_all[1].append(predicted2)
        predicted_all[2].append(predicted3)
        targets_all.append(targets)
#     print(f'Pertb norm before: {norm_b}, and after clamping {norm_a}')
    cf20 = False
    for i in range(0, 3):
        flat_predict = torch.cat(predicted_all[i]).to(device)
        flat_target = torch.cat(targets_all).to(device)
        num_samples = flat_predict.shape[0]
#         acc = int((flat_predict.float() == flat_target.float()).sum()) / float(num_samples) * 100
        
        p_label_list.append(flat_predict)
        class_num = 10
        if ds == 'CIFAR10':
            class_names = ('airplane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        elif ds == 'CIFAR20':
            class_names = [0]*20
            cf20=True
            class_num = 20
        else:
            class_names =  ['airplane',
                 'bird',
                 'car',
                 'cat',
                 'deer',
                 'dog',
                 'horse',
                 'monkey',
                 'ship',
                 'truck']
           
        
        clustering_stats_adv = eval_cus.check(flat_target, flat_predict, class_num, class_names, 
                                        compute_confusion_matrix=True, 
                    confusion_matrix_file=None, cf20=cf20, output_file2=f"RUC_c100_{clamp}_{i}n{norm/len(testloader)}.pdf")
        acc2 = clustering_stats_adv['ACC']
        nmi = clustering_stats_adv['NMI']
        ari = clustering_stats_adv['ARI']
        
#         nmi = metrics.normalized_mutual_info_score(flat_target.cpu().numpy(), flat_predict.cpu().numpy())
#         ari = metrics.adjusted_rand_score(flat_target.cpu().numpy(), flat_predict.cpu().numpy())
        acc_list.append(acc2)
        acc_list.append(nmi)
        acc_list.append(ari)
        
    
    return acc_list, p_label_list, norm/len(testloader)

def test_ruc_cls(net, net2, testloader, device, class_num, pretrained_G, clamp, ds, cls):
    net.eval()
    net2.eval()
    
    predicted_all = [[] for i in range(0,3)]
    targets_all = []
    acc_list = []
    p_label_list = []
    j = 1
    no = False
    
    for batch_idx, (inputs, _, _, targets, indexes) in enumerate(testloader):
        batchSize = inputs.size(0)
#         targets, inputs = targets.to(device), inputs.to(device)
        
        if no:
            index = np.where(targets != cls)
        else:
            index = np.where(targets == cls)
        inputs = inputs[index]
        targets = targets[index]
        
        targets, inputs = targets.to(device), inputs.to(device)
#         x_clas = x_clas.to(device)
#         perturbation = pretrained_G(x_clas)
        
        perturbation = pretrained_G(inputs)
        
#         norm_b = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
        norm_b = torch.mean(torch.linalg.norm(perturbation.view(perturbation.shape[0], -1), dim=1, ord = 2))
        perturbation = torch.clamp(perturbation, -clamp, clamp)
        
        norm_a = torch.mean(torch.linalg.norm(perturbation.view(perturbation.shape[0], -1), dim=1, ord = 2))
        adv_imgs = perturbation + inputs
        
        
#         adv_imgs = torch.clip(adv_imgs, -1, 1)
        
        logit = net(adv_imgs)
        logit2 = net2(adv_imgs)
        _, predicted = torch.max(logit, 1)
        _, predicted2 = torch.max(logit2, 1)
        _, predicted3 = torch.max(logit + logit2, 1)
        
        if j<=2:
            plot_examples(inputs, targets, ds)
            plot_examples(adv_imgs, predicted3, ds)
            j += 1
        
        predicted_all[0].append(predicted)
        predicted_all[1].append(predicted2)
        predicted_all[2].append(predicted3)
        targets_all.append(targets)
    print(f'Pertb norm before: {norm_b}, and after clamping {norm_a}')
    
    for i in range(0, 3):
        flat_predict = torch.cat(predicted_all[i]).to(device)
        flat_target = torch.cat(targets_all).to(device)
        num_samples = flat_predict.shape[0]
        acc = int((flat_predict.float() == flat_target.float()).sum()) / float(num_samples) * 100
        acc_list.append(acc)
        p_label_list.append(flat_predict)
        nmi = metrics.normalized_mutual_info_score(flat_target.cpu().numpy(), flat_predict.cpu().numpy())
        ari = metrics.adjusted_rand_score(flat_target.cpu().numpy(), flat_predict.cpu().numpy())
        acc_list.append(nmi)
        acc_list.append(ari)
        
    
    return acc_list, p_label_list