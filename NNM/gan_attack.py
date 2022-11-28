import torch.nn as nn
import torch
import numpy as np
import models_clu
import torch.nn.functional as F
import torchvision
import os

from sklearn.metrics import normalized_mutual_info_score as nmi
import matplotlib.pyplot as plt

models_path = './models/'
device = 'cuda'


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def plot_examples(images, labels):
    print(type(images[0]), type(labels))
    print(images.shape)
    w = 10
    h = 10
    fig = plt.figure(figsize=(10, 20))
    columns = 11
    rows = 12
    for i in range(0, columns*rows -4):
        img = np.random.randint(10, size=(h,w))
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(images[i].detach().cpu().reshape(28,28), cmap="gray")
        plt.title('#{}: {}'.format(i, labels[i]))
    plt.show()
    
class GAN_Attack:
    def __init__(self,
                 device,
                 model,
                 model_num_labels,
                 image_nc,
                 box_min,
                 box_max,
                 dataset):
        output_nc = image_nc
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model
        self.input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max
        self.batch_no = 0
        self.dataset = dataset
        self.queries = 0

        self.gen_input_nc = image_nc
        self.netG = models_clu.Generator(self.gen_input_nc, image_nc).to(device)
        self.netDisc = models_clu.Discriminator(image_nc).to(device)

        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=0.001)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=0.001)

        if not os.path.exists(models_path):
            os.makedirs(models_path)
    
    def _loss(self, X, adv_X):
        loss_device = 'cuda'
        
        res = self.model(X, forward_pass='return_all')
        output = res['output']
        
#         print(len(output))
  
#         prob = F.softmax(output, dim=1)
    
        res = self.model(adv_X, forward_pass='return_all')
        output2 = res['output']
  
#         prob2 = F.softmax(output, dim=1)
        
        self.queries += 1
        dist_loss = torch.tensor(0.).to('cuda')
        for i in range(len(output)):
            diff_vec = output2[i] - output[i] #
            sample_dist_loss = torch.matmul(diff_vec.view(1, -1),
                                            diff_vec.view(-1, 1))
            dist_loss += 0.5 * torch.squeeze(sample_dist_loss)
        
        return dist_loss # is beta * ||f(x) - Cluster||^2_2
                
    def train_batch(self, p, labels):
        torch.cuda.empty_cache()
        x = p.to('cuda')
        for i in range(1):
            perturbation = self.netG(x)

            # add a clipping trick
            adv_images = torch.clamp(perturbation, -0.15, 0.15) + x
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

            self.optimizer_D.zero_grad()
            pred_real = self.netDisc(x)
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
            loss_D_real.backward()

            pred_fake = self.netDisc(adv_images.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            loss_D_fake.backward()
            loss_D_GAN = loss_D_fake + loss_D_real
            self.optimizer_D.step()
            
        # optimize G
        for i in range(1):
            self.optimizer_G.zero_grad()

            # cal G's loss in GAN
            pred_fake = self.netDisc(adv_images)
            loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_G_fake.backward(retain_graph=True)

            # calculate perturbation norm
            C = 0.1
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
#             loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))
            
            loss_adv = self._loss(x, adv_images) #works
            loss_adv = -loss_adv  #works
        
            adv_lambda = 30
            pert_lambda = 1
            loss_G = adv_lambda*loss_adv + pert_lambda*loss_perturb
            loss_G.backward()
            self.optimizer_G.step()

        return loss_D_GAN.item(), loss_G.item(), loss_perturb.item(), loss_adv.item()

    def train(self, train_dataloader, epochs):
        for epoch in range(1, epochs+1):

            if epoch == 50:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.0001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.0001)
            if epoch == 80:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.00001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.00001)
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            loss_device = torch.device("cuda")

            for batch_idx, (inputs, targets) in enumerate(train_dataloader):
#             for batch in train_dataloader:
#                 inputs = batch['image']
#                 targets = batch['target']
                torch.cuda.empty_cache()
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch = self.train_batch(inputs, targets)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch

            # print statistics
            self.batch_no += 1
            num_batch = len(train_dataloader)
            print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
             \nloss_perturb: %.3f, loss_adv: %.3f, \n" %
                  (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
                   loss_perturb_sum/num_batch, loss_adv_sum/num_batch))
            
            print("|No. of queries to the mdodel int this epoch: ", self.queries)

            # save generator
            if epoch%10==0:
                netG_file_name = models_path + 'netG_cc_' + str(self.dataset)+'_epoch_' + str(epoch) + '.pth'
                torch.save(self.netG.state_dict(), netG_file_name)

