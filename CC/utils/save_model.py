import os
import torch


def save_model(model_path, model, optimizer, current_epoch, dataset):
    out = os.path.join(model_path, "retrain_{}_checkpoint_{}.tar".format(dataset, current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)
