import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from typing import *
from tqdm import tqdm
from BasicalClass import BasicModule
from BasicalClass import common_ten2numpy
from BasicalClass import common_get_maxpos, common_predict, common_cal_accuracy
from Metric import BasicUncertainty


# A two-layer MLP class for learning the projection function
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x



class HidenDataset(Dataset):
    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y
        self.lenth = len(x)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.lenth


def build_loader(x, y, batch_size):
    dataset = HidenDataset(x, y)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    return data_loader


class PVScore(BasicUncertainty):
    def __init__(self, instance: BasicModule, device):
        super(PVScore, self).__init__(instance, device)
        self.train_sub_model(lr=1e-3, epoch=10)
        # if len(os.listdir(os.path.join(instance.save_dir, instance.__class__.__name__))) == 0:
        #     self.train_sub_model(lr=1e-2, epoch=10)

    def cal_svc(self, pred_vec, original_pred):
        pred_vec = self.softmax(pred_vec) # shape: B X k
        pred_order = torch.argsort(pred_vec, dim=1)
        sub_pred = pred_order[:, -1]
        sec_pos = pred_order[:, -2]
        second_vec = pred_vec[torch.arange(len(pred_vec)), sec_pos]
        cor_index = torch.where(sub_pred == original_pred) # shape: (N, )
        err_index = torch.where(sub_pred != original_pred) # shape: (N, )

        SVscore = torch.zeros([len(pred_vec)])
        lsh = torch.zeros([len(pred_vec)])
        lsh[cor_index] = second_vec[cor_index]
        tmp = pred_vec[torch.arange(len(pred_vec)), original_pred][cor_index]
        SVscore[cor_index] = tmp / (tmp + lsh[cor_index])

        lsh[err_index] = pred_vec[torch.arange(len(pred_order)), sub_pred][err_index]
        tmp = pred_vec[torch.arange(len(pred_vec)), original_pred][err_index]
        SVscore[err_index] = 1 - lsh[err_index] / (lsh[err_index] + tmp) # shape: (N, )

        return SVscore

    @staticmethod
    def get_pvscore(sv_score_list, snapshot, score_type=0):
        snapshot = torch.tensor(snapshot, dtype=torch.float32)
        if score_type == 0:
            weight = snapshot
        elif score_type == 1:
            weight = torch.log(snapshot)
        elif score_type == 2:
            weight = torch.exp(snapshot)
        else:
            raise ValueError("Not supported score type")
        weight = weight / torch.sum(weight)
        weight_svc = [sv_score_list[i].view([-1]) * weight[i] for i in range(len(weight))]
        weight_svc = torch.stack(weight_svc, dim=0)
        weight_svc = torch.sum(weight_svc, dim=0).view([-1])
        return weight_svc

    def train_sub_model(self, lr, epoch):
        print("Train sub models ...")
        sub_res_list, sub_num, label = self.instance.get_hiddenstate(self.train_loader, self.device)
        for i, sub_res in enumerate(sub_res_list):
            linear = nn.Linear(len(sub_res[1]), self.class_num).to(self.device)
            # linear = MLP(len(sub_res[1]), 1024, self.class_num).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(linear.parameters(), lr=lr)
            data_loader = build_loader(sub_res, label, self.train_batch_size)
            linear.train()
            for _ in range(epoch):
                for x, y in data_loader:
                    x = x.to(self.device) # B X d
                    y = y.to(self.device).view([-1]) # B
                    optimizer.zero_grad()
                    pred = linear(x)
                    loss = criterion(pred, y)
                    print('Batch loss: ', loss.item())
                    loss.backward()
                    optimizer.step()
                    # detach
                    x = x.detach().cpu()
                    y = y.detach().cpu()
                    pred = pred.detach().cpu()

            linear.eval()
            _, pred_y, _ = common_predict(
                data_loader, linear, device=self.device, train_sub=True,
                module_id=self.module_id
            )
            acc = common_cal_accuracy(pred_y, self.train_y)
            print('feature number for sub-model is', len(sub_res[0]), ', finish training the sub-model', sub_num[i],
                  'for', self.instance.__class__.__name__, ', accuracy is', acc.item())
            save_path = self.get_submodel_path(sub_num[i])
            torch.save(linear, save_path)
            print('save sub model in', save_path)

    def get_submodel_path(self, index):
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        dir_name = self.save_dir + '/' + self.instance.__class__.__name__ + '/'
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        if not self.instance.load_poor:
            save_path = dir_name + str(index) + '.h5'
        else:
            save_path = dir_name + str(index) + '_poor.h5'
        return save_path

    def get_submodel_prediction(self, data_loader):
        logits = []
        sub_res_list, sub_num, y = self.instance.get_hiddenstate(data_loader, self.device)
        for i in range(len(sub_num)):
            save_path = self.get_submodel_path(sub_num[i])
            linear_model = torch.load(save_path, map_location=self.device)
            linear_model.eval()
            hidden = sub_res_list[i]
            data_loader = build_loader(hidden, y, self.test_batch_size)
            pred_pos, pred_y, _ = common_predict(
                data_loader, linear_model, self.device, train_sub=True, 
                module_id=self.module_id
            )
            logits.append(pred_pos)
            print('test accuracy for', self.__class__.__name__, 
                  'submodel ', sub_num[i], 
                  'is', torch.sum(y.eq(pred_y), dtype=torch.float).item() / len(y))
        return logits, y, sub_num

    def get_svscore(self, data_loader, pred_y):
        logit_list, labels, sub_num = self.get_submodel_prediction(data_loader)
        svscore_list = []
        for sub_pred_pos in logit_list:
            svscore = self.cal_svc(sub_pred_pos, pred_y)
            svscore_list.append(svscore.view([-1, 1]))
        svscore_list = torch.cat(svscore_list, dim=1)
        svscore_list = torch.transpose(svscore_list, 0, 1)
        # Predictions
        logit_mean = torch.stack(logit_list, dim=0).mean(dim=0) # N X k
        pred_mean = torch.argmax(logit_mean, dim=1) # N
        
        return svscore_list, sub_num, logit_mean, pred_mean, labels 

    def _uncertainty_calculate(self, data_loader):
        print('Dissector uncertainty evaluation ...')
        weight_list = [0, 1, 2]
        uncertainty = []
        _, pred_y, _ = common_predict(
            data_loader, self.model, self.device, 
            module_id=self.module_id
        )
        # pred_y = pred_y.to(self.device)
        svscore_list, sub_num, logits, preds, labels = self.get_svscore(data_loader, pred_y)
        for weight in weight_list:
            pv_score = self.get_pvscore(svscore_list, sub_num, weight).detach().cpu()
            uncertainty.append(common_ten2numpy(pv_score))
        
        return self.eval_uncertainty(logits, preds, labels, uncertainty)


