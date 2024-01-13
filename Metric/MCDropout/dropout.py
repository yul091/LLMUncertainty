import torch
from torch.nn import functional as F
from scipy.stats import entropy
from BasicalClass import (
    BasicModule, 
    common_predict, 
    common_ten2numpy, 
    common_get_maxpos,
)
import numpy as np
from Metric import BasicUncertainty
from tqdm import tqdm

class ModelActivateDropout(BasicUncertainty):
    def __init__(self, instance: BasicModule, device, iter_time):
        super(ModelActivateDropout, self).__init__(instance, device)
        self.iter_time = iter_time

    def extract_metric(self, data_loader, orig_pred_y):
        res = 0
        self.model.train()
        for _ in range(self.iter_time):
            _, pred, _ = common_predict(
                data_loader, self.model, self.device, 
                module_id=self.module_id
            )
            res = res + pred.eq(orig_pred_y)
        self.model.eval()
        res = common_ten2numpy(res.float() / self.iter_time)
        return res
    
    
    def forward(self, *input, **kwargs):
        ws_sum = None
        sample_logits = []
        self.model.train()
        
        # Stochastic variational inference
        for _ in range(self.iter_time): 
            logit = self.model(*input, **kwargs) # B X k
            # UE estimation
            ws = common_get_maxpos(logit) # B
            if ws_sum is None:
                ws_sum = ws
            else:
                ws_sum += ws
            sample_logits.append(logit.detach().cpu()) # B X k
            
        sample_logits = torch.stack(sample_logits, dim=0) # iter_time X B X k
        logit_mean = sample_logits.mean(dim=0) # B X k
        pred_mean = torch.argmax(logit_mean, dim=1) # B
        ws_mean = ws_sum / self.iter_time # B
        pv_score = - self._pv(sample_logits) # PV: B
        bald_score = - self._bald(sample_logits) # BALD: B
        
        return ws_mean, pv_score, bald_score, logit_mean, pred_mean


    @staticmethod
    def label_chgrate(orig_pred, prediction):
        """
        orig_pred: N
        prediction: N X iter_time
        """
        _, repeat_num = np.shape(prediction) # N, iter_time
        tmp = np.tile(orig_pred.reshape([-1, 1]), (1, repeat_num)) # N X iter_time
        return np.sum(tmp == prediction, axis=1, dtype=np.float) / repeat_num
    
    def _pv(self, logits: torch.Tensor): # iter_time X N X k
        prob_scores = F.softmax(logits, dim=-1).numpy() # iter_time X N X k
        pv_scores = np.var(prob_scores, axis=0).mean(axis=1) # N
        return pv_scores
    
    def _bald(self, logits: torch.Tensor): # iter_time X N X k
        prob_scores = F.softmax(logits, dim=-1).numpy() # iter_time X N X k
        prob_mean = prob_scores.mean(axis=0) # \bar{p_k}: N X k
        mean_prob_uncertainties = entropy(prob_mean, axis=-1) # -sum p_k log p_k: N
        mean_sample_entropy = - entropy(prob_scores, axis=-1).mean(axis=0) # mean_t sum_k (p_k log p_k): N 
        return mean_sample_entropy + mean_prob_uncertainties # N

    def _uncertainty_calculate(self, data_loader):
        
        print('Stochastic variational inference ...')
        ws_list, pv_list, bald_list, logit_list, pred_list, y_list = [], [], [], [], [], []
        self.model.to(self.device)

        if self.module_id == 0: # code summary
            for i, ((sts, paths, eds), y, length) in enumerate(data_loader):
                sts = sts.to(self.device)
                paths = paths.to(self.device)
                eds = eds.to(self.device)
                y = torch.tensor(y, dtype=torch.long)
                ws_mean, pv_score, bald_score, logit, pred_y = self(sts, paths, eds, length)
                logit_list.append(logit)
                pred_list.append(pred_y)
                y_list.append(y)
                ws_list.append(ws_mean)
                pv_list.append(pv_score)
                bald_list.append(bald_score)

        elif self.module_id == 1: # code completion
            for i, (input, y, _) in tqdm(enumerate(data_loader), total=len(data_loader)):
                input = input.to(self.device)
                ws_mean, pv_score, bald_score, logit, pred_y = self(input)
                logit_list.append(logit)
                pred_list.append(pred_y)
                y_list.append(y)
                ws_list.append(ws_mean)
                pv_list.append(pv_score)
                bald_list.append(bald_score)
                
        else:
            raise TypeError()
        
        ws_scores = np.concatenate(ws_list, axis=0) # N
        pv_scores = np.concatenate(pv_list, axis=0) # N
        bald_scores = np.concatenate(bald_list, axis=0) # N
        logits = torch.cat(logit_list, dim=0) # N X k
        preds = torch.cat(pred_list, dim=0) # N
        labels = torch.cat(y_list, dim=0) # N
        return self.eval_uncertainty(logits, preds, labels, [ws_scores, pv_scores, bald_scores])
