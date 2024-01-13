import numpy as np
from tqdm import tqdm
from scipy.stats import entropy
import torch
import torch.nn.functional as F
from typing import List
from BasicalClass import (
    BasicModule,
    common_get_maxpos,
)
from Metric import BasicUncertainty


class Ensemble(BasicUncertainty):
    
    def __init__(self, instances: List[BasicModule], device):
        super(Ensemble, self).__init__(instances[0], device)
        self.instances = instances
        self.num_models = len(instances)
        self.model_list = []
        for i in range(self.num_models):
            self.model_list.append(self.instances[i].get_model())
            
    def forward(self, *input, **kwargs):
        ws_sum = None
        sample_logits = []
        # print("\nDeep ensemble forward: ")
        with torch.no_grad():
            # Stochastic variational inference
            for i in range(self.num_models): 
                self.model_list[i].eval()
                logit = self.model_list[i](*input, **kwargs) # B X k
                # print(f'model {i}, logit: {logit}')
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
        ws_mean = ws_sum / self.num_models # B
        pv_score = - self._pv(sample_logits) # PV: B
        bald_score = - self._bald(sample_logits) # BALD: B
        
        return ws_mean, pv_score, bald_score, logit_mean, pred_mean
    
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
        
        print('Deep ensemble inference ...')
        ws_list, pv_list, bald_list, logit_list, pred_list, y_list = [], [], [], [], [], []
        for i in range(self.num_models):
            self.model_list[i].to(self.device)

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