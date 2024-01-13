import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Union
from abc import ABCMeta, abstractmethod
from BasicalClass import (
    common_ten2numpy, 
    common_predict,
    common_cal_accuracy, 
    common_ten2numpy,
    common_get_entropy,
    common_get_confidence,
    spearmanr,
)
from BasicalClass import BasicModule



class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece



class BasicUncertainty(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self, instance: BasicModule, device):
        super(BasicUncertainty, self).__init__()
        self.instance = instance
        self.device = device
        self.train_batch_size = instance.train_batch_size
        self.test_batch_size = instance.test_batch_size
        self.model = instance.model.to(device)
        self.class_num = instance.class_num
        self.save_dir = instance.save_dir
        self.module_id = instance.module_id
        self.softmax = nn.Softmax(dim=1)
        self.nll_criterion = nn.CrossEntropyLoss()
        self.ece_criterion = _ECELoss()
        self.test_path = instance.test_path
        
        # handle train data and oracle
        self.train_y = instance.train_y
        self.train_pred_pos, self.train_pred_y =\
            instance.train_pred_pos, instance.train_pred_y
        self.train_loader = instance.train_loader
        self.train_num = len(self.train_y)
        self.train_oracle = np.int32(
            common_ten2numpy(self.train_pred_y).reshape([-1]) == \
                common_ten2numpy(self.train_y).reshape([-1])
        )
        # handle val data and oracle
        self.val_y = instance.val_y
        self.val_pred_pos, self.val_pred_y = \
            instance.val_pred_pos, instance.val_pred_y
        self.val_loader = instance.val_loader
        self.val_num = len(self.val_y)
        self.val_oracle = np.int32(
            common_ten2numpy(self.val_pred_y).reshape([-1]) == \
                common_ten2numpy(self.val_y).reshape([-1])
        )
        
        if self.test_path is not None:
            self.test_y = instance.test_y
            self.test_pred_pos, self.test_pred_y = \
                instance.test_pred_pos, instance.test_pred_y
            self.test_loader = instance.test_loader
            self.test_num = len(self.test_y)
            self.test_oracle = np.int32(
                common_ten2numpy(self.test_pred_y).reshape([-1]) == \
                    common_ten2numpy(self.test_y).reshape([-1])
            )
        else:
            self.test_y1, self.test_y2, self.test_y3 = \
                instance.test_y1, instance.test_y2, instance.test_y3
            self.test_pred_pos1, self.test_pred_y1 = \
                instance.test_pred_pos1, instance.test_pred_y1
            self.test_loader1 = instance.test_loader1
            self.test_num1 = len(self.test_y1)
            self.test_oracle1 = np.int32(
                common_ten2numpy(self.test_pred_y1).reshape([-1]) == \
                    common_ten2numpy(self.test_y1).reshape([-1])
            )
            self.test_pred_pos2, self.test_pred_y2 = \
                instance.test_pred_pos2, instance.test_pred_y2
            self.test_loader2 = instance.test_loader2
            self.test_num2 = len(self.test_y2)
            self.test_oracle2 = np.int32(
                common_ten2numpy(self.test_pred_y2).reshape([-1]) == \
                    common_ten2numpy(self.test_y2).reshape([-1])
            )
            self.test_pred_pos3, self.test_pred_y3 = \
                instance.test_pred_pos3, instance.test_pred_y3
            self.test_loader3 = instance.test_loader3
            self.test_num3 = len(self.test_y3)
            self.test_oracle3 = np.int32(
                common_ten2numpy(self.test_pred_y3).reshape([-1]) == \
                    common_ten2numpy(self.test_y3).reshape([-1])
            )
            

    @abstractmethod
    def _uncertainty_calculate(self, data_loader):
        return common_predict(data_loader, self.model, self.device, module_id=self.module_id)

    def run(self):
        score = self.get_uncertainty()
        self.save_uncertaity_file(score)
        print('finish score extract for class', self.__class__.__name__)
        return score

    def get_uncertainty(self):
        train_score = self._uncertainty_calculate(self.train_loader)
        val_score = self._uncertainty_calculate(self.val_loader)
        if self.test_path is not None:
            test_score = self._uncertainty_calculate(self.test_loader)
            result = {
                'train': train_score,
                'dev': val_score,
                'test': test_score
            }
        else:
            test_score1 = self._uncertainty_calculate(self.test_loader1)
            test_score2 = self._uncertainty_calculate(self.test_loader2)
            test_score3 = self._uncertainty_calculate(self.test_loader3)
            result = {
                'train': train_score,
                'dev': val_score,
                'test1': test_score1,
                'test2': test_score2,
                'test3': test_score3
            }
        return result
    
    
    def eval_uncertainty(
        self, 
        logits: Union[Tensor, List[Tensor]], 
        preds: Union[Tensor, List[Tensor]], 
        labels: Tensor, 
        uncertainty: Union[np.ndarray, List[np.ndarray]],
    ):
        """
        Calculate Accuracy, NLL, ECE, and Spearman's rank correlation
        """
        if isinstance(logits, list):
            nll, ece, acc, truths, entropies, confidencies = [], [], [], [], [], []
            for logit, pred in zip(logits, preds):
                nll.append(self.nll_criterion(logit, labels).item())
                ece.append(self.ece_criterion(logit, labels).item())
                acc.append(common_cal_accuracy(pred, labels).item())
                truths.append(common_ten2numpy(pred.eq(labels)))
                entropies.append(common_get_entropy(logit))
                confidencies.append(common_get_confidence(logit, labels))
            preds = [common_ten2numpy(pred) for pred in preds]
        else:
            nll = [self.nll_criterion(logits, labels).item()]
            ece = [self.ece_criterion(logits, labels).item()]
            acc = [common_cal_accuracy(preds, labels).item()]
            truths = [common_ten2numpy(preds.eq(labels))]
            preds = [common_ten2numpy(preds)]
            entropies = [common_get_entropy(logits)]
            confidencies = [common_get_confidence(logits, labels)]
        
        if isinstance(uncertainty, list):
            corrs = []
            for i, ue in enumerate(uncertainty):
                if isinstance(logits, list) and len(uncertainty) == len(logits):
                    rank_correlation, _ = spearmanr(ue, truths[i])
                else:
                    rank_correlation, _ = spearmanr(ue, truths[0])
                corrs.append(rank_correlation)
        else:
            rank_correlation, _ = spearmanr(uncertainty, truths[0])
            uncertainty = [uncertainty]
            corrs = [rank_correlation]
        print('Acc: %s, NLL: %s, ECE: %s, Rank correlation: %s' % (acc, nll, ece, corrs))
        
        return {
            'UE_scores': uncertainty,
            'preds': preds,
            'truths': truths,
            'labels': common_ten2numpy(labels),
            'entropies': entropies,
            'confidencies': confidencies,
            'nll': nll,
            'ece': ece,
            'acc': acc,
            'rank_correlation': corrs,
        }
    

    def save_uncertaity_file(self, score_dict):
        data_name = self.instance.__class__.__name__
        uncertainty_type = self.__class__.__name__
        save_name = self.save_dir + '/' + data_name + '/' + uncertainty_type + '.res'
        if not os.path.isdir(os.path.join(self.save_dir, data_name)):
            os.mkdir(os.path.join(self.save_dir, data_name))
        torch.save(score_dict, save_name)
        print('get result for dataset %s, uncertainty type is %s' % (data_name, uncertainty_type))