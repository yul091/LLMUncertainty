import os
import torch
from torch import nn, optim
from torch.nn import functional as F
from BasicalClass import (
    common_predict, 
    common_get_maxpos, 
)
from BasicalClass import BasicModule
from Metric import BasicUncertainty


class ModelWithTemperature(BasicUncertainty):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, instance: BasicModule, device, temperature=None):
        super(ModelWithTemperature, self).__init__(instance, device)
        if temperature is None:
            self.temperature = nn.Parameter(torch.ones(1) * 1.5)
            self.set_temperature(self.val_loader)
        else:
            self.temperature = temperature
        torch.save(self.temperature, os.path.join(
            instance.save_dir, instance.__class__.__name__, 'temperature.tmp'
        ))

    # modified forward for code summary task
    def forward(self, *input, **kwargs):
        # here since the model is code_summary model, the input has to be changed
        logits = self.model(*input, **kwargs)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        # First: collect all the logits and labels for the validation set
        logits, _, labels = common_predict(valid_loader, self.model, self.device, module_id=self.module_id)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = self.nll_criterion(logits, labels).item()
        before_temperature_ece = self.ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            loss = self.nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = self.nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = self.ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        return self

    def _uncertainty_calculate(self, data_loader):
        logits, preds, labels = common_predict(data_loader, self, self.device, module_id=self.module_id)
        uncertainty = common_get_maxpos(logits) # winning score as uncertainty
        return self.eval_uncertainty(logits, preds, labels, uncertainty)

