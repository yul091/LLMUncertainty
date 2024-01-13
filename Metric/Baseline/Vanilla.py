from BasicalClass import BasicModule
from BasicalClass import (
    common_get_maxpos, 
    common_predict,
)
from Metric import BasicUncertainty


class Vanilla(BasicUncertainty):
    def __init__(self, instance: BasicModule, device):
        super(Vanilla, self).__init__(instance, device)
        
    def forward(self, *input, **kwargs):
        return self.model(*input, **kwargs)

    def _uncertainty_calculate(self, data_loader):
        logits, preds, labels = common_predict(data_loader, self.model, self.device, module_id=self.module_id)
        uncertainty = common_get_maxpos(logits)
        return self.eval_uncertainty(logits, preds, labels, uncertainty)