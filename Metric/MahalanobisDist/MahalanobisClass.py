import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from BasicalClass import BasicModule
from BasicalClass import common_ten2numpy
from Metric import BasicUncertainty
from BasicalClass import IS_DEBUG, DEBUG_NUM


class Mahalanobis(BasicUncertainty):
    def __init__(self, instance: BasicModule, device):
        super(Mahalanobis, self).__init__(instance, device)
        self.hidden_num = 1
        self.u_list, self.std_value = self.preprocess(instance.train_loader)
        self.lr = self.train_logic(instance.train_loader, instance.train_truth)

    def train_logic(self, data_loader, ground_truth):
        train_res = self.extract_metric(data_loader)
        train_res = train_res.reshape([-1, self.hidden_num]) # N X D
        print("train_res: ", train_res.shape)
        lr = LogisticRegression(C=1.0, penalty='l2', tol=0.01)
        lr.fit(train_res, ground_truth)
        print(lr.score(train_res, ground_truth))
        return lr

    def preprocess(self, data_loader):
        fx, y = self.get_penultimate(data_loader) # N X V, N
        fx, y = fx.to(self.device), y.to(self.device)
        u_list, std_value = [], None
        for target in tqdm(range(self.class_num)): # int
            fx_tar = fx[y == target] # N' X V
            mean_val = fx_tar.mean(dim=0) # V
            std_val = (fx_tar - mean_val).transpose(dim0=0, dim1=1).mm((fx_tar - mean_val)) # V X V
            u_list.append(mean_val.cpu())
            # std_list.append(std_val)
            if std_value is None:
                std_value = std_val
            else:
                std_value += std_val
            
        # std_value = sum(std_list) / len(y)
        std_value /= len(y)
        std_value = torch.inverse(std_value)
        return u_list, std_value

    def get_penultimate(self, data_loader):
        pred_pos, pred_list, y_list = [], [], []
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            if self.module_id == 0: # code summary
                for i, ((sts, paths, eds), y, length) in tqdm(enumerate(data_loader), total=len(data_loader)):
                    torch.cuda.empty_cache()
                    sts = sts.to(self.device)
                    paths = paths.to(self.device)
                    eds = eds.to(self.device)
                    y = torch.tensor(y, dtype=torch.long)
                    output = self.model(sts, paths, eds, length, self.device)
                    _, pred_y = torch.max(output, dim=1)
                    # Detach
                    sts = sts.detach().cpu()
                    paths = paths.detach().cpu()
                    eds = eds.detach().cpu()
                    pred_y = pred_y.detach().cpu()
                    output = output.detach().cpu()

                    pred_list.append(pred_y)
                    pred_pos.append(output)
                    y_list.append(y)
                    
            elif self.module_id == 1: # code completion
                for i, (input, y, _) in tqdm(enumerate(data_loader), total=len(data_loader)):
                    torch.cuda.empty_cache()
                    input = input.to(self.device)
                    output = self.model(input)
                    _, pred_y = torch.max(output, dim=1)
                    # Detach
                    input = input.detach().cpu()
                    pred_y = pred_y.detach().cpu()
                    output = output.detach().cpu()
                    # Measure accuracy and record loss
                    pred_list.append(pred_y)
                    pred_pos.append(output)
                    y_list.append(y.long())
            else:
                raise TypeError()
        
        # return torch.cat(pred_list, dim=0), torch.cat(y_list, dim=0)
        return torch.cat(pred_pos, dim=0), torch.cat(y_list, dim=0)

    def extract_metric(self, data_loader, mini_batch: int = 1000):
        print("Extracting metric...")
        fx, _ = self.get_penultimate(data_loader) # N X V, N
        fx = fx.to(self.device)
        score = []
        for i in tqdm(range(0, fx.shape[0], mini_batch)):
            fx_batch = fx[i:i + mini_batch] # mini_batch X V
            score_batch = []
            for target in range(self.class_num):
                u = self.u_list[target].to(self.device) # V
                tmp = (fx_batch - u).mm(self.std_value) # mini_batch X V
                tmp = tmp.mm((fx_batch - u).transpose(dim0=0, dim1=1)) # mini_batch X mini_batch
                tmp = tmp.diagonal().reshape([-1, 1]) # mini_batch X 1
                score_batch.append(-tmp.cpu())
            score.append(torch.cat(score_batch, dim=1)) # mini_batch X V
                
        score = torch.cat(score, dim=0) # N X C
        score = common_ten2numpy(torch.max(score, dim=1)[0]) # N
        return score

    def _uncertainty_calculate(self, data_loader):
        metric = self.extract_metric(data_loader).reshape([-1, self.hidden_num])
        result = self.lr.predict_proba(metric)[:, 1] # N
        return result

