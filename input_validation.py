import sys
sys.dont_write_bytecode = True
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Union, Any
import torch
from BasicalClass.common_function import *


class InputValidator:
    def __init__(self, metric_dir: str, save_dir: str, shift: str):
        self.shift = shift
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.vanilla = torch.load(os.path.join(metric_dir, 'Vanilla.res'))
        self.temp = torch.load(os.path.join(metric_dir, 'ModelWithTemperature.res'))
        self.pv = torch.load(os.path.join(metric_dir, 'PVScore.res'))
        self.dropout = torch.load(os.path.join(metric_dir, 'ModelActivateDropout.res'))
        self.mutation = torch.load(os.path.join(metric_dir, 'Mutation.res'))
        self.ensemble = torch.load(os.path.join(metric_dir, 'Ensemble.res'))
        # self.truth = torch.load(os.path.join(metric_dir, 'truth.res'))
        
        
    def get_filtered_stats(
        self, 
        UE_dict: Dict[str, Any],
        UE_name: str, 
        split: str, 
        res: Dict[str, Any],
        threshold: float = None,
        coverage: float = None,
        using_coverage: bool = False,
    ):
        all_scores = UE_dict[split]['UE_scores'] # List of scores
        for i, scores in enumerate(all_scores):
            original_size = len(scores)
            if using_coverage:
                remain_size = int(coverage * original_size)
                idx = np.argsort(scores)[::-1][:remain_size]
                # print(f"original size: {scores.shape}, remain size: {remain_size}, idx: {idx}")
                threshold = scores[idx[-1]]
            else:
                idx = np.where(scores >= threshold)[0]
                coverage = len(idx) / original_size
                
            if UE_name != 'mutation':
                truths = UE_dict[split]['truths'][0]
            else:
                truths = UE_dict[split]['truths'][i]
                
            auc = common_get_auc(truths[idx], scores[idx])
            aupr = common_get_aupr(truths[idx], scores[idx])
            brier = common_get_brier(truths[idx], scores[idx])
            acc = np.mean(truths[idx])
        
            res[UE_name][split]['coverage'].append(coverage)
            res[UE_name][split]['threshold'].append(threshold)
            res[UE_name][split][f'accuracy-{i}'].append(acc)
            res[UE_name][split][f'AUC-{i}'].append(auc)
            res[UE_name][split][f'AUPR-{i}'].append(aupr)
            res[UE_name][split][f'Brier-{i}'].append(brier)
        
            print('[{} set | {} sub-method {} | threshold {}]: coverage {:.2f}, acc {:.2f}, AUC {:.2f}, AUPR {:.2f}, Brier {:.2f}'.format(
                split, UE_name, i, threshold, coverage, acc, auc, aupr, brier,
            ))


    def filtering(self, res, split='dev', threshold=None, coverage=None, use_coverage=False):
        for ue_name, ue_dict in [
            ('vanilla', self.vanilla), 
            ('temperature', self.temp), 
            ('mutation', self.mutation), 
            ('dropout', self.dropout), 
            ('ensemble', self.ensemble),
            ('dissector', self.pv),
        ]:
            self.get_filtered_stats(
                UE_dict=ue_dict,
                UE_name=ue_name,
                split=split,
                res=res,
                threshold=threshold,
                coverage=coverage,
                using_coverage=use_coverage,
            )


    def run(self, strategy='threshold'):
        # evaluate on test dataset
        value_range = np.arange(0.01, 1.01, 0.01)
        if strategy == 'threshold':
            threshold = value_range
            coverage = None
        else:
            threshold = None
            coverage = value_range
        
        res = {
            'threshold': threshold, 
            'coverage': coverage,
            'vanilla': {}, 
            'temperature': {}, 
            'mutation': {},
            'dropout': {},
            'ensemble': {},
            'dissector': {},
        }

        if self.shift:
            splits = ['dev', 'test1', 'test2', 'test3']
        else:
            splits = ['dev', 'test']
            
        for split in splits:
            res['vanilla'][split] = defaultdict(list)
            res['temperature'][split] = defaultdict(list)
            res['mutation'][split] = defaultdict(list)
            res['dropout'][split] = defaultdict(list)
            res['ensemble'][split] = defaultdict(list)
            res['dissector'][split] = defaultdict(list)
            
            if strategy == 'threshold':
                for threshold in tqdm(value_range):
                    self.filtering(
                        res=res, 
                        split=split, 
                        threshold=threshold, 
                        coverage=coverage,
                        use_coverage=False,
                    )
            else:
                for coverage in tqdm(value_range):
                    self.filtering(
                        res=res, 
                        split=split, 
                        threshold=threshold, 
                        coverage=coverage,
                        use_coverage=True,
                    )
            
        # Save file 
        if strategy == 'threshold':
            torch.save(res, os.path.join(self.save_dir, 'filter.res'))
        else:
            torch.save(res, os.path.join(self.save_dir, 'filter_coverage.res'))



if __name__ == "__main__":
    import warnings
    # Turn off all warnings
    warnings.filterwarnings("ignore")
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--shift_type', type=str, default='different_time',
                        choices=[
                            'different_time', 
                            'different_author', 
                            'different_project',
                            'case_study'
                        ],
                        help='Type of code distribution shift')
    parser.add_argument('--task', type=str, default='code_summary',
                        choices=['code_summary', 'code_completion'],
                        help='Type of program task')
    parser.add_argument('--model', type=str, default='codebert')
    parser.add_argument('--uncertainty_dir', type=str, default='Uncertainty_Results',
                        help='Directory where the uncertainty results are stored')
    parser.add_argument('--out_dir', type=str, default='Uncertainty_Eval/filter',
                        help='Directory where the filtering results are stored')
    parser.add_argument('--strategy', type=str, default='threshold',
                        choices=['threshold', 'coverage'],
                        help='Type of filtering strategy')
    
    args = parser.parse_args()
    task = args.task
    shift_type = args.shift_type
    model_type = args.model
    strategy = args.strategy
    uncertainty_dir = args.uncertainty_dir
    out_dir = args.out_dir
    module = 'CodeSummary_Module' if task == 'code_summary' else 'CodeCompletion_Module'
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    metric_dir = f'{uncertainty_dir}/{shift_type}/{model_type}/{module}'
    save_dir = f'{out_dir}/{shift_type}/{model_type}/{module}'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    shift = True if shift_type != 'case_study' else False
    filter = InputValidator(
        metric_dir=metric_dir,
        save_dir=save_dir, 
        shift=shift,
    )

    filter.run(strategy=strategy)

    # Turn on warnings again
    warnings.filterwarnings("default")





