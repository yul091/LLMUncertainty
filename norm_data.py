import pickle
import json
import os
from tqdm import tqdm
from typing import List, Dict, Tuple
from preprocess.utils import BASEDICT


def build_dict(dataset: List[str]):
    base_dict = BASEDICT.copy()
    token2index, path2index, func2index = base_dict.copy(), base_dict.copy(), base_dict.copy()
    for i, data in tqdm(enumerate(dataset)):
        data = data.strip().lower()
        target, code_context = data.split()[0], data.split()[1:]
        target = target.split('|')
        func_name = target[0]
        if func_name not in func2index:
            func2index[func_name] = len(func2index)
        for context in code_context:
            st, path, ed = context.split(',')
            if st not in token2index:
                token2index[st] = len(token2index)
            if ed not in token2index:
                token2index[ed] = len(token2index)
            if path not in path2index:
                path2index[path] = len(path2index)
    with open(DIR + '/' + 'tk.pkl', 'wb') as f:
        pickle.dump([token2index, path2index, func2index], f)
    print("finish dictionary build", len(token2index), len(path2index), len(func2index))


def tk2index(tk_dict, k):
    if k not in tk_dict:
        return tk_dict['____UNKNOW____']
    return tk_dict[k]


def norm_data(data_type, suffix='.c2v'):
    file_name = DATA_DIR + '/'+ FILENAME + '.' + data_type + suffix
    with open(file_name, 'r') as f:
        dataset = f.readlines()
    with open(DIR + '/' + 'tk.pkl', 'rb') as f:
        token2index, path2index, func2index = pickle.load(f)
    newdataset = []
    for i, data in enumerate(dataset):
        data = data.strip().lower()
        target, code_context = data.split()[0], data.split()[1:]
        target = target.split('|')
        func_name = target[0]
        label = tk2index(func2index, func_name)
        newdata = []
        for context in code_context:
            st, path, ed = context.split(',')
            newdata.append(
                [tk2index(token2index, st), tk2index(path2index, path), tk2index(token2index, ed)]
            )
        newdataset.append([newdata, label])
    save_file = DIR + '/' + data_type + '.pkl'
    with open(save_file, 'wb') as f:
        pickle.dump(newdataset, f)
    print("finish normalize dataset", data_type)


def main(DATA_DIR, DIR, suffix='.c2v'):
    with open(DATA_DIR + '/' + FILENAME + '.train' + suffix, 'r') as f:
        dataset = f.readlines()
        print('dataset number is ', len(dataset))
    if not os.path.isdir(DIR):
        os.mkdir(DIR)
    build_dict(dataset)
    norm_data('train', suffix=suffix)
    norm_data('val', suffix=suffix)
    try:
        norm_data('test1', suffix=suffix)
        norm_data('test2', suffix=suffix)
        norm_data('test3', suffix=suffix)
    except:
        norm_data('test', suffix=suffix)
   


if __name__ == '__main__':
    file_dict = {
        'case_study': 'case_study',
        'different_author': 'different_author',
        'different_time': 'different_time',
        'different_project': 'different_project',
    }
    for FILENAME, GEN_FILE in file_dict.items():
        TRG_DIR = 'dataset/code_summary' 
        DIR = os.path.join(TRG_DIR, GEN_FILE)
        DATA_DIR = os.path.join(TRG_DIR, FILENAME)
        main(DATA_DIR, DIR, suffix='.c2v')
