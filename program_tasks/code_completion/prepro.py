import os
import sys
sys.dont_write_bytecode = True
import javalang
from tqdm import tqdm
import torch
from program_tasks.code_completion.util import create_tsv_file
from collections import defaultdict
import tokenize
import random


def parse_java(src_folder: str, dest_dir: str, dest_file_name: str, token_dict: dict):
    """
    src_folder: data/main/different_project/train
    dest_dir: dataset/code_completion/different_project
    dest_file_name: train.txt
    token_dict: {}
    """
    token_dict[dest_file_name] = defaultdict(int)

    with open(os.path.join(dest_dir, dest_file_name), 'w') as write_file:
        for f in os.listdir(src_folder):
            subfolder = os.path.join(src_folder, f)
            if os.path.isdir(subfolder): # train/project_name/java
                print('Tokenizing java snippets in {} ...'.format(subfolder))
                project_file_list = os.listdir(subfolder)
                if 'different_time' in dest_dir or 'different_project' in dest_dir:
                    print("[TIMELINE SHIFT] only using 10% of files !")
                    project_file_list = random.sample(project_file_list, int(len(project_file_list) * 0.1))
                for file_path in tqdm(project_file_list):
                    if file_path.endswith(".java"):
                        try:
                            file = open(os.path.join(subfolder, file_path), 'r', encoding='utf-8')
                            file_string = ' '.join(file.read().splitlines()) # read in oneline
                        except:
                            file = open(os.path.join(subfolder, file_path), 'r', encoding='iso-8859-1')
                            file_string = ' '.join(file.read().splitlines()) # read in oneline
                        try:
                            tokens = list(javalang.tokenizer.tokenize(file_string)) # ast java parse
                        except:
                            print('----------------------------Error string----------------------------')
                            continue
                        # print('token list: \n', [x.value for x in tokens])
                        for tok in tokens:
                            token_dict[dest_file_name][tok.value] += 1
                        
                        token_str = " ".join([x.value for x in tokens])
                        # write new line each time, unicode escape
                        write_file.write(
                            token_str.encode('unicode_escape').decode('utf-8') + '\n')
                        
            else: # project_name/java
                if subfolder.endswith(".java"):
                    file = open(os.path.join(subfolder, file_path), 'r')
                    file_string = ' '.join(file.read().splitlines()) # read in oneline
                    tokens = list(javalang.tokenizer.tokenize(file_string)) # ast java parse
                    for tok in tokens:
                        token_dict[dest_file_name][tok.value] += 1

                    token_str = " ".join([x.value for x in tokens])
                    # write new line each time, unicode escape
                    write_file.write(
                        token_str.encode('unicode_escape').decode('utf-8') + '\n')

        write_file.close()



def parse_python(src_folder, dest_dir, dest_file_name, token_dict):
    token_dict[dest_file_name] = defaultdict(int)

    with open(os.path.join(dest_dir, dest_file_name), 'w') as write_file:
        for f in os.listdir(src_folder):
            subfolder = os.path.join(src_folder, f)
            if os.path.isdir(subfolder): 
                print('tokenizing java code in {} ...'.format(subfolder))
                for file_path in tqdm(os.listdir(subfolder)):
                    if file_path.endswith(".py"):
                        with open(os.path.join(subfolder, file_path), 'rb') as f:
                            tokens = tokenize.tokenize(f.readline)
                            # print('token list: \n', [x.value for x in tokens])
                            token_str = " ".join([tok.string for tok in tokens])
                            # write new line each time, unicode escape
                            write_file.write(
                                token_str.encode('unicode_escape').decode('utf-8') + '\n'
                            )
            else: # project_name/java
                if subfolder.endswith(".py"):
                    with open(subfolder, 'rb') as f:
                        tokens = tokenize.tokenize(f.readline)
                        # print('token list: \n', [x.value for x in tokens])
                        token_str = " ".join([x.string for x in tokens])
                        # write new line each time, unicode escape
                        write_file.write(
                            token_str.encode('unicode_escape').decode('utf-8') + '\n'
                        )
            
        write_file.close()



if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='data/main/different_project', help='Input dataset directory')
    parser.add_argument("--dest_dir", type=str, default='dataset/code_completion/different_project', help='Output dataset directory')
    parser.add_argument("--language", default='java', type=str, help='Language of the dataset')
    args = parser.parse_args()
    ###############################################################################
    random.seed(42)
    
    # Handle java files
    data_dir = args.data_dir
    dest_dir = args.dest_dir
    language = args.language
    data_type = ['train', 'dev', 'test1', 'test2', 'test3'] \
        if 'different' in data_dir else ['train', 'dev', 'test']
    # data_dir = 'data/case_study'
    # data_type = ['train', 'val', 'test']
    java_dict = {
        k + '.txt': os.path.join(data_dir, k) # 'train': data_dir/train/
        for k in data_type
    }
    token_dict = {} # save token hist in dict
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for name, src in java_dict.items(): # Generate .txt files
        if language == 'java':
            parse_java(src, dest_dir, name, token_dict)
        elif language == 'python':
            # parse_python(src, dest_dir, name, token_dict)
            # copy original file to dest dir
            os.system(f"cp {src}.txt {dest_dir}/")

    for name in java_dict:
        origin_file = os.path.join(dest_dir, name)
        dest_file = origin_file.rstrip('.txt') + '.tsv'
        print(f"origin_file: {origin_file}, dest file: {dest_file}")
        create_tsv_file(origin_file, dest_file)
        # Generate .tsv files

    # save token dict
    torch.save(token_dict, os.path.join(dest_dir, 'token_hist.res'))
    ###############################################################################
    
    ###############################################################################
    # # Handle python files
    # data_dir = 'python_data'
    # data_type = ['train', 'val', 'test1', 'test2', 'test3']
    # java_dict = {
    #     k + '.txt': os.path.join(data_dir, k) # 'train': data_dir/train/
    #     for k in data_type
    # }

    # dest_dir = "data/code_completion/python_project"
    # token_dict = {} # save token hist in dict
    # if not os.path.exists(dest_dir):
    #     os.makedirs(dest_dir)

    # for name, src in java_dict.items():
    #     parse_python(src, dest_dir, name, token_dict)

    # for name in java_dict:
    #     origin_file = os.path.join(dest_dir, name)
    #     dest_file = origin_file.rstrip('.txt') + '.tsv'
    #     create_tsv_file(origin_file, dest_file)

    # # save token dict
    # torch.save(token_dict, os.path.join(dest_dir, 'token_hist.res'))
    ###############################################################################