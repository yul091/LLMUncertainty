import os
import re
import json
import javalang
import logging
from tqdm import tqdm
import argparse
from collections import Counter
from argparse import Namespace
from io import TextIOWrapper
import numpy as np


lits = json.load(open("literals.json"))

def process_string(token, special_chars={" ": "U+0020", ",": "U+002C"}):
    str_quote_options = ["'''", '"""', "'", '"']
    start_quote = ""
    end_quote = ""
    qualifier_regex = r"^[a-z]+"
    qualifier_match = re.search(qualifier_regex, token)
    # string qualifiers like 'r' for regex, 'f' for formatted string, 'b' for bytes, 'u' for unicode, etc (or combination of them)
    qualifier = "" if not qualifier_match else qualifier_match[0]
    # token string without qualifiers
    token_string = re.sub(qualifier_regex, "", token)
    # string literal without quotes
    str_lit = token_string
    for q in str_quote_options:
        if token_string.startswith(q):
            start_quote = q
            str_lit = str_lit[len(q) :]
            if token_string.endswith(q):
                end_quote = q
                str_lit = str_lit[: -len(q)]
            break
    use_char = False
    if len(str_lit) == 1 and start_quote == "'":
        use_char = True
    for sc in special_chars:
        str_lit = str_lit.replace(sc, special_chars[sc])
    if not use_char:
        ret = (
            f"{qualifier}{start_quote}<STR_LIT:{str_lit}>{end_quote}"
            if str_lit in lits['str']
            else f"{qualifier}{start_quote}<STR_LIT>{end_quote}"
        )
    else:
        ret = (
            f"{qualifier}{start_quote}<CHAR_LIT:{str_lit}>{end_quote}"
            if str_lit in lits['char']
            else f"{qualifier}{start_quote}<CHAR_LIT>{end_quote}"
        )
    return ret


def preprocess_and_statisitics(file_dir: str, file_name: str, wf: TextIOWrapper):
    with open(os.path.join(file_dir, file_name), 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()

    num_tokens = 0
    num_files = 0
    num_functions = 0
    file_token_counter = {}

    try:
        # Tokenize the entire content
        tokens = list(javalang.tokenizer.tokenize(content))
        new_data = []

        for tok in tokens:
            if "String" in str(type(tok)) or "Character" in str(type(tok)):
                token = process_string(tok.value)
            elif "Integer" in str(type(tok)) or "FloatingPoint" in str(type(tok)):
                if tok.value in lits['num']:
                    token = f"<NUM_LIT:{tok.value}>"
                else:
                    token = "<NUM_LIT>"
            else:
                token = tok.value
            new_data.append(token)
            
        file_token_counter = Counter(new_data)
        num_tokens += len(new_data)
        data = "<s> " + " ".join(new_data) + " </s>"
        wf.write(data + "\n")

    except Exception as e:
        print(f"Error processing {file_name}: {e}")
    
    # Count Java files
    if file_name.endswith(".java"):
        num_files += 1

        # Count number of functions using javalang parser
        try:
            tree = javalang.parse.parse(content)
            num_functions += len(list(tree.filter(javalang.tree.MethodDeclaration)))
        except Exception as e:
            print(f"Error parsing {file_name} for function count: {e}")

    return num_tokens, num_files, num_functions, file_token_counter



def preprocess(args: Namespace, file_name: str, file_type: str):
    wf = open(os.path.join(args.output_dir, file_name), 'w')
    total_tokens = 0
    total_files = 0
    total_functions = 0
    token_counter = Counter()
    
    print(f"Processing {file_type}...")
    project_dir = os.path.join(args.base_dir, file_type)
    for java_project in os.listdir(project_dir):
        print(f"Java project: {java_project}")
        file_dir = os.path.join(project_dir, java_project)
        for file in tqdm(os.listdir(file_dir)):
            if file.endswith(".java"):
                num_tokens, num_files, num_functions, file_token_counter = preprocess_and_statisitics(file_dir, file, wf)
                total_tokens += num_tokens
                total_files += num_files
                total_functions += num_functions
                token_counter += file_token_counter
            
    logging.info(f"{file_type} is done!")
    logging.info(f"#Tokens: {total_tokens}, #Files: {total_files}, #Functions: {total_functions}")
    wf.close()
    return token_counter


def counter_to_prob_dist(counter):
    total_tokens = sum(counter.values())
    prob_dist = {token: count / total_tokens for token, count in counter.items()}
    return prob_dist


# In practice, there might be tokens present in P that are not in Q and vice-versa. 
# To handle this, we assume a very small probability for missing tokens.
def kl_divergence(P, Q):
    epsilon = 1e-10  # Small value to handle log(0) and division by zero
    kl_div = 0
    all_tokens = set(P.keys()) | set(Q.keys())

    for token in all_tokens:
        p = P.get(token, epsilon)
        q = Q.get(token, epsilon)
        kl_div += p * np.log(p / q)

    return kl_div
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="case_study", type=str, 
                        help="The downloaded data path")
    parser.add_argument("--output_dir", default="preprocessed/case_study", type=str, 
                        help="The output directory")
    parser.add_argument("--train_file", default=None, type=str, 
                        help="The train file name")
    parser.add_argument("--dev_file", default=None, type=str, 
                        help="The dev file name")
    parser.add_argument("--test_file", default=None, type=str, 
                        help="The shifted file name")
    parser.add_argument("--test1_file", default=None, type=str, 
                        help="The shifted1 file name")
    parser.add_argument("--test2_file", default=None, type=str, 
                        help="The shifted2 file name")
    parser.add_argument("--test3_file", default=None, type=str, 
                        help="The shifted3 file name")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    logging.basicConfig(
        filename=os.path.join(args.output_dir, 'statistics.log'), 
        filemode='w', # overwrite / append is 'a' (default)
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    if args.train_file is not None:
        train_counter = preprocess(args, file_name=args.train_file, file_type="train")
        train_prob_dist = counter_to_prob_dist(train_counter) # convert counters to probability distributions
    if args.dev_file is not None:
        dev_counter = preprocess(args, file_name=args.dev_file, file_type="dev")
        dev_prob_dist = counter_to_prob_dist(dev_counter)
        if args.train_file is not None:
            kl_train_dev = kl_divergence(train_prob_dist, dev_prob_dist)
            logging.info(f"KL divergence between train and dev: {kl_train_dev}")
    if args.test_file is not None:
        test_counter = preprocess(args, file_name=args.test_file, file_type="test")
        test_prob_dist = counter_to_prob_dist(test_counter)
        if args.train_file is not None:
            kl_train_test = kl_divergence(train_prob_dist, test_prob_dist)
            logging.info(f"KL divergence between train and test: {kl_train_test}")
    if args.test1_file is not None:
        test1_counter = preprocess(args, file_name=args.test1_file, file_type="test1")
        test1_prob_dist = counter_to_prob_dist(test1_counter)
        if args.train_file is not None:
            kl_train_test1 = kl_divergence(train_prob_dist, test1_prob_dist)
            logging.info(f"KL divergence between train and test1: {kl_train_test1}")
    if args.test2_file is not None:
        test2_counter = preprocess(args, file_name=args.test2_file, file_type="test2")
        test2_prob_dist = counter_to_prob_dist(test2_counter)
        if args.train_file is not None:
            kl_train_test2 = kl_divergence(train_prob_dist, test2_prob_dist)
            logging.info(f"KL divergence between train and test2: {kl_train_test2}")
    if args.test3_file is not None:
        test3_counter = preprocess(args, file_name=args.test3_file, file_type="test3")
        test3_prob_dist = counter_to_prob_dist(test3_counter)
        kl_train_test3 = kl_divergence(train_prob_dist, test3_prob_dist)
        logging.info(f"KL divergence between train and test3: {kl_train_test3}")