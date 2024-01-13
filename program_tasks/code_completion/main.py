from __future__ import print_function
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, '/home/yuli/UncertaintyPL/')
import os
import shutil
import argparse
import datetime
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from preprocess.checkpoint import Checkpoint
from transformers import (
    RobertaConfig,
    GPT2Config,
    LlamaConfig,
)
from program_tasks.code_completion.vocab import VocabBuilder
from program_tasks.code_completion.dataloader import Word2vecLoader
from program_tasks.code_completion.util import AverageMeter, accuracy, adjust_learning_rate
from models.code_analysis.model_cc import (
    Code2vecForClassification,
    CodeBertaForClassification,
    GraphCodeBertForClassification,
    BiLSTMForClassification,
    CodeBertForClassification, 
    CodeGPTForClassification,
    CodeLlamaForClassification,
)


def preprocess_data():
    print("===> creating vocabs ...")
    train_path = args.train_data
    val_path = args.val_data
    if args.test_data is None:
        test_path1 = args.test_data1
        test_path2 = args.test_data2
        test_path3 = args.test_data3
    else:
        test_path = args.test_data
    
    pre_embedding_path = args.embedding_path
    if args.embedding_type == 0:
        d_word_index, embed = torch.load(pre_embedding_path)
        print('load existing embedding vectors, name is ', pre_embedding_path)
    elif args.embedding_type == 1:
        v_builder = VocabBuilder(path_file=train_path)
        d_word_index, embed = v_builder.get_word_index(min_sample=args.min_samples)
        print('create new embedding vectors, training from scratch')
    elif args.embedding_type == 2:
        v_builder = VocabBuilder(path_file=train_path)
        d_word_index, embed = v_builder.get_word_index(min_sample=args.min_samples)
        embed = torch.randn([len(d_word_index), args.embedding_dim]).cuda()
        print('create new embedding vectors, training the random vectors')
    else:
        raise ValueError('unsupported type')

    if embed is not None:
        if type(embed) is np.ndarray:
            embed = torch.tensor(embed, dtype=torch.float).cuda()
        assert embed.size()[1] == args.embedding_dim

    if not os.path.exists(args.res_dir):
        os.mkdir(args.res_dir)

    train_loader = Word2vecLoader(train_path, d_word_index, batch_size=args.batch_size)
    val_loader = Word2vecLoader(val_path, d_word_index, batch_size=args.batch_size)
    if args.test_data is None:
        val_loader1 = Word2vecLoader(test_path1, d_word_index, batch_size=args.batch_size)
        val_loader2 = Word2vecLoader(test_path2, d_word_index, batch_size=args.batch_size)
        val_loader3 = Word2vecLoader(test_path3, d_word_index, batch_size=args.batch_size)
        return d_word_index, embed, train_loader, val_loader, \
            val_loader1, val_loader2, val_loader3
    else:
        test_loader = Word2vecLoader(test_path, d_word_index, batch_size=args.batch_size)
        return d_word_index, embed, train_loader, val_loader, test_loader


def train(train_loader, model, criterion, optimizer):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()
    for i, (input, target, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # print("input ({}): ".format(input.size(), input))
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0][0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()


def test(val_loader, model, val_name):
    top1 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    for i, (input, target, _) in tqdm(enumerate(val_loader)):
        input = input.cuda()
        target = target.cuda()
        # compute output
        output = model(input)
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        top1.update(prec1[0][0], input.size(0))
    try:
        res = {f'{val_name} acc': top1.avg.item()}
    except:
        res = {f'{val_name} acc': top1.avg}
    return res


def main(args):
    try:
        d_word_index, embed, train_loader, val_loader, \
            test_loader1, test_loader2, test_loader3 = preprocess_data()
    except:
        d_word_index, embed, train_loader, val_loader, test_loader = preprocess_data()
    vocab_size = len(d_word_index)
    print('vocab size: {}, num of batches: {}'.format(vocab_size, len(train_loader)))

    # load ckpt if necessary
    pretrained_model = None
    if args.load_ckpt:
        latest_checkpoint_path = Checkpoint.get_latest_checkpoint(args.res_dir)
        resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
        model = resume_checkpoint.model
        optimizer = resume_checkpoint.optimizer
        start_epoch = resume_checkpoint.epoch
    else:
        if args.model_type == 'lstm':
            model = BiLSTMForClassification(vocab_size, embed, hidden_size=args.embedding_dim)
        elif args.model_type == 'codebert':
            config_class = RobertaConfig
            pretrained_model = 'microsoft/codebert-base'
            config = config_class.from_pretrained(
                pretrained_model, num_labels=vocab_size, 
                use_cache=False, hidden_size=args.embedding_dim,
            )
            model = CodeBertForClassification(config)
        elif args.model_type == 'codegpt':
            config_class = GPT2Config
            pretrained_model = 'microsoft/CodeGPT-small-java-adaptedGPT2'
            config = config_class.from_pretrained(
                pretrained_model, num_labels=vocab_size, 
                use_cache=False, n_embd=args.embedding_dim,
            )
            model = CodeGPTForClassification(config)
        elif args.model_type == 'code2vec':
            model = Code2vecForClassification(vocab_size, embed, hidden_size=args.embedding_dim)
        elif args.model_type == 'codeberta':
            config_class = RobertaConfig
            pretrained_model = 'huggingface/CodeBERTa-small-v1'
            config = config_class.from_pretrained(
                pretrained_model, num_labels=vocab_size, 
                use_cache=False, hidden_size=args.embedding_dim,
            )
            model = CodeBertaForClassification(config)
        elif args.model_type == 'graphcodebert':
            config_class = RobertaConfig
            pretrained_model = 'microsoft/graphcodebert-base'
            config = config_class.from_pretrained(
                pretrained_model, num_labels=vocab_size, 
                use_cache=False, hidden_size=args.embedding_dim,
            )
            model = GraphCodeBertForClassification(config)
        elif args.model_type == 'codellama':
            config_class = LlamaConfig
            pretrained_model = 'codellama/CodeLlama-7b-hf'
            config = config_class.from_pretrained(
                pretrained_model, num_labels=vocab_size, 
                use_cache=False, hidden_size=args.embedding_dim,
            )
            model = CodeLlamaForClassification(config)
        else:
            raise TypeError('Undefined Model Type!')
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        start_epoch = 1

    model = model.cuda()
    
    if args.do_train:
        criterion = nn.CrossEntropyLoss()
        print(f'Training dataset size: {train_loader.n_samples}')
        time_cost = None
        best_val_acc = 0
        best_ckpt_dir = None
        t1 = datetime.datetime.now()

        for epoch in range(start_epoch, args.epochs + 1):
            st = datetime.datetime.now()
            train(train_loader, model, criterion, optimizer)
            ed = datetime.datetime.now()
            if time_cost is None:
                time_cost = ed - st
            else:
                time_cost += (ed - st)

            print(epoch, 'cost time', ed - st)
            res_val = test(val_loader, model, 'dev')
            if args.test_data is None:
                res1 = test(test_loader1, model, 'test1')
                res2 = test(test_loader2, model, 'test2')
                res3 = test(test_loader3, model, 'test3')
                merge_res = {**res_val, **res1, **res2, **res3} # merge all the test results
            else:
                res_test = test(test_loader, model, 'test')
                merge_res = {**res_val, **res_test}
            print(merge_res)

            # Save best model checkpoint
            if res_val['dev acc'] > best_val_acc:
                if best_ckpt_dir is not None:
                    shutil.rmtree(best_ckpt_dir)
                Checkpoint(model, optimizer, epoch, merge_res).save(args.res_dir)
                best_val_acc = res_val['dev acc']
                best_ckpt_dir = Checkpoint.get_latest_checkpoint(args.res_dir)

        print('Avg time cost', time_cost / args.epochs)
        t2 = datetime.datetime.now()
        print('Total time cost', t2 - t1)
    else:
        print(f" ** Using zero-shot model {pretrained_model} for evaluation **")
        
    if args.do_eval:
        model.eval()
        res_val = test(val_loader, model, 'dev')
        if args.test_data is None:
            res1 = test(test_loader1, model, 'test1')
            res2 = test(test_loader2, model, 'test2')
            res3 = test(test_loader3, model, 'test3')
            merge_res = {**res_val, **res1, **res2, **res3} # merge all the test results
        else:
            res_test = test(test_loader, model, 'test')
            merge_res = {**res_val, **res_test}
        print(merge_res)


if __name__ == '__main__':
    import random
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch_size', default=2048, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument('--embedding_dim', default=120, type=int, metavar='N', help='embedding size')
    parser.add_argument('--model_type', default='codebert', type=str, 
                        choices=['codebert', 'lstm', 'codegpt', 'code2vec', 'codeberta', 'graphcodebert', 'codellama'], 
                        help='model architecture')
    parser.add_argument('--layers', default=2, type=int, metavar='N', help='number of rnn layers')
    parser.add_argument('--min_samples', default=5, type=int, metavar='N', help='min number of tokens')
    parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
    parser.add_argument('--rnn', default='LSTM', choices=['LSTM', 'GRU'], help='rnn module type')
    parser.add_argument('--mean_seq', default=False, action='store_true', help='use mean of rnn output')
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--weight_name', type=str, default='1', help='model name')
    parser.add_argument('--embedding_path', type=str, default='embedding_vec100_1/fasttext.vec')
    parser.add_argument('--ensemble_models', type=int, default=1, help='number of ensemble models')
    parser.add_argument('--train_data', type=str, default='data/code_completion/different_time/train.tsv',)
    parser.add_argument('--do_train', action='store_true', help='do training')
    parser.add_argument('--do_eval', action='store_true', help='do evaluation')
    parser.add_argument('--val_data', type=str, default='data/code_completion/different_time/dev.tsv', help='model name')
    parser.add_argument('--test_data', type=str, default=None, help='model name')
    parser.add_argument('--test_data1', type=str, default=None, help='model name')
    parser.add_argument('--test_data2', type=str, default=None, help='model name')
    parser.add_argument('--test_data3', type=str, default=None, help='model name')
    parser.add_argument('--embedding_type', type=int, default=1, choices=[0, 1, 2])
    parser.add_argument('--experiment_name', type=str, default='code_completion')
    parser.add_argument('--res_dir', type=str, default=None)
    parser.add_argument('--load_ckpt', default=False, action='store_true', help='use pretrained model')
    args = parser.parse_args()

    if args.ensemble_models > 1:
        for i in range(args.ensemble_models):
            # Optionally set a different seed for each training to ensure diversity
            random.seed(i)
            print(f'Training ensemble model {i} ...')
            original_res_dir = args.res_dir
            args.res_dir = os.path.join(original_res_dir, f'ensemble_model-{i}')
            if not os.path.exists(args.res_dir):
                os.makedirs(args.res_dir)
            main(args)
            args.res_dir = original_res_dir
    else:
        main(args)