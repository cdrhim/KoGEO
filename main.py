import torch
from torch.utils.data import (DataLoader, RandomSampler)
import argparse
import logging
import os

import json
import torch_optimizer as optim

import time
import torch.nn.functional as F
# from preprocess_model import LCSTSProcessor
from preprocess import LCSTSProcessor

from model import MTL

import random
from scheduler import LinearWarmUpScheduler,EnDeDiffWarmup

import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

import sys
import pandas as pd
from tqdm import tqdm

from transformers import AutoConfig,AutoTokenizer, AutoModel
from transformers import AdamW

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "auto": (AutoConfig, AutoModel, AutoTokenizer)
}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
parser = argparse.ArgumentParser()
# Required parameters
parser.add_argument("--data_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The input data path. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--model_path",
                    default='/',
                    type=str,
                    required=False,
                    help="The pretrained model path. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--pretrained_lm",
                    default=None,
                    type=str,
                    required=True,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                         "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                         "bert-base-multilingual-cased, bert-base-chinese.")
parser.add_argument("--model_type",
                    default='GEO',
                    type=str,
                    required=False,
                    help="GEO, aux, base, MTL_generation, MTL_classification, classification")
parser.add_argument("--task_name",
                    default=None,
                    type=str,
                    required=True,
                    help="math word problem solving task name , alg514, draw, mawps, Math23k, CC, IL")
parser.add_argument("--encoder_parameter_initialization",
                    action='store_true',
                    help="Decide whether to update the parameters of the encoder. True : encoder parameter initialization , not using pre-trained parameter.")

parser.add_argument("--GPU_index",
                    default='-1',
                    type=str,
                    help="Designate the GPU index that you desire to use. Should be str. -1 for using all available GPUs.")
parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--encoder_learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial encoder learning rate for Adam.")
parser.add_argument("--decoder_learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial decoder learning rate for Adam.")
parser.add_argument("--num_train_epochs",
                    default=100,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion",
                    default=0,
                    type=str,
                    help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
parser.add_argument("--max_src_len",
                    default=100,
                    type=int,
                    help="Max sequence length for source text. Sequences will be truncated or padded to this length")
parser.add_argument("--eval_interval",
                    default=10,
                    type=int,
                    help="Duration of evaluation")
parser.add_argument("--max_tgt_len",
                    default=50,
                    type=int,
                    help="Max sequence length for target text. Sequences will be truncated or padded to this length")
parser.add_argument("--beam_size",
                    default=3,
                    type=int,
                    help="Beam search size")
parser.add_argument("--best_beam",
                    default=3,
                    type=int,
                    help="Best beam size")
parser.add_argument("--numberofdecoder",
                    default=3,
                    type=int,
                    help="Number of decoders")
parser.add_argument("--decoder_head",
                    default=8,
                    type=int,
                    help="Number of decoders")
parser.add_argument("--decoder_feedforwardL",
                    default=8,
                    type=int,
                    help="decoder feed forward layers")
parser.add_argument("--train_batch_size",
                    default=8,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--decoder_config",
                    default=None,
                    type=str,
                    help="Configuration file for decoder. Must be in JSON format.")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--optimizer", default="WAdam", type=str, )
parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")

parser.add_argument("--print_every",
                    default=10,
                    type=int,
                    help="Print loss every k steps.")
parser.add_argument("--window_size",
                    default=3,
                    type=int,
                    help="INC window size.")
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--output_dir",
                    default=None,
                    type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--test_title",
                    default='test.tsv',
                    type=str,
                    help="")
parser.add_argument("--do_train",
                    action='store_true',
                    help="")
parser.add_argument("--do_label_smoothing",
                    action='store_true',
                    help="")
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument("--save_every",
                    action='store_true',
                    help="")
parser.add_argument('--seed',
                    type=int,
                    default=2,
                    help="random seed for initialization")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cal_performance(logits, ground, smoothing=True, rewards=None):
    ground = ground[:, 1:]
    if rewards is not None:
        rewards = rewards.view(-1, 1).expand(-1, logits.shape[1])
    logits = logits.view(-1, logits.size(-1))
    ground = ground.contiguous().view(-1)
    loss = cal_loss(logits, ground, smoothing=smoothing, rewards=rewards)

    pad_mask = ground.ne(processor.indexes['PAD'])
    pred = logits.max(-1)[1]
    correct = pred.eq(ground)
    correct = correct.masked_select(pad_mask).sum().item()
    return loss, correct


def cal_loss(logits, ground, smoothing=True, rewards=None):
    def label_smoothing(logits, labels, rewards=None):
        eps = 0.1
        num_classes = logits.size(-1)
        one_hot = torch.zeros_like(logits).scatter(1, labels.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (num_classes - 1)
        log_prb = F.log_softmax(logits, dim=1)
        if rewards is not None:
            one_hot *= rewards.reshape(-1, 1)
        non_pad_mask = labels.ne(processor.indexes['PAD'])
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()
        return loss

    if smoothing:
        loss = label_smoothing(logits, ground, rewards=rewards)
    else:
        loss = F.cross_entropy(logits, ground, ignore_index=processor.indexes['PAD'])
    return loss

def prediction(model,batch, bs, nb, greedy=False, train=False,n_gpu=2):
    if not greedy:
        if processor.type != 'base':
            if n_gpu > 1:
                pred, scores = model.module.beam_decode(batch[0], batch[1],batch[4], batch[5],
                                                        beam_size=bs, n_best=nb)
            else:
                pred, scores = model.beam_decode(batch[0], batch[1], batch[4], batch[5],
                                                 beam_size=bs, n_best=nb)
        else:
            if n_gpu > 1:
                pred, scores = model.module.beam_decode(batch[0], batch[1],beam_size=bs,n_best=nb)
            else:
                pred, scores = model.beam_decode(batch[0], batch[1],beam_size=bs, n_best=nb)
    elif greedy:
        if n_gpu > 1:
            pred, scores = model.module.greedy_decode(batch[0], batch[1])
        else:
            pred, scores = model.greedy_decode(batch[0], batch[1])

    outpred = []
    for subPred in pred:
        temp = []
        for p in subPred:
            p += [0] * (args.max_tgt_len - len(p))
            temp.append(p)
        outpred.append(temp)
    return outpred, scores

def evaluation(model, eval_dataloader, beam_size, best_beam, device = None, tokenizer= None, n_gpu =None):
    model.eval()

    evaluation_log = {}

    hyp_list = []
    ref_list = []
    src_list = []
    idx_list = []

    answers = []
    topk_answers = []

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)

            pred, scores = prediction(model,batch, beam_size, best_beam, n_gpu=n_gpu)
            src, tgt = batch[0], batch[2]

            for ti in range(len(batch[0])):
                sample_src = \
                    " ".join(tokenizer.convert_ids_to_tokens(src[ti].cpu().numpy())).replace('▁', ' ') + '\n'
                sample_tgt = \
                    " ".join(tokenizer.convert_ids_to_tokens(tgt[ti].cpu().numpy())).split(tokenizer.cls_token)[1].split(tokenizer.sep_token)[0] + '\n'
                sample_preds = [
                    " ".join(tokenizer.convert_ids_to_tokens(pred[ti][pt])).split(tokenizer.sep_token)[0].replace(
                        '▁', '').replace('##', '')
                    + '\n'
                    for pt in range(len(pred[ti]))]

                hyp_list.append(sample_preds[0])
                ref_list.append(sample_tgt[1:])
                src_list.append(sample_src)
                idx_list.append(str(int(batch[-1][ti].cpu())))

                solvings = []
                if n_gpu > 1:
                    for i in range(best_beam):
                        solving, _, _ = model.module.checkAnswer(torch.tensor(pred[ti][i]), tgt[ti], batch[-1][ti],
                                                                 processor)
                        solvings.append(solving)

                else:
                    for i in range(best_beam):
                        solving, _, _ = model.checkAnswer(torch.tensor(pred[ti][i]), tgt[ti], batch[-1][ti], processor)
                        solvings.append(solving)

                answers.append(solvings[0])

                if True in solvings:
                    topk_answers.append(True)
                else:
                    topk_answers.append(False)

    evaluation_log["equation_accuracy"] = sum(np.array(answers)) / len(ref_list)
    evaluation_log["equation_topk_accuracy"] = np.sum(topk_answers) / len(ref_list)

    evaluation_log["matching"] = answers
    evaluation_log["matching_top{}".format(beam_size)] = topk_answers

    evaluation_log["hyp_list"] = hyp_list
    evaluation_log["ref_list"] = ref_list
    evaluation_log["src_list"] = src_list
    evaluation_log["idx_list"] = idx_list

    return evaluation_log

def main(args):
    # Initialize tensorboard writer. tensorboard is visualization tool for debuging.
    # You can use tensorboard writer by typing tensorboard --logdir ./savefile in cmd console.
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir,
                                                   time.strftime('model_%m-%d-%H:%M:%S',
                                                                 time.localtime()) + "_{}_".format(valid_num) + experiment_name))

    # Gpu setting
    # If you want to use single GPU, you can set local_rank = number of index(ex. 0 or 1)
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() and not args.no_cuda else 0
    else:
        torch.cuda.set_device(args.local_rank)

        device = torch.device("cuda", args.local_rank)
        n_gpu = 1

    logger.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))

    # Set random seed.
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory. You can change output directory name by replacing model_path.
    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        model_path = os.path.join(args.output_dir, time.strftime('model_%m-%d-%H:%M:%S',
                                                                 time.localtime()) + "_{}_".format(valid_num) + experiment_name)
        os.makedirs(model_path, exist_ok=True)
        logger.info(f'Saving model to {model_path}.')

    # Load existing config or create decoder config.
    if args.decoder_config is not None:
        with open(args.decoder_config, 'r') as f:
            decoder_config = json.load(f)
    else:
        if os.path.isdir(args.pretrained_lm):
            with open(os.path.join(args.pretrained_lm, 'bert_config.json'), 'r') as f:
                lm_config = json.load(f)
        else:
            lm_config = AutoConfig.from_pretrained(args.pretrained_lm)

        decoder_config = {}

        decoder_config['len_max_seq'] = args.max_tgt_len
        decoder_config['d_model'] = lm_config.hidden_size
        decoder_config['n_layers'] = args.numberofdecoder
        decoder_config['n_head'] = args.decoder_head
        decoder_config['d_k'] = args.decoder_feedforwardL
        decoder_config['window_size'] = args.window_size
        decoder_config['vocab_size'] = lm_config.vocab_size

        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_lm)
        lm_config.vocab_size = tokenizer.vocab_size


    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # train data load
    logger.info('Loading train examples...')
    if not os.path.exists(os.path.join(args.data_dir, 'train.tsv')):
        raise ValueError(f'train.csv does not exist.')
    train_examples = processor.get_examples(os.path.join(args.data_dir, 'train.tsv'), base = processor.type == 'base')
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    logger.info('Converting train examples to features...')
    train_features = processor.convert_examples_to_features(train_examples, args.max_src_len, args.max_tgt_len,
                                                            tokenizer,INC_window_size = args.window_size,
                                                            learning_type=processor.type)
    train_data = processor.create_dataset(train_features, processor.type)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=False)

    example = train_examples[0]
    example_feature = train_features[0]

    logger.info("*** Example ***")
    logger.info("guid: %s" % (example.guid))
    logger.info("src text: %s" % example.src)
    logger.info("src_ids: %s" % " ".join([str(x) for x in example_feature.src_ids]))
    logger.info("src_mask: %s" % " ".join([str(x) for x in example_feature.src_mask]))
    logger.info("tgt text: %s" % example.tgt)
    logger.info("tgt_ids: %s" % " ".join([str(x) for x in example_feature.tgt_ids]))
    logger.info("tgt_mask: %s" % " ".join([str(x) for x in example_feature.tgt_mask]))
    logger.info('Building dataloader...')

    # evaluation data load
    if not os.path.exists(os.path.join(args.data_dir, args.test_title)):
        logger.info('No eval data found in data directory. Eval will not be performed.')
        eval_dataloader = None
    else:
        logger.info('Loading eval dataset...')
        eval_examples = processor.get_examples(os.path.join(args.data_dir, args.test_title),
                                               base = processor.type == 'base')
        logger.info('Converting eval examples to features...')
        eval_features = processor.convert_examples_to_features(eval_examples, args.max_src_len, args.max_tgt_len,
                                                               tokenizer,INC_window_size = args.window_size,
                                                            learning_type=processor.type)
        eval_data = processor.create_dataset(eval_features, processor.type)
        # eval_sampler = RandomSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, batch_size=args.train_batch_size, drop_last=False)

        if os.path.isfile(os.path.join(args.data_dir, 'test.tsv')) and args.test_title == 'test.tsv' and args.task_name == 'draw':
            test_examples = processor.get_examples(os.path.join(args.data_dir, 'test.tsv'),base = processor.type == 'base')

            test_features = processor.convert_examples_to_features(test_examples, args.max_src_len, args.max_tgt_len,
                                                                   tokenizer,INC_window_size = args.window_size,
                                                            learning_type=processor.type)
            if args.do_train == False:
                eval_data = processor.create_dataset(test_features,processor.type)
                # eval_sampler = RandomSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, batch_size=args.train_batch_size, drop_last=False)

            else:
                del test_examples
                del test_features

    # total step calculation
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Load model
    model = MTL(args.pretrained_lm, decoder_config, device, lm_config, tokenizer, processor)

    # Load parameter
    if os.path.isfile(args.model_path):
        model.load_state_dict(torch.load(args.model_path))
    else:
        model.init_weights(encoder_RandomInit=args.encoder_parameter_initialization)

    model.to(device)

    # Assign hyper parameter to each components
    encoder = ['encoder', 'numberDistance', 'numberDistanceCounts']
    decoder = ['decoder', 'generator']
    param_optimizer = list(model.named_parameters())

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in encoder) and
                    not any(nd in n for nd in decoder)], 'lr': args.encoder_learning_rate},
        {'params': [p for n, p in param_optimizer if
                    any(nd in n for nd in decoder) and not any(nd in n for nd in encoder)],
         'lr': args.decoder_learning_rate},
        {'params': [p for n, p in param_optimizer if
                    not any(nd in n for nd in encoder) and not any(
                        nd in n for nd in decoder)]}]

    # Choose optimizer function
    if args.optimizer == 'Lamb':
        optimizer = optim.Lamb(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    elif args.optimizer == 'adabound':
        optimizer = optim.AdaBound(optimizer_grouped_parameters, lr=args.learning_rate)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(optimizer_grouped_parameters, lr=args.learning_rate)
    elif args.optimizer == 'yogi':
        optimizer = optim.Yogi(optimizer_grouped_parameters, lr=args.learning_rate)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    # Choose scheduler
    if ',' in args.warmup_proportion:
        args.warmup_proportion = [float(wp) for wp in args.warmup_proportion.split(',')]
        scheduler = EnDeDiffWarmup(optimizer, warmup=args.warmup_proportion,
                                   total_steps=t_total)
    else:
        args.warmup_proportion = float(args.warmup_proportion)
        scheduler = LinearWarmUpScheduler(optimizer, warmup=args.warmup_proportion,
                                          total_steps=t_total)

    if os.path.isfile(os.path.join(args.pretrained_lm, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.pretrained_lm, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Data parallel training
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    # train
    model.train()
    global_step = 0

    # save config file
    lm_config.to_json_file(os.path.join(model_path, 'bert_config.json'))
    with open(os.path.join(model_path, 'bert_config.json'), 'r') as f:
        lm_config = json.load(f)
    outputconfig = {'bert_config': lm_config, 'decoder_config': decoder_config}
    with open(os.path.join(model_path, 'config.json'), 'w') as f:
        json.dump(outputconfig, f)

    best_accuracy = 0

    set_seed(seed=args.seed)
    if args.do_train:
        for i in range(int(args.num_train_epochs)):
            # do training
            model.train()

            tr_loss, tr_logging_loss = 0.0, 0.0
            g_loss, g_logging_loss, ip_loss, ip_logging_loss = 0.0, 0.0, 0.0, 0.0
            op_loss,op_loging_loss = 0.0, 0.0

            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)

                if processor.type == 'MTL_generation':
                    logits = model(src=batch[0], src_mask=batch[1], tgt=batch[2], Normtgt=batch[5], Normtgt_loc=batch[6],
                                   Normsrc = batch[7],Normsrc_labels = batch[8], processors=processor)
                    logits[0], _ = cal_performance(logits[0], batch[5], smoothing=args.do_label_smoothing)

                    if n_gpu > 1:
                        logits = [loss.mean() for loss in logits]
                    if args.gradient_accumulation_steps > 1:
                        logits = [loss / args.gradient_accumulation_steps for loss in logits]

                    loss = logits[0]
                    op_loss += logits[1].item()
                    loss = loss + logits[1]

                elif processor.type == 'classification':
                    logits = model(src=batch[0], src_mask=batch[1], tgt=batch[2], Normtgt=batch[5],
                                   Normtgt_loc=batch[6],
                                   Normsrc=batch[7], Normsrc_labels=batch[8], processors=processor)

                    if n_gpu > 1:
                        logits = [loss.mean() for loss in logits]
                    if args.gradient_accumulation_steps > 1:
                        logits = [loss / args.gradient_accumulation_steps for loss in logits]

                    loss = logits[0]

                elif processor.type == 'MTL_classification':
                    logits = model(src=batch[0], src_mask=batch[1], tgt=batch[2], Normtgt=batch[5], Normtgt_loc=batch[6],
                                   Normsrc = batch[7],Normsrc_labels = batch[8], processors=processor)

                    if n_gpu > 1:
                        logits = [loss.mean() for loss in logits]
                    if args.gradient_accumulation_steps > 1:
                        logits = [loss / args.gradient_accumulation_steps for loss in logits]

                    loss = logits[0]
                    op_loss += logits[1].item()
                    loss = loss + logits[1]

                elif processor.type != 'base':
                    logits = model(src=batch[0], src_mask=batch[1], tgt=batch[2],
                                         gd_loc_label=batch[4], ip_loc_label=batch[5], processors=processor)

                    # calculation of label smoothing loss
                    logits[0], _ = cal_performance(logits[0], batch[2], smoothing=args.do_label_smoothing)

                    if n_gpu > 1:
                        logits = [loss.mean() for loss in logits]
                    if args.gradient_accumulation_steps > 1:
                        logits = [loss / args.gradient_accumulation_steps for loss in logits]

                    g_loss += logits[1].item()
                    ip_loss += logits[2].item()

                    loss = logits[0]
                    loss = loss + logits[1]
                    loss = loss + logits[2]

                else:
                    logits = model(src=batch[0], src_mask=batch[1], tgt=batch[2], processors=processor)
                    loss, _ = cal_performance(logits, batch[2], smoothing=args.do_label_smoothing)

                tr_loss += loss.item()
                loss.backward()

                nb_tr_examples += batch[0].size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                    scheduler.step()
                    global_step += 1

                    loss_scalar = (tr_loss - tr_logging_loss) / args.gradient_accumulation_steps

                    logs = {}
                    logs["encoder_learning_rate"] = scheduler.get_lr()[0]
                    logs["decoder_learning_rate"] = scheduler.get_lr()[1]

                    logs["train_loss"] = loss_scalar

                    logs["group_diff_loss"] = (g_loss - g_logging_loss) / args.gradient_accumulation_steps
                    logs["implicit_pair_loss"] = (ip_loss - ip_logging_loss) / args.gradient_accumulation_steps

                    logs["operator_loss"] = (op_loss - op_loging_loss) / args.gradient_accumulation_steps

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)

                    tr_logging_loss = tr_loss

                    g_logging_loss = g_loss
                    ip_logging_loss = ip_loss

                    op_loging_loss = op_loss

            # do evaluation
            if args.output_dir is not None:
                state_dict = model.module.state_dict() if n_gpu > 1 else model.state_dict()

                if args.save_every:
                    torch.save(state_dict, os.path.join(model_path, 'model{}.bin'.format(i)))
                else:
                   torch.save(state_dict, os.path.join(model_path, 'Modellast.bin'))

                logger.info('Model saved')
                logger.info(f'Epoch {i} finished.')

            if eval_dataloader is not None and i % args.eval_interval == 0:
                evaluation_log = evaluation(model, eval_dataloader, args.beam_size, args.best_beam, tokenizer=tokenizer,device = device,n_gpu=n_gpu)
                print('eval acc : {}'.format(evaluation_log["equation_accuracy"]))
                print('eval{}_{}'.format(args.best_beam, evaluation_log["equation_topk_accuracy"]))

                maxacc = np.max([evaluation_log["equation_accuracy"]])

                if best_accuracy < evaluation_log["equation_accuracy"]:
                    best_accuracy = evaluation_log["equation_accuracy"]

                    output_dir = os.path.join(model_path, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Save model checkpoint
                    state_dict = model.module.state_dict() if n_gpu > 1 else model.state_dict()
                    torch.save(state_dict, os.path.join(model_path, 'BestModel.bin'.format(i)))
                    # tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

                    fequation_accuracy = maxacc
                    logger.info('BestModel saved')
                    df = pd.DataFrame.from_dict(evaluation_log)
                    df.to_csv(os.path.join(output_dir, 'BestBertAbsSum.csv'), index=False, sep='\t')

                else:
                    df = pd.DataFrame.from_dict(evaluation_log)
                    df.to_csv(os.path.join(model_path, 'AbsSum{}.csv'.format(i)), index=False, sep='\t')

                tb_writer.add_scalar('eval_{}'.format(args.task_name), evaluation_log["equation_accuracy"], global_step)
                tb_writer.add_scalar('eval{}_{}'.format(args.best_beam, args.task_name),
                                     evaluation_log["equation_topk_accuracy"], global_step)

                logger.info(f'Source: {evaluation_log["src_list"][0]}')
                logger.info(f'Beam Generated: {evaluation_log["hyp_list"][0]}')
                logger.info(f'Epoch {i} finished.')
                tb_writer.close()

    else:
        evaluation_log = evaluation(model, eval_dataloader, args.beam_size, args.best_beam, tokenizer=tokenizer,device = device,n_gpu=n_gpu)

        print('eval : ', str(evaluation_log["equation_accuracy"]))
        print('eval{}_{}'.format(args.best_beam, evaluation_log["equation_topk_accuracy"]))

        # df = pd.DataFrame.from_dict(evaluation_log)
        # df.to_csv(saved_model_path.replace('BestModel.bin', 'BestModelTest.csv'), index=False, sep='\t')
        # print('df saved in {}'.format(model_path))
    logger.info('Training finished')

if __name__ == "__main__":
    var_dict = \
        {
        "model_type" : "GEO",
        # ex) "GEO","aux","base","MTL_generation","MTL_classification",classification
        "do_train": "",
        # ex) If you only want to run the test, please comment out
        'task_name': "klg514",
        # ex) 'alg514', 'mawps', 'Math23k','IL','draw' , 'klg514',
        "test_title" : 'test.tsv',
        # ex) 'dev.tsv' , 'test.tsv'
        "data_dir" : '{}/{}/fold{}/tokens/',
        # ex) '{}/{}/fold{}/tokens/', '{}/{}/tokens/'
        "output_dir":r'./savefile/result',
        "eval_interval":"10",
        # ex) duration of evaluation during test
        'train_batch_size': '16',
        'num_train_epochs': '501',
        'pretrained_lm': 'monologg/koelectra-base-v3-discriminator',
        ## 'pretrained_lm': 'google/electra-base-discriminator',
        # ex) 'google/electra-base-discriminator' , 'hfl/chinese-electra-base-discriminator','monologg/koelectra-base-v3-discriminator'
        # 'model_path': r'/home/dg/PycharmProjects/Coling_JournalofSuperComputing/savefile/result/alg514/model_01-31-20:16:32_0_16_150_1e-4_2e-4_google/electra-base-discriminator_GEO/BestModel.bin',
        # ex) If you want to use saved model, please comment out and replace the model_path to './savefile/...'
        # "encoder_parameter_initialization":"",
        # ex) If you only want to train the encoder using random intialized parameter, please comment out
        # "no_cuda":"",
        # ex) If you only want to train using cpu, please comment out
        # 'save_every':"",
        "do_label_smoothing":"",
        # ex) If you want to use label smoothing loss calcuation, please comment out
        'learning_rate': '1e-4',
        'encoder_learning_rate': '1e-4',
        'decoder_learning_rate': '2e-4',
        'max_src_len': "150",
        'weight_decay': '0.0',
        'local_rank' : "0",
        # ex) 0 , 1
        "optimizer": "Wadam",
        "window_size": "3",
        # ex) 3 , 4 ...
        "warmup_proportion": "0.1",
        'gradient_accumulation_steps': "1",
        "max_tgt_len": '50',
        "beam_size" : '3',
        "best_beam" : '3',
        "numberofdecoder": "3",
        "decoder_head" : "8",
        "decoder_feedforwardL" : "2048",
    }
    processor = LCSTSProcessor()
    processor.type = var_dict["model_type"]
    if 'classification' in processor.type:
        var_dict["beam_size"] = 1
        var_dict["best_beam"] = 1

    if var_dict["task_name"] != 'draw' and var_dict["task_name"] != 'mathqa' and var_dict["task_name"] != 'Math23k':
        valid_first = 0
        valid_last = 5
    else:
        valid_first = 1
        valid_last = 2

    root = os.path.dirname(os.path.realpath(__file__))
    var_dict["output_dir"] = os.path.join(root, var_dict["output_dir"],r'{}'.format(var_dict["task_name"]))

    experiment_name = '{}_{}_{}_{}_{}'.format(var_dict["train_batch_size"], var_dict["max_src_len"],
                                              var_dict["encoder_learning_rate"], var_dict["decoder_learning_rate"],
                                              '{}_{}'.format(var_dict["pretrained_lm"], processor.type))

    for key in list(var_dict.keys()):
        sys.argv.append('--{}'.format(key))
        item = var_dict[key]
        if item != "":
            sys.argv.append(var_dict[key])

    for valid_num in range(valid_first,valid_last):
        args = parser.parse_args()
        args.data_dir = var_dict["data_dir"].format(os.path.join(root, 'dataset'), var_dict["task_name"],valid_num)
        main(args)
