import torch
from torch.utils.data import TensorDataset
import csv
import logging
from utils import convert_to_unicode
from tqdm import tqdm
import numpy as np
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example."""

    def __init__(self, guid, src, tgt=None, dep=None,numdep=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            src: string. The untokenized text of the target sequence.
            tgt: (Optional) string. The untokenized text of the target.
        """
        self.guid = guid
        self.src = src
        self.tgt = tgt
        self.dep = dep
        self.numdep = numdep
class InputFeatures():
    """A single set of features of data."""

    # def __init__(self, src_ids, src_mask, tgt_ids, tgt_mask):
    #     self.src_ids = src_ids
    #     self.src_mask = src_mask
    #     self.tgt_ids = tgt_ids
    #     self.tgt_mask = tgt_mask

    def __init__(self, src_ids, src_mask, tgt_ids, tgt_mask, id,dep_ids = None,dep_masks = None,src_length = None,tgt_locs = None,
                 Normtgt_ids=None, Normtgt_locs=None, Normsrc_tokens = None,Normsrc_labels =None):
        self.src_ids = src_ids
        self.src_mask = src_mask
        self.tgt_ids = tgt_ids
        self.tgt_mask = tgt_mask

        self.Normtgt_ids = Normtgt_ids
        self.Normtgt_locs = Normtgt_locs
        self.Normsrc_tokens = Normsrc_tokens
        self.Normsrc_labels = Normsrc_labels

        self.id = id
        self.dep_ids = dep_ids
        self.dep_masks = dep_masks
        self.src_length = src_length
        self.tgt_locs = tgt_locs
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class LCSTSProcessor(DataProcessor):
    """Processor for the LCSTS data set."""
    def __init__(self):
        self.numbers = {}
        self.answers = {}

        self.answer_equations = {}
        self.answer_index = {}
        self.answer_lists = []

        self.false_equations = {}
        self.filtered_tokenToIds = {0:0}
        self.filtered_IdstoTokens = {0:0}

        self.sums = {}
        self.number_idx = {}
        self.idx_num = {}
        self.indexes = {}
        self.op_idx = {}
        self.op_num = {}
        self.seq_len = 0

        self.num_groupLimit = 20*2

        self.num_tokens = {}
        self.fusion_layer = True
        self.type = ''
        self.operators = ['-', '/', '*', '+']

        self.labeldict = {}
        self.labellist = []

        self.op_norm_labeldict = {}
        self.op_norm_labellist =[]
    def get_examples(self, data_path,base):
        """See base class."""
        return self._create_examples(self._read_tsv(data_path),base)

    def _create_examples(self, lines, base = False):
        examples = []
        # equation = 1

        for n,data in enumerate(lines):
            # lines: id, summary, text
            # if n > 0:
                guid = data[0].replace('\ufeff','')
                src = convert_to_unicode(data[1].replace('N','N ').replace('C','C '))
                tgt = convert_to_unicode(data[2]).replace('N','N ').replace('C','C ')

                try:
                    self.answers[int(guid)] = \
                    list(set([float(answer) for answer in data[4].split(' ')]))
                except:
                    self.answers[int(guid)] = ''
                temnum = {}
                for dn,number in enumerate(data[5].split(' ')):
                    temnum['n{}'.format(dn)]= number
                self.numbers[int(guid)] = temnum
                self.false_equations[int(guid)] = []

                if not base:
                    dep = convert_to_unicode(data[3])
                    numdep = convert_to_unicode(data[-1])
                    examples.append(InputExample(guid=guid, src=src, tgt=tgt,
                                                 dep=[int(f) for f in dep.split(' ') if f != ''] + [-10],
                                                 numdep=[int(f) for f in numdep.split(' ') if f != '']))
                else:
                    examples.append(InputExample(guid=guid, src=src, tgt=tgt))
        return examples

    def convert_examples_to_features(self,examples, src_max_seq_length, tgt_max_seq_length, tokenizer, INC_window_size=3,
                                     learning_type=''):
        """Loads a data file into a list of `InputBatch`s."""
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        self.seq_len = src_max_seq_length

        if len(self.sums) == 0:
            self.operatorToken = tokenizer.convert_tokens_to_ids('operator')
            self.padding = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            for i in range(40):
                self.sums[sum(range(i + 2))] = i + 1
                self.number_idx[i] = tokenizer.convert_tokens_to_ids(str(i))
                self.idx_num[tokenizer.convert_tokens_to_ids(str(i))] = i

            self.number_idx['n'] = tokenizer.convert_tokens_to_ids('n')
            self.number_idx['number'] = tokenizer.convert_tokens_to_ids('number')

            self.indexes['EOS'] = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
            self.indexes['BOS'] = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
            self.indexes['CLS'] = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
            self.indexes['SEP'] = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
            self.indexes['PAD'] = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            self.indexes['UNK'] = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
            self.indexes['MASK'] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

            self.num_tokens = {}

            for nums in range(30):
                self.num_tokens['n{}'.format(nums)] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('n{}'.format(nums)))
            for o in self.operators+['op']:
                self.op_idx[o] = tokenizer.convert_tokens_to_ids(str(o))
                self.op_num[tokenizer.convert_tokens_to_ids(str(o))] = o

        INCids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize('n')[-1]),
                  tokenizer.convert_tokens_to_ids(tokenizer.tokenize('N')[-1])]

        features = []
        meanLength = []
        tgtlength = []
        for (ex_index, example) in enumerate(tqdm(examples, desc='examples')):
            if 'bart' not in str(tokenizer):
                src_tokens = tokenizer.tokenize(example.src)
                tgt_tokens = tokenizer.tokenize(example.tgt)

                meanLength.append(len(src_tokens))
                tgtlength.append(len(example.tgt.replace('N ','N').replace('C ','C').split(" ")))

                if len(src_tokens) > src_max_seq_length- 2:
                    src_tokens = src_tokens[:(src_max_seq_length- 2)]
                if len(tgt_tokens) > tgt_max_seq_length - 2:
                    tgt_tokens = tgt_tokens[:(tgt_max_seq_length - 2)]

                src_ids = tokenizer.encode(src_tokens, add_special_tokens=True)
                tgt_ids = tokenizer.encode(tgt_tokens, add_special_tokens=True)
                meanLength.append(len(src_ids))

                for nt,id in enumerate(tgt_ids):
                    if id not in self.filtered_tokenToIds:
                        tg = len(self.filtered_tokenToIds)
                        self.filtered_tokenToIds[id] = tg
                        self.filtered_IdstoTokens[tg] = id

                template = ' '.join([str(t) for t in tgt_ids[1:-1]])
                if template not in self.answer_lists:
                    self.answer_lists.append(template)
                self.answer_equations[int(example.guid)] = [template]
                self.answer_index[int(example.guid)] = self.answer_lists.index(template)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                src_mask = [1] * len(src_ids)
                tgt_mask = [1] * len(tgt_ids)
            else:
                src = tokenizer.batch_encode_plus([example.src], max_length=128, return_tensors='pt')
                src_ids = list(src['input_ids'].numpy()[0])
                src_mask = list(src['attention_mask'].numpy()[0])
                tgt = tokenizer.batch_encode_plus([example.tgt], max_length=128, return_tensors='pt')
                self.answer_equations[int(example.guid)] = \
                [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in tgt['input_ids']][0].rstrip().lstrip()
                tgt_ids = list(tgt['input_ids'].numpy()[0])
                tgt_mask = list(tgt['attention_mask'].numpy()[0])

            Normtgt_tokens = []
            Normsrc_labels = []
            Normtgt_locs = []
            for nt,tok in enumerate(tgt_tokens):
                if tok not in self.operators:
                    Normtgt_tokens.append(tok)
                else:
                    Normtgt_tokens.append('op')
                    Normsrc_labels.append(self.operators.index(tok))
                    Normtgt_locs.append(len(src_ids)-1 + nt)

            Normtgt_ids = tokenizer.encode(Normtgt_tokens, add_special_tokens=True)
            Normsrc_tokens = src_ids[:-1] + Normtgt_ids[1:]
            Normsrc_tokens += [self.padding] * (src_max_seq_length - len(Normsrc_tokens))
            Normsrc_tokens = Normsrc_tokens[:src_max_seq_length]

            Normsrc_labels += [self.padding] * (tgt_max_seq_length - len(Normsrc_labels))
            Normtgt_padding = [self.padding] * (tgt_max_seq_length - len(Normtgt_ids))
            Normtgt_locs += [0] * (tgt_max_seq_length - len(Normtgt_locs))
            if 'classification' not in learning_type:
                Normtgt_ids += Normtgt_padding
                Normtgt_ids = Normtgt_ids[:tgt_max_seq_length]

            elif learning_type == 'classification':
                if tgt_ids not in self.op_norm_labellist:
                    self.op_norm_labellist.append(tgt_ids)
                    self.op_norm_labeldict[self.op_norm_labellist.index(tgt_ids)] = tgt_ids
                op_norm_label = self.op_norm_labellist.index(tgt_ids)
                Normtgt_ids = [op_norm_label]

            else:
                if Normtgt_ids not in self.op_norm_labellist:
                    self.op_norm_labellist.append(Normtgt_ids)
                    self.op_norm_labeldict[self.op_norm_labellist.index(Normtgt_ids)] = Normtgt_ids
                op_norm_label = self.op_norm_labellist.index(Normtgt_ids)
                Normtgt_ids = [op_norm_label]

            src_length = len(src_ids) if len(src_ids) < src_max_seq_length else src_max_seq_length
            # Zero-pad up to the sequence length.
            src_padding = [self.padding] * (src_max_seq_length - len(src_ids))
            tgt_padding = [self.padding] * (tgt_max_seq_length - len(tgt_ids))
            # tgt_loc_padding = [self.padding] * (tgt_max_seq_length - len(tgt_locs))

            src_mask += [0] * (src_max_seq_length - len(src_ids))
            src_ids += src_padding
            tgt_mask += [0] * (tgt_max_seq_length - len(tgt_ids))
            tgt_ids += tgt_padding
            tgt_locs = list(range(len(tgt_ids)))

            src_ids = src_ids[:src_max_seq_length]
            src_mask = src_mask[:src_max_seq_length]
            assert len(src_ids) == src_max_seq_length
            assert len(tgt_ids) == tgt_max_seq_length
            assert len(tgt_locs) == tgt_max_seq_length

            if learning_type == 'aux' or learning_type == 'GEO':
                dep_src_idx = []
                dep_src_idx_counts = []
                num = 0

                for INC in INCids:
                    INC_locs = torch.where(torch.tensor(src_ids) == INC)[0]
                    if len(INC_locs) != 0:
                        break

                for nidx,id in enumerate(INC_locs):
                    # 다음 수가 실제 상수인지 확인
                    try:
                        int(tokenizer.convert_ids_to_tokens(src_ids[id + 1]).replace('##',''))
                    except:
                        continue

                    id = int(id)

                    count = example.numdep[num]

                    if count == 0:
                        count = -2

                    temp = [id + wn for wn in range(-1 * INC_window_size, INC_window_size + 1)]
                    temp = np.array(temp)
                    temp[np.array(temp) < 0] = 0

                    if True in list(np.array(temp) > src_max_seq_length-1):
                        temp[np.array(temp) > src_max_seq_length-1]  = src_max_seq_length-1

                    temp = list(temp)

                    dep_src_idx_counts.extend(temp+[count])
                    dep_src_idx.extend(temp)

                    num +=1

                out_dep = []
                initial_loc = 0

                window_size = INC_window_size*2 + 1

                for i in example.dep:
                    if i == 0:
                        i = -2
                    if i == -10:
                        break
                    if i == -1:
                        initial_loc += 1
                        pre_loc = initial_loc
                    else:
                        pre_loc += 1
                        if i > 19:
                            continue
                        temp = dep_src_idx[window_size*(initial_loc-1):window_size*initial_loc]+\
                               dep_src_idx[window_size*(pre_loc-1):window_size*pre_loc]

                        if len(temp) == window_size*2:
                            out_dep.extend(temp+[i])

                if len(out_dep) > (window_size*2+1)*self.num_groupLimit:
                    out_dep = out_dep[:(window_size*2+1)*self.num_groupLimit]

                else:
                    dep_padding = [0] * ((window_size*2+1)*self.num_groupLimit - len(out_dep))
                    out_dep += dep_padding

                dep_padding = [0] * ((window_size + 1) * self.num_groupLimit-len(dep_src_idx_counts))

                dep_src_idx_counts += dep_padding

                features.append(InputFeatures(src_ids=src_ids,
                                              src_mask=src_mask,
                                              tgt_ids=tgt_ids,
                                              tgt_mask=tgt_mask,
                                              id=example.guid,dep_ids = out_dep,dep_masks=dep_src_idx_counts,src_length=src_length,tgt_locs = tgt_locs,
                                              Normtgt_ids = Normtgt_ids,Normtgt_locs = Normtgt_locs,Normsrc_tokens= Normsrc_tokens,Normsrc_labels=Normsrc_labels))
            else:
                features.append(InputFeatures(src_ids=src_ids,
                                              src_mask=src_mask,
                                              tgt_ids=tgt_ids,
                                              tgt_mask=tgt_mask,
                                              id = example.guid,src_length=src_length,tgt_locs = tgt_locs,Normtgt_ids = Normtgt_ids,
                                              Normtgt_locs = Normtgt_locs,Normsrc_tokens= Normsrc_tokens,Normsrc_labels=Normsrc_labels))
        print('max : {}'.format(np.max(meanLength)))
        print('min : {}'.format(np.min(meanLength)))
        print('median : {}'.format(np.median(meanLength)))
        print('std : {}'.format(np.std(meanLength)))
        print('tgtmax : {}'.format(np.max(tgtlength)))
        print('tgtmin : {}'.format(np.min(tgtlength)))
        print('tgtmedian : {}'.format(np.median(tgtlength)))
        print('tgtstd : {}'.format(np.std(tgtlength)))
        return features

    def create_dataset(self,features,learning_type):
        all_src_ids = torch.tensor([f.src_ids for f in features], dtype=torch.long)
        all_src_mask = torch.tensor([f.src_mask for f in features], dtype=torch.long)
        all_tgt_ids = torch.tensor([f.tgt_ids for f in features], dtype=torch.long)
        all_tgt_mask = torch.tensor([f.tgt_mask for f in features], dtype=torch.long)
        all_tgt_locs = torch.tensor([f.tgt_locs for f in features], dtype=torch.long)
        all_ids = torch.tensor([int(f.id) for f in features], dtype=torch.long)
        all_src_length = torch.tensor([f.src_length for f in features], dtype=torch.long)
        all_Normtgt_ids = torch.tensor([f.Normtgt_ids for f in features], dtype=torch.long)
        all_Normtgt_locs = torch.tensor([f.Normtgt_locs for f in features], dtype=torch.long)
        all_Normsrc_tokens = torch.tensor([f.Normsrc_tokens for f in features], dtype=torch.long)
        all_Normsrc_labels = torch.tensor([f.Normsrc_labels for f in features], dtype=torch.long)
        if learning_type == 'aux' or learning_type == 'GEO':
            all_dep_ids = torch.tensor([f.dep_ids for f in features], dtype=torch.long)
            all_dep_ids_counts = torch.tensor([f.dep_masks for f in features], dtype=torch.long)
            train_data = TensorDataset(all_src_ids, all_src_mask, all_tgt_ids, all_tgt_mask,all_dep_ids,all_dep_ids_counts,
                                       all_tgt_locs,all_Normtgt_ids,all_Normtgt_locs,all_Normsrc_tokens,all_Normsrc_labels,all_src_length, all_ids)
        else:
            train_data = TensorDataset(all_src_ids, all_src_mask, all_tgt_ids, all_tgt_mask, all_tgt_locs,all_Normtgt_ids,all_Normtgt_locs,all_Normsrc_tokens,
                                       all_Normsrc_labels,
                                       all_src_length, all_ids)
        return train_data