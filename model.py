import torch.nn as nn
import torch
import numpy as np

from transformer.Models import TransformerDecoder

from transformer.Beam import Beam
import copy

from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from calculator import solver

from transformers import AutoConfig,AutoTokenizer, AutoModel


class MTL(nn.Module):
    def __init__(self, encoder_model_path, decoder_config, device, encoder_config, tokenizer, processors):
        super().__init__()
        self.learning_type = processors.type
        self.device = device
        self.solvers = solver()

        self.window_size = decoder_config['window_size']*2 + 1

        self.tokenizer = tokenizer
        self.vocab_size = decoder_config['vocab_size']

        # Load encoder parameters from transformer libraries.
        self.encoder = AutoModel.from_pretrained(encoder_model_path, config=encoder_config)
        if self.learning_type != 'MTL_classification':
            self.tgt_embeddings = nn.Embedding(decoder_config['vocab_size'], decoder_config['d_model'], padding_idx=0)
            self.decoder = TransformerDecoder(
                decoder_config['n_layers'],
                decoder_config['d_model'],
                heads=decoder_config['n_head'],
                d_ff=decoder_config['d_k'],
                dropout=0.1,
                embeddings=self.tgt_embeddings, operatorNums=processors.op_num
            )
            gen_func = nn.LogSoftmax(dim=-1)
            self.generator = nn.Sequential(nn.Linear(decoder_config['d_model'], decoder_config['vocab_size']), gen_func)
            self.generator[0].weight = self.decoder.embeddings.weight

        self.len_max_src = processors.seq_len
        self.len_max_seq = decoder_config['len_max_seq']

        # Create a linear layer according to the learning type.
        if self.learning_type == 'GEO':
            # convert concatenation of INC pair relation state and INC contextual state to one-dimension.
            self.conversion = nn.Linear((self.window_size) * 3 * decoder_config['d_model'], decoder_config['d_model'])
            self.OP3FLayer = nn.Linear(decoder_config['d_model'] * 2, decoder_config['d_model'])

        if 'MTL' in self.learning_type:
            if 'classification' in self.learning_type:
                self.op_norm_labeldict = processors.op_norm_labeldict
                self.template_classifier = nn.Linear(encoder_config.hidden_size, len(self.op_norm_labeldict))
            self.operator_classifier = nn.Linear(encoder_config.hidden_size, len(processors.operators))

        elif self.learning_type == 'classification':
            self.op_norm_labeldict = processors.op_norm_labeldict
            self.template_classifier = nn.Linear(encoder_config.hidden_size, len(self.op_norm_labeldict))

        if self.learning_type == 'aux' or self.learning_type == 'GEO':
            self.groupDiff = nn.Linear(((self.window_size) * 2) * encoder_config.hidden_size, 20)
            self.implicitPair = nn.Linear((self.window_size) * encoder_config.hidden_size, 9)

        # parameter setting
        self.dropOut = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)

        self.sums = processors.sums
        self.number_idx = processors.number_idx
        self.idx_num = processors.idx_num
        self.indexes = processors.indexes

        self.op_idx = processors.op_idx
        self.op_num = processors.op_num
        self.num_tokens = processors.num_tokens

        self.search_range = self.window_size

    def init_weights(self,encoder_RandomInit=False):
        if encoder_RandomInit:
            for module in self.encoder.modules():
                if isinstance(module, (nn.Linear)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
        if self.learning_type != 'MTL_classification':
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                    # module.weight.data.fill_(1.0)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()

    def forward(self, src, src_mask, tgt, gd_loc_label=[], ip_loc_label=[],processors=[],Normtgt=[],Normtgt_loc=[],Normsrc = [],Normsrc_labels = []):
        # shift right
        tgt = tgt[:, :-1]

        # cross-entropy loss
        loss_cl = CrossEntropyLoss()

        encoder_hidden_states = self.encoder(src, attention_mask=src_mask)

        if self.learning_type == 'MTL_generation':
            Normtgt = Normtgt[:, :-1]

            dec_state = self.decoder.init_decoder_state(src, encoder_hidden_states[0])
            logits_, _ = self.decoder(Normtgt, encoder_hidden_states[0], dec_state)
            logits = self.generator.forward(logits_)

            conditions = torch.where((Normtgt_loc<self.len_max_src) & (0<Normtgt_loc))

            Normsrc_labels = Normsrc_labels[conditions]
            conditions = [conditions[0],Normtgt_loc[conditions]]

            Normsrc_hiddens = self.encoder(Normsrc)[0][conditions]
            Normsrc_logits = self.operator_classifier(Normsrc_hiddens)
            Normsrc_loss = loss_cl(Normsrc_logits.view(-1,len(processors.operators)),
                                      Normsrc_labels)

            logits = [logits,Normsrc_loss]
            return logits

        elif self.learning_type == 'MTL_classification':
            ## Template classification
            logits = self.template_classifier(encoder_hidden_states[0][:,0,:])
            logits = loss_cl(logits,Normtgt.view(-1))

            ## Operator classification
            conditions = torch.where((Normtgt_loc<self.len_max_src) & (0<Normtgt_loc))
            Normsrc_labels = Normsrc_labels[conditions]
            conditions = [conditions[0],Normtgt_loc[conditions]]
            Normsrc_hiddens = self.encoder(Normsrc)[0][conditions]
            Normsrc_logits = self.operator_classifier(Normsrc_hiddens)
            Normsrc_loss = loss_cl(Normsrc_logits.view(-1,len(processors.operators)),
                                      Normsrc_labels)
            logits = [logits,Normsrc_loss]
            return logits

        elif self.learning_type == 'classification':
            ## Template classification
            logits = self.template_classifier(encoder_hidden_states[0][:, 0, :])
            logits = loss_cl(logits, Normtgt.view(-1))
            return [logits]

        elif self.learning_type == 'aux' or self.learning_type == 'GEO':
            # create empty decoder state
            dec_state = self.decoder.init_decoder_state(src, encoder_hidden_states[0])

            # calculate decoder logits
            decoder_hidden_states, _ = self.decoder(tgt, encoder_hidden_states[0], dec_state)

            # calculate group difference loss
            INC_relationState, gd_labels, gd_batch_idx = self.extract_states_label(gd_loc_label, encoder_hidden_states[0],groupdiff=True)

            group_logit = self.groupDiff(self.dropOut(INC_relationState).view(INC_relationState.size()[0], -1))
            group_loss = loss_cl(group_logit,torch.tensor(gd_labels).view(-1).type(torch.int64).to(self.device))

            ## group_loss scaling
            group_loss *= len(gd_batch_idx) / len(gd_labels)

            # calculate group difference loss
            INC_contexualState, ip_labels, ip_batch_idx = self.extract_states_label(ip_loc_label,encoder_hidden_states[0],groupdiff=False)

            ip_logit = self.implicitPair(self.dropOut(INC_contexualState).view(INC_contexualState.size()[0], -1))
            ip_loss = loss_cl(ip_logit, torch.tensor(ip_labels).view(-1).type(torch.int64).to(self.device))

            ## ip_loss scaling
            ip_loss *= len(ip_batch_idx) / len(ip_labels)

            if self.learning_type == 'GEO':
                # preprocess for op3f layer
                # get pair mean of INC relation state
                INC_relationState_mean = []

                starts = 0
                for nb, batch in enumerate(gd_batch_idx):
                    INC_relationState_mean.append(torch.mean(INC_relationState[starts:starts + len(batch)], 0))
                    starts += len(batch)

                INC_relationState_mean = torch.stack(INC_relationState_mean)

                # get mean of contextual state and check single number problem
                INC_contexualState_mean = []
                starts = 0

                plural_number_bool = []
                for nb, batch in enumerate(ip_batch_idx):
                    if len(batch) > 1:
                        plural_number_bool.append(True)
                    else:
                        plural_number_bool.append(False)
                    INC_contexualState_mean.append(torch.mean(INC_contexualState[starts:starts + len(batch)], 0))
                    starts += len(batch)
                INC_contexualState_mean = torch.stack(INC_contexualState_mean)

                plural_number_bool = torch.tensor(plural_number_bool).type(INC_contexualState_mean.type()).type(torch.bool)

                Irs_shape = INC_relationState_mean.shape

                if False in plural_number_bool:
                    empty = torch.zeros((len(INC_contexualState_mean), Irs_shape[1], Irs_shape[2])).type(INC_relationState_mean.type())
                    empty[plural_number_bool] = INC_relationState_mean
                    INC_relationState_mean = empty

                # op3f layer calculation
                combined_states = self.conversion(torch.cat((INC_relationState_mean, INC_contexualState_mean), 1).view(len(INC_contexualState_mean), 1, -1))
                logits = self.OP3FLayer(torch.cat((decoder_hidden_states, combined_states.expand(decoder_hidden_states.shape)), 2))
                logits = self.generator.forward(logits)

            else:
                logits = self.generator.forward(decoder_hidden_states)

            logits = [logits, group_loss, ip_loss]
            return logits
        else:
            dec_state = self.decoder.init_decoder_state(src, encoder_hidden_states[0])
            logits_, _ = self.decoder(tgt, encoder_hidden_states[0], dec_state)
            logits = self.generator.forward(logits_)

            return logits

    def toTokens(self, tokids, tokenizer, seq_token='[SEP]'):
        # convert ids to tokens
        output = " ".join(tokenizer.convert_ids_to_tokens(tokids)).split(seq_token)[0]
        return output.replace('▁', '').replace('##', '')

    def checkAnswer(self, ids, tgt, idx, processor):
        # Check if the predicted tgt can actually yield the correct answer
        operators = ['*', '-', '+', '/', '=']

        solving = False

        seq_num = processor.indexes['SEP']
        cls_num = processor.indexes['CLS']

        guid = int(idx)
        ids = ids.cpu()

        number_dict = processor.numbers[guid]

        if seq_num in ids:
            ids = ids[:torch.where(ids == seq_num)[0][0]]
        if cls_num in ids:
            ids = ids[torch.where(ids == cls_num)[0][0] + 1:]

        strids = ' '.join(list(ids.cpu().numpy().astype(str)))

        if strids not in processor.answer_equations[guid]:
            tokens = self.toTokens(ids, self.tokenizer)
            tokens = tokens.replace('x ', 'x').replace('n ', 'n').replace('number ', 'n').replace('数 ','n').replace('c 1 . ne g', '(-1)').replace(' . ',
                                                                                                        '.').replace(
                'eo s ', 'eos').replace('e os ', 'eos').replace('eos ', 'eos').replace('c', '').replace('  ', ' ').rstrip().lstrip()

            symbols = list(set([tok for tok in tokens.split(' ') if 'x' in tok]))

            tokens = tokens.split('=')
            tokens = [token.rstrip().lstrip() for token in tokens]

            if ('eos0' not in tokens[-1] and 'eos1' not in tokens[-1]) or len(tokens[-1]) > 4:
                return solving, tgt, None

            for token in tokens[:-1]:
                symbol_lists = token.split(' ')
                for sym in symbol_lists:
                    try:
                        if sym not in operators and sym != '(-1)':
                            float(sym.replace('x', '').replace('n', ''))
                    except:
                        return solving, tgt, None

            try:
                infix_exp_lists = []
                for expr in tokens[:-1]:
                    infix_exp_lists.append(self.solvers.getInfix(expr + ' ='))
            except:
                return solving, tgt, None

            if len(infix_exp_lists) != len(symbols):
                return solving, tgt, None

            try:
                for ni, inexp in enumerate(infix_exp_lists):
                    for key in number_dict:
                        infix_exp_lists[ni] = infix_exp_lists[ni].replace(key, number_dict[key])

                solutions = self.solvers.solve(infix_exp_lists, symbols)
                temp_answers = np.array(sorted(processor.answers[guid]))
                if len(solutions) == 0 or len(solutions) != len(temp_answers):
                    return solving, tgt, None

                if np.equal(np.array(sorted(solutions)), temp_answers).sum() == len(solutions):
                    ids = torch.cat((torch.tensor([cls_num]), ids, torch.tensor([seq_num]),
                                     torch.zeros(self.len_max_seq - len(ids) - 3).type(ids.type())))

                    processor.answer_equations[guid].append(strids)

                    origin = self.toTokens(
                        torch.tensor(np.array(processor.answer_equations[guid][0].split(' ')).astype(int)),
                        self.tokenizer).replace('x ', 'x').replace('n ', 'n').replace('c 1 . ne g', '(-1)').replace(
                        ' . ', '.').replace('eo s ', 'eos').replace('e os ', 'eos').replace('c ', 'c').replace('  ',
                                                                                                               ' ').rstrip().lstrip()
                    new = self.toTokens(torch.tensor(np.array(strids.split(' ')).astype(int)), self.tokenizer).replace(
                        'x ', 'x').replace('n ', 'n').replace('c 1 . ne g', '(-1)').replace(' . ', '.').replace('eo s ',
                                                                                                                'eos').replace(
                        'e os ', 'eos').replace('c ', 'c').replace('  ', ' ').rstrip().lstrip()

                    notinorin = [n for n in new.split(' ') if n not in origin.split(' ')]

                    solving = True
                    if tgt is not None:
                        nos = [no for no in notinorin if 'c' in no or 'x' in no]

                        if len(nos) == 0 and origin.split(' ')[-1] == new.split(' ')[-1] and origin != new:
                            tgt = ids.to(self.device)

                return solving, tgt, np.array(list(set(solutions)))

            except:
                return solving, tgt, None

        else:
            solving = True
            return solving, tgt, np.array(processor.answers[guid])

    def extract_states_label(self, loc_label, encoder_hiddenstates, groupdiff = True):
        # Extract INC contextual state or INC pair relation state from encoder hiddenstates
        batch_idx = []
        tensors = []
        labels = []

        if groupdiff:
            window_size = self.window_size* 2 + 1
        else:
            window_size = self.window_size + 1

        pre = 0

        for num in range(loc_label.size()[0]):
            loc = torch.where(loc_label[num] != self.indexes['PAD'])[0]

            if len(loc) != 0:
                last_loc = torch.tensor([torch.max(loc) + 1, len(loc_label[num])]).min()
                if last_loc == 1:
                    continue
            else:
                continue

            row, col = 0, 0
            idx_lists = []
            for zi in range(last_loc // window_size):
                if len(loc_label[num]) - 1 < zi * window_size + 1:
                    continue

                target = loc_label[num][zi * window_size:zi * window_size + (window_size - 1)]
                if zi == 0:
                    col += 1
                elif pre != loc_label[num][zi * window_size + 1]:
                    row += 1
                    col = row + 1
                else:
                    col += 1

                pre = loc_label[num][zi * window_size + 1]
                tensors.append(encoder_hiddenstates[num, target, :].view(1, window_size - 1, -1))

                label = int(loc_label[num][zi * window_size + (window_size - 1)])
                if label == -2:
                    label = 0
                labels.append(label)

                if groupdiff:
                    idx_lists.append([row, col])
                else:
                    idx_lists.append([row])

            batch_idx.append(idx_lists)
            pre = len(labels)
        tensors = torch.cat(tensors).contiguous()

        return tensors, labels, batch_idx

    def beam_decode(self, src_seq, src_mask, gd_loc_label=[], ip_loc_label=[], INC_relationState_mean=None, INC_contexualState_mean=None,
                    beam_size=3, n_best=3):
        ''' Translation work in one batch '''

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            ''' Indicate the position of an instance in a tensor. '''
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, beam_size):
            ''' Collect tensor parts associated to active instances. '''
            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * beam_size, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def collate_active_info(
                src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list, INC_relationState_mean, INC_contexualState_mean):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, beam_size)
            active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, beam_size)

            if self.learning_type == 'aux' or  self.learning_type == 'GEO' :
                active_INC_relationState= collect_active_part(INC_relationState_mean, active_inst_idx, n_prev_active_inst, beam_size)
                active_INC_contexualState = collect_active_part(INC_contexualState_mean, active_inst_idx, n_prev_active_inst, beam_size)
            else:
                active_INC_relationState = INC_relationState_mean
                active_INC_contexualState = INC_contexualState_mean
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_src_seq, active_src_enc, active_inst_idx_to_position_map, active_INC_relationState, active_INC_contexualState

        def beam_decode_step(
                inst_dec_beams, len_dec_seq, src_seq, enc_output, inst_idx_to_position_map, beam_size,
                INC_relationState_mean, INC_contexualState_mean):
            ''' Decode and update beam status, and then return active beam idx '''

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)

                return dec_partial_seq

            def predict_word(dec_seq, src_seq, enc_output, n_active_inst, beam_size, INC_relationState_mean, INC_contexualState_mean):
                dec_state = self.decoder.init_decoder_state(src_seq, enc_output)

                decoder_hidden_states, _ = self.decoder(dec_seq, enc_output, dec_state)

                if self.learning_type == 'GEO':
                    combined_states = self.conversion(torch.cat((INC_relationState_mean, INC_contexualState_mean), 1).view(len(INC_contexualState_mean), 1, -1))
                    logits = self.OP3FLayer(torch.cat((decoder_hidden_states, combined_states.expand(decoder_hidden_states.shape)), 2))
                    dec_output = self.generator.forward(logits)
                else:
                    dec_output = self.generator.forward(decoder_hidden_states)

                # output token
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h

                word_prob = nn.functional.log_softmax(dec_output, dim=1)
                word_prob = word_prob.view(n_active_inst, beam_size, -1)

                return word_prob

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]
                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)

            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            word_prob = predict_word(dec_seq, src_seq, enc_output, n_active_inst, beam_size, INC_relationState_mean, INC_contexualState_mean)

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)

            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores

        with torch.no_grad():
            src_seq_orin = copy.deepcopy(src_seq)

            src_enc = self.encoder(src_seq, attention_mask=src_mask)[0]
            n_inst, len_s, d_h = src_enc.size()

            if 'classification' not in self.learning_type:
                if self.learning_type == 'aux' or self.learning_type == 'GEO':
                    INC_relationState_mean, group_batch_idx ,plural_number_bool= self.extract_mean_hiddenstates(gd_loc_label, src_enc)
                    INC_contexualState_mean, ip_batch_idx, plural_number_bool = self.extract_mean_hiddenstates(ip_loc_label, src_enc,
                                                                                              groupdiff=False)

                    Irs_shape = INC_relationState_mean.shape

                    if False in plural_number_bool:
                        empty = torch.zeros((len(INC_contexualState_mean), Irs_shape[1], Irs_shape[2])).type(
                            INC_relationState_mean.type())
                        empty[plural_number_bool] = INC_relationState_mean
                        INC_relationState_mean = empty

                    INC_relationState_mean = INC_relationState_mean.repeat(1, beam_size, 1) \
                        .view(len(INC_relationState_mean) * beam_size, INC_relationState_mean.shape[1], d_h)
                    INC_contexualState_mean = INC_contexualState_mean.repeat(1, beam_size, 1) \
                        .view(len(INC_contexualState_mean) * beam_size, INC_contexualState_mean.shape[1], d_h)

                src_seq_orin = src_seq_orin.repeat(1, beam_size).view(n_inst * beam_size, src_seq_orin.shape[1])
                src_seq = src_seq.repeat(1, beam_size).view(n_inst * beam_size, src_seq.shape[1])
                src_enc = src_enc.repeat(1, beam_size, 1).view(n_inst * beam_size, len_s, d_h)
                src_mask = src_mask.repeat(1, beam_size).view(n_inst * beam_size, src_mask.shape[1])

                # -- Prepare beams
                inst_dec_beams = [Beam(beam_size, device=self.device, indexes=self.indexes) for _ in range(n_inst)]

                # -- Bookkeeping for active or not
                active_inst_idx_list = list(range(n_inst))
                inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

                # -- Decode
                for len_dec_seq in range(1, self.len_max_seq + 1):
                    active_inst_idx_list = beam_decode_step(
                        inst_dec_beams, len_dec_seq, src_seq, src_enc, inst_idx_to_position_map, beam_size,
                        INC_relationState_mean, INC_contexualState_mean)

                    if not active_inst_idx_list:
                        break  # all instances have finished their path to <EOS>

                    src_seq, src_enc, inst_idx_to_position_map, INC_relationState_mean, INC_contexualState_mean = \
                        collate_active_info(src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list,
                                            INC_relationState_mean, INC_contexualState_mean)

                batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, n_best)

                if self.learning_type == 'MTL_generation':
                    ridx = 0

                    src_row = []
                    src_col = []

                    operator_row = []
                    operator_col = []
                    for row in range(len(batch_hyp)):
                        for b in range(beam_size):
                            src = src_seq_orin[ridx]

                            tmpbatch = torch.tensor(batch_hyp[row][b]).type(src_seq.type())

                            tmp = torch.cat((src[:torch.sum(src_mask[ridx])-1],tmpbatch,torch.tensor([self.indexes['SEP']]).type(src.type())))

                            if len(tmp) < self.len_max_src:
                                tmp = torch.cat((tmp,torch.zeros(self.len_max_src-len(tmp)).type(tmp.type())))
                            else:
                                tmp = tmp[:self.len_max_src]

                            src_seq_orin[ridx] = tmp[:self.len_max_src]

                            src_condition = torch.where(tmp==self.op_idx['op'])[0]
                            op_condition = torch.where(tmpbatch==self.op_idx['op'])[0]
                            if len(src_condition) != 0:
                                op_condition = op_condition[:len(src_condition)]
                                src_row.extend([ridx]*len(src_condition))
                                operator_row.extend([[row,b]]*len(op_condition))
                                src_col.extend(list(src_condition.cpu().numpy()))
                                operator_col.extend(list(op_condition.cpu().numpy()))

                            ridx += 1

                    operator_hiddens = self.encoder(src_seq_orin)[0]
                    if len(src_row) != 0:
                        opertor_logit = self.operator_classifier(operator_hiddens[src_row,src_col])
                        opertor_index = list(torch.max(opertor_logit, 1)[1].cpu().numpy())
                        # operators = [self.op_num[self.operators.keys()[i]] for i in opertor_index]
                        count = 0
                        for row,b in operator_row:
                            if batch_hyp[row][b][operator_col[count]] != self.op_idx['op']:
                                print("error")
                            batch_hyp[row][b][operator_col[count]] = self.op_idx[list(self.op_idx.keys())[opertor_index[count]]]
                            count += 1
            else:
                batch_scores, batch_idxex, batch_idxexOperator, value_scores, batch_hyp  = [],[],[],[],[]
                template_index = self.template_classifier(src_enc[:,0,:])
                template_index = list(torch.max(template_index, 1)[1].cpu().numpy())
                if self.learning_type == 'classification':
                    batch_hyp = [[self.op_norm_labeldict[i]] for i in template_index]
                else:
                    batch_hyp = [[i] for i in template_index]
                    ridx = 0
                    src_row = []
                    src_col = []

                    operator_row = []
                    operator_col = []
                    for row in range(len(template_index)):
                        for b in range(1):
                            src = src_seq_orin[ridx]
                            batch_hyp[row][b]= self.op_norm_labeldict[batch_hyp[row][b]][1:]
                            tmpbatch = torch.tensor(batch_hyp[row][b]).type(src_seq.type())
                            tmp = torch.cat((src[:torch.sum(src_mask[ridx]) - 1], tmpbatch,
                                             torch.tensor([self.indexes['SEP']]).type(src.type())))
                            # tmp = torch.cat((src[:torch.sum(src_mask[ridx]) - 1], tmpbatch))
                            if len(tmp) < self.len_max_src:
                                tmp = torch.cat((tmp, torch.zeros(self.len_max_src - len(tmp)).type(tmp.type())))
                            else:
                                tmp = tmp[:self.len_max_src]
                            src_seq_orin[ridx] = tmp[:self.len_max_src]

                            src_condition = torch.where(tmp == self.op_idx['op'])[0]
                            op_condition = torch.where(tmpbatch == self.op_idx['op'])[0]
                            if len(src_condition) != 0:
                                op_condition = op_condition[:len(src_condition)]
                                src_row.extend([ridx] * len(src_condition))
                                operator_row.extend([[row, b]] * len(op_condition))
                                src_col.extend(list(src_condition.cpu().numpy()))
                                operator_col.extend(list(op_condition.cpu().numpy()))

                        ridx += 1

                    operator_hiddens = self.encoder(src_seq_orin)[0]
                    if len(src_row) != 0:
                        opertor_logit = self.operator_classifier(operator_hiddens[src_row, src_col])

                        opertor_index = list(torch.max(opertor_logit, 1)[1].cpu().numpy())
                        # operators = [self.op_num[self.operators.keys()[i]] for i in opertor_index]
                        count = 0
                        for row, b in operator_row:
                            if batch_hyp[row][b][operator_col[count]] != self.op_idx['op']:
                                print("error")
                            batch_hyp[row][b][operator_col[count]] = self.op_idx[
                                list(self.op_idx.keys())[opertor_index[count]]]
                            count += 1

        return batch_hyp, batch_scores

    def extract_mean_hiddenstates(self, loc_label, src_enc, groupdiff=True):
        if groupdiff:
            states, labels, batch_idx = self.extract_states_label(loc_label, src_enc)

        else:
            states, labels, batch_idx = self.extract_states_label(loc_label, src_enc,groupdiff=False)

        starts = 0
        tensor_mean = []
        plural_number_bool = []
        for key, batch in enumerate(batch_idx):
            tensor_mean.append(torch.mean(states[starts:starts + len(batch)], 0))
            if len(batch) >1:
                plural_number_bool.append(True)
            else:
                plural_number_bool.append(False)
            starts += len(batch)

        tensor_mean = torch.stack(tensor_mean)
        return tensor_mean, batch_idx, plural_number_bool

    def greedy_decode(self, src_seq, src_mask, train=False):
        # with torch.no_grad():
        enc_output = self.encoder(src_seq, attention_mask=src_mask)[0]
        n_inst, len_s, d_h = enc_output.size()
        inst_dec_beams = [Beam(1, device=self.device, indexes=self.indexes) for _ in range(n_inst)]

        # -- Bookkeeping for active or not
        dec_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
        dec_seq = torch.stack(dec_seq).to(self.device)
        dec_seq = dec_seq.view(-1, 1)
        pred_lists = torch.tensor(list(range(n_inst)))

        filtered_list = copy.deepcopy(pred_lists)
        dec_logits = []

        for i in range(self.len_max_seq - 1):
            if len(filtered_list) == 0:
                break
            dec_state = self.decoder.init_decoder_state(src_seq[filtered_list], enc_output[filtered_list])
            dec_outputLogits, _ = self.decoder(dec_seq[filtered_list], enc_output[filtered_list], dec_state)
            logits = self.generator.forward(dec_outputLogits)
            logits = logits[:, -1, :]  # Pick the last step: (bh * bm) * d_h
            dec_logits.append(logits.view(len(filtered_list), 1, -1))

            dec_outputprob = nn.functional.log_softmax(logits, dim=1)
            dec_output = dec_outputprob.max(-1)[1]

            temp = torch.ones(n_inst, 1).type(dec_output.type()) * self.indexes['SEP']
            temp[filtered_list] = dec_output.view(-1, 1)

            dec_seq = torch.cat((dec_seq, temp), 1).contiguous()

            filtered_list = filtered_list[dec_output != self.indexes['SEP']]
        if train:
            dec_logits = torch.cat(dec_logits, 1)
            return dec_logits, enc_output
        else:
            preds = [[list(seq.cpu().numpy())] for seq in dec_seq]
            return preds, [], [], [], []



