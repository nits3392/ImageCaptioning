from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

import numpy as np
from json import encoder

import sys
import time
import os
import json
import h5py
import random
import argparse

import multiprocessing

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

    
class DataLoader(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = Iterator(split, self)
        self.iterators[split] = 0

    def __init__(self):
        self.info = json.load(open('data/cocotalk.json'))
        self.vocab_size = len(self.info['ix_to_word'])
        self.h5_label_file = h5py.File('data/cocotalk_label.h5', 'r', driver='core')
        seq_size = self.h5_label_file['labels'].shape
        self.ix_to_word = self.info['ix_to_word']        
        self.seq_length = seq_size[1]
        self.batch_size=8
        self.seq_per_img=5
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]
        self.num_images = self.label_start_ix.shape[0]
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
        self.iterators = {'train': 0, 'val': 0, 'test': 0}
        
        self._prefetch_process = {}
        for split in self.iterators.keys():
            self._prefetch_process[split] = Iterator(split, self)

    def get_vocab(self):
        return self.ix_to_word

    def get_batch(self, split, batch_size=None):
        fc_batch = [] 
        att_batch = []
        label_batch = np.zeros([8 * 5, self.seq_length + 2], dtype = 'int')
        mask_batch = np.zeros([8 * 5, self.seq_length + 2], dtype = 'float32')

        new_epoch = False

        infos = []
        gts = []

        for i in range(8):
            import time
            t_start = time.time()
            tmp_fc, tmp_att,ix, tmp_new_epoch = self._prefetch_process[split].get()
            fc_batch += [tmp_fc] * 5
            att_batch += [tmp_att] * 5
            ix1 = self.label_start_ix[ix] - 1
            ix2 = self.label_end_ix[ix] - 1
            ncap = ix2 - ix1 + 1

            if ncap < 5:
                seq = np.zeros([5, self.seq_length], dtype = 'int')
                for q in range(5):
                    ixl = random.randint(ix1,ix2)
                    seq[q, :] = self.h5_label_file['labels'][ixl, :self.seq_length]
            else:
                ixl = random.randint(ix1, ix2 - 5 + 1)
                seq = self.h5_label_file['labels'][ixl: ixl + 5, :self.seq_length]
            
            label_batch[i * 5 : (i + 1) * 5, 1 : self.seq_length + 1] = seq

            if tmp_new_epoch:
                new_epoch = True

            gts.append(self.h5_label_file['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]])

            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)

        t_start = time.time()
        nonzeros = np.array(list(map(lambda x: (x != 0).sum()+2, label_batch)))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1

        data = {}
        data['fc_feats'] = np.stack(fc_batch)
        data['att_feats'] = np.stack(att_batch)
        data['labels'] = label_batch
        data['gts'] = gts
        data['masks'] = mask_batch 
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'new_epoch': new_epoch}
        data['infos'] = infos

        return data

    def __getitem__(self, index):
        return np.load(os.path.join('data/cocotalk_fc', str(self.info['images'][index]['id']) + '.npy')), np.load(os.path.join('data/cocotalk_att', str(self.info['images'][index]['id']) + '.npz'))['feat'], index

    def __len__(self):
        return len(self.info['images'])

class Iterator():
    def __init__(self, split, dataloader):
        self.split = split
        self.dataloader = dataloader
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,batch_size=1,sampler=self.dataloader.split_ix[self.split][self.dataloader.iterators[self.split]:],pin_memory=True,num_workers=multiprocessing.cpu_count(),collate_fn=lambda x: x[0]))
    
    def get(self):
        max_index = len(self.dataloader.split_ix[self.split])
        new_epoch = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            random.shuffle(self.dataloader.split_ix[self.split])
            new_epoch = True
        self.dataloader.iterators[self.split] = ri_next

        tmp = self.split_loader.next()
        if new_epoch:
            self.split_loader = iter(data.DataLoader(dataset=self.dataloader,batch_size=1,sampler=self.dataloader.split_ix[self.split][self.dataloader.iterators[self.split]:],pin_memory=True,num_workers=multiprocessing.cpu_count(),collate_fn=lambda x: x[0]))
        return tmp + [new_epoch]
    
def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)



class AttentionModel(nn.Module):
    def __init__(self, vocab_size, seq_len):
        super(AttentionModel, self).__init__()
        self.vocab_size = vocab_size        
        self.input_encoding_size = 512
        self.rnn_type = "lstm"
        self.rnn_size = 512
        self.num_layers = 1
        self.drop_prob_lm = 0.5
        self.fc_feat_size = 2048
        self.att_feat_size = 2048
        self.att_hid_size = 512
        self.seq_length = seq_len
        self.ss_prob = 0.0
        self.linear = nn.Linear(self.fc_feat_size, self.num_layers * self.rnn_size) # feature to rnn_size
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.dropout = nn.Dropout(self.drop_prob_lm)
        self.init_weights()
        
        self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size + self.att_feat_size, 
                self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)

        self.ctx2att = nn.Linear(self.att_feat_size, self.att_hid_size)
        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, fc_feats):
        image_map = self.linear(fc_feats).view(-1, self.num_layers, self.rnn_size).transpose(0, 1)
        return (image_map, image_map)            

    def forward(self, fc_feats, att_feats, seq):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(fc_feats)

        outputs = []

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0:
                sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[-1].data)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    it = Variable(it, requires_grad=False)
            else:
                it = seq[:, i].clone()          
            if i >= 1 and seq[:, i].data.sum() == 0:
                break

            xt = self.embed(it)
            att_size = att_feats.numel() // att_feats.size(0) // self.att_feat_size
            att = att_feats.view(-1, self.att_feat_size)

            att = self.ctx2att(att)                            
            att = att.view(-1, att_size, self.att_hid_size)     
            att_h = self.h2att(state[0][-1])                    
            att_h = att_h.unsqueeze(1).expand_as(att)           
            dot = att + att_h                                   
            dot = F.tanh(dot)                                   
            dot = dot.view(-1, self.att_hid_size)               
            dot = self.alpha_net(dot)                           
            dot = dot.view(-1, att_size)                                                        

            weight = F.softmax(dot)
            att_feats_ = att_feats.view(-1, att_size, self.att_feat_size) 
            att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)

            output, state = self.rnn(torch.cat([xt, att_res], 1).unsqueeze(0), state)
            output = output.squeeze(0)
            output = F.log_softmax(self.logit(self.dropout(output)))
            outputs.append(output)

        return torch.cat([_.unsqueeze(1) for _ in outputs], 1)
    def sample(self, fc_feats, att_feats, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(fc_feats)

        seq = []
        seqLogprobs = []
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.data.new(batch_size).long().zero_()
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu() 
                else:
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1).cuda()
                sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False)) 
                it = it.view(-1).long() 

            xt = self.embed(Variable(it, requires_grad=False))

            if t >= 1:
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it) #seq[t] the input of t+2 time step
                seqLogprobs.append(sampleLogprobs.view(-1))
            att_size = att_feats.numel() // att_feats.size(0) // self.att_feat_size
            att = att_feats.view(-1, self.att_feat_size)

            att = self.ctx2att(att)                             
            att = att.view(-1, att_size, self.att_hid_size)     
            att_h = self.h2att(state[0][-1])                    
            att_h = att_h.unsqueeze(1).expand_as(att)           
            dot = att + att_h                                   
            dot = F.tanh(dot)                                   
            dot = dot.view(-1, self.att_hid_size)               
            dot = self.alpha_net(dot)                           
            dot = dot.view(-1, att_size)                                                       

            weight = F.softmax(dot)
            att_feats_ = att_feats.view(-1, att_size, self.att_feat_size) 
            att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)

            output, state = self.rnn(torch.cat([xt, att_res], 1).unsqueeze(0), state)
            output = output.squeeze(0)
            logprobs = F.log_softmax(self.logit(self.dropout(output)))

        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)

def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out

def language_eval(preds):
    sys.path.append("coco-caption")
    annFile = 'coco-caption/annotations/captions_val2014.json'
    encoder.FLOAT_REPR = lambda o: format(o, '.3f')
    cache_path = os.path.join('abc_val.json')
    coco = COCO(annFile)
    valids = coco.getImgIds()
    preds_filt = [p for p in preds if p['image_id'] in valids]
    json.dump(preds_filt, open(cache_path, 'w'))
    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score
    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)
    return out

def eval_split(model, crit, loader, args):
    verbose = True
    num_images = 10
    split = 'val'
    lang_eval =  0
    dataset = 'coco'
    beam_size =  1
    
    model.eval()
    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size
        if data.get('labels', None) is not None:
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]
            tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
            fc_feats, att_feats, labels, masks = tmp
            loss = crit(model(fc_feats, att_feats, labels), labels[:,1:], masks[:,1:]).data[0]
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img], 
            data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
        tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        fc_feats, att_feats = tmp
        seq, _ = model.sample(fc_feats, att_feats)
        sents = decode_sequence(loader.get_vocab(), seq)
        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            cmd = 'cp "' + os.path.join(args.data_dir,data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' 
            #print(cmd)
            print('image %s: %s' %(entry['image_id'], entry['caption']))
            os.system(cmd)
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()
        if data['bounds']['new_epoch']:
            break
        if num_images >= 0 and n >= num_images:
            break
    lang_stats = language_eval(predictions)
    return lang_stats

def main():
    loader = DataLoader()
    iteration = 0
    epoch = 0
    model = AttentionModel(loader.vocab_size,loader.seq_length)
    model.cuda()
    model.train()
    crit = LanguageModelCriterion()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default='/scratch/ns3664/data')

    args = parser.parse_args()
    if args.eval != 0:
        #Evaluate
        
        model.load_state_dict(torch.load('checkpoint_dir/model.pth'))
        lang_stat=eval_split(model, crit, loader, args)
        print(lang_stat)
    else:
        #Train
        
        while True:
            data = loader.get_batch('train')

            torch.cuda.synchronize()
            start = time.time()

            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]
            tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
            fc_feats, att_feats, labels, masks = tmp

            optimizer.zero_grad()
            loss = crit(model(fc_feats, att_feats, labels), labels[:,1:], masks[:,1:])
            loss.backward()
            clip_gradient(optimizer, 0.1)
            optimizer.step()
            train_loss = loss.data[0]
            torch.cuda.synchronize()
            print("Training step:{},epoch:{},loss:{:.3f}".format(iteration, epoch, train_loss))

            iteration += 1

            # One epoch is of 14160 steps when batch size is 8
            if iteration>0 and iteration % 14160 == 0:
                epoch += 1
                frac = epoch // 3
                decay_factor = 0.8  ** frac
                current_lr = 5e-4 * decay_factor
                for group in optimizer.param_groups:
                    group['lr'] = current_lr


            # Validation after 1000 steps and save model after that
            if (iteration % 100 == 0):
                checkpoint_path = os.path.join('checkpoint_dir', 'model.pth')
                torch.save(model.state_dict(), checkpoint_path)
                optimizer_path = os.path.join('checkpoint_dir', 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)

            if epoch >= 30:
                break

main()
