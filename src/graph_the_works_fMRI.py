import random

import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler, SequentialSampler
from .utils import calculate_accuracy, Cutout, calculate_accuracy_by_labels, calculate_FP, calculate_FP_Max
from .newtrainer import Trainer
from src.utils import EarlyStopping, EarlyStoppingACC, EarlyStoppingACC_and_Loss
from torchvision import transforms
import matplotlib.pylab as plt
import matplotlib.pyplot as pl
import torchvision.transforms.functional as TF
import torch.nn.utils.rnn as tn
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import torch.nn.utils.rnn as tn
from sklearn.linear_model import LogisticRegression
from itertools import combinations, product
from torch.nn import TripletMarginLoss
from torch import int as tint, long, short, Tensor
from random import sample
from sys import maxsize
from collections import Counter
import csv
import time
import math

class Classifier(nn.Module):
    def __init__(self, num_inputs1, num_inputs2):
        super().__init__()
        self.network = nn.Bilinear(num_inputs1, num_inputs2, 1)

    def forward(self, x1, x2):
        return self.network(x1, x2)


class the_works_trainer(Trainer):
    def __init__(self, model, config, device, device_encoder, tr_labels, val_labels, test_labels, test_labels2="", trial="",
                 crossv="", gtrial="", tr_FNC="tr_FNC", val_FNC="val_FNC",test_FNC="test_FNC"):
        super().__init__(model, device)
        self.config = config
        self.device_encoder = device_encoder
        self.tr_labels = tr_labels
        self.test_labels = test_labels
        self.test_labels2= test_labels2
        self.val_labels = val_labels
        self.tr_FNC = tr_FNC
        self.val_FNC = val_FNC
        self.test_FNC = test_FNC
        self.criterion2 = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.78))
        self.patience = self.config["patience"]
        self.dropout = nn.Dropout(0.65).to(device)
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.sample_number = config['sample_number']
        self.path = config['path']
        self.oldpath = config['oldpath']
        self.fig_path = config['fig_path']
        self.p_path = config['p_path']
        self.PT = config['pre_training']
        self.device = device
        self.gain = config['gain']
        self.train_epoch_loss, self.train_batch_loss, self.eval_epoch_loss, self.eval_batch_loss, self.eval_batch_accuracy, self.train_epoch_accuracy = [], [], [], [], [], []
        self.train_epoch_roc, self.eval_epoch_roc = [], []
        self.eval_epoch_CE_loss, self.eval_epoch_E_loss, self.eval_epoch_lstm_loss = [], [], []
        self.test_accuracy = 0.
        self.test_auc = 0.
        self.test_precision = 0.
        self.test_recall = 0.
        self.test_loss = 0.
        self.n_heads = 1
        self.edge_weights= ""
        self.temporal_edge_weights = ""
        self.edge_weights_sum = ""
        self.attention_region = ""
        self.attention_time = ""
        self.attention_weights = ""
        self.attention_component = ""
        self.attention_time_embeddings  = ""
        self.FNC = ""
        self.trials = trial
        self.gtrial = gtrial
        self.exp = config['exp']
        self.cv = crossv
        self.test_targets = ""
        self.test_targets2 =""
        self.test_predictions = ""
        self.regions_selected = ""
        self.means_labels = ""
        self.loss_criteria = nn.L1Loss()
        self.triplet_loss_function = TripletMarginLoss(margin=0.5)
        self.lr = config['lr']
        self.dropout = nn.Dropout(0.65).to(self.device)

        if self.exp in ['UFPT', 'NPT']:
            self.optimizer = torch.optim.Adam(self.model.parameters(),lr=config['lr'])
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=9e-5, max_lr=config['lr'], step_size_up= 2000, cycle_momentum=False)
        # else:
        #     if self.PT in ['milc', 'milc-fMRI', 'variable-attention', 'two-loss-milc']:
        #         self.optimizer = torch.optim.Adam(list(self.decoder.parameters()),lr=config['lr'], eps=1e-5)
        #     else:
        #         self.optimizer = torch.optim.Adam(list(self.decoder.parameters()) + list(self.attn.parameters())
        #                                                + list(self.lstm.parameters()) + list(self.key_layer.parameters())
        #                                           + list(self.value_layer.parameters()) + list(self.query_layer.parameters())
        #                                           +  list(self.multihead_attn.parameters()),
        #                                           lr=config['lr'], eps=1e-5)

        self.early_stopper = EarlyStopping("self.model_backup",  patience=self.patience, verbose=False,
                                           wandb="self.wandb", name="model",
                                           path=self.path, trial=self.trials)
        self.transform = transforms.Compose([Cutout(n_holes=1, length=80)])
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.001, epochs=80,
        #                                                 steps_per_epoch=5, pct_start=0.2,
        #                                                 div_factor=0.001 / config['lr'], final_div_factor=10000,verbose=True)

    def find_value_ids(self, it, value):
        """
        Args:
            it: list of any
            value: query element

        Returns:
            indices of the all elements equal x0
        """
        if isinstance(it, np.ndarray):
            inds = list(np.where(it == value)[0])
        else:  # could be very slow
            inds = [i for i, el in enumerate(it) if el == value]
        return inds

    def _check_input_labels(self, labels):
        """
        The input must satisfy the conditions described in
        the class documentation.

        Args:
            labels: labels of the samples in the batch
        """
        labels_counter = Counter(labels)
        assert all(n > 1 for n in labels_counter.values())
        assert len(labels_counter) > 1

    def mysample(self, features, labels):
        if isinstance(labels, Tensor):
            labels = labels.tolist()
        self._check_input_labels(labels)
        ids_anchor, ids_pos, ids_neg = self.mysample2(features, labels=labels)
        return features[ids_anchor], features[ids_pos], features[ids_neg]

    def mysample2(self, features, labels):
        num_labels = len(labels)

        triplets = []
        for label in set(labels):
            ids_pos_cur = set(self.find_value_ids(labels, label))
            ids_neg_cur = set(range(num_labels)) - ids_pos_cur

            pos_pairs = list(combinations(ids_pos_cur, r=2))

            tri = [(a, p, n) for (a, p), n in product(pos_pairs, ids_neg_cur)]
            triplets.extend(tri)

        triplets = sample(triplets, min(len(triplets), maxsize))
        ids_anchor, ids_pos, ids_neg = zip(*triplets)

        return list(ids_anchor), list(ids_pos), list(ids_neg)

    def TripletLoss(self, features, labels):
        """
        Args:
            features: features with shape [batch_size, features_dim]
            labels: labels of samples having batch_size elements

        Returns: loss value

        """
        # labels_list = convert_labels2list(labels)

        features_anchor, features_positive, features_negative = self.mysample(
            features=features, labels=labels
        )

        loss = self.triplet_loss_function(
            anchor=features_anchor, positive=features_positive, negative=features_negative,
        )
        return loss

    def generate_batch(self, episodes, mode):
        if self.sample_number == 0:
            total_steps = sum([len(e) for e in episodes])
        else:
            total_steps = self.sample_number

        if mode == 'train' or mode == 'eval':
            BS = self.batch_size
        else:
            BS = self.batch_size#math.ceil(episodes.shape[0]/5)


        # print('episodes shape = ', episodes.shape)
        # print(len(episodes))
        # episodes = episodes[150:182,:,:,:]
        # episodes = episodes.permute(0,2,1,3).reshape(32 * 100, 160, 1)
        # print(len(episodes))
        # packed = tn.pack_sequence(episodes, enforce_sorted=False)
        # return
        if mode == 'seq':
            # sampler = BatchSampler(RandomSampler(range(len(episodes)),
            #                                      replacement=True),
            #                        BS, drop_last=False)
            sampler = BatchSampler(SequentialSampler(range(len(episodes))),
                                   BS, drop_last=False)
        else:
            sampler = BatchSampler(RandomSampler(range(len(episodes)),
                                             replacement=False),
                               BS, drop_last=False)

        for indices in sampler:
            # print('length of episodes', len(episodes))
            # print(indices)
            # print('episode shape',episodes.shape)
            # print(episodes[199,:,:,:])
            episodes_batch = [episodes[x,:,:,:] for x in indices]

            # episodes_batch  = torch.stack(episodes_batch)
            # episodes_batch = episodes_batch.permute(0,2,1,3).reshape(32 * 100, 160, 1)
            # packed = tn.pack_sequence(episodes_batch, enforce_sorted=False)
            # return

            ts_number = torch.LongTensor(indices)
            i = 0
            sx = []
            # for episode in episodes_batch:
            #     # Get all samples from this episode
            #     # mean = episode.mean()
            #     # sd = episode.std()
            #     # episode = (episode - mean) / sd
            #     sx.append(episode)
            yield torch.stack(episodes_batch).to(self.device_encoder), ts_number.to(self.device_encoder)


    def get_prediction_loss(self, preds, target, variance=5e-5, add_const=False):
        neg_log_p = ((preds - target) ** 2 / (2 * variance))
        if add_const:
            const = 0.5 * np.log(2 * np.pi * variance)
            neg_log_p += const
        return neg_log_p.sum() / (target.size(0) * target.size(1))

    def mixup_data(self, x, y, alpha=1.0, device='cuda'):

        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        mixed_y = lam * y + (1 - lam) * y[index]

        ones = mixed_y >= 0.5
        zeros = mixed_y < 0.5
        mixed_y[zeros] = 0
        mixed_y[ones] = 1
        return mixed_x, y_a, y_b, mixed_y, lam

    def do_one_epoch(self, epoch, episodes, mode):

        epoch_loss, epoch_loss2, epoch_loss3, accuracy, steps, epoch_acc, epoch_roc, epoch_roc2, epoch_prec, epoch_recall = 0., 0., 0, 0., 0., 0., 0., 0., 0., 0.
        epoch_CE_loss, epoch_E_loss, epoch_lstm_loss = 0., 0., 0.,
        accuracy1, accuracy2, accuracy, FP = 0., 0., 0., 0.
        epoch_loss_mi, epoch_loss_mse, epoch_accuracy, epoch_accuracy2, epoch_FP,epoch_total_loss = 0., 0., 0., 0.,0., 0.
        all_logits=''

        data_generator = self.generate_batch(episodes, mode)
        for sx, ts_number in data_generator:
            FNC=""
            # print('sx shape', sx.shape)
            # sx = sx.permute(0,2,1,3).reshape(sx.shape[0] * 100, 160, 1)
            # packed = tn.pack_sequence(sx, enforce_sorted=False)
            # return

            # mean = sx.mean()
            # sd = sx.std()
            # sx = (sx - mean) / sd
            loss = 0.
            loss2 = 0.
            loss3 = 0.
            diag = 0.
            ndiag = 0.
            lam = 0.
            loss_total , loss_pred=0.,0.
            CE_loss, E_loss, lstm_loss = 0., 0., 0.
            targets = ""
            targets2 = ""
            if mode == 'train':
                targets = self.tr_labels[ts_number]
                # FNC = self.tr_FNC[ts_number,:]

            elif mode == 'eval':
                targets = self.val_labels[ts_number]
                # FNC = self.val_FNC[ts_number, :]

            else:
                targets = self.test_labels[ts_number]
                # if self.test_labels2 != "":
                #     targets2=self.test_labels2[ts_number]
                #     targets2=targets2.to(self.device)
                # FNC = self.test_FNC[ts_number, :, :]
            # print('sx shape = ', sx.shape)
            # sx = sx.reshape(32 * 160, 100, 1)
            # packed = tn.pack_sequence(sx, enforce_sorted=False)
            # return
            # if mode == 'test' or mode == 'tst':
            #     print(targets[:15])
            #
            #     f = targets == 1
            #     m = targets == 0
            #     print(torch.sum(f))
            #     print(torch.sum(m))
            targets = targets.long()
            targets = targets.to(self.device)
            # logits, FC, _, FC_sum, attention_time,attention_weights, means_logits,selected_indices,ENC_from_means = self.model(sx, targets, mode, self.device, epoch)


            logits, kl_loss, FC, FC_temporal = self.model(sx, targets, mode, self.device, epoch)



            loss = F.cross_entropy(logits, targets)




            # loss =  loss + loss_mse + loss2
            if mode == 'train' or mode == 'eval':
                # loss = loss + kl_loss
                loss, CE_loss, E_loss, lstm_loss = self.add_regularization(loss)

            # print("reg time", time.time() - t)
            t = time.time()
            # accuracy2, roc2, pred2 = self.acc_and_auc(logits2.detach(), mode, targets.detach())
            # loss_total = loss + loss2 + loss3
            # accuracy2, roc2 = self.acc_and_auc(encoder_logits.detach(), mode, targets.detach())
            # print("auc time", time.time() - t)
            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()
            # t = time.time()
            # loss = loss + loss2
            # accuracy = (accuracy + accuracy2)/2
            # roc = (roc + roc2)/2
            # if mode == "test":
            #     epoch += torch.sqrt(loss.detach().item())
            epoch_loss += loss.detach().item()
            # epoch_loss2 += (loss2.detach().item())
            # epoch_loss3 += (loss3.detach().item())

            # epoch_loss_mi += loss2.detach().item()
            # epoch_loss_mse += loss3.detach().item()
            # epoch_total_loss = epoch_loss + epoch_loss_mi + epoch_loss_mse
            # epoch_accuracy += accuracy.detach().item()
            # epoch_accuracy2 += accuracy2#.detach().item()

            if all_logits == '':
                all_logits = logits.detach()
                all_targets = targets.detach()
            else:
                all_logits = torch.cat((all_logits, logits.detach()), dim=0)
                all_targets = torch.cat((all_targets, targets.detach()), dim=0)



            if mode == 'train' or mode == 'eval':
                # epoch_CE_loss += CE_loss.detach().item()
                epoch_E_loss += E_loss
                # epoch_lstm_loss += lstm_loss.detach().item()
            # if mode != 'train':
            #     epoch_roc += roc
            #     epoch_prec += prec
            #     epoch_recall += recall
                # epoch_roc2 += roc2

            if mode == 'test':
                if self.edge_weights == "":

                    self.edge_weights = FC.detach()
                    # self.temporal_edge_weights = FC_temporal.detach()
                    # self.edge_weights_sum = FC_sum.detach()
                    # self.attention_region = attention_region.detach()
                    # self.attention_time = attention_time.detach()
                    # self.attention_weights = attention_weights.detach()
                    # self.means_labels = selected_indices.detach()
                    # self.regions_selected = region_indices.detach()
                    # self.attention_component = attention_component.detach()
                    # self.attention_time_embeddings = attention_time_embeddings.detach()
                    # self.test_targets = all_targets
                    # if self.test_labels2!="":
                    #     self.test_targets2 = targets2.detach()
                    # self.test_predictions = pred
                    # self.regions_selected = regions_selected
                    # self.FNC = FNC
                else:
                    self.edge_weights = torch.cat((self.edge_weights,FC.detach()),dim=0)
                    # self.temporal_edge_weights = torch.cat((self.temporal_edge_weights, FC_temporal.detach()), dim=0)
                    # self.edge_weights_sum = torch.cat((self.edge_weights_sum, FC_sum.detach()), dim=0)
                    # self.attention_time = torch.cat((self.attention_time, attention_time.detach()), dim=0)
                    # if self.test_labels2 != "":
                    #     self.test_targets2 = torch.cat((self.test_targets2, targets2.detach()), dim=0)
                    # self.attention_weights = torch.cat((self.attention_weights, attention_weights.detach()), dim=0)
                    # self.means_labels = torch.cat((self.means_labels, selected_indices.detach()), dim=0)

                    # self.regions_selected = torch.cat((self.regions_selected, region_indices.detach()), dim=0)
            del loss
            del loss2
            # del loss3
            del targets
            del diag
            del FC
            del logits
            del loss_total
            # del pred
            #print("junk time", time.time() - t)
            steps += 1

        accuracy, roc, pred, prec, recall = self.acc_and_auc(all_logits, mode, all_targets)
        if mode != 'train':
            epoch_roc += roc * steps
            epoch_prec += prec * steps
            epoch_recall += recall * steps
        # epoch_accuracy += accuracy * steps
        #
        epoch_accuracy += accuracy.detach().item() * steps

        #t = time.time()
        if mode == "eval":
            self.eval_batch_accuracy.append(epoch_accuracy / steps)
            self.eval_epoch_loss.append(epoch_loss / steps)
            self.eval_epoch_roc.append(epoch_roc / steps)
            self.eval_epoch_CE_loss.append(epoch_CE_loss / steps)
            self.eval_epoch_E_loss.append(epoch_E_loss / steps)
            self.eval_epoch_lstm_loss.append(epoch_lstm_loss / steps)
        elif mode == 'train':
            self.train_epoch_loss.append(epoch_loss / steps)
            self.train_epoch_accuracy.append(epoch_accuracy / steps)
        if epoch % 1 == 0:
          self.log_results(epoch, epoch_loss / steps, epoch_loss / steps, epoch_loss / steps, epoch_accuracy / steps,
                       epoch_accuracy2 / steps, epoch_roc / steps, epoch_roc2 / steps, epoch_prec / steps, epoch_recall / steps, prefix=mode)
        if mode == "eval" and epoch > -1:
            best_on =   (epoch_accuracy / steps)# + (epoch_roc / steps) # +1/(epoch_loss / steps) +
            self.early_stopper(epoch_loss / steps, best_on, self.model, 0, epoch=epoch)
        if mode == 'test':
            self.test_accuracy = epoch_accuracy / steps
            self.test_auc = epoch_roc / steps
            self.test_loss = epoch_loss / steps
            self.test_precision = epoch_prec / steps
            self.test_recall = epoch_recall / steps
            self.test_targets = all_targets
        #print("last time", time.time() - t)
        return epoch_loss / steps

    def acc_and_auc(self, logits, mode, targets):
        # print(targets)
        # N = logits.size(0)
        # sig = torch.zeros(N, 2).to(self.device)
        sig = torch.softmax(logits, dim=1)
        values, indices = sig.max(1)

        # sig = torch.sigmoid(logits).reshape(-1)
        # indices = (sig > 0.5).int()
        roc = 0.
        prec = 0.
        rec = 0.
        acc = 0.
        # return acc, roc, indices, prec, rec
        # y_scores = sig.detach().gather(1, targets.to(self.device).long().view(-1,1))
        if 1 in targets and 0 in targets:
            if mode != 'train':
                y_scores = (sig.detach()[:, 1]).float()
                roc = roc_auc_score(targets.to('cpu'), y_scores.to('cpu'))
                prec = precision_score(targets.to('cpu'), indices.to('cpu'))
                rec = recall_score(targets.to('cpu'), indices.to('cpu'))
        accuracy = calculate_accuracy_by_labels(indices, targets)

        if mode == 'test':
            print(indices)
            print(targets)

        return accuracy, roc, indices, prec, rec


    def add_regularization(self, loss, ortho_loss=0.0):
        reg = 1e-6
        E_loss = 0.
        lstm_loss = torch.zeros(1).to(self.device)
        orth_loss = torch.zeros(1).to(self.device)
        attn_loss = 0.
        mha_loss = 0.
        CE_loss = loss
        encoder_loss=torch.zeros(1).to(self.device)

        # ortho_loss = reg * ortho_loss

        # for name, param in self.model.garo_key_components.named_parameters():
        #     if 'bias' not in name:
        #         param_flat = param.view(param.shape[0], -1)
        #         sym = torch.mm(param_flat, torch.t(param_flat))
        #         sym -= torch.eye(param_flat.shape[0],device=self.device)
        #         orth_loss = orth_loss + (reg * sym.abs().sum())
        # for name, param in self.model.garo_query_components.named_parameters():
        #     if 'bias' not in name:
        #         param_flat = param.view(param.shape[0], -1)
        #         sym = torch.mm(param_flat, torch.t(param_flat))
        #         sym -= torch.eye(param_flat.shape[0],device=self.device)
        #         orth_loss = orth_loss + (reg * sym.abs().sum())

        for name, param in self.model.gta_embed.named_parameters():
            if 'bias' not in name:
                lstm_loss += (reg * torch.norm(param,p=1))

        for name, param in self.model.gta_attend.named_parameters():
            if 'bias' not in name:
                lstm_loss += (reg * torch.norm(param,p=1))

        # for name, param in self.model.encoder.named_parameters():
        #     if 'bias' not in name:
        #         encoder_loss += (reg * torch.norm(param,p=1))

        # for name, param in self.model.up_sample.named_parameters():
        #     if 'bias' not in name:
        #         lstm_loss += (reg * torch.norm(param,p=1))

        # for name, param in self.model.embedder.named_parameters():
        #     if 'bias' not in name:
        #         encoder_loss += (reg * torch.norm(param,p=1))

        # for name, param in self.model.transformer_encoder.named_parameters():
        #     if 'bias' not in name:
        #         encoder_loss += (reg * torch.norm(param,p=1))



        # for name, param in self.model.encoder.named_parameters():
        #     if 'bias' not in name:
        #         encoder_loss += (reg * torch.sum(torch.abs(param)))

        # for name, param in self.model.lstm.named_parameters():
        #     if 'bias' not in name:
        #         lstm_loss += (reg * torch.sum(torch.abs(param)))
        #
        # for name, param in self.model.lstm_decoder2.named_parameters():
        #     if 'bias' not in name:
        #         lstm_loss += (reg * torch.sum(torch.abs(param)))
        #
        # for name, param in self.model.lstm_decoder3.named_parameters():
        #     if 'bias' not in name:
        #         lstm_loss += (reg * torch.sum(torch.abs(param)))


        # for name, param in self.model.lstm_decoder_time.named_parameters():
        #     if 'bias' not in name:
        #         lstm_loss += (reg * torch.sum(torch.abs(param)))


        # for name, param in self.model.decoder.named_parameters():
        #     if 'bias' not in name:
        #         lstm_loss += (reg * torch.sum(torch.abs(param)))
        # for name, param in self.model.attn_time.named_parameters():
        #     if 'bias' not in name:
        #         lstm_loss += (reg * torch.sum(torch.abs(param)))
        # for name, param in self.model.mlp.named_parameters():
        #     if 'bias' not in name:
        #         lstm_loss += (reg * torch.norm(param,p=1))

        # for name, param in self.model.mlp_before_lstm.named_parameters():
        #     if 'bias' not in name:
        #         lstm_loss += (reg * torch.sum(torch.abs(param)))

        # for name, param in self.model.attn_region.named_parameters():
        #     if 'bias' not in name:
        #         lstm_loss += (reg * torch.sum(torch.abs(param)))
        #
        # for name, param in self.model.attn_spatial.named_parameters():
        #     if 'bias' not in name:
        #         lstm_loss += (reg * torch.sum(torch.abs(param)))

        # for name, param in self.model.attn_weight.named_parameters():
        #     if 'bias' not in name:
        #         lstm_loss += (reg * torch.sum(torch.abs(param)))

        # for name, param in self.model.key_layer_temporal.named_parameters():
        #     if 'bias' not in name:
        #         lstm_loss += (reg * torch.sum(torch.abs(param)))
        #
        # for name, param in self.model.value_layer_temporal.named_parameters():
        #     if 'bias' not in name:
        #         lstm_loss += (reg * torch.sum(torch.abs(param)))
        #
        # for name, param in self.model.query_layer_temporal.named_parameters():
        #     if 'bias' not in name:
        #         lstm_loss += (reg * torch.sum(torch.abs(param)))
        #
        # for name, param in self.model.multihead_attn_temporal.named_parameters():
        #     if 'bias' not in name:
        #         lstm_loss += (reg * torch.norm(param,p=1))
        # for name, param in self.model.garo_key.named_parameters():
        #     if 'bias' not in name:
        #         lstm_loss += (reg * torch.sum(torch.abs(param)))
        # for name, param in self.model.garo_query.named_parameters():
        #     if 'bias' not in name:
        #         lstm_loss += (reg * torch.sum(torch.abs(param)))

        # for name, param in self.model.key_layer2.named_parameters():
        #     if 'bias' not in name:
        #         lstm_loss += (reg * torch.sum(torch.abs(param)))
        #
        # for name, param in self.model.value_layer2.named_parameters():
        #     if 'bias' not in name:
        #         lstm_loss += (reg * torch.sum(torch.abs(param)))
        #
        # for name, param in self.model.query_layer2.named_parameters():
        #     if 'bias' not in name:
        #         lstm_loss += (reg * torch.sum(torch.abs(param)))
        #
        # for name, param in self.model.multihead_attn2.named_parameters():
        #     if 'bias' not in name:
        #         lstm_loss += (reg * torch.sum(torch.abs(param)))


        loss = loss +  lstm_loss.to(self.device) #+ encoder_loss
        return loss, CE_loss, E_loss, lstm_loss

    def validate(self, val_eps):

        model_dict = torch.load(os.path.join(self.p_path, 'encoder' + self.trials + '.pt'), map_location=self.device)
        self.encoder.load_state_dict(model_dict)
        self.encoder.eval()
        self.encoder.to(self.device)

        model_dict = torch.load(os.path.join(self.p_path, 'lstm' + self.trials + '.pt'), map_location=self.device)
        self.lstm.load_state_dict(model_dict)
        self.lstm.eval()
        self.lstm.to(self.device)

        # model_dict = torch.load(os.path.join(self.p_path, 'decoder' + self.trials + '.pt'), map_location=self.device)
        # self.decoder.load_state_dict(model_dict)
        # self.decoder.eval()
        # self.decoder.to(self.device)

        mode = 'eval'
        self.do_one_epoch(0, val_eps, mode)
        return self.test_auc

    def load_model_and_test(self, tst_eps):
        print('Best model was', self.early_stopper.epoch_saved)
        model_dict = torch.load(os.path.join(self.path, 'model' + self.trials + '.pt'), map_location=self.device)
        self.model.load_state_dict(model_dict)
        self.model.eval()
        #self.encoder.to(self.device_encoder)

        # model_dict = torch.load(os.path.join(self.path, 'lstm' +  self.trials + '.pt'), map_location=self.device)
        # self.lstm.load_state_dict(model_dict)
        # self.lstm.eval()
        # self.lstm.to(self.device)
        #
        # model_dict = torch.load(os.path.join(self.path, 'attn' +  self.trials + '.pt'), map_location=self.device)
        # self.attn.load_state_dict(model_dict)
        # self.attn.eval()
        # self.attn.to(self.device)
        #
        # model_dict = torch.load(os.path.join(self.path, 'cone' + self.trials + '.pt'), map_location=self.device)
        # self.decoder.load_state_dict(model_dict)
        # self.decoder.eval()
        # self.decoder.to(self.device)

        #model_dict = torch.load(os.path.join(self.path, 'key' + self.trials + '.pt'), map_location=self.device)
        #self.key_layer.load_state_dict(model_dict)
        #self.key_layer.eval()
        #self.key_layer.to(self.device)

        #model_dict = torch.load(os.path.join(self.path, 'value' + self.trials + '.pt'), map_location=self.device)
        #self.value_layer.load_state_dict(model_dict)
        #self.value_layer.eval()
        #self.value_layer.to(self.device)

        #model_dict = torch.load(os.path.join(self.path, 'query' + self.trials + '.pt'), map_location=self.device)
        #self.query_layer.load_state_dict(model_dict)
        #self.query_layer.eval()
        #self.query_layer.to(self.device)

        #model_dict = torch.load(os.path.join(self.path, 'mha' + self.trials + '.pt'), map_location=self.device)
        #self.multihead_attn.load_state_dict(model_dict)
        #self.multihead_attn.eval()
        #self.multihead_attn.to(self.device)


        mode = 'test'
        self.do_one_epoch(0, tst_eps, mode)

    def save_loss_and_auc(self):

        with open(os.path.join(self.path, 'all_data_information' + self.trials + '.csv'), 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            self.train_epoch_loss.insert(0, 'train_epoch_loss')
            wr.writerow(self.train_epoch_loss)

            self.train_epoch_accuracy.insert(0, 'train_epoch_accuracy')
            wr.writerow(self.train_epoch_accuracy)

            self.eval_epoch_loss.insert(0, 'eval_epoch_loss')
            wr.writerow(self.eval_epoch_loss)

            self.eval_batch_accuracy.insert(0, 'eval_batch_accuracy')
            wr.writerow(self.eval_batch_accuracy)

            self.eval_epoch_roc.insert(0, 'eval_epoch_roc')
            wr.writerow(self.eval_epoch_roc)

            self.eval_epoch_CE_loss.insert(0, 'eval_epoch_CE_loss')
            wr.writerow(self.eval_epoch_CE_loss)

            self.eval_epoch_E_loss.insert(0, 'eval_epoch_E_loss')
            wr.writerow(self.eval_epoch_E_loss)

            self.eval_epoch_lstm_loss.insert(0, 'eval_epoch_lstm_loss')
            wr.writerow(self.eval_epoch_lstm_loss)

    def train(self, tr_eps, val_eps, tst_eps):
        print('lr = ',self.lr)
        # TODO: Make it work for all modes, right now only it defaults to pcl.
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[4, 30, 128, 256, 512, 700, 800, 2500], gamma=0.15)
        #
        # print(self.test_labels.shape[0])
        # return
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5,factor=0.25,cooldown=0,verbose=True )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=4,factor=0.5,cooldown=0,verbose=True )
        #
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

        if self.PT in ['DECENNT']:
            if self.exp in ['UFPT', 'FPT']:
                print('in ufpt and fpt')
                model_dict = torch.load(os.path.join(self.oldpath, 'model' + '.pt'), map_location=self.device)
                self.model.load_state_dict(model_dict)
                self.model.to(self.device)
            else:
                self.model.init_weight(PT=self.exp)
        #
        else:
            self.model.init_weight(PT=self.exp)

        saved = 0

        for e in range(self.epochs):
            #print(self.exp)
            if self.exp in ['UFPT', 'NPT']:
                self.model.train()

            else:
                self.model.eval()

            mode = "train"
            # tr_eps = tr_eps.permute(0, 2, 1, 3).contiguous().reshape(292 * 100, 160, 1)
            # packed = tn.pack_sequence(tr_eps, enforce_sorted=False)
            # return
            #t = time.time()
            val_loss = self.do_one_epoch(e, tr_eps, mode)
            #print("train time", time.time()-t)
            self.model.eval()
            mode = "eval"
            #t = time.time()
            #print("====================================VALIDATION START===============================================")
            val_loss = self.do_one_epoch(e, val_eps, mode)

            # mode="tst"
            # _ = self.do_one_epoch(e, tst_eps, mode)
            # mode = 'check'
            # self.do_one_epoch(e, tst_eps, mode)
            #print("val time",time.time()-t)
            #print("====================================VALIDATION END===============================================")
            #mode = "test"
            #junk = self.do_one_epoch(e, tst_eps, mode)
            # if val_loss < 0.65:
            scheduler.step(val_loss)

            if self.early_stopper.early_stop:
                self.early_stopper(0, 0, self.model, 1, epoch=e)
                saved = 1
                break


        if saved == 0:
            # print('saving')
            self.early_stopper(0, 0, self.model, 1,epoch=e)
            saved = 1

        self.save_loss_and_auc()
        self.load_model_and_test(tst_eps)

        # tr_eps=tr_eps.to('cpu')
        # self.tr_labels=self.tr_labels.to('cpu')
        #
        # tst_eps=tst_eps.to('cpu')
        # self.test_labels=self.test_labels.to('cpu')

        ###################################################################################################
        # lr_auc, acc, lr_auc2, acc2 = 0,0,0,0
        # mode = 'LR'
        # list_FC_train_top = []
        # list_FC_train_bottom =[]
        # for i in range(7):
        #     s = i*22
        #     e = s+22
        #     _, FC_train_top, FC_train_bottom, _, _,_ = self.model(tr_eps[s:e,:,:,:], self.tr_labels[s:e], mode, self.device, 0)
        #     list_FC_train_top.append(FC_train_top)
        #     list_FC_train_bottom.append(FC_train_bottom)
        # FC_train_top = torch.stack(list_FC_train_top)
        # FC_train_bottom = torch.stack(list_FC_train_bottom)
        # # print(FC_train_top.shape)
        # FC_train_top = FC_train_top.reshape(7 * 22,100,100)
        # FC_train_bottom = FC_train_bottom.reshape(7 * 22, 100, 100)
        # FC_train_top=FC_train_top.detach()
        # FC_train_bottom=FC_train_bottom.detach()
        # model = LogisticRegression(penalty='l2', solver='liblinear', max_iter=100, random_state=2)
        # FC_train_top = FC_train_top.to('cpu')
        # model.fit(FC_train_top.reshape(FC_train_top.shape[0],-1).numpy(), self.tr_labels.to('cpu'))
        # mode = 'LR'
        #
        # _, FC_test_top, FC_test_bottom, _, _,_ = self.model(tst_eps, self.test_labels, mode, self.device, 0)
        # FC_test_top = FC_test_top.detach()
        # FC_test_bottom = FC_test_bottom.detach()
        # FC_test_top=FC_test_top.to('cpu')
        # lr_probs = model.predict_proba(FC_test_top.reshape(FC_test_top.shape[0],-1).numpy())
        # lr_probs = lr_probs[:, 1]
        # lr_auc = roc_auc_score(self.test_labels.to('cpu'), lr_probs)
        # acc = model.score(FC_test_top.reshape(FC_test_top.shape[0],-1).numpy(),self.test_labels.to('cpu') )
        #
        # print(
        #     " Top 5 auc: {}, accuracy: {}".format(
        #         lr_auc, acc
        #     ))
        # # ###################################################################################################
        # FC_train_bottom=FC_train_bottom.to('cpu')
        # model = LogisticRegression(penalty='l2', solver='liblinear', max_iter=100, random_state=2)
        # model.fit(FC_train_bottom.reshape(FC_train_bottom.shape[0],-1).numpy(), self.tr_labels.to('cpu'))
        # FC_test_bottom = FC_test_bottom.to('cpu')
        # lr_probs = model.predict_proba(FC_test_bottom.reshape(FC_test_bottom.shape[0],-1).numpy())
        # lr_probs = lr_probs[:, 1]
        # lr_auc2 = roc_auc_score(self.test_labels.to('cpu'), lr_probs)
        # acc2 = model.score(FC_test_bottom.reshape(FC_test_bottom.shape[0],-1).numpy(),self.test_labels.to('cpu') )
        #
        # print(
        #     " Bottom 5 auc: {}, accuracy: {}".format(
        #         lr_auc2, acc2
        #     ))
        # print('logistic regression auc score for bottom 5 % = ', lr_auc)

        ###################################################################################################

        # f = pl.figure()
        #
        # pl.plot(self.train_epoch_loss[1:], label='train_total_loss')
        # pl.plot(self.eval_epoch_loss[1:], label='val_total_loss')
        # pl.plot(self.eval_epoch_CE_loss[1:], label='val_CE_loss')
        # pl.plot(self.eval_epoch_E_loss[1:], label='val_Enc_loss')
        # pl.plot(self.eval_epoch_lstm_loss[1:], label='val_lstm_loss')
        # # #
        # #
        # pl.xlabel('epochs')
        # pl.ylabel('loss')
        # pl.legend()
        # pl.show()
        # f.savefig(os.path.join(self.fig_path, 'all_loss.png'), bbox_inches='tight')
        #
        # f = pl.figure()
        # #
        # pl.plot(self.train_epoch_accuracy[1:], label='train_acc')
        # pl.plot(self.eval_batch_accuracy[1:], label='val_acc')
        # pl.plot(self.eval_epoch_roc[1:], label='val_auc')
        #
        # #
        #
        # pl.xlabel('epochs')
        # pl.ylabel('acc/auc')
        # pl.legend()
        # pl.show()
        # f.savefig(os.path.join(self.fig_path, 'acc.png'), bbox_inches='tight')

        # return self.test_accuracy, self.test_auc, self.test_loss, e
        #
        # print('-----------------')
        # print(self.test_auc)
        # print(self.test_labels.shape[0])
        # print(self.test_auc*self.test_labels.shape[0])
        # print('-----------------')
        # torch.cuda.empty_cache()
        return self.test_accuracy, self.test_auc, self.test_loss, e, self.test_precision, self.test_recall, self.edge_weights#, self.test_targets, #self.edge_weights_sum,self.attention_time#,self.test_targets#,self.test_targets2#, self.attention_weights
        # return self.test_accuracy, self.test_auc, self.test_loss, e, self.test_precision, self.test_recall, lr_auc, acc,lr_auc2, acc2, self.edge_weights, self.edge_weights_sum,\
        # self.attention_time,FC_test_top,FC_test_bottom.reshape(FC_test_top.shape)#, self.attention_component#, self.attention_time_embeddings#, self.test_targets, self.test_predictions, self.regions_selected, self.FNC
        # return self.early_stopper.val_acc_max



    def log_results(self, epoch_idx, epoch_loss2, epoch_loss3, epoch_loss, epoch_test_accuracy, epoch_accuracy2, epoch_roc,
                    epoch_roc2, prec, recall, prefix=""):
        print(
            "{} CV: {}, Trial: {}, Epoch: {}, Loss: {}, Accuracy: {}, roc: {}, prec: {}, recall:{}".format(
                prefix.capitalize(),
                self.cv,
                self.trials,
                epoch_idx,
                epoch_loss,
                # epoch_loss_mse,
                # epoch_loss_mi,
                epoch_test_accuracy,
                epoch_roc,
                prec,
                recall
            ))
