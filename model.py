import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from argparse import ArgumentParser
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class LightningInterface(pl.LightningModule):
    def __init__(self, threshold=0.5, **kwargs):
        super().__init__()
        self.best_f1 = 0.
        self.threshold = threshold
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCEWithLogitsLoss()
        self.automatic_optimization = False
        self.validation_step_outputs = []

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        if type(y_hat) == tuple:
            y_hat, attn_scores,sc = y_hat
            
        opt = self.optimizers()
        opt.zero_grad()
        loss = self.criterion(y_hat, y)
        self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=0.1)
        opt.step()
        tensorboard_logs = {'train_loss': loss}
        # import pdb; pdb.set_trace()
        # self.log('lr', self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0], on_step=True)
        # self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=True)
        return {'loss': loss, 'log': tensorboard_logs}

    # def training_epoch_end(self, output) -> None:
    #     self.log('lr', self.hparams.lr)
    
    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        if type(y_hat) == tuple:
            y_hat, attn_scores,sc = y_hat
        yy, yy_hat = y.detach().cpu().numpy(), y_hat.sigmoid().detach().cpu().numpy()
        loss = self.criterion(y_hat, y)
        self.validation_step_outputs.append({'val_loss': loss, "labels": yy, "probs": yy_hat})
        return {'val_loss': loss, "labels": yy, "probs": yy_hat}

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     all_labels = np.concatenate([x['labels'] for x in outputs])
    #     all_probs = np.concatenate([x['probs'] for x in outputs])
    #     all_preds = (all_probs > self.threshold).astype(float)
    #     acc = np.mean(all_labels == all_preds)
    #     p = precision_score(all_labels, all_preds)
    #     r = recall_score(all_labels, all_preds)
    #     f1 = f1_score(all_labels, all_preds)
    #     self.best_f1 = max(self.best_f1, f1)
    #     if self.current_epoch == 0:  # prevent the initial check modifying it
    #         self.best_f1 = 0
    #     # return {'val_loss': avg_loss, 'val_acc': avg_acc, 'hp_metric': self.best_acc}
    #     tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc, 'val_p': p, 'val_r': r, 'val_f1': f1, 'hp_metric': self.best_f1}
    #     # import pdb; pdb.set_trace()
    #     self.log_dict(tensorboard_logs)
    #     self.log("best_f1", self.best_f1, prog_bar=True, on_epoch=True)
    #     return {'val_loss': avg_loss, 'log': tensorboard_logs}
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        all_labels = np.concatenate([x['labels'] for x in outputs])
        all_probs = np.concatenate([x['probs'] for x in outputs])
        all_preds = (all_probs > self.threshold).astype(float)
        acc = np.mean(all_labels == all_preds)
        p = precision_score(all_labels, all_preds)
        r = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        self.best_f1 = max(self.best_f1, f1)
        if self.current_epoch == 0:  # prevent the initial check modifying it
            self.best_f1 = 0
        # return {'val_loss': avg_loss, 'val_acc': avg_acc, 'hp_metric': self.best_acc}
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc, 'val_p': p, 'val_r': r, 'val_f1': f1, 'hp_metric': self.best_f1}
        # import pdb; pdb.set_trace()
        self.log_dict(tensorboard_logs)
        self.log("best_f1", self.best_f1, prog_bar=True, on_epoch=True)
        self.validation_step_outputs.clear()  # free memory
        log_text = f"val  f1:{f1} , bestf1:{self.best_f1} val_loss:{avg_loss}\n"
        with open('log.text','a') as f:
            f.write(log_text)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}
        
        
        
        
    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        if type(y_hat) == tuple:
            y_hat, attn_scores,sc = y_hat
        yy, yy_hat = y.detach().cpu().numpy(), y_hat.sigmoid().detach().cpu().numpy()
        return {'test_loss': self.criterion(y_hat, y), "labels": yy, "probs": yy_hat}
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        all_labels = np.concatenate([x['labels'] for x in outputs])
        all_probs = np.concatenate([x['probs'] for x in outputs])
        all_preds = (all_probs > self.threshold).astype(float)
        acc = np.mean(all_labels == all_preds)
        p = precision_score(all_labels, all_preds)
        r = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        log_text = f"test  f1:{f1} , bestf1:{self.best_f1} val_loss:{avg_loss}\n"
        with open('log.text','a') as f:
            f.write(log_text)
        return {'test_loss': avg_loss, 'test_acc': acc, 'test_p': p, 'test_r': r, 'test_f1': f1}

    def on_after_backward(self):
        pass
        # can check gradient
        # global_step = self.global_step
        # if int(global_step) % 100 == 0:
        #     for name, param in self.named_parameters():
        #         self.logger.experiment.add_histogram(name, param, global_step)
        #         if param.requires_grad:
        #             self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        return parser


class Classifier(LightningInterface):
    def __init__(self, threshold=0.5, lr=5e-5, model_type="/mnt/proj/bert-tiny", **kwargs):
        super().__init__(threshold=threshold, **kwargs)

        self.model_type = model_type
        self.model = BERTFlatClassifier(model_type)
        self.lr = lr
        # self.lr_sched = lr_sched
        self.save_hyperparameters()
        #print(self.hparams)

    def forward(self, x):
        x = self.model(**x)
        return x

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = LightningInterface.add_model_specific_args(parser)
        parser.add_argument("--threshold", type=float, default=0.5)
        parser.add_argument("--lr", type=float, default=2e-4)
        # parser.add_argument("--lr_sched", type=str, default="none")
        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=4, hidden_size=64, activation=nn.ReLU()):
        """
        MOE Args:
            input_dim: dimension of input
            output_dim: dimension of output
            num_experts: number of experts
            hidden_size: dimension of FNN hidden layer
        """
        super().__init__()
        self.num_experts = num_experts

        # 专家网络列表：每个专家是一个简单的MLP
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                activation,
                nn.Linear(hidden_size, output_dim)
            ) for _ in range(num_experts)
        ])

        # 门控网络：学习样本到专家的权重分布
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),  # 直接输出专家数量维度的logits
            nn.Softmax(dim=1)                   # 转换为概率分布
        )

    def forward(self, x):
        """
        input:  [batch_size, input_dim]
        output: [batch_size, output_dim]
        """
        # 门控网络计算权重 [batch_size, num_experts]
        gate_weights = self.gate(x)
        
        # 计算所有专家的输出 [num_experts, batch_size, output_dim]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        # 维度变换用于矩阵乘法 [batch_size, num_experts, 1]
        gate_weights = gate_weights.unsqueeze(-1)
        #print(expert_outputs.shape,gate_weights.shape)
        # 加权求和 [batch_size, output_dim]
        output = torch.sum(gate_weights * expert_outputs, dim=1)
        return output

    
# using model
class BERTHierClassifierTransAbs(nn.Module):
    def __init__(self, model_type, num_heads=8, num_trans_layers=6, max_posts=64, freeze=False, pool_type="first") -> None:
        super().__init__()
        self.model_type = model_type
        self.num_heads = num_heads
        self.num_trans_layers = num_trans_layers
        self.pool_type = pool_type
        self.post_encoder = AutoModel.from_pretrained(model_type)
        if freeze:
            for name, param in self.post_encoder.named_parameters():
                param.requires_grad = False
        self.hidden_dim = self.post_encoder.config.hidden_size
        self.max_posts = max_posts
        
        
        self.time_embedding = nn.Embedding(24, 1)
        self.text_effect_projection = nn.Linear(384, 1)
        #self.time_effect_projection = nn.Linear(64, 1)
        self.basic_fector = nn.Parameter(torch.Tensor([1]))
        self.basic_timeline = nn.Parameter(torch.Tensor(1,768))
        self.activation = nn.Tanh()
        
        self.history_proj = nn.Linear(384, 768)
        self.fusion = nn.MultiheadAttention(768, 8)
        #self.norm = RMSNorm(768)
        
        
        
        # batch_first = False
        self.pos_emb = nn.Parameter(torch.Tensor(max_posts, self.hidden_dim))
        nn.init.xavier_uniform_(self.pos_emb)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu')
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        
        self.ana_fusion = nn.MultiheadAttention(768, 8,batch_first=True)
        
        self.expert1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gate1 = nn.Linear(self.hidden_dim, 1)
        self.expert2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gate2 = nn.Linear(self.hidden_dim, 1)
        
        self.dropout = nn.Dropout(self.post_encoder.config.hidden_dropout_prob)
        self.clf = nn.Linear(self.hidden_dim, 1)
        
    
    def forward(self, batch, **kwargs):
        feats = []
        attn_scores = []
        sleep_schedule = []
        for user_feats in batch:
            embs = user_feats["embs"]  #(seqlen, 384)
            timelines = user_feats["timepoints_emb"]#(seqlen, 768)
            timeindex = user_feats["timeindex"] #(seqlen)
            seqlen = timelines.shape[0]
            
            text_factor = self.text_effect_projection(embs)
            factor = self.time_embedding(timeindex)
            #factor = self.activation(text_factor) #(seqlen,1)
            #print(text_factor.shape,factor.shape,timelines.shape)
            factor = factor + self.basic_fector
            timeline_emb = factor * timelines
            timeline_emb = torch.cat((self.basic_timeline,timeline_emb),dim=0)
            integration_timeline = timeline_emb.sum(0).view(1,1,-1) / (seqlen/4)
            #raise Exception(integration_timeline.shape)
            
            history_feature = self.history_proj(torch.mean(embs,dim=0))
            
            post_outputs = self.post_encoder(user_feats["input_ids"], user_feats["attention_mask"], user_feats["token_type_ids"])
            analyze_outputs = self.post_encoder(user_feats["input_ids_a"], user_feats["attention_mask_a"], user_feats["token_type_ids_a"])
            # [num_posts, seq_len, hidden_size] -> [num_posts, 1, hidden_size]
            if self.pool_type == "first":
                x = post_outputs.last_hidden_state[:, 0:1, :]
                a = analyze_outputs.last_hidden_state[:, 0:1, :]
            elif self.pool_type == 'mean':
                x = mean_pooling(post_outputs.last_hidden_state, user_feats["attention_mask"]).unsqueeze(1)
                a = mean_pooling(analyze_outputs.last_hidden_state, user_feats["attention_mask_a"]).unsqueeze(1)
            #raise Exception(x.shape,a.shape)
            try:
                x_a ,att = self.ana_fusion(x,a,a)
            except Exception:
                raise Exception("报错",x.shape,a.shape)
            x = x_a + x
            #x = x+a
            x=torch.cat((history_feature.view(1,1,-1),x),dim=0)
            
            x_,att = self.fusion(x,integration_timeline,integration_timeline)
            x = x_ + x
            # positional embedding for posts
            x = x + self.pos_emb[:x.shape[0], :].unsqueeze(1)
            #raise Exception(x.dtype,timeline.dtype)
            x = torch.cat((x,integration_timeline),dim=0)
            
            
            x = self.user_encoder(x).squeeze(1) # [num_posts, hidden_size]
            # [num_posts, ]
            x1 = self.expert1(x)
            x2 = self.expert2(x)
            g1 = self.gate1(x)
            g2 = self.gate2(x)
            
            #raise Exception(x1.shape,g1.shape,x2.shape,g2.shape)
            feat = g1.squeeze(-1) @ x1 + g2.squeeze(-1) @ x2
            #raise Exception(feat.shape)
            feats.append(feat)
            attn_scores.append(feat)
            sleep_schedule.append(integration_timeline)
        feats = torch.stack(feats)
        #raise Exception(feats.shape)
        x = self.dropout(feats)
        logits = self.clf(x).squeeze(-1)
        #logits = self.moe_clf(x).squeeze(-1)
        # [bs, num_posts]
        return logits, attn_scores,sleep_schedule[-1]






class HierClassifier(LightningInterface):
    def __init__(self, threshold=0.5, lr=5e-5, model_type="/mnt/proj/bert-tiny", user_encoder="none", num_heads=8, num_trans_layers=2, freeze_word_level=False, pool_type="first", vocab_size=30522, **kwargs):
        super().__init__(threshold=threshold, **kwargs)

        self.model_type = model_type
        self.model = BERTHierClassifierTransAbs(model_type, num_heads, num_trans_layers, freeze=freeze_word_level, pool_type=pool_type)
       
        self.lr = lr
        # self.lr_sched = lr_sched
        self.save_hyperparameters()
        #print(self.hparams)

    def forward(self, x):
        x = self.model(x)
        return x

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = LightningInterface.add_model_specific_args(parser)
        parser.add_argument("--threshold", type=float, default=0.5)
        parser.add_argument("--lr", type=float, default=2e-4)
        # parser.add_argument("--trans", action="store_true")
        parser.add_argument("--user_encoder", type=str, default="none")
        parser.add_argument("--pool_type", type=str, default="first")
        parser.add_argument("--num_heads", type=int, default=8)
        parser.add_argument("--num_trans_layers", type=int, default=2)
        parser.add_argument("--freeze_word_level", action="store_true")
        # parser.add_argument("--lr_sched", type=str, default="none")
        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer