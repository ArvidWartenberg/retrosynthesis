import math
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from time import time
import argparse
import pdb

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader

chars = " ^$#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy"
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
VOCAB_SIZE=len(chars)
TGT_VOCAB_SIZE, SRC_VOCAB_SIZE = VOCAB_SIZE,VOCAB_SIZE
MAX_SEQ_LEN=160
PAD_IDX, BOS_IDX, EOS_IDX = 0,1,2

file = open("/home/arvid/data/USTPO_paper_5x/USTPO_5x_parsed.pickle",'rb')
data = pickle.load(file)



class ReactionDataset(Dataset):

    def __init__(self, data, split, maxlen=160, rep=" ^$#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy"):
        self.split = split
        self.data = data[self.split]
        self.maxlen = maxlen
        self.rep = rep
        self.char_to_ix = { ch:i for i,ch in enumerate(rep) }
        self.ix_to_char = { i:ch for i,ch in enumerate(rep) }
        # Add augmentation methods here later
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        rs_smiles = self.data[index]["rs"]
        if rs_smiles[0] == ".":
            rs_smiles = rs_smiles[1:]
        ps_smiles = self.data[index]["ps"]
        
        
        rs_smiles = self.ix_to_char[BOS_IDX] + rs_smiles + self.ix_to_char[EOS_IDX] + (self.maxlen-len(rs_smiles))*" "
        ps_smiles = ps_smiles + (self.maxlen-len(ps_smiles)+2)*" "
        
        # Augment smiles here for train
        
        rs = torch.tensor([self.char_to_ix[char] for char in rs_smiles])
        ps = torch.tensor([self.char_to_ix[char] for char in ps_smiles])
        
        return {
            "rs": rs.to(dtype=torch.int64),
            "ps": ps.to(dtype=torch.int64)
            #'rs': F.one_hot(rs.to(dtype=torch.int64), num_classes=len(self.rep)),
            #'ps':  F.one_hot(ps.to(dtype=torch.int64), num_classes=len(self.rep))
        }

datasets = {}
dataloaders = {}
for split in ['train', 'eval']:
    datasets[split] = ReactionDataset(data=data,
                                   split=split)

    dataloaders[split] = DataLoader(datasets[split],
                                    batch_size=32,
                                    shuffle=(split != 'test'),
                                    num_workers=4,
                                    pin_memory=False)# Was True before.


def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX)
    tgt_padding_mask = (tgt == PAD_IDX)
    tgt_padding_mask[tgt == EOS_IDX] = True
    return src_mask, tgt_mask, src_padding_mask.permute(1,0), tgt_padding_mask.permute(1,0)

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 200):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        token_embedding = token_embedding
        return self.dropout((token_embedding + self.pos_embedding[:token_embedding.size(0), :]))
    
    
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                    N_ENCODERS: int=6,
                    N_DECODERS: int=6,
                    EMBEDDING_SIZE: int=512,
                    N_HEADS: int=8,
                    SRC_VOCAB_SIZE: int=VOCAB_SIZE,
                    TGT_VOCAB_SIZE: int=VOCAB_SIZE,
                    DIM_FF: int=512,
                    DROPOUT: float=0.0):
        super(Seq2SeqTransformer, self).__init__()
        
        
        self.transformer = Transformer(d_model=EMBEDDING_SIZE,
                                       nhead=N_HEADS,
                                       num_encoder_layers=N_ENCODERS,
                                       num_decoder_layers=N_DECODERS,
                                       dim_feedforward=DIM_FF,
                                       dropout=DROPOUT)
        
        self.generator = nn.Linear(EMBEDDING_SIZE, TGT_VOCAB_SIZE)
        
        self.src_tok_emb = TokenEmbedding(SRC_VOCAB_SIZE, EMBEDDING_SIZE)
        self.tgt_tok_emb = TokenEmbedding(TGT_VOCAB_SIZE, EMBEDDING_SIZE)
        
        self.positional_encoding = PositionalEncoding(
            EMBEDDING_SIZE, dropout=DROPOUT)

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor, src_padding_mask: Tensor):
        return self.transformer.encoder(src=self.positional_encoding(
                            self.src_tok_emb(src)), mask=src_mask, src_key_padding_mask=src_padding_mask)

    def decode(self, tgt: Tensor, memory: Tensor, memory_key_padding_mask: Tensor):
        
        return self.transformer.decoder(tgt=self.positional_encoding(self.tgt_tok_emb(tgt)),
                                        memory=memory,
                                        memory_key_padding_mask=memory_key_padding_mask)
    
def loss_fn(logits, tgt, tgt_padding_mask):
    #Shift outs and labels one step
    logits = logits[:-1,:,:].permute(1,0,2)
    one_hot_targets = F.one_hot(tgt, num_classes=VOCAB_SIZE)[1:,:,:].permute(1,0,2)
    weights = (~tgt_padding_mask[:,:-1]).long().unsqueeze(-1).tile([1,1,TGT_VOCAB_SIZE])
    
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    loss = torch.sum(torch.mean(criterion(logits, one_hot_targets.float())*weights,axis=0),axis=(0,1))
    return loss




class CosineAnnealingWarmup():

    def __init__(self,
                 max_lr : float = 1e-3,
                 min_lr : float = 1e-4,
                 warmup_steps : int = 10,
                 base_lr: float = 1e-4,
                 gamma : float = 1,
                 last_epoch : int = 200
        ):
        
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        self.base_lr = base_lr
        
        
    
    def get_lr(self, step):
        if step <= self.warmup_steps:
            return (self.max_lr - self.base_lr)*step / self.warmup_steps + self.base_lr
        else:
            return self.base_lr + (self.max_lr - self.base_lr) \
                    * (1+ math.cos(math.pi * (step - self.warmup_steps)/(step*1.5- self.warmup_steps))) / 2
    
class up_down_down_down():
    def __init__(self,a,b,c):
        self.a = a
        self.b = b
        self.c = c
    
    def get_lr(self,step):
        if step <= 20:
            return a
        elif step <= 50:
            return b
        elif step <= 90:
            return a
        else:
            return c
    

def train_epoch(model, optimizer, split):
    if split == "train": model.train()
    else: model.eval()
        
    losses = 0
    for batch in dataloaders[split]: #tqdm(dataloaders[split], colour='WHITE'):
        src = batch["ps"].to(DEVICE).permute(1,0)
        tgt = batch["rs"].to(DEVICE).permute(1,0)


        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src=src, tgt=tgt)

        logits = model(src=src,tgt=tgt,src_mask=src_mask,
                            tgt_mask=tgt_mask,
                            src_padding_mask=src_padding_mask,
                            tgt_padding_mask=tgt_padding_mask,
                            memory_key_padding_mask=src_padding_mask)


        optimizer.zero_grad()

        loss = loss_fn(logits, tgt, tgt_padding_mask)
        if split=="train": loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(dataloaders[split])

def training_monitoring(model, optimizer, scheduler, n_epochs, args):
    
    model_dir = os.path.expanduser("~/models/transformers/")
    time()
    from datetime import datetime
    date_time = datetime.fromtimestamp(time())
    model_str = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    os.mkdir(model_dir + model_str)
    os.mkdir(model_dir + model_str + "/logdir")
    print("tensorboard --logdir " + model_dir + model_str + "/logdir/ --port=6006")
    
    
    
    with open(model_dir + model_str + '/params.pickle', 'wb') as handle:
        pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    best_loss = 999999999
    running_losses = {'train': [], 'eval': []}    
        
    # Loop
    writer = SummaryWriter(model_dir + model_str + "/logdir/")
    
    for epoch in range(n_epochs):
        losses = {}
        print()
        
        lr = scheduler.get_lr(epoch)
        print("Beginning of epoch %i LR: %.4f"%(epoch, lr), end="")
        for g in optimizer.param_groups:
            g['lr'] = lr
        
        for split in ["train", "eval"]:
            t1 = time()
            print("..." + split, end="")
            #print("Epoch " + str(epoch) + " " + split + " progression:")
            epoch_loss = train_epoch(model=model, optimizer=optimizer, split=split)
            
            losses[split] = epoch_loss
            running_losses[split].append(epoch_loss)
            writer.add_scalar('Loss/'+split, epoch_loss, epoch+1)
            
            if split == 'eval' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = model.state_dict()
                    torch.save(model.state_dict(),model_dir + model_str + "/weights")
            print("...finished in %.2f s! "%(time()-t1), end="")
        print()
        print("Epoch %i losses are; Train: %.2f // Eval: %.2f."%(epoch,losses["train"],losses["eval"]))
        
        
def parseArguments(default_args):
        # Create argument parser
        parser = argparse.ArgumentParser()

        # Optional arguments
        for key in default_args.keys():
            parser.add_argument("--"+key, help=key, default=default_args[key][0], type=default_args[key][1])
        
        parser.add_argument("--checkpoint", help="Load model checkpoint", required=False)
        parser.add_argument("--device", help="Pick device", required=False)
        
        args = parser.parse_args()

        return args
    
def main():
    default_args= {"N_ENCODERS": [6, int],
                    "N_DECODERS": [6, int],
                    "EMBEDDING_SIZE": [512, int],
                    "N_HEADS": [8, int],
                    "DIM_FF": [512, int],
                    "DROPOUT": [0.1, float],
                    "LR": [1e-5, float],
                    "EPOCHS": [200, int],
                    "BATCH_SIZE": [32, int],
                    "VOCAB_SIZE": [VOCAB_SIZE, int],
                    "MAX_SEQ_LEN": [MAX_SEQ_LEN, int],
                    "MAX_LR": [1e-3, float],
                    "MIN_LR": [1e-4, float],
                    "WARMUP_STEPS": [10, int],
                    "BASE_LR": [1e-4, float],
                    "GAMMA": [1.0, float]}
    args = parseArguments(default_args=default_args).__dict__
    
    #pdb.set_trace()
    print()
    print("RUN PARAMS: ")
    for a in args:
        print(str(a) + ": " + str(args[a]))
        
    if args["device"] is not None: dev=args["device"]
    else: dev="1"
    global DEVICE
    DEVICE = torch.device('cuda:'+dev if torch.cuda.is_available() else 'cpu')
    torch.torch.cuda.set_device(DEVICE)
    print("Using device " + str(torch.torch.cuda.current_device()) + "/" + str(torch.cuda.device_count())
          +", name: " + str(torch.cuda.get_device_name(0)))
    
    #torch.manual_seed(1337)

    chemFormer = Seq2SeqTransformer(N_ENCODERS=args["N_ENCODERS"],
                                    N_DECODERS=args["N_DECODERS"],
                                    EMBEDDING_SIZE=args["EMBEDDING_SIZE"],
                                    N_HEADS=args["N_HEADS"],
                                    SRC_VOCAB_SIZE=SRC_VOCAB_SIZE,
                                    TGT_VOCAB_SIZE=TGT_VOCAB_SIZE,
                                    DIM_FF=args["DIM_FF"],
                                    DROPOUT=args["DROPOUT"]).to(DEVICE)
    
    if args["checkpoint"] is not None:
        models_dir = "/home/arvid/models/transformers/"
        chemFormer.load_state_dict(torch.load(models_dir + args["checkpoint"] + "/weights"))
        print("Successfully loaded checkpoint: " + models_dir + args["checkpoint"] + "/weights")   
 

    optimizer = torch.optim.Adam(chemFormer.parameters(), lr=args["LR"], betas=(0.9, 0.98), eps=1e-9)
    
    scheduler_cos = CosineAnnealingWarmup(max_lr=args["MAX_LR"],
                                     min_lr=args["MIN_LR"],
                                     warmup_steps=args["WARMUP_STEPS"],
                                     base_lr=args["BASE_LR"],
                                     gamma=args["GAMMA"],
                                     last_epoch=args["EPOCHS"])
    scheduler_me = up_down_down_down(1e-4,1e-3,1e-5)
    
    training_monitoring(model=chemFormer, optimizer=optimizer, scheduler=scheduler_me, n_epochs=args["EPOCHS"], args=args)
        

if __name__ == '__main__':
    main()