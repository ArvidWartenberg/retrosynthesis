from modules import *
from Trainer import *
from Transformer import *
from Reaction_dataset import *
from Tools import *
from constants import *

os.system("nvidia-smi")

def parseArguments(default_args):
        # Create argument parser
        parser = argparse.ArgumentParser()

        # Optional arguments
        for key in default_args.keys():
            parser.add_argument("--"+key,
                                help=key,
                                default=default_args[key][0],
                                type=default_args[key][1])
        
        parser.add_argument("--checkpoint", help="Load model checkpoint", required=False)
        parser.add_argument("--name", help="Name run", required=False)
        parser.add_argument("--debug", help="Debug", required=False)
        parser.add_argument("--port", help="Tensorboard port", default="6006", required=False)
        parser.add_argument("--dataset", help="Tensorboard port", default="non-aug", required=False)
        
        args = parser.parse_args().__dict__
        args["device"] = "cuda:" + args["device"]

        return args
        
         
def main():
    os.chdir("..")
    home_dir = os.getcwd()
    os.chdir("retrosynthesis_cluster_training/")
    
    args = parseArguments(default_args=default_args)
    args["SEED"] = torch.seed()
    #torch.manual_seed(1337)
    args["home_dir"] = home_dir
    
    print("RUN PARAMS: ")
    for a in args:
        print(str(a) + ": " + str(args[a]))
        

    DEVICE = torch.device(args["device"] if torch.cuda.is_available() else 'cpu')
    torch.torch.cuda.set_device(DEVICE)
    print("Using device " + str(torch.torch.cuda.current_device()) + "/" + str(torch.cuda.device_count())
          +", name: " + str(torch.cuda.get_device_name(0)))
    
    torch.cuda.empty_cache()
    gc.collect()
    
    if args["dataset"] == "non-aug": file = open(args["home_dir"] + "/data/USTPO_not_augmented/data.pickle","rb")
    else: file = open(args["home_dir"] + "/data/USTPO_paper_5x/USTPO_5x_parsed.pickle",'rb')
    
    data = pickle.load(file)
    if args["debug"] is not None: 
        data = {"train": data["train"][0:64], "eval": data["eval"][0:64]}
    datasets = {}
    dataloaders = {}
    for split in ['train', 'eval']:
        datasets[split] = ReactionDataset(data=data,
                                       split=split,
                                         args=args)

        dataloaders[split] = DataLoader(datasets[split],
                                        batch_size=args["BATCH_SIZE"],
                                        shuffle=(split != 'test'),
                                        num_workers=8,
                                        pin_memory=False,
                                        drop_last=True)

        
    chemFormer = Seq2SeqTransformer(num_encoder_layers=args["N_ENCODERS"],
                                    num_decoder_layers=args["N_DECODERS"],
                                    emb_size=args["EMBEDDING_SIZE"],
                                    nhead=args["N_HEADS"],
                                    src_vocab_size=SRC_VOCAB_SIZE,
                                    tgt_vocab_size=TGT_VOCAB_SIZE,
                                    dim_feedforward=args["DIM_FF"],
                                    dropout=args["DROPOUT"],
                                    DEVICE=DEVICE).to(DEVICE)
    
    for p in chemFormer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        
    
    # TODO OUTDATED
    if args["checkpoint"] is not None:
        models_dir = args["home_dir"] + "/models/transformers/"
        chemFormer.load_state_dict(torch.load(models_dir + args["checkpoint"] + "/weights"))
        print("Successfully loaded checkpoint: " + models_dir + args["checkpoint"] + "/weights")   
 
    
    optimizer = torch.optim.Adam(chemFormer.parameters(), lr=args["LR"], betas=(0.9, 0.98), eps=1e-9)
    print(args["EPOCHS"])
    
    
    if args["SCHEDULER"] == "noam":
        scheduler = NoamLR(optimizer=optimizer,
                           model_size=512,
                           warmup_steps=args["WARMUP_STEPS"],
                           last_epoch=-1)
    elif args["SCHEDULER"] == "cos":
        scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer,
                                                  first_cycle_steps=200,
                                                  cycle_mult=1.0,
                                                  max_lr=7e-4,
                                                  min_lr=1e-5,
                                                  warmup_steps=40,
                                                  gamma=0.5,
                                                  last_epoch=-1)
    #scheduler = None
    
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    trainer = Trainer(model=chemFormer,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      loss_fn=loss_fn,
                      dataloaders=dataloaders,
                      args=args)
    trainer.training_monitoring()
        
# Example: python3 run.py --device 0 --name my_run
if __name__ == '__main__':
    main()