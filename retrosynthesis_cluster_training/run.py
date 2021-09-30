from modules import *
from training import *
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
            parser.add_argument("--"+key, help=key, default=default_args[key][0], type=default_args[key][1])
        
        parser.add_argument("--checkpoint", help="Load model checkpoint", required=False)
        parser.add_argument("--name", help="Name run", required=False)
        
        args = parser.parse_args()

        return args

def main():
    
    args = parseArguments(default_args=default_args).__dict__
    torch.manual_seed(1337)
    
    print("RUN PARAMS: ")
    for a in args:
        print(str(a) + ": " + str(args[a]))
        
    global DEVICE
    DEVICE = torch.device("cuda:" + args["device"] if torch.cuda.is_available() else 'cpu')
    torch.torch.cuda.set_device(DEVICE)
    print("Using device " + str(torch.torch.cuda.current_device()) + "/" + str(torch.cuda.device_count())
          +", name: " + str(torch.cuda.get_device_name(0)))
    
    
    file = open("/home/arvid/data/USTPO_paper_5x/USTPO_5x_parsed.pickle",'rb')
    data = pickle.load(file)
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

        
    chemFormer = Seq2SeqTransformer(num_encoder_layers=args["N_ENCODERS"],
                                    num_decoder_layers=args["N_DECODERS"],
                                    emb_size=args["EMBEDDING_SIZE"],
                                    nhead=args["N_HEADS"],
                                    src_vocab_size=SRC_VOCAB_SIZE,
                                    tgt_vocab_size=TGT_VOCAB_SIZE,
                                    dim_feedforward=args["DIM_FF"],
                                    dropout=args["DROPOUT"],
                                    DEVICE=DEVICE).to(DEVICE)
    
    if args["checkpoint"] is not None:
        models_dir = "/home/arvid/models/transformers/"
        chemFormer.load_state_dict(torch.load(models_dir + args["checkpoint"] + "/weights"))
        print("Successfully loaded checkpoint: " + models_dir + args["checkpoint"] + "/weights")   
 

    optimizer = torch.optim.Adam(chemFormer.parameters(), lr=args["LR"], betas=(0.9, 0.98), eps=1e-9)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    training_monitoring(model=chemFormer,
                        optimizer=optimizer,
                        n_epochs=args["EPOCHS"],
                        dataloaders=dataloaders,
                        loss_fn=loss_fn,
                        DEVICE=DEVICE,
                        args=args)
        

if __name__ == '__main__':
    main()