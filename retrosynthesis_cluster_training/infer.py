from modules import *
from Trainer import *
from Transformer import *
from Reaction_dataset import *
from Tools import *
from constants import *
from Inferrer import *

#os.system("nvidia-smi")

def parseArguments():
        # Create argument parser
        parser = argparse.ArgumentParser()
        
        parser.add_argument("--model", help="date/time model indicator", required=False)
        parser.add_argument("--algorithm", help="search algorithm", default="greedy", required=False)
        parser.add_argument("--device", help="choose device", default="0", required=False)
        parser.add_argument("--weights", help="Choose checkpoint", default="best_acc", required=False)
        parser.add_argument("--dataset", help="choose dataset", default="non-augmented", required=False)
        parser.add_argument("--n_infer", help="num inference points", default="all", required=False)
        parser.add_argument("--k", help="top k to be reported", default=5, type=int, required=False)
        parser.add_argument("--beam_size", help="beam size", default=5, type=int, required=False)
        # TODO add dataset argument
        
        args = parser.parse_args().__dict__
        args["device"] = "cuda:" + args["device"]

        return args

def main():
    
    os.chdir("..")
    home_dir = os.getcwd()
    os.chdir("retrosynthesis_cluster_training/")
    
    # Parse specified inference settings
    settings = parseArguments()
    model_dir = home_dir + "/models/chemformers/" + settings["model"] + "/"
    
    # Load original experiment args
    args = pickle.load(open(model_dir + "params.pickle", "rb"))
    args["home_dir"] = home_dir

    DEVICE = torch.device(settings["device"] if torch.cuda.is_available() else 'cpu')
    torch.torch.cuda.set_device(DEVICE)
    print("Using device " + str(torch.torch.cuda.current_device()) + "/" + str(torch.cuda.device_count())
          +", name: " + str(torch.cuda.get_device_name(0)))
    
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load specified dataset
    if settings["dataset"] == "non-augmented": file = open(home_dir + "/data/USTPO_not_augmented/data.pickle","rb")
    else: file = open(home_dir + "/data/USTPO_paper_5x/USTPO_5x_parsed.pickle",'rb')
    data = pickle.load(file)
    
    datasets = {}
    datasets["eval"] = ReactionDataset(data=data,
                                       split="eval",
                                       args=args)
    
    
    # Init model and load checkpoint
    chemFormer = Seq2SeqTransformer(num_encoder_layers=args["N_ENCODERS"],
                                    num_decoder_layers=args["N_DECODERS"],
                                    emb_size=args["EMBEDDING_SIZE"],
                                    nhead=args["N_HEADS"],
                                    src_vocab_size=SRC_VOCAB_SIZE,
                                    tgt_vocab_size=TGT_VOCAB_SIZE,
                                    dim_feedforward=args["DIM_FF"],
                                    dropout=args["DROPOUT"],
                                    DEVICE=DEVICE).to(DEVICE)
     
    chemFormer.eval()
    
    chemFormer.load_state_dict(torch.load(model_dir + settings["weights"]))
    print("Successfully loaded checkpoint: " + model_dir + settings["weights"])   
    args["model_dir"] = model_dir
    
    inferrer = Inferrer(model=chemFormer,
                        datasets=datasets,
                        settings=settings,
                        args=args)
    
    inferrer.infer_and_write()

# Example: python3 infer.py --model 2021-10-12/13:53:03 --weights weights_epoch_900 --n_infer all --device=3 --algorithm greedy
if __name__ == '__main__':
    main()