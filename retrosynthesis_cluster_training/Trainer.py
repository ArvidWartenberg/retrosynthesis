from constants import *
from modules import *
from Transformer import *


class Trainer:
    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 loss_fn,
                 dataloaders,
                 args):
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.dataloaders = dataloaders
        self.args = args
        self.DEVICE = args["device"]
        

    # Here, we build our training & monitoring methods

    # Code for running one split (name of method is faulty)
    def run_one_split(self, split):

        # put model in right split state
        self.model.eval()
        if split=="train": self.model.train()
        

        # Iterate over batches in split
        losses = 0
        with torch.set_grad_enabled(split == "train"):
            for batch in self.dataloaders[split]:
                #torch.tensor().to(dtype=torch.int64)
                # Prepare inputs, outputs and masks
                src = batch["ps"].to(dtype=torch.int64).permute(1,0)
                tgt = batch["rs"].to(dtype=torch.int64).permute(1,0)

                src = src.to(self.DEVICE)
                tgt = tgt.to(self.DEVICE)

                tgt_input = tgt[:-1, :]
                tgt_out = tgt[1:, :]

                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src=src,
                                                                                     tgt=tgt_input,
                                                                                     DEVICE=self.DEVICE)
                if split=="train": self.optimizer.zero_grad()

                # Let model compute logits
                logits = self.model(src=src,
                                    trg=tgt_input,
                                    src_mask=src_mask,
                                    tgt_mask=tgt_mask,
                                    src_padding_mask=src_padding_mask,
                                    tgt_padding_mask=tgt_padding_mask,
                                    memory_key_padding_mask=src_padding_mask)

                # compute loss and backprop
                loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

                if split=="train": 
                    loss.backward()
                    self.optimizer.step()
                  
                losses += loss.item()
                
                
        return losses / len(self.dataloaders[split])

    # This method manages training by iterating over epochs, saving some data to a tensorboard along the way
    def training_monitoring(self):

        folder_dir = os.path.expanduser("~/models/chemformers/") 
        day_dir = datetime.now().strftime("%Y-%m-%d")
        if day_dir not in os.listdir(folder_dir):
            os.mkdir(folder_dir + day_dir)

        total_dir_path = folder_dir+day_dir+"/"
        
        date_time = datetime.fromtimestamp(time())
        if self.args["name"] is None:
            model_str = datetime.now().strftime("%H:%M:%S")
        else: 
            model_str = self.args["name"]
        os.mkdir(total_dir_path + model_str)
        os.mkdir(total_dir_path + model_str + "/logdir")
        print("tensorboard --logdir " + total_dir_path + " --port=6006")
        print("tensorboard --logdir " + total_dir_path + model_str + "/logdir/ --port=6006")



        with open(total_dir_path + model_str + '/params.pickle', 'wb') as handle:
            pickle.dump(self.args,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL)


        best_loss = 999999999
        running_losses = {'train': [], 'eval': []}    

        # Loop
        writer = SummaryWriter(total_dir_path + model_str + "/logdir/")

        for epoch in range(self.args["EPOCHS"]):
            losses = {}
            print()

            print("Beginning of epoch %i"%(epoch), end="")

            for split in ["train", "eval"]:
                t1 = time()
                print("..." + split, end="")
                
                epoch_loss = self.run_one_split(split=split)
                losses[split] = epoch_loss
                running_losses[split].append(epoch_loss)
                
                writer.add_scalar('Loss/'+split, epoch_loss, epoch+1)
                
                print("...finished in %.2f s! "%(time()-t1), end="")
                if (split == 'eval' and epoch_loss < best_loss) or (self.args["debug"] is not None and epoch_loss < best_loss):
                    print("Best epoch...saving weights... ", end="")
                    best_loss = epoch_loss
                    best_model_wts = self.model.state_dict()
                    torch.save(self.model.state_dict(), total_dir_path + model_str + "/weights")
                
                torch.save(self.model.state_dict(), total_dir_path + model_str + "/weights_last_epoch")
            
            self.scheduler.step()
            writer.add_scalar('lr/', self.scheduler.get_lr()[0], epoch+1)
            print()
            print("Epoch %i losses are; Train: %.4f // Eval: %.4f."%(epoch,losses["train"],losses["eval"]))