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
        accuracy = 0
        top_3_accuracy = 0
        top_5_accuracy = 0
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
                flat_targets = tgt_out.reshape(-1)
                loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), flat_targets)

                if split=="train": 
                    loss.backward()
                    self.optimizer.step()
                
                #pdb.set_trace()
                top_3_accuracy += torch.sum(torch.any(((torch.flip(torch.argsort(logits,dim=-1),dims=[-1])[:,:,0:3] == tgt_out.unsqueeze(-1).tile([1,1,3]))*((tgt_out != PAD_IDX)).unsqueeze(-1).tile([1,1,3])).to(torch.int32),axis=-1))/torch.where(flat_targets != PAD_IDX)[0].shape[0]
                
                top_5_accuracy += torch.sum(torch.any(((torch.flip(torch.argsort(logits,dim=-1),dims=[-1])[:,:,0:5] == tgt_out.unsqueeze(-1).tile([1,1,5]))*((tgt_out != PAD_IDX)).unsqueeze(-1).tile([1,1,5])).to(torch.int32),axis=-1))/torch.where(flat_targets != PAD_IDX)[0].shape[0]
                
                non_pad_ixs = torch.where(flat_targets != PAD_IDX)
                predicted_chars = torch.argmax(logits.reshape(-1, logits.shape[-1]),axis=1)
                
                flat_targets = flat_targets[non_pad_ixs]
                predicted_chars = predicted_chars[non_pad_ixs]
                accuracy += torch.sum(predicted_chars == flat_targets)/flat_targets.shape[0]
                
                losses += loss.item()
                
                
        return losses/len(self.dataloaders[split]), accuracy/len(self.dataloaders[split]), top_3_accuracy/len(self.dataloaders[split]), top_5_accuracy/len(self.dataloaders[split])

    # This method manages training by iterating over epochs, saving some data to a tensorboard along the way
    def training_monitoring(self):
        
        folder_dir = os.path.expanduser(self.args["home_dir"] + "/models/chemformers/") 
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
        print("tensorboard --logdir " + total_dir_path + " --port="+self.args["port"])
        print("tensorboard --logdir " + total_dir_path + model_str + "/logdir/ --port="+self.args["port"])



        with open(total_dir_path + model_str + '/params.pickle', 'wb') as handle:
            pickle.dump(self.args,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL)


        best_loss = 999999999
        best_acc = 0
        running_losses = {'train': [], 'eval': []}    

        # Loop
        writer = SummaryWriter(total_dir_path + model_str + "/logdir/")

        for epoch in range(self.args["EPOCHS"]):
            losses = {}
            accuracies = {}
            top_3_accuracies = {}
            top_5_accuracies = {}
            print()

            print("Beginning of epoch %i"%(epoch), end="")

            for split in ["train", "eval"]:
                t1 = time()
                print("..." + split, end="")
                
                epoch_loss, accuracy, top_3_accuracy, top_5_accuracy = self.run_one_split(split=split)
                print("...finished in %.2f s! "%(time()-t1), end="")
                
                losses[split] = epoch_loss
                accuracies[split] = accuracy
                top_3_accuracies[split] = top_3_accuracy
                top_5_accuracies[split] = top_5_accuracy
                running_losses[split].append(epoch_loss)
                
                if (split == 'eval' and epoch_loss < best_loss) or (self.args["debug"] is not None and epoch_loss < best_loss):
                    best_loss = epoch_loss
                    best_model_wts = self.model.state_dict()
                    torch.save(self.model.state_dict(), total_dir_path + model_str + "/best_loss")
                
                if epoch%50==0 and split == "eval":
                    torch.save(self.model.state_dict(), total_dir_path + model_str + "/epoch_%i"%epoch)
                    
                if accuracy > best_acc and split == "eval":
                    best_acc = accuracy
                    torch.save(self.model.state_dict(), total_dir_path + model_str + "/best_acc")
                    
                if split == "train": torch.save(self.model.state_dict(), total_dir_path + model_str + "/latest")
            
            writer.add_scalar("Training/train loss", losses['train'], epoch+1)
            writer.add_scalar("Training/eval loss", losses['eval'], epoch+1)
            writer.add_scalar("Training/_lr", self.scheduler.get_lr()[0], epoch+1)
            #writer.add_scalar("Metrics/top-1 train", accuracies['train'], epoch+1)
            writer.add_scalar("Metrics/top-1 eval", accuracies['eval'], epoch+1)
            #writer.add_scalar("Metrics/top-3 train", top_3_accuracies['train'], epoch+1)
            writer.add_scalar("Metrics/top-3 eval", top_3_accuracies['eval'], epoch+1)
            #writer.add_scalar("Metrics/top-5 train", top_5_accuracies['train'], epoch+1)
            writer.add_scalar("Metrics/top-5 eval", top_5_accuracies['eval'], epoch+1)
            
            print()
            print("Epoch %i losses are; Train: %.4f // Eval: %.4f."%(epoch,losses["train"],losses["eval"]))
            
            self.scheduler.step()