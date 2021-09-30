from constants import *
from modules import *
from Transformer import *


# Here, we build our training & monitoring methods

# Code for running one split (name of method is faulty)
def run_one_epoch(model, optimizer, loss_fn, dataloaders, split, DEVICE):
    
    # put model in right split state
    model.eval()
    if split=="train": model.train()
    
    # Iterate over batches in split
    losses = 0
    for batch in dataloaders[split]:
        
        # Prepare inputs, outputs and masks
        src = batch["ps"].permute(1,0)
        tgt = batch["rs"].permute(1,0)

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        
        tgt_input = tgt[:-1, :]
        tgt_out = tgt[1:, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, DEVICE=DEVICE)
        
        # Let model compute logits
        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        
        # compute loss and backprop
        optimizer.zero_grad()
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
    
             
    return losses / len(dataloaders[split])

# This method manages training by iterating over epochs, saving some data to a tensorboard along the way
def training_monitoring(model, optimizer, loss_fn, n_epochs, dataloaders, DEVICE, args):
    
    model_dir = os.path.expanduser("~/models/transformers/")
    time()
    from datetime import datetime
    date_time = datetime.fromtimestamp(time())
    if args["name"] is None:
        model_str = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    else: 
        model_str = args["name"]
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
        
        print("Beginning of epoch %i"%(epoch), end="")
        
        for split in ["train", "eval"]:
            t1 = time()
            print("..." + split, end="")
            epoch_loss = run_one_epoch(model=model,
                                       optimizer=optimizer,
                                       loss_fn=loss_fn,
                                       dataloaders=dataloaders,
                                       split=split,
                                       DEVICE=DEVICE)
            
            losses[split] = epoch_loss
            running_losses[split].append(epoch_loss)
            writer.add_scalar('Loss/'+split, epoch_loss, epoch+1)
            
            print("...finished in %.2f s! "%(time()-t1), end="")
            if split == 'eval' and epoch_loss < best_loss:
                print("Best epoch...saving weights... ", end="")
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
                torch.save(model.state_dict(),model_dir + model_str + "/weights")
        print()
        print("Epoch %i losses are; Train: %.4f // Eval: %.4f."%(epoch,losses["train"],losses["eval"]))