from modules import *
from constants import *
from Transformer import *
from Tools import *
from Reaction_dataset import *

class Inferrer:
    def __init__(self,
                 model,
                 datasets,
                 settings,
                 args,
                 split="eval"):
        
        self.model = model
        self.datasets = datasets
        self.settings = settings
        self.split = split
        self.DEVICE = settings["device"]
        self.args = args
        
        # Gen set for items to infer
        data_ixs = np.arange(0,len(datasets[split]))
        #np.random.shuffle(data_ixs)
        self.Q = set()
        if self.settings["n_infer"] == "all": 
            for e in data_ixs: self.Q.add(e)
        else: 
            for e in data_ixs[0:int(self.settings["n_infer"])]: self.Q.add(e)
                
    
    def greedy_decode(self):
        inferred = {}

        # Init src and tgt tensors, free_ixs_tracker & element tracker
        capacity = 32
        src = torch.zeros([MAX_SEQ_LEN, capacity]).to(self.DEVICE)
        tgt = torch.zeros([MAX_SEQ_LEN, capacity]).to(self.DEVICE)
        element_tracker = {}
        free_ixs_tracker = set()
        for e in range(capacity): free_ixs_tracker.add(e)
        j = 0
        # Infer while PQ nonempty
        
        pbar = tqdm(total=len(self.Q))
        while self.Q != set() or len(free_ixs_tracker) != capacity:
            j += 1
            # Track elements and fill a batch
            while free_ixs_tracker != set() and self.Q != set():
                # Pop data item for a free ix in tensor, and record where it is put & ix where to get next word
                free_ix = free_ixs_tracker.pop()
                data_ix = self.Q.pop()
                element_tracker[free_ix] = [data_ix, 0]
                data = torch.tensor(self.datasets[self.split].__getitem__(data_ix)["ps"]).to(self.DEVICE)
                src[:,free_ix] = data
                tgt[:,free_ix] = PAD_IDX
                tgt[0,free_ix] = BOS_IDX

            # Calculate all masks
            src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask = create_mask(src=src,
                                                                                 tgt=tgt,
                                                                                 DEVICE=self.DEVICE)
            # Compute embedding
            memory = self.model.encode(src=src,
                                       src_mask=src_mask,
                                       src_key_padding_mask=src_key_padding_mask)

            # Decode until a sequence finishes
            while True:

                # Calculate all masks again
                src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask = create_mask(src=src,
                                                                                             tgt=tgt,
                                                                                             DEVICE=self.DEVICE)
                logits = self.model.decode(tgt=tgt,
                                           memory=memory,
                                           tgt_mask=tgt_mask,
                                           memory_key_padding_mask=src_key_padding_mask,
                                           tgt_key_padding_mask=tgt_key_padding_mask)
                probs = self.model.generator(logits)

                # Iterate over tracked seqs
                should_break = False
                for k,v in element_tracker.items():
                    if element_tracker[k][0] in inferred: continue
                    # Find next word for each seq, record that seq expands
                    if element_tracker[k][1] == MAX_SEQ_LEN - 1:
                        inferred[element_tracker[k][0]] = None
                        free_ixs_tracker.add(k)
                        should_break = True

                    else:
                        next_word = torch.argmax(probs[element_tracker[k][1], k,:])

                        element_tracker[k][1] += 1

                        tgt[element_tracker[k][1], k] = next_word

                        # If seq ends, need to save result & remove from capacity
                        if next_word == EOS_IDX:
                            seq = tgt[:,k]
                            seq = tokens_to_smiles(seq[torch.where(seq != PAD_IDX)][1:-1].cpu().numpy())
                            gt = self.datasets[self.split].__getitem__(element_tracker[k][0])["rs"]
                            gt = tokens_to_smiles(gt[np.where(gt != PAD_IDX)][1:-1])

                            inferred[element_tracker[k][0]] = {"pred": [{"p": 1, "output": seq}], "tgt": gt}
                            free_ixs_tracker.add(k)
                            should_break = True
                            pbar.update(1)

                if should_break == True: break
        pbar.close()
        return inferred
    
    def beam_search_one(self, data_ix, beam_size=10, top_k=5, split="eval"):
        tiebreaker = count()

        # Get item
        data = self.datasets[split].__getitem__(data_ix)

        src = data["ps"]
        src = torch.tensor(src[np.where(src != PAD_IDX)]).to(self.DEVICE).unsqueeze(-1)

        gt = data["rs"][1:]
        tgt_out = gt[np.where(gt != PAD_IDX)]
        tgt_out = tokens_to_smiles(tgt_out[:-1])

        # Compute embedding
        memory = self.model.encode(src=src,
                                   src_mask=None,
                                   src_key_padding_mask=None).tile([1,beam_size,1])

        # Prepare prompting target tensor and add to heap

        """
        We need multiple data structures for tracking the beams. If we only had one, 
        problems might arise with shorter non-terminated seqs are kept instead
        of a longer seq, which is incorrect.
        """

        # init heap with one compute prompt
        top_k_heap = []
        heappush(top_k_heap, (0.00, next(tiebreaker), {"seq": np.array([BOS_IDX],dtype="float64"), "len": 1, "compute": True}))

        # run until all seqs terminate
        while True:

            # place items in compute/finished
            compute_items = []
            finished_items = []
            while top_k_heap != []:
                ll, _, info = heappop(top_k_heap)
                if info["compute"]: compute_items.append((ll, info))
                else: finished_items.append((ll, info))

            # termination condition
            if len(finished_items) == beam_size or compute_items == []: break

            # Produce tgt in tensor for this iter
            tgt_compute = PAD_IDX*torch.ones([MAX_SEQ_LEN,len(compute_items)]).to(self.DEVICE)
            for i in range(len(compute_items)):
                compute_item = compute_items[i][1]
                compute_seq = torch.tensor(compute_item["seq"]).to(self.DEVICE)
                tgt_compute[0:compute_item["len"],i] = compute_seq


            # Prepare memory
            sliced_memory = memory[:,0:len(compute_items),:]

            # Gen masks
            src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask = create_mask(src=sliced_memory,
                                                                                         tgt=tgt_compute,
                                                                                         DEVICE=self.DEVICE)

            # run through decoder
            output_embeddings = self.model.decode(tgt=tgt_compute,
                                                  memory=sliced_memory,
                                                  tgt_mask=tgt_mask,
                                                  memory_key_padding_mask=None,
                                                  tgt_key_padding_mask=tgt_key_padding_mask)
            logits = self.model.generator(output_embeddings)
            softmaxed_logits = F.softmax(logits, dim=-1)
            log_likelihoods = torch.log10(softmaxed_logits)



            # Place all finished seqs in top_k_heap
            for ll, info in finished_items: heappush(top_k_heap, (ll, next(tiebreaker), info))


            # Track top k seqs and update top_k_heap
            for i in range(len(compute_items)):
                ll, info = compute_items[i]

                for word in range(VOCAB_SIZE):
                    word_ll = log_likelihoods[info["len"]-1, i, word].item()
                    new_len = info["len"] + 1
                    old_seq = info["seq"]
                    compute = not word==EOS_IDX


                    if new_len > MAX_SEQ_LEN: continue

                    # Fill heap with anything if not full
                    if len(top_k_heap) < beam_size:
                        new_seq = np.concatenate((old_seq,np.array([word])))

                        heappush(top_k_heap, (ll+word_ll,
                                              next(tiebreaker),
                                              {"seq": new_seq,
                                               "len": new_len,
                                               "compute": compute}))
                    # Else we pop and compare such that heap does not become overfilled
                    else:
                        ll_vs, _, info_vs = heappop(top_k_heap)
                        if ll_vs < ll+word_ll:
                            new_seq = np.concatenate((old_seq,np.array([word])))
                            heappush(top_k_heap, (ll+word_ll,
                                                  next(tiebreaker),
                                                  {"seq": new_seq,
                                                   "len": new_len,
                                                   "compute": compute}))
                        else: heappush(top_k_heap, (ll_vs, next(tiebreaker), info_vs))


        lls =  np.array([e[0] for e in finished_items])
        argsort = torch.argsort(torch.tensor(lls)).numpy()[::-1]
        probabilities = F.softmax(torch.tensor(lls[argsort[0:top_k]]), dim=0)

        top_k_list = []
        for i in range(min(len(argsort), top_k)):
            ix = argsort[i]
            seq = finished_items[ix][1]["seq"]
            smiles = tokens_to_smiles(seq[1:-1])
            top_k_list.append({"p": probabilities[i].item(), "output": smiles})

        return (tgt_out, top_k_list)

    def beam_search(self, beam_size=10, top_k=5, split="eval"):
        inferred = {}
        pbar = tqdm(total=len(self.Q))
        while self.Q != set():
            data_ix = self.Q.pop()
            tgt_out, top_k_list = self.beam_search_one(data_ix, beam_size=beam_size, top_k=top_k, split=split)
            inferred[data_ix] = {"pred": top_k_list, "tgt": tgt_out}
            pbar.update(1)
        pbar.close()
        return inferred
            
            
    def calculate_accuracies(self, inferred, top_k):
        top_k_TPs = np.zeros(top_k)
        top_k_tots = np.zeros(top_k)
        for data_ix, item in inferred.items():
            if item is None: continue
            tgt_out, predictions = item["tgt"], item["pred"]
            correcly_predicted = False
            correct_at_top_k = np.zeros(top_k)
            correct_at_top_k = np.zeros(top_k)
            for i in range(top_k):
                prediction = predictions[i]
                if smiles_matching(out_smiles = prediction["output"], tgt_smiles = tgt_out): correcly_predicted = True
                correct_at_top_k[i] = int(correcly_predicted)

            top_k_tots += np.ones(top_k)
            top_k_TPs += correct_at_top_k
        return top_k_TPs/top_k_tots
    
    def save_results(self, inferred, accuracies, top_k):
        results_dir = self.args["model_dir"] + "results/"
        if not os.path.isdir(results_dir): os.mkdir(results_dir)
        
        test_results_name = results_dir + datetime.now().strftime("%H:%M:%S")+".pickle"
        
        with open(test_results_name, 'wb') as handle:
            pickle.dump({"settings": self.settings,
                        "args": self.args,
                        "accuracies": accuracies,
                        "inferred": inferred}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        result_string = test_results_name.split("formers/")[1].split(".pick")[0]
        result_string += " "*(45-len(result_string)) + str(self.settings["n_infer"]) + " pts"
        result_string = result_string + " "*(60-len(result_string))
        
        if self.settings["algorithm"] == "beam":
            for i in range(top_k):
                if i in [0, 2, 4, 9]:
                    result_string += "Top-%i: %.1f  "%(i+1,accuracies[i]*100)
        else: result_string += "Greedy: %.1f  "%(accuracies[0]*100)
            
        
        with open(self.args["home_dir"] + '/RESULTS.txt', "a") as fhandle:
            fhandle.write(f'{result_string}\n')
        print("Saved inferred results, predictions & targets to " + test_results_name)
        print("Wrote to RESULTS.txt in home dir.")
        print()
        
    def infer_and_write(self):
        print()
        print()
        print()
        print("Recieved MODEL, key hyperparameters are:")
        print("N_ENCODERS: %i"%self.args["N_ENCODERS"])
        print("N_DECODERS: %i"%self.args["N_DECODERS"])
        print("DIM_FF: %i"%self.args["DIM_FF"])
        print("N_HEADS: %i"%self.args["N_HEADS"])
        print("EMBEDDING_SIZE: %i"%self.args["EMBEDDING_SIZE"])
        
        if self.settings["algorithm"] == "greedy":
            top_k = 1
            print()
            print()
            print("Running GREEDY DECODE inference on " + self.settings["n_infer"] 
                  + " examples from the " + self.settings["dataset"] + " DATASET")
            print("Using model " + self.settings["model"] + " with weights: " + self.settings["weights"])
            inferred =  self.greedy_decode()
            print("FINISHED & RECIEVED INFERENCE RESULTS")
            print()
            print("CALCULATING GREEDY ACCURACY")
            
        
        if self.settings["algorithm"] == "beam":
            top_k = self.settings["k"]
            beam_size = self.settings["beam_size"]
            print()
            print()
            print("Running BEAM SEARCH inference on " + self.settings["n_infer"] 
                  + " examples from the " + self.settings["dataset"] + " DATASET")
            print("Using model " + self.settings["model"] + " with weights: " + self.settings["weights"])
            inferred =  self.beam_search(beam_size=10, top_k=5, split=self.split)
            print("FINISHED & RECIEVED INFERENCE RESULTS")
            print()
            print("CALCULATING top-k ACCURACIES")
        
        accuracies = self.calculate_accuracies(inferred, top_k)
        print()
        if self.settings["algorithm"] == "beam":
            for i in range(top_k):
                print("Top-%i Accuracy: %.1f"%(i+1,accuracies[i]*100))    
        else: print("GREEDY Accuracy: %.1f"%(accuracies[0]*100))
        print()
        print()
        print()
        self.save_results(inferred, accuracies, top_k)
        
        