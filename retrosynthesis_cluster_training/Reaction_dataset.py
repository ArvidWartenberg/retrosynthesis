from modules import *
from constants import *


# ReactionDataset class
class ReactionDataset(Dataset):

    def __init__(self, 
                 data, 
                 split,
                 args,
                 maxlen=MAX_SEQ_LEN,
                 rep=" ^$#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy"):
        
        self.split = split
        self.data = data[self.split]
        self.maxlen = maxlen
        self.rep = rep
        self.args = args
        self.char_to_ix = { ch:i for i,ch in enumerate(rep) }
        self.ix_to_char = { i:ch for i,ch in enumerate(rep) }
        # Add augmentation methods here later
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        rs_smiles = self.data[index]["rs"]
        ps_smiles = self.data[index]["ps"]
        if rs_smiles[0] == ".":
            rs_smiles=rs_smiles[1:]
        if ps_smiles[0] == ".":
            ps_smiles=ps_smiles[1:]
        
        
        rs_smiles = self.ix_to_char[BOS_IDX] + rs_smiles + self.ix_to_char[EOS_IDX] + (self.maxlen-len(rs_smiles)-2)*" "
        ps_smiles = ps_smiles + (self.maxlen-len(ps_smiles))*" "
        
        # Augment smiles here for train
        
        rs = np.array([self.char_to_ix[char] for char in rs_smiles])
        ps = np.array([self.char_to_ix[char] for char in ps_smiles])
        
        return {
            "rs": rs,
            "ps": ps
        }
