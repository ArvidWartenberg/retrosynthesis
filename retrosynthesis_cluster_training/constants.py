chars = " ^$#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy"
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
VOCAB_SIZE=len(chars)
TGT_VOCAB_SIZE, SRC_VOCAB_SIZE = VOCAB_SIZE,VOCAB_SIZE
MAX_SEQ_LEN=160
PAD_IDX, BOS_IDX, EOS_IDX = 0,1,2

default_args= {"N_ENCODERS": [6, int],
                    "N_DECODERS": [6, int],
                    "EMBEDDING_SIZE": [512, int],
                    "N_HEADS": [8, int],
                    "DIM_FF": [512, int],
                    "DROPOUT": [0.1, float],
                    "LR": [1e-4, float],
                    "EPOCHS": [200, int],
                    "BATCH_SIZE": [32, int],
                    "MAX_SEQ_LEN": [MAX_SEQ_LEN, int],
                    "device": ["1", str]}

