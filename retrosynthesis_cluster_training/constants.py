chars = " ^$#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy"
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
VOCAB_SIZE=len(chars)
TGT_VOCAB_SIZE, SRC_VOCAB_SIZE = VOCAB_SIZE,VOCAB_SIZE
MAX_SEQ_LEN=162
PAD_IDX, BOS_IDX, EOS_IDX = 0,1,2

default_args= {"N_ENCODERS": [3, int],
                    "N_DECODERS": [3, int],
                    "EMBEDDING_SIZE": [512, int],
                    "N_HEADS": [8, int],
                    "DIM_FF": [512, int],
                    "DROPOUT": [1e-1, float],
                    "LR": [1e-1, float],
                    "EPOCHS": [1200, int],
                    "BATCH_SIZE": [64, int],
                    "MAX_SEQ_LEN": [MAX_SEQ_LEN, int],
                    "WARMUP_STEPS": [40, int],
                    "device": ["1", str]}

