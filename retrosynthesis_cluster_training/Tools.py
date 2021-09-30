# greedy_decode will be used in a later stage to test our model
def greedy_decode(model, src, src_mask, max_len, start_symbol, DEVICE):
    src = src.to(DEVICE).unsqueeze(-1)
    src_mask = src_mask.to(DEVICE)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        
        out = model.decode(ys, memory, tgt_mask)
        probs = model.generator(out)
        next_word = torch.argmax(probs, axis=2)[-1,:].squeeze()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys
