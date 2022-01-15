import os
import argparse
import os
import argparse
from functools import partial

import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F

from models.utils import build_vocab, pad_sequence

def main():

    source_vocab = "data/de-en/source.vocab"
    target_vocab = "data/de-en/target.vocab"

    ### Create vocabulary, token-to-index, index-to-token
    src_vocab, (src_tok2id, src_id2tok) = build_vocab(source_vocab, special_tokens=True) 
    tgt_vocab, (tgt_tok2id, tgt_id2tok) = build_vocab(target_vocab, special_tokens=True)
    src_vocab_size = len(src_id2tok)
    tgt_vocab_size = len(tgt_id2tok)
    
    # Create `tokenizer_collector` for trainer function
    tokenizer_collector = dict()
    tokenizer_collector["src_tok2id"] = src_tok2id
    tokenizer_collector["src_id2tok"] = src_id2tok
    tokenizer_collector["tgt_tok2id"] = tgt_tok2id
    tokenizer_collector["tgt_id2tok"] = tgt_id2tok

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ckpt = "results/word/L-4_D-256_H-8/ckpt/epoch-25.pt"
    model = torch.load(ckpt)
    model.eval()
    model.to(device)

    inp = ["diese funktion ersetzt die funktion pg_lounlink ( ) .", "dies wird nur zu problemen führen .", "es gibt 4 möglichkeiten , eine dba datenbank zu öffnen :",
          "siehe auch odbc_commit ( ) und odbc_rollback ( ) .", "diese funktion wurde in 4.0 hinzugefügt ."]
    print(inp)
    # inp = ["gutes design des datenbankschemas , und die applikation wird mit ihren größten befürchtungen fertig .", 
    #       "gutes design des datenbankschemas , und die applikation wird mit ihren größten befürchtungen fertig ."]
    inp = [ pad_sequence(e.split(), 20, 2) for e in inp]
    to_id_fn = lambda x : [ src_tok2id[tok] for tok in x]
    inp_id = [ to_id_fn(sent) for sent in inp]
    inp_id = torch.tensor(inp_id).to(device)
    
    out = model.beam_decode(inp_id,
                            k=4,
                            max_len=20,
                            sos_idx=tgt_tok2id["[CLS]"],
                            eos_idx=tgt_tok2id["[SEP]"],
                            src_pad_idx=src_tok2id["[PAD]"],
                            tgt_pad_idx=tgt_tok2id["[PAD]"],
                            device=device)
      
    decoded_out = out[0].cpu().detach().numpy()
    log = out[1]
    # print(log)
    for sent_lst in decoded_out:
        print(" ".join([ tgt_id2tok[tok] for tok in sent_lst]))
    
    # print("batch k")
    # batch_k = out[2].cpu().detach().numpy()
    # for sent_lst in batch_k:
        # print(" ".join([ tgt_id2tok[tok] for tok in sent_lst]))


if __name__ == "__main__":
    main()
