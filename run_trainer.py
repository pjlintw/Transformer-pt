"""Train the Transformer with trainer."""
import os
import argparse
import logging
import json
import pathlib
import sys
import pickle
import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset, load_metric

from models.transformer import Transformer
from models.utils import (create_transformer_masks, 
    convert_tensor_to_tokens,
    save_k_exmaple_from_tensor,
    check_k_exmaple_from_tensor, 
    build_vocab,
    pad_sequence)
from models.transformer_blocks import WarmupScheduler


def get_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="")
    # Model
    parser.add_argument('--output_dir', type=str, default='tmp/')
    parser.add_argument('--max_seq_length', type=int, default=20)
    parser.add_argument('--source_vocab', type=str, required=True)
    parser.add_argument('--target_vocab', type=str, required=True)
    
    # Modeling Transformer
    parser.add_argument('--tf_layers', type=int, default=4)
    parser.add_argument('--tf_dims', type=int, default=128) 
    parser.add_argument('--tf_heads', type=int, default=8)
    parser.add_argument('--tf_dropout_rate', type=float, default=0.1)
    parser.add_argument('--tf_shared_emb_layer', type=bool, default=False)
    parser.add_argument('--tf_learning_rate', type=float, default=1e-2)

    # Training
    parser.add_argument('--dataset_script', type=str, required=True)
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_eval', type=bool, default=True)
    parser.add_argument('--do_predict', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--mle_epochs', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=50)
    parser.add_argument('--max_train_samples', type=int)
    parser.add_argument('--max_val_samples', type=int)
    parser.add_argument('--max_test_samples', type=int)

    parser.add_argument('--logging_steps', type=int, default=10)
    # Debugging
    parser.add_argument('--debugging', type=bool, default=False, 
                        help="If debugging. Small batch and sentence length will be used \
                        and the features of models will be printed.")
    return parser.parse_args()


def compute_accuracy(real, pred, pad_idx):
    """Compute accuracy."""
    # Non-pad is True and pad is False
    mask = ~real.eq(pad_idx)
    num_predictions = mask.sum()
    # All padding are not taking into account and
    # non-pad token will be keep as True or False
    corrects = (real == pred)
    num_corrects  = (corrects*mask).sum()
    return  num_corrects / num_predictions
    

def train_generator_MLE(model, 
                        train_dataset,
                        eval_dataset,
                        test_dataset,
                        opt, 
                        logging_steps=50, 
                        epochs=1,
                        tokenizer_dict=None,
                        args=None):
    """Pre-train the generator with MLE."""
    # Prepare for `decode_batch`
    src_pad_idx =tokenizer_dict["src_tok2id"]["[PAD]"]
    SOS_IDX = tokenizer_dict["tgt_tok2id"]["[CLS]"]
    EOS_IDX = tokenizer_dict["tgt_tok2id"]["[SEP]"]
    PAD_IDX = tokenizer_dict["tgt_tok2id"]["[PAD]"]

    # nn.NLLLoss: use log-softmax as input 
    # nn.CrossEntropyLoss: use logit as input
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    for epoch in range(epochs):
        msg = 'epoch %d : ' % (epoch + 1)
        logging.info(msg)
        print(msg)
        total_loss = 0
        total_accuracy = 0
        for step, features_dict in enumerate(train_dataset):
            opt.zero_grad()

            # Encoder input: (batch_size, seq_len)
            src = features_dict["source_ids"]
            # Decoder input and target: (batch_size, seq_len-1)
            tgt_inp = features_dict["target_input"]
            tgt_out = features_dict["target_output"]
            
            enc_padding_mask = features_dict["enc_padding_mask"]
            combined_mask = features_dict["combined_mask"]
            dec_padding_mask =features_dict["dec_padding_mask"]
            
            logits, attn = model(src=src,
                                 tgt=tgt_inp,
                                 training=True,
                                 enc_padding_mask=enc_padding_mask,
                                 look_ahead_mask=combined_mask,
                                 dec_padding_mask=dec_padding_mask,
                                 device=args.device)

            # 2D (batch_size*(seq_len-1), tgt_vocab_size)
            two_d_logits = logits.reshape(-1, logits.shape[-1])
            # print("2D logit shape", two_d_logits.shape)
            # print("tgt out shape", tgt_out.reshape(-1))

            # `two_d_logits` is (batch_size * (seq_seq-1), target_vocab_size)
            # `tgt_out` is (batch_size * (seq_seq-1))
            loss = loss_fn(two_d_logits, tgt_out.reshape(-1))
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(generator.parameters(), 0.5)
            opt.step()

            pred = logits.argmax(-1)
            acc = compute_accuracy(real=tgt_out,
                                   pred=pred,
                                   pad_idx=PAD_IDX)

            if (step) % logging_steps == 0:
                msg = f"Model: generator, Step: {step}, Loss: {loss.item():.2f}, Accuracy: {acc:.2f}"
                logging.info(msg)
                print(msg)

            if (step+1) == args.max_steps:
                msg = f"Reach max steps, Model: generator, Step: {step+1}, Loss: {loss.item():.2f}"
                logging.info(msg)

            # Decode batch
            # pred_sentences = decode_batch(gen_inp, id2tok, unk_idx)
            # print(pred_sentences)

        # Save output file
        # Save model
        if (epoch+1) % 1 == 0:
            ### Saving model ###
            pt_file_tf = get_output_dir(args.output_dir, f"ckpt/epoch-{epoch+1}.pt")
            torch.save(model, pt_file_tf)
            msg = f"Saving model to: {pt_file_tf}"
            logging.info(msg)
            print(msg)

            if args.do_eval:
                msg = f"### Evaluate ###"
                logging.info(msg)
                print(msg)
                avg_loss, avg_acc = model.evaluate(eval_dataset, args, PAD_IDX, compute_accuracy)
                msg = f"Evaluation Epoch: {epoch+1}, avg Loss: {avg_loss:.2f}, avg Accuracy: {avg_acc:.2f}"
                logging.info(msg)
                print(msg)
            
            ### Perform prediction on test set ###
            if args.do_predict:
                output_file = get_output_dir(args.output_dir, f"test.epoch-{epoch+1}.pred")
                wf = open(output_file, "w")
                for step, features_dict in enumerate(test_dataset):
                    src = features_dict["source_ids"]
                    source_tokens = features_dict["source_tokens"]
                    target_tokens = features_dict["target_tokens"]
                    # (batch_size, seq_len)
                    output = model.sample(inp=src,
                                          max_len=20,
                                          sos_idx=SOS_IDX,
                                          eos_idx=EOS_IDX,
                                          src_pad_idx=src_pad_idx,
                                          tgt_pad_idx=PAD_IDX,
                                          device=args.device,
                                          decode_strategy="greedy")
                    # print("output", output)
                    preds = output.cpu().detach().numpy()
                    for sent_ids, src_sent, tgt_sent in zip(preds, source_tokens, target_tokens):
                        sent = " ".join([ tokenizer_dict["tgt_id2tok"][tok_id] for tok_id in sent_ids ])
                        wf.write(f"{sent}\t{src_sent}\t{tgt_sent}\n")
                wf.close()
                msg = f"Saving the translation result of test set to: {output_file}"
                logging.info(msg)
                print(msg)            


def get_output_dir(output_dir, file):
    """Joint path for output directory."""
    return pathlib.Path(output_dir,file)


def build_dirs(output_dir,logger):
    """Build hierarchical directories."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Create folder for output directory: {output_dir}")


def decode_batch(inp, id2tok, unk_idx, batch=True):
    """Convert word indices into words.

    Args:
      inp: (batch_size, seq_max_len)
      id2tok: dictionary.
      unk_idx: int.
      batch: bool.
    """
    batch_example = list()
    if batch:
        for pred_ids in inp:
            batch_example.append([ id2tok[int(w_idx)] if int(w_idx) in id2tok else id2tok[unk_idx] for w_idx in pred_ids ])

    return batch_example


def main():
    # Argument parser
    args = get_args()
    SEED = 13

    cuda_is_available = torch.cuda.is_available()
    args.device = "cuda:0" if cuda_is_available else "cpu"
    torch.manual_seed(SEED) 

    # Create output dir
    output_dir = args.output_dir

    # Logger
    logger = logging.getLogger(__name__)
    build_dirs(output_dir, logger)
    build_dirs(pathlib.Path(output_dir, "ckpt"), logger)
    
    log_file = get_output_dir(output_dir, 'example.log')
    logging.basicConfig(filename=log_file,
                        filemode="w",
                        format="%(asctime)s, %(msecs)d %(name)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S",
                        level=logging.INFO)
    logger.info(args)
    # Saving arguments
    write_path = get_output_dir(output_dir, 'hyparams.txt')
    with open(write_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        logger.info(f"Saving hyperparameters to: {write_path}")

    ########## Load dataset from script. ##########
    # 'wiki-table-questions.py'
    datasets = load_dataset(args.dataset_script)
    logger.info("Loading Datasets")
    # print(datasets)
    

    ### Create vocabulary, token-to-index, index-to-token
    src_vocab, (src_tok2id, src_id2tok) = build_vocab(args.source_vocab, special_tokens=True)
    tgt_vocab, (tgt_tok2id, tgt_id2tok) = build_vocab(args.target_vocab, special_tokens=True)
    
    src_vocab_size = len(src_id2tok)
    tgt_vocab_size = len(tgt_id2tok)

    if args.debugging:
        print("source ids [CLS], [UNK], [SEP], [PAD]", src_tok2id["[CLS]"],src_tok2id["[UNK]"],src_tok2id["[SEP]"],src_tok2id["[PAD]"])
        print("target ids [CLS], [UNK], [SEP], [PAD]", tgt_tok2id["[CLS]"],tgt_tok2id["[UNK]"],tgt_tok2id["[SEP]"],tgt_tok2id["[PAD]"])

    # Create `tokenizer_collector` for trainer function
    tokenizer_collector = dict()
    tokenizer_collector["src_tok2id"] = src_tok2id
    tokenizer_collector["src_id2tok"] = src_id2tok
    tokenizer_collector["tgt_tok2id"] = tgt_tok2id
    tokenizer_collector["tgt_id2tok"] = tgt_id2tok
    

    ########## Load the custom model, tokenizer and config ##########
    def tokenize_fn(example, max_seq_len, ):
        """Add special tokens to input sequence and padd the max lengths."""
        feature_dict = dict()
        # token_ids = list()
    
        source_sent  = example["source"]
        target_sent  = example["target"]
        # sent_len = len(tokens)
        
        ### Add special token and pad ###
        # 2 for [CLS] and [SEP]
        padded_source_sentence_lst = pad_sequence(source_sent, max_seq_len, 2)
        padded_target_sentence_lst = pad_sequence(target_sent, max_seq_len, 2)
        
        # Add padded tokens 
        feature_dict["source_tokens"] = padded_source_sentence_lst
        feature_dict["target_tokens"] = padded_target_sentence_lst

        # Add padded ids
        token_ids = [ src_tok2id[tok] if tok in src_tok2id else src_tok2id["[UNK]"] for tok in padded_source_sentence_lst ]
        feature_dict["source_ids"] = torch.tensor(token_ids)

        condition_token_ids = [ tgt_tok2id[tok] if tok in tgt_tok2id else tgt_tok2id["[UNK]"] for tok in padded_target_sentence_lst ]
        feature_dict["target_ids"] = torch.tensor(condition_token_ids)

        return feature_dict

    # Create tokenize_fn
    tokenize_fn = partial(tokenize_fn, max_seq_len=args.max_seq_length)

    ### Truncate  number of examples ###
    if args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset = train_dataset.map(
            tokenize_fn
        )

    if args.do_eval:
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(args.max_val_samples))
        eval_dataset = eval_dataset.map(
            tokenize_fn
        )

    if args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(args.max_test_samples))
        test_dataset = test_dataset.map(
            tokenize_fn
        )

    
    ### Feature ###
    # `token_ids`, `labels` for training and loss computation
    def generate_batch(data_batch, device):
        """Package feature as mini-batch."""
        features_dict = dict()

        id_batch = list()
        src_token_batch, tgt_token_batch = list(), list()
        src_id_batch, tgt_id_batch = list(), list()
        
        # print("data batch", data_batch)
        for batch_group in data_batch:
            id_batch.append(batch_group["id"])
            src_token_batch.append(batch_group["source_tokens"])
            tgt_token_batch.append(batch_group["target_tokens"])
            src_id_batch.append(batch_group["source_ids"])
            tgt_id_batch.append(batch_group["target_ids"])

        
        batch_src_pt = torch.tensor(src_id_batch)
        batch_tgt_pt = torch.tensor(tgt_id_batch)

        batch_tgt_inp = batch_tgt_pt[:, :-1]
        batch_tgt_out = batch_tgt_pt[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_transformer_masks(batch_src_pt, 
                                                                                     batch_tgt_inp, 
                                                                                     src_pad_idx=src_tok2id["[PAD]"],
                                                                                     tgt_pad_idx=tgt_tok2id["[PAD]"],
                                                                                     device=args.device)
        #  Batch of token list: (batch_size, seq_len)
        features_dict["source_tokens"] = src_token_batch 
        features_dict["target_tokens"] =  tgt_token_batch
        # Batch of source/target id list: (batch_size, seq_len)
        features_dict["source_ids"] = batch_src_pt
        features_dict["target_ids"] = batch_tgt_pt
        # Batch of target id list for decoder: (batch_size, seq_len-1)
        features_dict["target_input"] = batch_tgt_inp
        features_dict["target_output"] = batch_tgt_out
        # Masks
        features_dict["enc_padding_mask"] = enc_padding_mask
        features_dict["combined_mask"] = combined_mask
        features_dict["dec_padding_mask"] = dec_padding_mask

        if device:
            for k in features_dict:
                if torch.is_tensor(features_dict[k]):
                    features_dict[k] = features_dict[k].to(device)
            
        return features_dict

    # Construct model
    model = Transformer(num_layers=args.tf_layers,
                        d_model=args.tf_dims,
                        num_head=args.tf_heads,
                        intermediate_dim=args.tf_dims*4,
                        input_vocab_size=src_vocab_size,
                        target_vocab_size=tgt_vocab_size,
                        src_max_len=args.max_seq_length,
                        tgt_max_len=args.max_seq_length,
                        padding_idx=src_tok2id["[PAD]"],
                        shared_emb_layer=args.tf_shared_emb_layer, # Whether use embeeding layer from encoder
                        rate=args.tf_dropout_rate)
    model.to(args.device)

    #model.apply(init_weights)
    logging.info(model.encoder)
    
    generate_batch_fn = partial(generate_batch, device=args.device)
    ### Fetch dataset iterator
    train_iter = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, collate_fn=generate_batch_fn)
    eval_iter = DataLoader(eval_dataset, batch_size=args.batch_size,
                           shuffle=False, collate_fn=generate_batch_fn)
    test_iter = DataLoader(test_dataset, batch_size=args.batch_size,
                           shuffle=False, collate_fn=generate_batch_fn)

    
    ### train model ###
    print("train the Transformer")
    gen_optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9,0.98), eps=1e-9)
    gen_optimizer = WarmupScheduler(model_size=args.tf_dims,
                                    factor=2,
                                    warmup=4000,
                                    optimizer=gen_optimizer)
    

    train_generator_MLE(model=model,
                        train_dataset=train_iter,
                        eval_dataset=eval_iter,
                        test_dataset=test_iter,
                        opt=gen_optimizer,
                        logging_steps=args.logging_steps,
                        epochs=args.mle_epochs,
                        tokenizer_dict=tokenizer_collector,
                        args=args)
    



if __name__ == "__main__":
    main()
