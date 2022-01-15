"""Implementation of Transformer."""
from .utils import create_transformer_masks
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, 
                 num_layers,
                 d_model, num_head,
                 intermediate_dim,
                 input_vocab_size,
                 target_vocab_size,
                 src_max_len,
                 tgt_max_len,
                 padding_idx,
                 shared_emb_layer=None, # Whether use embeeding layer from encoder
                 rate=0.1):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.pad_idx = padding_idx

        # (vocab_size, emb_dim)
        self.embedding_layer = nn.Embedding(input_vocab_size, d_model)
    
        self.encoder = TransformerEncoder(num_layers, d_model, num_head,
                                          intermediate_dim,
                                          input_vocab_size,
                                          src_max_len, 
                                          rate)

        if shared_emb_layer is True:
            self.shared_emb_layer = self.embedding_layer
        else:
            self.shared_emb_layer = shared_emb_layer
        # print(self.shared_emb_layer)
        self.decoder = TransformerDecoder(num_layers, d_model, num_head,
                                         intermediate_dim,
                                         target_vocab_size,
                                         tgt_max_len,
                                         self.shared_emb_layer,  # share embedding
                                         rate)
        self.final_layer = nn.Linear(d_model, target_vocab_size)
    
        
    def forward(self, src, tgt, training, enc_padding_mask,
                look_ahead_mask, dec_padding_mask, cuda):
        """Forward propagate for transformer.
        
        Args:
          src: (batch_size, src_max_len)
        """
        # Mapping
        src = self.embedding_layer(src)
        src = torch.mul(src, (self.d_model**(1/2)))

        # (batch_size, inp_seq_len, d_model)
        enc_out = self.encoder(src, training, enc_padding_mask, gpu=cuda) #.cuda()

        # print("type of decoder input", type(tgt))
        # print("decoder input", tgt)
        # (batch_size, tgt_seq_len, d_model)
        dec_output, dec_attn = self.decoder(x=tgt, 
                                            enc_output=enc_out, 
                                            training=training, 
                                            look_ahead_mask=look_ahead_mask,
                                            padding_mask=dec_padding_mask,
                                            gpu=cuda)

        # (batch_size, tgt_seq_len, target_vcoab_size)
        final_output = self.final_layer(dec_output)

        return final_output, dec_attn


    def sample(self, inp, max_len, sos_idx, eos_idx, src_pad_idx, tgt_pad_idx, device, temperature=None, decode_strategy="greedy"):
        """Forward propagate for transformer.
        
        Args:
          inp:
          max_len:
          temperature
          sos_idx
          eos_idx

        Returns:
          out: (batch_size, max_len)
        """
        if torch.is_tensor(inp):
            pass
        else:
            inp = torch.tensor(inp, device=device)

        # Gumbel-Softmax tricks
        batch_size = inp.shape[0]
        #sampled_ids = torch.zeros(batch_size, max_len).type(torch.LongTensor)

        # (batch_size, 1)
        # Create a tensor on CPU by default
        output = torch.tensor([sos_idx]*batch_size).unsqueeze(1)
        if device:
            output=output.to(device)
        assert output.shape[-1] == 1 
        
        for i in range(max_len-1):
            # print(output)

            # enc_pad_mask, combined_mask, dec_pad_mask
            enc_padding_mask, combined_mask, dec_padding_mask = create_transformer_masks(inp,
                                                                                         output, 
                                                                                         src_pad_idx=src_pad_idx,
                                                                                         tgt_pad_idx=tgt_pad_idx,
                                                                                         device=device)
            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, _ = self.forward(inp,     # (bathc_size, 1)
                                          output,  # (batch_size, 1-TO-MAXLEN)
                                          False,
                                          enc_padding_mask,
                                          combined_mask,
                                          dec_padding_mask,
                                          cuda=device)
            
            # Select the last word from the seq_len dimension
            # (batch_size, 1, vocab_size) to (batch_size, voacb_size) 
            predictions = predictions[: ,-1:, :].squeeze() 
            # print("preds", predictions.shape)

            if decode_strategy == "greedy":
                predicted_idx = torch.argmax(predictions, dim=-1).unsqueeze(1)
                # print(predicted_idx.shape)
                # print(predicted_idx)
            elif decode_strategy == "gumbel":
                # (batch_size, 1)
                # assert inp.shape[-1] = 1
                gumbel_distribution = gumbel_softmax_sample(predictions, temperature,gpu=cuda)
                # (batch_size, vocab_size)
                # print("gumbel", gumbel_distribution.shape)

                # (batch_sizes) to (bathc_size, 1)
                predicted_idx = torch.argmax(gumbel_distribution, dim=-1).unsqueeze(1)

            # print("pred idx", predicted_idx.shape)
            output = torch.cat((output, predicted_idx), 1)
            # Update along with col
            #sampled_ids[:,i] = predicted_idx.squeeze()
        return output
    

    def beam_decode(self, inp, k, max_len, sos_idx, eos_idx, src_pad_idx, tgt_pad_idx, device):
        """Beam-search decoding.

        Args:
            inp (tensor): Input with shape [batch_size, seq_len]. 
            k (int): Exporation of k hypothsis.
            max_len (int): Maximum length of inference. 
            sos_idx (int): Index of start of sentence symbol.
            eos_idx (int): Index of end of sentence symbol.
            src_pad_idx (int): Index of PAD symbol.
            tgt_pad_idx (int): Index of PAD symbol.
            device (str): Device should cuda or cpu.
            
        [batch_size*k, seq_len] -> [batch_size*k*k, seq_len] -> [batch_size*k, seq_len]

        In the loop of beam search, we do one beam decoding step.
        that means:
            1. Inference k hypothsis for k sample.
               We have [batch_size*k*k, 1] for log prob and token indices
            
            2. Compute the log prob for (batch_size*k*k) samples and
               Select the indices by the k maximum values out of k*k hypothesis.
               We then have the log prob matrix with shape [batch_size, k]
            
            3. Select the new hypothesis with shape [batch_size, 1] using the log prob matrix
            
            4. Concatenate it with `decoded_output` at the dim=1. It become [batch_size, current_seq_len+1]
            
            5. Repeat k times If not reach the `max_len`. Otherwise, do nothing.

        decoded_log_probs: shape [batch_size*k*k, 1]
        decoded_indices: shape [batch_size*k, 1]

        Returns:
            output (Tensor): Decoded batch. 
            log_prob: Log probability over a batch  with shape [batch_size,] 
        """
        if torch.is_tensor(inp):
            pass
        else:
            inp = torch.tensor(inp, device=device)
   
        ### Repeat k times: [batch_size, seq_len] -> [batch_size*k, seq_leg] ###
        ### Initialize the log probs and decoded output. Both has the shape (batch_size, 1]. ###
        batch_size = inp.shape[0]   
        decoded_output = torch.tensor([sos_idx]*(batch_size), device=device).unsqueeze(1)
        enc_padding_mask, combined_mask, dec_padding_mask = create_transformer_masks(inp,
                                                                                     decoded_output,
                                                                                     src_pad_idx=src_pad_idx,
                                                                                     tgt_pad_idx=tgt_pad_idx,
                                                                                     device=device)
        logits, _ = self.forward(inp, decoded_output, False, enc_padding_mask, combined_mask,
                                 dec_padding_mask, cuda=device)
        # bath are [batch_size, vocab] -> [batch_size, k] 
        log_probs = F.log_softmax(logits[:,-1:,:].squeeze(), dim=-1)
        k_log_probs, k_indices = log_probs.topk(k, dim=1)
        # print("k_log_probs outside of loop", k_log_probs)
        # print("k_indieces outside of loop", k_indices)
        # To [batch_size*k, 1]
        k_indices = k_indices.view(-1,1)
        # Initialize `decoded_output` [batch_size, 1] -> [batch_size*k, 1]
        decoded_log_probs = k_log_probs.view(-1,1)
        decoded_output = torch.cat((decoded_output.repeat(k, 1), k_indices), 1)
        # print("first cat", decoded_output)
        # print("added", decoded_log_probs)

        ### Repeat k times: [batch_size, seq_len] -> [batch_size*k, seq_leg] ###
        # inp = inp.repeat(k,1)
        inp = inp.repeat(1,k).view(batch_size*k, -1)
        # print("inp repeat k times at dim=1", inp)
 
        n=1
        ### Beam search steps ###
        # [batch_size*k, seq_len] -> [batch_size*k*k, seq_len+1] -> [batch_size*k, seq_len+1]
        # [batch_size*k, current_seq_len].repeat(k,1) + [batch_size*k*k,1] -> [batch_size*k*k, 1]
        for idx in range(0, max_len-2):
            # enc_pad_mask, combined_mask, dec_pad_mask
            enc_padding_mask, combined_mask, dec_padding_mask = create_transformer_masks(inp,
                                                                                         decoded_output,
                                                                                         src_pad_idx=src_pad_idx,
                                                                                         tgt_pad_idx=tgt_pad_idx,
                                                                                         device=device)
            ### 1. Inference k sample ###
            # Repeat the decode input k times as the new input
            # Inference first token based on [CLS]
            # logits.shape == [batch_size*k, seq_len, vocab_size] -> [batch_size*k, k]
            logits, _ = self.forward(inp,
                                     decoded_output,
                                     False,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask,
                                     cuda=device)
            
            # Select the logits of last token
            # [batch_size*k, seq_len, vocab_size] -> [batch_size*k, vocab_size]
            logits = logits[:,-1:,:].squeeze()
            log_probs = F.log_softmax(logits, dim=-1)

            # both with shape [batch_size*k, k]
            k_log_probs, k_indices = log_probs.topk(k, dim=1)
            # print("raw k_log_probs", k_log_probs)
            # print("raw k_indieces", k_indices)

            # To [batch_size*k*k, 1]
            k_log_probs = k_log_probs.view(-1,1)
            k_indices = k_indices.view(batch_size,-1)        
            # print("k_log_probs view in [batch*k*k, 1]", k_log_probs)
            # print("k_indices view in [batch, k*k]", k_indices)

            ### Stack  and get the indices with maximum log prob in each k hypothesis. ###
            # expand to [b*k*k,1]
            # While decoding, we keep k*k hypothesis for an sample
            # Therefore, we need duplicate the log_probs matrix k times
            # View in  [batch_size*k, 1]
            decoded_log_probs = decoded_log_probs.repeat(1, k).view(-1,1)
            # print("decoded_log_probs repeat k at dim=1", decoded_log_probs)
            decoded_log_probs += k_log_probs
            # print("decoded_log_probs added by k_log_probs", decoded_log_probs)
            
            # print("decoded_log_probs shape", decoded_log_probs.shape)
            # [batch_size, k*k]
            decoded_log_probs = decoded_log_probs.view(batch_size, -1)
            # print("decoded_prob_probs view in [batch,k*k]", decoded_log_probs)

            # Argmax return the indices of k samples. 
            # (1) Slecting [batch_size, k] out of [batch_size, k*k]
            # (2) To make the indices become the indices of a (batch*k*k)-wise length
            # We need to add the local index with i*(k*k)
            #largetest_log_prob_indices = decoded_log_probs.argmax(dim=-1) + idx_diff # k top
            # [batch_size, k*k] -> [batch_size, k]
            k_log_prob, k_log_prob_indices = decoded_log_probs.topk(k, dim=1) 
            # print("k_log_prob", k_log_prob)
            # print("k_log_prob_indices", k_log_prob_indices)
             
            # print(largetest_log_prob_indices)
            ### Select by `largetest_log_prob_indices` at dim=0 ###
            # make [b*k*k,1] -> [b*k, 1]
            # a. for decoded_log_probs. To decoded_log_probs with shape [batch_size*k, 1]
            decoded_log_probs = k_log_prob.view(-1, 1)
            # print("decoded log_probs view in [batch*k,1]", decoded_log_probs)

            # b. for k_indices
            idx_diff = (torch.arange((batch_size)) * (k*k)).view(-1,1)
            k_log_prob_indices = (k_log_prob_indices + idx_diff).view(-1)
            k_indices = torch.index_select(k_indices.view(-1,1), 0, k_log_prob_indices)
            # print("k_log_prob_indices", k_log_prob_indices)
            # print("k_indices", k_indices)
            
            # c. for decoded_output
            # Make [batch_size*k,k, 1] first, then select
            seq_len = decoded_output.shape[-1]
            # print("decoded_output before", decoded_output)
            decoded_output = decoded_output.repeat(1, k).view(batch_size*k*k, seq_len)
            # print("decoded_output repeat k times in last dims and view in [batch*k*k, seq]", decoded_output)
            decoded_output = torch.index_select(decoded_output, 0, k_log_prob_indices)
            # print("decoded_output index_select with k_log_prob_indices", decoded_output)
            # Cat [batch_size*k, current_seq_len] + [batch_size*k, 1] ->
            decoded_output = torch.cat((decoded_output, k_indices), 1)
            # print("decoded_output cat with k_indices", decoded_output)

            if n == 100:
                return
            n+=1
            #print("decoded log prob", decoded_log_probs)
            #print("decoded output", decoded_output)
            
            # If not reach max_len, then repeat k times. Otherwise jump
            # [batch_size, seq_len+1] -> [batch_size*k, seq_len+1]
            if decoded_output.shape[1] == max_len:
                batch_k_output = decoded_output
                idx_diff = (torch.arange((batch_size)) * (k)).view(-1,1)
                _, largest_indices = decoded_log_probs.view(batch_size, -1).topk(1, dim=1)
                largest_indices = (largest_indices+idx_diff).view(-1)
                decoded_output = torch.index_select(decoded_output, 0, largest_indices)
                                
        return decoded_output, decoded_log_probs, batch_k_output


    def evaluate(self, dataset, args, pad_idx, acc_fn):
        """Perform evaluate."""
        self.eval()
        
        total_loss = 0
        total_accuracy = 0
        step = 0
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
        for features_dict in dataset:
            # Encoder input: (batch_size, seq_len)
            src = features_dict["source_ids"]
            # Decoder input and target: (batch_size, seq_len-1)
            tgt_inp = features_dict["target_input"]
            tgt_out = features_dict["target_output"]
            
            enc_padding_mask = features_dict["enc_padding_mask"]
            combined_mask = features_dict["combined_mask"]
            dec_padding_mask =features_dict["dec_padding_mask"]
            
            logits, attn = self.forward(src=src,
                                        tgt=tgt_inp,
                                        training=False,
                                        enc_padding_mask=enc_padding_mask,
                                        look_ahead_mask=combined_mask,
                                        dec_padding_mask=dec_padding_mask,
                                        cuda=args.gpu)

            two_d_logits = logits.reshape(-1, logits.shape[-1])
            loss = loss_fn(two_d_logits, tgt_out.reshape(-1))

            pred = logits.argmax(-1)
            acc = acc_fn(real=tgt_out,
                         pred=pred,
                         pad_idx=pad_idx)
            total_loss += loss.item()
            total_accuracy += acc
            step+= 1
        avg_loss = total_loss/ step
        avg_acc = total_accuracy / step
        return avg_loss, avg_acc


def sample_gumbel(shape, eps=1e-20, device=None):
    """Sample from Gumbel(0, 1)"""
    
    # The drawn nosie is created by default in CPU
    noise = torch.rand(shape, device=device)
    return -torch.log(-torch.log(noise+eps)+eps)


def gumbel_softmax_sample(logits, temperature, gpu):
    """Sample from Gumbel softmax distribution.
    Reference:
        1. Gumbel distribution: https://en.wikipedia.org/wiki/Gumbel_distribution
        2. Inverse Tranform Sampling: https://en.wikipedia.org/wiki/Inverse_transform_sampling
    """
    y = logits + sample_gumbel(shape=logits.shape, device=gpu)
    return nn.functional.softmax(y/temperature, dim=-1)


if __name__ == "__main__":
    transf = Transformer(2,512,8,2048,8500, 8000,10000,6000, -100, None)

    i = torch.randint(0, 200, (64,38))  
    tgt_i = torch.randint(0,200, (64, 36))

    output, attn = transf(i,
                          tgt_i,
                          training=False,
                          enc_padding_mask=None,
                          look_ahead_mask=None,
                          dec_padding_mask=None)

    print(output.shape)
