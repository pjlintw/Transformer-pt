

import torch

def beam_decode(inp, k, max_len, sos_idx, eos_idx, src_pad_idx, tgt_pad_idx, device):
    
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
    
    batch_size = inp.shape[0]

    ### Initialize the log probs and decoded output. Both has the shape (batch_size*k, 1]. ###
    decooded_output = torch.tensor([sos_idx]*(batch_size*k), device=device).unsqueeze(1)
    decoded_log_probs = torch.zeros((batch_size*k,1), device=device)

    ### Beam search steps ###
    # [batch_size*k, seq_len] -> [batch_size*k*k, seq_len+1] -> [batch_size*k, seq_len+1]
    # [batch_size*k, current_seq_len].repeat(k,1) + [batch_size*k*k,1] -> [batch_size*k*k, 1]
    for idx in range(0, max_len-1):
        ### enc_pad_mask, combined_mask, dec_pad_mask
        enc_padding_mask, combined_mask, dec_padding_mask = create_transformer_masks(inp,
                                                                                     decoded_output,
                                                                                     src_pad_idx=src_pad_idx,
                                                                                     tgt_pad_idx=tgt_pad_idx,
                                                                                     device=device)
        ### 1. Inference k sample ###
        # Repeat the decode input k times as the new input
        # Inference first token based on [CLS]
        # [batch_size*k, vocab_size] -> [batch_size*k, k]
        logits, _ = self.forward(inp,
                                 decoded_output,
                                 False,
                                 enc_padding_mask,
                                 combined_mask,
                                 dec_padding_mask,
                                 cuda=device)
        # logits = torch.tensor([[1.,2.,3.,4.,5.]]).repeat(batch_size*k, 1)
        # print("Inference: \n ", logits)
        # print("Inference shape: \n", logits.shape)

        # both with shape [batch_size*k, k] -> [batch_size*k*k, 1]
        k_lop_probs, k_indices = logits.topk(k)
        k_lop_probs = k_lop_probs.view(-1,1)
        k_indices = k_indices.view(batch_size,-1)
        
        ### Stack  and get the indices with maximum log prob in each k hypothesis. ###
        # expand to [b*k*k,1]
        # While decoding, we keep k*k hypothesis for an sample
        # Therefore, we need duplicate the log_probs matrix k times
        # Reshape to [batch_size*k*k, 1]
        decoded_log_probs = decoded_log_probs.repeat(k, 1)
        decoded_log_probs += k_lop_probs
        
        # print("decoded_log_probs shape", decoded_log_probs.shape)
        # [batch_size, k*k]
        decoded_log_probs = decoded_log_probs.view(batch_size, -1)
        
        # Argmax return the indices of k samples. 
        # (1) Slecting [batch_size, k] out of [batch_size, k*k]
        # (2) To make the indices become the indices of a (batch*k*k)-wise length
        # We need to add the local index with i*(k*k)
        #largetest_log_prob_indices = decoded_log_probs.argmax(dim=-1) + idx_diff # k top
        # [batch_size, k*k] -> [batch_size, k]
        k_log_prob, k_log_prob_indices = decoded_log_probs.topk(k, dim=1) 
        
        # print(largetest_log_prob_indices)
        ### Select by `largetest_log_prob_indices` at dim=0 ###
        # make [b*k*k,1] -> [b*k, 1]
        # a. for decoded_log_probs. To decoded_log_probs with shape [batch_size*k, 1]
        decoded_log_probs = k_log_prob.view(-1, 1)
        print("new",decoded_log_probs)
        
        # b. for k_indices
        idx_diff = (torch.arange((batch_size)) * (k*k)).view(-1,1)
        k_log_prob_indices = (k_log_prob_indices + idx_diff).view(-1)
        k_indices = torch.index_select(k_indices.view(-1,1), 0, k_log_prob_indices)
        print("k ", k_indices)
        print(k_indices.shape)
        # c. for decooded_output
        # Make [batch_size*k,k, 1] first, then select
        # print(decooded_output)
        decooded_output = torch.index_select(decooded_output.repeat(k,1), 0, k_log_prob_indices)
        #print(decooded_output)
        
        # Cat [batch_size*k, current_seq_len] + [batch_size*k, 1] ->
        decooded_output = torch.cat((decooded_output, k_indices), 1)
        print("out",decooded_output)
        print(decooded_output.shape)
        #print("decoded log prob", decoded_log_probs)
        #print("decoded output", decooded_output)
        
        # If not reach max_len, then repeat k times. Otherwise jump
        # [batch_size, seq_len+1] -> [batch_size*k, seq_len+1]
        if decooded_output.shape[1] == max_len:
            idx_diff = (torch.arange((batch_size)) * (k)).view(-1,1)
            _, largest_indices = decoded_log_probs.view(batch_size, -1).topk(1, dim=1)
            largest_indices = (largest_indices+idx_diff).view(-1)
            decooded_output = torch.index_select(decooded_output, 0, largest_indices)
    # print(decooded_output)
    # print(decooded_output.shape)
    return decooded_output, decoded_log_probs


def main():    
    inp = torch.tensor([[10.,20.,30.],[10.,20.,30.]])

    print(inp)

    beam_decode(inp,
                max_len=6,
                k=2,
                sos_idx=1,
                eos_idx=100,
                src_pad_idx=123,
                tgt_pad_idx=456,
                device="cpu")





if __name__ == "__main__":
    main()
