"""Extract POS tags and split dataset."""
import os
import argparse
import random

from collections import Counter
from pathlib import Path
from nltk.tokenize import word_tokenize


def get_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--source_file', type=str, default="de-en/PHP.de-en.de")
    parser.add_argument('--target_file', type=str, default="de-en/PHP.de-en.en")

    parser.add_argument('--output_dir', type=str, default='de-en')
    parser.add_argument('--eval_samples', type=int, default=1000)
    parser.add_argument('--test_samples', type=int, default=1000)

    return parser.parse_args()


def preprocess_sentence(sentence, lower=True):
    """Preprocess sentence

    Aegs:
        sentence: str.
        lower: bool.
    
    Returns:
      examples: str list.
    """
    sent = sentence.strip()
    if lower:
        sent = sent.lower()
    return " ".join(word_tokenize(sent))


def sentence_parse(sentence, label_list):
    """Parse sentence into list of text, and label.

    Args:
      sentence: str, sentence line.
      label_list: list of class in string.

    Returns:
      example: tuple of word list, sentence and index
              
    Example:
        ("what"
         4,
         "what is it ?")
    """
    sentence_list = sentence.split()

    # question mark not separate
    if sentence_list[-1][-1] == "?":
        # Remove and add 
        sentence_list[-1] = sentence_list[-1][:-2]
        sentence_list +=  ["?"]

    sent_len = len(sentence_list)
    label = None

    # check first 5 words have question works
    for w_idx in range(min(5, sent_len)):
        for q_word in label_list:
            if q_word == sentence_list[w_idx]:
                label = q_word 
                break
        
    # Return None if no label
    if label == None:
        return None

    examples = (label, sent_len, " ".join(sentence_list))
    return examples


def build_examples(source_file, target_file):
    """Create data example of features.
    
    Args:
      source_file: str, file path to source file.
      target_file: str, file path to target file. 

    Returns:
      examples: List of tuple contains (1) sentence idex , (2) soruce sentence and 
                (3) target sentence.
                
    """
    # List of tuple containing (source sentence, target sentence)
    examples = list()
    
    checkSet = set()
    with open(source_file, "r") as src_f, open(target_file, "r") as tgt_f:
        src_lst = src_f.readlines()
        tgt_lst = tgt_f.readlines()

        for idx, (src_sent, tgt_sent) in enumerate(zip(src_lst, tgt_lst)):
            # Continue if duplicated sample
            if src_sent in checkSet:
                continue
            src_tokenized_sent = preprocess_sentence(src_sent)
            tgt_tokenized_sent = preprocess_sentence(tgt_sent)
            example = (idx, src_tokenized_sent , tgt_tokenized_sent)
            examples.append(example)
            checkSet.update([src_sent])
    print("Number of source sentences:", len(src_lst))
    print("Number of sentence pairs after duplicated removal:", len(examples))
    return examples


def save_example_to_file(examples, output_file):
    """Save examples.
    Args:
      examples: List of examples
      output_file: wirte file.
    """
    # as format "what   4   what is it ?"
    with open(output_file, 'w') as wf:
        for (idx, src_sent, tgt_sent) in examples:    
            # Separate by tab
            wf.write(f"{idx}\t{src_sent}\t{tgt_sent}\n")
    

def save_vocab(examples, src_output_file, tgt_output_file):
    """Save examples.

    Args:
      examples: List of examples
      output_file: wirte file.
    """
    src_vocab_cnt = Counter()
    tgt_vocab_cnt = Counter()

    for (_, src_sent, tgt_sent) in examples:
        src_vocab_cnt.update(src_sent.split())
        tgt_vocab_cnt.update(tgt_sent.split())

    with open(src_output_file, 'w') as source_wf:
        for (word, freq) in src_vocab_cnt.most_common():    
            # Separate by tab
            source_wf.write(f"{word}\t{freq}\n")

    with open(tgt_output_file, 'w') as target_wf:
        for (word, freq) in tgt_vocab_cnt.most_common():    
            # Separate by tab
            target_wf.write(f"{word}\t{freq}\n")
    print("Saving {} vocabulary to {}".format(len(src_vocab_cnt), src_output_file))
    print("Saving {} vocabulary to {}".format(len(tgt_vocab_cnt), tgt_output_file))


def main():
    # Argument parser
    args = get_args()
    SEED = 49

    # Collect examples from `sample.conll`
    examples = build_examples(source_file=args.source_file,
                              target_file=args.target_file)
    num_examples = len(examples)

    print("Loading {} examples".format(num_examples))

    # # Shuffle 
    random.Random(SEED).shuffle(examples)
    print("Seed {} is used to shuffle examples".format(SEED))
    
    # Write `sample.tsv`
    write_file = Path(args.output_dir, "sample.tsv")
    save_example_to_file(examples=examples,
                         output_file=write_file)
    print("Saving {} examples to {}".format(num_examples, write_file))

    ### Train, validation, test splits. ###
    n_eval = int(args.eval_samples)
    n_test = int(args.test_samples)

    # Spliting datasets
    train_examples = examples[:-n_eval-n_test]
    eval_start = -(n_eval+n_test)
    eval_examples = examples[eval_start:-n_test]
    test_examples = examples[-n_test: ]
    
    # Write `sample.train`
    write_file = Path(args.output_dir, "sample.train")
    save_example_to_file(examples=train_examples,
                         output_file=write_file)
    print("Saving {} examples to {}".format(len(train_examples), write_file))


    # Write `sample.dev`
    write_file = Path(args.output_dir, "sample.dev")
    save_example_to_file(examples=eval_examples,
                         output_file=write_file)
    print("Saving {} examples to {}".format(len(eval_examples), write_file))
    

    # Write `sample.test`
    write_file = Path(args.output_dir, "sample.test")
    save_example_to_file(examples=test_examples,
                         output_file=write_file)
    print("Saving {} examples to {}".format(len(test_examples), write_file))


    # Write vocab
    src_write_file = Path(args.output_dir, "source.vocab")
    tgt_write_file = Path(args.output_dir, "target.vocab")
    save_vocab(examples=examples,
               src_output_file=src_write_file,
               tgt_output_file=tgt_write_file)


if __name__ == '__main__':
    main()
