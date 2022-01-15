"""Train WordPiece tokenizer and generate splits."""
import argparse

from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase


def get_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--source_file', type=str, default="de-en/PHP.de-en.de")
    parser.add_argument('--target_file', type=str, default="de-en/PHP.de-en.en")

    parser.add_argument('--output_dir', type=str, default='de-en/subword')
    parser.add_argument('--use_wikitext', type=bool, default=True)
    
    return parser.parse_args()


def save_vocab(vocab_dict, output_file):
    """Save examples.

    Args:
      vocab_dict: List of examples
      output_file: wirte file.
    """
    with open(output_file, 'w') as wf:
        for word in vocab_dict:    
            # Separate by tab
            freq = 100 # 
            wf.write(f"{word}\t{freq}\n")
    print("Saving {} vocabulary to {}".format(len(vocab_dict), output_file))


def get_tokenizer():
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.normalizer = normalizers.Sequence([Lowercase()])
    return tokenizer

def main():
    # Argument parser
    args = get_args()

    tokenizer_de = get_tokenizer()
    tokenizer_en = get_tokenizer()

    trainer = WordPieceTrainer(special_tokens=["[UNK]"],
                               continuing_subword_prefix="##")

    if args.use_wikitext:
        de_files = [args.source_file]
        try:
            en_files = [args.target_file] +  [f"wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
            print("Using wikitext-103-raw for English corpus")
        except:    
            de_files = [args.target_file]    
    else:
        de_files = [args.source_file]
        en_files = [args.target_file]
        

    # Train respective tokenizer
    print("training German tokenizer")
    tokenizer_de.train(de_files, trainer)
    tokenizer_de.save(args.output_dir+"/tokenizer-de.json")
    
    print("training English tokenizer")
    tokenizer_en.train(en_files, trainer)
    tokenizer_en.save(args.output_dir+"/tokenizer-en.json")
    
    # Save vocabulary
    save_vocab(tokenizer_de.get_vocab(), args.output_dir+"/source.vocab")
    save_vocab(tokenizer_en.get_vocab(), args.output_dir+"/target.vocab")


if __name__ == "__main__":
    main()

