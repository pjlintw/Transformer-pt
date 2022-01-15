"""Dataset loading script for PHP De-En corpus sparated by tab."""
from tokenizers import Tokenizer
import datasets

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = "PHPDeEn"

class PHPDeEnConfig(datasets.BuilderConfig):
    """BuilderConfig for OntoNotes 4.0"""

    def __init__(self, **kwargs):
        """BuilderConfig for OntoNotes 4.0.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(PHPDeEnConfig, self).__init__(**kwargs)


class PHPDeEn(datasets.GeneratorBasedBuilder):
    """OntoNotes 4.0."""
    BUILDER_CONFIGS = [
        PHPDeEnConfig(name='PHPDeEn_subword', description="PHP De-En dataset.")
    ]
    
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("int32"),
                    # "length": datasets.Value("int32"),
                    "source": datasets.Sequence(datasets.Value("string")),
                    "target": datasets.Sequence(datasets.Value("string"))
                }
            )
        )

    def _split_generators(self, dl_manager):
        
        train_file = "./data/de-en/sample.train"
        dev_file = "./data/de-en/sample.dev"
        test_file ="./data/de-en/sample.test"
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_file}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": dev_file}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_file}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)

        de_file = "./data/de-en/subword/tokenizer-de.json"
        en_file = "./data/de-en/subword/tokenizer-en.json"
        logger.info("⏳ Loading German tokenizers from = %s", de_file)
        logger.info("⏳ Loading English tokenizers from = %s", en_file)

        tokenizer_de =  Tokenizer.from_file(de_file)
        tokenizer_en =  Tokenizer.from_file(en_file)
    
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            
            for line in f:
                line_list = line.strip().split("\t")
                idx, src_sent, tgt_sent = line_list
                
                if len(line_list) != 3:
                    print(line_list)
                yield guid, {
                    "id": int(idx),
                    "source": tokenizer_de.encode(src_sent).tokens,
                    "target": tokenizer_en.encode(tgt_sent).tokens
                }


if __name__ == "__main__":
    WikiTableQuestions(__name__)





























