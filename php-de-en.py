"""Dataset loading script for PHP De-En corpus sparated by tab."""
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
        PHPDeEnConfig(name='PHPDeEn', description="PHP De-En dataset.")
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
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            
            for line in f:
                line_list = line.strip().split("\t")
                idx, src_sent, tgt_sent = line_list

                if len(line_list) != 3:
                    print(line_list)
                yield guid, {
                    "id": int(idx),
                    "source": src_sent.split(),
                    "target": tgt_sent.split()
                }


if __name__ == "__main__":
    WikiTableQuestions(__name__)





























