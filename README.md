# Transformer in Pytorch
[**Data**](#dataset-and-preprocessing) | [**Play with Transformer**](#transformer-for-german-english-Translation) | [**Hyperparameter Guide**](hyperparameter-uide)

The repository works on the implementation of Transformer in Pytorch and the trainer for Transformer on the PHP German-English corpus.

We use `PHP German-English` dataset which contains `11k` non-duplicated German-English sentence pairs for the training split and `1k` for the development set and the test set respectively. By executing the trainer script, the model is trained in supervised manner and all the output files, including the logger, hyperparameter, checkpoint and predictions will be saved in an output folder after done with training.

To reproduce the results, follow the steps below.

* New Januray 15th, 2021: Transformer in Pytorch
* New Januray 15th, 2021: Beam search decoding

## References for the Implementations

For quickly checking the implemtations of Transformer, we provide the references for linkinkg the functionalities to specific file.

#### 1. Core Functions in Transformer

- Transformer: `transformer.py`
- TransformerDecoder: `decoder.py`
- TransformerEncoder: `encoder.py`
- WarmupScheduler: `transformer_blocks.py`
- Positional Encoding: `transformer_blocks.py`
- Scaled Dot-Product Attention: `transformer_blocks.py`
- MultiHeadAttention: `transformer_blocks.py`
- FeedForwardBlock: `transformer_blocks.py`


#### 2. Support Functions for training

- Duplicated examples removal: `data_preprocess.py`
- Hyperparameter chossing: `get_args` in `run_trainer.py`
- Trainer: `train_generator_MLE` in `run_trainer.py`
- Masking: `create_transformer_masks` in `utils.py`

#### 3. Prediction

- Prediction on Test set: `test.epoch-#N.pred`


## Files Structure

There are many files in the result folder. The files are exported by the trainer script after done with the training precedure.

The figure below uses a `results/word/L-4_D-128_H-8` as example and explains the purpose of these files. Note that `#N` refers to a number with the file. 


```
results/word/L-4_D-128_H-8
|--ckpt
|  └-- epoch-#N.pt     # Checkpoint of Transformer saved at N-th epoch
|-- example.log        # Log file
|-- hyparams.txt       # Hyperparameters for modeling and training
└── test.epoch-#N.pred # Prediction on test set at the N-th epoch
```


## Installation

### Python version

* Python >= 3.8

### Environment

Create an environment from file and activate the environment.

```
conda env create -f environment.yml
conda activate transformer-pt
```

If conda fails to create an environment from `environment.yml`. This may be caused by the platform-specific build constraints in the file. Try to create one by installing the important packages manually. The `environment.yml` was built in macOS.

**Note**: Running `conda env export > environment.yml` will include all the 
dependencies conda automatically installed for you. Some dependencies may not work in different platforms.
We suggest you to use the `--from-history` flag to export the packages to the environment setting file.
Make sure `conda` only exports the packages that you've explicitly asked for.

```
conda env export > environment.yml --from-history
```

## Dataset and Preprocessing

### PHP German-English corpus

We use the `PHP-German-English` parallel corpus as the dataset for the training of the transformer. The source-target files should be in the `data/de-en` folder. You can download the dataset via the link:  [German-English parallel corpus](https://opus.nlpl.eu/download.php?f=PHP/v1/moses/de-en.txt.zip). 

To preprocess the dataset and fetch the statistical information, move the working directory to the `data` folder.

```
cd data
```

In the following steps, the script extract the sentence from `PHP.de-en.de` and `PHP.de-en.en`, then split them into `sample.train`, `sample.dev` and `sample.test`.


### Preprocessing and Dataset Splitting

The files `PHP.de-en.de` and `PHP.de-en.en` contain duplicated samples for German-English sentences. We extract the non-duplicated pairs and 
use German sentences as the soruce language and English sentences as target language.

Running `data_preprocess.py` will extract `id`, `source sentence` and `target sentence`, then write them to `sample.tsv` in which the items  are separated by tab.  

The arguments `--source_file`, `--source_file` and `--output_dir` recevie the repositories of source file, target file and the output folder respectively. 

The file `sample.tsv` contain all extracted examples and the splits: `sample.train`, `sample.dev` and `sample.test` are for the network training.  The examples will be shuffled in the scripts and split into `train`, `validation` and `test` files according to the arguments of `--eval_samples` and `--test_samples`. They decide the number of samples for dev and test splits. We select 11782 for the training set, 1000 for validation and test sets respectively after performing duplication removal. To preprocess and split the datasets, you need to run the code below. 


```python
python data_preprocess.py \
  --source_file de-en/PHP.de-en.de \
  --target_file de-en/PHP.de-en.de \
  --output_dir de-en \
  --eval_samples 1000 \
  --test_samples 1000
```

These output files for building datasets will be under the path `--output_dir`. You will get the result.

```
Number of source sentences: 39707
Number of sentence pairs after duplicated removal: 11782
Loading 11782 examples
Seed 49 is used to shuffle examples
Saving 11782 examples to de-en/sample.tsv
Saving 9782 examples to de-en/sample.train
Saving 1000 examples to de-en/sample.dev
Saving 1000 examples to de-en/sample.test
Saving 16737 vocabulary to de-en/source.vocab
Saving 16737 vocabulary to de-en/target.vocab
```

Make sure that you pass the correct **data file** to the `--source_file` and `--target_file` arguments and they have enough examples for splitting out development and test sets. The output files may have no example, if  the number of examples in the source-target files are less than the number of eval and test examples.


### Using dataset loading script for German-English corpus

We use our dataset loading script `php-de-en.py` for creating dataset when runing our training script. The script builds the train, validation and test sets from the dataset splits obtained by the `data_preprocess.py` program. 
Make sure the dataset split files `sample.train`, `sample.dev` , and `sample.test` are included in the datasets folder `data/de-en`.

If you get an error message like:

```
pyarrow.lib.ArrowTypeError: Could not convert 1 with type int: was not a sequence or recognized null for conversion to list type
```

You may have run other datasets in the same folder before. The Huggingface already created `.arrow` files once you run a loading script. These files are for reloading the datasets quickly.

Try to move the dataset you would like to use to the other folder and modify the path in the loading script. 

Or delete the relevant folder and files in the `.cache` for datasets. `cd ~/USERS_NAME/.cache/huggingface/datasets/` and `rm -r *`. This means that all the loading records will be removed and
 Hugging Face will create the `.arrows` files again, including the previous loading records. 

### Subword Tokenizer

To perform on subword-level translation, we provide subword tokenization based on `WordPiece`. The `run_tokenizer.py` script trains an `WordPiece` tokenizer on the `PHP.de-en.de` and `PHP.de-en.en` files. 

Note that the tokenizer trainer are taking the sentecne file as input. Therefore you need the `PHP.de-en.de` and `PHP.de-en.en` file before training the subword tokenizer. 

Once the files are ready, you can create the folder for subword files

```
cd data
mkdir subword
```

and run:

```python
python run_tokenizer.py \
	--output_dir subword \
	--source_vocab de-en/PHP.de-en.de \
	--target_vocab de-en/PHP.de-en.en
```

Similar to `data_preprocess.py`. This will generate the 
`source.vocab`, `target.vocab` in subword-level under `--output_dir` folder but no data split and `tokenizer-de.json` and `tokenizer-en.json` files will be exported for tokenizer re-loading in the subword-level data script.

If one wants train the transformer on subword tokenization, see the section: train on subword-level corpus.


## Transformer for German-English Translation

Transformer is an encoder-decoder architecture. Given a dataset consisting of the parallel sentences, the transformer encodes the source sentence and decodes a sentence in target langauge token by token. To do so, we implement a trainer for the training of the network on PHP German-English corpus.

Note that dataset script and vocab files are required.

### Train with Transformer

To train the Transformer, you can run `run_trainer.py` with the arguments for hyperparameters and dataset loading and vocab files. The `--source_vocab` and `--target_vocab` and `dataset_script` are required for running the trainer. The other arguments is optional. For detailed information about the arguments, please check the hyperparameter guide below.

 
```python
python run_trainer.py \
	--output_dir tmp \
	--source_vocab data/de-en/source.vocab \
	--target_vocab data/de-en/target.vocab \
	--tf_layers 4 \
	--tf_dims  128 \
	--tf_heads 8 \
	--dataset_script php-de-en.py \
	--max_seq_length 20 \
	--batch_size 64 \
	--do_train True \
	--do_eval True \
	--do_predict True \
	--mle_epochs 30
```

The arguments `--tf_layers`,  `--tf_dims`, `--tf_heads`, `--tf_dropout_rate`, `--tf_shared_emb_layer` and `--tf_learning_rate` design the Transformer's architecture. 

Regarding the training arguments, the Transformer was trained with maximum log-likelihood with `--mle_epochs` epochs using `--batch_size` mini-batch. The trainer saves the checkpoint of the model every 5 epochs and the model will be automatically evaluated and inference on the test set if `--do_eval==True` and `--do_predict==True`.

Note that all the output files, including the logger, hyperparameter, checkpoint and predictions will be saved in the `--output_dir`.


### Train on subword-level corpus.

It is necessary to provide the subword vocaublaries and data loading script. To do so, one need to specify the files to `--source_vocab`, `--target_vocab` and `--dataset_script`.

For example, you can run with the command:

```python
python run_trainer.py \
	--output_dir tmp \
	--source_vocab data/de-en/subword/source.vocab \
    --target_vocab data/de-en/subword/target.vocab \
    --dataset_script php-de-en_subword.py
```



### Hyperparameter search with Trainer script

To evaluate the architecture with different hyperparameter settings, we prepare the shell script for running the transformer with `num_layers` ranging from 2 to 8, `num_dimension` from 64 to 512 and `num_head` from 1 to 32. The script experiements 60 transformers varied in the hyperparameter setting.

You can simply run the shell script:


```
. ./run_trainer.sh
```

### Hyperparameter Guide

This section intorudces the argument in `run_trainer.py`.


| Parameters | description | default | Type |
|---|---|---|---|
| `output_dir`  | output directory  | tmp | str  |
| `max_seq_length`| maximum sequence length | 20 | int |
| `source_vocab` | source vocab file | - | str |
| `target_vocab` | target vocab file | -  | str |
| `tf_layers` | number of layers | 4  | int  |
| `tf_dims ` | number of d_model | 128 | int |
| `tf_heads` | number of heads | 8 | int |
| `tf_dropout_rate` | dropout rate  | 0.1 | float  |
| `tf_shared_emb_layer` | wether to share embedding layer  | False | bool  |
| `tf_learning_rate` | learning rate | 1e-2 | float |
| `dataset_script` | dataset loading script  | - | str  |
| `do_train` | wether to train the model  | True |  bool |
| `do_eval` | wether to evaluate | True | bool |
| `do_predict` | wether to inference on test set | True | bool |
| `batch_size` | batch size | 64 | int |
| `mle_epochs` | number of epochs | 10 | bool |
| `max_train_samples` | maximum training examples | - | int |
| `max_val_samples` | maximum develpment examples | - | int |
| `max_test_samples ` | maximum test examples | - | int |
| `debugging` | debugging mode | False  | bool |

### Contact Information

For the help or the issues using the code, please submit a GitHub issue or contact the author via `pjlintw@gmail.com`.



