[**中文说明**](./README.md) | [**English**](./readme_en.md)

<p align="center">
    <img src="./pics/banner.svg" width="500"/>
</p>

This project provides a Chinese-oriented BERT pre-training model, which aims to enrich Chinese natural language processing resources and provide a variety of Chinese pre-training model options. Experts and scholars are welcome to download and use, and jointly promote and develop the construction of Chinese resources.
This project is based on Google's official BERT: https://github.com/google-research/bert
Other related resources: - Chinese BERT pre-training model: https://github.com/ymcui/Chinese-BERT-wwm


## NEWS

**2023/5/27 Release Sentence-BERT: Drogo, Base, Large, Multi-Lingual-Base.**   
Sentence-BERT works well only for text vectorization representation, please refer to the sentence_transformers project.


<details>
<summary>Historical news</summary>

2023/5/23 Release Stark-BERT: Eddard, Lyarra, Rickard, Lyanna, including tf, pytorch models. 
12-layer, 768-hidden, 12-heads, 102.3M/105.1M parameters training 200,000 steps.  

2023/5/22 Release Night King BERT: bert_night-king_36L_cn, including tf and pytorch models. 
Chinese model bert_night-king: its detailed parameters are 36-layer, 1024-hidden, 16-heads, 476.7M parameters 

2023/5/16 released Chinese pre-training models BERT-Tiny-CN, BERT-Mini-CN. 

Trained for 100k steps from the news corpus. The hyperparameters are basically the same as Google BERT. 

* BERT-Tiny: masked_lm_accuracy=22.74%, NSP_accuracy=100%. 
* BERT-Mini: masked_lm_accuracy=33.54%, NSP_accuracy=100%. 
* The above word segmentation MASK method uses Google's default method: case-sensitive, word segmentation according to Chinese characters. '
* The vocabulary adopts the default vocabulary of 21128 words in Google Chinese.  

2023/5/9 Fix download link 

2021/2/6 All models already support Pytorch, Tensorflow1 and Tensorflow2, please call or download through the transformers library.

2021/2/6The models published in this directory can be accessed to [hugging face transformers](https://github.com/huggingface/transformers) in the future, view [quick load](#quick load)

2021/2/6 `bert_12L_cn`It can be downloaded.
</details>

## Content guidance
| Chapter                                                                     | Description                                        |
|-----------------------------------------------------------------------------|----------------------------------------------------|
| [introduction](#Introduction)                                               | Introduce the basic principles of BERT-wwm         |
| [Model Download](#Model_Download)                                           | The Chinese pre-training BERT download links       |
| [Baseline System Effects](#Baseline_System_Effects)                         | Some baseline system effects are listed            |
| [Pretraining](#Pretraining)                                                 | Description of pre-training details                |
| [Downstream_task_fine-tuning_details](#Downstream_task_fine-tuning_details) | Description of downstream task fine-tuning details |
| [FAQ](#FAQ)                                                                 | FAQ                                                |
| [Citation](#Citation)                                                       | Technical reports                                  |


## Introduction

Whole Word Masking (wwm), temporarily translated as whole word Mask or whole word Mask , is an upgraded version of BERT released by Google on May 31, 2019, which mainly changes the training sample generation strategy in the original pre-training stage. To put it simply, the original WordPiece-based word segmentation method will divide a complete word into several subwords. When generating training samples, these divided subwords will be randomly masked. In the whole word Mask , if part of a WordPiece subword of a complete word is masked, other parts belonging to the same word will also be masked, that is, the whole word Mask .
It should be noted that the mask here refers to a generalized mask (replaced with [MASK]; keep the original vocabulary; randomly replaced with another word), not limited to the case where words are replaced with [MASK] tags . For more detailed instructions and examples, please refer to: #4
the BERT-base, Chinese officially released by Google , Chinese is segmented at the granularity of characters , and Chinese word segmentation (CWS) in traditional NLP is not considered. We applied the method of full-word Mask to Chinese, used Chinese Wikipedia (including Simplified and Traditional) for training, and used a word segmentation tool, that is, Mask all the Chinese characters that make up the same word .


The following text shows an example of the generation of the full word Mask . Note: For ease of understanding, only the [MASK] tag is considered in the following example.

| Description             | example|
|:------------------------| :--------- |
| The original text       | Use a language model to predict the probability of the next word. |
| Word segmentation text  | Use a language model to predict the probability of the next word. |
| The original mask input | Use the language [MASK] type to [MASK] measure the pro [MASK] ##lity of the next word. |
| Full word mask input    | Use language [MASK] [MASK] to [MASK] [MASK] [MASK] [MASK] [MASK] of the next word. |

## Model_Download


| dataset                                                             | owner      | model                                                     | language | layers | hidden | head | Parameter amount             |
|---------------------------------------------------------------------|------------|-----------------------------------------------------------|------|-------|--------|------|-----------------|
| news[corpus-3]                                                      | Brian Shen | [bert_tiny_cn_tf],[bert_tiny_cn_pt]                       | cn   | 2     | 128    | 2    | 3.2M            |
| news[corpus-3]                                                      | Brian Shen | [bert_mini_cn_tf], [bert_mini_cn_pt]                      | cn   | 4     | 256    | 4    | 8.8M            |
| Middle School Reading Comprehension                                 | Brian Shen | [bert_2L_cn]                                              | cn   | 2     | 768    | 4    | 16.8M           |
| Middle School Reading Comprehension                                 | Brian Shen | [bert_6L_cn]                                              | cn   | 6     | 768    | 12   | 45.1M           |
| Chinese Wikipedia                                                   | Google     | [chinese_L-12_H-768_A-12_tf],[chinese_L-12_H-768_A-12_pt] | cn   | 12    | 768    | 12   | 102.3M[model-1] |
| Chinese Wikipedia                                                   | Brian Shen | [bert_tywin_12L_cn]                                       | cn   | 12    | 768    | 12   | 102.3M          |
| Chinese Wikipedia                                                   | Brian Shen | [bert_tyrion_12L_cn]                                      | cn   | 12    | 768    | 12   | 102.3M          |
| Chinese Wikipedia, other encyclopedias, news, questions and answers | Brian Shen | [roberta-3L_cn-alpha]                                     | cn   | 3     | 768    | 12   | 38.5M           |
| Middle School Reading Comprehension                                 | Brian Shen | [roberta-3L_cn-beta]                                      | cn   | 3     | 1024   | 16   | 61.0M           |
| Chinese Wikipedia, other encyclopedias, news, questions and answers | Brian Shen | [bert_sansa_12L_cn]                                       | cn   | 12    | 768    | 12   | 102.3M          |
| Chinese comments                                                    | Brian Shen | [bert_eddard_12L_cn_tf],[bert_eddard_12L_cn_pt]           | cn   | 12    | 768    | 12   | 102.3M          |
| Chinese comments                                                    | Brian Shen | [bert_lyarra_12L_cn_tf],[bert_lyarra_12L_cn_pt]           | cn   | 12    | 768    | 12   | 105.1M          |
| Chinese comments                                                    | Brian Shen | [bert_rickard_12L_cn_tf],[bert_rickard_12L_cn_pt]         | cn   | 12    | 768    | 12   | 105.1M          |
| Chinese comments                                                    | Brian Shen | [bert_lyanna_12L_cn_tf],[bert_lyanna_12L_cn_pt]           | cn   | 12    | 768    | 12   | 105.1M          |
| Chinese Wikipedia, other encyclopedias, news, questions and answers | Brian Shen | [bert_24L_cn]                                             | cn   | 24    | 1024   | 16   | 325.5M          |
| QA                                                                  | Brian Shen | [bert_arya_24L_cn]                                        | cn   | 24    | 1024   | 16   | 325.5M          |
| QA                                                                  | Brian Shen | [bert_daenerys_24L_cn]                                    | cn   | 24    | 1024   | 16   | 325.5M          |
| news[corpus-4]                                                      | Brian Shen | [bert_night-king_36L_cn_tf],[bert_night-king_36L_cn_pt]   | cn   | 36    | 1024   | 16   | 476.7M          |
| English Text                                                        | Brian Shen | [stsb_drogo_L-12_H-768_A-12]                              | en     | 12    | 768    | 12   | 109.5M          |
| English Text                                                        | Brian Shen | [stsb_L-12_H-768_A-12]                                    | en     | 12    | 768    | 12   | 124.6M          |
| English Text                                                        | Brian Shen | [stsb_L-24_H-1024_A-16]                                   | en     | 24    | 1024   | 16   | 355.3M          |
| Multi-Lingual Text                                                  | Brian Shen | [stsb-multi_L-12_H-768_A-12]                              | global | 12    | 768    | 12   | 278M          |


> base : 12-layer, 768-hidden, 12-heads, 102.3M parameters 
> large : 24-layer, 1024-hidden, 16-heads, 325.5M parameters 
> giant : 36-layer, 1024-hidden, 16-heads, 476.7M parameters

> [corpus-1] General data includes: questions and answers and other data, the total size is 12.5MB, the number of records is 10,000, and the number of words is 72,000. 
> [corpus-2] When loading pytorch and tf2 models, if transformers load xla error, please modify the value of xla_device in config.json by yourself . If you fine-tune on gpu, you need to set it to false. If you fine-tune on tpu, you need to set it to true. 
> [corpus-3] news corpus: 5000 pieces of 2021 news, about 13MB in size. 
> [corpus-4] News corpus: multiple news articles in 2021, about 200GB in size. 
> [model-1] Chinese-Bert-Base: The parameter volume of Chinese BERT-Base is calculated to be 102.3M, while the parameter volume of Google English BERT-Base is 110M. The difference should be caused by the inconsistent number of vocabulary. Statistics script count.py .


[bert_tiny_cn_tf]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/tf1/chinese_L-2_H-128_A-2.zip
[bert_tiny_cn_pt]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/pt/chinese_L-2_H-128_A-2.zip
[bert_mini_cn_tf]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/tf1/chinese_L-4_H-256_A-4.zip
[bert_mini_cn_pt]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/pt/chinese_L-4_H-256_A-4.zip
[bert_2L_cn]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/pt/bert_L-2_H-768_A-4_cn.zip
[bert_6L_cn]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/pt/bert_L-6_H-768_A-12_cn.zip
[chinese_L-12_H-768_A-12_tf]: https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
[chinese_L-12_H-768_A-12_pt]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/pt/chinese_L-12_H-768_A-12.tgz
[bert_tywin_12L_cn]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/pt/bert_tywin_L-12_H-768_A-12_cn.zip
[bert_tyrion_12L_cn]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/pt/bert_tyrion_L-12_H-768_A-12_cn.zip
[roberta-3L_cn-alpha]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/pt/roberta_L-3_H-768_A-12_cn.zip
[roberta-3L_cn-beta]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/pt/roberta_L-3_H-1024_A-16_cn.zip
[bert_sansa_12L_cn]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/pt/roberta_sansa_L-12_H-768_A-12_cn.zip
[bert_eddard_12L_cn_tf]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/tf1/roberta_eddard_L-12_H-768_A-12_cn.zip
[bert_eddard_12L_cn_pt]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/pt/roberta_eddard_L-12_H-768_A-12_cn.zip
[bert_lyarra_12L_cn_tf]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/tf1/roberta_lyarra_L-12_H-768_A-12_cn_tf.zip
[bert_lyarra_12L_cn_pt]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/pt/roberta_lyarra_L-12_H-768_A-12_cn.zip
[bert_rickard_12L_cn_tf]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/tf1/roberta_rickard_L-12_H-768_A-12_cn_tf.zip
[bert_rickard_12L_cn_pt]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/pt/roberta_rickard_L-12_H-768_A-12_cn.zip
[bert_lyanna_12L_cn_tf]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/tf1/roberta_lyanna_L-12_H-768_A-12_cn_tf.zip
[bert_lyanna_12L_cn_pt]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/pt/roberta_lyanna_L-12_H-768_A-12_cn.zip
[bert_24L_cn]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/pt/roberta_L-24_H-1024_A-16_cn.zip
[bert_arya_24L_cn]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/pt/roberta_arya_L-24_H-1024_A-16_cn.zip
[bert_daenerys_24L_cn]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/pt/roberta_daenerys_L-24_H-1024_A-16_cn.zip
[bert_night-king_36L_cn_tf]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/tf1/roberta_night-king_L-36_H-1024_A-16_cn_tf.zip
[bert_night-king_36L_cn_pt]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/pt/roberta_night-king_L-36_H-1024_A-16_cn.zip
[stsb_drogo_L-12_H-768_A-12]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/pt/stsb_drogo_L-12_H-768_A-12.zip
[stsb_L-12_H-768_A-12]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/pt/stsb_L-12_H-768_A-12.zip
[stsb_L-24_H-1024_A-16]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/pt/stsb_L-24_H-1024_A-16.zip
[stsb-multi_L-12_H-768_A-12]: https://transformers-models.obs.cn-north-4.myhuaweicloud.com/bert/cn/pretrain/pt/stsb-multi_L-12_H-768_A-12.zip

### PyTorch/Tensorflow Versions

Some provide Tensorflow and Pytorch versions, and some only provide PyTorch versions.


### Instructions for Use

The bert_12L_cn model file size is about 454M and 1.3G .
The TensorFlow version is:
```
tf_chinese_BERT_base_L-12_H-768_A-12.zip
    |- bert_config.json                                    # Model parameters
    |- bert_model.ckpt.data-00000-of-00001          # Model weights
    |- bert_model.ckpt.index                        # Model Index
    |- bert_model.ckpt.data                         # Model meta
    |- vocab.txt                                            # Word segmentation vocabulary
```

Pytorch Version：

```
chinese_BERT_base_L-12_H-768_A-12.zip
    |- pytorch_model.bin     # Model weights
    |- config.json           # Model parameters
    |- vocab.txt             # Word segmentation vocabulary
```

`stsb`模型需要使用`sentence_transformers`库加载，先`pip install sentence_transformers`安装后使用。  
```python
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('stsb_L-24_H-1024_A-16')
embeddings = model.encode(sentences)
print(embeddings)
```


###Fast loading
Relying on [hugging face transformers 3.1.0](https://github.com/huggingface/transformers) The above models can be easily called.
```
tokenizer =  AutoTokenizer.from_pretrained("MODEL_NAME")
model =  AutoModel.from_pretrained("MODEL_NAME")

or

tokenizer = BertTokenizer.from_pretrained("MODEL_NAME")
model = BertModel.from_pretrained("MODEL_NAME")
```


## Baseline_System_Effects
In order to compare the baseline performance, we tested on the following English datasets. Compared with English BERT-Tiny, Chinese BERT-Tiny, Chinese BERT-Mini, Chinese BERT-wwm-ext, BERT-base and bert_12L_cn of this project.


| Model        | Score |  CoLA  | SST-2 |  MRPC   | STS-B |  QQP  |MNLI-m| MNLI-mm |QNLI(v2)|  RTE  | WNLI  |
|--------------|:-----:|:------:|:-----:|:-------:|:-----:|:-----:|:----:|:-------:|:------:|:-----:|:-----:|
| BERT-Tiny[1] | 65.13 | 69.12  | 79.12 |  70.34  | 42.73 | 79.81 |64.60 |  66.47  | 77.63  | 59.21 | 42.25 |
| BERT-Mini[1] | 65.93 | 69.12	 | 83.60 | 	72.79  | 45.27 | 76.01 |71.42 |  56.06  | 83.21  | 61.01 | 40.85 |
| BERT-Tiny-CN | 56.00 | 69.12  | 71.10 | 	68.38  | 24.33 | 73.79 |49.23 |  49.79  | 59.30  | 51.26 | 43.66 |
| BERT-Mini-CN | 58.70 | 69.12	 | 75.91 | 	68.38  | 25.40 | 76.09 |55.24 |  55.09  | 56.33  | 49.10 | 56.34 |

> [1] This is Google's BERT-Tiny/Mini model. On the GLUE test data set, the CoLA score is 0. This evaluation uses the same script to re-evaluate all models to compare the results.

We use the same training parameters for one round of training for each task, and other parameters are as follows:   

* max seq length: 128
* batch size: 4 
* learning rate: 2e-5

In conclusion, the Chinese BERT-Tiny/Mini trained using the news corpus [corpus-3] has achieved competitive results compared with the BERT-Tiny/Mini given by Google on the GLUE dataset.   
Compared with Google BERT-Tiny/Mini: Chinese BERT-Tiny(-9.13%/-9.93%), Chinese BERT-Mini(-6.43%/-7.23%); since these two models are trained on Chinese corpus It has achieved this effect in the English GLUE evaluation, which proves the ability of the model in English tasks. Analyze why the Chinese model can be fine-tuned on English tasks, because the Chinese model uses a vocabulary of 21K words from Google's official Chinese model, which contains a large number of common English words, so it has the potential to represent English text. However, since this model is not designed for English, its representation ability is 9.13%/9.93% and 6.43%/7.23% worse than the Google BERT-Tiny model, respectively.


| Model        | Score | SQUAD 1.1 | SQUAD-2 |
|--------------|:-----:|:---------:|:-------:|
| BERT-Tiny    | 45.27 |   39.88   |  50.66  |
| BERT-Mini    | 64.03 |   68.58   |  59.47  |
| BERT-Tiny-CN | 29.78 |   9.48    |  50.07  |
| BERT-Mini-CN | 31.76 |  13.45    |  50.06  |

We use the same training parameters for two rounds of training for each task, and the other parameters are as follows：  
* max seq length: 384
* batch size: 12
* learning rate: 3e-5
* doc stride: 128

In conclusion, since BERT-Tiny/Mini-CN is trained on Chinese corpus, it is 15%-33% worse than Google's BERT-Tiny/Mini in English reading comprehension/question answering tasks.


## Pre-trained_Word_Segmentation
TBERT-Tiny-CN and BERT-Mini-CN use Chinese word segmentation without lowercase conversion.
According to the details of the word MASK and other models, the bert_12L_cn model is taken as an example to illustrate the pre-training details.


### generate vocabulary
According to the official BERT tutorial steps, you first need to use Word Piece to generate a vocabulary. WordPiece is a subword tokenization algorithm for BERT, DistilBERT, and Electra. The algorithm, outlined in Japanese and Korean Speech Search (Schuster et al., 2012), is very similar to BPE.   
WordPiece first initializes the vocabulary to contain each character in the training data, and gradually learns a given number of merging rules. Unlike BPE, WordPiece does not choose the most frequent symbol pair, but the symbol pair that maximizes the likelihood of adding the training data to the vocabulary. So what exactly does this mean? Referring to the previous example, maximizing the likelihood of the training data is equivalent to finding the pair of symbols whose probability divided by the probability of its first symbol and then divided by the probability of its second symbol is the greatest of all symbol pairs. E. "u" followed by "g" will only merge if the probability of "ug" divided by "u", "g" is greater than any other symbol pair.   
Intuitively, WordPiece is slightly different from BPE in that it evaluates what it has lost by merging two symbols to make sure it is worth it.  
In this project, the vocabulary size we use is 21128, and the rest of the parameters adopt the default configuration in the official example.  


```
# Initialize a tokenizer
tokenizer = BertWordPieceTokenizer()
# Customize training
tokenizer.train(files=paths, vocab_size=21_128, min_frequency=0, special_tokens=[
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "<S>",
                "<T>"
                ])

```

### Algorithm of generating vocabulary
The following methods are not implementations of tokenizers, as expressed.   
Furthermore, for an English word (Chinese word segmentation is the same), according to the WP rule, it can be divided into multiple high-frequency segments. The sample code is as follows:

```
def tokenize(self, text):
  
  # Cut a paragraph into word piece. This is actually the greedy maximum forward matching algorithm.
  # eg：
  # input = "unaffable"
  # output = ["un", "##aff", "##able"]
 
  
  text = convert_to_unicode(text)
  
  output_tokens = []
  for token in whitespace_tokenize(text):
	  chars = list(token)
	  if len(chars) > self.max_input_chars_per_word:
		  output_tokens.append(self.unk_token)
		  continue
	  
	  is_bad = False
	  start = 0
	  sub_tokens = []
	  while start < len(chars):
		  end = len(chars)
		  cur_substr = None
		  while start < end:
			  substr = "".join(chars[start:end])
			  if start > 0:
				  substr = "##" + substr
			  if substr in self.vocab:
				  cur_substr = substr
				  break
			  end -= 1
		  if cur_substr is None:
			  is_bad = True
			  break
		  sub_tokens.append(cur_substr)
		  start = end
	  
	  if is_bad:
		  output_tokens.append(self.unk_token)
	  else:
		  output_tokens.extend(sub_tokens)
  return output_tokens
```

### Pretraining
The training parameters of the BERT-Tiny-CN and BERT-Mini-CN models are: * train_batch_size: 32 * max_seq_length: 128 * max_predictions_per_seq: 20 * num_train_steps: 100000 * num_warmup_steps: 5000 * learning_rate: 2e - 5
The training results are as follows:   
* BERT-Tiny: masked_lm_accuracy=22.74%, NSP_accuracy=100%. 
* BERT-Mini: masked_lm_accuracy=33.54%, NSP_accuracy=100%.

After obtaining the above data, as of February 6, 2021, use the WordPiece vocabulary (model) of BERT-wwm-ext (the WordPiece model based on general data will be used in the future), and officially start pre-training BERT.  
The reason why it is called bert_12L_cn is because only compared with BERT-wwm-ext , the rest of the parameters have not changed, mainly because the computing equipment is limited.  
The command used is as follows:

```
    from transformers import (
        CONFIG_MAPPING,
        MODEL_WITH_LM_HEAD_MAPPING,
        AutoConfig,
        BertConfig,
        RobertaConfig,
        AutoModelWithLMHead,
        BertForMaskedLM,
        RobertaForMaskedLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        HfArgumentParser,
        LineByLineTextDataset,
        PreTrainedTokenizer,
        TextDataset,
        Trainer,
        TrainingArguments,
        set_seed,
        BertTokenizer
    )
   
    TEMP="temp/"
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config_file', default='config/bert_config.json',
        help='model config file')
    
    args = parser.parse_args()
    with open(args.config_file) as fin:
        config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))
    
    bert_config = BertConfig.from_pretrained(config.bert_model_path, cache_dir=TEMP)
    
    WRAPPED_MODEL = BertForMaskedLM.from_pretrained(
                config.bert_model_path,
                from_tf=False,
                config=bert_config,
                cache_dir=TEMP,
            )
    for param in WRAPPED_MODEL.parameters():
        param.requires_grad = True
    WRAPPED_MODEL.train()
    
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_path)
    WRAPPED_MODEL.resize_token_embeddings(len(tokenizer))
    
    print("dataset maxl:", config.max_seq_len)
    
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=config.train_file_path,
        block_size=config.max_seq_len,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    
    """### Finally, we are all set to initialize our Trainer"""
    training_args = TrainingArguments(
        output_dir=TEMP,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=config.batch_size,
        save_steps=10_000,
        save_total_limit=2,
        tpu_num_cores=8,
    )
    
    trainer = Trainer(
        model=WRAPPED_MODEL,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        prediction_loss_only=True,
    )
    
    
    trainer.train(model_path=config.bert_model_path)
    WRAPPED_MODEL.to('cpu')
    trainer.save_model(output_dir=config.trained_model_path)
    torch.save(WRAPPED_MODEL.state_dict(), os.path.join(config.trained_model_path, 'pytorch_model.bin'))

```


## Downstream_task_fine-tuning_details

The device used for downstream task fine-tuning is Google Cloud GPU (16G HBM).  
The configuration of each task fine-tuning is briefly described below. See the project for the relevant code.



## FAQ

**Q: How to use this model?**  
A: How to use the Chinese BERT released by Google, this is how to use it. The text does not need to be segmented, and wwm only affects the pre-training process and does not affect the input of downstream tasks.

**Q: Is there any pre-training code provided?**  
A: Unfortunately, I can't provide the relevant code, please refer to #10 and #13 for implementation .

**Q: Where can I download the XX dataset?**  
A: Please check the data directory. The README.md under the task directory indicates the data source. For copyrighted content, please search by yourself or contact the original author to obtain the data.

**Q: Will there be plans to release a larger model? Such as BERT-large-wwm version?**  
A: If we get better results from the experiment, we will consider releasing a larger version.

**Q: You lied! Unable to reproduce the results 😂**  
A: In the downstream task, we adopted the simplest model. For example, for classification tasks, we directly use run_classifier.py (provided by Google). If the average value cannot be reached, it means that there is a bug in the experiment itself, please check carefully. There are many random factors in the highest value, and we cannot guarantee that the highest value will be achieved. Another recognized factor: reducing the batch size will significantly reduce the experimental effect. For details, please refer to the relevant Issues in the BERT and XLNet directories.

**Q: I trained better results than you!**  
A: Congratulations.

**Q: How long did the training take, and what equipment did you train on?**  
A: The training is done on the Google TPU v3 version (128G HBM). It takes about 4 hours to train BERT-wwm-base, and about 8 hours to train BERT-wwm-large.

**Q: The effect of BERT-wwm is not good for all tasks**  
A: The purpose of this project is to provide researchers with a variety of pre-training models, free to choose BERT, ERNIE, or BERT-wwm. We only provide experimental data, and we still have to keep trying in our own tasks to draw conclusions about the specific effect. One more model, one more choice.

**Q: Why are some data sets not tested?**  
A: Frankly speaking: 1) I don’t have the energy to find more data; 2) It’s unnecessary; 3) I don’t have money;

**Q: Briefly evaluate these models.**  
A: Each has its own focus and strengths. The research and development of Chinese natural language processing requires the joint efforts of many parties.

**Q: More details about the RoBERTa-wwm-ext model?**  
A: We integrated the advantages of RoBERTa and BERT-wwm, and made a natural combination of the two. The differences from the previous models in this catalog are as follows: 
>1) The wwm strategy is used for masking in the pre-training phase (but dynamic masking is not used) 
>2) Next Sentence Prediction (NSP) loss is simply canceled 
>3) Max_len=128 is no longer used Then use max_len=512 training mode, directly train max_len=512 
>4) Properly extend the number of training steps  

It should be noted that this model is not the original RoBERTa model, but a BERT model trained in a RoBERTa-like training method, that is, RoBERTa-like BERT. Therefore, when using downstream tasks and converting models, please use BERT instead of RoBERTa.



## Citation
If the contents in this catalogue are helpful to your research work, you are welcome to quote the following technical reports in the paper:


## Thank you
Project Author: Brian Shen. Twitter@dezhou. Please follow my Twitter, thank you!
During the construction of the project, the following warehouses have been referred to. Thank you here:
- BERT：https://github.com/google-research/bert
- Chinese pretrained model：https://github.com/ymcui/Chinese-BERT-WWM

## Disclaimer
This project is not [Bert official](https://github.com/google-research/bert) Published Chinese BERT model.
The content of the project is only for technical research reference, not as any conclusive basis.
Users can use the model freely within the scope of the license, but we are not responsible for the direct or indirect losses caused by using the content of the project.

## Follow us
Welcome to Zhihu column.
[Deep Learning Interest Group](https://www.zhihu.com/column/thuil)

## Problem feedback &amp; contribution
If you have any problems, please submit them in GitHub Issue. 
We do not operate, and encourage netizens to help each other solve problems. 
If you find implementation problems or are willing to jointly build the project, please submit a Pull Request.

