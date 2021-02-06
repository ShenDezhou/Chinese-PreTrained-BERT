[**中文说明**](./readme.md) | [**English**](./readme_en.md)

<p align="center">
    <img src="./pics/banner.svg" width="500"/>
</p>
<p align="center">
    <a href="https://github.com/shendezhou/Chinese-PreTrained-BERT/blob/master/LICENSE">
        <img alt="LICENCE" src="" />
    </a>
</p>

本项目提供了面向中文的BERT预训练模型，旨在丰富中文自然语言处理资源，提供多元化的中文预训练模型选择。
我们欢迎各位专家学者下载使用，并共同促进和发展中文资源建设。

本项目基于谷歌官方BERT：https://github.com/google-research/bert

其他相关资源：
- 中文BERT预训练模型：https://github.com/ymcui/Chinese-BERT-wwm

查看更多发布的资源：https://github.com/ 

## 新闻
**2021/2/6 所有模型已支持Pytorch和Tensorflow1以及Tensorflow2，请通过transformers库进行调用或下载。https://huggingface.co/**


<details>
<summary>历史新闻</summary>
2021/2/6 本目录发布的模型未来可接入[Huggingface-Transformers](https://github.com/huggingface/transformers)，查看[快速加载](#快速加载)

2021/2/6 `BERT-wwm-base-qa`已可下载，查看[模型下载](#模型下载)

2021/2/6 提供了在小规模通用语料（12.5MB）上训练的中文`BERT-wwm-base-qa`模型，查看[模型下载](#模型下载)
</details>

## 内容导引
| 章节 | 描述 |
|-|-|
| [简介](#简介) | 介绍BERT-wwm基本原理 |
| [模型下载](#模型下载) | 提供了中文预训练BERT下载地址 |
| [基线系统效果](#基线系统效果) | 列举了部分基线系统效果 |
| [预训练细节](#预训练细节) | 预训练细节的相关描述 |
| [下游任务微调细节](#下游任务微调细节) | 下游任务微调细节的相关描述 |
| [FAQ](#faq) | 常见问题答疑 |
| [引用](#引用) | 本目录的技术报告 |

## 简介
**Whole Word Masking (wwm)**，暂翻译为`全词Mask`或`整词Mask`，是谷歌在2019年5月31日发布的一项BERT的升级版本，主要更改了原预训练阶段的训练样本生成策略。
简单来说，原有基于WordPiece的分词方式会把一个完整的词切分成若干个子词，在生成训练样本时，这些被分开的子词会随机被mask。
在`全词Mask`中，如果一个完整的词的部分WordPiece子词被mask，则同属该词的其他部分也会被mask，即`全词Mask`。

**需要注意的是，这里的mask指的是广义的mask（替换成[MASK]；保持原词汇；随机替换成另外一个词），并非只局限于单词替换成`[MASK]`标签的情况。
更详细的说明及样例请参考：[#4](https://github.com/ymcui/Chinese-BERT-wwm/issues/4)**

同理，由于谷歌官方发布的`BERT-base, Chinese`中，中文是以**字**为粒度进行切分，没有考虑到传统NLP中的中文分词（CWS）。
我们将全词Mask的方法应用在了中文中，使用了中文维基百科（包括简体和繁体）进行训练，并且使用了分词工具，即对组成同一个**词**的汉字全部进行Mask。

下述文本展示了`全词Mask`的生成样例。
**注意：为了方便理解，下述例子中只考虑替换成[MASK]标签的情况。**

| 说明 | 样例 |
| :------- | :--------- |
| 原始文本 | 使用语言模型来预测下一个词的probability。 |
| 分词文本 | 使用 语言 模型 来 预测 下 一个 词 的 probability 。 |
| 原始Mask输入 | 使 用 语 言 [MASK] 型 来 [MASK] 测 下 一 个 词 的 pro [MASK] ##lity 。 |
| 全词Mask输入 | 使 用 语 言 [MASK] [MASK] 来 [MASK] [MASK] 下 一 个 词 的 [MASK] [MASK] [MASK] 。 |



## 模型下载
* **`BERT-wwm-base-qa`**：12-layer, 768-hidden, 12-heads, 110M parameters()
* **`BERT-wwm-large-qa`**：24-layer, 768-hidden, 16-heads, 330M parameters()

| 模型简称 | 语料 | Google下载 | 百度云下载[待传] |
| :------- | :--------- | :---------: | :---------: |
| **`BERT-wwm-base-qa, Chinese`** | **中文问答/<br/>通用数据<sup>[1][2]</sup>** | **[TensorFlow1](https://drive.google.com/drive/folders/11B-bU8I2BUvtel4100GkqmeprWPvkI8R?usp=sharing)** <br/>**[TensorFlow2](https://drive.google.com/drive/folders/1-yx_t3u5gAuxlrmVQnkv8msKlduX3HOh?usp=sharing)** <br/>**[PyTorch](https://drive.google.com/drive/folders/1-wnz-EC9znkZz-DW40Mhisr4PaejtctC?usp=sharing)** | **[TensorFlow1,密码:tfxl]()** <br/>**[TensorFlow2,密码:tfxl]()** <br/>**[PyTorch,密码:toxl]()** |
| **`BERT-wwm-large-qa, Chinese`** | **中文问答/<br/>通用数据<sup>[1][3]</sup>** | **[TensorFlow1](https://drive.google.com/drive/folders/11JvzcJvuhxYbVSNWUutC57-zP2Y1gGV5?usp=sharing)** <br/>**[TensorFlow2](https://drive.google.com/drive/folders/10COzeCg9AUCLWzIyPhkEC6Mpm2Od2OnP?usp=sharing)** <br/>**[PyTorch](https://drive.google.com/drive/folders/1-2U4oaOpbb5nlUOUBMdurxyoO7qMBBxf?usp=sharing)** | **[TensorFlow1,密码:tfxl]()** <br/>**[TensorFlow2,密码:tfxl]()** <br/>**[PyTorch,密码:toxl]()** |
| **`BERT-wwm-large-qa, Chinese`** | **中文问答/<br/>通用数据<sup>[1][4]</sup>** | **[TensorFlow1](https://drive.google.com/drive/folders/11U8Pd9hRaT3MEe7J6V05RC8_QMfPPccc?usp=sharing)** <br/>**[TensorFlow2](https://drive.google.com/drive/folders/10X8fxslV0Oo5jEbkK9tNGzmqpKT86xT0?usp=sharing)** <br/>**[PyTorch](https://drive.google.com/drive/folders/11Pd5sy2jRePz5AVlOdImRYNDSbCtwtAS?usp=sharing)** | **[TensorFlow1,密码:tfxl]()** <br/>**[TensorFlow2,密码:tfxl]()** <br/>**[PyTorch,密码:toxl]()** |

> [1] 通用数据包括：问答等数据，总大小12.5MB，记录数1万，字数7.2万。
> [2] 训练1轮，最大串长512，batch=2.
> [3] 训练1轮，最大串长128，batch=8.
> [4] 最大串长128，batch=8，训练1轮后；用最大串长256，batch=4，再训练一轮。

### PyTorch/Tensorflow版本

提供PyTorch版本，TF1和TF2版本。

### 使用说明

中国大陆境内建议使用百度云下载点，境外用户建议使用谷歌下载点，`BERT-wwm-base-qa`模型文件大小约**454M**和**1.3G**。 以TensorFlow版`BERT-wwm-base-qa, Chinese`为例，下载完毕后对zip文件进行解压得到：

```
tf_chinese_BERT_tiny_L-12_H-768_A-12.zip
    |- checkpoint                                           # 存盘点信息
    |- BERT_tiny_chinese.ckpt.data-00000-of-00001          # 模型权重
    |- BERT_tiny_chinese.ckpt.index                        # 模型index信息
    |- BERT_tiny_chinese.ckpt.data                         # 模型meta信息
    |- vocab.txt                                            # 分词词表
```

TensorFlow2版本为：

```
tf2_chinese_BERT_tiny_L-12_H-768_A-12.zip
    |- tf_model.h5           # 模型权重
    |- config.json           # 模型参数
    |- vocab.txt             # 分词词表
```

Pytorch版本为：

```
chinese_BERT_tiny_L-12_H-768_A-12.zip
    |- pytorch_model.bin     # 模型权重
    |- config.json           # 模型参数
    |- training_args.bin     # 模型训练信息
    |- vocab.txt             # 分词词表
```


### 快速加载
依托于[Huggingface-Transformers 3.1.0](https://github.com/huggingface/transformers) ，可轻松调用以上模型。
```
tokenizer = AutoTokenizer.from_pretrained("MODEL_NAME")
model = AutoModel.from_pretrained("MODEL_NAME")
```
其中`MODEL_NAME`对应列表如下：  

| 模型名 | MODEL_NAME |
| - | - |
| BERT-wwm-base-qa-Chinese | /chinese-BERT-wwm-base-qa<sup>[1]</sup>|
| BERT-wwm-large-qa-Chinese | /chinese-BERT-wwm-large-qa<sup>[1]</sup>|
> [1] 待上传,暂时需要手动下载。



## 基线系统效果
为了对比基线效果，我们在以下几个中文数据集上进行了测试。对比了中文BERT-wwm-ext、BERT-base以及本项目的BERT-wwm-base-qa。
时间及精力有限，并未能覆盖更多类别的任务，请大家自行尝试。



## 预训练细节
以下以`BERT-wwm-base-qa`模型为例，对预训练细节进行说明。

### 生成词表
按照BERT官方教程步骤，首先需要使用[Word Piece](https://pypi.org/project/tokenizers/) 生成词表。
WordPiece是用于BERT、DistilBERT和Electra的子词标记化算法。该算法在日语和韩语语音搜索（Schuster et al.，2012）中进行了概述，与BPE非常相似。WordPiece首先初始化词汇表以包含训练数据中的每个字符，并逐步学习给定数量的合并规则。与BPE不同的是，WordPiece并没有选择最频繁的符号对，而是选择将训练数据添加到词汇表中的可能性最大化的符号对。
那么这到底意味着什么呢？参照前面的示例，最大化训练数据的可能性相当于找到符号对，其概率除以其第一个符号的概率然后再除以其第二个符号的概率是所有符号对中最大的。E、 只有当“ug”除以“u”、“g”的概率大于任何其他符号对时，“u”后接“g”才会合并。直观地说，WordPiece与BPE稍有不同，它通过合并两个符号来评估它所失去的东西，以确保它是值得的。

在本项目中，我们使用的词表大小为21128，其余参数采用官方示例中的默认配置。

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

### 预训练
获得以上数据后，截止2021年2月6日，使用BERT-wwm-ext的WordPiece词表（模型），(未来将使用基于通用数据的WordPiece模型)，正式开始预训练BERT。
之所以叫`BERT-wwm-base-qa`是因为仅相比`BERT-wwm-ext`，其余参数没有变动，主要因为计算设备受限。
使用的命令如下：
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

## 下游任务微调细节
下游任务微调使用的设备是谷歌Cloud GPU（16G HBM），以下简要说明各任务精调时的配置。
**相关代码请查看[TBD]项目。**


## FAQ
**Q: 这个模型怎么用？**  
A: 谷歌发布的中文BERT怎么用，这个就怎么用。
**文本不需要经过分词，wwm只影响预训练过程，不影响下游任务的输入。**

**Q: 请问有预训练代码提供吗？**  
A: 很遗憾，我不能提供相关代码，实现可以参考 [#10](https://github.com/ymcui/Chinese-BERT-wwm/issues/10) 和 [#13](https://github.com/ymcui/Chinese-BERT-wwm/issues/13)。

**Q: 某某数据集在哪里下载？**  
A: 请查看`data`目录，任务目录下的`README.md`标明了数据来源。对于有版权的内容，请自行搜索或与原作者联系获取数据。

**Q: 会有计划发布更大模型吗？比如BERT-large-wwm版本？**  
A: 如果我们从实验中得到更好效果，会考虑发布更大的版本。

**Q: 你骗人！无法复现结果😂**  
A: 在下游任务中，我们采用了最简单的模型。比如分类任务，我们直接使用的是`run_classifier.py`（谷歌提供）。
如果无法达到平均值，说明实验本身存在bug，请仔细排查。
最高值存在很多随机因素，我们无法保证能够达到最高值。
另外一个公认的因素：降低batch size会显著降低实验效果，具体可参考BERT，XLNet目录的相关Issue。

**Q: 我训出来比你更好的结果！**  
A: 恭喜你。

**Q: 训练花了多长时间，在什么设备上训练的？**  
A: 训练是在谷歌TPU v3版本（128G HBM）完成的，训练BERT-wwm花费约1.5天，BERT-wwm-ext则需要数周时间（使用了更多数据需要迭代更充分）。
需要注意的是，预训练阶段我们使用的是`LAMB Optimizer`（[TensorFlow版本实现](https://github.com/ymcui/LAMB_Optimizer_TF)）。该优化器对大的batch有良好的支持。
在微调下游任务时，我们采用的是BERT默认的`AdamWeightDecayOptimizer`。

**Q: ERNIE是谁？**  
A: 本项目中的ERNIE模型特指百度公司提出的[ERNIE](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE)，而非清华大学在ACL 2019上发表的[ERNIE](https://github.com/thunlp/ERNIE)。

**Q: BERT-wwm的效果不是在所有任务都很好**  
A: 本项目的目的是为研究者提供多元化的预训练模型，自由选择BERT，ERNIE，或者是BERT-wwm。
我们仅提供实验数据，具体效果如何还是得在自己的任务中不断尝试才能得出结论。
多一个模型，多一种选择。

**Q: 为什么有些数据集上没有试？**  
A: 很坦率的说：
1）没精力找更多的数据；
2）没有必要； 
3）没有钞票；

**Q: 简单评价一下这几个模型**  
A: 各有侧重，各有千秋。
中文自然语言处理的研究发展需要多方共同努力。

**Q: 你预测下一个预训练模型叫什么？**  
A: 可能叫ZOE吧，ZOE: Zero-shOt Embeddings from language model

**Q: 更多关于`RoBERTa-wwm-ext`模型的细节？**  
A: 我们集成了RoBERTa和BERT-wwm的优点，对两者进行了一个自然的结合。
和之前本目录中的模型之间的区别如下:  
1）预训练阶段采用wwm策略进行mask（但没有使用dynamic masking）  
2）简单取消Next Sentence Prediction（NSP）loss  
3）不再采用先max_len=128然后再max_len=512的训练模式，直接训练max_len=512  
4）训练步数适当延长  

需要注意的是，该模型并非原版RoBERTa模型，只是按照类似RoBERTa训练方式训练出的BERT模型，即RoBERTa-like BERT。
故在下游任务使用、模型转换时请按BERT的方式处理，而非RoBERTa。

**Q: ？**  
A: 。


## 引用
如果本目录中的内容对你的研究工作有所帮助，欢迎在论文中引用下述技术报告：
https://arxiv.org/abs/
```
TBD
```


## 致谢
项目作者： tsinghuaboy

建设该项目过程中参考了如下仓库，在这里表示感谢：
- BERT：https://github.com/google-research/bert
- 中文BERT预训练模型：https://github.com/ymcui/Chinese-BERT-wwm


## 免责声明
本项目并非[BERT官方](https://github.com/google-research/bert) 发布的Chinese BERT模型。
该项目中的内容仅供技术研究参考，不作为任何结论性依据。
使用者可以在许可证范围内任意使用该模型，但我们不对因使用该项目内容造成的直接或间接损失负责。


## 关注我们
欢迎关注知乎专栏号。

[学习兴趣小组](https://www.zhihu.com/column/thuil)


## 问题反馈 & 贡献
如有问题，请在GitHub Issue中提交。  
我们没有运营，鼓励网友互相帮助解决问题。  
如果发现实现上的问题或愿意共同建设该项目，请提交Pull Request。  
