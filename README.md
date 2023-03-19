# MedDG

This is the code for the following paper:

[MedDG: an entity-centric medical consultation dataset for entity-aware medical dialogue generation](https://link.springer.com/chapter/10.1007/978-3-031-17120-8_35)

*Wenge Liu, Jianheng Tang, Yi Cheng, Wenjie Li, Yefeng Zheng, Xiaodan Liang;*


### Requirement

    pip install allenlp==0.9

### Usage

For the task of topic prediction  (e.g. run the LSTM baseline):

```shell
cd topic_predict
allennlp train ../config/topic_predict/lstm_baseline.json --include-package basline -s tmp/lstm1
```

For the task of response generation (e.g. run the seq2seq baseline):

```shell
cd generation
allennlp train ../config/generation/seq2seqattn.json --include-package seq2seq -s tmp/seq2seq1
```

### Dataset
https://drive.google.com/drive/folders/109WnXlNhmqttxYwb4EEscYGcA_-eq8Eg?usp=sharing
