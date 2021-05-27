# MedDG

This is the code for the following paper:

[MedDG: A Large-scale Medical Consultation Dataset for Building Medical Dialogue System](https://arxiv.org/pdf/2010.07497)

*Wenge Liu, Jianheng Tang, Jinghui Qin, Lin Xu, Zhen Li, Xiaodan Liang; Arxiv*


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
