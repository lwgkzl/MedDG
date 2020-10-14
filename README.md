# MedDG
a large-scale high-quality medical dialogue dataset



Installation：

pip install allenlp==0.9

Running :

For Task Topic Predict  (run lstm baseline for example):
cd topic_predict
allennlp train ../config/topic_predict/lstm_baseline.json --include-package basline -s tmp/lstm1

For Task generation (run seq2seq baseline for example):
cd generation
allennlp train ../config/generation/seq2seqattn.json --include-package seq2seq -s tmp/seq2seq1
