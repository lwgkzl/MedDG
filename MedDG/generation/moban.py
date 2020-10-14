import pickle
from tqdm import tqdm

with open('../data/0831/moban_dic.pk','rb') as f:
    moban_dic = pickle.load(f)

with open('../data/0831/gen_test_with_bert.pk','rb') as f:
    test_dataset = pickle.load(f)

def find_similary_key_subset(x):
    xx = x.split('.')
    most_key = 0
    ans = "无"
    for k in moban_dic.keys():
        aa = k.split('.')  # subset
        flag = 0
        for j in aa:
            if j not in xx:
                flag = 1
        if flag == 0:
            if len(aa) > most_key:
                most_key = len(aa)
                ans = k
    return ans

def find_similary_key(x):
    xx = x.split('.')
    most_key = 0
    ans = "无"
    jj = [i for i in moban_dic.keys()]
    random.shuffle(jj)
    for k in jj:
        aa = k.split('.')  # subset
        flag = 0
        for j in xx:
            if j not in aa:
                flag = 1
        if flag == 0 and len(aa) > most_key:
            most_key = len(aa)
            ans = k
    return ans




import random
from CY_DataReadandMetric import *
num = 0
kdm = KD_Metric()
bleu1 = NLTK_BLEU()
bleu4 = NLTK_BLEU(ngram_weights=(0, 0, 0, 1))
dist1 = Distinct1()
dist2 = Distinct2()

with open('../data/0831/160_last_topic2num.pk', 'rb') as f:
    topic2idx = pickle.load(f)

moban_ans = []
for dic in tqdm(test_dataset):
    bert_pre = dic['bert_word']
    key = '无'
    if len(bert_pre) > 0:
        xx = sorted(bert_pre)
        x = '.'.join(xx)
        if x not in moban_dic.keys():
            num += 1
            x = find_similary_key(x)
        key = x
    # gen = random.choice(list(moban_dic[key]))
    gen = random.choice(list(moban_dic[key]))
    # gen = dic['response']
    kdm([dic['response']], [gen])
    bleu1([dic['response']], [gen])
    bleu4([dic['response']], [gen])
    dist1([gen])
    dist2([gen])
    moban_ans.append(gen)
ans = {}
ans.update(kdm.get_metric(reset=False))
ans.update({"bleu1": bleu1.get_metric(reset=False)})
ans.update({"bleu4": bleu4.get_metric(reset=False)})
ans.update({"dist1": dist1.get_metric(reset=False)})
ans.update({"dist2": dist2.get_metric(reset=False)})
with open('moban_result.pk','wb') as f:
    pickle.dump(ans, f)
with open('moban_ans.pk','wb') as f:
    pickle.dump(moban_ans, f)
print(ans)

# len(moban_ans)