## Dataset address:

https://drive.google.com/file/d/1ASb-iSOt_v1Uqx53Qezt99KXCffRxz2D/view?usp=sharing

## Example
*.pk is raw data used for entity prediction tasks.  
example:  
  [{'id': 'dis',  
    'sentence': '口臭，经常腹痛什么原因（女，34岁）',  
    'sym': ['腹痛'],  
    'medical': [],  
    'check': [],  
    'prop': [],  
    'diease': []},  
   {'id': 'doc',  
    'sentence': '你好，经常腹疼多长时间了，疼距肚脐的什么位置。',  
    'sym': ['腹痛'],  
    'medical': [],  
    'check': [],  
    'prop': ['位置', '时长'],  
    'diease': []}]  


gen*.pk is the data used to generate the task.  
example:  
  [{'history': ['口臭，经常腹痛什么原因（女，34岁）'],  
    'next_sym': [63, 74, 77],  
    'response': '你好，经常腹疼多长时间了，疼距肚脐的什么位置。',  
    'id': 200000,  
    'history_with_topic': ['腹痛:口臭，经常腹痛什么原因（女，34岁）']},  
   {'history': ['口臭，经常腹痛什么原因（女，34岁）',  
     '你好，经常腹疼多长时间了，疼距肚脐的什么位置。',  
     '肚脐周围，坐久了感觉肚子不舒服。'],  
    'next_sym': [],
    'response': '口臭，除口腔卫生和口腔疾病外，全身某方面病也可引起。',  
    'id': 200001,  
    'history_with_topic': ['腹痛:口臭，经常腹痛什么原因（女，34岁）' 
     '腹痛位置时长:你好，经常腹疼多长时间了，疼距肚脐的什么位置。',  
     '胃肠不适:肚脐周围，坐久了感觉肚子不舒服。']}]  


## Tips
The file DataReader.ipynb shows how to extract data from *.pk file.  

Omissions are inevitable in data tagging, if you have found an error, please feel free to contact us by email!  






















