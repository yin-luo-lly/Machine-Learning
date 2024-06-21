计算机学院
课程设计报告 
（ 2023  ~2024 学年度     第 二学期 ）
课程名称	机器学习
设计名称	法律领域篇章级多事件检测
姓名	罗粮银	学号	20224950122
专业	人工智能	班级	本22智能02班
地点	 8-411       	教师	   熊东平             
         
一、课题设计背景
 信息抽取是自然语言处理领域的一项基础任务，涉及事件抽取、命名实体识别等多个子任务。事件检测作为事件抽取的一个子任务，其结果对后续的事件元素抽取任务产生着重要影响。由于一个完整的法律案件通常需要一段包含多个事件的较长文本进行描述，而且其中往往存在触发词不明显或者不包含触发词的事件，因此，常用的基于触发词进行事件检测模型不能很好的发挥作用。随着国家法律体系的细化、完善，司法部门日常需要处理大量的案件信息。为帮助司法办案人员快速理清案件的发展状况，掌握法律案件中包含哪些类型的事件，需要依据真实存在的法律案件信息（人名、时间等信息已重造）利用机器学习等相关技术，建立稳健的事件检测模型，用于判断法律案件中所包含的各个事件对应的事件类型，为后续抽取各事件所涉及的元素提供有利信息。
本设计针对法律案件中存在触发词不明显或者不包含触发词的事件，试图建立稳健的事件检测模型，用于判断法律案件中所包含的各个事件对应的事件类型，进而对后续的事件元素抽取任务提供支持。

二、设计方案概述
1.数据准备
(1) 从数据集构建训练集、验证集和测试集
(2) 使用Pandas进行数据统计分析，包括文本长度统计、标签分布统计等
(3) 使用Seaborn和Matplotlib进行数据可视化分析
2.模型训练
(1) 使用ERNIE-3.0预训练模型作为backbone
(2) 使用二分类交叉熵损失函数
(3) 使用线性学习率衰减策略
(4) 使用FGM进行对抗训练
(5) 评估指标使用AUC和F1-score
3.模型测试
(1) 使用训练得到的最好模型进行测试集预测
(2) 对预测结果进行后处理
(3) 保存预测结果到submit.json文件
4.数据处理
(1) 定义数据预处理函数
(2) 定义数据Batch处理函数
(3) 定义数据后处理函数
5.工具函数
(1) 定义数据集加载函数
(2) 定义模型评估函数
(3) 定义FGM对抗训练函数
(4) 定义预测结果后处理函数

三 、具体实现
 1.准备数据
数据格式如下：
{
"id": 1, 
"text": "赵四与妻子王五通过相亲认识，2011年登记结婚，婚后共生育三个孩子，后双方因感情不和，于2020年协议离婚，协议约定，离婚后，三个孩子在一年内跟随王五生活，赵四每月每个孩子支付2000元抚养费，2021年三个孩子向法院提起诉讼，要求赵四按照协议约定支付抚养费。", 
"classname": "婚姻家庭纠纷", 
"eventchain": [
        {"trigger": "结婚", "eventtype": "Marry", "argument": [{"husband": "赵四", "wife": "王五", "time": "2011年", "loc": ""}]},   
        {"trigger": "生育", "eventtype": "BeBorn", "argument": [{"per": "三个孩子", "time": "婚后", "loc": ""}]},  
        {"trigger": "离婚", "eventtype": "Other", "argument": [{"subjec": "赵四", "object": "王五", "context": "双方感情不和", "time": "2020年", "loc": ""}]}, 
        {"trigger": "诉讼", "eventtype": "Prosecute", "argument": [{"prosecutor": "孩子", "defendant": "赵四", "reason": "", "demand": "按照协议约定支付抚养费", "time": "2021年", "court": "法院"}]}], 
"caseresult": ["双方达成调解，被告同意按照协议约定支付抚养费。"
]
}
由于实际上目标任务是一个多分类任务，因此只保留了数据中id,text,event_type属性（其中event_type用label代替），分别生成训练集、验证集和测试集。
from utils import id_label, data_prepare

datapath = 'Data'

id2label, label2id = id_label(datapath)
train_dataset = data_prepare(datapath, label2id, mode='train')
dev_dataset = data_prepare(datapath, label2id, mode='dev')
test_dataset = data_prepare(datapath, label2id, mode='test')

2.数据探索
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from collections import Counter
from matplotlib.font_manager import FontProperties

（1）查看数据
  df_train = pd.DataFrame.from_dict(train_dataset)
  df_train.head() 
  df_train.info()

（2）文本长度分布
  # 训练集的文本长度描述
  df_train['lenth'] = df_train['text'].apply(lambda x:len(x))
  df_train['lenth'].describe()
  
  # 测试集的文本长度描述
  df_test = pd.DataFrame.from_dict(test_dataset)
  df_test['lenth'] = df_test['text'].apply(lambda x:len(x))
  df_test['lenth'].describe()
  fig, ax = plt.subplots(1,1,figsize=(12,4))

  ax = plt.hist(x=df_train['lenth'], bins=100)
  ax = plt.hist(x=df_test['lenth'], bins=100)

  plt.xlim([0, max(max(df_train['lenth']), max(df_test['lenth']))])
  plt.xlabel("length of sample")
  plt.ylabel("number of sample")
  plt.legend(['train_len','test_len'])

  plt.show()
  
  plt.figure(figsize=(12,4))
  ax = sns.distplot(df_train['lenth'], bins=100)
  ax = sns.distplot(df_test['lenth'], bins=100)
  plt.xlim([0, max(max(df_train['lenth']), max(df_test['lenth']))])
  plt.xlabel("length of sample")
  plt.ylabel("prob of sample")
  plt.legend(['train_len','test_len'])
  
  import scipy
  scipy.stats.ks_2samp(df_train['lenth'], df_test['lenth'])
  
（3）截断位置选择
     log_train_len = np.log(1+df_train['lenth'])
     log_test_len = np.log(1+df_test['lenth'])
     _, lognormal_ks_pvalue = scipy.stats.kstest(rvs=log_train_len, cdf='norm')
    print(lognormal_ks_pvalue)
    trans_data, lam = scipy.stats.boxcox(df_train['lenth']+1)
    print(scipy.stats.normaltest(trans_data))
    
    plt.figure(figsize=(12,4))
    ax = sns.distplot(log_train_len)
    ax = sns.distplot(log_test_len)
    plt.xlabel("log length of sample")
    plt.ylabel("prob of log")
    plt.legend(['train_len','test_len'])
   
（4）类别分析
  num_label = []
  for j in range(len(id2label)):
      m = 0
      for i in range(len(df_train['labels'])):
         m += df_train['labels'][i][j]
      num_label.append(m)
      
    objs = [df_train[['id', 'lenth']], pd.DataFrame(df_train['labels'].tolist())]
    ans1 = pd.concat(objs, axis=1)
    ans2 = pd.melt(ans1, var_name='id_label', value_name='labels', id_vars=['id', 'lenth'])
    ans2 = ans2[ans2['labels']!=0.0].reset_index(drop=True).drop('labels', axis=1)

   plt.figure()
   ax = sns.catplot(x='id_label', y='lenth', data=ans2, kind='strip')
   plt.xticks(range(len(id2label)), list(id2label.values()), rotation=45)
   
3.事件检测分类
（1）从本地文件创建数据集
  from paddlenlp.datasets import load_dataset
  from utils import id_label, data_prepare, data_split

  datapath = 'Data'

  id2label, label2id = id_label(datapath)
  train_dataset = data_prepare(datapath, label2id, mode='train')
  test_dataset = data_prepare(datapath, label2id, mode='dev')
 
  dataset = train_dataset + test_dataset # 训练集和验证集共同组成新的训练集
  train_dataset, test_dataset = data_split(dataset) # 划分数据集

  def read(dataset):
      for data in dataset:
          text, labels = data['text'], data['labels']
          yield {'text': text, 'labels': labels}

  train_dataset = load_dataset(read, dataset=train_dataset,lazy=False) # dataset是read的参数
  test_dataset = load_dataset(read, dataset=test_dataset,lazy=False)

  print(id2label, '\n\n', label2id, '\n\n')
  print("训练集样例:", train_dataset[291])
  print("测试集样例:", test_dataset[291])

（2)将文本数据处理成模型可以接受的 feature
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.data import Stack, Tuple, Pad, Dict
from functools import partial

model_name = "ernie-3.0-base-zh"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def convert_example(example, tokenizer, max_seq_length=640, is_test=False): # 数据预处理函数
    
    # tokenizer.encode方法能够完成切分token，映射token ID以及拼接特殊token
    tokenized_example = tokenizer.encode(text=example['text'], max_seq_len=max_seq_length, truncation=True)
    if not is_test:
        tokenized_example['labels'] = example['labels'] # 加上labels用于训练
    else:
        tokenized_example['ids'] = example['id']

    return tokenized_example

trans_func = partial( # 给convert_example传入参数
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=640,
    is_test=False
    )

train_dataset = train_dataset.map(trans_func)
test_dataset = test_dataset.map(trans_func)

print(train_dataset[0])
print(test_dataset[0])

（3）组成batch  
# collate_fn函数构造，将不同长度序列充到批中数据的最大长度，再将数据堆叠

collate_fn = lambda samples, fn=Dict({
    'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id),
    'token_type_ids': Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
    'labels': Stack(dtype="float32")
}): fn(samples)

from paddle.io import DataLoader, BatchSampler

train_batch_sampler = BatchSampler(train_dataset, batch_size=16, shuffle=True)
train_data_loader = DataLoader(dataset=train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate_fn)

test_batch_sampler = BatchSampler(test_dataset, batch_size=16, shuffle=False)
test_data_loader = DataLoader(dataset=test_dataset, batch_sampler=test_batch_sampler, collate_fn=collate_fn)
（4）定义模型网络和损失函数
from metric import MultiLabelReport
import paddle

max_steps = -1
epochs = 10
learning_rate = 2e-5
warmup_steps = 1000
id2label, label2id = id_label(datapath)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_classes=len(id2label))

num_training_steps = max_steps if max_steps > 0 else len(train_data_loader) * epochs

lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_steps)

# 梯度裁剪
clip = paddle.nn.ClipGradByNorm(clip_norm=1.0)

# 生成执行权重衰减所需的参数名称，所有偏差和分层参数被被排除
decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]

optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    epsilon=1e-8,
    parameters=model.parameters(),
    grad_clip=clip,
    weight_decay=0.0,
    apply_decay_param_fun=lambda x: x in decay_params)

criterion = paddle.nn.BCEWithLogitsLoss()
metric = MultiLabelReport()

（5）开始训练
import time
from eval import evaluate
import paddle.nn.functional as F

ckpt_dir = "ernie_ckpt" # 训练过程中保存模型参数的文件夹
def train(epochs, save_dir=ckpt_dir):
    model.train()
    best_f1_score = 0
    for epoch in range(1, epochs + 1):
        global_step = 0 # 迭代次数
        for step, batch in enumerate(train_data_loader, start=1):
            tic_train = time.time()
            length = len(train_data_loader)
            input_ids, token_type_ids, labels = batch
            # 计算模型输出、损失函数值、分类概率值、准确率、f1分数
            logits = model(input_ids, token_type_ids)
            loss = criterion(logits, labels)
            probs = F.sigmoid(logits)
            metric.update(probs, labels)
            auc, f1_score, _, _ = metric.accumulate()

            # 每迭代40次或batch训练完毕，打印损失函数值、准确率、f1分数、计算速度
            global_step += 1
            if global_step % 40 == 0 or global_step == length:
                print(
                    "epoch: %d, batch: %d, loss: %.5f, auc: %.5f, f1 score: %.5f, time: %.2f s"
                    % (epoch, step, loss, auc, f1_score, (time.time() - tic_train))) # 每个batch用时
                tic_train = time.time()
            
            # 梯度回传，更新参数
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

        # 每个epoch保存一次最佳模型参数
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        eval_f1_score = evaluate(model, criterion, metric, test_data_loader, id2label, if_return_results=False)
        if eval_f1_score > best_f1_score:
            best_f1_score = eval_f1_score
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

train(epochs=epochs, save_dir=ckpt_dir)

（6）对抗训练
class FGM(): # 对抗训练
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='embedding'):
        # emb_name这个参数要换成模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and emb_name in name:
                self.backup[name] = param.clone().numpy()
                norm = paddle.norm(param.grad) # 默认为2范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.stop_gradient = True
                    param.add_(r_at)
                    param.stop_gradient = False

    def restore(self, emb_name='embedding'):
        # emb_name这个参数要换成模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.stop_gradient and emb_name in name: 
                assert name in self.backup
                param = self.backup[name]
        self.backup = {}

ckpt_dir = "fgm_ernie_ckpt"

def FGM_train(epochs, save_dir=ckpt_dir):
    best_f1_score = 0
    model.train()
    fgm = FGM(model)
    for epoch in range(1, epochs + 1):
        global_step = 0 # 迭代次数
        for step, batch in enumerate(train_data_loader, start=1):
            tic_train = time.time()
            length = len(train_data_loader)
            input_ids, token_type_ids, labels = batch
            # 计算模型输出、损失函数值、分类概率值、准确率、f1分数
            logits = model(input_ids, token_type_ids)
            loss = criterion(logits, labels)
            probs = F.sigmoid(logits)
            metric.update(probs, labels)
            auc, f1_score, _, _ = metric.accumulate()

            # 每迭代40次或batch训练完毕，打印损失函数值、准确率、f1分数、计算速度
            global_step += 1
            if global_step % 40 == 0 or global_step == length:
                print(
                    "epoch: %d, batch: %d, loss: %.5f, auc: %.5f, f1 score: %.5f, time: %.2f s"
                    % (epoch, step, loss, auc, f1_score, (time.time() - tic_train)))
                tic_train = time.time()

            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 对抗训练
            fgm.attack() # embedding被修改了
            logits = model(input_ids, token_type_ids)
            loss = criterion(logits, labels)
            loss.backward() # 反向传播，在正常的grad基础上，累加对抗训练的梯度
            fgm.restore() # 恢复Embedding的参数
            
            # 参数更新
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

        # 每个epoch保存一次最佳模型参数
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        eval_f1_score = evaluate(model, criterion, metric, test_data_loader, id2label, if_return_results=False)
        if eval_f1_score > best_f1_score:
            best_f1_score = eval_f1_score
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

FGM_train(epochs=epochs, save_dir=ckpt_dir)

（7）提交结果
# 加载已经训练好的模型
model.set_dict(paddle.load('ernie_ckpt/model_state.pdparams'))

# 加载测试集
test_ds0 = LawDataset(datapath, 'test')
test_ds = MapDataset(test_ds0)

test_trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=640,
    is_test=True
    )

test_ds = test_ds.map(test_trans_func)

collate_fn = lambda samples, fn=Dict({
    'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id),
    'token_type_ids': Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
    'ids': Stack(dtype="int32")
}): fn(samples)

test_ds_batch_sampler = BatchSampler(test_ds, batch_size=16, shuffle=False)
test_ds_data_loader = DataLoader(dataset=test_ds, batch_sampler=test_ds_batch_sampler, collate_fn=collate_fn)

import paddle.nn.functional as F

def data_reprocess():
    # 生成预测结果
    ids_ = []
    y_prob = None
    for _, (input_ids, token_type_ids, ids) in enumerate(test_ds_data_loader,start=1):
        model.eval()
        logits = model(input_ids, token_type_ids)
        probs = F.sigmoid(logits)
        if y_prob is not None:
            y_prob = np.append(y_prob, probs.numpy(), axis=0)
        else:
            y_prob = probs.numpy()
        ids_.extend(ids)

    best_threshold = 0.32 
    # 参照https://aistudio.baidu.com/aistudio/projectdetail/4201483，使用了0.32作为阈值
    # 可以对训练好的模型遍历阈值来找到最佳阈值

    y_prob = y_prob > best_threshold
    results = []
    pos = 0
    for event in test_ds0:
        assert event['id'] == ids_[pos].item() # 确保是同一条信息
        event['event_chain'] = []
        for i in range(len(id2label)):
            if y_prob[pos][i] == True:
                event['event_chain'].append(id2label[i])
                
        pos+=1
        results.append(event)
    return results

results = data_reprocess()
print(results[:5])

# 加载对抗训练得到的模型
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_classes=len(id2label))
model.set_dict(paddle.load('fgm_ernie_ckpt/model_state.pdparams'))
results1 = data_reprocess()
print(results1[:5])

4.导出结果
 # 保存结果文件
with open('submit.json','w') as f:
    json.dump(results1,f)
四、结果及分析
竞赛项目的准确率及排名的截图：
测试运行结果截图：
此时模型在验证集上的最佳F1值表现为：
eval loss: 0.20442, auc: 0.97672, f1 score: 0.91613, precison: 0.94278, recall: 0.89096

五、总结
本次课程设计让我深刻体会到机器学习在信息抽取领域的强大能力，尤其是在法律领域篇章级多事件检测任务中，模型能够有效地识别并分类不同类型的事件，为后续的法律案件分析提供有力支持。同时，我也认识到机器学习模型的训练和应用并非易事，需要克服许多难题，比如：模型在测试集上取得了较好的成绩，但在实际的测试集中，模型预测的结果并不准确，因此需要进一步提升其泛化能力。
