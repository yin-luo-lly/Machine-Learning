import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from collections import Counter
from matplotlib.font_manager import FontProperties

from utils import id_label, data_prepare

datapath = 'Data'

id2label, label2id = id_label(datapath)
train_dataset = data_prepare(datapath, label2id, mode='train')
dev_dataset = data_prepare(datapath, label2id, mode='dev')
test_dataset = data_prepare(datapath, label2id, mode='test')

print(train_dataset[0], '\n\n', id2label, '\n\n', label2id)

df_train = pd.DataFrame.from_dict(train_dataset)
df_train.head()
df_train.info()

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

fig, ax = plt.subplots(1,1,figsize=(12,4))

ax = plt.hist(x=df_train['lenth'], bins=100)
ax = plt.hist(x=df_test['lenth'], bins=100)

plt.xlim([0, max(max(df_train['lenth']), max(df_test['lenth']))])
plt.xlabel("length of sample")
plt.ylabel("number of sample")
plt.legend(['train_len','test_len'])

plt.show()


import scipy
scipy.stats.ks_2samp(df_train['lenth'], df_test['lenth'])

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


num_label = []
for j in range(len(id2label)):
    m = 0
    for i in range(len(df_train['labels'])):
        m += df_train['labels'][i][j]
    num_label.append(m)

plt.figure()
plt.bar(x=range(len(id2label)), height=num_label)
plt.xlabel("label")
plt.ylabel("number of sample")
plt.xticks(range(len(id2label)), list(id2label.values()), rotation=45)
plt.show()

objs = [df_train[['id', 'lenth']], pd.DataFrame(df_train['labels'].tolist())]
ans1 = pd.concat(objs, axis=1)
ans2 = pd.melt(ans1, var_name='id_label', value_name='labels', id_vars=['id', 'lenth'])
ans2 = ans2[ans2['labels']!=0.0].reset_index(drop=True).drop('labels', axis=1)

plt.figure()
ax = sns.catplot(x='id_label', y='lenth', data=ans2, kind='strip')
plt.xticks(range(len(id2label)), list(id2label.values()), rotation=45)

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


from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.data import Stack, Tuple, Pad, Dict
from functools import partial

model_name = "ernie-3.0-base-zh"

tokenizer = AutoTokenizer.from_pretrained(model_name)


def convert_example(example, tokenizer, max_seq_length=640, is_test=False):  # 数据预处理函数

    # tokenizer.encode方法能够完成切分token，映射token ID以及拼接特殊token
    tokenized_example = tokenizer.encode(text=example['text'], max_seq_len=max_seq_length, truncation=True)
    if not is_test:
        tokenized_example['labels'] = example['labels']  # 加上labels用于训练
    else:
        tokenized_example['ids'] = example['id']

    return tokenized_example


trans_func = partial(  # 给convert_example传入参数
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=640,
    is_test=False
)

train_dataset = train_dataset.map(trans_func)
test_dataset = test_dataset.map(trans_func)

print(train_dataset[0])
print(test_dataset[0])

# collate_fn函数构造，将不同长度序列充到批中数据的最大长度，再将数据堆叠

collate_fn = lambda samples, fn=Dict({
    'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id),
    'token_type_ids': Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
    'labels': Stack(dtype="float32")
}): fn(samples)

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

import time
from eval import evaluate
import paddle.nn.functional as F

ckpt_dir = "ernie_ckpt"  # 训练过程中保存模型参数的文件夹


def train(epochs, save_dir=ckpt_dir):
    model.train()
    best_f1_score = 0
    for epoch in range(1, epochs + 1):
        global_step = 0  # 迭代次数
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
                    % (epoch, step, loss, auc, f1_score, (time.time() - tic_train)))  # 每个batch用时
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
# 释放显存分配器中空闲的显存
paddle.device.cuda.empty_cache()

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
        global_step = 0  # 迭代次数
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
            fgm.attack()  # embedding被修改了
            logits = model(input_ids, token_type_ids)
            loss = criterion(logits, labels)
            loss.backward()  # 反向传播，在正常的grad基础上，累加对抗训练的梯度
            fgm.restore()  # 恢复Embedding的参数

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

ckpt_dir = "fgm_ernie_ckpt"


def FGM_train(epochs, save_dir=ckpt_dir):
    best_f1_score = 0
    model.train()
    fgm = FGM(model)
    for epoch in range(1, epochs + 1):
        global_step = 0  # 迭代次数
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
            fgm.attack()  # embedding被修改了
            logits = model(input_ids, token_type_ids)
            loss = criterion(logits, labels)
            loss.backward()  # 反向传播，在正常的grad基础上，累加对抗训练的梯度
            fgm.restore()  # 恢复Embedding的参数

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

# 加载对抗训练得到的模型
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_classes=len(id2label))
model.set_dict(paddle.load('fgm_ernie_ckpt/model_state.pdparams'))
results1 = data_reprocess()
print(results1[:5])

# 保存结果文件
with open('submit.json','w') as f:
    json.dump(results1,f)




