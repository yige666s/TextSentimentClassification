import torch
import torch.nn as nn
from torch import optim # 优化器
from models import Model # 模型
from datasets import data_loader,text_Cls # 数据
from configs import Config # 配置

cfg = Config() # 定义配置

data_path = "sources/weibo_senti_100k.csv" # 数据集
data_stop_path = "sources/hit_stopword" # 停用词典
dict_path = "sources/dict"

dataset = text_Cls(dict_path,data_path,data_stop_path)
train_dataloader = data_loader(dataset,cfg)
cfg.pad_size = dataset.max_len_seq #设置最大填充长度
print(cfg.pad_size)

model_text_cls = Model(cfg) # 定义模型
model_text_cls.to(cfg.devices) # 走cuda
model_text_cls.load_state_dict(torch.load("models/0.pth"))

for i, batch in enumerate(train_dataloader):
    label,data = batch
    data = torch.tensor(data).to(cfg.devices) # 定义数据
    label = torch.tensor(label,dtype=torch.int64).to(cfg.devices) # 定义标签
    pred_softmax = model_text_cls.forward(data) #前向推理 
    pred = torch.argmax(pred_softmax,dim=1)
    out = torch.eq(pred, label)
    print(out)
    print(out.sum() * 1.0 /pred.size()[0]) # 统计准确率
        



