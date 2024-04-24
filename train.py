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
model_text_cls = Model(cfg) # 定义模型
model_text_cls.to(cfg.devices) # 走cuda

loss_func = nn.CrossEntropyLoss() #交叉熵损失
optimizer = optim.Adam(model_text_cls.parameters(), lr= cfg.learn_rate) # 配置网络参数和学习率

for epoch in range(cfg.num_epochs): # 迭代训练
    for i, batch in enumerate(train_dataloader):
        label,data = batch
        data = torch.tensor(data).to(cfg.devices) # 定义数据
        label = torch.tensor(label,dtype=torch.int64).to(cfg.devices) # 定义标签
        
        optimizer.zero_grad() #梯度置零
        pred = model_text_cls.forward(data) #前向推理
        loss_val = loss_func(pred,label) # 计算损失
        # print(pred)
        # print(label)
        
        print("epoch is {},ite is {},val is {}".format(epoch,i, loss_val))
        loss_val.backward() # 反向传播
        optimizer.step() # 参数更新
        
    if epoch % 10 == 0: # 每10个epoch保存一个模型
        torch.save(model_text_cls.state_dict(), "models/{}.pth".format(epoch))
        
        



