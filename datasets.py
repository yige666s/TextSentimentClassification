from torch.utils.data import Dataset, DataLoader
import jieba
import numpy as np
from configs import Config
# 加载字典
def read_dict(voc_dict_path):
  voc_dict = {}
  dict_list = open(voc_dict_path).readlines()
  for item in dict_list:
    item = item.split(",")
    voc_dict[item[0]] = int(item[1].strip())
  return voc_dict
  
# 加载数据
def load_data(data_path, data_stop_path):
    data_list =  open(data_path).readlines()[1:] # 读取数据去掉第一行
    stops_word = open(data_stop_path).readlines() # 读入停用词
    stops_word = [line.strip() for line in stops_word] # 去掉每行的换行符
    voc_dict = {} # Topk词典
    stops_word.append(" ")
    stops_word.append("\n")
    data = [] # 数据集字典
    max_len_seq = 0 # 最长词向量长度
    np.random.shuffle(data_list)
    for item in data_list[:]:
        label = item[0] # 拿到标签
        content = item[2:].strip() # 拿到内容并去掉换行符
        seg_list = jieba.cut(content, cut_all=False)    # 对语句进行分词
        seg_res = []
        for seg_item in seg_list:
            # print(seg_item)
            if seg_item in stops_word:  # 分词结果位于停用词词典则跳过该词
                continue
            seg_res.append(seg_item) # 每条句子去掉停用词后的结果
            if seg_item in voc_dict.keys():
                voc_dict[seg_item] = voc_dict[seg_item] + 1
            else:
                voc_dict[seg_item] = 1   
        if len(seg_res) > max_len_seq: # 统计最长词向量长度
            max_len_seq = len(seg_res)  
        data.append([label,seg_res])
    return data, max_len_seq

class text_Cls(Dataset):
    def __init__(self,voc_dic_path,data_path,data_stop_path):
        self.data_path = data_path # 数据集路径
        self.data_stop_path = data_stop_path # 停用词路径
        self.voc_dict = read_dict(voc_dic_path) # Token字典路径
        self.data, self.max_len_seq = load_data(self.data_path, self.data_stop_path) # 返回数据集和最长词向量长度
        np.random.shuffle(self.data) # 打乱数据集
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,item):
        data = self.data[item]
        label = int(data[0]) # 提取标签
        word_list = data[1] # 提取内容
        input_idx = [] # 内容对应的词向量
        for word in word_list:
            if word in self.voc_dict.keys(): # 在字典中则置为对应word的idx
                input_idx.append(self.voc_dict[word])
            else: # 如果对应word不在topK中则置为UNK
                input_idx.append(self.voc_dict["<UNK>"])
        if len(input_idx) < self.max_len_seq:
            input_idx += [self.voc_dict["<PAD>"]
                        for _ in range(self.max_len_seq - len(input_idx))] #PAD填充剩余部分
        data = np.array(input_idx)
        return label,data


def data_loader(dataset, config):
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=config.is_shuffle)

if __name__ == "__main__":
    data_path = "sources/weibo_senti_100k.csv" # 数据集
    data_stop_path = "sources/hit_stopword" # 停用词典
    dict_path = "sources/dict"
    cfg = Config()
    dataset = text_Cls(dict_path,data_path,data_stop_path)
    train_dataloader = data_loader(dataset,cfg)
    for i, batch in enumerate(train_dataloader):
        print(i,batch)
                          