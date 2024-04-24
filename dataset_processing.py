# 1. 语句分词来构建词典
# 2. 使用停用词典处理语料

import jieba

data_path = "sources/weibo_senti_100k.csv" # 数据集
data_stop_path = "sources/hit_stopword" # 停用词典
data_list =  open(data_path).readlines()[1:] # 读取数据去掉第一行
stops_word = open(data_stop_path).readlines() # 读入停用词
stops_word = [line.strip() for line in stops_word] # 去掉每行的换行符
voc_dict = {} # Topk词典
min_seq = 1 # 最小词频
top_n = 1000 # 词典长度为1000
UNK = "<UNK>" # Topk以外的词默认表示
PAD = "<PAD>" # 填充字段
stops_word.append(" ")
stops_word.append("\n")



# 1. 去除停用词，构建词典
for item in data_list:
    label = item[0] # 拿到标签
    content = item[2:].strip() # 拿到内容并去掉换行符
    seg_list = jieba.cut(content, cut_all=False)    # 对语句进行分词
    seg_res = []
    for seg_item in seg_list:
        # print(seg_item)
        if seg_item in stops_word:  # 分词结果位于停用词词典则跳过该词
            continue;
        seg_res.append(seg_item) # 每条句子去掉停用词后的结果
        if seg_item in voc_dict.keys():
            voc_dict[seg_item] = voc_dict[seg_item] + 1
        else:
            voc_dict[seg_item] = 1
    # print(content) # 原内容
    # print(seg_res) # 去掉停用词后内容

# 2. 统计词典中的Topk
voc_list = sorted([_ for _ in voc_dict.items() if _[1] > min_seq],
                  key =lambda x:x[1],
                  reverse=True)[:top_n] # 取前1000个词
voc_dict = {word_count[0] : idx for idx,word_count in enumerate(voc_list)} # 重新构建词典
voc_dict.update({UNK:len(voc_dict), PAD:len(voc_dict)+1})  #Topk以外的词定义为Unknow
# print(voc_dict) # 最终的词典
print(len(voc_dict))

ff = open("sources/dict","w") # 保存词典
for item in voc_dict.keys():
    ff.writelines("{},{}\n".format(item,voc_dict[item]))
ff.close()
