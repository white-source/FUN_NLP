import os
import glob
import re
import jieba
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer


def txt_path_to_txt():
    # 将训练数据的txt和测试数据的txt保存在txt中
    train_path = "/root/train_data/fudan_data/Fudan/train/"  # 训练数据存放位置
    test_path = "/root/train_data/fudan_data/Fudan/answer/"  # 测试数据存放位置
    train_txt_path = "/root/fxh/FUN_NLP/tensorflow-text-classification-master/data/Fudan/train.txt"
    test_txt_path = "/root/fxh/FUN_NLP/tensorflow-text-classification-master/data/Fudan/test.txt"
    # 用于获取目录下的所有文件夹，返回一个列表
    train_list = os.listdir(train_path)
    fp1 = open(train_txt_path, "a", encoding="utf-8")
    fp2 = open(test_txt_path, "a", encoding="utf-8")
    for train_dir in train_list:
        # 用于获取当前目录下指定的文件，返回的是一个字符串
        for txt in glob.glob(train_path + train_dir + "/*.txt"):
            fp1.write(txt + "\n")
    fp1.close()
    test_list = os.listdir(test_path)
    for test_dir in test_list:
        for txt in glob.glob(test_path + test_dir + "/*.txt"):
            fp2.write(txt + "\n")
    fp2.close()


def train_content_to_txt():
    train_txt_path = "/root/fxh/FUN_NLP/tensorflow-text-classification-master/data/Fudan/train.txt"  # 训练数据txt
    test_txt_path = "/root/fxh/FUN_NLP/tensorflow-text-classification-master/data/Fudan/test.txt"  # 测试数据txt

    train_content_path = "/root/fxh/FUN_NLP/tensorflow-text-classification-master/data/Fudan/train_jieba.txt"  # 存储文本和标签txt
    train_content_txt = open(train_content_path, "a", encoding="utf-8")

    def remove_punctuation(line, strip_all=True):
        if strip_all:
            rule = re.compile(u"[^a-zA-Z0-9\u4e00-\u9fa5]")
            line = rule.sub('', line)
        else:
            punctuation = """！？｡＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"""
            re_punctuation = "[{}]+".format(punctuation)
            line = re.sub(re_punctuation, "", line)
        return line.strip()

    train_txt = open(train_txt_path, "r", encoding="utf-8")

    for txt in train_txt.readlines():  # 读取每一行的txt
        txt = txt.strip()  # 去除掉\n
        content_list = []
        label_str = txt.split("/")[-1].split("-")[-1]  # 先用/进行切割，获取列表中的最后一个，再利用-进行切割，获取最后一个
        label_list = []
        # 以下for循环用于获取标签，遍历每个字符，如果遇到了数字，就终止
        for s in label_str:
            if s.isalpha():
                label_list.append(s)
            elif s.isalnum():
                break
            else:
                print("出错了")
        label = "".join(label_list)  # 将字符列表转换为字符串，得到标签
        # print(label)
        # 以下用于获取所有文本
        fp1 = open(txt, "r", encoding="gb18030", errors='ignore')  # 以gb18030的格式打开文件，errors='ignore'用于忽略掉超过该字符编码范围的字符
        for line in fp1.readlines():  # 读取每一行
            # line = remove_punctuation(line)
            line = jieba.lcut(line.strip(), cut_all=False)  # 进行分词，cut_all=False表明是精确分词，lcut()返回的分词后的列表
            content_list.extend(line)
        fp1.close()

        content_str = " ".join(content_list)  # 转成字符串
        # print(content_str)
        train_content_txt.write(content_str + "\t" + label + "\n")  # 将文本 标签存到txt中

    train_content_txt.close()


def test_content_to_txt():
    test_txt_path = "/root/fxh/FUN_NLP/tensorflow-text-classification-master/data/Fudan/test.txt"
    test_content_path = "/root/fxh/FUN_NLP/tensorflow-text-classification-master/data/Fudan/test_jieba.txt"
    test_content_txt = open(test_content_path, "a", encoding="utf-8")
    import re
    def remove_punctuation(line, strip_all=True):
        if strip_all:
            rule = re.compile(u"[^a-zA-Z0-9\u4e00-\u9fa5]")
            line = rule.sub('', line)
        else:
            punctuation = """！？｡＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"""
            re_punctuation = "[{}]+".format(punctuation)
            line = re.sub(re_punctuation, "", line)
        return line.strip()

    test_txt = open(test_txt_path, "r", encoding="utf-8")
    for txt in test_txt.readlines():
        txt = txt.strip()
        content_list = []
        label_str = txt.split("/")[-1].split("-")[-1]
        label_list = []
        # 以下for循环用于获取标签
        for s in label_str:
            if s.isalpha():
                label_list.append(s)
            elif s.isalnum():
                break
            else:
                print("出错了")
        label = "".join(label_list)
        # print(label)
        # 以下用于获取所有文本
        fp1 = open(txt, "r", encoding="gb18030", errors='ignore')
        for line in fp1.readlines():
            # line = remove_punctuation(line)
            line = jieba.lcut(line.strip(), cut_all=False)
            content_list.extend(line)
        fp1.close()
        content_str = " ".join(content_list)
        # print(content_str)
        test_content_txt.write(content_str + "\t" + label + "\n")

    test_content_txt.close()


# def label2dicNumber():
# 对标签映射成具体的数值
label = set()
with open("/root/fxh/FUN_NLP/tensorflow-text-classification-master/data/Fudan/train_jieba.txt", "r",
          encoding="utf-8") as fp:
    lines = fp.readlines()
    for line in lines:
        line = line.strip().split("\t")
        if len(line) == 2:
            label.add(line[1])
with open("/root/fxh/FUN_NLP/tensorflow-text-classification-master/data/Fudan/test_jieba.txt", "r",
          encoding="utf-8") as fp:
    lines = fp.readlines()
    for line in lines:
        line = line.strip().split("\t")
        if len(line) == 2:
            label.add(line[1])
label_to_idx = list(zip(label, range(len(label))))
label_to_idx_dict = {}
idx_to_label_dict = {}
for k, v in label_to_idx:
    label_to_idx_dict[k] = v
    idx_to_label_dict[v] = k
print(label_to_idx_dict)
print(idx_to_label_dict)

# 构建训练集
train_data = []
train_label = []
with open("/root/fxh/FUN_NLP/tensorflow-text-classification-master/data/Fudan/train_jieba.txt", "r",
          encoding="utf-8") as fp:
    lines = fp.readlines()
    for line in lines:
        line = line.strip().split("\t")
        if len(line) == 2:
            train_data.append([line[0]])
            train_label.append([label_to_idx_dict[line[1]]])
train_data = np.array(train_data)
train_label = np.array(train_label)
print(train_data.shape)
print(train_label.shape)
print(train_data[:2])
print(train_label[:2])

# 构建测试集合
test_data = []
test_label = []
with open("/root/fxh/FUN_NLP/tensorflow-text-classification-master/data/Fudan/test_jieba.txt", "r",
          encoding="utf-8") as fp:
    lines = fp.readlines()
    for line in lines:
        line = line.strip().split("\t")
        if len(line) == 2:
            test_data.append([line[0]])
            test_label.append([label_to_idx_dict[line[1]]])
test_data = np.array(test_data)
test_label = np.array(test_label)
print(test_data.shape)
print(test_label.shape)
print(test_data[:2])
print(test_label[:2])

# 打散训练集 要同时将数据和标签进行打乱
shuffle_ix = np.random.permutation(np.arange(len(train_data)))
train_row = list(range(len(train_data)))
random.shuffle(train_row)
print(train_row)
train_data = train_data[train_row, :]
train_label = train_label[train_row, :]
print(train_data[:2])
print(train_label[:2])

# 这里需要将二维数组转换一维数组，同时要将numpy数组转换为列表

train_data = train_data.reshape(len(train_data)).tolist()
print(train_data[:2])
tfidf_model = TfidfVectorizer()
sparse_result = tfidf_model.fit_transform(train_data)  # 得到tf-idf矩阵，稀疏矩阵表示法

for k, v in tfidf_model.vocabulary_.items():
    print(k, v)

test_data = test_data.reshape(len(test_data)).tolist()
print(test_data[:2])
test_sparse_result = tfidf_model.transform(test_data)

for k, v in tfidf_model.vocabulary_.items():
    print(k, v)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

mnb_count = MultinomialNB()
mnb_count.fit(sparse_result, np.ravel(train_label))  # 学习
mnb_count_y_predict = mnb_count.predict(test_sparse_result)  # 预测
print(classification_report(mnb_count_y_predict, test_label))
