from sklearn.feature_extraction.text import TfidfVectorizer

# 这里需要将二维数组转换一维数组，同时要将numpy数组转换为列表

train_data = train_data.reshape(len(train_data)).tolist()
print(train_data[:2])
tfidf_model = TfidfVectorizer()
sparse_result = tfidf_model.fit_transform(train_data)  # 得到tf-idf矩阵，稀疏矩阵表示法
