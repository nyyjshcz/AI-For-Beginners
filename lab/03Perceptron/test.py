import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

data_path="../../data/mnist.pkl"
if not os.path.exists(data_path):
    print(f"错误：找不到数据文件，请确认{data_path}路径正确！")
else:
    print(f"找到数据文件，正在加载{data_path}...")
    with open(data_path,'rb')as mnist_pickle:
        training_data,validation_data,test_data=pickle.load(mnist_pickle,encoding='latin1')
    MNIST={
        'Train':{
            'Features':training_data[0],
            'Labels':training_data[1]
        },
        'Test':{
            'Features': test_data[0],
            'Labels': test_data[1] 
        }
    }

train_features=MNIST['Train']['Features'].astype(np.float32)
train_labels=MNIST['Train']['Labels']
test_features=MNIST['Test']['Features'].astype(np.float32)
test_labels=MNIST['Test']['Labels']

def train(positive_examples, negative_examples, num_iterations=200, batch_size=1024, learning_rate=0.01, early_stop_threshold=0.999):
    num_dims = positive_examples.shape[1]
    weights = np.zeros((num_dims, 1))
    bias = 0.0
    pos_total = positive_examples.shape[0]
    neg_total = negative_examples.shape[0]
    report_frequency = 50
    for i in range(num_iterations):
        pos_batch = positive_examples[np.random.choice(pos_total, replace=True, size=batch_size)]
        neg_batch = negative_examples[np.random.choice(neg_total, size=batch_size, replace=True)]
        pos_z = np.dot(pos_batch, weights) + bias
        neg_z = np.dot(neg_batch, weights) + bias
        pos_mask = (pos_z <= 0).flatten()
        neg_mask = (neg_z >= 0).flatten()
        weight_gradient = np.sum(pos_batch[pos_mask], axis=0).reshape(weights.shape) - np.sum(neg_batch[neg_mask], axis=0).reshape(weights.shape)
        bias_gradient = pos_mask.sum() - neg_mask.sum()
        weights += learning_rate * weight_gradient
        bias += learning_rate * bias_gradient
        if i % report_frequency == 0 or i == num_iterations - 1:
            pos_acc = (np.dot(positive_examples, weights) + bias >= 0).mean()
            neg_acc = (np.dot(negative_examples, weights) + bias < 0).mean()
            print(f"  轮次 {i:3d}: 正样本准确率 {pos_acc*100:5.2f}% | 负样本准确率 {neg_acc*100:5.2f}%")
            if pos_acc >= early_stop_threshold and neg_acc >= early_stop_threshold:
                print(f"  在第 {i} 轮达到目标准确率，提前停止训练。")
                break
    
    return weights, bias

print("\n开始训练10个感知机(0-9)")
weights_list=[]
bias_list=[]
for digit in range(10):
    print(f"正在训练数字{digit}的感知机")
    pos_examples=train_features[train_labels==digit]
    neg_examples=train_features[train_labels!=digit]
    w,b=train(pos_examples, neg_examples)
    weights_list.append(w)
    bias_list.append(b)
    print(f"数字{digit}的感知机训练完成")

weight_matrix = np.hstack(weights_list)
bias_array = np.array(bias_list)

scores = (test_features @ weight_matrix) + bias_array
predictions = np.argmax(scores, axis=1)

accuracy = (predictions == test_labels).mean() * 100
print(f"\n测试集上的准确率: {accuracy:.2f}%")