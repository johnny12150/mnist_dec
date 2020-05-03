import tensorflow as tf
import numpy as np
from scipy.stats import entropy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances
from tensorflow.keras.datasets.mnist import load_data
# from tf_KNN import KNeighborsClassifier

(x_train, y_train), (x_test, y_test) = load_data("C:/Users/Wade/.keras/datasets/mnist.npz")
x_train, x_test = x_train / 255., x_test / 255.
x_train = x_train[:20000] # 沒label的
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 128 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 2 # 降維
epoch = 1
batch = 1000

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, 1])
# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def kNN(x_train, y_train, x_test, k):
    # 這邊就是上面定義的 l2 distance
    distance = tf.norm(tf.subtract(x_train, tf.expand_dims(x_test, 1)), axis=2)
    # 在算出來的 distance 裡面，找出 k 個最近的，回傳 index 與 value
    # tf.nn.top_k 預設回傳排序為大到小，加上 negative 可以逆序排列 
    # top_k_xvals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
    indices = tf.argsort(distance, axis=-1, direction='DESCENDING')
    topk = tf.constant(list(range(k)))
    top_k_indices = tf.gather(indices, topk, axis=-1)
    # 根據回傳的 index，到 y_train 找出對應的 target value
    prediction_indices = tf.gather(y_train, top_k_indices)
    # todo 累計每類各幾票, 以下不支援 2d
    prediction_indices = tf.reshape(prediction_indices, (-1,))
    _, _, count_of_predictions = tf.unique_with_counts(prediction_indices)
    proba = tf.div(count_of_predictions, k)
    # 回傳的是 1-hot 的 target vector，這邊將他改為整數
    # count_of_predictions = tf.reduce_sum(prediction_indices, axis=1)
    # 投票結果
    # prediction = tf.argmax(count_of_predictions, axis=1)
    # 機率
    # proba = tf.div(count_of_predictions, k)
    return proba


def dim_reduction(x, sig=False):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    if sig:
        return tf.nn.sigmoid(out_layer)
    return out_layer


def cal_loss(x, y):
    # 根據前 1000篇做 KNN, 算 label的總 entropy

    xt = tf.gather(x, tf.constant(list(range(batch, batch+10000))))
    xte = tf.gather(x, tf.constant(list(range(batch))))
    counts = kNN(xt, Y, xte, 50)  # shape: batch, classes
    counts = tf.cast(counts, 'float')
    loss = -tf.reduce_sum(counts)
    # loss = -tf.reduce_sum(counts * tf.log(counts))  # entropy
    return loss


sess = tf.Session()
sess.run(tf.global_variables_initializer())
logits = dim_reduction(X)
emb = dim_reduction(X, True)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
# opt_op = optimizer.minimize(cal_loss(logits, Y))

# grads_and_vars = optimizer.compute_gradients()
# opt_op = optimizer.apply_gradients(clipped_grads_and_vars)

# 將視覺化輸出
writer = tf.summary.FileWriter("TensorBoard/", graph=sess.graph)


def find_label(x):
    return np.where(np.in1d())


for i in range(epoch):
    # 先降成兩維
    feed_dict = dict()
    feed_dict.update({X: np.vstack([x_train, x_test])})
    out_emb = sess.run(logits, feed_dict=feed_dict)

    # todo 先用 knn找出各圖最近的圖
    knn = KNeighborsClassifier(n_neighbors=50, algorithm='kd_tree', n_jobs=4)
    knn.fit(out_emb[-10000:], y_test)
    predict_label = knn.predict(out_emb[:20000])  # 找每張圖最近的label
    # todo 根據各類找最近的 K個距離
    distance_table = pairwise_distances(out_emb[:20000], out_emb[-10000:], metric='euclidean', n_jobs=4)

    # pairwise handcraft
    # repeat_train = np.repeat(out_emb[:20000], 10000, axis=0).reshape(20000, 10000, -1)
    # repeat_target = np.tile(out_emb[-10000:], (20000, 1)).reshape(20000, 10000, -1)
    # custom = np.linalg.norm((repeat_train - repeat_target), axis=2)

    # todo 算各圖的 distances
    # 取target中各類label的index
    # targets = np.tile(y_test, (20000, 1))
    # targets_idx = np.zeros((20000, 10000))
    y_distance = np.zeros(20000)
    for it in range(len(predict_label)):
        match_dist = distance_table[it, np.isin(y_test, predict_label[it])]
        # 根據取出的distance排序, 取最小的K個
        y_distance[it] = np.sum(np.sort(match_dist)[:50], axis=0)

    for s in range(int(20000/ batch)):
        feed_dict = dict()
        # feed_dict.update({X: np.vstack([x_train[s:(s+1)*batch], x_test])})
        # todo 放原始 feature跟算過的 distance
        feed_dict.update({X: x_train[s:(s + 1) * batch]})
        sess.run([opt_op], feed_dict=feed_dict)

sess.close()
