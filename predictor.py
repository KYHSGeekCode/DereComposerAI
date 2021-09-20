import numpy as np

from sqlhelper import get_colors
from util import load_random_vectors, get_music_files


def target(N, X, Y, w):
    mrn = np.finfo(X.dtype).max
    thr = np.log(mrn) - 2.0
    s = 0
    for i in range(N):
        d = X[i, :].flatten() @ w
        yid = -Y[i] * d
        # print("yid", yid)
        # print("thr", thr)
        if yid >= thr:
            beta = thr - yid
            s = s + np.log(np.exp(beta) + np.exp(thr)) - beta
        else:
            s = s + np.log(1 + np.exp(yid))
    return s / N


def predict(xxxx, w):
    return xxxx @ w


def average_pooling(arr):
    x, y = arr.shape
    new_x, new_y = x // 8, y // 13
    arr = np.mean(arr.reshape(new_x, 8, new_y, 13), axis=(1, 3))
    return arr


# setup
N = 150
np.random.seed(0)

# Prepare X
files = get_music_files()
print(f"Loading random {N} features")
labels, vectors = load_random_vectors(files, N)

# Normalize X
minlen1 = min([v.shape[0] for v in vectors])
minlen1 = minlen1 // 104 * 104
print(minlen1)
vectors = [v[:minlen1, :] for v in vectors]
vectors = [average_pooling(v) for v in vectors]
minlen = min([v.shape[0] for v in vectors])
print(minlen)
print(labels[0], vectors[0].shape)
vectors = [v[:minlen, :] for v in vectors]
print(labels[0], vectors[0].shape)
X = np.array(vectors)
print(X.shape)

# prepare y
print("Preparing Y")
colors = get_colors('210920.db')
print(colors)
y = np.array([colors[x] for x in labels])
print(y)
ys = [0, 0, 0, 0]
for i in range(4):
    ys[i] = [1 if yy == (1 + i) else -1 for yy in y]
for yyy in ys:
    print(yyy)

# prepare weights
feature_size = vectors[0].shape[1] * minlen
print(feature_size)
scale = 10000000

with open("weights_10000000_5e-07_150", 'rb') as f:
    ws = np.load(f)

for iiii in range(N):
    xxxx = X[iiii, :].flatten()

    print("xxxx:", xxxx.shape)
    what = -1
    whatv = -1
    for wi in range(len(ws)):
        print("wi:", wi)
        print(ws[wi, :])
        correctY = ys[wi][iiii]
        print(correctY)
        # print("wswi", ws[wi].shape)
        res = np.exp(predict(xxxx, ws[wi, :]))
        print(res)
        if res > whatv:
            what = wi
            whatv = res
    print("what:", what, "val", whatv)

print(ws)