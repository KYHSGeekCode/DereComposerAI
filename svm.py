import numpy as np
from matplotlib import pyplot as plt

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


def grad(Xi, Yi, w):
    mrn = np.finfo(Xi.dtype).max
    thr = np.log(mrn) - 2.0
    son = -Yi * Xi
    momi = Yi * (Xi @ w)
    print("momi", momi)
    if momi >= thr:
        beta = thr - momi
        betaexp = np.exp(beta)
        return son * betaexp / (betaexp + np.exp(thr))
    mom = np.exp(momi) + 1
    return son / mom


def average_pooling(arr):
    x, y = arr.shape
    new_x, new_y = x // 8, y // 13
    arr = np.mean(arr.reshape(new_x, 8, new_y, 13), axis=(1, 3))
    return arr


def main():
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
    ws = [np.random.rand(feature_size) / scale] * len(ys)

    K = 30000
    alpha = 0.0000005
    f_vals = []
    for wi in range(len(ws)):
        # w[i]를 학습시킨다.
        they = ys[wi]
        print(they)
        f_val = []
        for kki in range(K):  # iterate SGD K times
            ind = np.random.randint(N)  # index
            sampledX = X[ind, :].flatten()
            # print(sampledX.shape)
            gradval = grad(sampledX, they[ind], ws[wi])
            # print("grad", gradval)
            ws[wi] -= alpha * gradval
            print("wsnorm", np.linalg.norm(ws[wi]))
            targetval = target(N, X, they, ws[wi])
            f_val.append(targetval)
            # print(targetval)
            if (kki + 1) % 1000 == 0:
                print(kki)
        f_vals.append(f_val)
        print(wi)

    with open(f"weights_{scale}_{alpha}_{N}", 'wb') as f:
        np.save(f, ws)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=14)
    for idx, f_val in enumerate(f_vals):
        plt.plot(list(range(K)), f_val, label=f"Stochastic Gradient Descent {idx}")
    plt.title(f"Scale:{scale}, alpha={alpha}, data={N}")

    # plt.plot(list(range(K)),np.linalg.norm(X@np.linalg.inv(X.T@X)@X.T@Y-Y)**2*np.ones(K), color = "red", label = "Optimal Value")
    plt.xlabel('Iterations')
    plt.ylabel(r'$f(\theta^k)$')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
