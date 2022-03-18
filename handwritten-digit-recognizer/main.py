from os import F_TEST
from nnet import Nnet
import numpy as np
import matplotlib.pyplot as plt
import read_MNIST
import random
import sys
import consts
import tqdm

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def run(nnet :Nnet, test):
    ll_a = test # last layer a
    for i in range(1, len(nnet.Ws)):
        ll_a = sigmoid((nnet.Ws[i] @ ll_a) + nnet.bs[i])
    
    return ll_a

def get_mx_ind(arr :np.array):
    return np.argmax(arr)

def test_nn(nnet :Nnet, test_set):
    passed = 0
    mse = 0
    for it in range(len(test_set)):
        out, ans = run(nnet, test_set[it][0]), test_set[it][1]
        mse += (np.square(out - ans)).mean(axis=None)
        if get_mx_ind(out) == get_mx_ind(ans):
            passed += 1
    
    return mse / len(test_set), passed/len(test_set)

def deriv(nnet :Nnet, test, ans):
    zs = ['_']
    a_s = [test]
    for i in range(1, len(nnet.Ws)):
        z = (nnet.Ws[i] @ a_s[i-1]) + nnet.bs[i]
        a = sigmoid(z)
        
        zs.append(z)
        a_s.append(a)
    
    da = ['_' for i in range(len(nnet.Ws))]
    dW = ['_' for i in range(len(nnet.Ws))]
    db = ['_' for i in range(len(nnet.Ws))]
    da[-1] = (a_s[-1] - ans)
    
    for i in range(len(nnet.Ws)-1, 0, -1):
        v = dsigmoid(zs[i]) * da[i]
        dW[i] = v @ a_s[i-1].T
        db[i] = v
        da[i-1] = nnet.Ws[i].T @ v
    
    return dW, db

def learn(nnet :Nnet, batch):
    sdW = [np.zeros(nnet.Ws[i].shape) if i > 0 else '_' for i in range(len(nnet.Ws))]
    sdb = [np.zeros(nnet.bs[i].shape) if i > 0 else '_' for i in range(len(nnet.Ws))]
    for test in batch:
        dW, db = deriv(nnet, test[0], test[1])
        for i in range(1, len(nnet.Ws)):
            sdW[i] += dW[i]
            sdb[i] += db[i]

    for i in range(1, len(nnet.Ws)):
        nnet.Ws[i] -= consts.LEARNING_RATE * (sdW[i] / len(batch))
        nnet.bs[i] -= consts.LEARNING_RATE * (sdb[i] / len(batch))
    # print(nnet.bs[3])

def train(nnet :Nnet, train_set, test_set):
    mses = []
    accs = []
    axis = []

    def measure(id):
        mse, acc = test_nn(nnet, test_set)
        mses.append(mse)
        accs.append(acc)
        axis.append(id)

    for _ in range(consts.EPOCH_COUNT):
        print("epoch num: {}".format(_+1))
        measure(_)
        random.shuffle(train_set)
        for i in tqdm.tqdm(range(0, len(train_set), consts.BATCH_SIZE)):
            learn(nnet, train_set[i:min(i + consts.BATCH_SIZE, len(train_set))])
    measure(consts.EPOCH_COUNT)


    plt.plot(axis, mses)
    plt.show()
    plt.plot(axis, accs)
    plt.show()

if __name__ == '__main__':
    print("loading tests.")
    train_set, test_set = read_MNIST.load_tests("tests/")
    try:
        if sys.argv[1] == 'sh': # shift
            sh_count = int(sys.argv[2]) # for example: python main.py sh 4
            for i in range(len(test_set)):
                test_set[i] = (
                    np.concatenate((np.zeros((28*sh_count, 1)), test_set[i][0][:-28*sh_count, :])),
                    test_set[i][1]
                )
    except:
        pass

    nnet = Nnet()
    train(nnet, train_set, test_set)
    mse, acc = test_nn(nnet, test_set)
    print(mse, acc)