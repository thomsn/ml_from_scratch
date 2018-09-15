from csv import reader
from decision_tree import create_tree, predict
from json import dump
from time import time


def stats(func, *argv):
    t0 = time()
    result = func(*argv)
    print('{:.6f} {}'.format(time() - t0, func.__name__))
    return result


def pre_processa():
    readr= reader(open('datasets/a/train.csv'))
    next(readr)
    data = [[int(item) for item in line] for line in readr]
    return [[line[2:] for line in data], [line[:2] for line in data]]


def pre_processb():
    data = [[[float(n) for n in line] for line in reader(open('datasets/b/features.csv'))],
            [[int(n) for n in line] for line in reader(open('datasets/b/labels.csv'))]]
    print("{} data points".format(len(data[0])))
    return data


def make_model(features, labels, depth_gas):
    return create_tree(features, labels, depth_gas)


def save_model(model):
    dump(model, open('tree.json', 'w+'),  indent=4)

def score_model(model, features, labels, depth):
    labels_pred = predict(model, features)
    correct = 0.0
    for i, _ in enumerate(labels_pred):
        correct += int(labels_pred[i] == labels[i])
    print(correct/len(labels))

GAS = 10

data = stats(pre_processb)
data.append(GAS)
model = stats(make_model, *data)
stats(save_model, model)
stats(score_model, model, *data)
