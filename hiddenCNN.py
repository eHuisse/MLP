import numpy as np
from matplotlib import pyplot as plt
from char_viz import *


class HiddenNN(object):
    def __init__(self):
        self.weight = []
        self.number_of_label = 4

    # definition d'une fonction sigmoid
    def sigmoid(self, x):
        '''
        Compute sigmoid function
        :param x: Input
        :return: Sig(x)
        '''
        x[x < -100] = -100
        x[x > 100] = 100
        return (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))

    def sigmop(self, x):
        '''
        Compute derivative sigmoid function
        :param x: Input
        :return: Sig(x)
        '''
        return 1 - self.sigmoid(x) ** 2

    def biasize(self, exemples):
        ones = np.ones((1, exemples.shape[1]))
        return np.vstack((ones, exemples))

    def mlpclass(self, y):
        return np.argsort(y, axis=0)[-1, :]

    def mlpdef(self, nb_cells_layers, nb_input):
        '''
        Generate random weight matrix
        :param nb_cell:
        :param nb_input:
        :return:
        '''
        self.weight = []
        self.weight.append(np.random.rand(nb_cells_layers[0], nb_input + 1) * 2 - 1)
        for i in range(1, len(nb_cells_layers)):
            self.weight.append(np.random.rand(nb_cells_layers[i], nb_cells_layers[i-1] + 1) * 2 - 1)

        return self.weight

    def score(self, y, true_results):
        if len(y) != len(true_results):
            raise ('waited result and actual result are not align')
        tmp = np.equal(y, true_results)
        s = sum(tmp)
        return s, s / len(y)

    def mlperror(self, y, target):
        #print('target = ', str(target))
        return y - target

    def mlprun(self, weight_mat, exemples):
        tmp = []
        for i in range(len(weight_mat)):
            #print('weightmati : ' + str(weight_mat[i]))
            tmp.append(self.sigmoid(np.dot(weight_mat[i], self.biasize(exemples))))
            exemples = tmp[i]
            #print('tmp : ' + str(tmp))
        return tmp

    def label2target(self, c):
        class_count = len(np.unique(c))
        tmp = np.ones((class_count, len(c))) * -1

        for i in range(len(c)):
            tmp[c[i], i] = 1

        return tmp

    def deltaout(self, error, state):
        #print('sigmop : ' + str(error * self.sigmop(state)))
        return error*self.sigmop(state)

    def unbiasize_Weight(self, weight):
        return weight[:, 1:]

    def deltahidden(self, state, following_weight, following_delta):
        #print('state : ' + str(state))
        #print('followW : ' + str(following_weight))
        #print('followD : ' + str(following_delta))
        #print('test' +str(np.dot(np.transpose(self.unbiasize_Weight(following_weight)), following_delta)))
        return self.sigmop(state)*np.dot(np.transpose(self.unbiasize_Weight(following_weight)), following_delta)

    def mlp1partialQ(self, target, train_base):
        delta = []
        partialQ= []
        result = self.mlprun(self.weight, x)
        #print('result : ' + str(result))

        # Outlayer
        error = self.mlperror(result[-1], target)
        #print('error : ' + str(error))

        delta.insert(0, self.deltaout(error[-1], CNN.mlprun(self.weight, x)[-1]))
        #print('deltaout : ', delta)

        for i in range(2, len(self.weight)+1):
            # first hidden layer
            delta.insert(0, self.deltahidden(result[-i], self.weight[-i+1], delta[-i+1]))
            #print("delta_" + str(i) + " : " + str(delta))

        partialQ.append(np.dot(delta[0], np.transpose(self.biasize(train_base))))
        for i in range(1, len(result)):
            partialQ.append(np.dot(delta[i], np.transpose(self.biasize(result[i-1]))))
            #print("partialQ :" + str(partialQ))
        return partialQ

    def train(self, train_base, train_label, couches, conv_speed):
        error_evolve = 3*(10**-3)
        error_prev = 0

        self.number_of_label = len(np.unique(train_label))
        couches[-1] = self.number_of_label

        weight = self.mlpdef(couches, train_base.shape[0])
        #print('weight : ' + str(weight))

        target = self.label2target(train_label)
        #print('target : ' + str(target))

        for i in range(3000):
            deltaQ = self.mlp1partialQ(target, train_base)
            print('deltaQ : ' + str(deltaQ))
            for i in range(len(self.weight)):
                self.weight[i] = self.weight[i] - conv_speed * deltaQ[i]
            #print('newweight : ' + str(self.weight))
            #print('error_prev n: ' + str(sum(self.sqrerror(self.mlperror(self.mlprun(self.weight, train_base)[-1], target)))))
        return self.weight

    def sqrerror(self, vecterror):
        return 0.5 * sum(vecterror ** 2)

    def mlperror(self, y, target):
        #print('target = ', str(target))
        return y - target



if __name__ == "__main__":

    basetrain = np.load('basetrain.npy')
    basetest = np.load('basetest.npy')
    labeltrain = np.load('labeltrain.npy')
    labeltest = np.load('labeltest.npy')
    #print(basetest.shape)

    # 4 jeux d’entrees pour un reseau a 5 cellules :
    x = np.transpose(np.array([[-1, 0, 0, 0, 1], [0, 1, 0, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [-1, 0, 0, 0, 1]]))
    labelx = np.array([0, 1, 2, 2, 0])

    couches = [3]
    nb_input = 5

    CNN = HiddenNN()

    #weight = CNN.mlpdef(couches, nb_input)
    #print('weight : ' + str(weight))
    #result = CNN.mlprun(weight, x)
    #print('result : ' + str(result))
    #target = CNN.label2target(labelx)
    #print('target : ' + str(target))

    #Outlayer
    #error = CNN.mlperror(result[-1], target)
    #print('error : ' + str(error))

    #deltaout = CNN.deltaout(error[-1], CNN.mlprun(weight, x)[-1])
    #print('deltaout : ', deltaout)

    #first hidden layer
    #delta_1 = CNN.deltahidden(CNN.mlprun(weight, x)[-2], weight[-1], deltaout)
    #print("delta_1" + str(delta_1))

    #delta_2 = CNN.deltahidden(CNN.mlprun(weight, x)[-3], weight[-2], delta_1)
    #print("delta_2" + str(delta_2))

    CNN.train(x, labelx, couches, 0.1)
    print(CNN.mlprun(CNN.weight, x))
    print(CNN.label2target(labelx))