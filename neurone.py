import numpy as np
from matplotlib import pyplot as plt
from char_viz import *



class neuralnetwork:
    def __init__(self, conv_speed=0.05):
        self.weight = np.array([])
        self.number_of_label = 0
        self.convergence_speed = conv_speed

    # definition d'une fonction sigmoid
    def sigmoid(self, x):
        '''
        Compute sigmoid function
        :param x: Input
        :param derivative: is derivate?
        :return: Sig(x)
        '''
        x[x < -2] = -2
        x[x > 2] = 2
        return (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))

    def sigmop(self, x):
        return 1 - self.sigmoid(x) ** 2

    def mlp1def(self, nb_input, nb_cell): #= self.number_of_label
        '''
        Generate random weight matrix
        :param nb_cell:
        :param nb_input:
        :return:
        '''
        return np.random.rand(nb_cell, nb_input + 1) * 2 - 1

    def mlp1run(self, weight_mat, exemples):
        return self.sigmoid(np.dot(weight_mat, self.biasize(exemples)))

    def mlp1runop(self, weight_mat, exemples):
        return self.sigmop(np.dot(weight_mat, self.biasize(exemples)))

    def mlpclass(self, y):
        return np.argsort(y, axis=0)[-1, :]

    def biasize(self, exemples):
        ones = np.ones((1, exemples.shape[1]))
        return np.vstack((ones, exemples))

    def score(self, y, true_results):
        if len(y) != len(true_results):
            raise ('waited result and actual result are not align')
        tmp = np.equal(y, true_results)
        s = sum(tmp)
        return s, s / len(y)

    def label2target(self, c):
        class_count = len(np.unique(c))
        tmp = np.ones((class_count, len(c))) * -1

        for i in range(len(c)):
            tmp[c[i], i] = 1
        return tmp

    def mlperror(self, y, target):
        #print('target = ', str(target))
        return y - target

    def mlp1partialQ(self, target, train_base):
        # on pese les entrées avec les poids
        #print('target : ' +str(target))
        tmp_result = self.mlp1run(self.weight, train_base)
        #print('tmp_result : ' + str(np.dot(self.weight, self.biasize(train_base))))
        # On calcule l'erreur sur chaque pesées par rapport au target
        error = self.mlperror(tmp_result, target)
        #print('error : ' + str(error))
        # On calcule la derivé des pesés avec la fonction de decision
        sigmaPrime = self.mlp1runop(self.weight, train_base)
        #print('sigmaPrime : ' + str(sigmaPrime))
        #print('tmp : ' +str(np.dot(sigmaPrime, np.transpose(self.biasize(train_base)))))
        deltaQ = error * sigmaPrime
        #print('deltaQ : ' + str(deltaQ))
        deltaQ = np.dot(deltaQ, np.transpose(self.biasize(train_base)))
        #print('deltaQ : ' + str(deltaQ))
        return deltaQ

    def sqrerror(self, vecterror):
        return 0.5 * sum(vecterror ** 2)

    def train(self, train_base, train_label):
        error_evolve = 3*(10**-3)
        error_prev = 0

        self.number_of_label = len(np.unique(train_label))
        #print('Nb label : ' + str(self.number_of_label))

        self.weight = NN.mlp1def(train_base.shape[0], self.number_of_label)
        target = self.label2target(train_label)

        #print('weight : ' +str(self.weight))
        #print('trainbase : ' +str(train_base))
        #while abs(error_prev - sum(self.sqrerror(self.mlperror(self.mlp1run(self.weight, train_base), target)))) > error_evolve:
            #error_prev = sum(self.sqrerror(self.mlperror(self.mlp1run(self.weight, train_base), target)))
        for i in range(100):
            deltaQ = self.mlp1partialQ(target, train_base)
            #print('deltaQ : ' + str(deltaQ))
            self.weight = self.weight - self.convergence_speed * deltaQ
            #print('newweight : ' + str(self.weight))
            print('error_prev n: ' + str(sum(self.sqrerror(self.mlperror(self.mlp1run(self.weight, train_base), target)))))
        return self.weight

    def setweight(self, weight):
        self.weight = weight

    def evaluate(self, base):
        return self.mlpclass(self.mlp1run(self.weight, base))


if __name__ == '__main__':

    basetrain = np.load('basetrain.npy')
    basetest = np.load('basetest.npy')
    labeltrain = np.load('labeltrain.npy')
    labeltest = np.load('labeltest.npy')
    #print(basetest.shape)

    # 3 jeux d’entrees pour un reseau a 5 cellules :
    x = np.array([[-1, 0, 0, 0, 1], [0, 1, 0, 1, 1], [0, 0, 1, 0, 0]])
    labelx = np.array([0, 1, 2])
    # rangement des exemples en colonne:
    x = np.transpose(x)

    test = basetest
    NN = neuralnetwork()

    #weight = NN.mlp1def(test.shape[0], 10) #attention nb neurone = a nb classe
    #print(weight)
    #result = NN.mlp1run(weight, test)
    #print('mlp res : ' + str(result))
    #print('mlp class : ' + str(NN.mlpclass(result)))
    #testtarg = NN.label2target(labeltest)
    #error = NN.mlperror(result, testtarg)
    #print("error : "+str((NN.mlperror(result, testtarg))))
    #print("quaderror : "+str(NN.sqrerror(error)))
    np.save('weight', NN.train(basetrain, labeltrain))
    test = NN.evaluate(basetest)
    print(NN.score(test, labeltest))

