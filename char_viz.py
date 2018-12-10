# import librairie numpy et matplotlib:
import numpy as np
import matplotlib.pyplot as plt


# affichage:
def char_viz(char):
    if len(char) != 784:
        raise('This is note a letter')

    plt.figure(1, figsize=(3, 3))
    plt.imshow(char.reshape(28,28), cmap=plt.cm.gray_r)
    plt.show()
    return 0