import pickle
import numpy as np
import matplotlib.pyplot as plt


def gen():
    index = [5, 1, 2]
    data = [2, 3, 4]
    i = 0
    while True:
        try:
            a = data[index[i]]
            i += 1
        except IndexError:
            i += 1
            print("error")
            continue
        yield a


def main():
    data = np.load("G://6years//out//demo_train_curve.npy")
    plt.plot(data[:, 0], data[:, 1])
    plt.show()


if __name__ == "__main__":
    main()
