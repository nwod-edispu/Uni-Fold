import pickle


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
    a = gen()
    print(next(a))


if __name__ == "__main__":
    main()
