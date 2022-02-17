import pickle


def main():
    feats = pickle.load(open("G://repo//Uni-Fold//example_data//features//101m_1_A//features.pkl", 'rb'))
    print(feats)


if __name__ == "__main__":
    main()
