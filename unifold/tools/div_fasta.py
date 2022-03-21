import os


def main():
    data_dir = "G:/23333/casp14"
    with open(os.path.join(data_dir, "casp14.fasta")) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if i == len(lines) - 1:
                break
            if i % 2 == 1:
                continue
            line1 = lines[i]
            line2 = lines[i + 1]
            name = line1.split()[0][1:]
            with open(os.path.join(data_dir, name + ".fasta"), "w") as fw:
                fw.write(line1)
                fw.write(line2)


if __name__ == "__main__":
    main()
