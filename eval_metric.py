import os
from unifold.tools.tmscore import eval_score

if __name__ == "__main__":
    target_dir = "/home/hanj/workplace/dataset/casp14_dataset/target/"
    predict_dir = "/home/hanj/workplace/dataset/casp14_dataset/predict/"

    score = 0.0
    for f in os.listdir(target_dir):
        ref_pdb = os.path.join(target_dir, f)
        name = os.path.splitext(f)[0]
        pred_pdb = os.path.join(predict_dir, name, "unrelaxed_unifold.pdb")
        tmscore = eval_score(ref_pdb, pred_pdb)
        print(name, ":", tmscore.get_tm_score())
        score += tmscore.get_tm_score()

    print(score / len(os.listdir(target_dir)))
