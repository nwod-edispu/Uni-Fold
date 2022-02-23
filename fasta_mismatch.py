import os
from unifold.data.mmcif_parsing import MmcifObject
from unifold.data.mmcif_parsing import parse as parse_mmcif_string
from unifold.data.pipeline import FeatureDict
from unifold.data.templates import _get_atom_positions as get_atom_positions
from Bio.PDB import protein_letters_3to1


def cif_to_fasta(mmcif_object: MmcifObject,
                 chain_id: str) -> str:
    residues = mmcif_object.seqres_to_structure[chain_id]
    residue_names = [residues[t].name for t in range(len(residues))]
    residue_letters = [protein_letters_3to1.get(n, 'X') for n in residue_names]
    filter_out_triple_letters = lambda x: x if len(x) == 1 else 'X'
    fasta_string = ''.join([filter_out_triple_letters(n) for n in residue_letters])
    return fasta_string


if __name__ == "__main__":
    fasta_dir = "/home/hanj/workplace/unifold_dataset/training_set/fasta/"
    cif_dir = "/home/hanj/workplace/unifold_dataset/training_set/mmcif/"
    cnt = 0
    for fasta in os.listdir(fasta_dir):
        parts = fasta.split("_")
        pdb_id = parts[0].strip()
        chain_id = parts[2].split(".")[0].strip()
        cif_path = os.path.join(cif_dir, pdb_id + ".cif")
        cif_string = open(cif_path, 'r').read()
        print(fasta)
        print(cif_path)
        # parse cif string
        mmcif_obj = parse_mmcif_string(
            file_id=pdb_id, mmcif_string=cif_string).mmcif_object
        # fetch useful labels
        if mmcif_obj is not None:
            # directly parses sequence from fasta. should be consistent to 'aatype' in input features (from .fasta or .pkl)
            sequence = cif_to_fasta(mmcif_obj, chain_id)

            # hj: The rosetta sequence is cropped, so we need to align the fasta sequence
            fasta_path = os.path.join(fasta_dir, fasta)
            with open(fasta_path) as f:
                f.readline()
                origin_seq = f.readline().strip()
            begin = sequence.find(origin_seq, 0, len(sequence))
            if begin == -1:
                cnt += 1
    print("fasta with no cif: ", cnt)
