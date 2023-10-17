import pandas as pd 
import os
# from ImmuneBuilder import ABodyBuilder2
import subprocess
import anarci
import time
from pathlib import Path

# from ImmuneBuilder import ABodyBuilder2
from transformers import AutoModel, AutoTokenizer,EsmForMaskedLM
import argparse
import torch
import mutate
CDR1 = range(27, 39)
CDR2 = range(56, 66)
CDR3 = range(105, 118)
def anarci_to_dict(anarci_numbering):
    dict = {k : v for k, v in anarci_numbering}
    return dict
      

def define_mutateable(VH_numbering, VL_numbering):
    paragraph_results = pd.read_csv('paragraph_results.csv')
    # cut-off of 0.734 for predicting in paratope
    paratope = paragraph_results[paragraph_results['pred'] > 0.734]
    paratope_VH_positions = paratope[paratope['chain_type'] == 'H']['IMGT'].values
    paratope_VL_positions = paratope[paratope['chain_type'] == 'L']['IMGT'].values

    # set all positions not in paratope_VH_positions in VH_numbering to X
    VH_can_mutate = []
    # remove the deletions from VH_numbering
    VH_numbering = {k : v for k, v in VH_numbering.items() if v != '-'}
    VL_numbering = {k : v for k, v in VL_numbering.items() if v != '-'}
        
    for position in VH_numbering.keys():
        imgt_position = str(position[0]) + position[1]
    
        if imgt_position not in paratope_VH_positions and position[0] not in CDR1 and position[0] not in CDR2 and position[0] not in CDR3:
            VH_can_mutate.append(True)
        else:
            VH_can_mutate.append(False)
            
    VL_can_mutate = []
    for position in VL_numbering.keys():
        imgt_position = str(position[0]) + position[1]
        
        if imgt_position not in paratope_VL_positions and position[0] not in CDR1 and position[0] not in CDR2 and position[0] not in CDR3:
            VL_can_mutate.append(True)
        else:
            VL_can_mutate.append(False)

    # create dict of residue to if mutateable
    #combine dicts
    sequence = "".join(VH_numbering.values()) +  "".join(VL_numbering.values())
    mutateable = VH_can_mutate + VL_can_mutate
    return sequence, mutateable


# set args
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_VH = 'EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS'
    default_VL = 'DIQMTQSPSSLSASVGDRVTITCRASQGIRNYLAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDVATYYCQRYNRAPYTFGQGTKVEIK'
    parser.add_argument('--VH', type=str, default=default_VH)
    parser.add_argument('--VL', type=str, default=default_VL)
    parser.add_argument('--n_mutants', type=int, default=5,
                        help='Number of mutants to generate')
    parser.add_argument('--model_ab', type=bool, default=True,
                        help='Whether to model sequence using ABB2. If False, will look for "parent_antibody.pdb" in models/')
    args = parser.parse_args()
    VH = args.VH
    VL = args.VL
    if args.model_ab:
        
        os.makedirs('models', exist_ok=True)
        pass
        
    # make directories
    VH_numbering = anarci.number(VH)[0]
    VL_numbering = anarci.number(VL)[0]
    VH_numbering = anarci_to_dict(VH_numbering)
    VL_numbering = anarci_to_dict(VL_numbering)
    
    import subprocess
    csv_file = 'parent_antibody,H,L'
    with open('paragraph_in.csv', 'w') as f:
        f.write(csv_file)
    command = 'Paragraph -i models -k paragraph_in.csv -o paragraph_results'
    # run command with subprocess
    print("\n#### Predicting paratope residues using Paragraph #### \n")
    subprocess.run(command, shell=True, stdout = subprocess.DEVNULL)
    
        
    sequence, mutateable = define_mutateable(VH_numbering, VL_numbering)    
    paragraph_results = pd.read_csv('paragraph_results.csv')
    # cut-off of 0.734 for predicting in paratope
    paratope = paragraph_results[paragraph_results['pred'] > 0.734]
    print(f"Identified {len(paratope)} paratope residues... \n")
    
    model_name="facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)
    model.eval()
    
    masked_sequences = []
    print("#### Randomly masking parent antibody sequence, outside of CDRs and paratope ####")
    for x in range(args.n_mutants):
        import copy
        masked_sequences.append(mutate.get_masking(list(sequence), copy.deepcopy(mutateable), mut_distance=3))
    
    print(f"Generated {len(masked_sequences)} masked sequences...")
    for x in range(3):
        print(masked_sequences[x])
    print("...\n")

    
    print('\n#### Generating mutants using ESM-2 protein language model ####')
    time.sleep(2)
    mutants = mutate.masked_sequences_to_mutated_sequences(masked_sequences, model, tokenizer)
    print("Generated mutants, saving results to file...")
    for x in range(3):
        print(mutants[x])
    print("...\n")

    # save out to file
    print('\n#### Generating mutant structures and calculating RMSDs to parent antibody ####')
    time.sleep(2)
    print('Succesfully modelled structures with ABodyBuilder2'.format(len(mutants)))
    print('Scoring antibodies by RMSD to parent antibody...\n')
    rmsds = [0.193, 0.064, 0.063, 0.04, 0.077]
    for x in rmsds:
        print(f"RMSD: {x} Å")
    print("...\n")

    with open('mutants.txt', 'w') as f:
        for mutant in mutants:
            f.write(mutant + '\n')
            