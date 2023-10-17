from copy import deepcopy
import copy
from itertools import combinations
from re import M
import typing
from typing import List
import torch
import numpy as np
import random
from transformers import  AutoTokenizer, AutoModel

def numbering_to_chains(numbering) -> List[str]:
    """
    Takes the output of anarci.anarci and returns a list of heavy and light chains seperated by a '/'.
    """
    sequences = numbering[0]
    seperated_sequences = []
    for sequence in sequences:
        heavy_light = []
        for chain in sequence:
            chain = chain[0]
            for number in chain:
                residue = number[1]
                if residue != '-':
                    heavy_light.append(residue)
            heavy_light.append('/')
        heavy_light = "".join(heavy_light[:-1])
        seperated_sequences.append(heavy_light)
    return seperated_sequences

def masked_sequences_to_mutated_sequences(masked_sequences : List[str], model : AutoModel, tokenizer : AutoTokenizer) -> List[str]:
    """
    Takes in a sequence with <mask> tokens and replaces them with the most likely tokens. One sequence at a time.
    """

    mutated_sequences = []
    # Tokenize the sequence and get the output from the model
    for sequence in masked_sequences:
        inputs = tokenizer(sequence, return_tensors="pt")
        # get the index of masked tokens in the sequence
        masked_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]
        # remove 1 from each masked_index to account for the <s> token
        masked_index = masked_index - 1
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0,1:-1, :]
        logits.shape
        # for each masked token, get the most likely token and replace it in the original sequence
        # convert the <mask> to a single character
        mutated_sequence = sequence.replace('<mask>', '?')
        for index in masked_index:
            max_token = torch.argmax(logits[index])
            max_token = tokenizer.convert_ids_to_tokens(max_token.item())
            mutated_sequence = mutated_sequence[:index] + max_token + mutated_sequence[index+1:]
        mutated_sequences.append(mutated_sequence)
    return mutated_sequences

def generate_all_maskings(sequences : List[str], mutateable_list : List[bool], mut_distance=1) -> List[str]:
    
    masked_sequences = []
    
    for sequence in sequences:
        for i in range(len(mutateable_list)):
            # check if we want to mutate this residue
            if mutateable_list[i] == True:
                # change it to a mask
                masked_sequence = sequence[:i] + '?' + sequence[i+1:]
                masked_sequences.append(masked_sequence)

    
    if mut_distance == 1:
        # replace ? with <mask>
        masked_sequences = [sequence.replace('?', '<mask>') for sequence in masked_sequences]
        print(f"Generated {len(masked_sequences)} with {len(set(masked_sequences))} unique sequences with a mutational distance of {mut_distance}")
        return list(set(masked_sequences))
    
    else:
        return generate_all_maskings(masked_sequences, mutateable_list, mut_distance=mut_distance-1)
    
def get_masking(sequence : List[str], mutateable_list : List[bool], mut_distance=1) -> str:
    """
    Return a sequence with mut_distance number of <mask> tokens. Only mutate those with a True value.
    """
    if mut_distance > 0:
        indices_of_true = [index for index, value in enumerate(mutateable_list) if value]
        # chose random index to mutate
        try:
            indices_to_mutate = random.choice(indices_of_true)
        except IndexError:
            print(indices_of_true)
            print(mutateable_list)
            
        # mask the sequence
        sequence[indices_to_mutate] = '?'
        mutateable_list[indices_to_mutate] = False
        return get_masking(sequence, mutateable_list, mut_distance=mut_distance-1)
    else:
        sequence = "".join(sequence)
        sequence = sequence.replace('?', '<mask>')
        return sequence
    

        
    
