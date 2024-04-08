import torch

from prepare_data.extract import extract
from prepare_data.devide_instances import devide

from utils.tools import insert_to_zero_dim


def add_background(inst, sem):
    return insert_to_zero_dim(inst, (sem[0] == 0))

def get_dataset(mode):
    assert mode in ['train', 'dev', 'test']
    imgs, sems, insts = extract(mode)
    insts = [add_background(devide(inst), sem)
             for inst, sem in zip(insts, sems)]
    return imgs, insts
