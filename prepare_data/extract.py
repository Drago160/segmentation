import pandas as pd
import torch
from torchvision import transforms as tr
import torch

from PIL import Image

def extract_pathes(mode):
    assert mode in ['train', 'dev', 'test']
    split = pd.read_csv('CVPPPSegmData/split.csv')
    split = split[split.split == mode]

    return split.img_path, split.sem_path, split.inst_path

def extract(mode):
    assert mode in ['train', 'dev', 'test']

    image_prepare = tr.Compose([
        tr.Resize(512),
        tr.CenterCrop(512),
        tr.ToTensor(),
    ])

    images = []
    sems = []
    insts = []

    img_pathes, sem_pathes, inst_pathes = extract_pathes(mode)
    for img_path, sem_path, inst_path in zip(img_pathes, sem_pathes, inst_pathes):
        img = Image.open('CVPPPSegmData/' + img_path)
        images.append(image_prepare(img)[:3])

        sem = Image.open('CVPPPSegmData/' + sem_path)
        sems.append(image_prepare(sem)[:3])

        inst = Image.open('CVPPPSegmData/' + inst_path)
        insts.append(image_prepare(inst)[:3])
    return torch.stack(images), torch.stack(sems), torch.stack(insts)


