import os
import random
import numpy as np
from PIL import Image
import time

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset

__all__ = ['dataset', 'split_dataset']

class Sealer(Dataset):
    def __init__(self, args, root_dir, mode, transform=None):
        self.args = args
        self.root_dir = root_dir
        self.transform = transform
        self.NUM_CLASSES = 1

        self.image_paths = []
        self.label_paths = []

        target = 'ori_target' if mode == 'original' else 'target'

        # 하위 폴더 탐색하여 이미지 파일 경로 수집
        for root, _, files in os.walk(root_dir):
            if mode in root:
                for file in files:
                    if file.endswith(".tif"):
                        image_path = os.path.join(root, file)
                        self.image_paths.append(image_path)
            
            if target == os.path.split(root)[-1]:
                for file in files:
                    if file.endswith(".tif"):
                        label_path = os.path.join(root, file)
                        self.label_paths.append(label_path)


    def __getitem__(self, index):
        # 이미지와 레이블 로드
        image = Image.open(self.image_paths[index])
        label = Image.open(self.label_paths[index])

        # 전처리 적용
        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)[0]

        return image, label

    def __len__(self):
        return len(self.image_paths)
    

def split_dataset(dataset, train_ratio):
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)

    # 데이터셋의 인덱스를 랜덤하게 섞음
    indices = list(range(dataset_size))
    random.shuffle(indices)

    # 학습 세트와 테스트 세트로 분할
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, test_dataset

    
if __name__ == '__main__':
    # 데이터 전처리를 위한 변환기 정의
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        ])

    # 커스텀 데이터셋 객체 생성
    root_dir = "train_data"  # 데이터셋이 있는 루트 폴더 경로
    datasets = Sealer(root_dir, transform=transform)

    train_ratio = 0.8  # 학습 세트의 비율 (0.8은 80%의 데이터가 학습 세트로 사용됨을 의미)
    train_dataset, test_dataset = split_dataset(datasets, train_ratio)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)