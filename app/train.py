import sys
print("🔍 sys.path:", sys.path)


# app/train.py
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from app.dataset import SignDataset      # ✅ 요거!
from app.model import SignLSTM           # ✅ 요것도 app.model 경로에서


def train_model(data_dir, label_map_path, output_model_path, input_size=225, hidden_size=128, batch_size=4, epochs=100):
    with open(label_map_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    
    label_set = set()
    for v in label_mapping.values():
        if isinstance(v, list):
            label_set.update(v)
        else:
            label_set.add(v)
    label_list = sorted(label_set)
    label2idx = {label: i for i, label in enumerate(label_list)}

    dataset = SignDataset(data_dir, label2idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SignLSTM(input_size, hidden_size, num_classes=len(label2idx))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            output = model(x_batch)
            loss = criterion(output, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), output_model_path)
    print(f"\n✅ 모델 저장 완료: {output_model_path}")