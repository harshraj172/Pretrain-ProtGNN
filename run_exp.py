import json
import numpy as np

import torch
import os.path as osp
import GCL.augmentors as A
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator
from torch_geometric.data import Data, DataLoader


def train(encoder_model, contrast_model, dataloader, optimizer, topk=1):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.edge_attr, data.batch)
        g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
        loss, acc = contrast_model(g1=g1, g2=g2, batch=data.batch, topk=topk)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    print(f"Top-{topk} Accuracy: {acc}")
    return epoch_loss


def test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        _, g, _, _, _, _ = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    result = SVMEvaluator(linear=True)(x, y, split)
    return result


def main():
    path = "mydata.json"
    batch_size = 128
    topk = (1,)
    
    data_lst = prepare_data(path, batch_size)
    dataloader = DataLoader(data_lst, batch_size=batch_size)
    node_dim = max(data_lst[0].x.size(-1), 1)
    edge_dim = max(data_lst[0].edge_attr.size(-1), 1)

    aug1 = A.Identity()
    aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=100),
                           A.NodeDropping(pn=0.1),
                           A.FeatureMasking(pf=0.1),
                           A.EdgeRemoving(pe=0.1)], 1)
    gconv = GConv(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=32, num_layers=5).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    contrast_model = DualBranchContrast(loss=InfoNCE(tau=0.2), mode='G2G').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)

    with tqdm(total=100, desc='(T)') as pbar:
        for epoch in range(1, 101):
            loss = train(encoder_model, contrast_model, dataloader, optimizer, topk)
            pbar.set_postfix({'loss': loss})
            pbar.update()
        print(f"loss = {loss}")

#     test_result = test(encoder_model, dataloader)
#     print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    device = torch.device('cuda')
    main()
