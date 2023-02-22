import torch

from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import roc_auc_score

from tqdm import tqdm

from constants import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_train_test_loaders(train_data, test_data) -> list[LinkNeighborLoader, LinkNeighborLoader]:
    train_loader = LinkNeighborLoader(
        train_data,
        num_neighbors=n_neighbors,
        batch_size=batch_size,
        edge_label_index=train_data.edge_label_index,
        neg_sampling_ratio=1.0,
        #shuffle=True,
    )

    test_loader = LinkNeighborLoader(
        test_data,
        num_neighbors=n_neighbors,
        batch_size=batch_size,
        edge_label_index=test_data.edge_label_index,
        neg_sampling_ratio=1.0,
        #shuffle=True,
    )
    return train_loader, test_loader


def train(model, optimizer, criterion, loader: LinkNeighborLoader) -> float:
    model.train()
    loss = 0
    for batch in tqdm(loader):
        optimizer.zero_grad()        
        batch.to(device)

        z = model.encode(batch.x, batch.edge_index)
        out = model.decode(z, batch.edge_label_index).view(-1)
        loss = criterion(out, batch.edge_label)
        loss.backward()
        optimizer.step()
    return loss

@torch.no_grad()
def test(model, loader: LinkNeighborLoader) -> float:
    model.eval()
    aucs = 0
    for batch in tqdm(loader):
        batch.to(device)
        
        z = model.encode(batch.x, batch.edge_index)
        out = model.decode(z, batch.edge_label_index).view(-1)
        aucs += roc_auc_score(batch.edge_label.cpu().numpy(), out.cpu().numpy())
    return aucs / len(loader)
