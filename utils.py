import torch
from structgen import protein_features 

def ProteinFeatures(top_k=30, num_rbf=16, features_type='full', direction='bidirectional'):
    features = protein_features.ProteinFeatures(
            top_k=top_k, num_rbf=num_rbf,
            features_type=features_type,
            direction=direction
    )
    return features

def completize(batch, batch_idx=1):
    L_max = len(batch['coords']['CA'])
    X = np.zeros([batch_idx, L_max, 4, 3])
    mask = np.zeros([batch_idx, L_max], dtype=np.float32)

    # Build the batch
    x = np.stack([batch['coords'][c] for c in ['N', 'CA', 'C', 'O']], 1)
    X[batch_idx-1,:len(x),:,:] = x
    mask[batch_idx-1, :L_max] = 1.

    # Remove NaN coords
    mask = mask * np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    isnan = np.isnan(X)
    X[isnan] = 0.

    # Conversion
    X = torch.from_numpy(X).float().cuda()
    mask = torch.from_numpy(mask).float().cuda()
    return X, mask
  
def to_torch_geom(E, E_idx):
    # get edge indices
    node1 = torch.tensor([], dtype=torch.long)
    for idx in range(E_idx.size(-2)):
        node1 = torch.cat((node1, torch.full((1, E_idx.size(-1)), idx)), axis=1)
    node1 = node1.repeat((E_idx.size(0), 1)).to(device)
    node2 = E_idx.view(E_idx.size(0), E_idx.size(1)*E_idx.size(2)).to(device)
    E_idx = torch.stack((node1, node2), 1).to(device)

    # get edge attr
    E = E.view(E.size(0), E.size(1)*E.size(2), E.size(3)).to(device)
   
    return E, E_idx.type(torch.LongTensor)
  
def prepare_data(path, batch_size):
    X = read_json(path)
    features = ProteinFeatures()
    data_lst = []
    for i, x in enumerate(X):
        hchain = completize(x) 
        V, E, E_idx = features(hchain[0], hchain[-1]) 
        E, E_idx = to_torch_geom(E, E_idx)
        data_lst.append(Data(x=V[0, :, :], edge_index=E_idx[0, :, :], edge_attr=E[0, :, :], batch=torch.tensor([int(i/batch_size)])))
    return data_lst
  
def accuracy(sim, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    mask = torch.eye(sim.shape[0], dtype=torch.bool).to(device)
    neg = sim[~mask].view(sim.shape[0], -1)
    pos = sim[mask].view(sim.shape[0], -1)
    output = torch.cat([pos, neg], dim=1)
    target = torch.zeros(output.shape[0], dtype=torch.long).to(device)
        
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(float(correct_k.mul_(100.0 / batch_size)))
        return res
