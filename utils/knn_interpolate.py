from pytorch3d.ops import knn_points
import torch

def knn_interpolate(x,pos1,pos2, k=3):
    
    pos1=torch.cat([torch.zeros_like(pos1)[:,0].reshape(-1,1), pos1],-1)[None]
    pos2=torch.cat([torch.zeros_like(pos2)[:,0].reshape(-1,1), pos2],-1)[None]
    
    knn=knn_points(pos2,pos1, K=k)
    idx=knn.idx[0]
    dists=knn.dists[0]
 
        
    weights = 1.0 / (dists + 1e-8)
    weights = weights / weights.sum(dim=1, keepdim=True)

   

    feats_knn = x[idx] 
    

    interpolated = (weights[..., None] * feats_knn).sum(dim=1)
    return interpolated
