import torch
import torch.nn.functional as F


def nt_xent_loss(z1, z2, temperature: float = 0.5):
    batch_size = z1.size(0)

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    z = torch.cat([z1, z2], dim=0)  

    similarity = torch.matmul(z, z.T)  
    similarity = similarity / temperature

    mask = torch.eye(2 * batch_size, device=z.device).bool()
    similarity = similarity.masked_fill(mask, float("-inf"))

    targets = torch.arange(batch_size, device=z.device)
    targets = torch.cat([targets + batch_size, targets], dim=0)

    loss = F.cross_entropy(similarity, targets)
    return loss