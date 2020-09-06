import torch
import torch.nn.functional as F


def cal_similarity(key_embeds,
                   ref_embeds,
                   method='dot_product',
                   temperature=-1):
    assert method in ['dot_product', 'cosine']

    if method == 'cosine':
        key_embeds = F.normalize(key_embeds, p=2, dim=1)
        ref_embeds = F.normalize(ref_embeds, p=2, dim=1)
        return torch.mm(key_embeds, ref_embeds.t())
    elif method == 'dot_product':
        if temperature > 0:
            dists = cal_similarity(key_embeds, ref_embeds, method='cosine')
            dists /= temperature
        else:
            return torch.mm(key_embeds, ref_embeds.t())
