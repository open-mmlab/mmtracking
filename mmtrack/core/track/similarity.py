import torch
import torch.nn.functional as F


def embed_similarity(key_embeds,
                     ref_embeds,
                     method='dot_product',
                     temperature=-1,
                     transpose=True):
    assert method in ['dot_product', 'cosine']

    if key_embeds.size(0) == 0 or ref_embeds.size(0) == 0:
        return torch.zeros((key_embeds.size(0), ref_embeds.size(0)),
                           device=key_embeds.device)

    if method == 'cosine':
        key_embeds = F.normalize(key_embeds, p=2, dim=1)
        ref_embeds = F.normalize(ref_embeds, p=2, dim=1)
    elif method == 'dot_product':
        if temperature > 0:
            sims = embed_similarity(
                key_embeds, ref_embeds, method='cosine', transpose=transpose)
            sims /= temperature
            return sims
    else:
        raise NotImplementedError()

    if transpose:
        ref_embeds = ref_embeds.t()
    return torch.mm(key_embeds, ref_embeds)
