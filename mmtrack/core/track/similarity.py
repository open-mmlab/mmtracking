import torch
import torch.nn.functional as F


def embed_similarity(key_embeds,
                     ref_embeds,
                     method='dot_product',
                     temperature=-1,
                     transpose=True):
    """Calculate feature similarity from embeddings.

    Args:
        key_embeds (Tensor): Shape (N1, C).
        ref_embeds (Tensor): Shape (N2, C) or (C, N2).
        method (str, optional): Method to calculate the similarity,
            options are 'dot_product' and 'cosine'. Defaults to
            'dot_product'.
        temperature (int, optional): Softmax temperature. Defaults to -1.
        transpose (bool, optional): Whether transpose `ref_embeds`.
            Defaults to True.

    Returns:
        Tensor: Similarity matrix of shape (N1, N2).
    """
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
