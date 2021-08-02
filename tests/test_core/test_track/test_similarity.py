import torch


def test_embed_similarity():
    from mmtrack.core import embed_similarity
    key_embeds = torch.randn(20, 256)
    ref_embeds = torch.randn(10, 256)

    sims = embed_similarity(
        key_embeds,
        ref_embeds,
        method='dot_product',
        temperature=-1,
        transpose=True)
    assert sims.size() == (20, 10)

    sims = embed_similarity(
        key_embeds,
        ref_embeds.t(),
        method='dot_product',
        temperature=-1,
        transpose=False)
    assert sims.size() == (20, 10)

    sims = embed_similarity(
        key_embeds,
        ref_embeds,
        method='dot_product',
        temperature=0.07,
        transpose=True)
    assert sims.size() == (20, 10)

    sims = embed_similarity(
        key_embeds,
        ref_embeds,
        method='cosine',
        temperature=-1,
        transpose=True)
    assert sims.size() == (20, 10)
    assert sims.max() <= 1
