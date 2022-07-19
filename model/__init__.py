from model.hdr_transformer import HDRTransformer


def fetch_net(params):
    if params.net_type == "hdr_transformer":
        net = HDRTransformer(embed_dim=60, depths=[6, 6, 6], num_heads=[6, 6, 6], mlp_ratio=2, in_chans=6)

    else:
        raise NotImplementedError
    return net