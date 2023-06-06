from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('custom_gnn')
def custom_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """

    # Use residual connections between the GNN layers.
    cfg.gnn.residual = False

    # Positional encodings argument group
    cfg.gt = CN()
    # Type of Graph Transformer layer to use
    cfg.gt.layer_type = 'test'
    # Number of Transformer layers in the model
    cfg.gt.layers = 3
    # Number of attention heads in the Graph Transformer
    cfg.gt.n_heads = 8
    # Size of the hidden node and edge representation
    cfg.gt.dim_hidden = 64
    # statistics argument group
    cfg.statistics = CN()
    # Memory Usage
    cfg.statistics.memory = 0
    # Total Time
    cfg.statistics.total_time = 0
    # Total Time Std
    cfg.statistics.total_time_std = 0
    # Avg Time
    cfg.statistics.avg_time = 0
    # Avg Time Std
    cfg.statistics.avg_time_std = 0
