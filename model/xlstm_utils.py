
# Set up environment (run this before running the script)
#   conda env create -n xlstm -f environment_pt220cu121.yaml
#   conda activate xlstm
#   pip install xlstm
#   pip install -e .


import os
import torch

from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig, sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig


def get_xlstm_model(embedding_dim, num_blocks, conv1d_kernel_size, qkv_proj_blocksize, num_heads):

    cfg = xLSTMBlockStackConfig(mlstm_block=mLSTMBlockConfig(mlstm=mLSTMLayerConfig(conv1d_kernel_size=conv1d_kernel_size, qkv_proj_blocksize=qkv_proj_blocksize, num_heads=num_heads)),
                                slstm_block=sLSTMBlockConfig(slstm=sLSTMLayerConfig(backend="cuda", num_heads=num_heads, conv1d_kernel_size=conv1d_kernel_size, bias_init="powerlaw_blockdependent"),
                                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu")),
                                context_length=256, num_blocks=num_blocks, embedding_dim=embedding_dim, slstm_at=[1]
                                )
    xlstm_stack = xLSTMBlockStack(cfg)

    return xlstm_stack
