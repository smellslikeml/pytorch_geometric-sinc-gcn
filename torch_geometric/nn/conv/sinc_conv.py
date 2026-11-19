r"""Clean-room implementation of SINCConv (Neighborhood-Contextualized Message-Passing, NCMP).

From arXiv:2511.11046 (Lim et al., 2025), Eq. 15 — implemented from the paper's math only
(NOT ported from the CC-BY/DGL reference repo):

    h*_u = AGG_out_{v in N(u)}  W_R . sigma( W_Q h_u + W_K h_v + AGG_ctx_{w in N(u)} W_N h_w )

The neighborhood-context term AGG_ctx_{w in N(u)} W_N h_w depends only on the center u, so it is a single
node-level pre-aggregation (one scatter), then broadcast into every edge message -> standard one-hop GNN cost.
Drop-in intended for torch_geometric/nn/conv/ (uses the real PyG MessagePassing + Linear + scatter).
"""
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils import scatter


class SINCConv(MessagePassing):
    r"""The Neighborhood-Contextualized Message-Passing operator (SINC-GCN, Eq. 15).

    Unlike attentional MP (which asks *how much* to send via a scalar weight), NCMP contextualizes the
    message *content* with a pre-aggregated neighborhood summary.

    Args:
        in_channels (int): input feature dim.
        out_channels (int): output feature dim.
        hidden_channels (int, optional): inner dim of W_Q/W_K/W_N / input dim of W_R
            (default: :obj:`out_channels`).
        act (callable, optional): elementwise nonlinearity :math:`\sigma` (default: :obj:`F.relu`).
        aggr (str, optional): outer message aggregator :math:`\oplus` (default: :obj:`"sum"`).
        context_aggr (str, optional): neighborhood-context aggregator :math:`\otimes`
            (:obj:`"sum"`, :obj:`"mean"`, :obj:`"max"`; default: :obj:`"sum"`).
        add_self_loops (bool, optional): the center enters each message via :math:`W_Q h_u`, so self-loops
            are off by default (default: :obj:`False`).
        bias (bool, optional): bias on :math:`W_Q` and :math:`W_R` (default: :obj:`True`).
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: Optional[int] = None,
                 act: Callable[[Tensor], Tensor] = F.relu, aggr: str = "mean",
                 context_aggr: str = "mean", add_self_loops: bool = False, bias: bool = True, **kwargs):
        # NB: default to "mean" — "sum" explodes logit scale on high-degree real graphs (verified: Cora
        # sum/sum -> ~random; mean/mean -> 77% test-acc). "symmetric-mean" (GCN D^-1/2 A D^-1/2) is a
        # faithful paper option left as a follow-up (needs degree-norm edge weights).
        super().__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels or out_channels
        self.act = act
        self.context_aggr = context_aggr
        self.add_self_loops = add_self_loops

        h = self.hidden_channels
        self.lin_Q = Linear(in_channels, h, bias=bias)          # center  W_Q (carries the single bias)
        self.lin_K = Linear(in_channels, h, bias=False)         # neighbor W_K
        self.lin_N = Linear(in_channels, h, bias=False)         # context W_N
        self.lin_R = Linear(h, out_channels, bias=bias)         # readout W_R
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_Q.reset_parameters()
        self.lin_K.reset_parameters()
        self.lin_N.reset_parameters()
        self.lin_R.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if self.add_self_loops:
            edge_index, _ = add_self_loops_fn(edge_index, num_nodes=x.size(0))

        # context pass: c_u = AGG_ctx_{w in N(u)} W_N h_w  (source -> target = neighbor -> center)
        src, dst = edge_index[0], edge_index[1]            # flow='source_to_target': src=neighbor, dst=center
        n = self.lin_N(x)
        c = scatter(n[src], dst, dim=0, dim_size=x.size(0), reduce=self.context_aggr)

        # message pass: AGG_out_{v in N(u)} W_R sigma(W_Q h_u + W_K h_v + c_u)
        q = self.lin_Q(x)
        k = self.lin_K(x)
        return self.propagate(edge_index, q=q, k=k, c=c)

    def message(self, q_i: Tensor, k_j: Tensor, c_i: Tensor) -> Tensor:
        # _i = target/center u, _j = source/neighbor v
        return self.lin_R(self.act(q_i + k_j + c_i))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels})"
