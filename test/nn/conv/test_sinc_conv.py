import pytest
import torch

from torch_geometric.nn import SINCConv


def test_sinc_conv():
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])

    conv = SINCConv(8, 16)
    assert str(conv) == 'SINCConv(8, 16)'

    out = conv(x, edge_index)
    assert out.size() == (4, 16)
    assert torch.isfinite(out).all()

    conv.reset_parameters()
    assert torch.isfinite(conv(x, edge_index)).all()


@pytest.mark.parametrize('aggr', ['sum', 'mean', 'max'])
@pytest.mark.parametrize('context_aggr', ['sum', 'mean', 'max'])
def test_sinc_conv_aggregators(aggr, context_aggr):
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    out = SINCConv(8, 6, aggr=aggr, context_aggr=context_aggr)(x, edge_index)
    assert out.size() == (4, 6)
    assert torch.isfinite(out).all()


def test_sinc_conv_hidden_self_loops_and_grad():
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])

    conv = SINCConv(8, 16, hidden_channels=32, add_self_loops=True)
    out = conv(x, edge_index)
    assert out.size() == (4, 16)

    out.sum().backward()
    for name, param in conv.named_parameters():
        assert param.grad is not None, f'no grad for {name}'
        assert torch.isfinite(param.grad).all()
