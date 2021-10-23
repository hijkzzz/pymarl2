import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
    Examples::
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: str = 'cpu') -> None:
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.device = device
        self.u_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.s_w = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.u_b = nn.Parameter(torch.Tensor(out_features))
            self.s_b = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.u_w, a=math.sqrt(3 / self.in_features))
        nn.init.constant_(self.s_w, 0.017)
        if self.bias is not None:
            nn.init.uniform_(self.u_b, a=math.sqrt(3 / self.in_features))
            nn.init.constant_(self.s_b, 0.017)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training: 
            e_w = torch.randn(self.s_w.shape, device=self.device)
            e_b = torch.randn(self.s_b.shape, device=self.device)
            weight = self.u_w + (self.s_w * e_w)
            bias = self.u_b + (self.s_b * e_b)
        else:
            weight = self.u_w
            bias = self.u_b
        return F.linear(input, weight, bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )