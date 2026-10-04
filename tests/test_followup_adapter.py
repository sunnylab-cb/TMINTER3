"""CPU tests for endpoint and null-aware pair adapters."""
import torch
from ppcbind.models.esm_pair import SymmetricOuterProductAdapter

torch.set_num_threads(2)
torch.manual_seed(49)
for endpoints, pooling in ((False, "softmax"), (True, "softmax"), (False, "null_softmax")):
    model = SymmetricOuterProductAdapter(16, 4, 8, 8, 0.0, 0, pooling, 4, 0.5, endpoints).eval()
    a, b = torch.randn(7, 16), torch.randn(11, 16)
    initial = model(a, b)
    assert torch.count_nonzero(initial[0]) == 0 and torch.count_nonzero(initial[1]) == 0
    torch.nn.init.normal_(model.site_residual[-1].weight, std=0.1)
    left, right, contact = model(a, b)
    right_swap, left_swap, contact_swap = model(b, a)
    torch.testing.assert_close(left, left_swap, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(right, right_swap, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(contact, contact_swap.T, atol=1e-6, rtol=1e-5)
    (left.square().mean() + right.square().mean() + contact.square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if pooling == "null_softmax":
        for length in (1, 10, 100):
            context, evidence = model._pool_one_axis(torch.ones(2, length, 8), torch.zeros(2, length))
            torch.testing.assert_close(evidence[:, 1], torch.full((2,), 0.5))
        _, low = model._pool_one_axis(torch.ones(2, 10, 8), torch.full((2, 10), -20.0))
        _, high = model._pool_one_axis(torch.ones(2, 10, 8), torch.full((2, 10), 20.0))
        assert (low[:, 1] < 1e-6).all() and (high[:, 1] > 0.999).all()
    print("PASS", endpoints, pooling)
