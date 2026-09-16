from __future__ import annotations

from types import SimpleNamespace

import torch

from offline_gspo import modeling


class _Core(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)
        self.shapes: list[tuple[int, int]] = []

    def forward(self, *, input_ids, attention_mask, **_):
        self.shapes.append(tuple(input_ids.shape))
        assert torch.all(attention_mask == 1)
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Core()
        self.lm_head = torch.nn.Linear(4, 16, bias=False)


def test_selected_logps_forward_each_segment_without_padding(monkeypatch) -> None:
    def full_softmax(hidden, weight, ids, *, bias, temperature):
        logits = hidden @ weight.T / temperature
        if bias is not None:
            logits = logits + bias
        return torch.log_softmax(logits.float(), dim=-1).gather(
            -1, ids.unsqueeze(-1)
        ).squeeze(-1)

    monkeypatch.setattr(
        modeling.LigerFusedLinearPPOBase,
        "chunk_forward",
        staticmethod(full_softmax),
    )
    model = _Model()
    input_ids = torch.tensor([[1, 2, 3, 0, 0], [1, 4, 5, 6, 7]])
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    values = modeling.selected_token_logps(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_rows=torch.tensor([0, 0, 1]),
        target_positions=torch.tensor([1, 2, 4]),
        temperatures=torch.tensor([0.7, 1.0]),
    )
    assert values.shape == (3,)
    assert model.model.shapes == [(1, 3), (1, 5)]
    values.sum().backward()
    assert model.model.embedding.weight.grad is not None
    assert model.lm_head.weight.grad is not None
