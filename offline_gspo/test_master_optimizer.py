from __future__ import annotations

import json

import torch

from offline_gspo.train_offline_gspo import (
    FP32MasterOptimizer,
    load_resume_checkpoint,
    rollback_metric_log,
    save_resume_checkpoint,
)


def test_fp32_master_accumulates_updates_below_one_bf16_ulp() -> None:
    actor = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        actor.weight.fill_(0.01)
    initial_actor = actor.weight.detach().clone()
    optimizer = FP32MasterOptimizer(actor, lr=5e-8, weight_decay=0.0)
    _, master = optimizer.pairs[0]
    initial_master = master.detach().clone()

    # A direct BF16 AdamW step at this LR is bit-identical.  The persistent
    # FP32 master crosses that quantization boundary after enough tiny steps.
    for _ in range(2_000):
        actor.weight.grad = torch.ones_like(actor.weight)
        optimizer.accumulate_model_grads()
        optimizer.optimizer.step()
        optimizer.sync_actor()
        optimizer.zero_grad()

    assert optimizer.optimizer.state[master]["exp_avg"].dtype == torch.float32
    assert not torch.equal(master.detach(), initial_master)
    assert not torch.equal(actor.weight.detach(), initial_actor)
    assert actor.weight.dtype == torch.bfloat16


def test_resume_round_trip_preserves_fp32_master_and_adam_state(tmp_path) -> None:
    actor = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    optimizer = FP32MasterOptimizer(actor, lr=1e-4, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer.optimizer, lr_lambda=lambda step: 1.0 / (step + 1)
    )
    actor.weight.grad = torch.ones_like(actor.weight)
    optimizer.accumulate_model_grads()
    optimizer.optimizer.step()
    optimizer.sync_actor()
    scheduler.step()
    expected_master = optimizer.pairs[0][1].detach().clone()
    checkpoint = tmp_path / "resume.pt"
    contract = {"dataset": "signed", "lr": 1e-4}
    save_resume_checkpoint(
        path=checkpoint,
        master_optimizer=optimizer,
        scheduler=scheduler,
        next_group_offset=7,
        optimizer_step=7,
        global_step=224,
        contract=contract,
    )

    restored_actor = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    restored = FP32MasterOptimizer(restored_actor, lr=1e-4, weight_decay=0.0)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored.optimizer, lr_lambda=lambda step: 1.0 / (step + 1)
    )
    counters = load_resume_checkpoint(
        path=checkpoint,
        master_optimizer=restored,
        scheduler=restored_scheduler,
        expected_contract=contract,
    )
    assert counters == (7, 7, 224)
    assert torch.equal(restored.pairs[0][1], expected_master)
    assert torch.equal(restored_actor.weight, expected_master.to(torch.bfloat16))
    restored_state = restored.optimizer.state[restored.pairs[0][1]]
    assert restored_state["exp_avg"].dtype == torch.float32
    assert restored_scheduler.state_dict() == scheduler.state_dict()


def test_metric_log_rolls_back_to_durable_optimizer_step(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "".join(
            f'{{"optimizer_step":{step},"loss":0.0}}\n'
            for step in (20, 40, 60)
        ),
        encoding="utf-8",
    )
    rollback_metric_log(path, optimizer_step=40)
    assert [
        record["optimizer_step"]
        for record in map(json.loads, path.read_text(encoding="utf-8").splitlines())
    ] == [20, 40]
