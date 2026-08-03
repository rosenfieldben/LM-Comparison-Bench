"""Experiment planning. Pure functions: what to run, in what order.

The runner's decisions are separated from the runner's effects on
purpose. Task order, rotation and the per-repeat seed are the parts a
reader has to be able to check, and they are checkable here without a
server, a database or a network. What remains in main.py is the part
that spends money.

Everything here is deterministic given the experiment's recorded fields.
That is the whole reproducibility claim: not that the providers will
answer the same way, which nobody can promise, but that the bench will
ASK the same things in the same order.
"""

import random
from typing import Any


def ordered_tasks(
    tasks: list[dict[str, Any]], seed: int | None
) -> list[dict[str, Any]]:
    """The task order for one experiment: file order, or a seeded shuffle.

    A seed present means shuffle and record the seed; absent means file
    order. Either way the order is reproducible from the experiment row,
    which is the property that matters. A shuffle with a seed nobody
    recorded would make an experiment unrepeatable by anyone including
    its author, so there is no unseeded shuffle option.

    random.Random rather than the module-level functions so this cannot
    disturb, or be disturbed by, any other use of randomness in the
    process. A shared global generator would make the order depend on
    what else happened to run first.
    """
    if seed is None:
        return list(tasks)
    out = list(tasks)
    random.Random(seed).shuffle(out)
    return out


def rotation_for(lineup_size: int, task_index: int, repeat_index: int) -> int:
    """Which rotation this cell uses, as an offset into the lineup.

    The position-effect answer. Models enter the semaphore in lineup
    order, so under saturation the first-listed model reliably gets a
    slot first and the last-listed model reliably waits. That is a
    systematic advantage to being listed first, and averaging over
    repeats does not remove it: it is the same direction every time.
    Rotating the entry order per cell turns a systematic effect into one
    that cancels, which is what makes repeats able to do statistical
    work.

    Derived rather than stored-and-incremented so it depends only on the
    cell's coordinates. A counter would make the rotation of a given cell
    depend on how many trials ran before it, which a resumed or partially
    halted experiment would change.
    """
    if lineup_size <= 0:
        return 0
    return (task_index + repeat_index) % lineup_size


def rotate(lineup: list[str], rotation: int) -> list[str]:
    """The lineup in this cell's entry order.

    Rotation, not shuffle: the relative order is preserved and only the
    starting point moves, so every model occupies every position an equal
    number of times over a full cycle. A shuffle would also break the
    systematic effect but would not balance it exactly, which matters at
    the small repeat counts a localhost bench actually runs.
    """
    if not lineup:
        return []
    offset = rotation % len(lineup)
    return lineup[offset:] + lineup[:offset]


def seed_for_repeat(base_seed: int | None, repeat_index: int) -> int | None:
    """The seed this repeat sends, or None to send no seed at all.

    Repeats exist to sample variation. Sending one fixed seed to every
    repeat would ask the provider for the same sample N times, which
    measures nothing and costs N times as much; deriving base + N keeps
    the whole set reproducible while letting each repeat differ.

    None stays None, which is rule one: an experiment that set no seed
    sends no seed, and each provider applies its own default. Deriving a
    seed from nothing would be the bench inventing a control the user
    never chose, and it would then appear in the record as one.
    """
    if base_seed is None:
        return None
    return base_seed + repeat_index


def plan_trials(
    tasks: list[dict[str, Any]],
    lineup: list[str],
    repeats: int,
    task_order_seed: int | None,
) -> list[dict[str, Any]]:
    """Every cell of the experiment, in the order the runner will run it.

    One flat list rather than nested loops in the runner, so the plan is a
    value that can be inspected, counted and asserted against without
    running anything. The runner's job is then to walk it, which is the
    part that can fail.

    Ordered task-major then repeat: all repeats of task one, then all of
    task two. The alternative (all tasks once, then again) would spread a
    halted experiment's completed trials thinly across every task, so a
    partial record would have one repeat of everything rather than a
    complete record of something. Complete cells are more useful.
    """
    out: list[dict[str, Any]] = []
    for task_index, task in enumerate(ordered_tasks(tasks, task_order_seed)):
        for repeat_index in range(repeats):
            rotation = rotation_for(len(lineup), task_index, repeat_index)
            out.append(
                {
                    "task": task,
                    "task_index": task_index,
                    "repeat_index": repeat_index,
                    "rotation_index": rotation,
                    "order": rotate(lineup, rotation),
                }
            )
    return out
