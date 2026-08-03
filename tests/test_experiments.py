"""Experiment planning: what runs, in what order, with which seed.

These are the parts of the runner that must be checkable without a
server, a database or a network, which is why they live in a pure module.
Everything asserted here is a property of the recorded fields alone.
"""

from hypothesis import given
from hypothesis import strategies as st

from bench.experiments import (
    ordered_tasks,
    plan_trials,
    rotate,
    rotation_for,
    seed_for_repeat,
)


def tasks(n):
    return [{"id": f"t{i}", "prompt": f"p{i}", "system": None} for i in range(n)]


def test_no_seed_means_file_order():
    """The default has to be the order the author wrote, because that is
    the only order they can see by opening the file."""
    assert [t["id"] for t in ordered_tasks(tasks(4), None)] == ["t0", "t1", "t2", "t3"]


def test_a_seed_shuffles_reproducibly():
    """Reproducible is the whole requirement. A shuffle nobody could
    repeat would make an experiment unrepeatable by its own author."""
    once = [t["id"] for t in ordered_tasks(tasks(8), 42)]
    twice = [t["id"] for t in ordered_tasks(tasks(8), 42)]

    assert once == twice
    assert set(once) == {f"t{i}" for i in range(8)}
    assert once != [f"t{i}" for i in range(8)]


def test_different_seeds_give_different_orders():
    assert ordered_tasks(tasks(8), 1) != ordered_tasks(tasks(8), 2)


def test_ordering_does_not_mutate_the_input():
    """The plan is derived from the dataset, never in place: a loader's
    output is read by more than one thing."""
    original = tasks(4)
    ordered_tasks(original, 42)

    assert [t["id"] for t in original] == ["t0", "t1", "t2", "t3"]


def test_the_shuffle_uses_its_own_generator():
    """A shared global generator would make the task order depend on what
    else in the process happened to draw a random number first, which is
    exactly the kind of hidden input this phase exists to eliminate."""
    import random

    random.seed(0)
    first = ordered_tasks(tasks(8), 42)
    for _ in range(100):
        random.random()
    second = ordered_tasks(tasks(8), 42)

    assert first == second


def test_rotation_advances_with_both_coordinates():
    assert rotation_for(3, 0, 0) == 0
    assert rotation_for(3, 0, 1) == 1
    assert rotation_for(3, 1, 0) == 1
    assert rotation_for(3, 1, 2) == 0


def test_rotation_is_derived_not_counted():
    """A given cell's rotation depends only on that cell's coordinates, so
    a halted or resumed experiment cannot change it."""
    assert rotation_for(4, 7, 2) == rotation_for(4, 7, 2)
    assert rotation_for(4, 7, 2) == (7 + 2) % 4


def test_rotate_preserves_relative_order():
    """Rotation, not shuffle. Every model occupies every position an equal
    number of times over a full cycle, which a shuffle would only manage
    on average and this bench runs small repeat counts."""
    lineup = ["a", "b", "c"]

    assert rotate(lineup, 0) == ["a", "b", "c"]
    assert rotate(lineup, 1) == ["b", "c", "a"]
    assert rotate(lineup, 2) == ["c", "a", "b"]
    assert rotate(lineup, 3) == ["a", "b", "c"]


def test_rotate_handles_the_degenerate_cases():
    assert rotate([], 3) == []
    assert rotate(["a"], 5) == ["a"]
    assert rotation_for(0, 1, 1) == 0


def test_a_full_cycle_puts_every_model_in_every_slot_equally():
    """The property that makes rotation the answer to the position effect,
    stated as a count rather than as a claim."""
    lineup = ["a", "b", "c"]
    seen = {m: [0] * 3 for m in lineup}
    for rotation in range(3):
        for slot, model in enumerate(rotate(lineup, rotation)):
            seen[model][slot] += 1

    assert all(counts == [1, 1, 1] for counts in seen.values())


def test_no_base_seed_sends_no_seed():
    """Rule one at the repeat level. Deriving a seed from nothing would be
    the bench inventing a control the user never chose, which would then
    appear in the record as one."""
    assert seed_for_repeat(None, 0) is None
    assert seed_for_repeat(None, 7) is None


def test_repeats_derive_distinct_seeds():
    """One fixed seed across repeats would buy the same sample N times,
    which measures nothing and costs N times as much."""
    assert [seed_for_repeat(100, i) for i in range(4)] == [100, 101, 102, 103]


def test_the_plan_is_task_major():
    """All repeats of task one, then task two. A halt then leaves complete
    cells rather than one repeat of everything."""
    plan = plan_trials(tasks(2), ["a", "b"], repeats=2, task_order_seed=None)

    assert [(c["task"]["id"], c["repeat_index"]) for c in plan] == [
        ("t0", 0),
        ("t0", 1),
        ("t1", 0),
        ("t1", 1),
    ]


def test_the_plan_carries_each_cells_entry_order():
    plan = plan_trials(tasks(1), ["a", "b", "c"], repeats=3, task_order_seed=None)

    assert [c["order"] for c in plan] == [
        ["a", "b", "c"],
        ["b", "c", "a"],
        ["c", "a", "b"],
    ]
    assert [c["rotation_index"] for c in plan] == [0, 1, 2]


@given(
    n_tasks=st.integers(min_value=1, max_value=8),
    n_models=st.integers(min_value=1, max_value=5),
    repeats=st.integers(min_value=1, max_value=5),
    seed=st.one_of(st.none(), st.integers(min_value=0, max_value=10_000)),
)
def test_the_plan_has_exactly_one_cell_per_task_repeat(
    n_tasks, n_models, repeats, seed
):
    """The count is the experiment's own trials_total divided by the
    lineup, and it has to hold for every shape, because that number is the
    denominator the conservation property is checked against."""
    lineup = [f"m{i}" for i in range(n_models)]

    plan = plan_trials(tasks(n_tasks), lineup, repeats, seed)

    assert len(plan) == n_tasks * repeats
    assert sum(len(c["order"]) for c in plan) == n_tasks * repeats * n_models
    # Every cell asks every model exactly once, whatever the rotation.
    for cell in plan:
        assert sorted(cell["order"]) == sorted(lineup)
