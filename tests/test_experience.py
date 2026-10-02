"""Integration checks: witnessed evidence drives actions; imagination stays read-only."""

from copy import deepcopy

import numpy as np
import pytest

import cadence as cd
from cadence.circuits import imagine


def cue(index: int, width: int) -> np.ndarray:
    return np.eye(width)[[index]]


def route(model: cd.SynapticMemory, start: int, goal: int) -> tuple[int, ...] | None:
    """Search reads acquired consequences; it never calls or updates the real world."""

    def predicted(room, action):
        return model.recall(cue(2 * room + action, 6))[0]

    plan = imagine(
        (start, 0),
        lambda s: [a for a in range(2) if predicted(s[0], a).max() > 0.5],
        lambda s, a: (int(predicted(s[0], a).argmax()), s[1] + 1),
        lambda s: float(s[0] == goal) - 0.01 * s[1],
        lambda s: s[0] == goal,
        depth=3,
    )
    if start == goal:
        return ()
    return next((f.sequence for f in plan.futures if f.state[0] == goal), None)


def request(model, words, events, room):
    meanings = words.recall(cue(0, 1))[0]  # the heard word ID
    winners = np.flatnonzero(np.isclose(meanings, meanings.max()))
    if meanings.max() <= 0 or len(winners) != 1:
        return None  # insufficient grounding: do not invent a referent
    location = events.recall(cue(int(winners[0]), 3))[0]
    return route(model, room, int(location.argmax())) if location.max() > 0.5 else None


def _experience(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    model = cd.SynapticMemory(np.arange(6), np.arange(6, 9), decay=1)
    words = cd.SynapticMemory(np.arange(1), np.arange(1, 4), decay=1, rate=0.25)
    events = cd.FastSynapses(np.arange(3), np.arange(3, 6), rule="delta")
    for store in (model, words, events):
        store.reset(1)
    room, seen = 0, False
    errors = []

    def move(action):
        nonlocal room, seen
        key = cue(2 * room + action, 6)
        prediction = model.recall(key).copy()  # before the outcome is available
        room = (room + (1 if action == 0 else -1)) % 3  # environment executes
        observed = cue(room, 3)
        errors.append(float(np.square(prediction - observed).sum()))
        model.observe(key, observed)  # repair from the actual consequence
        if room == 2 and not seen:
            events.observe(cue(1, 3), observed)  # object 1 is visible only once
            seen = True

    for moment in range(64):
        move(int(rng.integers(2)))
        if moment in (10, 20):
            # Ambiguous scenes share object 1; no dictionary target is supplied.
            candidates = (0, 1) if moment == 10 else (1, 2)
            words.observe(cue(0, 1), np.maximum.reduce([cue(i, 3) for i in candidates]))
    while room != 0:
        move(0)

    # Read the same experience under interventions. These copies never teach the live agent.
    choices = {}
    for variant in ("full", "no_events", "no_words", "wrong_world"):
        world_copy, word_copy, event_copy = deepcopy((model, words, events))
        if variant == "no_events":
            event_copy.reset(1)
        elif variant == "no_words":
            word_copy.clear()
        elif variant == "wrong_world":
            world_copy.strength = np.roll(world_copy.strength, 1, axis=2)
        choices[variant] = request(world_copy, word_copy, event_copy, room)
    plan = choices["full"]
    for action in plan or ():
        move(action)  # acting and learning continue through the very same operation
    return {
        "choices": choices,
        "reached_object": room == 2,
        "first_16_prediction_error": float(np.mean(errors[:16])),
        "last_16_prediction_error": float(np.mean(errors[-16:])),
        "single_sightings": events.writes,
        "boundary": "Supplied perception, request protocol and search; learned associations.",
    }


def _changing_rewards(seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    agent = cd.GenericBrain.build(4, 4, hidden=16, seed=seed)
    context = rng.integers(4, size=32)
    action = agent.step(np.eye(4)[context])
    rewards = []
    for moment in range(600):
        # The world changes its reward rule halfway through the same learning life.
        rewarded_action = (context + int(moment >= 300)) % 4
        reward = (action == rewarded_action).astype(float)
        rewards.append(float(reward.mean()))
        context = rng.integers(4, size=32)
        action = agent.step(np.eye(4)[context], reward=reward, done=np.ones(32, bool))
    return {
        "before_change": float(np.mean(rewards[250:300])),
        "just_after_change": float(np.mean(rewards[300:310])),
        "after_adapting": float(np.mean(rewards[-50:])),
    }


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_experience_actions_depend_on_acquired_memories(seed):
    result = _experience(seed)
    assert result["reached_object"] and result["single_sightings"] == 1
    assert result["choices"]["no_events"] is None
    assert result["choices"]["no_words"] is None
    assert result["choices"]["wrong_world"] != result["choices"]["full"]
    assert result["last_16_prediction_error"] < result["first_16_prediction_error"]


def test_search_cannot_know_unobserved_transitions_or_teach_itself():
    model = cd.SynapticMemory(np.arange(6), np.arange(6, 9), decay=1)
    model.reset(1)
    assert route(model, 0, 2) is None
    model.observe(cue(1, 6), cue(2, 3))  # witnessed 0 --action 1--> 2
    fast, slow, writes = model.strength.copy(), model.consolidated.copy(), model.writes
    assert route(model, 0, 2) == (1,)
    np.testing.assert_array_equal(model.strength, fast)
    np.testing.assert_array_equal(model.consolidated, slow)
    assert model.writes == writes


def test_reward_stream_adapts_without_restarting_its_brain():
    result = _changing_rewards()
    assert result["before_change"] > 0.9
    assert result["after_adapting"] > 0.9
    assert result["just_after_change"] < result["after_adapting"]
