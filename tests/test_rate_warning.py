"""Under per-synapse normalization the rates are absolute steps; large ones warn (issue 131)."""

import warnings

import pytest

import cadence as cd
from cadence.plasticity import ActorCriticConfig


def test_normalized_learner_with_a_large_rate_warns() -> None:
    with pytest.warns(RuntimeWarning, match="absolute per-synapse steps"):
        cd.LearnerConfig(normalize=0.99, momentum=0.9, eta=0.5)
    with pytest.warns(RuntimeWarning):
        cd.LearnerConfig(normalize=0.99, eta=0.003, eta_bias=0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cd.LearnerConfig(normalize=0.99, momentum=0.9, eta=0.003, eta_bias=0.02)  # the working range
        cd.LearnerConfig(eta=0.5)  # unnormalized: the recovered finite rule, no warning


def test_normalized_actor_with_a_large_rate_warns() -> None:
    with pytest.warns(RuntimeWarning, match="absolute per-synapse step"):
        ActorCriticConfig(normalize=0.99, momentum=0.9, eta=0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ActorCriticConfig(normalize=0.99, momentum=0.9, eta=0.002)
        ActorCriticConfig(eta=1.0)
