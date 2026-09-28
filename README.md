# Cadence

Brains built from one element: the **cortical column**, a bounded
self-reading settling patch - owned local state, typed boundary ports,
live readback of its own uncertainty, feedback that enters the executed
equations, transactional records, and one repair law that settles them.
A **Cortex** is a trainable hierarchy of column banks: coarse levels are
priors for fine levels, every level watches its own uncertainty, and
learning admits each witnessed transition exactly once.

Pure Python, zero dependencies. The games extra adds Atari.

```sh
pip install -e .            # library (stdlib only)
pip install -e '.[games]'   # + gymnasium/ALE for Atari
```

## Quickstart

```python
import ale_py, gymnasium as gym
from cadence import Cortex

gym.register_envs(ale_py)
def env_factory():
    return gym.make('ALE/Freeway-v5', obs_type='ram', frameskip=4,
                    repeat_action_probability=0.0)

cortex = Cortex.for_environment(env_factory, depth=2, width=1)

env = env_factory()
for episode in range(24):
    observation, _ = env.reset(seed=episode)
    done, total = False, 0.0
    while not done:
        action = cortex.act(observation)
        observation_next, reward, terminated, truncated, _ = env.step(action)
        cortex.learn(observation, action, float(reward), observation_next, terminated)
        observation, total = observation_next, total + reward
        done = terminated or truncated
    cortex.end_episode()
    print(episode, total)
```

No manual wiring: a calibration probe finds the bytes the body's own
actions move and the bytes the world moves, and builds the level maps.
Structure is configured like a small neural net - `depth` (hidden
levels) and `width` (bytes read by the finest level), or an explicit
`ladder` - while dynamics (decay, discount, optimism, epsilon) are
constructor keywords with documented defaults. On Freeway this learns
its first road crossings within a handful of episodes and reaches
15-20 crossings per two-minute episode by episode ~10, while random
and frozen-memory controls stay at zero; on Pong, width-1 wiring is
demonstrably too narrow - both results, with every control, live in
the development receipts (docs/VARIANTS.md).

## The pieces

| Piece | What it is |
| --- | --- |
| `CorticalColumn` | The element: scalar witness admission, read-only query, observer readback/feedback, exact Fraction inference on certified cluster forests. |
| `Cortex` | Hierarchy of value-column banks over calibrated contexts; act / learn / end_episode / value / snapshot / restore. |
| `calibrate` / `wire` | The probe and the layer builder (`depth`, `width`, `ladder`). |

Docs: [REFERENCE](docs/REFERENCE.md) (API and every parameter),
[VARIANTS](docs/VARIANTS.md) (flat, deep, wide - with the measured
evidence), [ELEMENT](docs/ELEMENT.md) (the law, the equations, the
qualification story and its limits), [MIGRATION](docs/MIGRATION.md)
(coming from 0.19 and earlier).

MIT license.
