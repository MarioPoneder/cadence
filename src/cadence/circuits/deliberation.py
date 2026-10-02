"""Isolated prospective search, advanced by the application's own event loop."""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Generic, TypeVar

S = TypeVar("S")
A = TypeVar("A")


@dataclass(frozen=True)
class Future(Generic[S, A]):
    action: A
    score: float
    sequence: tuple[A, ...]
    state: S


@dataclass(frozen=True)
class Deliberation(Generic[S, A]):
    futures: tuple[Future[S, A], ...]
    nodes: int
    depth: int


class _BudgetReached(Exception):
    pass


def _positive(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class Deliberator(Generic[S, A]):
    """Keep hypothetical work alive between actions, in bounded cooperative ticks.

    ``start(live)`` snapshots a new observation and cancels obsolete work. ``tick``
    continues iterative deepening, returning the last fully completed depth or
    None. Pause preserves work; cancel discards it. No thread starts implicitly.
    The application calls tick from its task loop, including while awaiting input.

    Values use the root actor's perspective; adversarial search alternates max/min.
    Callbacks must be bounded, read-only with respect to external objects, and use
    a fixed model/evaluator until the next start. Only transition may mutate its
    supplied isolated copy. Tick bounds visited nodes, not callback wall time.
    No imagined event writes real-world memories, actions or reward eligibility.
    """

    def __init__(
        self,
        actions: Callable[[S], Sequence[A]],
        transition: Callable[[S, A], S],
        evaluate: Callable[[S], float],
        terminal: Callable[[S], bool],
        *,
        depth: int = 6,
        max_nodes: int = 10000,
        adversarial: bool = False,
        prune: bool = False,
        clone: Callable[[S], S] = deepcopy,
    ) -> None:
        _positive(depth, "depth")
        _positive(max_nodes, "max_nodes")
        self.actions, self.transition = actions, transition
        self.evaluate, self.terminal = evaluate, terminal
        self.depth, self.max_nodes = depth, max_nodes
        self.adversarial, self.prune, self.clone = adversarial, prune, clone
        self.nodes = 0
        self.result: Deliberation[S, A] | None = None
        self.paused = False
        self.budget_exhausted = False
        self._work: Generator[Deliberation[S, A] | None, None, None] | None = None

    @property
    def pending(self) -> bool:
        """Unfinished work, including paused work."""
        return self._work is not None

    def start(self, live: S) -> None:
        """Snapshot a new real state, replacing the previous search and result."""
        initial = self.clone(live)
        self.cancel()
        self.nodes = 0
        self.budget_exhausted = False
        self.paused = False
        self._work = self._deepen(initial)

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def cancel(self) -> None:
        """Discard the continuation and its obsolete candidate actions."""
        if self._work is not None:
            self._work.close()
        self._work = None
        self.result = None

    def tick(self, nodes: int = 128) -> Deliberation[S, A] | None:
        """Visit at most ``nodes`` new positions; never execute a real action.

        Completed results are isolated copies, so caller inspection cannot corrupt
        a retained search. An incomplete depth never replaces a completed one.
        Once finished or paused, ticks do no work until start/resume respectively.
        """
        _positive(nodes, "nodes")
        if not self.paused and self._work is not None:
            try:
                for _ in range(nodes):
                    event = next(self._work)
                    if event is not None:
                        self.result = event
            except StopIteration:
                self._work = None
            except _BudgetReached:
                self._work = None
                self.budget_exhausted = True
            except Exception:
                self.cancel()
                raise
        if self.result is None:
            return None
        return Deliberation(
            tuple(
                Future(deepcopy(f.action), f.score, deepcopy(f.sequence), self.clone(f.state))
                for f in self.result.futures
            ),
            self.result.nodes,
            self.result.depth,
        )

    def _deepen(self, initial: S) -> Generator[Deliberation[S, A] | None, None, None]:
        for depth in range(1, self.depth + 1):
            yield from self._search(initial, depth)
            if self.terminal(initial) or len(self.actions(initial)) == 0:
                return

    def _search(self, initial: S, depth: int) -> Iterator[Deliberation[S, A] | None]:
        def visit(
            parent: S,
            action: A,
            remaining: int,
            maximize: bool,
            alpha: float = -float("inf"),
            beta: float = float("inf"),
        ) -> Generator[None, None, tuple[float, tuple[A, ...], S]]:
            if self.nodes >= self.max_nodes:
                raise _BudgetReached
            self.nodes += 1
            state = self.transition(self.clone(parent), action)
            choices = () if remaining == 0 or self.terminal(state) else self.actions(state)
            if len(choices) == 0:
                value = float(self.evaluate(state))
                if not isfinite(value):
                    raise ValueError("evaluator returned a nonfinite value")
                yield None
                return (value, (), state)
            yield None
            best: tuple[float, tuple[A, ...], S] | None = None
            for choice in choices:
                score, suffix, end = yield from visit(
                    state,
                    choice,
                    remaining - 1,
                    not maximize if self.adversarial else True,
                    alpha,
                    beta,
                )
                if best is None or (score > best[0] if maximize else score < best[0]):
                    best = (score, (choice, *suffix), end)
                if self.prune and self.adversarial:
                    if maximize:
                        alpha = max(alpha, score)
                    else:
                        beta = min(beta, score)
                    if alpha >= beta:
                        break
            assert best is not None
            return best

        if self.terminal(initial):
            yield Deliberation((), self.nodes, depth)
            return
        futures = []
        for action in self.actions(initial):
            # Fresh bounds preserve exact values for every root option.
            score, sequence, end = yield from visit(
                initial, action, depth - 1, not self.adversarial
            )
            futures.append(Future(action, score, (action, *sequence), end))
        yield Deliberation(tuple(sorted(futures, key=lambda f: -f.score)), self.nodes, depth)


def imagine(
    live: S,
    actions: Callable[[S], Sequence[A]],
    transition: Callable[[S, A], S],
    evaluate: Callable[[S], float],
    terminal: Callable[[S], bool],
    *,
    depth: int = 2,
    max_nodes: int = 10000,
    adversarial: bool = False,
    prune: bool = False,
    clone: Callable[[S], S] = deepcopy,
) -> Deliberation[S, A]:
    """Compare isolated futures synchronously at one depth, with a hard node budget.

    Values are from the root actor's perspective. Adversarial search alternates
    max/min; optional alpha-beta pruning preserves exact root-action scores.
    Callbacks obey the same isolation contract as Deliberator. A hard budget
    failure raises instead of silently returning a partial comparison.
    """
    planner = Deliberator(
        actions,
        transition,
        evaluate,
        terminal,
        depth=depth,
        max_nodes=max_nodes,
        adversarial=adversarial,
        prune=prune,
        clone=clone,
    )
    result = None
    try:
        for event in planner._search(clone(live), depth):
            if event is not None:
                result = event
    except _BudgetReached as exc:
        raise ValueError("imagination node budget exceeded") from exc
    assert result is not None
    return result
