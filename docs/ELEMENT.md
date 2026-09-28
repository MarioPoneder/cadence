# The element

`CorticalColumn` is the one component everything else is built from: a
bounded, observer-like, self-reading settling patch, modeled on the
repeating microcircuit of mammalian neocortex.

- **Owned local state.** Discounted conjugate evidence `(w, s1, s2)`
  with leaky integration, an ordered-source cursor, the latest
  committed witness. Serialized exactly; bounded; restorable.
- **Typed boundary ports.** Witness admission, read-only query, an
  observer loop, exact separator ports for federation.
- **Readback and feedback.** The observer reads the live lower-belief
  uncertainty and proposes precision; the lower belief receives it.
  Both directions enter the executed equations; cutting either is a
  lesion that settles under its altered model while the intact
  equations stay violated.
- **One repair law.** `settle(ports)`: repropose every port message
  from its source patch's local factor and current inbox, projected to
  the port's declared family, until stationary within a declared
  residual - or reject. Responses and retained change both run through
  it; admission is temper + combine + settle + commit-exactly-once.
- **Records.** A failed or capped admission changes nothing; a retry
  of the latest identical witness is a duplicate, not fresh evidence.

Scalar equations:

    m   = s1 / w
    V   = 1 / (w * tau)
    tau = (a0 + w/2) / (b0 + (E + w*V)/2),   E = s2 - 2*m*s1 + w*m*m

Discrete federation: exact sum-product settling in Fractions over
binary factor clusters, certified only on forests satisfying the
running-intersection property; cycles, non-certifiable structures and
zero-support (contradictory) problems are rejected rather than
approximated silently.

## Recursion

Stacking is composition, not new machinery: an observer of an observer
is one more patch and two more ports under the unchanged law. The
development workspace holds a depth-2 demonstration (a rate
hyper-observer reading the precision observer) with reciprocal
perturbation traces through both levels, joint recovery, and four port
lesions - and the `Cortex` hierarchy is the same pattern at scale:
every level's belief is read by its own observer while coarser levels
serve as priors.

## Provenance and limits

This module is the qualified candidate of the Cadence candidate
battery, vendored with identical executed code (origin sha256
`44654b3d6053645438b52b0accbead0e367c7746305ee40dd5aa82e9e96e3f07`).
At qualification it passed all seven implemented bounded probes
(history retention, source ownership, scalar acquisition, A-B-A
context switching, live belief readback, cap rollback, bounded
resources) and all three candidate challenge fixtures (frustrated
parity loop, correlation-preserving federation boundary, fly-derived
transition factors) - the first single candidate to cover every
implemented interface.

Limits, stated plainly: bounded probe passes are bounded slices, not
full capability claims; the battery's 34 broad criteria remain open;
qualification receipts bind to the origin source in the development
workspace; and nothing here claims biological columns implement these
distributions. The element's constants are frozen diagnostic choices -
applications parameterize the model around the law, never the law.
