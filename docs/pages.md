# The brain in a browser page

The whole-brain viewer, the atlas that lays a connectome out and the self-contained replay
page live in the examples repository, under
[cadence-examples/viewer](https://github.com/muellerberndt/cadence-examples/tree/main/viewer),
beside the browser engine every settling-brain page shares
([cadence-examples/engine](https://github.com/muellerberndt/cadence-examples/tree/main/engine)).
The library ships the brains alone: `Connectome`, the patches, `record_settlements` for the
settlings a page replays. The three quickstart brains behind a local page are
[cadence-examples/quickstart](https://github.com/muellerberndt/cadence-examples/tree/main/quickstart).

The [viewer reference](https://github.com/muellerberndt/cadence-examples/blob/main/viewer/reference.md)
contains the atlas API, renderer options and complete recording/page examples.
Cadence 0.18 no longer installs `cadence.atlas` or `cadence-demo`: clone the
examples, import `viewer.atlas` from that checkout, and run
`python -m quickstart.demo stream` from its root (`decide` and `body` are the other
choices). See the viewer's
setup instructions for applications in a different directory.
