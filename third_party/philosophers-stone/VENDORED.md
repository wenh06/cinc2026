# Vendored: philosophers-stone

Upstream: https://github.com/bdsp-core/philosophers-stone
Pinned commit: 0b1b49a8665e11157a107e0d76b9eedddd3b04c7
License: CC BY-NC 4.0 (see LICENSE in this directory)

Only `src/` and `LICENSE` are tracked.  The official image build context has no
`.git`, so git submodules cannot be used there; the runtime source is vendored
as ordinary files instead.  Do not re-introduce a submodule for this directory.
