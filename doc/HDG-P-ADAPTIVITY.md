# `p`-adaptivity for HDG — owned by another branch

**This plan is being executed on `gf-hdg-p-adaptivity`, not here, and that
branch carries the current version of this document** (480 lines against this
stub's, plus `HDG-P-ADAPTIVITY-CONSTRAIN.md` and `HDG-P-ADAPTIVITY-MEQ-MERGE.md`).
Steps 1 through 3 of the original plan are done there — a per-face trace order
behind two accessors, the surplus slots constrained, and `convdiff -pref` to
drive it — along with an `hp` demonstrator, a smoothness sensor and the
parallel port. Read it there; do not develop against this copy.

Kept as a stub rather than deleted because the trunk has the file, so a trunk
merge would resurrect it, and because two of its prerequisites were paid for
here and are worth naming:

* the HDG face quadrature now takes the trace element's order into account, so
  a face may legally be richer than its elements (commit "The HDG face
  quadrature never saw the trace element");
* `DarcyOperator` survives the NC mesh that variable order requires (commit
  "DarcyOperator dereferenced a null prolongation…").

Both are on the trunk and therefore on every descendant. Roadmap §7 carries
what the scoping measured and why the trace space is the whole job.
