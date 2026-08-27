# Reusing the symbolic factorisation in the direct solver wrappers

A plan, not an implementation.

`UMFPackSolver` and `KLUSolver` recompute the **symbolic analysis** on every
`SetOperator()` call. Any caller that refactorises a matrix whose sparsity
pattern does not change — a Newton iteration, an implicit time step, a
continuation loop, a parameter sweep — pays for it every time and uses it once.

Measured with SuiteSparse UMFPACK on a hybridized HDG trace matrix, ordering set
to `UMFPACK_ORDERING_METIS`:

| `n` | nnz | symbolic | numeric | backsolve | symbolic share of a factorisation |
|---|---|---|---|---|---|
| 12,544 | 246,784 | 20.8 ms | 72.1 ms | 10.9 ms | **22%** |
| 49,664 | 985,088 | 104.2 ms | 325.6 ms | 54.6 ms | **24%** |

So roughly a quarter of the linear cost of every Newton step is thrown away and
recomputed. The share rises with a better ordering, which is the perverse part:
asking for METIS rather than the default makes the discarded work more
expensive.

**This is a wrapper limitation, not a SuiteSparse one.** Both libraries separate
the phases precisely so that the analysis can be amortised —
`umfpack_*_symbolic` / `umfpack_*_numeric`, `klu_analyze` / `klu_factor` — and
reuse across changing values is their documented use.

## Where it stands today

**`UMFPackSolver::SetOperator`** (`linalg/solvers.cpp`) declares

```cpp
void *Symbolic;
```

as a **local variable**, runs `umfpack_*_symbolic` then `umfpack_*_numeric`, and
frees the symbolic object before returning. There is no way for a caller to
retain it.

**`KLUSolver`** already stores `Symbolic` as a member — because `klu_solve()`
requires it, not by design for reuse — and its `SetOperator` frees and rebuilds
it regardless. So it has the storage but not the behaviour.

Both are called once per iteration by `NewtonSolver::Mult`, which does
`prec->SetOperator(*grad)` inside its loop.

## The design question, which is when reuse is legitimate

A retained symbolic object is valid exactly when the **sparsity pattern** is
unchanged. Values may change arbitrarily. Three ways to establish that, of
increasing cost and coverage:

| level | test | cost | covers |
|---|---|---|---|
| 0 | none — always re-analyse | — | today's behaviour |
| 1 | same `SparseMatrix` object, same `GetI()`/`GetJ()` pointers, same `nnz` | `O(1)` | a matrix reassembled in place, which is the Newton and time-stepping case |
| 2 | pattern arrays compare equal | `O(nnz)` integer compare | a matrix rebuilt into a fresh object with the same structure |

Level 1 is the one that matters and is nearly free. Level 2 costs about 1 ms
where the symbolic costs 100 ms, so it is still overwhelmingly worth it, and it
is *exact* — no hashing, no trusting the caller.

**Proposal: opt in, then verify.** Default stays level 0, so no existing code
changes behaviour or performance. A caller that opts in gets level 1, escalating
to level 2 when the object identity test fails. The wrapper never takes the
caller's word for it, which is what keeps this from becoming a way to get silent
wrong answers when somebody's pattern does change.

```cpp
/// Retain the symbolic factorisation across SetOperator() calls and reuse it
/// whenever the sparsity pattern is unchanged. The pattern is checked, not
/// assumed. Off by default.
void SetReuseSymbolic(bool reuse = true);

/// How many symbolic analyses have actually been performed. For tests, and
/// for anyone who wants to know whether reuse is firing.
long GetNumSymbolicFactorizations() const;
```

The counter is what makes this testable without timing assertions, which rot.

## Scope

1. **`UMFPackSolver`** — promote `Symbolic` to a member, free it in the
   destructor and whenever the pattern check fails, and skip
   `umfpack_*_symbolic` when it holds. Both the `int` and `SuiteSparse_long`
   paths.
2. **`KLUSolver`** — same, and additionally expose **`klu_refactor()`**, which
   reuses the numeric pattern as well and is materially cheaper than a fresh
   `klu_factor()`. This is the case KLU was written for, and the larger win of
   the two.
3. **Leave `CPardisoSolver` alone in this change**, and `PardisoSolver` if it is
   ever enabled. Both have the same shape — analysis and factorisation are both
   inside `SetOperator` — and the same fix applies, but PARDISO's phase
   numbering makes it a separate, simpler patch and mixing them would obscure
   both.

## What could go wrong, and what it costs

**A stale symbolic object used against a changed pattern** is the failure to
design against, and it would be silent: UMFPACK would factor against the wrong
structure and return numbers. The pattern check is what prevents it, which is
why it is a check and not a caller promise.

**Ordering quality can drift.** `umfpack_*_symbolic` reads the values as well as
the pattern — under `UMFPACK_STRATEGY_AUTO` it uses them to choose between the
symmetric and unsymmetric strategies and to detect singletons. A symbolic object
computed for one set of values and reused against very different ones stays
*correct* but may carry a worse ordering than a fresh analysis would. This is a
performance caveat, not a correctness one, and it is worth a sentence in the
doxygen rather than a mechanism.

**Memory** rises by one retained symbolic object for the lifetime of the solver,
which is small next to the numeric factors it already holds.

## Acceptance

* A regression that factorises, changes the **values** in place, calls
  `SetOperator` again, and gets a solution agreeing with a freshly-factorised
  solver to round-off — with `GetNumSymbolicFactorizations()` equal to 1.
* The same with a **structural** change between calls, asserting the count goes
  to 2 and the answer is still right. This is the one that would catch a stale
  symbolic being reused.
* A rebuilt-but-identical matrix in a fresh `SparseMatrix`, asserting level 2
  fires and the count stays at 1.
* Existing behaviour unchanged with the flag off: same results, same counts.
* KLU covered symmetrically, including a `klu_refactor` path.

The measurement worth quoting afterwards is not a wall-clock ratio — that varies
with ordering and machine — but the symbolic share of a factorisation, which is
the fraction actually recovered. On the table above it is 22–24%.

## Provenance

Found from outside, by a Grad–Shafranov solver built on this tree that runs
Newton on a hybridized HDG trace system and pays the analysis on every step. The
numbers above are from that application's matrices, but nothing in the plan is
specific to it: any consumer of these wrappers inside an iteration has the same
problem, and the fix is in the wrapper rather than in any caller.
