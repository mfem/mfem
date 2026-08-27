# `ess_tdof_list` holds trace dofs and is documented as flux dofs

A one-line documentation defect, reported because it cost real time and would
cost it again.

`fem/darcy/darcyhybridization.hpp:237`:

```cpp
Array<int> ess_tdof_list;              ///< essential flux true DOFs
```

and the accessor at `:850`:

```cpp
/// Return a (read-only) list of all essential true DOFs.
const Array<int> &GetEssentialTrueDofs() const { return ess_tdof_list; }
```

**It holds essential TRACE true dofs, not flux.** `SetEssentialBC` fills it from
the constraint space (`darcyhybridization.cpp:258`):

```cpp
c_fes.GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list);
```

`c_fes` is the constraint — trace — space. Which is the useful thing and the
right thing: the reduced system is in the trace, so a caller checking anything
about the reduced operator needs exactly this list. `../meq` uses it that way
and
has done since stage 4, and it is correct: the indices match the unit rows the
`DIAG_ONE` policy leaves in the reduced Jacobian, exactly, at every polynomial
degree tried (64, 96 and 128 dofs at `k` = 1, 2, 3 on one benchmark).

## Why it is worth a line of anyone's time

Any caller that wants to compare `GetGradient` against a finite difference of
`Mult` **must** mask these rows: the residual is masked there and the Jacobian
carries a unit row, so including them makes the comparison meaningless. That is
the standard check on a Newton operator, so it is not an exotic need.

Reading the comment, this reporter concluded the accessor returned flux dofs,
that no trace-dof accessor existed, and wrote a substitute that detected unit
rows in the assembled Jacobian instead. That substitute silently found **zero**
rows on a problem where the essential trace condition had not in fact been
established, and the resulting Jacobian comparison was measuring an ill-posed
problem. Two wrong measurements were published from it before the control
caught them. The accessor was there the whole time.

## What would fix it

Either of:

```cpp
Array<int> ess_tdof_list;   ///< essential trace true DOFs, from the constraint space
```

or, if the name is worth matching to what it holds, an alias beside the existing
accessor:

```cpp
/// The essential true DOFs of the constraint (trace) space -- the rows the
/// diagonal policy pins in the reduced operator.
const Array<int> &GetEssentialTraceDofs() const { return ess_tdof_list; }
```

The comment is the part that matters; the name is a preference. `SetEssentialBC`
would read more clearly with a sentence saying which space it marks, since it
takes a *boundary attribute* marker and produces trace dofs, and nothing at the
call site says so.

## Provenance

Found while reporting a bug against `NLOrdering::LineariseThenCondense` — see
the Outcome section of `doc/HDG-LINEARISE-THEN-CONDENSE.md` — whose
demonstrator needed exactly this list and therefore carried a dependency on
`../meq` that it should not have needed. That demonstrator is now the unit test
"The reduced gradient is the derivative of the reduced residual", which uses
`GetEssentialTrueDofs()` for the purpose described above, so the accessor is
load-bearing on this branch and the comment above it is still wrong.
