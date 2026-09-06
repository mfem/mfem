# `IDASolver` links against IDAS, not IDA, and only by symbol coincidence

**From gffp, 2026-09-06.** A defect report, not a request. Branch:
`sundials-ida-integration`. It affects **both** build systems.

`linalg/sundials.hpp:47` includes `<ida/ida.h>` unconditionally under
`MFEM_USE_SUNDIALS`, and `:937` declares `class IDASolver`. Neither build
system ever looks for the IDA library.

## 1. What is missing

**CMake.** `CMakeLists.txt:390`

```cmake
set(SUNDIALS_COMPONENTS CVODES ARKODE IDAS KINSOL NVector_Serial)
```

and `config/cmake/modules/FindSUNDIALS.cmake:34` declares `IDAS` and no `IDA`:

```cmake
ADD_COMPONENT IDAS "include" idas/idas.h "lib" sundials_idas
```

**Makefile.** `config/defaults.mk:319`

```make
 -lsundials_arkode -lsundials_cvodes -lsundials_idas\
 -lsundials_nvecserial -lsundials_kinsol
```

So `-lsundials_ida` appears nowhere, in either build.

## 2. Why nothing reported it

**IDAS exports the same core symbols as IDA.** Measured on SUNDIALS 7.5.0,
counting defined `T` symbols in each shared library:

| symbol | `libsundials_ida.so` | `libsundials_idas.so` |
|---|---|---|
| `IDACreate`, `IDAInit`, `IDASolve`, `IDACalcIC`, `IDASetId` | 5 / 5 | 5 / 5 |

`-lsundials_idas` alone therefore satisfies every reference `IDASolver`
generates, and a build that has IDAS linked appears to work. MFEM is compiling
against **IDA's** declarations and resolving against **IDAS's**
implementation.

## 3. Why it matters

* **IDAS is optional.** A SUNDIALS built without the sensitivity packages has
  `libsundials_ida` and no `libsundials_idas`, and MFEM then fails to link with
  undefined `IDACreate` etc. — pointing at the caller rather than at the
  missing component, because `find_package(SUNDIALS REQUIRED ...)` succeeded.
* **It is an unsanctioned mixture.** IDA and IDAS are separate packages with
  separate `IDAMem` layouts and separate sonames — on the install here,
  `libsundials_ida.so.7` against `libsundials_idas.so.6` from one 7.5.0 release.
  The public entry points take opaque `void *`, which is why it happens to
  work, and nothing in SUNDIALS promises it will keep working.

## 4. Suggested fix, which is additive

**Keep IDAS.** It is requested independently, other consumers of a shared
install depend on the existing list, and needing IDA does not make IDAS
unnecessary.

`FindSUNDIALS.cmake`, alongside the existing `IDAS` line:

```cmake
ADD_COMPONENT IDA "include" ida/ida.h "lib" sundials_ida
```

`CMakeLists.txt:390`:

```cmake
set(SUNDIALS_COMPONENTS CVODES ARKODE IDA IDAS KINSOL NVector_Serial)
```

`config/defaults.mk:319`: add `-lsundials_ida` to `SUNDIALS_LIB`.

## 5. A trap in applying it, worth a line in the commit message

**`mfem_find_package` returns early on a warm cache**, so adding the component
does nothing on a tree that has already configured:

```cmake
// config/cmake/modules/MfemCmakeUtilities.cmake:235
if (${Prefix}_FOUND)
   return()
```

Verified here: after the change, `cmake .` reported `-- MFEM: using package
SUNDIALS` and left `SUNDIALS_IDA_LIBRARY` absent from the cache. The `elseif`
immediately below is the same trap one step further in — it accepts an existing
`${Prefix}_LIBRARIES` *silently*. Re-running the search needs three cache
entries deleted:

```sh
sed -i '/^SUNDIALS_\(FOUND\|LIBRARIES\|INCLUDE_DIRS\):/d' build/CMakeCache.txt
```

and **their `//` doc-comment lines deleted too**, or CMake rejects the whole
cache with `Parse error in cache file ... Offending entry:` naming a line
number and no entry, which does not obviously mean "dangling comment".

After that the cache picks up
`SUNDIALS_IDA_LIBRARY:FILEPATH=.../libsundials_ida.so` and `SUNDIALS_LIBRARIES`
carries both `libsundials_ida.so` and `libsundials_idas.so`.

---

# Reply, 2026-09-06

**Accepted and fixed as suggested.** Every claim reproduced here before
changing anything.

## 1. Reproduced

All four sites are as reported on `sundials-ida-integration`:
`linalg/sundials.hpp:47` includes `<ida/ida.h>` and `:937` declares
`IDASolver`; `CMakeLists.txt:390` lists `IDAS` and not `IDA`;
`FindSUNDIALS.cmake:34` declares `IDAS` only; `config/defaults.mk:319` links
`-lsundials_idas` only.

The symbol coincidence is confirmed on the install here: both
`libsundials_ida.so` and `libsundials_idas.so` export all five of `IDACreate`,
`IDAInit`, `IDASolve`, `IDACalcIC` and `IDASetId`. So does the soname
observation -- `install-mpi` carries `libsundials_ida.so.7` beside
`libsundials_idas.so.6` from one 7.5.0 release.

## 2. The fix, exactly the additive one suggested

`ADD_COMPONENT IDA` in `FindSUNDIALS.cmake`, `IDA` added to
`SUNDIALS_COMPONENTS` in `CMakeLists.txt`, and `-lsundials_ida` added to
`SUNDIALS_LIB` in `config/defaults.mk`, **before** `-lsundials_idas` so that
IDA's implementation is the one that resolves. IDAS is kept.

Verified in both build systems, each from a clean configuration rather than a
warm one:

* **CMake**, fresh build directory: `-- SUNDIALS: IDA:
  .../libsundials_ida.so`, with `SUNDIALS_IDA_LIBRARY` and
  `SUNDIALS_IDAS_LIBRARY` both in the cache.
* **Makefile**, out-of-source `make config MFEM_BUILD_DIR=<scratch>`: the link
  line carries `-lsundials_ida` and `-lsundials_idas`.

And the behaviour the fix is actually for, which neither of those shows:
linking a program against `-lsundials_ida -lsundials_idas` in that order and
running it under `LD_DEBUG=bindings` binds `IDACreate` to
**`libsundials_ida.so.7`**. Before the change there was no IDA on the line at
all and it could only have come from IDAS.

## 3. Your section 5 trap, and one we hit that is adjacent

The warm-cache early return is real and we avoided it rather than worked
around it: **verify a find-module change in a fresh build directory**, where
`${Prefix}_FOUND` is unset and the early return cannot fire. That is cheaper
than deleting three cache entries and their doc-comment lines, and it cannot
half-succeed.

The Makefile has the same shape one level down and it caught us first:
`make info` reads the *generated* `config/config.mk`, not `config/defaults.mk`,
so a `defaults.mk` edit shows no effect at all until `make config` re-runs.
Ours was four days stale and reported neither `-lsundials_ida` nor
`-lsundials_idas`, which looks exactly like the edit having failed.

**No further action wanted.** The report was accurate in every particular,
including the parts that are only visible with a second SUNDIALS install to
compare against.
