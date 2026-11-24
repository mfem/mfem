# MFEM Unit Tests

This directory contains MFEM's suite of unit tests, using the
[Catch2](https://github.com/catchorg/Catch2) unit testing framework.

## Test executables

MFEM's unit test suite includes a number of executables:

* `unit_tests`
* `sedov_tests`, testing a Sedov hydrodynamics case
* `tmop_pa_tests`, testing TMOP with partial assembly
* `debug_device_tests`, testing the memory manager with the fixed `debug` device

`unit_tests` contains serial and MPI tests when MFEM enables MPI. It runs serial
tests by default; `--parallel` initializes MPI and hypre and selects tests tagged
`[Parallel]`. This option must also be supplied for one-rank MPI runs.

Sedov and TMOP each have separate serial and MPI executables:
`sedov_tests` / `psedov_tests` and `tmop_pa_tests` / `ptmop_pa_tests`. The MPI
executables are built only when MFEM enables MPI and always run parallel tests.
For example:

```sh
./unit_tests
mpirun -np 4 ./unit_tests --parallel
./sedov_tests --device cpu
mpirun -np 4 ./psedov_tests --device cpu
mpirun -np 4 ./ptmop_pa_tests --device gpu
```

Each accepts `--device` to select an enabled MFEM backend (default: `cpu`).
Configured builds register runs for `gpu`, `raja-gpu`, `omp`, and `raja-omp` as
applicable; `raja-omp` requires OpenMP in both MFEM and RAJA. Sedov and serial
TMOP also run with `debug`, and Sedov runs with `gpu:uvm` when a GPU backend is
enabled. The serial `unit_tests` CPU path leaves device configuration to individual
tests; its parallel path explicitly configures even the CPU device.

## Basics of using Catch

To run the unit tests, any of the executables listed above can be run from the
command line, for example
```
./unit_tests
```
which will run _all_ serial unit tests. If you want to run only a specific test
case, it is possible to specify the name of the test case as an argument to the
test executable (the names of the test cases are given as the first argument to
the `TEST_CASE` macro in the source code). For example
```
./unit_tests "NCMesh PA diagonal"
```
will run only test case testing partial assembly of the diagonal on
non-conforming meshes. Test cases are optionally given one or more _tags_, which
can be used to group test cases together. Tags can also be specified on the
command line, for example
```
./unit_tests "[NCMesh]"
```
will run all the tests that relate to non-conforming meshes.

## Listing tests and tags

It is possible to list all test cases with the `-l` flag, for example
```
./unit_tests -l
```
will list all serial unit tests, and
```
./unit_tests -l "[NCMesh]"
```
will list all test cases with the given tag.
```
./unit_tests -t
```
will list all available tags, along with the number of test cases that are
assigned to each tag.

## Special tags

For the most part, tags are just used to group similar tests together according
to their subject matter. However, several specific tags have special meanings,
and those are:

* `[Parallel]`, which indicates a test that runs with `unit_tests --parallel`
  or in the MPI miniapp executables. Serial runs exclude parallel tests.
* `[GPU]`, which indicates that a test will be run when `--device` selects a backend
  other than `cpu`. These tests also run with the default CPU backend. With a
  non-CPU backend in a parallel run, only tests marked with both `[GPU]` and
  `[Parallel]` run.
* `[MFEMData]`, which indicates that a test requires access to a clone of the
  MFEM data repository (see the `--data` flag below), in order to run tests on
  some larger mesh files. By default, tests tagged with this tag are skipped,
  unless the `--data` flag is provided.

## Special command line arguments

In addition to the standard Catch command line arguments (which can be viewed
with the `-h` or `--help` flag), MFEM's unit tests support these additional
command line arguments:

* `--parallel`, which runs the MPI tests in `unit_tests`. Available only in MPI
  builds. The MPI Sedov and TMOP executables select their parallel path directly.

* `--device`, which selects the MFEM device backend (default: `cpu`).
  `debug_device_tests` always uses `debug` and does not offer this flag.

* `--all`, which enables some more thorough tests, at the expense of longer
  runtimes. This sets the global variable `launch_all_non_regression_tests` to
  true. Available in `unit_tests`, `tmop_pa_tests`, and `ptmop_pa_tests`.
* `--data`, which specifies a path to a clone of the MFEM [data
  repository](https://github.com/mfem/data), which contains some larger mesh
  files. If this argument is provided, tests tagged with `[MFEMData]` will be
  run, and they will have access to the files in the data repo through the
  `mfem_data_dir` global variable. Available in `unit_tests`.

## Test output and debug messages

By default, MFEM's unit tests display relatively little output (a couple of info
lines at the beginning, and a summary at the end with the number of test cases
and assertions that were run). If a test fails, some additional information
about the failing test will be printed. Output to `mfem::out` and `mfem::err` is
suppressed by default.

To enable more verbose test output, run the unit tests with the `-s` or
`--success` flag, which will print a message for every successful test
assertion, including some additional informational messages. With this option,
output to `mfem::out` and `mfem::err` is enabled.

## Writing unit tests

The following are some guidelines for developers writing unit tests:

* Give your test case a concise yet descriptive name. Whitespace is allowed.
* Tag your test with the relevant tags. Look at similar tags to see what
  relevant tags are in use. Class names are often used as tags. Another common
  tag is `[PartialAssembly]`. See also the section on [special
  tags](#special-tags).
* Do not use `std::cout` or `std::cerr` in your test. Prefer the Catch macros
  [`INFO`](https://github.com/catchorg/Catch2/blob/v2.x/docs/logging.md#top),
  [`CAPTURE`](https://github.com/catchorg/Catch2/blob/v2.x/docs/logging.md#quickly-capture-value-of-variables-or-expressions),
  and similar. If you need more control over the output, prefer `mfem::out` and
  `mfem::err`.
* Use the
  [`GENERATE`](https://github.com/catchorg/Catch2/blob/v2.x/docs/generators.md#top)
  macro instead of nested for-loops when testing many combinations of
  parameters.

## Customizing the shared runner

`RunCatchSession` accepts `CatchSessionOptions` to control the optional `--all`
and `--data` flags, enable the serial/MPI selector with `enable_parallel`,
configure the CPU device explicitly, or specify a fixed
backend. Its `configure_cli` hook extends Catch's parser before parsing arguments.
The values bound to custom options must remain alive until the session returns:

```cpp
int iterations = 10;
CatchSessionOptions options;
options.enable_all = options.enable_data = false;
options.configure_cli = [&](Catch::Session &session)
{
   session.cli(session.cli()
               | Catch::clara::Opt(iterations, "count")
                 ["--iterations"] ("number of iterations"));
};
return RunCatchSession(argc, argv, {"[MyTests]"}, options);
```

The optional `before_tests` hook runs after device configuration and can return a
nonzero exit code, for example `MFEM_SKIP_RETURN_VALUE` for an unsupported backend.
