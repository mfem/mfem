# MFEM Unit Tests

This directory contains MFEM's suite of unit tests, using the
[Catch2](https://github.com/catchorg/Catch2) unit testing framework.

## Test executables

MFEM's unit test suite includes a number of executables:

* `unit_tests`
* `gpu_unit_tests` if MFEM is compiled with CUDA/HIP support
* `sedov_tests_cpu`, `sedov_tests_debug` (and `sedov_tests_gpu` and
  `sedov_tests_gpu_uvm` if GPU is enabled), testing a Sedov hydrodynamics case
* `tmop_pa_tests_cpu`, `tmop_pa_tests_debug` (and `tmop_pa_tests_gpu` if GPU
  is enabled), testing TMOP with partial assembly

There are also parallel versions of these executables (prefixed with `p`), which
are built if MFEM is compiled with MPI enabled.

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

* `[Parallel]`, which indicates that a test will **not** be tested with the
  serial test executables, and will only be tested with the parallel executable
  (e.g. `punit_tests`). `punit_tests` will only run tests marked with
  `[Parallel]`.
* `[GPU]`, which indicates that a test will be tested with the GPU executables
  (e.g. `gpu_unit_tests`). These tests will still be run by the standard (CPU)
  executables. `gpu_unit_tests` will only run tests marked with `[GPU]`, and its
  parallel version `pgpu_unit_tests` will only run tests marked with _both_
  `[GPU]` and `[Parallel]`.
* `[MFEMData]`, which indicates that a test requires access to a clone of the
  MFEM data repository (see the `--data` flag below), in order to run tests on
  some larger mesh files. By default, tests tagged with this tag are skipped,
  unless the `--data` flag is provided.

## Special command line arguments

In addition to the standard Catch command line arguments (which can be viewed
with the `-h` or `--help` flag), MFEM's unit tests support two additional
command line arguments:

* `--all`, which enables some more thorough tests, at the expense of longer
  runtimes. This sets the global variable `launch_all_non_regression_tests` to
  true.
* `--data`, which specifies a path to a clone of the MFEM [data
  repository](https://github.com/mfem/data), which contains some larger mesh
  files. If this argument is provided, tests tagged with `[MFEMData]` will be
  run, and they will have access to the files in the data repo through the
  `mfem_data_dir` global variable.

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
