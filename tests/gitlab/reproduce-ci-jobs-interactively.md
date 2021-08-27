# HowTo: Reproduce CI jobs interactively.

We rely on Spack, driven by uberenv, to build MFEM dependencies automatically
in CI. Then we use a script that will be retrieve the configuration file
generated and use it to build MFEM before running the tests.

This process is straightforward to reproduce. However the steps are not easy
to extract from the CI configuration, hence this article.

**WARNING**

This will only work on the same machines as used for CI.

**NOTE**

The `build_and_test` script controlling the build of MFEM and its dependencies
has two modes:
- The CI mode, with no argument, will build the deps, then MFEM and then run
  tests for the specified spec. This assumes no pre-existing configuration file
  in `host-configs` directory.
- The Interactive mode, presented here, where we first build the dependencies
  and then use a configuration file in `host-configs` directory to configure
  MFEM build.

## Prerequisite: Retrieve Uberenv

```bash
tests/gitlab/get_mfem_uberenv
```

We have a script to automatically download and install an MFEM-tuned uberenv
instance.

Uberenv will be placed in `tests/uberenv`, along with spack configuration files
for each machine we currently support, and possibly some Spack packages we
patch.


## Install Dependencies

### Using the CI script

```bash
SPEC="%gcc@6.1.0 +sundials" ./tests/gitlab/build_and_test --deps-only
```

The CI script has three steps that can be run individually with the options
`--deps-only`, `--build-only`, `--test-only`.

We ask to build only the dependencies, and we need to provide a spack spec
through an environment variable.

Virtually any spec can be provided, but you should check which compilers are
defined in the spack configuration
(`tests/uberenv/spack-configs/<sys_type>/compilers.yaml).

As a result, dependencies will be installed under `uberenv_libs`.
A configuration file `hc-<infos>.mk` will be generated in `host-configs`, or
directly in mfem root directory when not using `--deps-only`.

**NOTE**

The `build_and_test` script behaves slightly differently between CI context and
elsewhere (depending on environment variable $CI). In CI, and if launched on
quartz, ruby or corona, the script will build and install dependencies in
`/dev/shm` for better performance. However, this is only valid if we don’t want
the installation to persist. Installation will happen locally to the uberenv
directory if not in CI context.

### Calling uberenv directly

```bash
python ./tests/uberenv/uberenv.py --spec="%gcc@6.1.0 +sundials"
```

This is essentially the command the CI script runs in the end.

**NOTE**

When using this command, the configuration file will be in the mfem root dir.

## Build and test MFEM

### Without using scripts

```bash
cp host-configs/hc-<spec-identification>.mk config/config.mk
make all -j 8
make test
```

The key here is obviously the configuration file. It will direct to build to
the dependencies location, and apply any option selected with MFEM spec
variants in Spack.

### Using the CI script

```bash
HOST_CONFIG=host-configs/hc-<infos>.mk ./tests/gitlab/build_and_test --build-only
```

We can also use the CI script to only build mfem from a given configuration
file. We could even use `--test-only` option, which would also build MFEM to
make sure to use the provided configuration file.

**NOTE**

The CI script can be used without option only if no configuration file is
present in the `host-configs` directory. That is because in this CI mode, we
only build one set of dependencies and MFEM target per clone of MFEM / CI job.

