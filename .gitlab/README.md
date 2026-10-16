                    Finite Element Discretization Library
                                   __
                       _ __ ___   / _|  ___  _ __ ___
                      | '_ ` _ \ | |_  / _ \| '_ ` _ \
                      | | | | | ||  _||  __/| | | | | |
                      |_| |_| |_||_|   \___||_| |_| |_|

                               https://mfem.org


This directory contains most of the GitLab CI configuration. MFEM runs both PR
and nightly testing on GitLab.

# Structure

## Top level

The root configuration file is `.gitlab-ci.yml` at the root of MFEM repo.
This file only defines one stage, in which we trigger several
sub-pipelines.

We use sub-pipelines to isolate the test for one combination of `machine`
and `test type`.

Machines typically include:

* Dane: Intel Sapphire Rapids
* Lassen: Power9 + Nvidia GPU
* Corona: AMD GPU

Test types include:

* Build and test: Spack driven build of dependencies, mfem build, mfem
  test
* Baseline: Script driven build of dependencies, thorough testing

⚠️ The sub-pipeline design allows to add a new machine or a new test type without
altering the scheduling, execution and displaying of the others.

## Sub-pipelines

Each file is this directory is the root configuration file for one
sub-pipeline. The naming reflects the corresponding couple (`machine`,
`test_type`).

Those files define the *stages* and the *jobs* for the sub-pipeline. They
also contain any configuration that cannot be shared. For the most part
though, the configuration is shared and is placed in `.gitlab/configs`.

We try to keep scripts out of the CI config and share them among similar
jobs. They are gathered in `.gitlab/scripts`.

## Scripts

Scripts specific to the CI only are in `.gitlab/scripts`. It is best practice
to keep scripts outside the CI configuration (no bash scripts embedded in a
yaml file) because it helps with readability, maintenance and also with
transition to another CI system.

⚠️ Most of the scripts there are driven by environment variables and do not have a
usage function. This should be improved.


# More testing

## Adding a new target to a build_and_test pipeline

`build_and_test` pipelines rely on Spack to install dependencies. Spack is
driven by Uberenv which helps freezing Spack configuration: the goal being to
point to specific commit in Spack and isolate its configuration so that it is
not influenced by the user environment. More documentation about this can be
found in `tests/gitlab`.

In the end, the MFEM target for which to build the dependencies is expressed
with a spack spec of MFEM, within the limits permitted by the MFEM spack
package.

In any build-and-test sub-pipeline a job basically consists in defining the
spack spec to use. Adding a job on Dane for example resumes to:

```yaml
<job_name>:
  variables:
    SPEC: "<spack_spec>"
  extends: .build_and_test_on_dane
```

The remaining and non trivial work is to make sure this spec is working. To
test a spec before adding it, or reproduce a CI configuration, please refer to
`tests/gitlab/reproduce-ci-jobs-interactively.md`.

⚠️ It is assumed that the spack spec applies to `mfem@develop`. That's why in the
CI all the specs start with the compiler or the variants to apply to mfem. The
mechanism still works with a full spec.
