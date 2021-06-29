                    Finite Element Discretization Library
                                   __
                       _ __ ___   / _|  ___  _ __ ___
                      | '_ ` _ \ | |_  / _ \| '_ ` _ \
                      | | | | | ||  _||  __/| | | | | |
                      |_| |_| |_||_|   \___||_| |_| |_|

                               https://mfem.org


This directory contains recommended git hooks, which are scripts that can be
used to improve your development experience with MFEM:

### The hooks

* `pre-commit` is a hook that will be applied before each commit and run
`astyle` on the code. This will ensure that your changes comply with the MFEM
code styling guidelines.

* `pre-push` is a hook that will be applied before each push to run a quick set
of tests that verify that your files headers are in compliance, and that you did
not add any large files to the repo.

### Setup

To setup the git hooks, run `make hooks`, which creates symlinks to the hooks in
the `.git/hooks` directory. Individual hooks can be enabled by manually creating
symlinks.

(You may also copy the script directly and customize it further, but this way
you may miss additional updates in the future.)

### Failures

The `branch-history` check can fail in some cases when the history is OK. For
example, when a large number of files were modified for a legitimate reason, or
when a picture was added for documentation.

If that is the case, make sure the failure is indeed justified, and rerun the
push command with the `--no-verify` option. This will skip the hooks, allowing
you to push those changes.
