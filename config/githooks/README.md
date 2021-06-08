                    Finite Element Discretization Library
                                   __
                       _ __ ___   / _|  ___  _ __ ___
                      | '_ ` _ \ | |_  / _ \| '_ ` _ \
                      | | | | | ||  _||  __/| | | | | |
                      |_| |_| |_||_|   \___||_| |_| |_|

                               https://mfem.org


This directory contains git hooks, scripts that can be used to improve your development experience in MFEM:

### The hooks

* `pre-commit` is a hook that will be applied before each commit and run `astyle` on the code. This will ensure that your changes comply with the style guidelines.

* `pre-push` is a hook that will be applied before each push to run a quick set of tests that will verify that your files headers are in compliance, and that you did not add any files too large, or binaries, to the repo.

### Setup

To setup a git hook, create a symlink to it from the `.git/hooks` directory.

```bash
  cd <mfem_root_dir>/.git/hooks
  ln -s ../../config/githooks/<hook_script> <hook_script>
```

You may also copy the script directly, if you want to personalize it. You just won’t benefit from updates automatically if you do so.

### Failures

The `branch-history` check can fail whereas the history is OK. This is the case when a large number of files was modified for a legitimate reason, or if a picture was added for documentation for example.

In such case, and only after making sure nothing is wrong, you can rerun the push command with the `--no-verify` option. This will skip the hooks, allowing you to push those changes.
