#!/bin/bash
# This script is meant be launch in the root directory of MFEM do compile all
# the dependencies of MFEM for all the specs in CI. By doing so, we obtain a
# Spack instance that can be used as an upstream.

# This patch applies shared permissions to Spack installations.
git apply tests/gitlab/upstream-permission.patch

# call uberenv for all specs in CI
git grep -e "^[^#]" .gitlab | grep "SPEC" \
                            | cut -d' ' -f6- \
                            | sed 's/"//g' \
                            | while read -r line; do
  python ./tests/uberenv/uberenv.py --spec="$line"
done

# We revert the patch to leave the repo as found.
git apply -R tests/uberenv/upstream-permission.patch
