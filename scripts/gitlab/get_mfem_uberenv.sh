#!/bin/bash

set -o errexit
set -o nounset

uberenv_url="ssh://git@czgitlab.llnl.gov:7999/mfem/mfem-uberenv.git"
uberenv_ref="c2f3497e9a392885058dd2ba93e2f8c071655726"

git clone ${uberenv_url} scripts/uberenv
cd scripts/uberenv
git fetch origin ${uberenv_ref}
git checkout ${uberenv_ref}
cd -
