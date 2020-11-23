#!/bin/bash

set -o errexit
set -o nounset

uberenv_url="ssh://git@czgitlab.llnl.gov:7999/bernede1/mfem-uberenv.git"
uberenv_ref="5001ad5b0c4759cacc55153e9dca08db154a4a1f"

git clone ${uberenv_url} scripts/uberenv
cd scripts/uberenv
git fetch origin ${uberenv_ref}
git checkout ${uberenv_ref}
cd -
