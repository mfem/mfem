#!/bin/bash

set -o errexit
set -o nounset

uberenv_url="ssh://git@czgitlab.llnl.gov:7999/bernede1/mfem-uberenv.git"
uberenv_ref="51b81411bb3d87bdf424aabd65d6f369badc5fd8"

git clone ${uberenv_url} scripts/uberenv
cd scripts/uberenv
git fetch origin ${uberenv_ref}
git checkout ${uberenv_ref}
cd -
