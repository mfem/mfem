#!/bin/bash

set -o errexit
set -o nounset

uberenv_url="ssh://git@czgitlab.llnl.gov:7999/bernede1/mfem-uberenv.git"
uberenv_ref="71d85ce03411259ec145197b6155801277f49229"

git clone ${uberenv_url} scripts/uberenv
cd scripts/uberenv
git fetch origin ${uberenv_ref}
git checkout ${uberenv_ref}
cd -
