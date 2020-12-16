#!/bin/bash

set -o errexit
set -o nounset

uberenv_url="https://github.com/mfem/mfem-uberenv.git"
uberenv_ref="c9f95ab4d51c283a5e18d8f850a52935a402f679"


[[ ! -d scripts/uberenv ]] && git clone ${uberenv_url} tests/uberenv
cd tests/uberenv
git fetch origin ${uberenv_ref}
git checkout ${uberenv_ref}
cd -
