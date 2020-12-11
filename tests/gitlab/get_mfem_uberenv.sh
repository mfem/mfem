#!/bin/bash

set -o errexit
set -o nounset

uberenv_url="https://github.com/mfem/mfem-uberenv.git"
uberenv_ref="0a8f8715ae328caab0690832d1e85e1f14adb018"


[[ ! -d scripts/uberenv ]] && git clone ${uberenv_url} tests/uberenv
cd tests/uberenv
git fetch origin ${uberenv_ref}
git checkout ${uberenv_ref}
cd -
