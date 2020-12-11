#!/bin/bash

set -o errexit
set -o nounset

uberenv_url="https://github.com/mfem/mfem-uberenv.git"
uberenv_ref="1012c077d22161d96e1da3ae70dc5a5e872de4e6"


[[ ! -d scripts/uberenv ]] && git clone ${uberenv_url} tests/uberenv
cd tests/uberenv
git fetch origin ${uberenv_ref}
git checkout ${uberenv_ref}
cd -
