# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.googlecode.com.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

# Check the number of changes introduced by the last commit
# (sum of # of files changed, lines inserted and deleted).
function check_commit_size()
{
    binary_files=""

    # Compute the number of total changes (files+insertions+deletions)
    # in the commit, e.g.
    # "3 files changed, 66 insertions(+), 45 deletions(-)" -> 111
    changes=$(git diff --shortstat HEAD^ \
              | sed -e 's/[^0-9,]//g' -e 's/,/+/g' | bc)

    if (( $changes > 50000 ))
    then
        msg "$(git log -1 --format=%H)"
        msg "ATTENTION: This commit is unusually large! (${changes} changes)"
        return 1
    fi

    return 0
}

# Check if files modified in the last commit are binaries.
function check_bin_in_commit()
{
    binary_files=""

    if [[ -n ${binary_files} ]]
    then
        msg "$(git log -1 --format=%H)"
        msg "ATTENTION: binary file(s) added or modified"
        echo "${binary_files}"
        return 1
    fi

    return 0
}

# Check all the commits in the current branch, starting with the common ancestor
# with master, for large and binary files
function check_branch()
{
    current_branch="$(git symbolic-ref --short HEAD)"
    reference_branch="master"
    common_ancestor="$(git merge-base ${current_branch} ${reference_branch})"

    binary_found=0
    large_commit=0

    while read -r rev; do
        git checkout -q "$rev"
        # Read: "if check_bin_in_commit fails"
        if ! check_bin_in_commit
        then
            binary_found=1
        fi
        if ! check_commit_size
        then
            large_commit=1
        fi
    done < <(git rev-list --reverse ${common_ancestor}..${current_branch})

    git checkout -q ${current_branch}

    return $binary_found || $large_commit
}

function msg()
{
    echo "history_checks: $*"
}

function err()
{
    msg $* 1>&2
}
