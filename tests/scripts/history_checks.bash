#! /bin/bash

function check_bin_in_commit() {

    binary_files=""

    changes=$(git diff --pretty=format:"" --shortstat HEAD^ \
              | sed -e 's/[^0-9,]//g' -e 's/,/+/g' | bc)

    if (( $changes > 50000 ))
    then
        msg "$(git log -1 --format=%H)"
        msg "ATTENTION: This commit is unusually large"
        return 1
    fi

    return 0
}

function check_commit_size() {

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

function check_branch() {

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

function msg() {
    echo "history_checks: $*"
}

function err() {
    msg $* 1>&2
}
