#! /bin/bash

function check_bin_in_commit() {

    binary_files=""

    for file in $(git diff --name-only HEAD^)
    do
        if [[ -e $file ]] && (! file --mime $file | grep -q text)
        then
            binary_files="${binary_files}   ${file}\n"
        fi
    done

    if [[ -n ${binary_files} ]]
    then
        msg "$(git log -1 --format=%H)"
        msg "ATTENTION: binary file(s) added or modified"
        echo "${binary_files}"
        #[[ ! $1 == "--safe" ]] && exit 1
        return 1
    fi

    return 0
}

function check_bin_in_branch() {

    current_branch="$(git symbolic-ref --short HEAD)"
    reference_branch="master"
    common_ancestor="$(git merge-base ${current_branch} ${reference_branch})"

    binary_found=0

    while read -r rev; do
        git checkout -q "$rev"
        # Read: "if check_bin_in_commit fails"
        if ! check_bin_in_commit
        then
            binary_found=1
        fi
    done < <(git rev-list --reverse ${common_ancestor}..${current_branch})

    git checkout -q ${current_branch}

    return $binary_found
}

function msg() {
    echo "history_checks: $*"
}

function err() {
    msg $* 1>&2
}
