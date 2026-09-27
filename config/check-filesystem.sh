#!/bin/sh
# Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

# Arguments are the C++ compiler command and its compile/link flags. Print the
# extra filesystem library needed, or "unavailable" on failure. Only link the
# probe; never execute it, so this also works with cross compilers.
probe_dir=$(mktemp -d "${TMPDIR:-/tmp}/mfem-filesystem.XXXXXXXX") || {
    echo unavailable
    exit 1
}
trap 'rm -rf "$probe_dir"' 0
trap 'exit 1' HUP INT TERM

cat > "$probe_dir/probe.cpp" <<'EOF'
#include <filesystem>
int main() { return std::filesystem::exists(".") ? 0 : 1; }
EOF

for library in none stdc++fs c++fs; do
    if [ "$library" = none ]; then
        if "$@" "$probe_dir/probe.cpp" -o "$probe_dir/probe" \
            > "$probe_dir/log" 2>&1; then
            exit 0
        fi
    elif "$@" "$probe_dir/probe.cpp" "-l$library" -o "$probe_dir/probe" \
        > "$probe_dir/log" 2>&1; then
        echo "-l$library"
        exit 0
    fi
done
cat "$probe_dir/log" >&2
echo unavailable
exit 1
