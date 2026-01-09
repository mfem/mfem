#!/usr/bin/env python3
import pandas as pd
import subprocess
import argparse
import re
import sys
from pathlib import Path

# Pattern for stripping JIT suffix: $jit$123456789$.kd
JIT_SUFFIX_RE = re.compile(r"\$jit\$\d+\$$")
RETURN_TYPE_RE = re.compile(
    r"""
    (?:void|int|double|float|auto|bool|long|short|
       unsigned|signed|char|size_t|
       #[\w:<>]+
       )          # or a fully qualified type
    \s+                   # space before function name
    #(?=[\w+]*\()      # lookahead: function-like thing
    """,
    re.VERBOSE,
)
KD_SUFFIX_RE = re.compile(r"\[clone\.kd\]")

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def drop_return_types(s: str) -> str:
    #return s
    return RETURN_TYPE_RE.sub("", s)

def normalize_lambdas(s: str) -> str:
    # {lambda(int)#1} → lambda(int)
    s = re.sub(r"\{\s*lambda\s*\(([^)]*)\)\s*#\d+\s*\}", r"lambda(\1)", s)

    # ::'lambda'(int) → ::lambda(int)
    s = re.sub(r"::'lambda\d*'\s*\(([^)]*)\)", r"::lambda(\1)", s)
    
    # :: lambda0(int) → ::lambda(int)
    s = re.sub(r"::lambda\d", r"::lambda", s)

    return s

def drop_kd_clone(s: str) -> str:
    return KD_SUFFIX_RE.sub("", s)

def demangle(names):
    """
    Use c++filt to demangle a list of symbols.
    Returns dict: mangled -> demangled
    """
    proc = subprocess.Popen(
        ["llvm-cxxfilt"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    input_data = "\n".join(names) + "\n"
    out, err = proc.communicate(input_data)

    if proc.returncode != 0:
        print("c++filt error:", err, file=sys.stderr)
        sys.exit(1)

    demangled = out.strip().splitlines()
    return dict(zip(names, demangled))


def normalize_jit_name(name):
    return JIT_SUFFIX_RE.sub("", name)


def load_and_process(csv_path, is_jit=False):
    type_dict = {
        "region" : str,
        "rocm.kernel.name" : str,
        "time_min" : float,
        "time_max" : float,
        "time_avg" : float
    }
    df = pd.read_json(csv_path, dtype=type_dict)
    # print( df["rocm.kernel.name"].dtype)
    if is_jit:
        df["rocm.kernel.name"] = df["rocm.kernel.name"].map(normalize_jit_name)
        demangled_names = demangle(df["rocm.kernel.name"])
        df["rocm.kernel.name"] = df["rocm.kernel.name"].map(demangled_names)
        # print(df.iloc[:10])
    df["rocm.kernel.name"] = df["rocm.kernel.name"].map(normalize_lambdas)
    df["rocm.kernel.name"] = df["rocm.kernel.name"].map(drop_return_types)
            
    return df


def main():
    parser = argparse.ArgumentParser(description="Compare JIT vs AOT rocprof CSVs")
    parser.add_argument("aot_csv", help="AOT rocprof CSV")
    parser.add_argument("jit_csv", help="JIT rocprof CSV")
    parser.add_argument("--filter", default="Mass|Diffusion",
                        help="Regex to filter kernel names (default: Mass|Diffusion)")

    args = parser.parse_args()

    print("Loading JIT...")
    jit_df = load_and_process(args.jit_csv, is_jit=True)

    print("Loading AOT...")
    aot_df = load_and_process(args.aot_csv, is_jit=False)

    
    jit = {}
    aot = {}
    renaming = {"rocm.kernel.name" : "KernelName",
                "time_avg" : "Time",
                "path" : "Program"}
    for bench in jit_df["path"].unique():
        jit[bench]=jit_df[jit_df["path"] == bench].rename(columns=renaming)
        aot[bench]=aot_df[aot_df["path"] == bench].rename(columns=renaming)

    # Merge on demangled kernel name
    merged_dfs={}
    # mega_df=pd.DataFrame(columns=["Program","KernelName","Speedup"])
    for key, aot_df in aot.items(): 
        merged_dfs[key] = pd.merge(
            aot_df, jit[key],
            on="KernelName",
            how="inner",
            suffixes=("_AOT", "_JIT")
        )
        merged_dfs[key]["Speedup"] = merged_dfs[key]["Time_AOT"] / merged_dfs[key]["Time_JIT"]
        
        merged_dfs[key][["Program_AOT", "KernelName", "Time_AOT", "Time_JIT", "Speedup"]].to_csv(key+".csv")




if __name__ == "__main__":
    main()
