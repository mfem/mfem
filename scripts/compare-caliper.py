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
    result = {}

    raw_names = set()

    lines = Path(csv_path).read_text().splitlines()[1:]
    kernel_times = {}
    
    for line in lines:
        data = re.split(r'\s\s+',line)

        if len(data) < 4:
            continue
        program = data[0].strip()
        if program not in kernel_times:
            kernel_times[program] = {}
        mangled_name = data[1].strip()
        time_avg = float(data[2].split()[1])
        if is_jit:
            mangled_name = normalize_jit_name(mangled_name)
        if mangled_name in kernel_times:
            # print("DUPLICATE", mangled_name)
            kernel_times[program][mangled_name] += time_avg
        else:
            kernel_times[program][mangled_name] = time_avg

    for program in kernel_times.keys():
        if is_jit:
            jit_to_mangled = {name : normalize_jit_name(name) for name in kernel_times[program]}
            demangled_names = demangle(jit_to_mangled.keys())
            kernel_times[program] = {drop_return_types(normalize_lambdas(demangled_names[jit_to_mangled[name]])) : kernel_times[program][name] for name in kernel_times[program]}
        else:
            kernel_times[program] = {drop_return_types(normalize_lambdas(name)) : kernel_times[program][name] for name in kernel_times[program]}
            
    return kernel_times


def main():
    parser = argparse.ArgumentParser(description="Compare JIT vs AOT rocprof CSVs")
    parser.add_argument("aot_csv", help="AOT rocprof CSV")
    parser.add_argument("jit_csv", help="JIT rocprof CSV")
    parser.add_argument("--filter", default="Mass|Diffusion",
                        help="Regex to filter kernel names (default: Mass|Diffusion)")

    args = parser.parse_args()

    print("Loading JIT...")
    jit = load_and_process(args.jit_csv, is_jit=True)
    # print(jit)
    # exit()
    print("Loading AOT...")
    aot = load_and_process(args.aot_csv, is_jit=False)


    jit_df = None
    aot_df = None
    for bench, data_dict in jit.items():
        jit[bench]=pd.DataFrame({"KernelName":data_dict.keys(), "Time":data_dict.values()}).groupby("KernelName", as_index=False)["Time"].agg('sum')

        # grouped = df.groupby("KernelName", as_index=False)["DurationNs"]
        #       .agg(['mean', 'std'])
        #       .rename(columns={"mean": "AvgDurationNs",
        #                       "std": "StdDurationNs"})
    for bench, data_dict in aot.items():
        aot[bench]=pd.DataFrame({"KernelName":data_dict.keys(), "Time":data_dict.values()}).groupby("KernelName", as_index=False)["Time"].agg('sum')

        
    # Merge on demangled kernel name
    merged_dfs={}
    mega_df=pd.DataFrame(columns=["Program","KernelName","Speedup"])
    for key, aot_df in aot.items(): 
        merged_dfs[key] = pd.merge(
            aot_df, jit[key],
            on="KernelName",
            how="inner",
            suffixes=("_AOT", "_JIT")
        )
        merged_dfs[key]["Speedup"] = merged_dfs[key]["Time_AOT"] / merged_dfs[key]["Time_JIT"]
        
        # merged_dfs[key].to_csv(key+".csv")




if __name__ == "__main__":
    main()
