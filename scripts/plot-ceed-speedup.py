import matplotlib.pyplot as plt

raw_data = """
BK1Mass/7                            15.300        1.880       8.14x
BP1Mass/7                           524.000       93.300       5.62x
BK6VectorDiffusion/1                  1.280        0.267       4.79x
BP6VectorDiffusion/1                 42.400        9.970       4.25x
BK2VectorMass/1                       0.863        0.206       4.19x
BP2VectorMass/1                      29.000        7.950       3.65x
BK4VectorDiffusion/3                 31.500        9.170       3.44x
BP4VectorDiffusion/3               1017.000      303.000       3.36x
BK6VectorDiffusion/3                 20.300        6.150       3.30x
BK6VectorDiffusion/4                 43.400       13.400       3.24x
BP6VectorDiffusion/3                660.000      206.000       3.20x
BP6VectorDiffusion/4               1411.000      446.000       3.16x
BK4VectorDiffusion/1                  3.540        1.150       3.08x
BK6VectorDiffusion/2                  6.490        2.110       3.08x
BK2VectorMass/7                     128.000       42.100       3.04x
BP4VectorDiffusion/1                115.000       38.000       3.03x
BP6VectorDiffusion/2                212.000       71.200       2.98x
BP2VectorMass/7                    4211.000     1477.000       2.85x
BK4VectorDiffusion/4                 62.100       25.800       2.41x
BK4VectorDiffusion/2                 12.300        5.170       2.38x
BP4VectorDiffusion/4               2003.000      843.000       2.38x
BP4VectorDiffusion/2                398.000      169.000       2.36x
BK2VectorMass/8                     152.000       67.400       2.26x
BK6VectorDiffusion/7                294.000      131.000       2.24x
BK2VectorMass/3                      10.500        4.760       2.21x
BP6VectorDiffusion/7               9476.000     4299.000       2.20x
BK3Diffusion/7                        9.620        4.390       2.19x
BP2VectorMass/8                    4993.000     2329.000       2.14x
BK1Mass/6                             2.360        1.110       2.13x
BP2VectorMass/3                     344.000      162.000       2.12x
BK4VectorDiffusion/7                366.000      175.000       2.09x
BK2VectorMass/6                      70.400       33.700       2.09x
BP4VectorDiffusion/7              11798.000     5712.000       2.07x
BK6VectorDiffusion/5                 84.800       42.200       2.01x
BK2VectorMass/5                      33.700       17.000       1.98x
BP6VectorDiffusion/5               2755.000     1390.000       1.98x
BP2VectorMass/6                    2334.000     1185.000       1.97x
BP3Diffusion/7                      340.000      174.000       1.95x
BK6VectorDiffusion/8                424.000      225.000       1.88x
BP6VectorDiffusion/8              13691.000     7368.000       1.86x
BK4VectorDiffusion/5                116.000       62.500       1.86x
BP4VectorDiffusion/5               3752.000     2040.000       1.84x
BK6VectorDiffusion/6                175.000       95.200       1.84x
BP2VectorMass/5                    1119.000      613.000       1.83x
BP6VectorDiffusion/6               5651.000     3123.000       1.81x
BK2VectorMass/2                       3.600        2.000       1.80x
BK2VectorMass/4                      19.000       10.600       1.79x
BK4VectorDiffusion/6                220.000      123.000       1.79x
BP4VectorDiffusion/6               7130.000     4031.000       1.77x
BP2VectorMass/2                     119.000       67.600       1.76x
BP2VectorMass/4                     628.000      358.000       1.75x
BK5Diffusion/8                      136.000       79.400       1.71x
BP5Diffusion/8                     4423.000     2594.000       1.71x
BK4VectorDiffusion/8                513.000      309.000       1.66x
BK3Diffusion/8                      165.000      100.000       1.65x
BP4VectorDiffusion/8              16552.000    10050.000       1.65x
BP3Diffusion/8                     5344.000     3262.000       1.64x
BK3Diffusion/5                        2.270        1.440       1.58x
BP1Mass/6                           110.000       69.800       1.58x
BP3Diffusion/5                       85.800       59.100       1.45x
BK3Diffusion/6                        3.720        2.760       1.35x
BK5Diffusion/6                        2.380        1.820       1.31x
BK5Diffusion/7                        4.410        3.430       1.29x
BP3Diffusion/6                      153.000      122.000       1.25x
BP5Diffusion/7                      174.000      143.000       1.22x
BK3Diffusion/1                        0.134        0.112       1.20x
BP5Diffusion/6                      110.000       92.200       1.19x
BP3Diffusion/1                        5.350        4.630       1.16x
BK1Mass/5                             0.837        0.754       1.11x
"""
#BK5Diffusion/1                        0.108        0.099       1.09x
#BP5Diffusion/1                        4.540        4.220       1.08x
#BK3Diffusion/4                        0.875        0.815       1.07x
#BP1Mass/5                            39.900       37.200       1.07x
#BK1Mass/1                             0.100        0.094       1.06x
#BP3Diffusion/4                       35.400       33.400       1.06x
#BK3Diffusion/2                        0.202        0.191       1.06x
#BK5Diffusion/2                        0.167        0.158       1.06x
#BP1Mass/1                             4.260        4.050       1.05x
#BK5Diffusion/4                        0.597        0.571       1.05x
#BP5Diffusion/2                        7.130        6.840       1.04x
#BP3Diffusion/3                       18.100       17.400       1.04x
#BK3Diffusion/3                        0.446        0.429       1.04x
#BP5Diffusion/4                       26.500       25.600       1.04x
#BP3Diffusion/2                        8.210        7.940       1.03x
#BK5Diffusion/3                        0.307        0.301       1.02x
#BK5Diffusion/5                        1.100        1.080       1.02x
#BP1Mass/2                             7.210        7.100       1.02x
#BP5Diffusion/3                       13.600       13.400       1.01x
#BP5Diffusion/5                       48.200       47.500       1.01x
#BP1Mass/4                            23.000       22.700       1.01x
#BK1Mass/2                             0.171        0.169       1.01x
#BK1Mass/4                             0.485        0.480       1.01x
#BP1Mass/3                            13.800       13.700       1.01x
#BK1Mass/3                             0.313        0.312       1.00x
#BK1Mass/8                            57.100       58.700       0.97x
#BP1Mass/8                          1879.000     1963.000       0.96x
names = []
speedups = []

for line in raw_data.strip().splitlines():
    parts = line.split()
    name = parts[0]
    speedup = float(parts[-1].replace("x", ""))
    names.append(name)
    speedups.append(speedup)

# Sort by speedup descending
sorted_data = sorted(zip(names, speedups), key=lambda x: x[1], reverse=True)
names, speedups = zip(*sorted_data)

# Plot
plt.figure(figsize=(20, 10))
plt.bar(names, speedups)
plt.axhline(1.0, linestyle="--", linewidth=1)

plt.ylabel("Speedup")
plt.xlabel("Benchmark")
plt.xticks(rotation=75, ha="right", fontsize=8)
#plt.figure(figsize=(24, 6))
plt.title("CEED benchmarks speedup using JIT compilation (MFEM)")

plt.tight_layout()
plt.savefig("ceed.png")
