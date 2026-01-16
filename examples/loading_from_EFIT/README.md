# Structure-Preserving Transfer of Grad-Shafranov Equilibria to Magnetohydrodynamic Solvers

This repository contains the implementation for structure-preserving transfer of Grad-Shafranov (GS) equilibria to magnetohydrodynamic (MHD) solvers using compatible finite element spaces within the de Rham complex.

## Overview

Magnetohydrodynamic simulations of magnetically confined plasmas require initial conditions that satisfy force balance, typically obtained from equilibria computed by Grad–Shafranov solvers. However, transferring these equilibria to MHD discretizations can introduce numerical errors that disturb the equilibrium. This work identifies and analyzes key error sources within finite element frameworks, focusing on the preservation of force balance and the divergence-free property of the magnetic field.

### Key Findings

- Errors primarily arise from (1) incompatible finite element spaces between GS and MHD solvers, (2) mesh misalignment, and (3) under-resolved gradients near the separatrix.
- Equilibria are best preserved when structure-preserving finite element spaces are employed, meshes are aligned and refined, and magnetic fields are projected into div-conforming spaces to maintain force balance.
- Projection into curl-conforming spaces, while less optimal for force balance, provides weak preservation of the divergence-free condition.

## Repository

The implementation is available in the MFEM repository:
- **Branch**: [`tds-load`](https://github.com/mfem/mfem/tree/tds-load)

## Mathematical Background

Assuming axisymmetry in a tokamak, the magnetic field $\mathbf{B}$ is represented in terms of the poloidal flux function $\psi(r,z)$ and the toroidal field function $f(\psi)$ as

$$\mathbf{B} = \nabla \times \left( \frac{\psi}{r}\,\mathbf{e}_\phi \right) + \frac{f(\psi)}{r}\,\mathbf{e}_\phi,$$

with poloidal and toroidal components

$$\mathbf{B}_p = \frac{1}{r}\nabla \psi \times \mathbf{e}_\phi, \qquad B_\phi = \frac{f(\psi)}{r}.$$

Under magnetohydrodynamic equilibrium, force balance requires

$$\mathbf{J} \times \mathbf{B} = \nabla p,$$

and we should verify that

$$[\mathbf{B} \times \mathbf{J}]_p = 0 \qquad [\mathbf{B} \times \mathbf{J}]_t = 0.$$

Additionally, the magnetic field must be divergence-free:

$$\nabla \cdot \mathbf{B} = 0.$$

## Citation

If you use this code in your research, please cite:

```bibtex
@article{zhang2025structure,
  title={Structure-Preserving Transfer of Grad-Shafranov Equilibria to Magnetohydrodynamic Solvers},
  author={Zhang, Rushan and Wimmer, Golo and Tang, Qi},
  journal={arXiv preprint arXiv:2511.07763},
  year={2025}
}
```
