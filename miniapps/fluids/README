# Incompressible Schrödinger Flow

This miniapp introduces the Incompressible Schrödinger Flow (ISF), a ℂ²-valued
Schrödinger equation that incorporates an incompressibility constraint to
model classical incompressible fluids. Drawing from Madelung's quantum
hydrodynamics, this approach provides straightforward implementation
and exhibits strong performance in capturing thin vortex behaviors.

___

Let Ψ be a two components wave function: [ϕ₁,ϕ₂]ᵀ, let p be the potential
to be chosen so that the fluid is incompressible, using a splitting method,
we solve the following incompressible Schrödinger equations:

iℏ ∂Ψ/∂t = -½ℏ²ΔΨ + pΨ, with the constraint: Re(<ΔΨ,iΨ>) = 0 and |Ψ|² = 1

which leads to the following steps:

 1. Linear Schrödinger equation solver
 2. Normalization of the wave function
 3. Pressure projection
 4. Enforce geometry constraints
 5. Compute velocity field for visualization

___

For more details see:

[1] Chern, Knöppel, Pinkall, Schröder and Weißmann.
    [Schrödinger’s Smoke](https://dl.acm.org/doi/10.1145/2897824.2925868).
    ACM Trans. Graph., 2016.

[2] Albert Chern. [Fluid Dynamics with Incompressible Schrödinger Flow](https://cseweb.ucsd.edu/~alchern/projects/PhDThesis/).
    Doctoral Dissertation, California Institute of Technology, 2017.
