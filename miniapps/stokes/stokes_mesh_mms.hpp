#ifndef MFEM_STOKES_MESH_MMS_HPP
#define MFEM_STOKES_MESH_MMS_HPP

#include "stokes_mms.hpp"

namespace mfem::stokes
{

/// NoSlip uses hole cutoffs; Smooth retains a nonzero analytic hole trace.
enum class MeshMMSCase
{
   NoSlip,
   Smooth
};

/// Return the CLI case name, or "unknown" for an invalid enum.
const char *MeshMMSCaseName(MeshMMSCase problem_case);

/** @brief Divergence-free MMS for the supplied 2x1[x1] five-hole meshes.

    A curl of a periodic scalar potential gives the velocity. The default
    NoSlip case uses radial cutoffs to make it identically zero within radius
    0.10 of each hole center. Smooth omits these cutoffs and has nonzero hole
    velocity. Both give zero velocity on the outer box. Pressure and
    viscosity are periodic with the box lengths. Variable viscosity is
    mu0*(1+0.25*product(cos(2*pi*x_i/L_i))). All spatial derivatives used by
    the force are analytic. The NoSlip potential is C4 at the cutoff endpoints;
    the Smooth potential is analytic. Filled inclusions use prescribed MMS interface
    constraints, not as an unconstrained material-transmission benchmark. */
class MeshManufacturedSolution
{
   int dimension;
   real_t viscosity_scale;
   bool variable_viscosity;
   MeshMMSCase problem_case;

public:
   FunctionCoefficient viscosity;
   VectorFunctionCoefficient viscosity_gradient, velocity, force;
   FunctionCoefficient pressure;

   MeshManufacturedSolution(int dimension, real_t viscosity_scale = 1.0,
                            bool variable_viscosity = false,
                            MeshMMSCase problem_case = MeshMMSCase::NoSlip);
   /// Case selected at construction; coefficient adapters use the same case.
   MeshMMSCase GetCase() const { return problem_case; }
   MeshManufacturedSolution(const MeshManufacturedSolution &) = delete;
   MeshManufacturedSolution &operator=(const MeshManufacturedSolution &) = delete;
   void Velocity(const Vector &point, Vector &value) const;
   void Gradient(const Vector &point, DenseMatrix &value) const;
   void Force(const Vector &point, Vector &value) const;
   real_t Viscosity(const Vector &point) const;
   void ViscosityGradient(const Vector &point, Vector &value) const;
   real_t Pressure(const Vector &point) const;
};

/// Collective error norms; pressure error is measured modulo constants.
StokesErrors ComputeMeshMMSErrors(const ParGridFunction &velocity,
                                 const ParGridFunction &pressure,
                                 MeshManufacturedSolution &exact,
                                 int integration_order);

/** @brief Implementation of the dimension-independent parallel MMS executable.

    MPI and HYPRE must be initialized by the caller. The dimension is inferred
    from the required --mesh; only full-dimensional 2D/3D meshes are accepted.
    The mesh is uniformly refined --refine-serial (-rs) times before MPI
    partitioning.
    Then levels 0 through --refine-parallel (-rp) are solved, with parallel
    uniform refinement between levels. Level 0 is the serially refined mesh;
    both refinement counts default to zero and must be nonnegative.
    A fresh solver/space pair is made at every parallel level.
    --mms-case smooth projects the analytic nonzero hole trace into the H1
    boundary space with a zero-net-flux constraint; outer-wall velocity stays
    zero. This MMS-only projection does not weaken solver compatibility checks.
    Reports separate analytic boundary error from prescribed-DOF enforcement.
    Every solve reports the outer iteration count; MUMPS
    reports zero. A failed solve produces a report and a nonzero exit status.
    Uniform polynomial H1/L2 high-order geometry is preserved under refinement.
    Reports include coordinate degree and resolved assembly quadrature.
    Optional --paraview writes velocity/pressure at every level into one PVD
    series under --output. Series time is the refinement level, not physical
    time. Output is disabled by default and is excluded from solve timing. */
int RunMeshMMS(int argc, char **argv);

}

#endif
