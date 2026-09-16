// Regression for prescribed passive outputs in the Helmholtz design filter.
//
// The configured forward map is affine,
//
//    G(x) = C F x + (I-C) c,
//
// so this test checks both exact prescribed values and the Jacobian-transpose
// identity q^T G'(x) p = p^T F^T C q using a centered finite difference.

#include "mfem.hpp"
#include "../../pde_filter.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

using namespace mfem;

namespace
{

real_t GlobalInnerProduct(MPI_Comm comm, const Vector &left,
                          const Vector &right)
{
   const real_t local_value = left * right;
   real_t global_value = 0.0;
   MPI_Allreduce(&local_value, &global_value, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, comm);
   return global_value;
}

real_t GlobalMaximum(MPI_Comm comm, real_t local_value)
{
   real_t global_value = 0.0;
   MPI_Allreduce(&local_value, &global_value, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_MAX, comm);
   return global_value;
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   const MPI_Comm comm = MPI_COMM_WORLD;
   Device device("cpu");

   {
      int ranks = 1;
      MPI_Comm_size(comm, &ranks);
      Mesh serial_mesh = Mesh::MakeCartesian2D(
         2 * ranks, 2, Element::QUADRILATERAL,
         /*generate_edges=*/true, 1.0, 1.0);
      ParMesh mesh(comm, serial_mesh);

      H1_FECollection filter_collection(/*order=*/1, /*dimension=*/2);
      L2_FECollection control_collection(
         /*order=*/0, /*dimension=*/2, BasisType::GaussLobatto);
      ParFiniteElementSpace filter_fes(&mesh, &filter_collection);
      ParFiniteElementSpace control_fes(&mesh, &control_collection);

      toopt::PDEFilterOptions options;
      options.filter_radius = 0.12;
      options.solver_rtol = 1e-13;
      options.solver_atol = 1e-15;
      options.solver_maxiter = 500;
      toopt::PDEFilter filter(filter_fes, control_fes, options);

      // Prescribe the top filtered boundary as a compact proxy for a passive
      // collection of H1 true DOFs.  The production driver uses the same API
      // for every H1 DOF touching a passive element.
      Array<int> top_marker(mesh.bdr_attributes.Max());
      top_marker = 0;
      MFEM_VERIFY(top_marker.Size() >= 3,
                  "Cartesian test mesh has unexpected boundary attributes.");
      top_marker[2] = 1; // Cartesian boundary attribute 3 is the top.
      Array<int> prescribed_tdofs;
      filter_fes.GetEssentialTrueDofs(top_marker, prescribed_tdofs);
      constexpr real_t prescribed_value = 1.0;
      filter.SetPrescribedOutputDofs(prescribed_tdofs, prescribed_value);
      filter.Assemble();

      long long local_prescribed = prescribed_tdofs.Size();
      long long global_prescribed = 0;
      MPI_Allreduce(&local_prescribed, &global_prescribed, 1,
                    MPI_LONG_LONG, MPI_SUM, comm);
      MFEM_VERIFY(global_prescribed > 0,
                  "Prescribed-output regression found no filtered DOFs.");

      Vector base(control_fes.GetTrueVSize());
      Vector direction(control_fes.GetTrueVSize());
      for (int i = 0; i < base.Size(); i++)
      {
         base[i] = 0.45 + 0.08 * std::sin(real_t(i + 1));
         direction[i] = 0.3 * std::cos(real_t(2 * i + 1));
      }

      Vector dual(filter_fes.GetTrueVSize());
      for (int i = 0; i < dual.Size(); i++)
      {
         dual[i] = 0.2 + 0.1 * std::sin(real_t(3 * i + 2));
      }

      Vector filtered;
      filter.Mult(base, filtered);
      real_t local_prescribed_error = 0.0;
      for (int i = 0; i < prescribed_tdofs.Size(); i++)
      {
         local_prescribed_error = std::max(
            local_prescribed_error,
            std::abs(filtered[prescribed_tdofs[i]] - prescribed_value));
      }
      const real_t prescribed_error =
         GlobalMaximum(comm, local_prescribed_error);

      constexpr real_t epsilon = 1e-5;
      Vector plus(base), minus(base);
      plus.Add(epsilon, direction);
      minus.Add(-epsilon, direction);
      Vector filtered_plus, filtered_minus;
      filter.Mult(plus, filtered_plus);
      filter.Mult(minus, filtered_minus);
      Vector finite_difference(filtered_plus);
      finite_difference -= filtered_minus;
      finite_difference /= 2.0 * epsilon;

      real_t local_prescribed_derivative = 0.0;
      for (int i = 0; i < prescribed_tdofs.Size(); i++)
      {
         local_prescribed_derivative = std::max(
            local_prescribed_derivative,
            std::abs(finite_difference[prescribed_tdofs[i]]));
      }
      const real_t prescribed_derivative =
         GlobalMaximum(comm, local_prescribed_derivative);

      Vector transpose_action;
      filter.MultTranspose(dual, transpose_action);
      const real_t forward_action =
         GlobalInnerProduct(comm, dual, finite_difference);
      const real_t transpose_scalar =
         GlobalInnerProduct(comm, direction, transpose_action);
      const real_t transpose_relative_error =
         std::abs(forward_action - transpose_scalar) /
         std::max({std::abs(forward_action),
                   std::abs(transpose_scalar), real_t(1e-14)});

      MFEM_VERIFY(prescribed_error <= 1e-15,
                  "Prescribed filtered value is not exact.");
      MFEM_VERIFY(prescribed_derivative <= 1e-12,
                  "Prescribed filtered DOF has a nonzero design derivative.");
      MFEM_VERIFY(transpose_relative_error <= 2e-6,
                  "Prescribed-output filter failed its Jacobian-transpose "
                  "finite-difference check.");

      if (Mpi::Root())
      {
         std::cout << std::scientific << std::setprecision(12)
                   << "Prescribed output max error: " << prescribed_error
                   << '\n'
                   << "Prescribed output max derivative: "
                   << prescribed_derivative << '\n'
                   << "Jacobian-transpose relative error: "
                   << transpose_relative_error << '\n'
                   << "PDE filter prescribed-output regression passed.\n";
      }
   }

   return 0;
}
