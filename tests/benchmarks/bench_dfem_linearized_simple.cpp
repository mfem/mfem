// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"

#ifdef MFEM_USE_MPI

#include "fem/dfem/doperator.hpp"
#include "fem/dfem/backends/global_qf/prelude.hpp"
#include "fem/dfem/backends/local_qf/prelude.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

using namespace mfem;
using namespace mfem::future;

#ifdef MFEM_USE_ENZYME
using dscalar_t = real_t;
#else
using dscalar_t = dual<real_t, real_t>;
#endif

template <int DIM>
struct DiffusionLocalMode
{
   using dvecd_t = tensor<dscalar_t, DIM>;
   using matd_t = tensor<real_t, DIM, DIM>;

   MFEM_HOST_DEVICE inline auto operator()(const dvecd_t &dudxi,
                                           const matd_t &J,
                                           const real_t &w,
                                           dvecd_t &dvdxi) const
   {
      const auto invJ = inv(J);
      const auto invJt = transpose(invJ);
      dvdxi = (dudxi * invJ) * invJt * det(J) * w;
   }
};

template <int DIM>
struct DiffusionGlobalMode
{
   void operator()(tensor_array<const real_t, DIM> &dudxi,
                   tensor_array<const real_t, DIM, DIM> &J,
                   tensor_array<const real_t> &w,
                   tensor_array<real_t, DIM> &dvdxi) const
   {
      mfem::forall<UseEnzyme>(J.size(), [=] MFEM_HOST_DEVICE (int q)
      {
         const auto invJ = inv(J(q));
         const auto invJt = transpose(invJ);
         dvdxi(q) = (dudxi(q) * invJ) * invJt * det(J(q)) * w(q);
      });
   }
};

template <int DIM, typename QFBackend> struct DiffusionQF;
template <int DIM> struct DiffusionQF<DIM, LocalQFBackend>
{
   using type = DiffusionLocalMode<DIM>;
};
template <int DIM> struct DiffusionQF<DIM, GlobalQFBackend>
{
   using type = DiffusionGlobalMode<DIM>;
};

struct Timings
{
   double forward = 0.0;
   double derivative = 0.0;
   real_t error = 0.0;
};

template <typename F>
double TimeIt(const int iterations, MPI_Comm comm, F &&f)
{
   MPI_Barrier(comm);
   MFEM_DEVICE_SYNC;

   StopWatch timer;
   timer.Start();
   for (int i = 0; i < iterations; i++)
   {
      f();
   }
   MFEM_DEVICE_SYNC;
   timer.Stop();

   const double local_time = timer.RealTime();
   double global_time = 0.0;
   MPI_Allreduce(&local_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, comm);
   return global_time / iterations;
}

template <int DIM>
Mesh MakeSimpleTensorMesh(int n)
{
   if constexpr (DIM == 2)
   {
      return Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL,
                                   true, 1.0, 1.0);
   }
   else
   {
      return Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON,
                                   1.0, 1.0, 1.0);
   }
}

template <int DIM>
int MeshSizeFromTargetDofs(const int order, const int target_dofs)
{
   const real_t dofs_1d = std::pow(static_cast<real_t>(target_dofs),
                                   1.0 / DIM);
   return std::max(1, static_cast<int>(std::ceil((dofs_1d - 1.0) / order)));
}

template <int DIM>
auto GetGlobalTrueVSize(const int order, const int mesh_n)
{
   Mesh smesh = MakeSimpleTensorMesh<DIM>(mesh_n);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   smesh.Clear();

   pmesh.EnsureNodes();
   const int p = std::max(order, pmesh.GetNodalFESpace()->GetMaxElementOrder());
   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace pfes(&pmesh, &fec);
   return pfes.GlobalTrueVSize();
}

template <int DIM, typename QFBackend>
Timings RunBackendCase(const int order, const int mesh_n, const int warmup,
                       const int iterations)
{
   Mesh smesh = MakeSimpleTensorMesh<DIM>(mesh_n);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   smesh.Clear();

   pmesh.EnsureNodes();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   ParFiniteElementSpace *mfes = nodes->ParFESpace();

   const int p = std::max(order, pmesh.GetNodalFESpace()->GetMaxElementOrder());
   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace pfes(&pmesh, &fec);
   const IntegrationRule *ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(),
                                             2 * p);

   Array<int> all_domain_attr;
   if (pmesh.attributes.Size() > 0)
   {
      all_domain_attr.SetSize(pmesh.attributes.Max());
      all_domain_attr = 1;
   }

   static constexpr int U = 0, Coords = 1;
   const auto in_fds = std::vector
   {
      FieldDescriptor{ U, &pfes },
      FieldDescriptor{ Coords, mfes }
   };
   const auto out_fds = std::vector{ FieldDescriptor{ U, &pfes } };

   DifferentiableOperator dop_mf(in_fds, out_fds, pmesh);
   typename DiffusionQF<DIM, QFBackend>::type mf_apply_qf;
   dop_mf.AddDomainIntegrator<QFBackend>(
      mf_apply_qf,
      Inputs<Gradient<U>, Gradient<Coords>, Weight> {},
      Outputs<Gradient<U>> {},
      *ir, all_domain_attr,
      Derivatives<U> {});

   Vector xtvec(pfes.GetTrueVSize()), ztvec(pfes.GetTrueVSize());
   xtvec.Randomize(567);

   Vector nodestv;
   nodes->GetTrueDofs(nodestv);

   // Move the fixed input state to device before warmup/timing so repeated
   // applies do not pay host-to-device copies through aliased L-vectors.
   xtvec.Read();
   nodestv.Read();

   MultiVector X{xtvec, nodestv};
   MultiVector Z{ztvec};
   auto ddop = dop_mf.GetDerivative(U, X, false);

   Vector dztvec(ztvec.Size());
   MultiVector DZ{dztvec};

   for (int i = 0; i < warmup; i++)
   {
      dop_mf.Mult(X, Z);
      ddop->Mult(X[0], DZ);
   }
   MFEM_DEVICE_SYNC;

   // One correctness check: this diffusion action is linear in U, so the
   // forward action and derivative action should agree for the same direction.
   dop_mf.Mult(X, Z);
   ddop->Mult(X[0], DZ);
   Vector diff(ztvec);
   diff -= dztvec;
   const real_t local_error = diff.Normlinf();
   real_t global_error = 0.0;
   MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_MAX,
                 pmesh.GetComm());

   Timings timings;
   timings.error = global_error;
   timings.forward = TimeIt(iterations, pmesh.GetComm(), [&]()
   {
      dop_mf.Mult(X, Z);
   });

   timings.derivative = TimeIt(iterations, pmesh.GetComm(), [&]()
   {
      ddop->Mult(X[0], DZ);
   });

   return timings;
}

template <int DIM>
void RunCase(const int order, const int mesh_n, const int warmup,
             const int iterations)
{
   const char *mesh_name = DIM == 2 ? "quad" : "hex";
   const auto tdofs = GetGlobalTrueVSize<DIM>(order, mesh_n);
   if (Mpi::WorldRank() == 0)
   {
      mfem::out << mesh_name << " #dofs=" << tdofs << std::endl;
   }

   const Timings global = RunBackendCase<DIM, GlobalQFBackend>(order, mesh_n,
                                                               warmup,
                                                               iterations);
   const Timings local = RunBackendCase<DIM, LocalQFBackend>(order, mesh_n,
                                                             warmup,
                                                             iterations);

   const real_t error = std::max(global.error, local.error);
   if (Mpi::WorldRank() == 0)
   {
      mfem::out << mesh_name << " Scalar Action Linearized timings (seconds):\n"
                << "  forward global=" << global.forward
                << ", derivative global=" << global.derivative
                << ", overhead=" << global.derivative / global.forward << "x\n"
                << "  forward local =" << local.forward
                << ", derivative local =" << local.derivative
                << ", overhead=" << local.derivative / local.forward << "x\n"
                << "  forward local/global=" << local.forward / global.forward
                << "x\n"
                << "  derivative local/global="
                << local.derivative / global.derivative << "x\n"
                << "  linf_error=" << error << "\n";
   }
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   int order = 3;
   int quad_mesh_n = 32;
   int hex_mesh_n = 8;
   int target_dofs = 0;
   int warmup = 3;
   int iterations = 25;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Finite element order.");
   args.AddOption(&quad_mesh_n, "-nq", "--quad-mesh-size",
                  "Number of quad elements per direction.");
   args.AddOption(&hex_mesh_n, "-nh", "--hex-mesh-size",
                  "Number of hex elements per direction.");
   args.AddOption(&target_dofs, "-nd", "--num-dofs",
                  "Target number of dofs. If positive, overrides -nq/-nh and "
                  "computes element counts from the order.");
   args.AddOption(&warmup, "-w", "--warmup", "Number of warmup iterations.");
   args.AddOption(&iterations, "-i", "--iterations",
                  "Number of timed iterations.");
   args.AddOption(&device_config, "-d", "--device",
                  "MFEM device configuration string.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::WorldRank() == 0) { args.PrintUsage(mfem::out); }
      return 1;
   }

   Device device(device_config);
   if (Mpi::WorldRank() == 0)
   {
      args.PrintOptions(mfem::out);
      device.Print(mfem::out);
      mfem::out << std::scientific << std::setprecision(6)
                << "\nForward DifferentiableOperator::Mult vs "
                << "GetDerivative(...)->Mult\n";
   }

   if (target_dofs > 0)
   {
      quad_mesh_n = MeshSizeFromTargetDofs<2>(order, target_dofs);
      hex_mesh_n = MeshSizeFromTargetDofs<3>(order, target_dofs);
      if (Mpi::WorldRank() == 0)
      {
         mfem::out << "target #dofs=" << target_dofs
                   << " -> quad n=" << quad_mesh_n
                   << ", hex n=" << hex_mesh_n << std::endl;
      }
   }

   RunCase<2>(order, quad_mesh_n, warmup, iterations);
   RunCase<3>(order, hex_mesh_n, warmup, iterations);

   return 0;
}

#else

int main(int, char *[])
{
   mfem::out << "This benchmark requires MFEM_USE_MPI=YES.\n";
   return MFEM_SKIP_RETURN_VALUE;
}

#endif // MFEM_USE_MPI
