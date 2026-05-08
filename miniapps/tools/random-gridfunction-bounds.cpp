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
//
//    ---------------------------------------------------------------------
//     Compute bounds of a random grid function on a generated tensor mesh
//    ---------------------------------------------------------------------
//
// This miniapp generates a 1D segment mesh or 2D quad mesh, builds a random
// discontinuous grid function, computes element-wise piecewise linear bounds,
// and visualizes the input field together with the lower and upper bounds.
//
// Compile with: make random-gridfunction-bounds
//
// Sample runs:
//   mpirun -np 4 random-gridfunction-bounds
//   mpirun -np 4 random-gridfunction-bounds -nx 64 -o 6 -ref 3 -d hip

#include "mfem.hpp"

#include <algorithm>
#include <type_traits>

using namespace mfem;
using namespace std;

void VisualizeField(ParMesh &pmesh, ParGridFunction &input,
                    char *title, int pos_x, int pos_y);

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   int dim = 2;
   int nx = 16;
   int order = 4;
   int num_comp = 2;
   int ref = 2;
   int niter = 1000;
   int seed = 12345;
   bool kernel_only = true;
   bool visualization = false;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-dim", "--dimension",
                  "Dimension of the generated tensor-product mesh (1 or 2).");
   args.AddOption(&nx, "-nx", "--num-elements",
                  "Number of elements in each mesh direction.");
   args.AddOption(&order, "-o", "--order",
                  "Polynomial degree of the random discontinuous field.");
   args.AddOption(&num_comp, "-nc", "--num-components",
                  "Number of vector components in the ParFiniteElementSpace.");
   args.AddOption(&ref, "-ref", "--piecewise-linear-ref-factor",
                  "Scaling factor for the resolution of the piecewise linear "
                  "bounds. If less than 2, the resolution is picked "
                  "automatically.");
   args.AddOption(&niter, "-ni", "--num-iters",
                  "Number of times to evaluate the bounds.");
   args.AddOption(&seed, "-rs", "--random-seed",
                  "Random seed used to initialize the field.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&kernel_only, "-ko", "--kernel-only",
                  "-no-ko", "--no-kernel-only",
                  "Run only PLBound::GetElementBoundsKernel on a prebuilt "
                  "element E-vector.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();

   MFEM_VERIFY(dim == 1 || dim == 2, "dim must be 1 or 2.");
   MFEM_VERIFY(nx > 0, "nx must be positive.");
   MFEM_VERIFY(order >= 0, "order must be non-negative.");
   MFEM_VERIFY(num_comp > 0, "num_comp must be positive.");
   MFEM_VERIFY(niter > 0, "niter must be positive.");

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   Mesh mesh = (dim == 1) ?
               Mesh::MakeCartesian1D(nx, 1.0) :
               Mesh::MakeCartesian2D(nx, nx, Element::QUADRILATERAL, true,
                                     1.0, 1.0);
   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   const int mesh_dim = pmesh.Dimension();
   L2_FECollection fec(order, mesh_dim, BasisType::GaussLobatto);
   ParFiniteElementSpace fes(&pmesh, &fec, num_comp, Ordering::byNODES);
   ParGridFunction input(&fes);
   input.Randomize(seed + Mpi::WorldRank());
   input.UseDevice(true);

   L2_FECollection fec_pc(0, mesh_dim);
   ParFiniteElementSpace fes_pc(&pmesh, &fec_pc, num_comp, Ordering::byNODES);
   ParGridFunction lowerb(&fes_pc), upperb(&fes_pc);
   Vector lower_vec, upper_vec;

   PLBound plb(&fes, ref*(fes.GetMaxElementOrder() + 1));
   if (kernel_only)
   {
      const FiniteElement &fe = *fes.GetTypicalFE();
      const int rdim = fe.GetDim();
      const int nd = fe.GetDof();
      const int fes_dim = fes.GetVDim();
      Vector e_vec(nd*fes_dim*fes.GetNE(), Device::GetDeviceMemoryType());
      e_vec.UseDevice(true);

      const ElementRestrictionOperator *elem_restr =
         fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
      MFEM_VERIFY(elem_restr != nullptr,
                  "Element restriction is required for kernel-only mode.");
      elem_restr->Mult(input, e_vec);

      for (int i = 0; i < niter; i++)
      {
         plb.GetElementBoundsKernel(rdim, fes_dim, e_vec, lower_vec, upper_vec);
      }
   }
   else
   {
      for (int i = 0; i < niter; i++)
      {
         input.GetElementBounds(plb, lower_vec, upper_vec);
      }
   }

   const real_t *lower_data = lower_vec.HostRead();
   const real_t *upper_data = upper_vec.HostRead();

   // Build a host reference from the lexicographic E-vector and the scalar
   // PLBound::GetNDBounds path to avoid re-entering the device dispatch.
   const bool use_dev = input.UseDevice();
   PLBound plb_host(&fes, ref*(fes.GetMaxElementOrder() + 1));
   Vector lower_ref, upper_ref;
   const FiniteElement &fe = *fes.GetTypicalFE();
   const int rdim = fe.GetDim();
   const int nd = fe.GetDof();
   const int nel = fes.GetNE();
   const int fes_dim = fes.GetVDim();
   Vector e_vec_ref(nd*fes_dim*nel);
   lower_ref.SetSize(nel*fes_dim);
   upper_ref.SetSize(nel*fes_dim);
   const ElementRestrictionOperator *elem_restr =
      fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   MFEM_VERIFY(elem_restr != nullptr,
               "Element restriction is required for host reference.");
   input.UseDevice(false);
   input.HostRead();
   elem_restr->Mult(input, e_vec_ref);
   input.UseDevice(use_dev);
   const real_t *e_ref_data = e_vec_ref.HostRead();

   for (int d = 0; d < fes_dim; d++)
   {
      for (int e = 0; e < nel; e++)
      {
         Vector coeff(nd);
         for (int i = 0; i < nd; i++)
         {
            coeff(i) = e_ref_data[i + nd*(d + fes_dim*e)];
         }
         Vector lower_c, upper_c;
         plb_host.GetNDBounds(rdim, coeff, lower_c, upper_c);
         lower_ref(e + d*nel) = lower_c.Min();
         upper_ref(e + d*nel) = upper_c.Max();
      }
   }
   const real_t *lower_ref_data = lower_ref.HostRead();
   const real_t *upper_ref_data = upper_ref.HostRead();

   MFEM_VERIFY(lower_vec.Size() == lower_ref.Size() &&
               upper_vec.Size() == upper_ref.Size(),
               "Reference element-bound vectors have inconsistent sizes.");

   real_t lower_diff = 0.0;
   real_t upper_diff = 0.0;
   for (int i = 0; i < lower_vec.Size(); i++)
   {
      lower_diff = std::max(lower_diff,
                            std::abs(lower_data[i] - lower_ref_data[i]));
   }
   for (int i = 0; i < upper_vec.Size(); i++)
   {
      upper_diff = std::max(upper_diff,
                            std::abs(upper_data[i] - upper_ref_data[i]));
   }
   MPI_Allreduce(MPI_IN_PLACE, &lower_diff, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MAX, pmesh.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &upper_diff, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MAX, pmesh.GetComm());

   const real_t verify_tol = std::is_same<real_t, float>::value ?
                             real_t(1.0e-5) : real_t(1.0e-12);
   MFEM_VERIFY(lower_diff <= verify_tol && upper_diff <= verify_tol,
               "Device element bounds do not match host reference.");

   lowerb = lower_vec;
   upperb = upper_vec;

   real_t lower_min = lowerb.Min();
   real_t upper_max = upperb.Max();
   MPI_Allreduce(MPI_IN_PLACE, &lower_min, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MIN, pmesh.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &upper_max, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MAX, pmesh.GetComm());

   if (Mpi::Root())
   {
      cout << "dim: " << mesh_dim << '\n'
           << "nx: " << nx << '\n'
           << "order: " << order << '\n'
           << "num components: " << num_comp << '\n'
           << "PL bound control-point factor: " << ref << '\n'
           << "iterations: " << niter << '\n'
           << "kernel-only mode: " << (kernel_only ? "yes" : "no") << '\n'
           << "host/device lower max diff: " << lower_diff << '\n'
           << "host/device upper max diff: " << upper_diff << '\n'
           << "global lower bound minimum: " << lower_min << '\n'
           << "global upper bound maximum: " << upper_max << endl;
   }

   if (visualization)
   {
      char title1[] = "Random input gridfunction";
      char title2[] = "Element-wise lower bound";
      char title3[] = "Element-wise upper bound";
      VisualizeField(pmesh, input, title1, 0, 0);
      VisualizeField(pmesh, lowerb, title2, 450, 0);
      VisualizeField(pmesh, upperb, title3, 900, 0);
   }

   return 0;
}

void VisualizeField(ParMesh &pmesh, ParGridFunction &input,
                    char *title, int pos_x, int pos_y)
{
   socketstream sock;
   if (pmesh.GetMyRank() == 0)
   {
      sock.open("localhost", 19916);
      sock << "solution\n";
   }
   pmesh.PrintAsOne(sock);
   input.SaveAsOne(sock);
   if (pmesh.GetMyRank() == 0)
   {
      sock << "window_title '" << title << "'\n"
           << "window_geometry "
           << pos_x << " " << pos_y << " " << 400 << " " << 400 << "\n"
           << "keys jRmclApppppppppppp//]]]]]]]]" << endl;
   }
}
