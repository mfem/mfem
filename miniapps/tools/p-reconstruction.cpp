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
#include <limits>
#include <string>
#include <unordered_map>

using namespace mfem;

using profile_t = std::function<real_t(const Vector&,const Vector&)>;

void L2Reconstruction(const GridFunction& src, GridFunction& dst);
std::unordered_map<std::string, profile_t> GetFieldProfiles();

int main(int argc, char* argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   // Default command-line options
   std::string reconstruction_method = "LOR_reconstruction";
   int refinement_levels = 0;
   int order_lo = 0;
   int order_ho = 3;
   int order_im = 3; // intermediate order, only used for LOR reconstruction method
   int lref = order_im+1;

   std::string field_profile = "plane";
   real_t field_kx = 2.0;
   real_t field_ky = 4.0;

   bool visualization = true;
   int visport = 19916;

   // example field profiles
   std::unordered_map<std::string, profile_t> field_profiles = GetFieldProfiles();
   // create CLI help string for profiles
   std::string field_profiles_help = "Profile of field to be reconstructed:";
   for (const auto& [name, _] : field_profiles)
   {
      field_profiles_help += "\n\t" + name;
   }

   // Parse options
   OptionsParser args(argc, argv);
   args.AddOption(&reconstruction_method, "-m", "--method",
                  "Reconstruction method: \"element_average_reconstruction\" or \"LOR_reconstruction\".");
   args.AddOption(&refinement_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order_ho, "-ho", "--order_ho",
                  "Finite element order (polynomial degree) for high-order space.");
   args.AddOption(&field_profile, "-f", "--field-profile",
                  field_profiles_help.c_str());
   args.AddOption(&field_kx, "-fx", "--field-kx",
                  "Value of kx in field profile");
   args.AddOption(&field_ky, "-fy", "--field-ky",
                  "Value of ky in field profile.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable GLVis visualization.");
   args.AddOption(&visport, "-visp", "--visualization-port",
                  "Use custom port number for GLVis.");
   args.ParseCheck();

   MFEM_VERIFY((reconstruction_method=="LOR_reconstruction"),
               "Currently only LOR_reconstruction is supported.");

   // define u(x,y) to be represented
   profile_t u_function = field_profiles.at(field_profile);
   const Vector k({field_kx, field_ky});
   std::function<real_t(const Vector&)> u_function_wrapper =
   [&](const Vector &x) { return u_function(x, k); };
   FunctionCoefficient u_function_exact(u_function_wrapper);

   // create simple 2D mesh
   Mesh mesh;
   Mesh mesh_im;

   const int num_x = 8;
   const int num_y = 8;

   order_im = order_ho;
   lref = order_im + 1;
   MFEM_VERIFY((num_x % lref) == 0 && (num_y % lref) == 0,
               "For LOR_reconstruction, lref = order_im (=order_ho) + 1 must divide both num_x and num_y.");
   int num_x_im = num_x / lref;
   int num_y_im = num_y / lref;

   Mesh serial_mesh_im = Mesh::MakeCartesian2D(num_x_im, num_y_im,
                                               Element::QUADRILATERAL);
   for (int i = 0; i < refinement_levels; i++)
   {
      serial_mesh_im.UniformRefinement();
   }
   ParMesh pmesh_im(MPI_COMM_WORLD, serial_mesh_im);
   serial_mesh_im.Clear();
   ParMesh pmesh = ParMesh::MakeRefined(pmesh_im, lref, BasisType::ClosedUniform);
   int dim = pmesh.Dimension();

   L2_FECollection fec_lo(order_lo, dim);
   L2_FECollection fec_hi(order_ho, dim);
   H1_FECollection fec_im(order_im, dim);

   ParFiniteElementSpace pfespace_lo(&pmesh, &fec_lo);
   ParFiniteElementSpace pfespace_hi(&pmesh, &fec_hi);
   ParFiniteElementSpace pfespace_im(&pmesh_im, &fec_im);

   ParGridFunction u_lo(&pfespace_lo);
   ParGridFunction u_hi(&pfespace_hi);
   ParGridFunction u_im(&pfespace_im);

   ParBilinearForm M_lo(&pfespace_lo);
   M_lo.AddDomainIntegrator(new MassIntegrator);
   M_lo.Assemble();
   M_lo.Finalize();

   ParBilinearForm M_im(&pfespace_im);
   M_im.AddDomainIntegrator(new MassIntegrator);
   M_im.Assemble();
   M_im.Finalize();

   ParBilinearForm M_hi(&pfespace_hi);
   M_hi.AddDomainIntegrator(new MassIntegrator);
   M_hi.Assemble();
   M_hi.Finalize();

   // Set up the right-hand side vector for the exact solution
   ParLinearForm b_lo(&pfespace_lo);
   DomainLFIntegrator *lf_integ = new DomainLFIntegrator(u_function_exact);
   const IntegrationRule &ir_rhs = IntRules.Get(pfespace_lo.GetFE(
                                                   0)->GetGeomType(), order_ho + 1);
   lf_integ->SetIntRule(&ir_rhs);
   b_lo.AddDomainIntegrator(lf_integ);
   b_lo.Assemble();

   L2ProjectionGridTransfer gt1(pfespace_im, pfespace_lo);
   L2ProjectionGridTransfer gt2(pfespace_im, pfespace_hi);

   const Operator &P1 = gt1.BackwardOperator();   // Prolongation 1 (LO->IM)
   const Operator &P2 = gt2.ForwardOperator();    // Prolongation 2 (IM->HO)

   // STEP 1: L2 projection onto u_lo
   std::unique_ptr<HypreParMatrix> M_par_lo(M_lo.ParallelAssemble());
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetOperator(*M_par_lo);
   cg.SetRelTol(1e-16);
   cg.SetMaxIter(1000);
   cg.SetPrintLevel(0);
   u_lo = 0.0;
   cg.Mult(b_lo, u_lo); // Solve: M * u_lo = b_lo
   u_lo.SetTrueVector();
   u_lo.SetFromTrueVector();

   // STEP 2: Prolongation 1 (LO->IM)
   P1.Mult(u_lo, u_im); // u_im = P1 * u_lo

   // STEP 3: Prolongation 2 (IM->HO)
   P2.Mult(u_im, u_hi); // u_hi = P2 * u_im

   // Visualization
   if (visualization)
   {
      char vishost[] = "localhost";
      socketstream glvis_u_lo(vishost, visport);
      socketstream glvis_u_hi(vishost, visport);
      if (glvis_u_lo && glvis_u_hi)
      {
         glvis_u_lo.precision(8);
         glvis_u_lo << "parallel " << pmesh.GetNRanks()
                    << " " << pmesh.GetMyRank() << "\n"
                    << "solution\n" << pmesh << u_lo
                    << "window_title 'Low-order'\n" << std::flush;
         glvis_u_hi.precision(8);
         glvis_u_hi << "parallel " << pmesh.GetNRanks()
                    << " " << pmesh.GetMyRank() << "\n"
                    << "solution\n" << pmesh << u_hi
                    << "window_title 'High-order'\n" << std::flush;
      }
   }

   real_t error = u_hi.ComputeL2Error(u_function_exact);
   if (Mpi::Root())
   {
      mfem::out.precision(16);
      mfem::out << "|| u_h - u ||_{L^2} = \n" << error << std::endl;
   }

   return 0;
}

std::unordered_map<std::string, profile_t> GetFieldProfiles()
{
   std::unordered_map<std::string, profile_t> field_profiles;
   // plane profile
   field_profiles["plane"] =
      [](const Vector &x, const Vector &k)
   {
      return 1.0 + x*k;
   };
   // sinusoidal profile
   field_profiles["sinusoidal"] =
      [](const Vector &x, const Vector &k)
   {
      real_t result = 1.0;
      for (int i=0; i < x.Size(); i++) { result *= std::sin(2.0*M_PI*k(i)*x(i)); }
      return result;
   };
   // exponential-sinusoidal profile
   field_profiles["exponential"] =
      [](const Vector &x, const Vector &k)
   {
      real_t result = 1.0;
      for (int i=0; i < x.Size(); i++)
      {
         result *= std::exp(x.Norml2()) * std::sin(2.0*M_PI*k(i)*x(i));
      }
      return result;
   };
   return field_profiles;
}
