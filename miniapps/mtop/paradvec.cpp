// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
#include <fstream>
#include <iostream>

#include "mtop_integrators.hpp"

int main(int argc, char *argv[])
{
   // Initialize MPI.
   int nprocs, myrank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

   // Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order = 1;
   bool static_cond = false;
   int ser_ref_levels = 1;
   int par_ref_levels = 1;
   double newton_rel_tol = 1e-7;
   double newton_abs_tol = 1e-12;
   int newton_iter = 10;
   int print_level = 1;
   bool visualization = false;

   mfem::OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels,
                    "-rs",
                    "--refine-serial",
                    "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels,
                    "-rp",
                    "--refine-parallel",
                    "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&visualization,
                     "-vis",
                     "--visualization",
                     "-no-vis",
                     "--no-visualization",
                     "Enable or disable GLVis visualization.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&newton_rel_tol,
                      "-rel",
                      "--relative-tolerance",
                      "Relative tolerance for the Newton solve.");
    args.AddOption(&newton_abs_tol,
                      "-abs",
                      "--absolute-tolerance",
                      "Absolute tolerance for the Newton solve.");
    args.AddOption(&newton_iter,
                      "-it",
                      "--newton-iterations",
                      "Maximum iterations for the Newton solve.");
   args.Parse();

   if (!args.Good())
   {
      if (myrank == 0)
      {
         args.PrintUsage(std::cout);
      }
      MPI_Finalize();
      return 1;
   }

   if (myrank == 0)
   {
      args.PrintOptions(std::cout);
   }

   //    Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   mfem::Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   //    Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   //    Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }

   // Define the Diffusion tensor
   mfem::Vector vc(6);
   vc[0]=1.0; vc[1]=1.0; vc[2]=1.0; //diagonal terms for the diffusion tensor
   vc[3]=0.0; vc[4]=0.0; vc[5]=0.0; //off-diagon terms
   mfem::VectorCoefficient* diffco=new mfem::VectorConstantCoefficient(vc);
   // Define the Heat source
   mfem::ConstantCoefficient* loadco=new mfem::ConstantCoefficient(1.0);
   // Define rection coefficient
   mfem::ConstantCoefficient* mu=new mfem::ConstantCoefficient(0.0);
   // Define the q-function
   mfem::QAdvectionDiffusionLSFEM* qfun=new mfem::QAdvectionDiffusionLSFEM(*diffco,*mu,*loadco);

   // Define FE-collections
   mfem::H1_FECollection sfec(order,dim);
   mfem::RT_FECollection qfec(order,dim);
   mfem::ParFiniteElementSpace* sfes=new mfem::ParFiniteElementSpace(&pmesh,&sfec,1);
   mfem::ParFiniteElementSpace* qfes=new mfem::ParFiniteElementSpace(&pmesh,&qfec,1);

   // Define FE collection and space for the density field
   mfem::L2_FECollection pfec(order, dim);
   mfem::ParFiniteElementSpace* pfes=new mfem::ParFiniteElementSpace(&pmesh,&pfec,1);
   // FE space for the velocity field
   mfem::ParFiniteElementSpace* vfes=new mfem::ParFiniteElementSpace(&pmesh,&sfec,dim);

   // Define the arrays for the nonlinear form
   mfem::Array<mfem::ParFiniteElementSpace*> asfes;
   mfem::Array<mfem::ParFiniteElementSpace*> apfes;

   asfes.Append(sfes);
   asfes.Append(qfes);
   apfes.Append(pfes);
   apfes.Append(vfes);

   // Define parametric block nonlinear form using single scalar H1 field
   // and L2 scalar density field
   mfem::ParParametricBNLForm* nf=new mfem::ParParametricBNLForm(asfes,apfes);
   // add the parametric integrator
   nf->AddDomainIntegrator(new mfem::ParametricAdvecDiffusLSFEM(*qfun));

   // Define true block vectors  for state, adjoint, resudual
   mfem::BlockVector solbv; solbv.Update(nf->GetBlockTrueOffsets());    solbv=0.0;
   mfem::BlockVector adjbv; adjbv.Update(nf->GetBlockTrueOffsets());    adjbv=0.0;
   mfem::BlockVector resbv; resbv.Update(nf->GetBlockTrueOffsets());    resbv=0.0;

   // Define true block vectors for parametric field and gradients
   mfem::BlockVector prmbv; prmbv.Update(nf->PrmGetBlockTrueOffsets()); prmbv=0.0;
   mfem::BlockVector grdbv; grdbv.Update(nf->PrmGetBlockTrueOffsets()); grdbv=0.0;

   //set the BC for the physics
   mfem::Array<mfem::Array<int> *> ess_bdr;
   mfem::Array<mfem::Vector*>      ess_rhs;
   ess_bdr.Append(new mfem::Array<int>(pmesh.bdr_attributes.Max()));
   ess_bdr.Append(new mfem::Array<int>(pmesh.bdr_attributes.Max()));
   ess_rhs.Append(nullptr);
   ess_rhs.Append(nullptr);
   (*ess_bdr[0]) = 1;
   (*ess_bdr[1]) = 0;
   nf->SetEssentialBC(ess_bdr,ess_rhs);
   delete ess_bdr[0];
   delete ess_bdr[1];

   // set the density field to 0.5
   prmbv.GetBlock(0)=0.5;
   // set the velocity to [1,1,1] in 3D and [1,1] in 2D
   prmbv.GetBlock(1)=1.0;
   // set the parametric fields in the BNLForm
   nf->SetPrmFields(prmbv);

   double energy= nf->GetEnergy(solbv);
   // compute the stiffness/tangent matrix for density prmbv=0.5
   //mfem::BlockOperator& A=nf->GetGradient(solbv);

   delete nf;

   delete qfun;

   delete vfes;
   delete pfes;
   delete qfes;
   delete sfes;
   delete mu;
   delete loadco;
   delete diffco;

   MPI_Finalize();
   return 0;

}
