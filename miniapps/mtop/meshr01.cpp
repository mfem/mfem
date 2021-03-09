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

//  The following example computes the gradients of a specified objective
//  function with respect to parametric fields. The objective function is
//  having the following form f(u(\rho)) where u(\rho) is a solution of a
//  specific state problem (in the example that is the diffusion equation),
//  and \rho is a parametric field discretized by finite elements. The
//  parametric field (also called density in topology optimization) controls
//  the coefficients of the state equation. For the considered case, the
//  density controls the diffusion coefficient within the computational domain.
//  For more information, the users are referred to:
//
//  Hinze, M.; Pinnau, R.; Ulbrich, M. & Ulbrich, S.
//  Optimization with PDE Constraints
//  Springer Netherlands, 2009
//
//  Bendsøe, M. P. & Sigmund, O.
//  Topology Optimization - Theory, Methods and Applications
//  Springer Verlag, Berlin Heidelberg, 2003
//

//Gyroid function

double Gyroid(const mfem::Vector &xx)
{
    const double period = 8.0 * M_PI;
    double x=xx[0]*period;
    double y=xx[1]*period;
    double z=0.0;
    if(xx.Size()==3)
    {
       z=xx[2]*period;
    }
    double val=std::sin(x)*std::cos(y) +
           std::sin(y)*std::cos(z) +
           std::sin(z)*std::cos(x);

    z=z+0.5*M_PI;
    double va1=std::sin(x)*std::cos(y) +
           std::sin(y)*std::cos(z) +
           std::sin(z)*std::cos(x);

    double rez=0.0;
    if(fabs(val)<0.2)
    {
        rez=1.0;
    }

    if(fabs(va1)<0.2)
    {
        rez=1.0;
    }

    return rez;
}

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
   int print_level = 0;
   double eta=0.5;
   double beta=4.0;
   double rho_min=1e-3;
   double filter_r=0.01;
   int filter_o=2;
   bool visualization = false;
   int objective=1;

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
   args.AddOption(&filter_o, "-of", "--order-filter",
                  "Finite element order for the PDE filter (>=1).");
   args.AddOption(&rho_min, "-rmin", "--rho-min",
                  "Minimal value of rho.");
   args.AddOption(&eta, "-eta", "--eta",
                  "Threshold value.");
   args.AddOption(&beta, "-beta", "--beta",
                  "Projection value.");
   args.AddOption(&objective, "-obj", "--obj",
                  "Objective:1) compliance; 2) p-norm(p=8).");
   args.AddOption(&filter_r, "-rf", "--radius-filter",
                  "PDE filter radius.");
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

   int max_par=par_ref_levels;
   for(int par_ref_levels=1; par_ref_levels<max_par;par_ref_levels++)
   {

       //    Define a parallel mesh by a partitioning of the serial mesh. Refine
       //    this mesh further in parallel to increase the resolution. Once the
       //    parallel mesh is defined, the serial mesh can be deleted.
       mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
       {
          for (int l = 0; l < par_ref_levels; l++)
          {
             pmesh.UniformRefinement();
          }
       }

       //define the PDE filter
       mfem::PDEFilterTO filter(pmesh,filter_r,filter_o);

       // Define the Diffusion coefficient
       mfem::ConstantCoefficient* diffco=new mfem::ConstantCoefficient(1.0);
       // Define the Heat source
       //mfem::ConstantCoefficient* loadco=new mfem::ConstantCoefficient(1.0);
       mfem::Coefficient* loadco=new mfem::FunctionCoefficient(Gyroid);
       // Define the q-function
       mfem::QLinearDiffusion* qfun=new mfem::QLinearDiffusion(*diffco,*loadco,1.0,rho_min,beta,eta);

       // Define FE collection and space for the state solution
       mfem::H1_FECollection sfec(order, dim);
       mfem::ParFiniteElementSpace* sfes=new mfem::ParFiniteElementSpace(&pmesh,&sfec,1);
       // Define FE collection and space for the density field
       mfem::L2_FECollection pfec(0, dim);
       mfem::ParFiniteElementSpace* pfes=new mfem::ParFiniteElementSpace(&pmesh,&pfec,1);

       // Define the arrays for the nonlinear form
       mfem::Array<mfem::ParFiniteElementSpace*> asfes;
       mfem::Array<mfem::ParFiniteElementSpace*> apfes;

       asfes.Append(sfes);
       apfes.Append(pfes);

       // Define parametric block nonlinear form using single scalar H1 field
       // and L2 scalar density field
       mfem::ParParametricBNLForm* nf=new mfem::ParParametricBNLForm(asfes,apfes);
       // add the parametric integrator
       nf->AddDomainIntegrator(new mfem::ParametricLinearDiffusion(*qfun));

       // Define true block vectors for state, adjoint, resudual
       mfem::BlockVector solbv; solbv.Update(nf->GetBlockTrueOffsets());    solbv=0.0;
       mfem::BlockVector adjbv; adjbv.Update(nf->GetBlockTrueOffsets());    adjbv=0.0;
       mfem::BlockVector resbv; resbv.Update(nf->GetBlockTrueOffsets());    resbv=0.0;
       // Define true block vectors for parametric field and gradients
       mfem::BlockVector prmbv; prmbv.Update(nf->PrmGetBlockTrueOffsets()); prmbv=0.0;
       mfem::BlockVector grdbv; grdbv.Update(nf->PrmGetBlockTrueOffsets()); grdbv=0.0;

       // Define grid function for the density
       mfem::ParGridFunction densgrf(pfes);
       {
           mfem::Coefficient* ls_coeff=new mfem::FunctionCoefficient(Gyroid);
           filter.Filter(*ls_coeff,densgrf);
           //densgrf.ProjectCoefficient(*ls_coeff);
           delete ls_coeff;
       }



       //set the BC for the physics
       mfem::Array<mfem::Array<int> *> ess_bdr;
       mfem::Array<mfem::Vector*>      ess_rhs;
       ess_bdr.Append(new mfem::Array<int>(pmesh.bdr_attributes.Max()));
       ess_rhs.Append(nullptr);
       (*ess_bdr[0]) = 1;
       nf->SetEssentialBC(ess_bdr,ess_rhs);
       delete ess_bdr[0];

       // set the density field to 0.5
       //prmbv=0.5;
       densgrf.GetTrueDofs(prmbv.GetBlock(0));
       // set the density as parametric field in the parametric BNLForm
       nf->SetPrmFields(prmbv); //set the density

       // compute the stiffness/tangent matrix for density prmbv=0.5
       mfem::BlockOperator& A=nf->GetGradient(solbv);
       mfem::HypreBoomerAMG* prec=new mfem::HypreBoomerAMG();
       prec->SetPrintLevel(print_level);
       // use only block (0,0) as in this case we have a single field
       prec->SetOperator(A.GetBlock(0,0));

       // construct block preconditioner for the BNLForm
       mfem::BlockDiagonalPreconditioner *blpr = new mfem::BlockDiagonalPreconditioner(nf->GetBlockTrueOffsets());
       blpr->SetDiagonalBlock(0,prec);

       //define the solvers
       mfem::GMRESSolver *gmres;
       gmres = new mfem::GMRESSolver(MPI_COMM_WORLD);
       gmres->SetAbsTol(newton_abs_tol/10);
       gmres->SetRelTol(newton_rel_tol/10);
       gmres->SetMaxIter(100);
       gmres->SetPrintLevel(print_level);
       gmres->SetPreconditioner(*blpr);
       gmres->SetOperator(A);


       // solve the problem
       solbv=0.0;
       nf->Mult(solbv,resbv); resbv.Neg(); //compute RHS
       // linear system solve
       gmres->Mult(resbv, solbv);

       // Compute the energy of the state system
       double energy = nf->GetEnergy(solbv);
       if(myrank==0){
           std::cout<<"energy ="<< energy<<std::endl;
       }

       // Define the block nonlinear form utilized for
       // representing the objective - use the state array from the BNLForm
       mfem::ParBlockNonlinearForm* ob=new mfem::ParBlockNonlinearForm(asfes);
       // Add the integrator for the objective
       //ob->AddDomainIntegrator(new mfem::DiffusionObjIntegrator());
       if(objective==1){
           ob->AddDomainIntegrator(new mfem::ThermalComplianceIntegrator(*loadco));
       }else
       {
           ob->AddDomainIntegrator(new mfem::DiffusionObjIntegrator(4));
       }


       // Compute the objective
       double obj=ob->GetEnergy(solbv);
       if(myrank==0){
           std::cout<<"total_dofs = "<<sfes->GlobalTrueVSize()<<" Objective = "<<obj<<std::endl;
       }


       //Dump out the data
       if(visualization)
       {
           mfem::ParaViewDataCollection *dacol=new mfem::ParaViewDataCollection("ParHeat",&pmesh);
           mfem::ParGridFunction gfdns(pfes); gfdns.SetFromTrueDofs(prmbv.GetBlock(0));
           // Define state grid function
           mfem::ParGridFunction gfsol(sfes); gfsol.SetFromTrueDofs(solbv.GetBlock(0));

           dacol->SetLevelsOfDetail(order);
           dacol->RegisterField("sol", &gfsol);
           dacol->RegisterField("dns", &gfdns);
           dacol->RegisterField("ode",&densgrf);

           dacol->SetTime(1.0);
           dacol->SetCycle(1);
           dacol->Save();

           delete dacol;
       }


       delete ob;
       delete gmres;
       delete blpr;
       delete prec;

       delete nf;
       delete pfes;
       delete sfes;

       delete qfun;
       delete loadco;
       delete diffco;
   }

   MPI_Finalize();
   return 0;
}
