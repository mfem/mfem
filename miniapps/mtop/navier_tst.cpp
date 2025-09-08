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

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "ns_operators.hpp"
#include "mms_coefficients.hpp"
using namespace mfem;

template<template<typename> typename VeloPotential,
         template<typename> typename Pres>
class Stokes2DForcing:public VectorCoefficient
{
public:
    Stokes2DForcing(std::shared_ptr<Coefficient> visc_):VectorCoefficient(2)
    {
        visc=visc_;
    }

    virtual void Eval(Vector &V, ElementTransformation &T,
                      const IntegrationPoint &ip) override
    {
        V.SetSize(2); V=0.0;
        real_t t=VectorCoefficient::GetTime();
        Vector tv(2);
        VectorCoefficient* gp;
        //add pressure contribution

        gp=press.GetGradient();
        gp->SetTime(t);
        gp->Eval(tv,T,ip); V.Add(1.0,tv);
        //add inertia
        gp=velo.TimeDerivative();
        gp->SetTime(t);
        gp->Eval(tv,T,ip); V.Add(1.0,tv);
        //add laplacian
        real_t mu=visc->Eval(T,ip);
        gp=velo.VectorLaplacian();
        gp->SetTime(t);
        gp->Eval(tv,T,ip); V.Add(-mu,tv);

    }

private:
    std::shared_ptr<Coefficient> visc;
    ADScalar2DCoeff<Pres> press;
    ADDivFree2DVelocity<VeloPotential> velo;
};

//Define the 2D potentials and pressure fields
template<typename fp_type>
class Velo2DPotential
{
public:
   fp_type operator()(fp_type t,fp_type x,fp_type y)
   {
       return tanh(t)*sin(2*M_PI*x)*sin(2*M_PI*y)*sin(x+y);
   }
};

template<typename fp_type>
class Pressure2D
{
public:
   fp_type  operator()(fp_type t,fp_type x,fp_type y)
   {
       return tanh(t)*sin(M_PI*x)*sin(M_PI*y);

   }
};


class RampVectorCoeff:public VectorCoefficient
{
public:
    RampVectorCoeff(VectorCoefficient& v, real_t t0_=0.0, real_t t1_=1.0):VectorCoefficient(v.GetVDim())
    {
        t0=t0_;
        t1=t1_;
        vv=&v;
    }

    void Eval(Vector &V, ElementTransformation &T,
                        const IntegrationPoint &ip) override
    {
        real_t ct=GetTime();
        vv->Eval(V,T,ip);
        if(ct>t1){return;}
        else
        if(ct<t0){ V*=0.0;}
        else
        {
            V*=((ct)/(t1-t0));
        }
    }

private:
    real_t t0;
    real_t t1;
    VectorCoefficient* vv;
};



int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   mfem::Mpi::Init(argc, argv);
   int myrank = mfem::Mpi::WorldRank();
   mfem::Hypre::Init();

   // Parse command-line options.
   const char *mesh_file = "./mini_flow2d_ball.msh";
   int order = 1;
   bool static_cond = false;
   int ser_ref_levels = 1;
   int par_ref_levels = 1;
   real_t newton_rel_tol = 1e-7;
   real_t newton_abs_tol = 1e-12;
   int newton_iter = 10;
   int print_level = 1;
   bool visualization = false;
   int ode_solver_type = 21;  // SDIRK33Solver

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
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  ODESolver::Types.c_str());
   args.Parse();
   if (!args.Good())
   {
      if (myrank == 0)
      {
         args.PrintUsage(std::cout);
      }
      return 1;
   }

   if (myrank == 0)
   {
      args.PrintOptions(std::cout);
   }

   // Read the (serial) mesh from the given mesh file on all processors.  We
   // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   // and volume meshes with the same code.
   mfem::Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement. We choose
   // 'ref_levels' to be the largest number that gives a final mesh with no
   // more than 10,000 elements.
   /*
   {
      int ref_levels =
         (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }*/

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }

   std::cout<<"My rank="<<pmesh.GetMyRank()<<std::endl;

   /*
   mfem::Vector bcz(dim); bcz=0.0;
   mfem::Vector bci(dim); bci=0.0; bci(0)=1.0;
   VectorConstantCoefficient cvcz(bcz);
   VectorConstantCoefficient cvci(bci);

   std::shared_ptr<mfem::VectorCoefficient> vcz(new RampVectorCoeff(cvcz));
   std::shared_ptr<mfem::VectorCoefficient> vci(new RampVectorCoeff(cvci));

   mfem::Vector vforce(dim); vforce=0.0; vforce(0)=1.0;
   */

   std::shared_ptr<mfem::VectorCoefficient>
           vc(new ADDivFree2DVelocity<Velo2DPotential>());

   std::shared_ptr<mfem::ConstantCoefficient>
           visc(new mfem::ConstantCoefficient(0.1));


   std::shared_ptr<mfem::VectorCoefficient>
           fc(new Stokes2DForcing<Velo2DPotential,Pressure2D>(visc));

   std::shared_ptr<mfem::Coefficient>
           pres_co(new ADScalar2DCoeff<Pressure2D>());

   ParGridFunction velo;
   ParGridFunction pres;
   BlockVector sol;
   ParGridFunction tvelo;
   ParGridFunction tpres;


   std::unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);
   //define the TimeDependentOperator
   TimeDependentStokes* top=new TimeDependentStokes(&pmesh,2,visc);

   top->AddVelocityBC(1,vc);
   top->AddVelocityBC(2,vc);
   top->AddVelocityBC(3,vc);
   top->AddVelocityBC(4,vc);

   top->SetVolForce(fc);

   top->Assemble();

   //start the time stepping

   tvelo.SetSpace(top->GetVelocitySpace()); tvelo=0.0;
   tpres.SetSpace(top->GetPressureSpace()); tpres=0.0;

   velo.SetSpace(top->GetVelocitySpace());	velo=0.0;
   pres.SetSpace(top->GetPressureSpace());  pres=0.0;
   sol.Update(top->GetTrueBlockOffsets());  sol=0.0;

   if(0==myrank){
       std::cout<<"s0="<<sol.BlockSize(0)<<" s1="<<sol.BlockSize(1)<<std::endl;
       std::cout<<" v tv="<<top->GetVelocitySpace()->GetTrueVSize()<<std::endl;
       std::cout<<" p tv="<<top->GetPressureSpace()->GetTrueVSize()<<std::endl;

   }

   top->SetEssVBC(0.0,velo);
   velo.GetTrueDofs(sol.GetBlock(0));
   pres.GetTrueDofs(sol.GetBlock(1));

   std::cout<<"Start projection!"<<std::endl;
   tvelo.ProjectCoefficient(*vc);
   tpres.ProjectCoefficient(*pres_co);
   std::cout<<"End projection!"<<std::endl;



       ParaViewDataCollection paraview_dc("flow", &pmesh);
       paraview_dc.SetPrefixPath("ParaView");
       paraview_dc.SetLevelsOfDetail(order);
       paraview_dc.SetDataFormat(VTKFormat::BINARY);
       paraview_dc.SetHighOrderOutput(true);
       paraview_dc.SetCycle(0);
       paraview_dc.SetTime(0.0);
       paraview_dc.RegisterField("velo",&velo);
       paraview_dc.RegisterField("pres",&pres);
       paraview_dc.RegisterField("velot",&tvelo);
       paraview_dc.RegisterField("prest",&tpres);
       paraview_dc.Save();


   ode_solver->Init(*top);
   real_t t  = 0.0;
   real_t tf = 0.5;
   real_t dt = 0.5;

   bool flag=true;
   int cycl=1;
   while(flag)
   {
       ode_solver->Step(sol,t,dt);
       velo.SetFromTrueDofs(sol.GetBlock(0));
       pres.SetFromTrueDofs(sol.GetBlock(1));

       vc->SetTime(t);
       pres_co->SetTime(t);
       tvelo.ProjectCoefficient(*vc);
       tpres.ProjectCoefficient(*pres_co);

       paraview_dc.SetCycle(cycl);
       paraview_dc.SetTime(t);
       paraview_dc.Save();

       cycl++;

       if(t>tf){flag=false;}

   }


   //free the time dependent operator
   delete top;

   MPI::Finalize();

   return 0;
}
