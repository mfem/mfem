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


#ifndef MFEM_MESH_MORPHER_HPP
#define MFEM_MESH_MORPHER_HPP

#include "mfem.hpp"
#include "../multiapp.hpp"


using namespace mfem;


/**
 * @brief Mesh morphing is modeled as a time dependent vector diffusion equation
 *
 *              dx/dt = κΔx
 *
 * with vector diffusion coefficient, κ.
 */
class MeshDiffusion : public Application
{
public: 

   // Mesh and finite element space
   ParMesh &mesh;
   ParFiniteElementSpace &fes;
   Array<int> offsets; ///< Offsets for the velocity and displacement

   /// Essential true dof array. Relevant for eliminating boundary conditions
   /// when using an H1 space.
   Array<int> ess_attr, ess_tdofs;

   /// Diffusion coefficient
   ConstantCoefficient kappa;

    /// Grid function for the mesh displacement variable
   mutable ParGridFunction x_gf; ///< Grid function for the mesh displacement variable x
   mutable ParGridFunction u_gf; ///< Grid function for the time derivative of x (velocity)

   /// Mass and Stiffness forms
   mutable ParBilinearForm Mform, Kform,  Kform_e;

   /// RHS form
   mutable ParLinearForm bform;

   /// Mass and Stiffness operators
   OperatorHandle M, K;
   mutable HypreParMatrix Mmat, Kmat,  Mmat_e, Kmat_e;

   /// Mass matrix and implicit solver
   mutable CGSolver M_solver;
   mutable GMRESSolver implicit_solver;
   HypreParMatrix *T = nullptr; // T = M + dt K

   /// Mass matrix preconditioner
   HypreSmoother M_prec;
   HypreSmoother T_prec;  // Preconditioner for the implicit solver

   real_t current_dt = -1.0; 

   /// Auxiliary vectors
   mutable Vector x_bc;
   mutable Vector b, z;
   bool diffuse_velocity = false;

public:

     MeshDiffusion(ParFiniteElementSpace &fes_,
               Array<int> ess_attr_,
               real_t kappa_ = 1.0e0,
               bool diffuse_velocity_ = false) : 
               Application(2*fes_.GetTrueVSize()),
               mesh(*fes_.GetParMesh()),
               fes(fes_),
               ess_attr(ess_attr_),
               kappa(kappa_),
               x_gf(ParGridFunction(&fes)),
               u_gf(ParGridFunction(&fes)),
               Mform(&fes),
               Kform(&fes),Kform_e(&fes),
               bform(&fes),
               M_solver(fes.GetComm()),
               implicit_solver(fes.GetComm()),
               b(fes_.GetTrueVSize()), z(fes_.GetTrueVSize()),
               diffuse_velocity(diffuse_velocity_)
     {

          fes.GetEssentialTrueDofs(ess_attr, ess_tdofs);

          offsets = Array<int>({0, fes.GetTrueVSize(), fes.GetTrueVSize()});
          offsets.PartialSum();

          Mform.AddDomainIntegrator(new VectorMassIntegrator);
          Kform.AddDomainIntegrator(new VectorDiffusionIntegrator(kappa));
          Kform_e.AddDomainIntegrator(new VectorDiffusionIntegrator(kappa));

          Mform.Assemble();
          Kform.Assemble();
          Kform_e.Assemble();

          bform.Assemble();
          
          Array<int> empty;
          Mform.FormSystemMatrix(ess_tdofs, Mmat_e);
          Kform_e.FormSystemMatrix(ess_tdofs, Kmat_e);
          Kform.FormSystemMatrix(empty, Kmat);
          
          
          M_solver.iterative_mode = false;
          M_solver.SetRelTol(1e-8);
          M_solver.SetAbsTol(0.0);
          M_solver.SetMaxIter(100);
          M_solver.SetPrintLevel(0);
          M_solver.SetOperator(Mmat_e);
          M_prec.SetType(HypreSmoother::Jacobi);
          M_solver.SetPreconditioner(M_prec);


          implicit_solver.iterative_mode = false;
          implicit_solver.SetRelTol(1e-8);
          implicit_solver.SetAbsTol(1e-5);
          implicit_solver.SetMaxIter(500);
          implicit_solver.SetPrintLevel(0);
          T_prec.SetType(HypreSmoother::Jacobi);
          implicit_solver.SetPreconditioner(T_prec);
     }

     void Mult(const Vector &x, Vector &k) const override
     {
          BlockVector xb(x.GetData(), offsets);
          BlockVector kb(k.GetData(), offsets);

          int id = 1; // Diffuse displacement
          if(diffuse_velocity) id = 0; // Diffuse velocity
          
          Kmat.Mult(xb.GetBlock(id), z);
          M_solver.Mult(z, kb.GetBlock(id));

          if(diffuse_velocity)
          {   // compute kx = u0
               kb.GetBlock(1) = xb.GetBlock(0);
          } 
          else
          {   // compute ku = dudt = d²x/dt² = K²x
               Kmat.Mult(z, kb.GetBlock(0));
          }

          kb.GetBlock(0).SetSubVector(ess_tdofs, 0.0);
          kb.GetBlock(1).SetSubVector(ess_tdofs, 0.0);
     }

     void ImplicitSolve(const real_t dt, const Vector &x, Vector &k)
     {
          BlockVector xb(x.GetData(), offsets);
          BlockVector kb(k.GetData(), offsets);

          if(current_dt != dt)
          {
               if (T) delete T;
               T = Add(1.0, Mmat_e, dt, Kmat_e);
               implicit_solver.SetOperator(*T);
               current_dt = dt;
          }

          int id = 1; // Diffuse displacement
          if(diffuse_velocity) id = 0; // Diffuse velocity

          Kmat.Mult(xb.GetBlock(id), z);
          z.Neg();
          implicit_solver.Mult(z, kb.GetBlock(id));

          if(diffuse_velocity)
          {   // compute kx = u0 + dt*ku
               kb.GetBlock(1) = xb.GetBlock(0);
               kb.GetBlock(1).Add(dt, kb.GetBlock(0));
          } 
          else
          {   // compute ku = dudt = d²x/dt² = d/dt(Kx) = K*kx
               Kmat_e.Mult(kb.GetBlock(1), kb.GetBlock(0));
          }
          
          kb.GetBlock(0).SetSubVector(ess_tdofs, 0.0);
          kb.GetBlock(1).SetSubVector(ess_tdofs, 0.0);
     }

     void Solve(const Vector &x, Vector &y) const override {}

     void Assemble() override {}


     void Update() override
     {
          fes.Update();
          x_gf.Update();
          u_gf.Update();

          Mform.Update();
          Kform.Update();
          Kform_e.Update();

          Mform.Assemble();
          Kform.Assemble();
          Kform_e.Assemble();
          bform.Assemble();
          
          Array<int> empty;
          Mform.FormSystemMatrix(ess_tdofs, Mmat_e);
          Kform_e.FormSystemMatrix(ess_tdofs, Kmat_e);
          Kform.FormSystemMatrix(empty, Kmat);
     }

     void PreProcess(Vector &x) override
     {
          Update();
          if(operation_id == OperationID::NONE || operation_id == OperationID::STEP)
          {   // Only do this pre-processing for operations that are not for multi-stage 
               // time stepping
               BlockVector xb(x.GetData(), offsets);               
               u_gf.GetTrueDofs(xb.GetBlock(0));
               x_gf.GetTrueDofs(xb.GetBlock(1));
          }
     }

     void PostProcess(Vector &x) override
     {
          if(operation_id == OperationID::NONE || operation_id == OperationID::STEP)
          {   // Only do this pre-processing for operations that are not for multi-stage 
               // time stepping
               BlockVector xb(x.GetData(), offsets);
               u_gf.SetFromTrueDofs(xb.GetBlock(0));
               x_gf.SetFromTrueDofs(xb.GetBlock(1));
          }
     }


     void Transfer(const Vector &x) override
     {
          Application::Transfer();
     }   

     ~MeshDiffusion() override
     {
          if (T) delete T;
     }
};






#endif
