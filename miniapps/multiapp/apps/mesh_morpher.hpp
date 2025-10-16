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

   /// Essential true dof array.
   Array<int> ess_attr, ess_tdofs;

   /// Diffusion coefficient
   ConstantCoefficient kappa;

   /// Grid function for the mesh displacement variable
   mutable ParGridFunction x_gf; 
   mutable ParGridFunction x_gf_bc;
   mutable ParGridFunction u_gf_bc;
   mutable ParGridFunction bc_send_gf; ///< Grid functions for transfering BCs

   /// Mass and Stiffness forms
   mutable ParBilinearForm Mform, Kform,  Kform_e;

   /// Mass and Stiffness operators
   OperatorHandle M, K;
   mutable HypreParMatrix Mmat, Kmat,  Mmat_e, Kmat_e;

   /// Mass matrix and implicit solver
   mutable CGSolver M_solver;
   mutable GMRESSolver implicit_solver;
   HypreParMatrix *T = nullptr; // T = M + dt K

   /// Mass matrix preconditioner
   HypreSmoother M_prec;
   HypreSmoother T_prec;

   real_t current_dt = -1.0; 

   /// Auxiliary vectors
   mutable Vector z, zv;
   bool updated = false;

public:

     MeshDiffusion(ParFiniteElementSpace &fes_,
                   Array<int> ess_attr_,
                   real_t kappa_ = 1.0e0) : 
                   Application(fes_.GetTrueVSize()),
                   mesh(*fes_.GetParMesh()),
                   fes(fes_),
                   ess_attr(ess_attr_),
                   kappa(kappa_),
                   x_gf(&fes),
                   x_gf_bc(&fes), u_gf_bc(&fes),
                   bc_send_gf(&fes),
                   Mform(&fes), Kform(&fes),Kform_e(&fes),
                   M_solver(mesh.GetComm()),
                   implicit_solver(mesh.GetComm()),
                   z(fes_.GetTrueVSize())
     {
          fes.GetEssentialTrueDofs(ess_attr, ess_tdofs);

          x_gf = 0.0;
          x_gf_bc = 0.0;
          u_gf_bc = 0.0;
          bc_send_gf = 0.0;

          // Setup field collection for output and transfer
          field_collection.SetName("Mesh-Diffusion");
          field_collection.AddSourceField("Displacement", &x_gf);
          field_collection.AddSourceField("dxdt", &bc_send_gf);
          field_collection.AddField("Displacement_BC", &x_gf_bc);
          field_collection.AddField("Velocity_BC", &u_gf_bc);

          Mform.AddDomainIntegrator(new VectorMassIntegrator);
          Kform.AddDomainIntegrator(new VectorDiffusionIntegrator(kappa));
          Kform_e.AddDomainIntegrator(new VectorDiffusionIntegrator(kappa));

          Assemble();
          BuildSolvers();
     }

     void Assemble() override 
     {
          Mform.Assemble();
          Kform.Assemble();
          Kform_e.Assemble();

          Array<int> empty;
          Mform.FormSystemMatrix(ess_tdofs, Mmat_e);
          Kform_e.FormSystemMatrix(ess_tdofs, Kmat_e);
          Kform.FormSystemMatrix(empty, Kmat);
     }

     void Update() override
     {
          fes.Update();
          x_gf.Update();
          x_gf_bc.Update();
          u_gf_bc.Update();
          bc_send_gf.Update();

          Mform.Update();
          Kform.Update();
          Kform_e.Update();

          Assemble();
          updated = true;
     }

     void BuildSolvers()
     {
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
          Kmat.Mult(x, z);
          z.Neg();
          M_solver.Mult(z, k);
          k.SetSubVector(ess_tdofs, 0.0);
     }

     void ImplicitSolve(const real_t dt, const Vector &x, Vector &k)
     {
          if((current_dt != dt) || updated)
          {
               if (T) delete T;
               T = Add(1.0, Mmat_e, dt, Kmat_e);
               implicit_solver.SetOperator(*T);
               current_dt = dt;
               updated = false;
          }

          Kmat.Mult(x, z);
          z.Neg();
          implicit_solver.Mult(z, k);

          if(IsCoupled())
          {
               x_gf_bc.GetTrueDofs(z); // contains the BC in terms of kx (dx/dt)
               for (int i = 0; i < ess_tdofs.Size(); i++)
               {
                    int idx = ess_tdofs[i];
                    k(idx) = z(idx);
               }
          }
          else
          {
               k.SetSubVector(ess_tdofs, 0.0);
          }
     }

     void ImplicitMult(const Vector &x, const Vector &k, Vector &v ) const override
     {}


     void PreProcess(Vector &x) override
     {}

     void PostProcess(Vector &x) override {}

     void Transfer(const Vector &x) override
     {
          field_collection.Transfer("Displacement", x);

          zv.SetSize(fes.GetTrueVSize());
          u_gf_bc.GetTrueDofs(zv);
          Kmat.Mult(x, z);
          z.Neg();
          for (int i = 0; i < ess_tdofs.Size(); i++)
          {
               int idx = ess_tdofs[i];
               z(idx) = zv(idx);
          }
          field_collection.Transfer("dxdt", z);
     }

     void Transfer(const Vector &u, const Vector &k, real_t dt = 0.0) override
     {
          field_collection.Transfer("dxdt", k);
     }

     ~MeshDiffusion() override
     {
          if (T) delete T;
     }
};


#endif
