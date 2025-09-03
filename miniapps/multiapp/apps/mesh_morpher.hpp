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
 * @brief Wave-based mesh morphing time dependent operator
 *
 *              du/dt = c ∇²x 
 *              dx/dt = u
 *
 * c is the wave speed.
 */
class MeshWave : public Application
{
public:

   // Mesh and finite element space
   ParMesh &mesh;
   ParFiniteElementSpace &fes;
   Array<int> offsets;

   /// Essential true dof array. Relevant for eliminating boundary conditions
   Array<int> ess_attr, nat_attr;
   Array<int> ess_tdofs;

   /// Material properties
   ConstantCoefficient wave_speed;

   /// Grid functions for the displacement, velocity, and traction
   mutable ParGridFunction x_gf; ///< displacement
   mutable ParGridFunction u_gf; ///< velocity (dx/dt)


   /// Mass and Stiffness forms
   mutable ParBilinearForm Mform, Mform_e, Kform,  Kform_e;

   /// RHS form
   mutable ParLinearForm bform;
   mutable Vector b;

   /// Mass and Stiffness operators
   OperatorHandle M, K;
   mutable HypreParMatrix Mmat, Kmat, Mmat_e, Kmat_e;


   /// Mass matrix and implicit solver
   BlockOperator *implicit_op   = nullptr; ///< Block linear operator for the Navier-Stokes system
   FGMRESSolver implicit_solver;   
   CGSolver M_solver;

   /// Mass matrix preconditioner
   HypreSmoother M_prec;
   HypreBoomerAMG A_prec_amg;
   mutable BlockLowerTriangularPreconditioner *pc = nullptr;
   mutable Solver *K_pc = nullptr;

    /// Auxiliary vectors for the implicit solve
   mutable Vector z, x_bc, u_bc;


   ConstantCoefficient zero, one, negative_one;

   mutable HypreParMatrix Imat; ///< Identity matrix for implicit solve
   ParBilinearForm IdentityForm;
   real_t current_dt = -1.0;

public:
   MeshWave(ParFiniteElementSpace &fes_,
              Array<int> ess_attr_,
              real_t speed_ = 1.0) :
              Application(2*fes_.GetTrueVSize()),
              mesh(*fes_.GetParMesh()),              
              fes(fes_),
              ess_attr(ess_attr_),
              wave_speed(speed_*speed_),
              x_gf(ParGridFunction(&fes)),
              u_gf(ParGridFunction(&fes)),
              Mform(&fes), Mform_e(&fes),
              Kform(&fes), Kform_e(&fes),
              bform(&fes),
              implicit_solver(MPI_COMM_WORLD),              
              M_solver(MPI_COMM_WORLD),
              zero(0.0), one(1.0), negative_one(-1.0),              
              IdentityForm(&fes)
   {

        fes.GetEssentialTrueDofs(ess_attr, ess_tdofs);

        offsets = Array<int>({0, fes.GetTrueVSize(), fes.GetTrueVSize()});
        offsets.PartialSum();

        
        x_bc.SetSize(fes.GetTrueVSize());        
        u_bc.SetSize(fes.GetTrueVSize());        
        b.SetSize(fes.GetTrueVSize());
        z.SetSize(Height());
        

        Mform.AddDomainIntegrator(new VectorMassIntegrator);
        Kform.AddDomainIntegrator(new VectorDiffusionIntegrator(wave_speed));

        Mform_e.AddDomainIntegrator(new VectorMassIntegrator);
        Kform_e.AddDomainIntegrator(new VectorDiffusionIntegrator(wave_speed));

        Mform.Assemble();
        Kform.Assemble();

        Mform_e.Assemble();
        Kform_e.Assemble();

        bform.Assemble();

        Array<int> empty;
        Mform.FormSystemMatrix(ess_tdofs, Mmat);
        Kform.FormSystemMatrix(ess_tdofs, Kmat);

        Mform_e.FormSystemMatrix(empty, Mmat_e);
        Kform_e.FormSystemMatrix(empty, Kmat_e);

        // Mform.FormLinearSystem(empty, x_gf, bform, Mmat, b, x_bc);
        // Kform.FormLinearSystem(empty, u_gf, bform, Kmat, b, u_bc);

        // Create the identity matrix for implicit solve
        Array<int> all_dofs(fes.GetTrueVSize());
        std::iota(std::begin(all_dofs), std::end(all_dofs), 0);
        IdentityForm.AddDomainIntegrator(new VectorMassIntegrator(zero));
        IdentityForm.Assemble();
        IdentityForm.FormSystemMatrix(all_dofs, Imat);


        implicit_op = new BlockOperator(offsets);
        implicit_op->SetBlock(0, 0, &Kmat);
        implicit_op->SetBlock(0, 1, &Mmat);
        implicit_op->SetBlock(1, 0, &Mmat);
        implicit_op->SetBlock(1, 1, &Mmat);
        

        
        M_solver.iterative_mode = false;
        M_solver.SetRelTol(1e-8);
        M_solver.SetAbsTol(1e-8);
        M_solver.SetMaxIter(1000);
        M_solver.SetPrintLevel(0);
        M_solver.SetOperator(Mmat);
        M_prec.SetType(HypreSmoother::Jacobi);
        // M_solver.SetPreconditioner(M_prec);


        
        if(K_pc) delete K_pc;
        auto amg = new HypreBoomerAMG;
        HYPRE_BoomerAMGSetSmoothType(*amg, 5);
        amg->SetOperator(Kmat);
        amg->SetSystemsOptions(2, true);
        amg->SetPrintLevel(0);
        K_pc = amg;

        pc = new BlockLowerTriangularPreconditioner(offsets);
        pc->SetBlock(0, 0, K_pc);
        pc->SetBlock(1, 0, &Imat);
        // pc->SetBlock(1, 1, &Imat);


        implicit_solver.iterative_mode = false;
        implicit_solver.SetRelTol(1e-5);
        implicit_solver.SetAbsTol(1e-5);
        implicit_solver.SetMaxIter(100);
        implicit_solver.SetKDim(100);
        implicit_solver.SetPrintLevel(0);
        implicit_solver.SetPreconditioner(*pc);
        implicit_solver.SetOperator(*implicit_op);
   }


   void Mult(const Vector &u, Vector &k) const override
   {  
        BlockVector ub(u.GetData(), offsets);
        BlockVector kb(k.GetData(), offsets);
        BlockVector zb(z.GetData(), offsets);        

        Vector &pos = ub.GetBlock(0);
        Vector &vel = ub.GetBlock(1);

        Vector &kx = kb.GetBlock(0);
        Vector &ku = kb.GetBlock(1);
        
        Vector &z0 = zb.GetBlock(0);

        Kmat.Mult(pos,z0);
        z0.Neg();

        kx = vel; // dx/dt = u
        M_solver.Mult(z0, ku);

        kx.SetSubVector(ess_tdofs, 0.0);
        ku.SetSubVector(ess_tdofs, 0.0);
   }

   void ImplicitSolve(const real_t dt, const Vector &u, Vector &k)
   {
        BlockVector ub(u.GetData(), offsets);
        BlockVector kb(k.GetData(), offsets);
        BlockVector zb(z.GetData(), offsets);        

        Vector &pos = ub.GetBlock(0);
        Vector &vel = ub.GetBlock(1);

        Vector &kx = kb.GetBlock(0);        
        Vector &ku = kb.GetBlock(1);

        Vector &z0 = zb.GetBlock(0);
        Vector &z1 = zb.GetBlock(1);

        if(current_dt != dt)
        {            
            implicit_op->SetBlockCoef(0, 0, dt);
            implicit_op->SetBlockCoef(1, 1, -dt);
            current_dt = dt;            
        }

        Kmat_e.Mult(pos,z0);
        Mmat_e.Mult(vel,z1);
        z0.Neg();        

        implicit_solver.Mult(zb, kb);

        kx.SetSubVector(ess_tdofs, 0.0);
        ku.SetSubVector(ess_tdofs, 0.0);
   }

   void Assemble() override
   {
        // Mform.Update();
        // Kform.Update();

        // Mform.Assemble();
        // Kform.Assemble();
        // Mform.FormSystemMatrix(ess_tdofs, Mmat);
        // Kform.FormSystemMatrix(ess_tdofs, Kmat);
        // Mform.FormLinearSystem(ess_tdofs, x_gf, bform, Mmat, b, x_bc);        
        // Kform.FormLinearSystem(ess_tdofs, u_gf, bform, Kmat, b, u_bc);
   }

   void Update() override
   {
        Mform.Update();
        Kform.Update();

        Mform.Assemble();
        Kform.Assemble();

        bform.Update();
        bform.Assemble();
   }

   void Transfer(const Vector &x) override
   {
        BlockVector xb(x.GetData(), offsets);    
        x_gf.SetFromTrueDofs(xb.GetBlock(0));
        u_gf.SetFromTrueDofs(xb.GetBlock(1));
        Application::Transfer();
   }   

   ~MeshWave() override
   {
        if (implicit_op) { delete implicit_op; }
        if (K_pc) { delete K_pc; }        
        if (pc) { delete pc; }
   }
};



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
          implicit_solver.SetMaxIter(100);
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

     void Solve(const Vector &x, Vector &y) const {}

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
          
          Array<int> empty;
          Mform.FormSystemMatrix(ess_tdofs, Mmat_e);
          Kform_e.FormSystemMatrix(ess_tdofs, Kmat_e);
          Kform.FormSystemMatrix(empty, Kmat);
     }

     void PreProcess(Vector &x) override
     {   
          if(operation_id == OperationID::NONE || operation_id == OperationID::STEP)
          {   // Only do this pre-processing for operations that are not for multi-stage 
               // time stepping
               Update();
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
