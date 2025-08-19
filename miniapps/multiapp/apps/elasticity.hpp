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


#ifndef MFEM_ELASTICITY_HPP
#define MFEM_ELASTICITY_HPP

#include "mfem.hpp"
#include "../multiapp.hpp"


using namespace mfem;

/**
 * @brief Elasticity time dependent operator
 *
 *              ρ•dU/dt = (-∇•σ + f)
 *              dX/dt = U
 *
 * where σ is the stress tensor, f is the body force, and ρ is the density.
 */
class Elasticity : public Application
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
   ConstantCoefficient mu, density, lambda;

   /// Grid functions for the displacement, velocity, and traction
   mutable ParGridFunction x_gf; ///< displacement
   mutable ParGridFunction u_gf; ///< velocity (dx/dt)
   mutable ParGridFunction u_gf_trans; ///< velocity (dx/dt) for transfer
   mutable ParGridFunction stress_gf; ///< traction

   /// Mass and Stiffness forms
   mutable ParBilinearForm Mform, Mrho_form, Kform, Kform_e;

   /// RHS form
   mutable ParLinearForm bform;
   mutable Vector b;
   real_t b_norm = 1.0;

   /// Mass and Stiffness operators
//    mutable OperatorHandle Mmat, Kmat, Kmat_e, Mrhomat;
   mutable HypreParMatrix Mmat, Kmat, Kmat_e, Mrhomat, Mmat_e;
     

   /// Force 
   VectorGridFunctionCoefficient fcoeff;
   ScalarVectorProductCoefficient scaled_fcoeff;

   /// Mass matrix and implicit solver
   BlockOperator *implicit_op   = nullptr; ///< Block linear operator for the Navier-Stokes system
   GMRESSolver implicit_solver;   
   CGSolver M_solver;
   IdentityOperator IdentityOp;


   /// Preconditioner
   HypreSmoother M_prec;
   mutable BlockLowerTriangularPreconditioner *pc = nullptr;
   mutable Solver *K_pc = nullptr;

    /// Auxiliary vectors for the implicit solve
   mutable Vector z;

   ConstantCoefficient zero, one, negative_one;

   real_t current_dt = -1.0;

public:
   Elasticity(ParFiniteElementSpace &fes_,
              Array<int> ess_attr_,
              Array<int> nat_attr_,
              real_t mu_ = 1.0,
              real_t density_ = 1.0,              
              real_t lambda_ = 1.0) :
              Application(2*fes_.GetTrueVSize()),
              mesh(*fes_.GetParMesh()),              
              fes(fes_),
              ess_attr(ess_attr_),
              nat_attr(nat_attr_),
              mu(mu_),
              density(density_),
              lambda(lambda_),
              x_gf(ParGridFunction(&fes)),
              u_gf(ParGridFunction(&fes)),
              u_gf_trans(ParGridFunction(&fes)),
              stress_gf(ParGridFunction(&fes)),
              Mform(&fes), Mrho_form(&fes),
              Kform(&fes), Kform_e(&fes),
              bform(&fes),
              fcoeff(VectorGridFunctionCoefficient(&stress_gf)),
              scaled_fcoeff(-1.0, fcoeff),
              implicit_solver(MPI_COMM_WORLD),              
              M_solver(MPI_COMM_WORLD),
              IdentityOp(fes.GetTrueVSize()),
              zero(0.0), one(1.0), negative_one(-1.0)
   {

        fes.GetEssentialTrueDofs(ess_attr, ess_tdofs);

        offsets = Array<int>({0, fes.GetTrueVSize(), fes.GetTrueVSize()});
        offsets.PartialSum();

        b.SetSize(fes.GetTrueVSize());
        z.SetSize(Height());

     //    Mform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
     //    Mrho_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
     //    Kform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
     //    Kform_e.SetAssemblyLevel(AssemblyLevel::PARTIAL);

        Mform.AddDomainIntegrator(new VectorMassIntegrator);
        Mrho_form.AddDomainIntegrator(new VectorMassIntegrator(density));
        Kform.AddDomainIntegrator(new ElasticityIntegrator(lambda, mu));
        Kform_e.AddDomainIntegrator(new ElasticityIntegrator(lambda, mu));

        Mform.Assemble();
        Mrho_form.Assemble();
        Kform.Assemble();
        Kform_e.Assemble();

        Array<int> empty;
        Mform.FormSystemMatrix(empty, Mmat);
        Mrho_form.FormSystemMatrix(empty, Mrhomat);
        Kform_e.FormSystemMatrix(ess_tdofs, Kmat_e);
        Kform.FormSystemMatrix(empty, Kmat);
     //    Kform.FormSystemMatrix(ess_tdofs, Kmat_e);


        implicit_op = new BlockOperator(offsets);
        implicit_op->SetBlock(0, 0, &Kmat_e);
        implicit_op->SetBlock(0, 1, &Mrhomat);
        implicit_op->SetBlock(1, 0, &IdentityOp);
        implicit_op->SetBlock(1, 1, &IdentityOp);

        if(nat_attr.Size() > 0)
        {
            bform.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(scaled_fcoeff), nat_attr);            
          //   bform.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(fcoeff));
        }

        bform.Assemble();
        bform.ParallelAssemble(b);
        if(nat_attr.Size() > 0) b_norm = b.Norml2();
        

        M_solver.iterative_mode = false;
        M_solver.SetRelTol(1e-8);
        M_solver.SetAbsTol(1e-8);
        M_solver.SetMaxIter(1000);
        M_solver.SetPrintLevel(0);
        M_solver.SetOperator(Mrhomat);
        M_prec.SetType(HypreSmoother::Jacobi);
        M_prec.SetOperator(Mrhomat);
        M_solver.SetPreconditioner(M_prec);

        
        if(K_pc) delete K_pc;
        auto amg = new HypreBoomerAMG;
        HYPRE_BoomerAMGSetSmoothType(*amg, 5);
        amg->SetOperator(Kmat_e);
        amg->SetSystemsOptions(2, true);
        amg->SetElasticityOptions(&fes);
        amg->SetPrintLevel(0);
        K_pc = amg;     

        pc = new BlockLowerTriangularPreconditioner(offsets);
     //    M_prec.SetOperator(Kmat_e);
     //    pc->SetBlock(0, 0, &M_prec);
        pc->SetBlock(0, 0, K_pc);
        pc->SetBlock(1, 0, &IdentityOp);
        pc->SetBlock(1, 1, &IdentityOp);

        implicit_solver.iterative_mode = false;
        implicit_solver.SetRelTol(1e-4);
        implicit_solver.SetAbsTol(0.0);
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
        z0.Add(1.0, b);

        z0 /= (b_norm*density.constant); // Normalize zb

        kx = vel; // dx/dt = u        
        M_solver.Mult(z0, ku);

        ku *= (b_norm*density.constant); // Scale back ku

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

        Kmat.Mult(pos,z0);
        z0.Neg();
        z0.Add(1.0, b);
        z1 = vel; // dx/dt = u

        if(current_dt != dt)
        {            
            implicit_op->SetBlockCoef(0, 0, dt);
            implicit_op->SetBlockCoef(1, 1, -dt);            
            current_dt = dt;
          //   dtc.constant = dt;
            Assemble();
        }

        zb /= (b_norm*density.constant); // Normalize zb
        implicit_solver.Mult(zb, kb);
        kb *= (b_norm*density.constant); // Scale back kb

        kx.SetSubVector(ess_tdofs, 0.0);
        ku.SetSubVector(ess_tdofs, 0.0);
   }

   void Assemble() override
   {
     //    Kform_e.Assemble();
     //    Kform_e.FormSystemMatrix(ess_tdofs, Kmat_e);
   }

   void Update() override
   {
        // Kform_e.Update(&fes);
        // Kform_e.Assemble();

        bform.Update();
        bform.Assemble();
        bform.ParallelAssemble(b);
        if(nat_attr.Size() > 0) b_norm = b.Norml2();
   }

    void PreProcess(Vector &x) override
    {
        Update();
    }

    void PostProcess(Vector &x) override
    {
        if(operation_id == OperationID::NONE || operation_id == OperationID::STEP)
        {   // Only do this pre-processing for operations that are not for multi-stage 
            // time stepping
          BlockVector xb(x.GetData(), offsets);
          x_gf.SetFromTrueDofs(xb.GetBlock(0));
          u_gf.SetFromTrueDofs(xb.GetBlock(1));
          u_gf_trans.SetFromTrueDofs(xb.GetBlock(1));
        }
    }


   void Transfer(const Vector &x) override
   {
        BlockVector xb(x.GetData(), offsets);
        u_gf_trans.SetFromTrueDofs(xb.GetBlock(1));
        Application::Transfer();
   }   

   ~Elasticity() override
   {
        if (implicit_op) { delete implicit_op; }
        if (K_pc) { delete K_pc; }        
        if (pc) { delete pc; }
   }
};



#endif
