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
   mutable ParGridFunction sigma_gf; ///< traction


   /// Mass and Stiffness forms
   mutable ParBilinearForm Mform, Mform_rho, Mform_dt, Kform, Kform_dt;

   /// RHS form
   mutable ParLinearForm bform;
   mutable Vector b;
   real_t b_norm = 0.0;

   /// Mass and Stiffness operators
   OperatorHandle M, K;
   mutable HypreParMatrix Mmat, Kmat, Mmat_dt, Kmat_dt, Mmat_rho, Mmat_e;

   /// Force 
   VectorGridFunctionCoefficient fcoeff;

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
   mutable Vector z;


   ConstantCoefficient zero, one, negative_one, dtc;
   ProductCoefficient lambdadt, mudt, negative_dtc;

   mutable HypreParMatrix Imat; ///< Identity matrix for implicit solve
   ParBilinearForm IdentityForm;
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
              sigma_gf(ParGridFunction(&fes)),
              Mform(&fes), Mform_rho(&fes), Mform_dt(&fes),
              Kform(&fes), Kform_dt(&fes),
              bform(&fes),
              fcoeff(VectorGridFunctionCoefficient(&sigma_gf)),
              implicit_solver(MPI_COMM_WORLD),              
              M_solver(MPI_COMM_WORLD),
              zero(0.0), one(1.0), negative_one(-1.0), dtc(1.0),
              lambdadt(lambda,dtc), mudt(mu,dtc), negative_dtc(-1.0, dtc),
              IdentityForm(&fes)
   {

        fes.GetEssentialTrueDofs(ess_attr, ess_tdofs);

        offsets = Array<int>({0, fes.GetTrueVSize(), fes.GetTrueVSize()});
        offsets.PartialSum();

        b.SetSize(fes.GetTrueVSize());
        z.SetSize(Height());

        Mform.AddDomainIntegrator(new VectorMassIntegrator);
        Mform_rho.AddDomainIntegrator(new VectorMassIntegrator(density));
        Mform_dt.AddDomainIntegrator(new VectorMassIntegrator(negative_dtc));
        Kform.AddDomainIntegrator(new ElasticityIntegrator(lambda, mu));
        Kform_dt.AddDomainIntegrator(new ElasticityIntegrator(lambdadt, mudt));

        Mform.Assemble();
        Mform_rho.Assemble();
        Mform_dt.Assemble();
        Kform.Assemble();
        Kform_dt.Assemble();

        Array<int> empty;
        Mform.FormSystemMatrix(ess_tdofs, Mmat);
        Mform_dt.FormSystemMatrix(ess_tdofs, Mmat_dt);
        Mform_rho.FormSystemMatrix(empty, Mmat_rho);
        Kform_dt.FormSystemMatrix(ess_tdofs, Kmat_dt);

        Kform.FormSystemMatrix(empty, Kmat);
        Mform.FormSystemMatrix(empty, Mmat_e);
        // Kform.FormSystemMatrix(empty, K0mat);

        // Create the identity matrix for implicit solve
        Array<int> all_dofs(fes.GetTrueVSize());
        std::iota(std::begin(all_dofs), std::end(all_dofs), 0);        
        IdentityForm.AddDomainIntegrator(new VectorMassIntegrator(zero));
        IdentityForm.Assemble();
        IdentityForm.FormSystemMatrix(all_dofs, Imat);


        implicit_op = new BlockOperator(offsets);
        implicit_op->SetBlock(0, 0, &Kmat_dt);
        implicit_op->SetBlock(0, 1, &Mmat_rho);
        implicit_op->SetBlock(1, 0, &Imat);
        implicit_op->SetBlock(1, 1, &Imat);
        

        bform.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(fcoeff), nat_attr);
        // bform.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(fcoeff));
        bform.Assemble();
        bform.ParallelAssemble(b);
        b_norm = b.Norml2();
        

        M_solver.iterative_mode = false;
        M_solver.SetRelTol(1e-8);
        M_solver.SetAbsTol(1e-8);
        M_solver.SetMaxIter(1000);
        M_solver.SetPrintLevel(0);
        M_solver.SetOperator(Mmat_rho);
        M_prec.SetType(HypreSmoother::Jacobi);
        // M_solver.SetPreconditioner(M_prec);


        
        if(K_pc) delete K_pc;
        auto amg = new HypreBoomerAMG;
        HYPRE_BoomerAMGSetSmoothType(*amg, 5);
        amg->SetOperator(Kmat_dt);
        amg->SetSystemsOptions(2, true);
        amg->SetElasticityOptions(&fes);
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
            implicit_op->SetBlockCoef(1, 1, -dt);
            dtc.constant = dt;
            Assemble();
            current_dt = dt;            
        }

        zb /= (b_norm*density.constant); // Normalize zb
        implicit_solver.Mult(zb, kb);
        kb *= (b_norm*density.constant); // Scale back kb

        kx.SetSubVector(ess_tdofs, 0.0);
        ku.SetSubVector(ess_tdofs, 0.0);
   }

   void Assemble() override
   {
        Mform_dt.Assemble();
        Kform_dt.Assemble();
        Mform_dt.FormSystemMatrix(ess_tdofs, Mmat_dt);
        Kform_dt.FormSystemMatrix(ess_tdofs, Kmat_dt);
   }

   void Update() override
   {
        Mform_dt.Update();
        Kform_dt.Update();

        Mform_dt.Assemble();
        Kform_dt.Assemble();

        bform.Update();
        bform.Assemble();
        bform.ParallelAssemble(b);
        b_norm = b.Norml2();
   }

   ~Elasticity() override
   {
        if (implicit_op) { delete implicit_op; }
        if (K_pc) { delete K_pc; }        
        if (pc) { delete pc; }
   }
};



#endif
