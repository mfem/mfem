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
 *              ρ•dv/dt = (-∇•σ + f)
 *              du/dt = v
 *
 * where σ is the stress tensor, f is the body force,
 * and ρ is the density.
 */
class Elasticity : public Application
{
public:

   // Mesh and finite element space
   ParMesh &mesh;
   ParFiniteElementSpace &fes;
   Array<int> offsets;

   /// Essential and natural dof array.
   Array<int> ess_attr, nat_attr;
   Array<int> ess_tdofs, nat_tdofs;

   /// Material properties
   ConstantCoefficient density, mu, lambda;

   /// Grid functions for the displacement, velocity, and traction
   mutable ParGridFunction x_gf; ///< displacement
   mutable ParGridFunction u_gf; ///< velocity (dx/dt)
   mutable ParGridFunction stress_gf; ///< traction
   mutable ParGridFunction bc_send_gf; ///< Grid functions for transfering BCs
   
   /// Mass and Stiffness forms
   mutable ParBilinearForm Mform, Kform, Kform_e;

   /// RHS form
   mutable ParLinearForm bform;
   mutable Vector b;

   /// Mass and Stiffness operators
   mutable HypreParMatrix Mmat, Kmat, Kmat_e, Mrhomat, Mmat_e;
   mutable HypreParMatrix *T = nullptr;
     
   /// Force 
   VectorGridFunctionCoefficient fcoeff;
   ScalarVectorProductCoefficient scaled_fcoeff;

   /// Mass matrix and implicit solver
   GMRESSolver implicit_solver;   
   CGSolver M_solver;

   /// Preconditioner
   HypreSmoother M_prec;
   mutable HypreBoomerAMG *amg = nullptr;
   mutable Solver *pc = nullptr;

    /// Auxiliary variables
   mutable Vector z;
   real_t current_dt = -1.0;
   bool updated = false;

public:
   Elasticity(ParFiniteElementSpace &fes_,
              Array<int> ess_attr_,
              Array<int> nat_attr_,
              real_t density_ = 1.0,
              real_t mu_ = 1.0,
              real_t lambda_ = 1.0) :
              Application(2*fes_.GetTrueVSize()),
              mesh(*fes_.GetParMesh()),
              fes(fes_),
              ess_attr(ess_attr_),
              nat_attr(nat_attr_),
              density(density_),
              mu(mu_),
              lambda(lambda_),
              x_gf(&fes), u_gf(&fes), stress_gf(&fes),
              bc_send_gf(&fes),
              Mform(&fes), Kform(&fes),
              Kform_e(&fes), bform(&fes),
              fcoeff(&stress_gf), scaled_fcoeff(-1.0, fcoeff),
              implicit_solver(mesh.GetComm()),
              M_solver(mesh.GetComm())
   {
        fes.GetEssentialTrueDofs(ess_attr, ess_tdofs);
        fes.GetEssentialTrueDofs(nat_attr, nat_tdofs);

        x_gf = 0.0;
        u_gf = 0.0;
        stress_gf = 0.0;
        bc_send_gf = 0.0;

        // Setup field collection for output and transfer
        field_collection.SetName("Elasticity");
        field_collection.AddField("Displacement", &x_gf);
        field_collection.AddField("Velocity", &u_gf);
        field_collection.AddField("Traction", &stress_gf);
        field_collection.AddSourceField("Displacement_BC", &bc_send_gf);
        field_collection.AddSourceField("Velocity_BC", &bc_send_gf);

        offsets = Array<int>({0, fes.GetTrueVSize(), fes.GetTrueVSize()});
        offsets.PartialSum();

        Mform.AddDomainIntegrator(new VectorMassIntegrator(density));
        Kform.AddDomainIntegrator(new ElasticityIntegrator(lambda, mu));
        Kform_e.AddDomainIntegrator(new ElasticityIntegrator(lambda, mu));

        if(nat_attr.Size() > 0)
        {
            bform.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(scaled_fcoeff), nat_attr);
        }

        b.SetSize(fes.GetTrueVSize());
        z.SetSize(fes.GetTrueVSize());
        Assemble();
        BuildSolvers();
   }

   /// Assemble linear and bilinear forms; called if mesh is updated
   void Assemble() override 
   {
        AssembleLinearForms();
        AssembleBilinearForms();
   }

   /// Assemble linear forms for traction
   void AssembleLinearForms()
   {
        bform.Assemble();
        b.SetSize(fes.GetTrueVSize());
        bform.ParallelAssemble(b);
   }

   /// Assemble bilinear forms for mass and stiffness
   void AssembleBilinearForms()
   {
        Mform.Assemble();
        Kform.Assemble();
        Kform_e.Assemble();

        Array<int> empty;
        Mform.FormSystemMatrix(ess_tdofs, Mrhomat);
        Kform_e.FormSystemMatrix(ess_tdofs, Kmat_e);
        Kform.FormSystemMatrix(empty, Kmat);
   }

   /// Update finite element space and re-assemble forms
   /// if the mesh has changed
   void Update() override
   {
        fes.Update();

        u_gf.Update();
        x_gf.Update();
        stress_gf.Update();
        bc_send_gf.Update();
    
        Mform.Update();
        Kform.Update();
        Kform_e.Update();
        bform.Update();

        Assemble();
        updated = true;
   }

   /// Build implicit solver and AMG preconditioner
   void BuildSolvers()
   {
        M_solver.iterative_mode = false;
        M_solver.SetRelTol(1e-8);
        M_solver.SetAbsTol(1e-8);
        M_solver.SetMaxIter(1000);
        M_solver.SetPrintLevel(0);
        M_solver.SetOperator(Mrhomat);
        M_prec.SetType(HypreSmoother::Jacobi);
        M_prec.SetOperator(Mrhomat);
        M_solver.SetPreconditioner(M_prec);

        if(amg) delete amg;
        amg = new HypreBoomerAMG;
        HYPRE_BoomerAMGSetSmoothType(*amg, 5);
        amg->SetOperator(Kmat_e);
        amg->SetSystemsOptions(2, true);
        amg->SetElasticityOptions(&fes);
        amg->SetPrintLevel(0);
        pc = amg;

        implicit_solver.iterative_mode = false;
        implicit_solver.SetRelTol(1e-4);
        implicit_solver.SetAbsTol(0.0);
        implicit_solver.SetMaxIter(500);
        implicit_solver.SetKDim(200);
        implicit_solver.SetPrintLevel(0);
        implicit_solver.SetPreconditioner(*pc);
   }

   /// Apply operator
   void Mult(const Vector &u, Vector &k) const override
   {
        BlockVector ub(u.GetData(), offsets);
        BlockVector kb(k.GetData(), offsets);

        Vector &vel = ub.GetBlock(0);
        Vector &pos = ub.GetBlock(1);

        Vector &ku = kb.GetBlock(0);
        Vector &kx = kb.GetBlock(1);
        
        Kmat.Mult(pos,z);
        z.Neg();
        z.Add(1.0, b);

        kx = vel; // dx/dt = u
        M_solver.Mult(z, ku);

        ku.SetSubVector(ess_tdofs, 0.0);
        kx.SetSubVector(ess_tdofs, 0.0);
   }

   /// Solve implicit system in Schur complement form
   void ImplicitSolve(const real_t dt, const Vector &u, Vector &k)
   {
        BlockVector ub(u.GetData(), offsets);
        BlockVector kb(k.GetData(), offsets);

        Vector &vel = ub.GetBlock(0);
        Vector &pos = ub.GetBlock(1);

        Vector &ku = kb.GetBlock(0);
        Vector &kx = kb.GetBlock(1);

        AssembleLinearForms();
        Kmat.Mult(pos,z);
        z.Neg();
        z.Add(1.0, b);
        Kmat.AddMult(vel,z,-dt); // z = z  - dt*K*vel

        if((current_dt != dt) || updated)
        {
            if (T) { delete T; }
            current_dt = dt;
            T = Add(1.0, Mrhomat, dt*dt, Kmat_e);
            implicit_solver.SetOperator(*T);
            amg->SetOperator(*T);
            updated = false;
        }

        implicit_solver.Mult(z, ku);
        add(1.0, vel, dt, ku, kx);

        ku.SetSubVector(ess_tdofs, 0.0);
        kx.SetSubVector(ess_tdofs, 0.0);
   }

   /// Computes residual of the elasticity equations
   void ImplicitMult(const Vector &x, const Vector &k, Vector &v ) const override
   {
        BlockVector xb(x.GetData(), offsets);
        BlockVector kb(k.GetData(), offsets);
        BlockVector vb(v.GetData(), offsets);

        // Apply the implicit operator
        Mrhomat.Mult(kb.GetBlock(0), vb.GetBlock(0)); // v = A(x) + M*k
        vb.GetBlock(0).Add(-1.0, b);
        Kmat.AddMult(xb.GetBlock(1), vb.GetBlock(0), 1.0); // v = K*x

        vb.GetBlock(1) = kb.GetBlock(1); // v = u
        vb.GetBlock(1) -= xb.GetBlock(0); // v = k

        vb.GetBlock(0).SetSubVector(ess_tdofs, 0.0);
        vb.GetBlock(1).SetSubVector(ess_tdofs, 0.0);
   }

    void PreProcess(Vector &x) override {}

    void PostProcess(Vector &x) override {}


   void Transfer(const Vector &x) override
   {
        BlockVector xb(x.GetData(), offsets);
        field_collection.Transfer("Velocity_BC", xb.GetBlock(0));
        field_collection.Transfer("Displacement_BC", xb.GetBlock(1));
   }

   void Transfer(const Vector &u, const Vector &k, real_t dt = 0.0) override
   {
        BlockVector kb(k.GetData(), offsets);        
        field_collection.Transfer("Velocity_BC", kb.GetBlock(0));
        field_collection.Transfer("Displacement_BC", kb.GetBlock(1));
   }

   ~Elasticity() override
   {
        if (amg) { delete amg; }
        if (T) { delete T; }
   }
};



#endif
