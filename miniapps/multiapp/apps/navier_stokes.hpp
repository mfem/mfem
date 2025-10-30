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


#ifndef MFEM_NAVIER_STOKES_HPP
#define MFEM_NAVIER_STOKES_HPP

#include "mfem.hpp"
#include "../multiapp.hpp"


using namespace mfem;

/// @brief Compute deviatoric stress on a boundary
class DeviatoricStressCoefficient : public VectorCoefficient
{
protected:
    ParGridFunction *p_gf  = nullptr;  ///< Grid function for pressure
    ParGridFunction *u_gf  = nullptr;  ///< Grid function for velocity
    Coefficient *viscosity = nullptr;  ///< Coefficient for the kinematic viscosity
    Coefficient *density   = nullptr;  ///< Coefficient for the density
    bool scaled_pressure   = false;    ///< True if pressure is density-scaled

    DenseMatrix dudx, tau; ///< Velocity gradient tensor
    Vector normal;

public:
    // For velocity and pressure case
    DeviatoricStressCoefficient(ParGridFunction *p_gf_,
                                ParGridFunction *u_gf_,
                                Coefficient *viscosity_,
                                Coefficient *density_,
                                bool scaled_pressure_ = false) :
                                VectorCoefficient(p_gf_->FESpace()->GetMesh()->Dimension()),
                                p_gf(p_gf_), u_gf(u_gf_),
                                viscosity(viscosity_), density(density_), scaled_pressure(scaled_pressure_)
    {
        int ndim = p_gf->FESpace()->GetMesh()->Dimension();
        normal.SetSize(ndim);
        if(u_gf)
        {
            dudx.SetSize(ndim, vdim);
            tau.SetSize(ndim, vdim);
        }
    }

    // For pressure-only case
    DeviatoricStressCoefficient(ParGridFunction *p_gf_) :
                                DeviatoricStressCoefficient(p_gf_,NULL,NULL,NULL) {}

    /// Evaluate the deviatoric stress tensor at the given point.
    void Eval(Vector &v, ElementTransformation &T,
              const IntegrationPoint &ip) override
    {
        v.SetSize(vdim);
        v=0.0;

        const DenseMatrix &jacobian = T.Jacobian();

        MFEM_ASSERT( (jacobian.Height() - 1 == jacobian.Width()), 
                     "Incorrect Jacobian dimension. Coefficient only supported for boundary elements.");

        CalcOrtho(jacobian, normal);
        const double scale = normal.Norml2();
        normal /= scale;

        real_t rho = density ? density->Eval(T, ip) : 1.0;

        if(p_gf)
        {   // Add pressure term to the deviatoric stress tensor
            real_t pscale = scaled_pressure ? rho : 1.0;
            for (int i = 0; i < vdim; ++i)
            {
                v(i) = -pscale*p_gf->GetValue(T, ip) * normal(i);
            }
        }

        if(u_gf)
        {
            real_t nu  = viscosity ? viscosity->Eval(T, ip) : 1.0;
            u_gf->GetVectorGradient(T, dudx);

            for (int i = 0; i < vdim; i++)
            {
                for (int j = 0; j < vdim; j++)
                {
                    tau(i,j) = rho*nu*(dudx(i,j) + dudx(j,i));
                }
            }
            tau.AddMult(normal, v);
        }
    }
};

/**
 * @brief Navier-stokes time dependent operator
 *  du/dt = - div(u u) + nu * laplacian(u) - grad(p) + f
 *       div(u) = 0
 * 
 *   M du/dt = [ -N(u) + L, -G][u]
 *         0 = [ D        ,  0][p]
 */
class NavierStokes : public Application
{
protected:

  class NewtonResidual : public TimeDependentOperator
  {
  protected:
    NavierStokes *app;
    const Vector *u = nullptr; ///< Pointer to the input vector, used in Mult() methods
    mutable future::FDJacobian grad;
    real_t dt= 0.0;     ///< Time step size, used for time-dependent applications
    mutable Vector upk; ///< Temporary vector for u + dt*k

  public:
    NewtonResidual(NavierStokes *app_) : TimeDependentOperator(app_->Width()), 
                                         app(app_), grad(*this,1e-6), upk(Width()) {}
    virtual void SetTimeStep(real_t dt_){ dt = dt_;}
    virtual void SetState(const Vector *u_){u = u_;}
    virtual void Mult(const Vector &k, Vector &y) const override
    {
        add(1.0,*u, dt, k, upk); // upk = u + dt*k
        app->ImplicitMult(upk, k, y); // y = f(upk,k,t)
    }
    Operator& GetGradient(const Vector &k) const override {
        // grad.Update(k);
        // return const_cast<future::FDJacobian&>(grad);
        add(1.0,*u, dt, k, upk); // upk = u + dt*k 
        return app->GetGradient(upk);
    }
};

public:

   // Mesh and finite element spaces
   ParMesh &mesh;
   ParFiniteElementSpace &u_fes, &p_fes;
   Array<int> offsets; ///< Offsets for the velocity and pressure spaces

   /// Essential true dof array. Relevant for eliminating boundary conditions
   Array<int> u_ess_attr, p_ess_attr;
   Array<int> u_ess_tdofs, p_ess_tdofs;

   ConstantCoefficient viscosity, density, inv_density, compressibility; // Fluid properties

   mutable ParGridFunction u_gf, p_gf; ///< Grid function for the velocity and pressure
   mutable ParGridFunction u_gf_bc, p_gf_bc; ///< Grid functions for enforcing/transfering BCs

   mutable ParBilinearForm Mpform, Muform, Kuform; ///< Mass and Stiffness forms
   mutable ParMixedBilinearForm Gform, Dform;   ///< Divergence and Gradient forms
   mutable ParNonlinearForm Nform; ///< Nonlinear form for the Navier-Stokes equations
   mutable ParNonlinearForm Nform_e; ///< Gradient for nonlinear form for the Navier-Stokes equations

   /// Mass matrix and implicit solver
   CGSolver M_solver;
   FGMRESSolver linear_solver;
   NewtonSolver newton_solver;

   mutable OperatorHandle Mumat, Mpmat;
   mutable OperatorHandle Kumat, Dmat, Gmat;
   mutable OperatorHandle Kumat_e, Dmat_e, Gmat_e, Mpmat_e;

   BlockOperator *implicit_op   = nullptr; ///< Block operator for the Navier-Stokes system
   BlockOperator *implicit_grad = nullptr; ///< Block linear operator for the Navier-Stokes gradient
   NewtonResidual *newton_residual = nullptr; ///< Newton residual operator

   /// Mass matrix preconditioner
   HypreSmoother M_prec;
   mutable BlockLowerTriangularPreconditioner *pc = nullptr;
   mutable Solver *Nu_pc = nullptr;
   mutable Vector z, pz, uz;

   ConstantCoefficient zero, one; ///< Zero coefficient for boundary conditions
   ConstantCoefficient inv_dtc;

   IntegrationRules intrules;
   IntegrationRule ir, ir_nl;

   DeviatoricStressCoefficient stress_coeff;
   ParGridFunction stress_gf;
   VectorCoefficient *ale_velocity; ///< ALE velocity for moving mesh problems

   real_t current_dt = -1.0; ///< Current time in the simulation
   bool updated = false;

public:
   NavierStokes(ParFiniteElementSpace &u_fes_,
                ParFiniteElementSpace &p_fes_,
                Array<int> u_ess_attr,
                Array<int> p_ess_attr,
                real_t density_ = 1.0,
                real_t viscosity_ = 1.0,
                real_t compressibility_ = 1.0,
                bool scaled_pressure = true,
                VectorCoefficient *ale_velocity_ = nullptr) :
                Application(u_fes_.GetTrueVSize()+p_fes_.GetTrueVSize()),
                mesh(*p_fes_.GetParMesh()),
                u_fes(u_fes_), p_fes(p_fes_),
                u_ess_attr(u_ess_attr), p_ess_attr(p_ess_attr),
                viscosity(viscosity_), density(density_),
                inv_density(scaled_pressure ? 1.0 : 1.0/density_),
                compressibility(compressibility_),
                u_gf(&u_fes), p_gf(&p_fes),
                u_gf_bc(&u_fes), p_gf_bc(&p_fes),
                Mpform(&p_fes), Muform(&u_fes),
                Kuform(&u_fes), Gform(&p_fes, &u_fes),
                Dform(&u_fes, &p_fes),
                Nform(&u_fes), Nform_e(&u_fes),
                linear_solver(mesh.GetComm()),
                newton_solver(mesh.GetComm()),
                zero(0.0), one(1.0), inv_dtc(1.0),
                stress_coeff(&p_gf, &u_gf, &viscosity, &density, scaled_pressure),
                stress_gf(&u_fes), ale_velocity(ale_velocity_)
   {
        u_fes.GetEssentialTrueDofs(u_ess_attr, u_ess_tdofs);
        p_fes.GetEssentialTrueDofs(p_ess_attr, p_ess_tdofs);

        u_gf = 0.0;
        p_gf = 0.0;
        stress_gf = 0.0;
        u_gf_bc   = 0.0;
        p_gf_bc   = 0.0;

        // Setup field collection for output and transfer
        field_collection.SetName("Navier-Stokes");
        field_collection.AddField("Velocity", &u_gf);
        field_collection.AddField("Pressure", &p_gf);
        field_collection.AddSourceField("Stress", &stress_gf);
        field_collection.AddField("Velocity_BC", &u_gf_bc);
        field_collection.AddField("Pressure_BC", &p_gf_bc);

        offsets = Array<int>({0, u_fes.GetTrueVSize(), p_fes.GetTrueVSize()});
        offsets.PartialSum();

        auto geom_type = u_fes.GetFE(0)->GetGeomType();
        ir    = intrules.Get(geom_type, (int)(2*(u_fes.GetOrder(0)+1) - 3));
        ir_nl = intrules.Get(geom_type, (int)(ceil(1.5 * 2*(u_fes.GetOrder(0)+1) - 3)));

        Mpform.AddDomainIntegrator(new MassIntegrator(compressibility,&ir));
        Muform.AddDomainIntegrator(new VectorMassIntegrator(&ir));
        // Kuform.AddDomainIntegrator(new VectorDiffusionIntegrator(viscosity, &ir));
        Gform.AddDomainIntegrator(new GradientIntegrator(inv_density, &ir));
        Dform.AddDomainIntegrator(new VectorDivergenceIntegrator(&ir));
        Nform.AddDomainIntegrator(new VectorConvectionNLFIntegrator(one, *ale_velocity, &ir_nl));
        Nform.AddDomainIntegrator(new VectorDiffusionIntegrator(viscosity, &ir));

        Nform_e.AddDomainIntegrator(new VectorConvectionNLFIntegrator(one, *ale_velocity, &ir_nl));
        Nform_e.AddDomainIntegrator(new VectorDiffusionIntegrator(viscosity, &ir));
        Nform_e.AddDomainIntegrator(new VectorMassIntegrator(inv_dtc,&ir));

        // Muform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
        Mpform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
        Gform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
        Dform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
        Nform.SetAssemblyLevel(AssemblyLevel::PARTIAL);

        Assemble();
        BuildSolvers();
   }

   void Assemble() override
   {
        AssembleLinearForms();
        AssembleNonlinearForm();
        AssembleBilinearForms();
   }

   void AssembleLinearForms() {}

   void AssembleBilinearForms()
   {
        Mpform.Assemble();
        Muform.Assemble();
        // Kuform.Assemble();
        Dform.Assemble();
        Gform.Assemble();

        // Muform.FormSystemMatrix(u_ess_tdofs, Mumat);
        Mpform.FormSystemMatrix(p_ess_tdofs, Mpmat_e);
        Gform.FormRectangularSystemMatrix(p_ess_tdofs, u_ess_tdofs, Gmat_e);
        Dform.FormRectangularSystemMatrix(u_ess_tdofs, p_ess_tdofs, Dmat_e);

        Array<int> empty;
        Mpform.FormSystemMatrix(empty, Mpmat);
        Muform.FormSystemMatrix(empty, Mumat);
        Gform.FormRectangularSystemMatrix(empty, empty, Gmat);
        Dform.FormRectangularSystemMatrix(empty, empty, Dmat);

        if(!implicit_op) implicit_op = new BlockOperator(offsets);
        implicit_op->SetBlock(0, 0, &Nform);
        implicit_op->SetBlock(0, 1, Gmat.Ptr());
        implicit_op->SetBlock(1, 0, Dmat.Ptr());

        if(!implicit_grad) implicit_grad = new BlockOperator(offsets);
        implicit_grad->SetBlock(0, 1, Gmat_e.Ptr());
        implicit_grad->SetBlock(1, 0, Dmat_e.Ptr());
        implicit_grad->SetBlock(1, 1, Mpmat_e.Ptr());
   }

   void AssembleNonlinearForm()
   {
        Nform.Setup();
        Nform_e.SetEssentialTrueDofs(u_ess_tdofs);
        Nform_e.Setup();
   }

   void Update() override
   {
        u_fes.Update();
        p_fes.Update();

        u_gf.Update();
        p_gf.Update();
        stress_gf.Update();
        p_gf_bc.Update();
        u_gf_bc.Update();

        Mpform.Update();
        Muform.Update();
        // Kuform.Update();
        Dform.Update();
        Gform.Update();
        Nform.Update();
        Nform_e.Update();

        Assemble();
        updated = true;
   }

   void BuildSolvers()
   {
        M_solver.iterative_mode = false;
        M_solver.SetRelTol(1e-8);
        M_solver.SetMaxIter(100);
        M_prec.SetType(HypreSmoother::Jacobi);
        M_solver.SetPreconditioner(M_prec);
        M_solver.SetOperator(*Mumat.Ptr());

        linear_solver.iterative_mode = false;
        linear_solver.SetRelTol(1e-4);
        linear_solver.SetAbsTol(1e-4);
        linear_solver.SetMaxIter(500);
        linear_solver.SetKDim(300);
        linear_solver.SetPrintLevel(0);
        // pc = new BlockLowerTriangularPreconditioner(offsets);
        // pc->SetBlock(1, 0, Dmat_e.Ptr());
        // linear_solver.SetPreconditioner(*pc);

        if(newton_residual) delete newton_residual;
        newton_residual = new NewtonResidual(this);
        newton_solver.iterative_mode = true;
        newton_solver.SetRelTol(0.0);
        newton_solver.SetAbsTol(1e-4);
        newton_solver.SetMaxIter(30);
        newton_solver.SetPrintLevel(0);
        newton_solver.SetOperator(*newton_residual);
        newton_solver.SetPreconditioner(linear_solver);
   }

   void UpdatePreconditioner() const
   {
      if(Nu_pc) delete Nu_pc;

      auto amg = new HypreBoomerAMG;
      HYPRE_BoomerAMGSetSmoothType(*amg, 5);
      amg->SetOperator(*static_cast<HypreParMatrix*>(&implicit_grad->GetBlock(0, 0)));
      amg->SetSystemsOptions(2, true);
      amg->SetPrintLevel(0);
      Nu_pc = amg;
      pc->SetBlock(0, 0, Nu_pc);
   }

   Operator& GetGradient(const Vector &x) const override
   {
        BlockVector xb(x.GetData(), offsets);
        implicit_grad->SetBlock(0, 0, &Nform_e.GetGradient(xb.GetBlock(0)));
        implicit_grad->SetBlockCoef(0,0,current_dt);
        implicit_grad->SetBlockCoef(0,1,current_dt);
        implicit_grad->SetBlockCoef(1,0,current_dt);

        if(pc) UpdatePreconditioner();

        return *implicit_grad;
   }

   void Mult(const Vector &x, Vector &k) const override
   {
        BlockVector xb(x.GetData(), offsets);
        BlockVector kb(k.GetData(), offsets);

        implicit_op->Mult(xb, kb);

        M_solver.Mult(kb.GetBlock(0), z);
        kb.GetBlock(0) = z;

        kb.GetBlock(0).SetSubVector(u_ess_tdofs, 0.0);
        kb.GetBlock(1).SetSubVector(p_ess_tdofs, 0.0);
   }

   void ImplicitSolve(const real_t dt, const Vector &u, Vector &k) override
   {
        BlockVector ub(u.GetData(), offsets);
        BlockVector kb(k.GetData(), offsets);

        if((current_dt != dt) || updated)
        {
            inv_dtc.constant = 1.0/dt;
            current_dt = dt;
            AssembleNonlinearForm();
            updated = false;
        }

        if(IsCoupled())
        {   // Enforce velocity BCs through initial guess
            Vector &ku = kb.GetBlock(0);
            u_gf_bc.GetTrueDofs(z);
            for (int i = 0; i < u_ess_tdofs.Size(); i++)
            {
                int idx = u_ess_tdofs[i];
                ku(idx) = z(idx);
            }
        }
        else
        {
            kb.GetBlock(0) = 0.0;
            kb.GetBlock(1) = 0.0;
        }

        Vector zero_vec;
        newton_residual->SetTimeStep(dt);
        newton_residual->SetState(&u);
        newton_solver.Mult(zero_vec, k); // Solve the nonlinear system
   }

   // Computes residual of the Navier-Stokes equations
   void ImplicitMult(const Vector &x, const Vector &k, Vector &v ) const override
   {
        BlockVector xb(x.GetData(), offsets);
        BlockVector kb(k.GetData(), offsets);
        BlockVector vb(v.GetData(), offsets);

        // Apply the implicit operator
        implicit_op->Mult(xb, vb); // v = A(x)
        Mumat->AddMult(kb.GetBlock(0), vb.GetBlock(0), 1.0); // v = A(x) + M*k
        Mpmat->AddMult(kb.GetBlock(1), vb.GetBlock(1), 1.0); // v = A(x) + M*k

        vb.GetBlock(0).SetSubVector(u_ess_tdofs, 0.0);
        vb.GetBlock(1).SetSubVector(p_ess_tdofs, 0.0);
   }

    void PreProcess(Vector &x) override {}

    void PostProcess(Vector &x) override {}

    void Transfer(const Vector &x) override
    {
        BlockVector xb(x.GetData(), offsets);
        u_gf.SetFromTrueDofs(xb.GetBlock(0));
        p_gf.SetFromTrueDofs(xb.GetBlock(1));
        stress_gf.ProjectBdrCoefficient(stress_coeff,u_ess_attr);
        field_collection.Transfer("Stress");
    }

    void Transfer(const Vector &u, const Vector &k, real_t dt = 0.0) override
    {
        BlockVector ub(u.GetData(), offsets);
        BlockVector kb(k.GetData(), offsets);

        uz.SetSize(u_fes.GetTrueVSize());
        pz.SetSize(p_fes.GetTrueVSize());

        // compute stage-updated velocity and pressure
        add(1.0,ub.GetBlock(0), dt, kb.GetBlock(0), uz);
        add(1.0,ub.GetBlock(1), dt, kb.GetBlock(1), pz);

        u_gf.SetFromTrueDofs(uz);
        p_gf.SetFromTrueDofs(pz);
        stress_gf.ProjectBdrCoefficient(stress_coeff,u_ess_attr);
        field_collection.Transfer("Stress");
    }

   ~NavierStokes()
   {
        if(implicit_op) delete implicit_op;
        if(implicit_grad) delete implicit_grad;
        if(newton_residual) delete newton_residual;
        if(Nu_pc) delete Nu_pc;
        if(pc) delete pc;
    }
};



#endif
