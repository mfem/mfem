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

#include "multiapp.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

FieldTransfer* FieldTransfer::Select(ParFiniteElementSpace *src,
                                     ParFiniteElementSpace *tar, 
                                     Type type)
{
    switch (type)
    {
        case Type::NATIVE:
            return new NativeTransfer(src, tar);
        case Type::GSLIB:
            return new GSLibTransfer(src, tar);
        default:
            MFEM_ABORT("Unknown FieldTransfer scheme: " << static_cast<int>(type));
    }
}

OperatorCoupler* OperatorCoupler::Select(CoupledOperator *op, 
                                         Scheme scheme)
{
    switch (scheme)
    {
        case Scheme::MONOLITHIC:
        {
            real_t fd_eps = 1e-6;
            return new JacobianFreeFullCoupler(op, fd_eps);
        }
        case Scheme::ADDITIVE_SCHWARZ:
            return new AdditiveSchwarzCoupler(op);
        case Scheme::ALTERNATING_SCHWARZ:
            return new AlternatingSchwarzCoupler(op);
        default:
            MFEM_ABORT("Unknown coupling scheme: " << static_cast<int>(scheme));
    }
}

CoupledOperator::~CoupledOperator()
{
    if(solver && own_solver) delete solver;
    if(op_coupler && own_op_coupler) delete op_coupler;

    for(int i=0; i < nops; i++)
    {
        if(operators_owned[i] && operators[i]) delete operators[i];
    }
}

void CoupledOperator::SetOperatorCoupler(OperatorCoupler* op, bool own)
{
    if(op_coupler && own_op_coupler) delete op_coupler;
    op_coupler = op;
    own_op_coupler = own;
    coupler_type = op_coupler->GetType();
}

void CoupledOperator::Initialize(bool do_ops)
{
    if (do_ops)
    {
        for (auto &op : operators)
        {
            op->Initialize();
        }
    }
}

void CoupledOperator::Assemble(bool do_ops)
{
    if (do_ops)
    {
        for (auto &op : operators)
        {
            op->Assemble();
        }
    }

    // Check block offsets against operator size
    Array<int> true_offsets(Size()+1);
    bool offset_consistent = true;
    true_offsets = 0;
    int max_size = 0;

    for (int i=0; i < nops; i++)
    {
        auto op = GetOperator(i);
        int block_size = offsets[i+1]-offsets[i];
        true_offsets[i+1] = true_offsets[i] + op->Width();
        if (block_size != op->Width())
        {
            offset_consistent = false;
        }
    }
    if (!offset_consistent)
    {
        MFEM_WARNING("Block offsets inconsistent with operator sizes."
                        "Using default offsets.");
        offsets = true_offsets;
        max_op_size = max_size;
    }

    if(op_coupler && own_op_coupler) delete op_coupler;
    op_coupler = OperatorCoupler::Select(this, coupler_type);
    if(solver) solver->SetOperator(*op_coupler);
}

void CoupledOperator::Finalize(bool do_ops)
{
    if (do_ops)
    {
        for (auto &op : operators)
        {
            op->Finalize();
        }
    }
}

void CoupledOperator::PreProcess(Vector &x, bool do_ops) 
{
    if (do_ops)
    {
        BlockVector xb(x.GetData(), offsets);
        for (int i=0; i < nops; i++)
        {
            Vector &xi = xb.GetBlock(i);
            operators[i]->PreProcess(xi);
        }
    }
}

void CoupledOperator::PostProcess(Vector &x, bool do_ops) 
{
    if (do_ops)
    {
        BlockVector xb(x.GetData(), offsets);
        for (int i=0; i < nops; i++)
        {
            Vector &xi = xb.GetBlock(i);
            operators[i]->PostProcess(xi);
        }
    }
}

void CoupledOperator::SetOperationID(OperationID id, bool do_ops)
{
    Application::SetOperationID(id);
    if (do_ops)
    {
        for (auto &op : operators)
        {
            op->SetOperationID(id);
        }
    }
}

void CoupledOperator::SetTime(const real_t t_) 
{
    TimeDependentOperator::SetTime(t_);
    if(op_coupler) op_coupler->SetTime(t_);
    for (auto &op : operators)
    {
        op->SetTime(t_);
    }
}

void CoupledOperator::Transfer(const Vector &x)
{
    BlockVector xb(x.GetData(), offsets);
    for (int i=0; i < nops; i++)
    {
        operators[i]->Transfer(xb.GetBlock(i));
    }
}

void CoupledOperator::Transfer(const Vector &u, const Vector &k, real_t dt)
{
    BlockVector ub(u.GetData(), offsets);
    BlockVector kb(k.GetData(), offsets);
    for (int i=0; i < nops; i++)
    {
        operators[i]->Transfer(ub.GetBlock(i), kb.GetBlock(i), dt);
    }
}

void CoupledOperator::Mult(const Vector &x, Vector &y) const
{
    if(op_coupler && coupler_type != Scheme::NONE)
    {
        if(solver) {
            op_coupler->SetOperationID(OperationID::MULT);
            op_coupler->SetInput(&x);
            solver->Mult(b,y);
        }
        else {
            op_coupler->Mult(x,y);
        }
    }
    else
    {
        BlockVector xb(x.GetData(), offsets);
        BlockVector yb(y.GetData(), offsets);

        for (int i=0; i < nops; i++)
        {
            operators[i]->SetOperationID(OperationID::MULT);
            operators[i]->Mult(xb.GetBlock(i), yb.GetBlock(i));
        }
    }
}

void CoupledOperator::ImplicitSolve(const real_t dt, const Vector &x, Vector &k ){

    if(op_coupler && coupler_type != Scheme::NONE)
    {
        if(solver) {
            op_coupler->SetOperationID(OperationID::IMPLICIT_SOLVE); ///< OperatorCoupler::Mult() -> OperatorCoupler::ImplicitSolve() 
            op_coupler->SetTimeStep(dt);
            op_coupler->SetInput(&x);
            solver->Mult(b,k);
        }
        else {
            op_coupler->ImplicitSolve(dt,x,k);
        }                
    }
    else
    {
        BlockVector xb(x.GetData(), offsets);
        BlockVector kb(k.GetData(), offsets);

        for (int i=0; i < nops; i++)
        {
            Vector &xi = xb.GetBlock(i);
            Vector &ki = kb.GetBlock(i);
            operators[i]->SetOperationID(OperationID::IMPLICIT_SOLVE);
            operators[i]->ImplicitSolve(dt,xi,ki); ///< Solve the implicit system for the application
        }
    }
}

void CoupledOperator::Step(Vector &x, real_t &t, real_t &dt)
{
    if(op_coupler && coupler_type != Scheme::NONE)
    {
        if(solver) {
            op_coupler->SetOperationID(OperationID::STEP); ///< OperatorCoupler::Mult() -> OperatorCoupler::Mult() 
            op_coupler->SetTimeStep(dt); ///< Set the time step for the ODE Solver
            op_coupler->SetTime(t);
            op_coupler->SetInput(&x);
            solver->Mult(b,x);
        }
        else {
            op_coupler->Step(x,t,dt);
        }                
    }
    else
    {
        BlockVector xb(x.GetData(), offsets);
        for (int i=0; i < nops; i++)
        {
            real_t t0 = t;   ///< Store the current time
            real_t dt0 = dt; ///< Store the current time step
            Vector &xi = xb.GetBlock(i);
            operators[i]->SetOperationID(OperationID::STEP);
            operators[i]->Step(xi,t0,dt0); ///< Advance the time step for application
        }
        t += dt; ///< Update the time after all applications have been stepped forward
                 ///< NOTE: does not work for adaptive time-stepping
    }
}

void CoupledOperator::ImplicitMult(const Vector &u, const Vector &k, Vector &v) const 
{
    BlockVector ub(u.GetData(), offsets);
    BlockVector kb(k.GetData(), offsets);
    BlockVector vb(v.GetData(), offsets);

    for (int i=0; i < nops; i++)
    {
        Vector &ui = ub.GetBlock(i);
        Vector &ki = kb.GetBlock(i);
        Vector &vi = vb.GetBlock(i);
        operators[i]->SetOperationID(OperationID::IMPLICIT_MULT);
        operators[i]->ImplicitMult(ui,ki,vi); ///< Solve the implicit system for the application
    }
}

void CoupledOperator::ExplicitMult(const Vector &u, Vector &v) const
{
    BlockVector ub(u.GetData(), offsets);
    BlockVector vb(v.GetData(), offsets);

    for (int i=0; i < nops; i++)
    {
        Vector &ui = ub.GetBlock(i);
        Vector &vi = vb.GetBlock(i);
        operators[i]->SetOperationID(OperationID::EXPLICIT_MULT);
        operators[i]->ExplicitMult(ui,vi); ///< Solve the implicit system for the application
    }
}




// AdditiveSchwarzCoupler methods
void AdditiveSchwarzCoupler::Mult(const Vector &x, Vector &y) const
{
    /// This is use to call either ImplicitSolve or Step when Solver::Mult()
    /// calls Solver.Operator::Mult()
    if(GetOperationID() == OperationID::IMPLICIT_SOLVE)
    {
        y=x; // input vector passed as initial guess for k in ImpliicitSolve
        ImplicitSolve(timestep,*input,y);
        return;
    }
    else if(GetOperationID() == OperationID::STEP)
    {
        y=x; // input vector passed as initial condition in Step
        real_t t_ = t, dt = timestep; 
        Step(y,t_,dt);
        return;
    }    

    int nops = coupled_op->Size();
    const Array<int> offsets = coupled_op->GetBlockOffsets();

    BlockVector xb(x.GetData(), offsets);
    BlockVector yb(y.GetData(), offsets);

    for (int i=0; i < nops; i++)
    {
        Vector &xi = xb.GetBlock(i);
        Vector &yi = yb.GetBlock(i);
        auto op = coupled_op->GetOperator(i);
        op->Transfer(xi,yi,0.0);
    }

    for (int i=0; i < nops; i++)
    {
        Vector &xi = xb.GetBlock(i);
        Vector &yi = yb.GetBlock(i);
        auto op = coupled_op->GetOperator(i);
        op->SetOperationID(OperationID::MULT);
        
        op->PreProcess(xi); ///< Postprocess the data for the application
        op->Mult(xi,yi);
        op->PostProcess(yi); ///< Postprocess the data for the application
    }
}

void AdditiveSchwarzCoupler::ImplicitSolve(const real_t dt, const Vector &x, Vector &k ) const
{
    int nops = coupled_op->Size();
    const Array<int> offsets = coupled_op->GetBlockOffsets();

    BlockVector xb(x.GetData(), offsets);
    BlockVector kb(k.GetData(), offsets);

    for (int i=0; i < nops; i++)
    {
        Vector &xi = xb.GetBlock(i);
        Vector &ki = kb.GetBlock(i);
        auto op = coupled_op->GetOperator(i);
        op->Transfer(xi,ki,dt);
    }

    for (int i=0; i < nops; i++)
    {
        Vector &xi = xb.GetBlock(i);
        Vector &ki = kb.GetBlock(i);

        auto op = coupled_op->GetOperator(i);
        op->SetOperationID(OperationID::IMPLICIT_SOLVE);

        op->PreProcess(xi);
        op->ImplicitSolve(dt,xi,ki);
        op->PostProcess(ki);
    }
}

void AdditiveSchwarzCoupler::Step(Vector &x, real_t &t_, real_t &dt) const
{
    int nops = coupled_op->Size();
    const Array<int> offsets = coupled_op->GetBlockOffsets();

    BlockVector xb(x.GetData(), offsets);

    for (int i=0; i < nops; i++)
    {
        Vector &xi = xb.GetBlock(i);
        auto op = coupled_op->GetOperator(i);
        op->Transfer(xi);
    }

    // TODO: Add time-interpolation to enable different time step for each operator; 
    //       currently, all operators are stepped forward with the same time step
    for (int i=0; i < nops; i++)
    {
        real_t ti = t_;  ///< Store the current time
        real_t dti = dt; ///< Store the current time step

        Vector &xi = xb.GetBlock(i);
        auto op = coupled_op->GetOperator(i);
        op->SetOperationID(OperationID::STEP);
        
        op->PreProcess(xi);
        op->Step(xi,ti,dti);
        op->PostProcess(xi);
    }

    t_ += dt; ///< Update the time after all applications have been stepped forward
               ///< NOTE: does not work for adaptive time-stepping
}


// AlternatingSchwarzCoupler methods
void AlternatingSchwarzCoupler::Mult(const Vector &x, Vector &y) const
{
    /// This is use to call either ImplicitSolve or Step when Solver::Mult()
    /// calls Solver.Operator::Mult()
    if(GetOperationID() == OperationID::IMPLICIT_SOLVE)
    {
        y=x; // input vector passed as initial guess for k in ImpliicitSolve
        ImplicitSolve(timestep,*input,y);
        return;
    }
    else if(GetOperationID() == OperationID::STEP)
    {
        y=x; // input vector passed as initial condition in Step
        real_t t_ = t, dt = timestep; 
        Step(y,t_,dt);
        return;
    }    


    int nops = coupled_op->Size();
    const Array<int> offsets = coupled_op->GetBlockOffsets();

    BlockVector xb(x.GetData(), offsets);
    BlockVector yb(y.GetData(), offsets);

    for (int i=0; i < nops; i++)
    {
        Vector &xi = xb.GetBlock(i);
        Vector &yi = yb.GetBlock(i);
        auto op = coupled_op->GetOperator(i);
        op->SetOperationID(OperationID::MULT);

        op->PreProcess(xi);
        op->Mult(xi,yi);
        op->PostProcess(yi);
        op->Transfer(xi,yi,0.0);
    }
}

void AlternatingSchwarzCoupler::ImplicitSolve(const real_t dt, const Vector &x, Vector &k ) const
{
    int nops = coupled_op->Size();
    const Array<int> offsets = coupled_op->GetBlockOffsets();

    BlockVector xb(x.GetData(), offsets);
    BlockVector kb(k.GetData(), offsets);

    for (int i=0; i < nops; i++)
    {
        Vector &xi = xb.GetBlock(i);
        Vector &ki = kb.GetBlock(i);

        auto op = coupled_op->GetOperator(i);
        op->SetOperationID(OperationID::IMPLICIT_SOLVE);

        op->PreProcess(xi);
        op->ImplicitSolve(dt,xi,ki);
        op->PostProcess(ki);
        op->Transfer(xi,ki,dt);
    }
}

void AlternatingSchwarzCoupler::Step(Vector &x, real_t &t_, real_t &dt) const
{
    int nops = coupled_op->Size();
    const Array<int> offsets = coupled_op->GetBlockOffsets();

    BlockVector xb(x.GetData(), offsets);

    for (int i=0; i < nops; i++)
    {
        real_t ti = t_;   ///< Store the current time
        real_t dti = dt;  ///< Store the current time step

        Vector &xi = xb.GetBlock(i);
        auto op = coupled_op->GetOperator(i);
        op->SetOperationID(OperationID::STEP);

        op->PreProcess(xi);
        op->Step(xi,ti,dti);
        op->PostProcess(xi);
        op->Transfer(xi);
    }

    t_ += dt; ///< Update the time after all applications have been stepped forward
               ///< NOTE: does not work for adaptive time-stepping
}



// JacobianFreeFullCoupler methods
void JacobianFreeFullCoupler::Mult(const Vector &k, Vector &y) const
{
    add(1.0,*input,timestep,k,u); // u = u + dt*k
    coupled_op->Transfer(u);
    coupled_op->ImplicitMult(u,k,y); //compute residual y = f(u,k,t)
}

Operator& JacobianFreeFullCoupler::GetGradient(const Vector &k) const
{
    grad.Update(k);
    return const_cast<future::FDJacobian&>(grad);
}

}

#endif // MFEM_USE_MPI