#include "multiapp.hpp"


namespace mfem
{


CoupledApplication::~CoupledApplication()
{
    if(solver && own_solver) delete solver;
    if(coupling_op && own_coupling_op) delete coupling_op;        
}

void CoupledApplication::SetCouplingOperator(CouplingOperator* op, bool own){
    if(coupling_op && own_coupling_op) delete coupling_op;
    coupling_op = op;
    own_coupling_op = own;
}


CouplingOperator *CoupledApplication::BuildCouplingOperator(){

    if (coupling_op){
        delete coupling_op;
        coupling_op = nullptr;
    }

    if (scheme == Scheme::MONOLITHIC)
    {
        coupling_op = new MonolithicOperator(this, 1e-6); // eps = 1e-6
    }
    else if (scheme == Scheme::ADDITIVE_SCHWARZ)
    {
        coupling_op = new AdditiveSchwarzOperator(this);
    }
    else if (scheme == Scheme::ALTERNATING_SCHWARZ)
    {
        coupling_op = new AlternatingSchwarzOperator(this);
    }

    return coupling_op;
}

void CoupledApplication::Initialize(bool do_ops)
    {
        if (do_ops)
        {
            for (auto &op : operators)
            {
                op->Initialize();
            }
        }
    }


void CoupledApplication::Assemble(bool do_ops)
{
    if (do_ops)
    {
        for (auto &op : operators)
        {
            op->Assemble();
        }
    }

    BuildCouplingOperator();                
    if(coupling_op && solver) solver->SetOperator(*coupling_op);
}

void CoupledApplication::Finalize(bool do_ops)
{
    if (do_ops)
    {
        for (auto &op : operators)
        {
            op->Finalize();
        }
    }
    
    if (coupling_op)
    {
        MFEM_VERIFY(solver,"Solver to be used for coupling not defined; set solver using SetSolver(Solver*)");
        if(coupling_op->IsLinear())
        {  // Use b to store the rhs for the linear implicit solve.
            // b=0 (or unused) for nonlinear Newton solves
            b.SetSize(Width());
        }
    }
}


void CoupledApplication::PreProcess(Vector &x, bool do_ops) 
{
    if (do_ops)
    {
        BlockVector xb(x.GetData(), offsets);
        for (int i=0; i < nops; i++)
        {
            Vector &xi = xb.GetBlock(i);
            operators[i]->SetOperationID(OperationID::NONE);
            operators[i]->PreProcess(xi);
        }
    }
}


void CoupledApplication::PostProcess(Vector &x, bool do_ops) 
{
    if (do_ops)
    {
        BlockVector xb(x.GetData(), offsets);
        for (int i=0; i < nops; i++)
        {
            Vector &xi = xb.GetBlock(i);
            operators[i]->SetOperationID(OperationID::NONE);
            operators[i]->PostProcess(xi);
        }
    }
}    


void CoupledApplication::Transfer(const Vector &x)
{
    BlockVector xb(x.GetData(), offsets);
    for (int i=0; i < nops; i++)
    {
        operators[i]->Transfer(xb.GetBlock(i));
    }
}


void CoupledApplication::Mult(const Vector &x, Vector &y) const
{
    if(coupling_op && scheme != Scheme::NONE)
    {
        coupling_op->SetOperationID(OperationID::MULT);
        coupling_op->Mult(x,y);
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


void CoupledApplication::ImplicitSolve(const real_t dt, const Vector &x, Vector &k ){

    if(coupling_op && scheme != Scheme::NONE)
    {
        coupling_op->SetOperationID(OperationID::IMPLICIT_SOLVE);        
        coupling_op->SetTimeStep(dt); ///< Set the time step for the implicit operator
        coupling_op->SetState(&x); ///< Set the input vector for the implicit operator
        solver->Mult(b,k); ///< Solve the implicit system using the solver (b is unused for FPISolver)
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


void CoupledApplication::ImplicitMult(const Vector &u, const Vector &k, Vector &v) const 
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


void CoupledApplication::ExplicitMult(const Vector &u, Vector &v) const
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


void CoupledApplication::Step(Vector &x, real_t &t, real_t &dt)
{
    if(coupling_op && scheme != Scheme::NONE)
    {
        coupling_op->SetOperationID(OperationID::STEP);
        coupling_op->Step(x,t,dt);
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


void AdditiveSchwarzOperator::Mult(const Vector &x, Vector &y) const
{
    if(GetOperationID() == OperationID::IMPLICIT_SOLVE)
    {// if the operation is an implicit solve (set by CoupledApplication::ImplicitSolve), 
     // then Mult was called by the Solver
        y=x; // input vector passed as initial guess for k in ImpliicitSolve
        const_cast<AdditiveSchwarzOperator*>(this)->ImplicitSolve(dt,*u,y);
        return;
    }

    int nops = coupled_operator->Size();
    const Array<int> offsets = coupled_operator->GetOffsets();

    BlockVector xb(x.GetData(), offsets);
    BlockVector yb(y.GetData(), offsets);

    for (int i=0; i < nops; i++)
    {
        Vector &xi = xb.GetBlock(i);
        Vector &yi = yb.GetBlock(i);
        auto op = coupled_operator->GetOperator(i);
        op->SetOperationID(OperationID::MULT);
        op->Mult(xi,yi);
    }

    // Transfer data. If app is steady-state and Mult solves F(x) = 0, then
    // dt_ = 0 and x is transferred to all applications. If app is explicit, time-dependent,
    // then Mult computes dxdt = F(x), updates z = xi + dt*dxdt, and transfers z to all applications.
    for (int i=0; i < nops; i++)
    {
        Vector &xi = xb.GetBlock(i);
        Vector &yi = yb.GetBlock(i);
        int isize  = xi.Size();

        z.SetSize(isize);
        add(1.0,xi,dt,yi,z); ///< Compute the solution (z = xi + dt*ki)
        auto op = coupled_operator->GetOperator(i);
        op->Transfer(z);  ///< Transfer the data to all applications
    }
}


void AdditiveSchwarzOperator::ImplicitSolve(const real_t dt, const Vector &x, Vector &k )
{
    int nops = coupled_operator->Size();
    const Array<int> offsets = coupled_operator->GetOffsets();

    BlockVector xb(x.GetData(), offsets);
    BlockVector kb(k.GetData(), offsets);

    for (int i=0; i < nops; i++)
    {
        Vector &xi = xb.GetBlock(i);
        Vector &ki = kb.GetBlock(i);
        int nxi    = xi.Size();
        auto op = coupled_operator->GetOperator(i);

        z.SetSize(nxi);
        add(1.0,xi,dt,ki,z); ///< Compute the solution (z = xi + dt*ki)
        op->Transfer(z);     ///< Transfer the data to all applications
    }

    for (int i=0; i < nops; i++)
    {
        Vector &xi = xb.GetBlock(i);
        Vector &ki = kb.GetBlock(i);

        auto op = coupled_operator->GetOperator(i);
        op->SetOperationID(OperationID::IMPLICIT_SOLVE);

        z = xi;
        op->PreProcess(z); ///< Postprocess the data for the application
        op->ImplicitSolve(dt,z,ki); ///< Solve the implicit system for the application
        op->PostProcess(ki); ///< Postprocess the data for the application        
    }
}


void AdditiveSchwarzOperator::Step(Vector &x, real_t &t_, real_t &dt_)
{
    int nops = coupled_operator->Size();
    const Array<int> offsets = coupled_operator->GetOffsets();
    
    BlockVector xb(x.GetData(), offsets);

    // Step forward all operators
    // TODO: Add time-interpolation to enable different time step for each operator; 
    //       currently, all operators are stepped forward with the same time step
    for (int i=0; i < nops; i++)
    {
        real_t t0 = t_;   ///< Store the current time
        real_t dt0 = dt_; ///< Store the current time step

        Vector &xi = xb.GetBlock(i);
        auto op = coupled_operator->GetOperator(i);
        op->SetOperationID(OperationID::STEP);
        
        op->PreProcess(xi); ///< Preprocess the data for the application
        op->Step(xi,t0,dt0); ///< Advance the time step for application
        op->PostProcess(xi); ///< Postprocess the data for the application
    }

    // Transfer the data after all applications have been stepped forward
    for (int i=0; i < nops; i++)
    {
        Vector &xi = xb.GetBlock(i);
        auto op = coupled_operator->GetOperator(i);
        op->Transfer(xi);
    }

    t_ += dt_; ///< Update the time after all applications have been stepped forward
               ///< NOTE: does not work for adaptive time-stepping
}


void AlternatingSchwarzOperator::Mult(const Vector &x, Vector &y) const
{
    int nops = coupled_operator->Size();
    const Array<int> offsets = coupled_operator->GetOffsets();

    BlockVector xb(x.GetData(), offsets);
    BlockVector yb(y.GetData(), offsets);

    for (int i=0; i < nops; i++)
    {
        Vector &xi = xb.GetBlock(i);
        Vector &yi = yb.GetBlock(i);
        auto op = coupled_operator->GetOperator(i);
        op->SetOperationID(OperationID::MULT);
        op->Mult(xi,yi);

        int isize  = xi.Size();
        z.SetSize(isize);
        add(1.0,xi,dt,yi,z); ///< Compute the solution (z = xi + dt*ki)                
        op->Transfer(z);  ///< Transfer the data to all applications
    }
}


void AlternatingSchwarzOperator::ImplicitSolve(const real_t dt, const Vector &x, Vector &k )
{
    int nops = coupled_operator->Size();
    const Array<int> offsets = coupled_operator->GetOffsets();

    BlockVector xb(x.GetData(), offsets);
    BlockVector kb(k.GetData(), offsets);
    
    for (int i=0; i < nops; i++)
    {
        Vector &xi = xb.GetBlock(i);
        Vector &ki = kb.GetBlock(i);
        int isize  = xi.Size();

        z.SetSize(isize);
        add(1.0,xi,dt,ki,z); ///< Compute the solution (z = xi + dt*ki)

        auto op = coupled_operator->GetOperator(i);     
        op->SetOperationID(OperationID::IMPLICIT_SOLVE);

        op->Transfer(z);  ///< Transfer the data to all applications
        op->ImplicitSolve(dt,xi,ki); ///< Solve the implicit system for the application
    }
}


void AlternatingSchwarzOperator::Step(Vector &x, real_t &t_, real_t &dt_)
{
    int nops = coupled_operator->Size();
    const Array<int> offsets = coupled_operator->GetOffsets();

    BlockVector xb(x.GetData(), offsets);

    for (int i=0; i < nops; i++)
    {
        real_t t0 = t_;   ///< Store the current time
        real_t dt0 = dt_; ///< Store the current time step

        Vector &xi = xb.GetBlock(i);
        auto op = coupled_operator->GetOperator(i);
        op->SetOperationID(OperationID::STEP);

        op->PreProcess(xi); ///< Preprocess the data for the application
        op->Step(xi,t0,dt0); ///< Advance the time step for application
        op->PostProcess(xi); ///< Postprocess the data for the application
        op->Transfer(xi);    ///< Transfer the data to all applications
    }

    t_ += dt_; ///< Update the time after all applications have been stepped forward
               ///< NOTE: does not work for adaptive time-stepping
}


void MonolithicOperator::Mult(const Vector &k, Vector &y) const {
    add(du,*u,dt,k,upk); // upk = u + dt*k
    coupled_operator->Transfer(upk);
    coupled_operator->ImplicitMult(upk,k,y); //y = f(upk,k,t)
}


Operator& MonolithicOperator::GetGradient(const Vector &k) const {
    grad.Update(k);
    return const_cast<future::FDJacobian&>(grad);
}





}