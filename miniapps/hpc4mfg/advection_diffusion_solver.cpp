#include "advection_diffusion_solver.hpp"
#include "NLGLSIntegrator.hpp"

namespace mfem {

// double analytic_T(const Vector &x)
// {
//    double T = std::sin( x(0)*x(1) );
//    //double T = x(0)*x(1)*x(0);
//    return T;
// }

// double analytic_solution(const Vector &x)
// {
//    double s = x(1)*std::cos( x(0)*x(1) ) + x(0)*std::cos( x(0)*x(1) ) + x(1)*x(1)*std::sin( x(0)*x(1) ) + x(0)*x(0)*std::sin( x(0)*x(1) );

//    // double s = -x(1)*x(1)*std::sin( x(0)*x(1) ) - x(0)*x(0)*std::sin( x(0)*x(1) );
//    //double s = 2*x(0)*x(1)+x(0)*x(0)-2*x(1);
//    return s;
// }


void Advection_Diffusion_Solver::FSolve()
{
    bool pa = false;
    bool ea = false;
    bool fa = false;

        
    //    Define the solution vector x as a finite element grid function
    //    corresponding to fespace. Initialize x with initial guess of zero.
    ParGridFunction x(fes);
    x = 0.0;

    FunctionCoefficient T0(analytic_T);
    FunctionCoefficient s0(mfem::analytic_solution);
    solgf.ProjectCoefficient(T0);
    sol = solgf;

    ess_tdofv.DeleteAll();

    // set the boundary conditions
    {
        for(auto it=bcc.begin();it!=bcc.end();it++){
            mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
            ess_bdr=0;
            ess_bdr[it->first -1]=1;
            mfem::Array<int> ess_tdof_list;
            fes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
            ess_tdofv.Append(ess_tdof_list);
            //solgf.ProjectBdrCoefficient(*(it->second),ess_bdr);
        }

        // copy BC values from the grid function to the solution vector
        // {
        //     solgf.GetTrueDofs(rhs);
        //     for(int ii=0;ii<ess_tdofv.Size();ii++)
        //     {
        //         sol[ess_tdofv[ii]]=rhs[ess_tdofv[ii]];
        //     }
        // }
    }

    //the BC are setup in the solution vector sol

    std::cout<<"BC dofs size="<<ess_tdofv.Size()<<std::endl;

    // FIXME
    ParGridFunction tU_GF(fes_u);
    tU_GF = 1.0;

    VectorGridFunctionCoefficient velocity(&tU_GF);

//        ParGridFunction *u = new ParGridFunction(fes);
//    u->ProjectCoefficient(u0);
//    HypreParVector *U = u->GetTrueDofs();

    double sigma = -1.0;       //"One of the three DG penalty parameters, typically +1/-1."
    double kappa = -1.0;       //"One of the three DG penalty parameters, should be positive. Negative values are replaced with (order+1)^2."
    ConstantCoefficient one(1.0);
    ConstantCoefficient zero(0.0);

    constexpr double alpha = 1.0;

    //    Set up the linear form b(.) which corresponds to the right-hand side of
    //    the FEM linear system.
    if(b==nullptr)
    {
        b = new ParLinearForm(fes);

        //b->AddDomainIntegrator(new NLGLSIntegrator( materials[0], &tU_GF, &s0));
        b->AddDomainIntegrator(new DomainLFIntegrator(s0));
        //b->AddBdrFaceIntegrator(
            //new DGDirichletLFIntegrator(zero, one, sigma, kappa));
        b->Assemble();
    }


    
    //    Set up the bilinear form a(.,.) on the finite element space
    if(a==nullptr)
    {
        a = new ParBilinearForm(fes);

        if (pa)
        {
            a->SetAssemblyLevel(AssemblyLevel::PARTIAL);
        }
        else if (ea)
        {
            a->SetAssemblyLevel(AssemblyLevel::ELEMENT);
        }
        else if (fa)
        {
            a->SetAssemblyLevel(AssemblyLevel::FULL);
        }        

        // add diffusion integrators
        a->AddDomainIntegrator(new DiffusionIntegrator(one));
        //a->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
        //a->AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));                   // weak BC

        // add advection integrators
        a->AddDomainIntegrator(new ConvectionIntegrator(velocity, alpha));
        //a->AddInteriorFaceIntegrator( new NonconservativeDGTraceIntegrator(velocity, alpha));
        //a->AddBdrFaceIntegrator( new NonconservativeDGTraceIntegrator(velocity, alpha));         // weak BC

        // add gls integrator
        a->AddDomainIntegrator(new NLGLSIntegrator(materials[0], &tU_GF));

        //a->SetEssentialTrueDofs(ess_tdofv);

        a->Assemble();
        a->Finalize();

    }

    //allocate the preconditioner and the linear solver
    if(prec==nullptr){
        prec = new HypreBoomerAMG();
        prec->SetPrintLevel(print_level);
    }

    if(ls==nullptr){
        ls = new CGSolver(pmesh->GetComm());
        ls->SetAbsTol(linear_atol);
        ls->SetRelTol(linear_rtol);
        ls->SetMaxIter(linear_iter);
        ls->SetPrintLevel(print_level);
        ls->SetPreconditioner(*prec);
    }

   HypreParMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdofv, sol, *b, A, X, B);

    // HypreParMatrix *A = a->ParallelAssemble();
    // HypreParVector *B = b->ParallelAssemble();
    // HypreParVector *X = solgf.ParallelProject();



    ls->SetOperator(A);
    ls->Mult(B, X);

    solgf = X;     // copy solution

    std::cout<<" ----- end ----"<<std::endl;

}

void Advection_Diffusion_Solver::Postprocess()
{
    FunctionCoefficient s0(mfem::analytic_solution);
    FunctionCoefficient T0(analytic_T);

    ParGridFunction solExact(fes);
    solExact.ProjectCoefficient(T0);

    ParGridFunction solError(fes);
    solError = solgf;
    solError -= solExact;

   if( true )
   {
      mPvdc = new ParaViewDataCollection("AdvDiff", pmesh);
      mPvdc->SetDataFormat(VTKFormat::BINARY32);
      mPvdc->SetHighOrderOutput(true);
      //mPvdc->SetLevelsOfDetail(mCtk.order);
      mPvdc->SetCycle(0);
      mPvdc->SetTime(0.0);
      mPvdc->RegisterField("Temperature", &solgf);
      mPvdc->RegisterField("TAnalytic", &solExact);
      mPvdc->RegisterField("TError", &solError);
      mPvdc->Save();
   }
}

}
