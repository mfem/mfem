#include "advection_diffusion_solver.hpp"
#include "../navier/adv_diff_cg.hpp"
#include "NLGLSIntegrator.hpp"

#include "petsc.h"

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

//        ParGridFunction *u = new ParGridFunction(fes);
//    u->ProjectCoefficient(u0);
//    HypreParVector *U = u->GetTrueDofs();

    double sigma = -1.0;       //"One of the three DG penalty parameters, typically +1/-1."
    double kappa = -1.0;       //"One of the three DG penalty parameters, should be positive. Negative values are replaced with (order+1)^2."
    ConstantCoefficient one(1.0);
    ConstantCoefficient zero(0.0);

    constexpr double alpha = 1.0;

    //ParGridFunction avgGradTemp

    //    Set up the linear form b(.) which corresponds to the right-hand side of
    //    the FEM linear system.
    if(b==nullptr)
    {
        b = new ParLinearForm(fes);

        //b->AddDomainIntegrator(new NLGLSIntegrator( materials[0], &velocityGF, &s0));
        b->AddDomainIntegrator(new DomainLFIntegrator(s0));
        //b->AddBdrFaceIntegrator(
            //new DGDirichletLFIntegrator(zero, one, sigma, kappa));

        mfem::VectorCoefficient * diffVecCoeff = new Advection_Diffusion_Solver::RHSDiffCoeff( pmesh, avgGradTemp_, materials[0],  -1.0 );
        mfem::Coefficient * advCoeff = new Advection_Diffusion_Solver::RHSAdvCoeff( pmesh, vel_, avgGradTemp_,  -1.0 );

        b->AddDomainIntegrator(new DomainLFGradIntegrator(*diffVecCoeff));
        b->AddDomainIntegrator(new DomainLFIntegrator(*advCoeff));

        b->Assemble();

//         mfem::ParGridFunction oneGridFunction(fes);

//         oneGridFunction = 1.0;

// double tVal = b(oneGridFunction);
    
//         std::cout<<"Norm b: "<<tVal<<std::endl;

        delete diffVecCoeff;
        delete advCoeff;
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
        a->AddDomainIntegrator(new ConvectionIntegrator(*vel_, alpha));
        //a->AddInteriorFaceIntegrator( new NonconservativeDGTraceIntegrator(velocity, alpha));
        //a->AddBdrFaceIntegrator( new NonconservativeDGTraceIntegrator(velocity, alpha));         // weak BC

        // add gls integrator
        a->AddDomainIntegrator(new NLGLSIntegrator(materials[0], vel_, new NLGLSIntegrator::GLSCoefficient(pmesh, vel_,materials[0])));
        //a->AddDomainIntegrator(new AdvectionDiffusionGLSStabInt( &velocity, materials[0], 1.0));

        //a->SetEssentialTrueDofs(ess_tdofv);

        a->Assemble();
        a->Finalize();

    }

    HypreParMatrix A;
    HypreParVector B, X;

    if(true)
    {

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

        a->FormLinearSystem(ess_tdofv, sol, *b, A, X, B);

        // HypreParMatrix *A = a->ParallelAssemble();
        // HypreParVector *B = b->ParallelAssemble();
        // HypreParVector *X = solgf.ParallelProject();

        ls->SetOperator(A);
        ls->Mult(B, X);
    }
    else
    {
        if(false)
        {
         const char *petscrc_file = "";
         MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);

         PetscParMatrix A;
         a->SetOperatorType(Operator::PETSC_MATAIJ);
         a->FormLinearSystem(ess_tdofv, sol, *b, A, X, B);

        //PetscInitializeNoArguments();
        KSP tPetscKSPProblem;
        PC mpc;

        KSPCreate( PETSC_COMM_WORLD, &tPetscKSPProblem );
        KSPSetOperators( tPetscKSPProblem, A, A );

        // Build Preconditioner
        KSPGetPC( tPetscKSPProblem, &mpc );

        PCSetType( mpc, PCILU  );
        PCFactorSetDropTolerance( mpc, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT );
        PCFactorSetLevels( mpc, 0 );

        PetscInt maxits=1000;
        KSPSetTolerances( tPetscKSPProblem, 1.e-14, PETSC_DEFAULT, PETSC_DEFAULT, maxits );
        KSPSetType( tPetscKSPProblem, KSPFGMRES );
        //KSPSetType(tPetscKSPProblem,KSPPREONLY);
        KSPGMRESSetOrthogonalization( tPetscKSPProblem, KSPGMRESModifiedGramSchmidtOrthogonalization );
        KSPGMRESSetHapTol( tPetscKSPProblem, 1e-10 );
        KSPGMRESSetRestart( tPetscKSPProblem, 500 );

        KSPSetFromOptions( tPetscKSPProblem );

        // PetscInt       n = ess_tdofv.Size(), *idxs;
        // const int      *data = ess_tdofv.GetData();

        // PetscMalloc1(n,&idxs);

        // for (PetscInt i=0; i<n; i++) { idxs[i] = data[i] + 0; }

        // IS dir;
        // ISCreateGeneral(PETSC_COMM_WORLD,n,idxs,PETSC_OWN_POINTER,&dir);


        // PCBDDCSetDirichletBoundaries(mpc,dir);


        PetscParVector B_petsc(A, true, false);
        PetscParVector X_petsc(A, true, false);

        B_petsc.PlaceMemory(B.GetMemory());
        X_petsc.PlaceMemory(X.GetMemory(),true);

            //         MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullsp);
            // KSPSetNullSpace(ksp, nullsp);
            // MatNullSpaceDestroy(&nullsp);

    if(true)
    {
        PetscViewer tViewerRes;
        PetscViewerCreate( PETSC_COMM_WORLD, &tViewerRes );
        PetscViewerSetType( tViewerRes, PETSCVIEWERASCII  );
        PetscViewerFileSetName( tViewerRes, "Residual_Norms.txt" );

        PetscViewerAndFormat *tViewerAndFormatRes;
        PetscViewerAndFormatCreate( tViewerRes, PETSC_VIEWER_DEFAULT, &tViewerAndFormatRes );

        KSPMonitorSet( tPetscKSPProblem,
                       reinterpret_cast< int(*)( KSP, int, double, void* ) >( KSPMonitorTrueResidualNorm ),
                       tViewerAndFormatRes,
                       NULL );
    }
           
        if(true)
        {
            MatNullSpace nullsp;
            MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullsp);
            MatSetNullSpace(A,nullsp);
            //KSPSetNullSpace(tPetscKSPProblem, nullsp);
            MatNullSpaceDestroy(&nullsp);
        }

        KSPView ( tPetscKSPProblem, PETSC_VIEWER_STDOUT_WORLD);
        KSPSolve(tPetscKSPProblem, B_petsc, X_petsc);

        B_petsc.ResetMemory();
        X_petsc.ResetMemory();

        KSPDestroy( &tPetscKSPProblem );
        }
        else{

        const char *petscrc_file = "";
        MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL); 

        bool use_nonoverlapping = false;
        //  Use PETSc to solve the linear system.
        //      Assemble a PETSc matrix, so that PETSc solvers can be used natively.
        PetscParMatrix A;
        a->SetOperatorType(use_nonoverlapping ?
                        Operator::PETSC_MATIS : Operator::PETSC_MATAIJ);
        a->FormLinearSystem(ess_tdofv, sol, *b, A, X, B);
        // if (myid == 0)
        // {
             std::cout << "done." <<std:: endl;
            std:: cout << "Size of linear system: " << A.M() <<std:: endl;
        // }
        PetscPCGSolver *pcg = new PetscPCGSolver(A);
   
        // The preconditioner for the PCG solver defined below is specified in the
        // PETSc config file, since a Krylov solver in PETSc can also
        // customize its preconditioner.
        PetscPreconditioner *prec = NULL;
        if (use_nonoverlapping)
        {
            // Compute dofs belonging to the natural boundary


   
            // Auxiliary class for BDDC customization
            PetscBDDCSolverParams opts;
            // Inform the solver about the finite element space
            opts.SetSpace(fes);
            // Inform the solver about essential dofs
            opts.SetEssBdrDofs(&ess_tdofv);
            // Inform the solver about natural dofs
           // opts.SetNatBdrDofs(&ess_tdofv);
            // Create a BDDC solver with parameters
            prec = new PetscBDDCSolver(A,opts);
            pcg->SetPreconditioner(*prec);
        }
   
        pcg->SetMaxIter(2000);
        pcg->SetTol(1e-12);
        pcg->SetPrintLevel(2);
        pcg->Mult(B, X);
        delete pcg;
        delete prec;
        }
      }

    solgf = X;     // copy solution

    std::cout<<" ----- end ----"<<std::endl;

    // We finalize PETSc
    // MFEMFinalizePetsc(); 

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

      std::cout<<"Error Norm: "<<solError.Norml2()<<std::endl;
   }
}

double Advection_Diffusion_Solver::RHSAdvCoeff::Eval(
    mfem::ElementTransformation & T,
    const IntegrationPoint & ip)
    {
        Vector vel(dim_);
        Vector avgGradTemp(dim_);

        vel_->Eval( vel, T,ip );

        vel *= sign_;
        avgGradTemp_->Eval( avgGradTemp, T,ip );

        return vel * avgGradTemp;
    }

void Advection_Diffusion_Solver::RHSDiffCoeff::Eval(
    mfem::Vector & V,
    mfem::ElementTransformation & T,
    const IntegrationPoint & ip)
    {
        Vector avgGradTemp(vdim);
        Vector flux(vdim);
        V.SetSize(vdim);
        DenseMatrix matTensor(vdim,vdim);

        avgGradTemp_->Eval( avgGradTemp, T,ip );

        avgGradTemp *= sign_;

        MaterialCoeff_->Eval( matTensor, T,ip );

        matTensor.Mult(avgGradTemp, V );
    }

}
