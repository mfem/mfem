#include "Stokes.hpp"

#ifdef MFEM_USE_PETSC
#include "petsc.h"
#endif

namespace mfem {
        namespace stokes{

void Stokes::SetDensityCoeff(    
   enum DensityCoeff::PatternType aGeometry,
   enum DensityCoeff::ProjectionType aProjectionType)
{
   mDensCoeff = new DensityCoeff;

   mDensCoeff->SetThreshold(meta);
   mDensCoeff->SetPatternType(aGeometry);

   mDensCoeff->SetProjectionType(aProjectionType);
}

void Stokes::FSolve()
{

    bool verbose = true;
    int dim = 2;
    int SetPrintLevel = 1;

    const char *device_config = "cpu";
    Device device(device_config);

    double BrinmannPen = 1000.0;
    bool   BrinkannBC = false;

    Array<int> block_offsets(3); // number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = fes_u->GetVSize();
    block_offsets[2] = fes_p->GetVSize();
    block_offsets.PartialSum();

    Array<int> block_trueOffsets(3); // number of variables + 1
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = fes_u->TrueVSize();
    block_trueOffsets[2] = fes_p->TrueVSize();
    block_trueOffsets.PartialSum();

    MemoryType mt = device.GetMemoryType();
    BlockVector x(block_offsets, mt), rhs(block_offsets, mt);
    BlockVector trueX(block_trueOffsets, mt), trueRhs(block_trueOffsets, mt);
    BlockVector trueBCRhs(block_trueOffsets, mt);

    trueBCRhs = 0.0;
    trueX= 0.0;

    ess_tdofv.DeleteAll();
    ess_tdofp.DeleteAll();

    if(BrinkannBC)
    {
        // compute indicater LF for DBC
        mfem::ParLinearForm VolParLinearForm_u(fes_u_vol);
        mfem::ParLinearForm VolParLinearForm_p(fes_p);

        VolParLinearForm_p.AddDomainIntegrator(new DomainLFIntegrator(*mDensCoeff,4,4));

        //VolParLinearForm_u.Assemble();
        VolParLinearForm_p.Assemble();

        //double maxValLF_u = VolParLinearForm_u.Max();
        double maxValLF_p = VolParLinearForm_p.Max();

        if( true )
        {
            //mfem::ParGridFunction GFVolParLinearForm_u(fes_u_vol);
            mfem::ParGridFunction GFVolParLinearForm_p(fes_p);

           // GFVolParLinearForm_u = VolParLinearForm_u;
            GFVolParLinearForm_p = VolParLinearForm_p;

            mPvdc = new ParaViewDataCollection("StokesIndicatorfield", pmesh);
            mPvdc->SetDataFormat(VTKFormat::BINARY32);
            mPvdc->SetHighOrderOutput(true);
            mPvdc->SetCycle(0);
            mPvdc->SetTime(0.0);
           // mPvdc->RegisterField("VelocityIndicatorField",& GFVolParLinearForm_u);
            mPvdc->RegisterField("PreassureIndicatorField",& GFVolParLinearForm_p);
            mPvdc->Save();
        }

        //VolParLinearForm_u.Print();

        int dim=pmesh->Dimension();
        {
            mfem::ParGridFunction IndicatorGF_u(fes_u); IndicatorGF_u = 0.0;
            int counter = 0;

            for( int Ik =0; Ik<pmesh->GetNE(); Ik++)
            {
                mfem::Array< int > Vertexvdofs;
                fes_p->GetElementVDofs(Ik,Vertexvdofs);

                for( int Ii =0; Ii<Vertexvdofs.Size(); Ii++)
                {
                    if(VolParLinearForm_p[Vertexvdofs[Ii]] <= (maxValLF_p - 1e-12))
                    {
                       break;
                    }

                    mfem::Array< int > ElementVdofs;
                    fes_u->GetElementVDofs(Ik,ElementVdofs);

                    for( int Ia =0; Ia<ElementVdofs.Size(); Ia++)
                    {
                        if( IndicatorGF_u[ElementVdofs[Ia]] == 0.0 )
                        {
                            IndicatorGF_u[ElementVdofs[Ia]] = 1.0;
                            counter ++;
                        }
                    }   
                }
            }

            ess_tdofv.SetSize(counter);
            counter=0;
                    
            for( int Ij =0; Ij<IndicatorGF_u.Size(); Ij++)
            {
                if( IndicatorGF_u[Ij] == 1 )
                {
                    ess_tdofv[counter++] = Ij;
                }
            }

             std::cout<<"size: "<<ess_tdofv.Size()<<std::endl;

            for( int Ik =0; Ik<VolParLinearForm_p.Size(); Ik++)
            {
                if(VolParLinearForm_p[Ik] >= (maxValLF_p - 1e-10))
                {
                    mfem::Array< int > Vertexvdofs;
                    fes_p->GetVertexVDofs(Ik,Vertexvdofs);

                    ess_tdofp.Append(Vertexvdofs[0]);
                }
            }
        }
    }
    else
    {
  
    // set the boundary conditions
        for(auto it=bcc_u.begin();it!=bcc_u.end();it++){
            mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
            ess_bdr=0;
            ess_bdr[it->first -1]=1;
            mfem::Array<int> ess_tdof_list;
            fes_u->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
            ess_tdofv.Append(ess_tdof_list);
            solgf.ProjectBdrCoefficient(*(it->second),ess_bdr);
        }
      
        //copy BC values from the grid function to the solution vector
        {
            solgf.GetTrueDofs(trueRhs.GetBlock(0));
            for(int ii=0;ii<ess_tdofv.Size();ii++)
            {
                trueX.GetBlock(0)[ess_tdofv[ii]]=trueRhs.GetBlock(0)[ess_tdofv[ii]];
            }
        }

            // set the boundary conditions
        for(auto it=bcc_p.begin();it!=bcc_p.end();it++){
            mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
            ess_bdr=0;
            ess_bdr[it->first -1]=1;
            mfem::Array<int> ess_tdof_list;
            fes_p->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
            ess_tdofp.Append(ess_tdof_list);
            solgf_p.ProjectBdrCoefficient(*(it->second),ess_bdr);
        }

        //copy BC values from the grid function to the solution vector
        {
            solgf_p.GetTrueDofs(trueRhs.GetBlock(1));
            for(int ii=0;ii<ess_tdofp.Size();ii++)
            {
                trueX.GetBlock(1)[ess_tdofp[ii]]=trueRhs.GetBlock(1)[ess_tdofp[ii]];
            }
        }
    }
    //the BC are setup in the solution vector sol

    std::cout<<"BC dofs size="<<ess_tdofv.Size()<<std::endl;


    //-------------------------------------------------------



    //    Set up the linear form b(.) which corresponds to the right-hand side of
    //    the FEM linear system.

    // trueX = 0.0;
    // trueX.GetBlock(0)=solgf;    
    // x.GetBlock(0)=solgf;  
    // trueX.GetBlock(1)=solgf_p;    
    // x.GetBlock(1)=solgf_p;  

    // trueRhs = 0.0;
    // trueRhs.GetBlock(0)=solgf;    
    // rhs.GetBlock(0)=solgf;  
    // trueRhs.GetBlock(1)=solgf_p;    
    // rhs.GetBlock(1)=solgf_p;  

    mfem::stokes::BrinkPenalAccel * tBrinkAcc = new mfem::stokes::BrinkPenalAccel(pmesh->Dimension());

    tBrinkAcc->SetDensity(mDensCoeff);
    tBrinkAcc->SetBrinkmannPenalization(BrinmannPen);
    tBrinkAcc->SetVel(&solgf);        //FIXME
    tBrinkAcc->SetParams( mnx, mny, mnz, ma);

    if(b==nullptr)
    {
        b = new ParLinearForm;
        b->Update(fes_u, rhs.GetBlock(0), 0);
        b->AddDomainIntegrator(new mfem::VectorDomainLFIntegrator(*tBrinkAcc));
        b->Assemble();
        b->SyncAliasMemory(rhs);
        b->ParallelAssemble(trueRhs.GetBlock(0));
        trueRhs.GetBlock(0).SyncAliasMemory(trueRhs);
    }

    
    HypreParMatrix *M = NULL;
    HypreParMatrix *B = NULL;

    ::mfem::ConstantCoefficient lambda_func(0.0);
    ::mfem::ConstantCoefficient mu_func(1.0 ); //1.0e-6
    ::mfem::ConstantCoefficient one(1.0);

    ::mfem::Coefficient * MassCoeff = 
    new ::mfem::ProductCoefficient(1.0*BrinmannPen,*mDensCoeff);

    //    Set up the bilinear form a(.,.) on the finite element space
    if(a_u==nullptr)
    {
        Array<int> empty;
        a_u = new ParBilinearForm(fes_u);
        a_p = new ParBilinearForm(fes_p);
        a_up = new ParMixedBilinearForm(fes_p, fes_u);   // trail space | test space
        a_pu = new ParMixedBilinearForm(fes_u, fes_p);

        // add vector diffusion integrators
        a_u->AddDomainIntegrator(new ElasticityIntegrator(lambda_func,mu_func));
        a_u->AddDomainIntegrator(new VectorMassIntegrator (*MassCoeff));
        //a_p->AddDomainIntegrator(new MassIntegrator(one));

        // add advection integrators
        a_up->AddDomainIntegrator(new TransposeIntegrator(new VectorDivergenceIntegrator()));
        //a_pu->AddDomainIntegrator(new VectorDivergenceIntegrator());  //GradientIntegrator()
        a_pu->AddDomainIntegrator(new VectorDivergenceIntegrator());  

        
    }
        std::cout<<"Assemble System"<<std::endl;
        a_u->Assemble();
        a_p->Assemble();
        a_up->Assemble();
        a_pu->Assemble();

        a_u->Finalize();
        a_p->Finalize();
        a_up->Finalize();
        a_pu->Finalize();

        BlockOperator *StokesOp = new BlockOperator(block_trueOffsets);

        HypreParMatrix * L_uu = a_u->ParallelAssemble();
        HypreParMatrix * L_pp = a_p->ParallelAssemble();
        HypreParMatrix * D_up = a_up->ParallelAssemble();
        HypreParMatrix * D_pu = a_pu->ParallelAssemble();

        //--------------------------------------------------------------

        // OperatorHandle A, B_new, C;
        // a_u->FormSystemMatrix(ess_tdofv, A);
        // a_up->FormRectangularSystemMatrix(ess_tdofp, ess_tdofv, B_new);
        // //a_pu.FormRectangularSystemMatrix(ess_tdofp, ess_tdofv, D);
        // a_p->FormSystemMatrix(ess_tdofp, C);

        // OperatorHandle Bt_new(B_new.As<HypreParMatrix>()->Transpose());

        // We can combine the blocks into a monolithic HypreParMatrix
        // Array2D<HypreParMatrix*> blocks(2,2);
        // blocks(0,0) = A.As<HypreParMatrix>();
        // blocks(1,0) = B.As<HypreParMatrix>();
        // blocks(0,1) = Bt.As<HypreParMatrix>();
        // blocks(1,1) = C.As<HypreParMatrix>();
        // OperatorHandle D(HypreParMatrixFromBlocks(blocks));
        
        // ;;mfem::ParBilinearForm *Linv = new ParBilinearForm(fes_u);
        // Linv->AddDomainIntegrator(new InverseIntegrator(new ElasticityIntegrator(lambda_func,mu_func)));
        // Linv->Assemble();
        // Linv->Finalize();

        // HypreParMatrix * L_uu_inv = Linv->ParallelAssemble();   delete Linv;

        //=========================================

        //(*D_up) *= -1.0;
        //TransposeOperator * D_up = new TransposeOperator(D_pu);

      //=====================================================================

        // StokesOp->SetBlock(0,0, A.As<HypreParMatrix>());
        // StokesOp->SetBlock(1,1, C.As<HypreParMatrix>());
        // StokesOp->SetBlock(0,1, B_new.As<HypreParMatrix>(), -1.0);
        // StokesOp->SetBlock(1,0, Bt_new.As<HypreParMatrix>(), -1.0);

        StokesOp->SetBlock(0,0, L_uu);
        StokesOp->SetBlock(1,1, L_pp);
        StokesOp->SetBlock(0,1, D_up, -1.0);
        StokesOp->SetBlock(1,0, D_pu, -1.0);

        // Array2D<HypreParMatrix*> blocks(2,2);
        // blocks(0,0) = L_uu;
        // blocks(1,0) = L_pp);
        // blocks(0,1) = D_up;
        // blocks(1,1) = CD_pu;  
        // OperatorHandle D(HypreParMatrixFromBlocks(blocks));         

    HypreParMatrix * Mass_p = nullptr;
    {
        mfem::ParBilinearForm* bf=new mfem::ParBilinearForm(fes_p);
        bf->AddDomainIntegrator(new mfem::MassIntegrator(one));
        bf->Assemble();
        bf->Finalize();
        Mass_p=bf->ParallelAssemble();
        delete bf;

        // Mass mat with 1-\rho
    }

        std::cout<<"Preconditioner"<<std::endl;

    // setup block prec
    //                 P = [ diag(M)         0         ]
    //                     [  0       B diag(M)^-1 B^T ]
    //
    //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
    //     pressure Schur Complement.

    BlockDiagonalPreconditioner *StokesPr = new BlockDiagonalPreconditioner(
        block_trueOffsets);  

    HypreParMatrix *LinvBt = NULL;
    HypreParVector *Ldiag = NULL;
    HypreParMatrix *S = NULL;
    Vector Md_PA;
    Solver *invL, *invS;

if(true)
{
    //static_cast<mfem::BlockMatrix*>(StokesOp)->EliminateRowsCols(ess_tdofv, trueX, trueRhs);

    mfem::HypreParMatrix* A00=static_cast<mfem::HypreParMatrix*>(&(StokesOp->GetBlock(0,0)));
    mfem::HypreParMatrix* A01=static_cast<mfem::HypreParMatrix*>(&(StokesOp->GetBlock(0,1)));
    mfem::HypreParMatrix* A10=static_cast<mfem::HypreParMatrix*>(&(StokesOp->GetBlock(1,0)));
    mfem::HypreParMatrix* A11=static_cast<mfem::HypreParMatrix*>(&(StokesOp->GetBlock(1,1)));
    
    StokesOp->Mult( trueX, trueRhs);
    trueRhs.Neg();

    mfem::HypreParMatrix* A00elim=A00->EliminateRowsCols(ess_tdofv);
     mfem::HypreParMatrix* A11elim=A11->EliminateRowsCols(ess_tdofp );
     Mass_p->EliminateRowsCols(ess_tdofp );
    mfem::HypreParMatrix* A10elim=A10->EliminateCols(ess_tdofv);  A10->EliminateRows(ess_tdofp);
    mfem::HypreParMatrix* A01elim=A01->EliminateCols(ess_tdofp);  A01->EliminateRows(ess_tdofv);

    for(int ii=0;ii<ess_tdofv.Size();ii++)
    {
        trueRhs.GetBlock(0)[ess_tdofv[ii]]=trueX.GetBlock(0)[ess_tdofv[ii]];
    }
        for(int ii=0;ii<ess_tdofp.Size();ii++)
    {
        trueRhs.GetBlock(1)[ess_tdofp[ii]]=trueX.GetBlock(1)[ess_tdofp[ii]];
    }

        // std::fstream out("full.mat",std::ios::out);
        // StokesOp->PrintMatlab(out);
        // out.close();

        // ad

    
    
    Ldiag = new HypreParVector(MPI_COMM_WORLD, A00->GetGlobalNumRows(), A00->GetRowStarts());
    A00->GetDiag(*Ldiag);

    LinvBt = A10->Transpose();
    LinvBt->InvScaleRows(*Ldiag);
    S = ParMult(A10, LinvBt);
    //S->operator *=(-1.0);


    mfem::HypreBoomerAMG* invLL = new HypreBoomerAMG(*A00);
    invLL->SetSystemsOptions(dim);
    invLL->SetElasticityOptions(fes_u);
    invLL->SetPrintLevel(print_level);
    invL = invLL;
    invS = new HypreBoomerAMG(*Mass_p);
    //    invS = new HypreBoomerAMG(*S);

    // invL->iterative_mode = false;
    // invS->iterative_mode = false;
       
    StokesPr->SetDiagonalBlock(0, invL);
    StokesPr->SetDiagonalBlock(1, invS);
}
else{
        // HypreParMatrix D_pu_new(*D_pu);

        // D_pu_new *= -1.0;

        // RAPOperator APrec(*D_up, *L_uu_inv, D_pu_new);

    // Ldiag = new HypreParVector(MPI_COMM_WORLD, L_uu->GetGlobalNumRows(),
    //                           L_uu->GetRowStarts());
    // L_uu->GetDiag(*Ldiag);

    // LinvBt = D_pu->Transpose();
    // LinvBt->InvScaleRows(*Ldiag);
    // S = ParMult(D_pu, LinvBt);

    // Ldiag = new HypreParVector(MPI_COMM_WORLD, L_uu->GetGlobalNumRows(), L_uu->GetRowStarts());
    // L_uu->GetDiag(*Ldiag);
    // //Ldiag->operator *=(9);
    // LinvBt = D_pu->Transpose();
    // LinvBt->InvScaleRows(*Ldiag);
    // S = ParMult(D_pu, LinvBt);
    // S->operator *=(-1.0);

    // // // invL = new HypreDiagScale(*L_uu);
    // // // invS = new HypreBoomerAMG(*S);
    // mfem::HypreBoomerAMG* invLL = new HypreBoomerAMG(*L_uu);
    // invLL->SetSystemsOptions(dim);
    // invLL->SetElasticityOptions(fes_u);
    // invLL->SetPrintLevel(print_level);
    // invL = invLL;
    // invS = new HypreBoomerAMG(*S);

    // invL->iterative_mode = false;
    // invS->iterative_mode = false;
       
    // StokesPr->SetDiagonalBlock(0, invL);
    // StokesPr->SetDiagonalBlock(1, invS);
}

    std::cout<<"Solve"<<std::endl;

    // Solve
    int maxIter( 5000);
    double rtol(1.e-7);
    double atol(1.e-10);

    if( true)
    {
        mfem::GMRESSolver solver(MPI_COMM_WORLD);
        solver.SetAbsTol(atol);
        solver.SetRelTol(rtol);
        solver.SetMaxIter(maxIter);
        solver.SetKDim(1000);
        solver.SetPrintLevel(verbose);
        solver.SetOperator(*StokesOp);
        solver.SetPreconditioner(*StokesPr);

        //trueX.Print();

        //trueX = 0.0;    
        solver.Mult(trueRhs, trueX);
    }
    else
    {
        MINRESSolver solver(MPI_COMM_WORLD);
        solver.SetAbsTol(atol);
        solver.SetRelTol(rtol);
        solver.SetMaxIter(maxIter);
        solver.SetOperator(*StokesOp);
        solver.SetPreconditioner(*StokesPr);
        solver.SetPrintLevel(verbose);
        //StrueX = 0.0;
        solver.Mult(trueRhs, trueX);
    }

    std::cout<<"Postprocess"<<std::endl;

    // postprocess
    ParGridFunction *u(new ParGridFunction);
    ParGridFunction *p(new ParGridFunction);
    u->MakeRef(fes_u, x.GetBlock(0), 0);
    p->MakeRef(fes_p, x.GetBlock(1), 0);
    u->Distribute(&(trueX.GetBlock(0)));
    p->Distribute(&(trueX.GetBlock(1)));

    solgf = *u;

    if( true )
    {
       ParGridFunction *d_gf = new ParGridFunction(*p);
       mDensCoeff->SetProjectionType(stokes::DensityCoeff::ProjectionType::continuous);
       d_gf->ProjectCoefficient(*mDensCoeff);
       mDensCoeff->SetProjectionType(stokes::DensityCoeff::ProjectionType::zero_one);

       mPvdc = new ParaViewDataCollection("Stokes2D", pmesh);
       mPvdc->SetDataFormat(VTKFormat::BINARY32);
       mPvdc->SetHighOrderOutput(true);
       mPvdc->SetCycle(0);
       mPvdc->SetTime(0.0);
       mPvdc->RegisterField("Velocity", u);
       mPvdc->RegisterField("Preassure", p);
       mPvdc->RegisterField("densitu", d_gf);
       mPvdc->Save();
    }

    delete MassCoeff;
    delete tBrinkAcc;

}

 void Stokes::Postprocess()
 {
      int vdim = fes_u->GetVDim();

      // initilaize integradl of velozity vector
      mfem::Vector tAverageVel(vdim);       tAverageVel = 0.0;
      double tVolume = 0.0;

      for (int e = 0; e < fes_u->GetNE(); ++e)
      {
         const FiniteElement *fe = fes_u->GetFE(e);
         const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                               fe->GetOrder());
         ElementTransformation *tr = fes_u->GetElementTransformation(e);

         for (int i = 0; i < ir.GetNPoints(); ++i)
         {
            const IntegrationPoint &ip = ir.IntPoint(i);
            tr->SetIntPoint(&ip);

            double w = tr->Weight() * ip.weight;

            Vector tVal;

            solgf.GetVectorValue( e, ip, tVal);

            tVal *= w;
            tAverageVel += tVal;

            tVolume += w;
         }
      }

      double tTotalVol;
      MPI_Allreduce(&tVolume, &tTotalVol, 1, MPI_DOUBLE, MPI_SUM,
                  pmesh->GetComm());

      for( int Ik = 0; Ik < vdim; Ik ++)
      {
         double tVal = tAverageVel(Ik); 
         double tTotalVal;

         MPI_Allreduce(
            &tVal,
            &tTotalVal, 
            1, 
            MPI_DOUBLE, 
            MPI_SUM,
            pmesh->GetComm());

            tAverageVel(Ik) = tTotalVal / tTotalVol;
      }

      mfem::VectorConstantCoefficient AvgVelCoeff(tAverageVel);
      mfem::ParGridFunction           AvgVelGF(fes_u);
      AvgVelGF.ProjectCoefficient(AvgVelCoeff);


        solMinusAvgGF  = solgf;
        solMinusAvgGF  -= AvgVelGF;


    //   if (mpi.Root())
    //   {
    //      std::string tString = "./OutputFile_";

    //      Ascii tAsciiWriter( tString, FileMode::NEW );

    //      for( int Ik = 0; Ik < vdim; Ik ++)
    //      {
    //         tAsciiWriter.print(stringify( tVelVal(Ik) ));
    //      }
         
    //      tAsciiWriter.save();
    //   }
}

}

}