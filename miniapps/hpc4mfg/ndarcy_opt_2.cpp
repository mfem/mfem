
//#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include "MMA.hpp"
#include "hpc4solvers.hpp"
#include "hpc4mat.hpp"

int main(int argc, char *argv[])
{
   // 1. Initialize MPI
   int num_procs, myrank;
   mfem::MPI_Session mpi(argc, argv);
   // MPI_Init(&argc, &argv);
   // MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   // MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

   // Define Caliper ConfigManager
#ifdef MFEM_USE_CALIPER
   cali::ConfigManager mgr;
#endif

   // Caliper instrumentation
   MFEM_PERF_FUNCTION;

   // 2. Parse command-line options
   const char *mesh_file = "./cross2D.msh";
   int ser_ref_levels = 1;
   int par_ref_levels = 0;
   int order = 1;
   bool visualization = true;
   double newton_rel_tol = 1e-4;
   double newton_abs_tol = 1e-6;
   int newton_iter = 10;
   int print_level = 0;

   double rel_tol = 1e-7;
   double abs_tol = 1e-15;
   double fradius = 0.05;
   int tot_iter = 100;
   int max_it = 100;

   bool ConstSensFD =false;
   bool ObjSensFD =false;
   bool dQdpFD =false;
   bool dQdsFD =false;
   bool dRdsFD =false;
   bool BreakAfterFirstIt = false;
   bool initializeRandom = false;

   bool initializeSol = false;
   bool restartDesign = false;

   const char *petscrc_file = "";
  
   const char* cali_config = "runtime-report";

   mfem::OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels,
                  "-rp",
                  "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order,
                  "-o",
                  "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&visualization,
                  "-vis",
                  "--visualization",
                  "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&newton_rel_tol,
                  "-rel",
                  "--relative-tolerance",
                  "Relative tolerance for the Newton solve.");
   args.AddOption(&newton_abs_tol,
                  "-abs",
                  "--absolute-tolerance",
                  "Absolute tolerance for the Newton solve.");
   args.AddOption(&newton_iter,
                  "-it",
                  "--newton-iterations",
                  "Maximum iterations for the Newton solve.");
   args.AddOption((&print_level), "-prt", "--print-level", "Print level.");
   args.AddOption(&cali_config, "-p", "--caliper",
                  "Caliper configuration string.");
   args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                     "PetscOptions file to use.");
   args.Parse();
   if (!args.Good())
   {
      if (myrank == 0)
      {
         args.PrintUsage(std::cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myrank == 0)
   {
      args.PrintOptions(std::cout);
   }
   mfem::MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);

   // Caliper configuration
#ifdef MFEM_USE_CALIPER
   mgr.add(cali_config);
   mgr.start();
#endif

   //    Read the (serial) mesh from the given mesh file on all processors. We
   //    can handle triangular, quadrilateral, tetrahedral and hexahedral meshes
   //    with the same code.
   mfem::Mesh *mesh = new mfem::Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   mfem::ParMesh *pmesh = new mfem::ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // Allocate the nonlinear diffusion solver
   mfem::NLDiffusion* solver=new mfem::NLDiffusion(pmesh,2);

   //add boundary conditions
   //solver->AddDirichletBC(2,0.05);
   solver->AddDirichletBC(3,0.0);


   if(true)
   {
      mfem::ParaViewDataCollection paraview_dc("InitialMesh", pmesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);
      paraview_dc.Save();
   }

   mfem::ParFiniteElementSpace *fes = solver->GetFES();

      // build h1 desing space
   int orderDesing = 1;
   ::mfem::H1_FECollection desFECol_H1(orderDesing, dim);
   ::mfem::ParFiniteElementSpace desFESpace_scalar_H1(pmesh, &desFECol_H1 );

   ::mfem::H1_FECollection anaFECol_H1(2, dim);
   ::mfem::ParFiniteElementSpace anaVelFESpace_scalar_H1(pmesh, &desFECol_H1,2 );

   //gradients with respect to the desing field
   mfem::Vector objgrad(desFESpace_scalar_H1.GetTrueVSize()); objgrad=0.0; //of the objective
   mfem::Vector volgrad(desFESpace_scalar_H1.GetTrueVSize()); volgrad=0.0; //of the volume contr.

   // set esing variable bounds
   mfem::Vector xxmax(desFESpace_scalar_H1.GetTrueVSize()); 
   mfem::Vector xxmin(desFESpace_scalar_H1.GetTrueVSize()); 

   xxmax=0.5;
   xxmin=0.1;

   std::cout<<"maxBdrAttr: "<<pmesh->bdr_attributes.Max()<<std::endl;

   // desing variable vector
   mfem::ParGridFunction desingVarVec(&desFESpace_scalar_H1); desingVarVec=0.3;
   mfem::Vector vdens; vdens.SetSize(desFESpace_scalar_H1.GetTrueVSize()); vdens=0.3;

   if(initializeRandom)
   {
      for(int Ij = 0; Ij< desFESpace_scalar_H1.GetTrueVSize(); Ij++)
      {
         desingVarVec[Ij] = rand() / double(RAND_MAX)*0.5;
         vdens[Ij] = rand() / double(RAND_MAX)*0.5;
      }
   }

   mfem::ParGridFunction IdGF(&desFESpace_scalar_H1); IdGF=0.0;
   for(int Ij = 0; Ij< desFESpace_scalar_H1.GetTrueVSize(); Ij++)
   {
      IdGF[Ij] = Ij;
   }

   desingVarVec.SetFromTrueDofs(vdens);

   //desingVarVec.Print();

   std::cout<<"add material"<<std::endl;
   //add material
   //mfem::BasicNLDiffusionCoefficient* tMatCoeff = new mfem::DarcyCoefficient();
   mfem::BasicNLDiffusionCoefficient* tMatCoeff = new mfem::SurrogateNLDiffusionCoefficient();
   solver->AddMaterial(tMatCoeff);
   mfem::Coefficient * DesingCoeff = new mfem::GridFunctionCoefficient(&desingVarVec);
   solver->AddDesignCoeff(DesingCoeff);
   solver->AddDesignGF(&desingVarVec);
   std::cout<<"solve"<<std::endl;
   //solve
   //solver->FSolve();
   std::cout<<"opt"<<std::endl;

   //----------------------------------------------------
   // Optimization
   //  mfem::FilterSolver* fsolv = nullptr;

   // if( true)
   // {
   //    //allocate the filter
   //    fsolv=new mfem::FilterSolver(1.0,&pmesh);
   //    fsolv->SetSolver(1e-8,1e-12,100,0);
   //    fsolv->AddBC(1,1.0);
   //    fsolv->AddBC(2,1.0);
   //    fsolv->AddBC(3,1.0);
   //    fsolv->AddBC(4,1.0);
   // }

   if(false)
   {
      mfem::Vector locationVector(2);
      mfem::Vector & tSol = solver->GetSol();

      //std::cout<<"SolSize: "<<tSol.Size()<<std::endl;

      for(int Ij = 0; Ij< fes->GetTrueVSize(); Ij++)
      {
         //pmesh->GetNode(Ij, &locationVector[0]);

         //const double * pCoords(static_cast<const double*>(locationVector));

         tSol[Ij] = rand() / double(RAND_MAX)*0.01;
      }
   }

   if(initializeSol)
   {
      mfem::Vector & tSol = solver->GetSol();

      std::string tStringIn = "Darcy_FieldVec";
      int n = 6;
      std::string tWorlsRank = std::to_string( mpi.WorldRank());
    
      int precision = n - tWorlsRank.size();
      std::string s = std::string(precision, '0').append(tWorlsRank);

      tStringIn= tStringIn +"."+s;

      std::ifstream inp(tStringIn);
      mfem::ParGridFunction FieldGF(pmesh, inp);

      tSol = FieldGF;
   }

   if(restartDesign)
   {
      std::string tStringIn = "Darcy_DesingVarVec";
      int n = 6;
      std::string tWorlsRank = std::to_string( mpi.WorldRank());
    
      int precision = n - tWorlsRank.size();
      std::string s = std::string(precision, '0').append(tWorlsRank);

      tStringIn= tStringIn +"."+s;

      std::ifstream inp(tStringIn);
      mfem::ParGridFunction tLoadGF(pmesh, inp);

      desingVarVec = tLoadGF;

      for( int Ik=0;Ik<desingVarVec.Size();Ik++)
      {
         if(desingVarVec[Ik] < 0.1){desingVarVec[Ik]=0.1;}
      }
   }



   mfem::NativeMMA* mma = nullptr;
   {
       double a=0.0;
       double c=1000.0;
       double d=0.0;
       mma=new mfem::NativeMMA(MPI_COMM_WORLD,1, objgrad,&a,&c,&d);
   }

   double max_ch=0.01; //max design change

   double energyDissip; //energy dissipation  // cpl
   double vol; //volume
   double ivol; //intermediate volume

   double max_vol = 1.00;
   double maxVolAllowed = max_vol*0.65;

   mfem::ParaViewDataCollection paraview_dc("TopOpt", pmesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);

   if(true)
   {
      std::cout<<"opt iter"<<std::endl;

      for(int i=1;i<max_it;i++)
      {
         //solve desing variable and solve
         //solver->SetDensity(vdens,0.8);
         solver->FSolve();

         mfem::ParGridFunction tPreassureGF;
         solver->GetSol(tPreassureGF);

         std::cout<<"opt solve"<<std::endl;

         // evaluate obj and volume
         mfem::EnergyDissipationObjective tObj;
         tObj.SetNLDiffusionSolver( solver );
         tObj.SetDesignFES( &desFESpace_scalar_H1 );
         tObj.SetDesField( desingVarVec );
         tObj.SetNLDiffusionCoeff( tMatCoeff );

         mfem::VolumeQoI tContraint;
         tContraint.SetDesignFES( &desFESpace_scalar_H1 );
         tContraint.SetDesField( desingVarVec );

         energyDissip =tObj.Eval();
         vol          =tContraint.Eval();

         if(myrank==0)
         {
             std::cout<<"it: "<<i<<" | obj= "<<energyDissip<<" | vol= "<<vol<<" | Constraint: "<< vol-maxVolAllowed<<std::endl;
         }

         tObj      .Grad(desingVarVec,objgrad);
         tContraint.Grad(desingVarVec,volgrad);

         std::cout<<"EVAL: "<<std::endl;


         mfem::EnergyDissipationIntegrator_1 * ImpEnergyDispintgr = new mfem::EnergyDissipationIntegrator_1;
         ImpEnergyDispintgr->SetPreassure(&tPreassureGF);
         ImpEnergyDispintgr->SetNLDiffusionCoeff(tMatCoeff);
         ImpEnergyDispintgr->SetDesingField(&desingVarVec);
         mfem::ParLinearForm ParLinerFormDQDp(fes);
         ParLinerFormDQDp.AddDomainIntegrator(ImpEnergyDispintgr);

         ParLinerFormDQDp.Assemble();

         //ParLinerFormDQDp.Print();
         
         solver->ASolve(ParLinerFormDQDp);

         mfem::ParGridFunction tAdjointGF;
         solver->GetAdj(tAdjointGF);

         std::cout<<"Adjoint Solve done"<<std::endl;

         // ---- posmultiply -----

         mfem::AdjointPostIntegrator * AdjointPostIntegrator = new mfem::AdjointPostIntegrator;
         AdjointPostIntegrator->SetAdjoint(&tAdjointGF);
         AdjointPostIntegrator->SetNLDiffusionCoeff(tMatCoeff);
         AdjointPostIntegrator->SetDesingField(&desingVarVec);
         AdjointPostIntegrator->SetPreassure(&tPreassureGF);
         mfem::ParLinearForm ParLinerFormPostAdjoint(&desFESpace_scalar_H1);
         ParLinerFormPostAdjoint.AddDomainIntegrator(AdjointPostIntegrator);

         ParLinerFormPostAdjoint.Assemble();

         //objgrad.Print();
         //std::cout<<" --------------------------aaa--------------------"<<std::endl;
         //ParLinerFormPostAdjoint.Print();

         objgrad -= ParLinerFormPostAdjoint;

        //objgrad *= -1.0;
         

         paraview_dc.SetCycle(i);
         paraview_dc.SetTime(i*1.0);
         paraview_dc.RegisterField("design",&desingVarVec);
         paraview_dc.RegisterField("pressure",&tPreassureGF);
         paraview_dc.RegisterField("IdGF",&IdGF);  
         mfem::ParGridFunction objGradGF(&desFESpace_scalar_H1); objGradGF = objgrad;

         paraview_dc.RegisterField("ObjGrad",&objGradGF);  

         ::mfem::VectorCoefficient* VelVoeff = new mfem::VelCoefficient(
            tMatCoeff,
            &tPreassureGF,
            &desingVarVec  ); 

         mfem::ParGridFunction velGF(&anaVelFESpace_scalar_H1);
         velGF.ProjectCoefficient(*VelVoeff);

         paraview_dc.RegisterField("velocity",&velGF);
         delete VelVoeff;
    
         paraview_dc.Save();

          //compute the gradients of obj and constraint wrt desing variable
          //solver->MeanCompliance(ograd);
          //ivobj->Grad(vdens,vgrad);

         {
              // impose desing variable bounds - set xxmin and xxmax
              xxmin=desingVarVec; xxmin-=max_ch;
              xxmax=desingVarVec; xxmax+=max_ch;
              for(int li=0;li<xxmin.Size();li++)
              {
                  if(xxmin[li]<0.1)
                  {
                     xxmin[li]=0.1;
                  }
                  if(xxmax[li]>0.5)
                  {
                     xxmax[li]=0.5;
                  }
              }
         }

         //   //compute the original gradients
         //   fsolv->MultTranspose(ograd,ogrado);
         //   fsolv->MultTranspose(vgrad,vgrado);
         
         
         if(ConstSensFD)
         {
            double epsilon = 1e-8;
            mfem::ParGridFunction tFD_sens(fes); tFD_sens = 0.0;
            for( int Ia = 0; Ia<desingVarVec.Size(); Ia++)
            {
               desingVarVec[Ia] +=epsilon;

               mfem::VolumeQoI tContraintFD_1;
               tContraintFD_1.SetDesignFES( fes );
               tContraintFD_1.SetDesField( desingVarVec );
               double volFD_1  =tContraint.Eval();

               desingVarVec[Ia] -=2.0*epsilon;
               mfem::VolumeQoI tContraintFD_2;
               tContraintFD_2.SetDesignFES( fes );
               tContraintFD_2.SetDesField( desingVarVec );
               double volFD_2  =tContraint.Eval();

               desingVarVec[Ia] +=epsilon;

               tFD_sens[Ia] = (volFD_1-volFD_2)/(2.0*epsilon);
            }
               volgrad.Print();
               std::cout<<"  ----------  FD Diff ------------"<<std::endl;
               tFD_sens.Print();

               std::cout<<"  ---------- Analytic - FD Diff ------------"<<std::endl;
               mfem::ParGridFunction tFD_diff(fes); tFD_diff = 0.0;
               tFD_diff = volgrad;
               tFD_diff -=tFD_sens;
               tFD_diff.Print();

                  std::cout<<"norm: "<<tFD_diff.Norml2()<<std::endl;
         }

         if(ObjSensFD)
         {
            double epsilon = 1e-8;
            mfem::ParGridFunction tFD_sens(fes); tFD_sens = 0.0;
            for( int Ia = 0; Ia<desingVarVec.Size(); Ia++)
            {
                  desingVarVec[Ia] +=epsilon;

                  // Allocate the nonlinear diffusion solver
                  mfem::NLDiffusion solver_FD1(pmesh,2);
                  //solver_FD1.AddDirichletBC(2,0.05);
                  solver_FD1.AddDirichletBC(3,0.0);
                  solver_FD1.AddMaterial(tMatCoeff);
                  solver_FD1.AddDesignGF(&desingVarVec);
                  solver_FD1.FSolve();
                  // mfem::ParGridFunction tPreassureGF_FD;
                  // solver->GetSol(tPreassureGF_FD);

                  mfem::EnergyDissipationObjective tObjFD_1;
                  tObjFD_1.SetNLDiffusionSolver( &solver_FD1 );
                  tObjFD_1.SetDesignFES( fes );
                  tObjFD_1.SetDesField( desingVarVec );
                  tObjFD_1.SetNLDiffusionCoeff( tMatCoeff );
                  double energyDissipFD_1 =tObjFD_1.Eval();

                  desingVarVec[Ia] -=2.0*epsilon;
                  mfem::NLDiffusion solver_FD2(pmesh,2);
                  //solver_FD2.AddDirichletBC(2,0.05);
                  solver_FD2.AddDirichletBC(3,0.0);
                  solver_FD2.AddMaterial(tMatCoeff);
                  solver_FD2.AddDesignGF(&desingVarVec);
                  solver_FD2.FSolve();
                  // mfem::ParGridFunction tPreassureGF_FD;
                  // solver->GetSol(tPreassureGF_FD);

                  mfem::EnergyDissipationObjective tObjFD_2;
                  tObjFD_2.SetNLDiffusionSolver( &solver_FD2 );
                  tObjFD_2.SetDesignFES( fes );
                  tObjFD_2.SetDesField( desingVarVec );
                  tObjFD_2.SetNLDiffusionCoeff(tMatCoeff );
                  double energyDissipFD_2 =tObjFD_2.Eval();

                  desingVarVec[Ia] +=epsilon;

                  tFD_sens[Ia] = (energyDissipFD_1-energyDissipFD_2)/(2.0*epsilon);
                  std::cout<<"Var number: "<< Ia<< " Analytic: "<<objgrad[Ia] << " FD: "<< tFD_sens[Ia]<<std::endl;
            }
            
            objgrad.Print();
            std::cout<<"  ---------- FD obj ------------"<<std::endl;
            tFD_sens.Print();
            std::cout<<"  ---------- Analytic - FD Diff ------------"<<std::endl;
            mfem::ParGridFunction tFD_diff(fes); tFD_diff = 0.0;
            tFD_diff = objgrad;
            tFD_diff -=tFD_sens;
            tFD_diff.Print();

            std::cout<<"norml2: "<<tFD_diff.Norml2()<<"normllinf: "<<tFD_diff.Normlinf()<<"max/min: "<<tFD_diff.Max()<<" / "<<tFD_diff.Min()<<std::endl;
         }

         if(dQdpFD)
         {
            double epsilon = 1e-8;
            mfem::ParGridFunction tFD_sens(fes); tFD_sens = 0.0;
            for( int Ia = 0; Ia<tPreassureGF.Size(); Ia++)
            {
               tPreassureGF[Ia] +=epsilon;

               mfem::EnergyDissipationObjective tObj_FD1;
               tObj_FD1.SetNLDiffusionSolver( solver );
               tObj_FD1.SetPreassure( &tPreassureGF);
               tObj_FD1.SetDesignFES( fes );
               tObj_FD1.SetDesField( desingVarVec );
               tObj_FD1.SetNLDiffusionCoeff( tMatCoeff );

               double energyDissipFD1 =tObj_FD1.Eval();

               tPreassureGF[Ia] -=2.0*epsilon;
               
               mfem::EnergyDissipationObjective tObj_FD2;
               tObj_FD2.SetNLDiffusionSolver( solver );
               tObj_FD2.SetPreassure( &tPreassureGF);
               tObj_FD2.SetDesignFES( fes );
               tObj_FD2.SetDesField( desingVarVec );
               tObj_FD2.SetNLDiffusionCoeff( tMatCoeff );

               double energyDissipFD2 =tObj_FD2.Eval();

               tPreassureGF[Ia] +=epsilon;

               tFD_sens[Ia] = (energyDissipFD1-energyDissipFD2)/(2.0*epsilon);
            }
               ParLinerFormDQDp.Print();
               std::cout<<"  ----------  FD Diff ------------"<<std::endl;
               tFD_sens.Print();

               std::cout<<"  ---------- Analytic - FD Diff ------------"<<std::endl;
               mfem::ParGridFunction tFD_diff(fes); tFD_diff = 0.0;
               tFD_diff = ParLinerFormDQDp;
               tFD_diff -=tFD_sens;
               tFD_diff.Print();
                                       std::cout<<"norm: "<<tFD_diff.Norml2()<<std::endl;
         }

         if(dQdsFD)
         {
            double epsilon = 1e-8;
            mfem::ParGridFunction tFD_sens(fes); tFD_sens = 0.0;
            for( int Ia = 0; Ia<desingVarVec.Size(); Ia++)
            {
               desingVarVec[Ia] +=epsilon;

               mfem::EnergyDissipationObjective tObj_FD1;
               tObj_FD1.SetNLDiffusionSolver( solver );
               tObj_FD1.SetPreassure( &tPreassureGF);
               tObj_FD1.SetDesignFES( fes );
               tObj_FD1.SetDesField( desingVarVec );
               tObj_FD1.SetNLDiffusionCoeff( tMatCoeff );

               double energyDissipFD1 =tObj_FD1.Eval();

               desingVarVec[Ia] -=2.0*epsilon;
               
               mfem::EnergyDissipationObjective tObj_FD2;
               tObj_FD2.SetNLDiffusionSolver( solver );
               tObj_FD2.SetPreassure( &tPreassureGF);
               tObj_FD2.SetDesignFES( fes );
               tObj_FD2.SetDesField( desingVarVec );
               tObj_FD2.SetNLDiffusionCoeff( tMatCoeff );

               double energyDissipFD2 =tObj_FD2.Eval();

               desingVarVec[Ia] +=epsilon;

               tFD_sens[Ia] = (energyDissipFD1-energyDissipFD2)/(2.0*epsilon);
            }
               objgrad.Print();
               std::cout<<"  ----------  FD Diff ------------"<<std::endl;
               tFD_sens.Print();

               std::cout<<"  ---------- Analytic - FD Diff ------------"<<std::endl;
               mfem::ParGridFunction tFD_diff(fes); tFD_diff = 0.0;
               tFD_diff = objgrad;
               tFD_diff -=tFD_sens;
               tFD_diff.Print();
                                       std::cout<<"norm: "<<tFD_diff.Norml2()<<std::endl;
         }

         if(dRdsFD)
         {
            double epsilon = 1e-8;
            mfem::ParGridFunction tFD_sens(fes); tFD_sens = 0.0;
            for( int Ia = 0; Ia<desingVarVec.Size(); Ia++)
            {
               desingVarVec[Ia] +=epsilon;

               mfem::EnergyDissipationObjective tObj_FD1;
               tObj_FD1.SetNLDiffusionSolver( solver );
               tObj_FD1.SetPreassure( &tPreassureGF);
               tObj_FD1.SetDesignFES( fes );
               tObj_FD1.SetDesField( desingVarVec );
               tObj_FD1.SetNLDiffusionCoeff( tMatCoeff );

               double energyDissipFD1 =tObj_FD1.Eval();

               desingVarVec[Ia] -=2.0*epsilon;
               
               mfem::EnergyDissipationObjective tObj_FD2;
               tObj_FD2.SetNLDiffusionSolver( solver );
               tObj_FD2.SetPreassure( &tPreassureGF);
               tObj_FD2.SetDesignFES( fes );
               tObj_FD2.SetDesField( desingVarVec );
               tObj_FD2.SetNLDiffusionCoeff( tMatCoeff );

               double energyDissipFD2 =tObj_FD2.Eval();

               desingVarVec[Ia] +=epsilon;

               tFD_sens[Ia] = (energyDissipFD1-energyDissipFD2)/(2.0*epsilon);
            }
               objgrad.Print();
               std::cout<<"  ----------  FD Diff ------------"<<std::endl;
               tFD_sens.Print();

               std::cout<<"  ---------- Analytic - FD Diff ------------"<<std::endl;
               mfem::ParGridFunction tFD_diff(fes); tFD_diff = 0.0;
               tFD_diff = objgrad;
               tFD_diff -=tFD_sens;
               tFD_diff.Print();
                                       std::cout<<"norm: "<<tFD_diff.Norml2()<<std::endl;
         }

         if(BreakAfterFirstIt)
         {
             mfem::mfem_error("break before update");
         }

         double con=vol-maxVolAllowed;
         mma->Update(desingVarVec,objgrad,&con,&volgrad,xxmin,xxmax);

         std::string tDesingName = "DesingVarVec";
         desingVarVec.Save( tDesingName.c_str() );

         std::string tFieldName = "FieldVec";
         tPreassureGF.Save( tFieldName.c_str() );

         //save the design
         {
            //  paraview_dc.SetCycle(i);
            //  paraview_dc.SetTime(i*1.0);
            //  paraview_dc.Save();
         }
      }

   }


    delete mma;
    delete DesingCoeff;
   // delete vobj;
   // delete ivobj;
   // delete fsolv;

   delete solver;
   delete pmesh;
   // Flush output before MPI_finalize
#ifdef MFEM_USE_CALIPER
   mgr.flush();
#endif
   MPI_Finalize();

   return 0;
}


// if()
// {
//    //allocate the filter
//    mfem::FilterSolver* fsolv=new mfem::FilterSolver(1.0,&pmesh);
//    fsolv->SetSolver(1e-8,1e-12,100,0);
//    fsolv->AddBC(1,1.0);
//    fsolv->AddBC(2,1.0);
//    fsolv->AddBC(3,1.0);
//    fsolv->AddBC(4,1.0);
// }


//    mfem::ParGridFunction pgdens(fsolv->GetFilterFES());
//    mfem::ParGridFunction oddens(fsolv->GetDesignFES());
//    mfem::Vector vdens; vdens.SetSize(fsolv->GetFilterFES()->GetTrueVSize()); vdens=0.0;
//    mfem::Vector vtmpv; vtmpv.SetSize(fsolv->GetDesignFES()->GetTrueVSize()); vtmpv=0.5;

//    fsolv->Mult(vtmpv,vdens);
//    pgdens.SetFromTrueDofs(vdens);


//    AlcoaBracket* alco=new AlcoaBracket(&pmesh,1);
//    alco->SetDesignFES(pgdens.ParFESpace());
//    alco->SetDensity(vdens);
//    //alco->Solve();
//    //mfem::ParGridFunction disp;
//    //alco->GetSol(4,1,1,1,disp);

//    //check gradients
//    /*
//    {
//        mfem::Vector prtv;
//        mfem::Vector tmpv;
//        mfem::Vector tgrad;
//        mfem::Vector fgrad;
//        prtv.SetSize(vtmpv.Size());
//        tmpv.SetSize(vtmpv.Size());
//        tgrad.SetSize(vtmpv.Size());
//        fgrad.SetSize(vdens.Size()); fgrad=0.0;
//        double val=alco->MeanCompliance();
//        alco->MeanCompliance(fgrad);
//        fsolv->MultTranspose(fgrad,tgrad);
//        prtv.Randomize();
//        double nd=mfem::InnerProduct(pmesh.GetComm(),prtv,prtv);
//        double td=mfem::InnerProduct(pmesh.GetComm(),prtv,tgrad);
//        td=td/nd;
//        double lsc=1.0;
//        double lqoi;
//        for(int l=0;l<10;l++){
//            lsc/=10.0;
//            prtv/=10.0;
//            add(prtv,vtmpv,tmpv);
//            fsolv->Mult(tmpv,vdens);
//            alco->SetDensity(vdens);
//            alco->Solve();
//            lqoi=alco->MeanCompliance();
//            double ld=(lqoi-val)/lsc;
//            if(myrank==0){
//                std::cout << "dx=" << lsc <<" FD approximation=" << ld/nd
//                          << " adjoint gradient=" << td
//                          << " err=" << std::fabs(ld/nd-td) << std::endl;
//            }
//        }
//    }*/

//    mfem::PVolumeQoI* vobj=new mfem::PVolumeQoI(fsolv->GetFilterFES());
//    //mfem::VolumeQoI* vobj=new mfem::VolumeQoI(fsolv->GetFilterFES());
//    vobj->SetProjection(0.2,8.0);//threshold 0.2

//    //compute the total volume
//    double tot_vol;
//    {
//        vdens=1.0;
//        tot_vol=vobj->Eval(vdens);
//    }
//    double max_vol=0.5*tot_vol;
//    if(myrank==0){ std::cout<<"tot vol="<<tot_vol<<std::endl;}


//    mfem::MFEMFinalizePetsc();
//    MPI_Finalize();
//    return 0;
// }
