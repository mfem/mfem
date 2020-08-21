//                                MFEM modified from Example 10 and 16
//
// Compile with: make imAMRMHDp
//
// Sample runs:
// mpirun -n 4 imAMRMHDp -m Meshes/xperiodic-new.mesh -rs 4 -rp 0 -o 3 -i 3 -tf 1 -dt .1 -usepetsc --petscopts petscrc/rc_debug -s 3 -shell -amrl 3 -ltol 1e-3 -derefine
//
// Description:  this function only supports amr and implicit solvers
// Author: QT

#include "mfem.hpp"
#include "myCoefficient.hpp"
#include "myIntegrator.hpp"
#include "imResistiveMHDOperatorp.hpp"
#include "AMRResistiveMHDOperatorp.hpp"
#include "BlockZZEstimator.hpp"
#include "PCSolver.hpp"
#include "InitialConditions.hpp"
#include <memory>
#include <iostream>
#include <fstream>

#ifndef MFEM_USE_PETSC
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
#endif
double beta;
double Lx;  
double lambda;
double resiG;
double ep=.2;
int icase = 1;


//this is an AMR update function for VSize (instead of TrueVSize)
//It is only called in the initial stage of AMR to generate an adaptive mesh
void AMRUpdate(BlockVector &S, BlockVector &S_tmp,
               Array<int> &offset,
               ParGridFunction &phi,
               ParGridFunction &psi,
               ParGridFunction &w,
               ParGridFunction &j)
{
   ParFiniteElementSpace* H1FESpace = phi.ParFESpace();

   //update fem space
   H1FESpace->Update();

   int fe_size = H1FESpace->GetVSize();

   //update offset vector
   offset[0] = 0;
   offset[1] = fe_size;
   offset[2] = 2*fe_size;
   offset[3] = 3*fe_size;
   offset[4] = 4*fe_size;

   S_tmp = S;
   S.Update(offset);
    
   const Operator* H1Update = H1FESpace->GetUpdateOperator();

   H1Update->Mult(S_tmp.GetBlock(0), S.GetBlock(0));
   H1Update->Mult(S_tmp.GetBlock(1), S.GetBlock(1));
   H1Update->Mult(S_tmp.GetBlock(2), S.GetBlock(2));
   H1Update->Mult(S_tmp.GetBlock(3), S.GetBlock(3));

   phi.MakeRef(H1FESpace, S, offset[0]);
   psi.MakeRef(H1FESpace, S, offset[1]);
     w.MakeRef(H1FESpace, S, offset[2]);
     j.MakeRef(H1FESpace, S, offset[3]);

   S_tmp.Update(offset);
   H1FESpace->UpdatesFinished();
}

//this is an update function for block vector of TureVSize
void AMRUpdateTrue(BlockVector &S, 
               Array<int> &true_offset,
               ParGridFunction &phi,
               ParGridFunction &psi,
               ParGridFunction &w,
               ParGridFunction &j)
{
   FiniteElementSpace* H1FESpace = phi.FESpace();

   //++++Update the GridFunctions so that they match S
   phi.SetFromTrueDofs(S.GetBlock(0));
   psi.SetFromTrueDofs(S.GetBlock(1));
   w.SetFromTrueDofs(S.GetBlock(2));

   //update fem space
   H1FESpace->Update();

   // Compute new dofs on the new mesh
   phi.Update();
   psi.Update();
   w.Update();
   
   // Note j stores data as a regular gridfunction
   j.Update();

   int fe_size = H1FESpace->GetTrueVSize();

   //update offset vector
   true_offset[0] = 0;
   true_offset[1] = fe_size;
   true_offset[2] = 2*fe_size;
   true_offset[3] = 3*fe_size;

   // Resize S
   S.Update(true_offset);

   // Compute "true" dofs and store them in S
   phi.GetTrueDofs(S.GetBlock(0));
   psi.GetTrueDofs(S.GetBlock(1));
     w.GetTrueDofs(S.GetBlock(2));

   H1FESpace->UpdatesFinished();
}

int main(int argc, char *argv[])
{
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   //++++Parse command-line options.
   const char *mesh_file = "./Meshes/xperiodic-square.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 0;
   int order = 2;
   int ode_solver_type = 2;
   double t_final = 5.0;
   double t_change = 0.;
   double dt = 0.0001;
   double visc = 1e-3;
   double resi = 1e-3;
   bool visit = false;
   bool use_petsc = false;
   bool use_factory = false;
   bool yRange = false;
   bool useStab = false; //use a stabilized formulation (explicit case only)
   const char *petscrc_file = "";

   //----amr coefficients----
   int amr_levels=0;
   double ltol_amr=1e-5;
   bool derefine = false;
   int precision = 8;
   int nc_limit = 1;         // maximum level of hanging nodes
   int ref_steps=4;
   double err_ratio=.1;
   double err_fraction=.5;
   double derefine_ratio=.2;
   //----end of amr----
   
   beta = 0.001; 
   Lx=3.0;
   lambda=5.0;

   bool visualization = true;
   int vis_steps = 10;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refineP",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&amr_levels, "-amrl", "--amr-levels",
                  "AMR refine level.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Backward Euler, 3 - L-stable SDIRK23, 4 - L-stable SDIRK33,\n\t"
                  "            22 - Implicit Midpoint, 23 - SDIRK23, 24 - SDIRK34.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&t_change, "-tchange", "--t-change",
                  "dt change time; reduce to half.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&icase, "-i", "--icase",
                  "Icase: 1 - wave propagation; 2 - Tearing mode.");
   args.AddOption(&itau, "-itau", "--itau",
                  "Itau options.");
   args.AddOption(&ijacobi, "-ijacobi", "--ijacobi",
                  "Number of jacobi iteration in preconditioner");
   args.AddOption(&im_supg, "-im_supg", "--im_supg",
                  "supg options in formulation");
   args.AddOption(&i_supgpre, "-i_supgpre", "--i_supgpre",
                  "supg preconditioner options in formulation");
   args.AddOption(&ex_supg, "-ex_supg", "--ex_supg",
                  "supg options in explicit formulation");
   args.AddOption(&visc, "-visc", "--viscosity",
                  "Viscosity coefficient.");
   args.AddOption(&resi, "-resi", "--resistivity",
                  "Resistivity coefficient.");
   args.AddOption(&ALPHA, "-alpha", "--hyperdiff",
                  "Numerical hyprediffusion coefficient.");
   args.AddOption(&beta, "-beta", "--perturb",
                  "Pertubation coefficient in initial conditions.");
   args.AddOption(&ltol_amr, "-ltol", "--local-tol",
                  "Local AMR tolerance.");
   args.AddOption(&err_ratio, "-err-ratio", "--err-ratio",
                  "AMR component ratio.");
   args.AddOption(&err_fraction, "-err-fraction", "--err-fraction",
                  "AMR error fraction in estimator.");
   args.AddOption(&derefine_ratio, "-derefine-ratio", "--derefine-ratio",
                  "AMR derefine error ratio.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&ref_steps, "-refs", "--refine-steps",
                  "Refine or derefine every n-th timestep.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&usesupg, "-supg", "--implicit-supg", "-no-supg",
                  "--no-implicit-supg",
                  "Use supg in the implicit solvers.");
   args.AddOption(&useStab, "-stab", "--explicit-stab", "-no-stab","--no-explitcit-stab",
                  "Use supg in the explicit solvers.");
   args.AddOption(&maxtau, "-max-tau", "--max-tau", "-no-max-tau", "--no-max-tau",
                  "Use max-tau in supg.");
   args.AddOption(&useFull, "-useFull", "--useFull",
                  "version of Full preconditioner");
   args.AddOption(&usefd, "-fd", "--use-fd", "-no-fd",
                  "--no-fd",
                  "Use fd-fem in the implicit solvers.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&derefine, "-derefine", "--derefine-mesh", "-no-derefine",
                  "--no-derefine-mesh",
                  "Derefine the mesh in AMR.");
   args.AddOption(&yRange, "-yrange", "--y-refine-range", "-no-yrange",
                  "--no-y-refine-range",
                  "Refine only in the y range of [-.6, .6] in AMR.");
   args.AddOption(&use_petsc, "-usepetsc", "--usepetsc", "-no-petsc",
                  "--no-petsc",
                  "Use or not PETSc to solve the nonlinear system.");
   args.AddOption(&use_factory, "-shell", "--shell", "-no-shell",
                  "--no-shell",
                  "Use user-defined preconditioner factory (PCSHELL).");
   args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                  "PetscOptions file to use.");
   args.AddOption(&iUpdateJ, "-updatej", "--update-j",
                  "UpdateJ: 0 - no boundary condition used; 1 - Dirichlet used on J boundary.");
   args.Parse();

   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (icase==2)
   {
      resiG=resi;
   }
   else if (icase==3 || icase==4 || icase==5 || icase==6)
   {
      lambda=.5/M_PI;
      resiG=resi;
   }
   else if (icase==1)
   {
       resi=.0;
       visc=.0;
   }
   else if (icase!=1)
   {
       if (myid == 0) cout <<"Unknown icase "<<icase<<endl;
       MPI_Finalize();
       return 3;
   }
   if (myid == 0) args.PrintOptions(cout);

   if (use_petsc)
   {
      MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);
   }

   //++++Read the mesh from the given mesh file.    
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   //++++Define the ODE solver used for time integration. Several implicit
   //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
   //    backward Euler methods are available.
   ODESolver *ode_solver=NULL;
   switch (ode_solver_type)
   {
      // Implict L-stable methods 
      case 1: ode_solver = new BackwardEulerSolver; break;
      case 3: ode_solver = new SDIRK23Solver(2); break;
      case 4: ode_solver = new SDIRK33Solver; break;
      // Implicit A-stable methods (not L-stable)
      case 12: ode_solver = new ImplicitMidpointSolver; break;
      case 13: ode_solver = new SDIRK23Solver; break;
      case 14: ode_solver = new SDIRK34Solver; break;
     default:
         if (myid == 0) cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         delete mesh;
         if (use_petsc) { MFEMFinalizePetsc(); }
         MPI_Finalize();
         return 3;
   }

   //++++Refine the mesh to increase the resolution.    
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }
   Array<int> ordering;
   mesh->GetHilbertElementOrdering(ordering);
   mesh->ReorderElements(ordering);
   mesh->EnsureNCMesh();    //note after this call all the mesh_level=0!!

   //amr_levels+=ser_ref_levels; this is not needed any more

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }
   amr_levels+=par_ref_levels;

   H1_FECollection fe_coll(order, dim);
   ParFiniteElementSpace fespace(pmesh, &fe_coll); 

   HYPRE_Int global_size = fespace.GlobalTrueVSize();
   if (myid == 0)
      cout << "Number of total scalar unknowns: " << global_size << endl;

   //this is a periodic boundary condition in x and Direchlet in y 
   Array<int> ess_bdr(fespace.GetMesh()->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;  //set attribute 1 to Direchlet boundary fixed
   if(ess_bdr.Size()!=1)
   {
    if (myid == 0) cout <<"ess_bdr size should be 1 but it is "<<ess_bdr.Size()<<endl;
    delete ode_solver;
    delete pmesh;
    if (use_petsc) { MFEMFinalizePetsc(); }
    MPI_Finalize();
    return 2;
   }

   //-----------------------------------Generate adaptive grid---------------------------------
   //the first part of the code is copied from an explicit code to have a good initial adapative mesh
   //If there is a simple way to initialize the mesh, then we can drop this part.
   //But last time I tried, the solver has some issue in terms of wrong ordering and refined levels 
   //after an adaptive mesh is saved and loaded. This is a simple work around for now.
   int fe_size = fespace.GetVSize();
   Array<int> fe_offset(5);
   fe_offset[0] = 0;
   fe_offset[1] = fe_size;
   fe_offset[2] = 2*fe_size;
   fe_offset[3] = 3*fe_size;
   fe_offset[4] = 4*fe_size;

   BlockVector *vxTmp = new BlockVector(fe_offset);
   ParGridFunction psiTmp, phiTmp, wTmp, jTmp;
   phiTmp.MakeRef(&fespace, vxTmp->GetBlock(0), 0);
   psiTmp.MakeRef(&fespace, vxTmp->GetBlock(1), 0);
     wTmp.MakeRef(&fespace, vxTmp->GetBlock(2), 0);
     jTmp.MakeRef(&fespace, vxTmp->GetBlock(3), 0);
   phiTmp=0.0;
     wTmp=0.0;

   int sdim = pmesh->SpaceDimension();
   BilinearFormIntegrator *integ = new DiffusionIntegrator;
   ParFiniteElementSpace flux_fespace1(pmesh, &fe_coll, sdim), flux_fespace2(pmesh, &fe_coll, sdim);
   BlockZZEstimator estimatorTmp(*integ, psiTmp, *integ, phiTmp, flux_fespace1, flux_fespace2);

   ThresholdRefiner refinerTmp(estimatorTmp);
   refinerTmp.SetTotalErrorGoal(1e-7);    // total error goal (stop criterion)
   refinerTmp.SetLocalErrorGoal(1e-7);    // local error goal (stop criterion)
   refinerTmp.SetMaxElements(500000);
   refinerTmp.SetMaximumRefinementLevel(par_ref_levels+1);
   refinerTmp.SetNCLimit(nc_limit);

   AMRResistiveMHDOperator *exOperator = new AMRResistiveMHDOperator(fespace, ess_bdr, visc, resi);
   BlockVector *vxTmp_old = new BlockVector(*vxTmp);
   exOperator->assembleProblem(ess_bdr); 

   //psi is needed to get solution started
   if (icase==1)
   {
        FunctionCoefficient psiInit(InitialPsi);
        psiTmp.ProjectCoefficient(psiInit);
   }
   else if (icase==2)
   {
        FunctionCoefficient psiInit2(InitialPsi2);
        psiTmp.ProjectCoefficient(psiInit2);
   }
   else if (icase==3)
   {
        FunctionCoefficient psiInit3(InitialPsi3);
        psiTmp.ProjectCoefficient(psiInit3);
   }
   else if (icase==4)
   {
        FunctionCoefficient psiInit4(InitialPsi4);
        psiTmp.ProjectCoefficient(psiInit4);
   }
   psiTmp.SetTrueVector();

   if (icase==1)
   {
        FunctionCoefficient jInit(InitialJ);
        jTmp.ProjectCoefficient(jInit);
   }
   else if (icase==2)
   {
        FunctionCoefficient jInit2(InitialJ2);
        jTmp.ProjectCoefficient(jInit2);
   }
   else if (icase==3)
   {
        FunctionCoefficient jInit3(InitialJ3);
        jTmp.ProjectCoefficient(jInit3);
   }
   else if (icase==4)
   {
        FunctionCoefficient jInit4(InitialJ4);
        jTmp.ProjectCoefficient(jInit4);
   }
   jTmp.SetTrueVector();

   for (int ref_it = 1; ref_it<5; ref_it++)
   {
     exOperator->UpdateJ(*vxTmp, &jTmp);
     refinerTmp.Apply(*pmesh);
     if (refinerTmp.Refined()==false)
     {
         break;
     }
     else
     {
         if (myid == 0) cout<<"Initial mesh refine..."<<endl;
         AMRUpdate(*vxTmp, *vxTmp_old, fe_offset, phiTmp, psiTmp, wTmp, jTmp);
         pmesh->Rebalance();
         //---Update problem---
         AMRUpdate(*vxTmp, *vxTmp_old, fe_offset, phiTmp, psiTmp, wTmp, jTmp);
         exOperator->UpdateProblem();
         exOperator->assembleProblem(ess_bdr); 
     }
   }
   if (myid == 0) cout<<"Finish initial mesh refine..."<<endl;
   global_size = fespace.GlobalTrueVSize();
   if (myid == 0)
      cout << "Number of total scalar unknowns becomes: " << global_size << endl;
   delete vxTmp_old;
   delete vxTmp;
   delete exOperator;
   //-----------------------------------End of generating adaptive grid---------------------------------

   //-----------------------------------Initial solution on adaptive grid---------------------------------
   fe_size = fespace.TrueVSize();
   Array<int> fe_offset3(4);
   fe_offset3[0] = 0;
   fe_offset3[1] = fe_size;
   fe_offset3[2] = 2*fe_size;
   fe_offset3[3] = 3*fe_size;

   BlockVector vx(fe_offset3);
   ParGridFunction phi, psi, w, j(&fespace); 
   phi.MakeTRef(&fespace, vx, fe_offset3[0]);
   psi.MakeTRef(&fespace, vx, fe_offset3[1]);
     w.MakeTRef(&fespace, vx, fe_offset3[2]);

   //+++++Set the initial conditions, and the boundary conditions
   FunctionCoefficient phiInit(InitialPhi);
   phi.ProjectCoefficient(phiInit);
   phi.SetTrueVector();
   phi.SetFromTrueVector(); 

   if (icase==1)
   {
        FunctionCoefficient psiInit(InitialPsi);
        psi.ProjectCoefficient(psiInit);
   }
   else if (icase==2)
   {
        FunctionCoefficient psiInit2(InitialPsi2);
        psi.ProjectCoefficient(psiInit2);
   }
   else if (icase==3)
   {
        FunctionCoefficient psiInit3(InitialPsi3);
        psi.ProjectCoefficient(psiInit3);
   }else if (icase==4)
   {
        FunctionCoefficient psiInit4(InitialPsi4);
        psi.ProjectCoefficient(psiInit4);
   }
   psi.SetTrueVector();
   psi.SetFromTrueVector(); 

   FunctionCoefficient wInit(InitialW);
   w.ProjectCoefficient(wInit);
   w.SetTrueVector();
   w.SetFromTrueVector();
   
   //++++Initialize the MHD operator, the GLVis visualization    
   ResistiveMHDOperator oper(fespace, ess_bdr, visc, resi, use_petsc, use_factory);
   if (icase==2)  //add the source term
   {
       oper.SetRHSEfield(E0rhs);
   }
   else if (icase==3 || icase==4)     
   {
       oper.SetRHSEfield(E0rhs3);
   }

   //set initial J
   FunctionCoefficient jInit1(InitialJ), jInit2(InitialJ2), 
                       jInit3(InitialJ3), jInit4(InitialJ4), *jptr;
   if (icase==1)
       jptr=&jInit1;
   else if (icase==2)
       jptr=&jInit2;
   else if (icase==3)
       jptr=&jInit3;
   else if (icase==4)
       jptr=&jInit4;
   j.ProjectCoefficient(*jptr);
   j.SetTrueVector();
   oper.SetInitialJ(*jptr);

   //-----------------------------------AMR for the real computation---------------------------------
   BlockZZEstimator estimator(*integ, psi, *integ, j, flux_fespace1, flux_fespace2);
   estimator.SetErrorRatio(err_ratio); //we define total_err = err_1 + ratio*err_2

   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(err_fraction);   // here 0.0 means we use local threshold; default is 0.5
   refiner.SetTotalErrorGoal(ltol_amr);  // total error goal (stop criterion)
   refiner.SetLocalErrorGoal(0.0);  // local error goal (stop criterion)
   refiner.SetMaxElements(5000000);
   refiner.SetMaximumRefinementLevel(amr_levels);
   refiner.SetNCLimit(nc_limit);
   if (yRange)
       refiner.SetYRange(-.6, .6);

   ThresholdDerefiner derefiner(estimator);
   derefiner.SetThreshold(derefine_ratio*ltol_amr);
   derefiner.SetNCLimit(nc_limit);

   bool derefineMesh = false;
   bool refineMesh = false;
   //-----------------------------------AMR---------------------------------

   socketstream vis_phi, vis_j, vis_psi, vis_w;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      vis_phi.open(vishost, visport);
      if (!vis_phi)
      {
          if (myid==0)
          {
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
            cout << "GLVis visualization disabled.\n";
          }
         visualization = false;
      }
      else
      {
         vis_phi << "parallel " << num_procs << " " << myid << "\n";
         vis_phi.precision(8);
         vis_phi << "solution\n" << *pmesh << phi;
         vis_phi << "window_size 800 800\n"<< "window_title '" << "phi'" << "keys cm\n";
         vis_phi << flush;

         vis_j.open(vishost, visport);
         vis_j << "parallel " << num_procs << " " << myid << "\n";
         vis_j.precision(8);
         vis_j << "solution\n" << *pmesh << j;
         vis_j << "window_size 800 800\n"<< "window_title '" << "current'" << "keys cm\n";
         vis_j << flush;
         MPI_Barrier(MPI_COMM_WORLD);//without barrier, glvis may not open

         vis_w.open(vishost, visport);
         vis_w << "parallel " << num_procs << " " << myid << "\n";
         vis_w.precision(8);
         vis_w << "solution\n" << *pmesh << w;
         vis_w << "window_size 800 800\n"<< "window_title '" << "omega'" << "keys cm\n";
         vis_w << flush;
         MPI_Barrier(MPI_COMM_WORLD);
      }
   }

   double t = 0.0;
   oper.SetTime(t);
   ode_solver->Init(oper);

   // Create data collection for solution output: either VisItDataCollection for
   // ascii data files, or SidreDataCollection for binary data files.
   DataCollection *dc = NULL;
   if (visit)
   {
      if (icase==1)
      {
        dc = new VisItDataCollection("case1", pmesh);
        dc->RegisterField("psi", &psi);
      }
      else if (icase==2)
      {
        dc = new VisItDataCollection("case2", pmesh);
        dc->RegisterField("psi", &psi);
        dc->RegisterField("phi", &phi);
        dc->RegisterField("omega", &w);
      }
      else
      {
        dc = new VisItDataCollection("case3", pmesh);
        dc->RegisterField("psi", &psi);
        dc->RegisterField("phi", &phi);
        dc->RegisterField("omega", &w);
      }
      dc->RegisterField("j", &j);

      bool par_format = false;
      dc->SetFormat(!par_format ?
                      DataCollection::SERIAL_FORMAT :
                      DataCollection::PARALLEL_FORMAT);
      dc->SetPrecision(5);
      dc->SetCycle(0);
      dc->SetTime(t);
      dc->Save();
   }

   MPI_Barrier(MPI_COMM_WORLD); 
   double start = MPI_Wtime();

   if (myid == 0) cout<<"Start time stepping..."<<endl;

   //++++Perform time-integration (looping over the time iterations, ti, with a time-step dt).
   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      double dt_real = min(dt, t_final - t);

      if ((ti % ref_steps) == 0)
      {
          refineMesh=true;
          refiner.Reset();
      }
      else
          refineMesh=false;

      /* 
       * here we derefine every 2*ref_steps but it is lagged by a step of 1.5*ref_steps
       * sometimes derefine could break down the preconditioner (maybe solutions are 
       * not so nice after a derefining projection?)
       */
      if ( derefine && (ti-ref_steps/2)%(2*ref_steps) ==0 ) //&& ti >  ref_steps ) //&& t<5.0) 
      {
          derefineMesh=true;
          derefiner.Reset();
      }
      else
          derefineMesh=false;

      //---the main solve step---
      ode_solver->Step(vx, t, dt_real);

      last_step = (t >= t_final - 1e-8*dt);
      if (last_step)
      {
          refineMesh=false;
          derefineMesh=false;
      }

      //update J and psi as it is needed in the refine or derefine step
      if (refineMesh || derefineMesh)
      {
          w.SetFromTrueDofs(vx.GetBlock(2));
          oper.UpdateJ(vx, &j);
          phi.SetFromTrueDofs(vx.GetBlock(0));
          psi.SetFromTrueDofs(vx.GetBlock(1));
      }

      if (myid == 0)
      {
          global_size = fespace.GlobalTrueVSize();
          cout << "Number of total scalar unknowns: " << global_size << endl;
          cout << "step " << ti << ", t = " << t <<endl;
      }

      //----------------------------AMR---------------------------------
      if (refineMesh)  refiner.Apply(*pmesh);
      if (refiner.Refined()==false || (!refineMesh))
      {
         if (derefine && derefineMesh && derefiner.Apply(*pmesh))
         {
            if (myid == 0) cout << "Derefined mesh..." << endl;

            //---Update solutions first---
            AMRUpdateTrue(vx, fe_offset3, phi, psi, w, j);
            oper.UpdateGridFunction();

            pmesh->Rebalance();

            //---Update solutions after rebalancing---
            AMRUpdateTrue(vx, fe_offset3, phi, psi, w, j);
            oper.UpdateGridFunction();

            //---assemble problem and update boundary condition---
            oper.UpdateProblem(ess_bdr); 
            oper.SetInitialJ(*jptr);    //somehow I need to reset the current bounary

            ode_solver->Init(oper);
         }
         else //mesh is not refined or derefined
         {
            if ( (last_step || (ti % vis_steps) == 0) )
            {
               if (visualization || visit)
               {
                  //for the plotting purpose we have to reset those solutions
                  phi.SetFromTrueDofs(vx.GetBlock(0));
                  psi.SetFromTrueDofs(vx.GetBlock(1));
                  w.SetFromTrueDofs(vx.GetBlock(2));
                  oper.UpdateJ(vx, &j);
               }

               if (visualization)
               {
                  vis_phi << "parallel " << num_procs << " " << myid << "\n";
                  vis_phi << "solution\n" << *pmesh << phi;
                  if (icase==1) 
                      vis_phi << "valuerange -.001 .001\n" << flush;
                  else
                      vis_phi << flush;

                  vis_j << "parallel " << num_procs << " " << myid << "\n";
                  vis_j << "solution\n" << *pmesh << j << flush;
                  vis_w << "parallel " << num_procs << " " << myid << "\n";
                  vis_w << "solution\n" << *pmesh << w << flush;
               }

               if (visit)
               {
                  dc->SetCycle(ti);
                  dc->SetTime(t);
                  dc->Save();
               }
            }

            if (last_step)
                break;
            else
                continue;
         }
      }
      else
      {
         if (myid == 0) cout<<"Mesh refine..."<<endl;

         //---Update solutions first---
         AMRUpdateTrue(vx, fe_offset3, phi, psi, w, j);
         oper.UpdateGridFunction();


         pmesh->Rebalance();

         //---Update problem after rebalancing---
         AMRUpdateTrue(vx, fe_offset3, phi, psi, w, j);
         oper.UpdateGridFunction();

         //---assemble problem and update boundary condition---
         oper.UpdateProblem(ess_bdr); 
         oper.SetInitialJ(*jptr);

         ode_solver->Init(oper);

     }
      //----------------------------AMR---------------------------------

      //++++always plot solutions when mesh is refined/derefined
      if (visualization || visit)
      {
         //J need to be updated again just for plotting
         //this may affect AMR??? no, the nc_limit was the issue
         //we cannot do nc_limit>1 with h dependent diffusion
         oper.UpdateJ(vx, &j);
      }

      if (visualization)
      {
         vis_phi << "parallel " << num_procs << " " << myid << "\n";
         vis_phi << "solution\n" << *pmesh << phi;
         if (icase==1) 
             vis_phi << "valuerange -.001 .001\n" << flush;
         else
             vis_phi << flush;

         vis_j << "parallel " << num_procs << " " << myid << "\n";
         vis_j << "solution\n" << *pmesh << j << flush;
         vis_w << "parallel " << num_procs << " " << myid << "\n";
         vis_w << "solution\n" << *pmesh << w << flush;
      }

      if (visit)
      {
         dc->SetCycle(ti);
         dc->SetTime(t);
         dc->Save();
      }

   }

   MPI_Barrier(MPI_COMM_WORLD); 
   double end = MPI_Wtime();

   //++++++Save the solutions.
   {
      phi.SetFromTrueDofs(vx.GetBlock(0));
      psi.SetFromTrueDofs(vx.GetBlock(1));
      w.SetFromTrueDofs(vx.GetBlock(2));
      oper.UpdateJ(vx, &j);

      ostringstream mesh_name, phi_name, psi_name, w_name, j_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      phi_name << "sol_phi." << setfill('0') << setw(6) << myid;
      psi_name << "sol_psi." << setfill('0') << setw(6) << myid;
      w_name << "sol_omega." << setfill('0') << setw(6) << myid;
      j_name << "sol_j." << setfill('0') << setw(6) << myid;

      ofstream omesh(mesh_name.str().c_str());
      omesh.precision(8);
      pmesh->Print(omesh);

      ofstream osol(phi_name.str().c_str());
      osol.precision(8);
      phi.Save(osol);

      ofstream osol3(psi_name.str().c_str());
      osol3.precision(8);
      psi.Save(osol3);

      ofstream osol4(w_name.str().c_str());
      osol4.precision(8);
      w.Save(osol4);

      ofstream osol5(j_name.str().c_str());
      osol5.precision(8);
      j.Save(osol5);

      //output v1 and v2 for a comparision
      ParGridFunction v1(&fespace), v2(&fespace);
      oper.computeV(&phi, &v1, &v2);

      ostringstream v1_name, v2_name;
      v1_name << "sol_v1." << setfill('0') << setw(6) << myid;
      v2_name << "sol_v2." << setfill('0') << setw(6) << myid;
      ofstream osol6(v1_name.str().c_str());
      osol6.precision(8);
      v1.Save(osol6);

      ofstream osol7(v2_name.str().c_str());
      osol7.precision(8);
      v2.Save(osol7);

      ParGridFunction b1(&fespace), b2(&fespace);
      oper.computeV(&psi, &b1, &b2);
      ostringstream b1_name, b2_name;
      b1_name << "sol_b1." << setfill('0') << setw(6) << myid;
      b2_name << "sol_b2." << setfill('0') << setw(6) << myid;
      ofstream osol8(b1_name.str().c_str());
      osol8.precision(8);
      b1.Save(osol8);

      ofstream osol9(b2_name.str().c_str());
      osol9.precision(8);
      b2.Save(osol9);
   }

   if (myid == 0) 
   { 
       cout <<"######Runtime = "<<end-start<<" ######"<<endl;
   }

   //+++++Free the used memory.
   delete ode_solver;
   delete pmesh;
   delete integ;
   delete dc;

   oper.DestroyHypre();

   if (use_petsc) { MFEMFinalizePetsc(); }

   MPI_Finalize();

   return 0;
}



