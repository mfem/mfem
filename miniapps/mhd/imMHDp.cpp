//                                MFEM modified from Example 10 and 16
//
// Compile with: make imMHDp
//
// Description:  It solves a time dependent resistive MHD problem 
//               There are three versions:
//               1. explicit scheme
//               2. implicit scheme using a very simple linear preconditioner
//               3. implicit scheme using physcis-based preconditioner
// Author: QT

#include "mfem.hpp"
#include "myCoefficient.hpp"
#include "myIntegrator.hpp"
#include "imResistiveMHDOperatorp.hpp"
#include "PCSolver.hpp"
#include "InitialConditions.hpp"
#include "localrefine.hpp"
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
int order = 2;
ParMesh *pmesh;

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
   int ode_solver_type = 2;
   double t_final = 5.0;
   double t_change = 0.;
   double dt = 0.0001;
   double visc = 1e-3;
   double resi = 1e-3;
   bool visit = false;
   bool paraview = false;
   bool use_petsc = false;
   bool use_factory = false;
   bool useStab = false; //use a stabilized formulation (explicit case only)
   bool local_refine = false;
   int local_refine_levels = 2;
   const char *petscrc_file = "";
   int part_method=1;   //part_method 0 or 1 gives good results for a static adaptive mesh
   beta = 0.001; 
   Lx=3.0;
   lambda=5.0;

   bool visualization = true;
   bool slowStart=false;    //the first step might take longer than usual
   bool saveOne=false;
   int vis_steps = 10;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refineP",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Backward Euler, 2 - Brailovskaya,\n\t"
                  "            3 - L-stable SDIRK23, 4 - L-stable SDIRK33,\n\t"
                  "            22 - Implicit Midpoint, 23 - SDIRK23, 24 - SDIRK34.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&t_change, "-tchange", "--t-change",
                  "dt change time; reduce to half.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&icase, "-i", "--icase",
                  "Icase: 1 - wave propagation; 2 - Tearing mode.");
   args.AddOption(&ijacobi, "-ijacobi", "--ijacobi",
                  "Number of jacobi iteration in preconditioner");
   args.AddOption(&im_supg, "-im_supg", "--im_supg",
                  "supg options in formulation");
   args.AddOption(&i_supgpre, "-i_supgpre", "--i_supgpre",
                  "supg preconditioner options in formulation");
   args.AddOption(&ex_supg, "-ex_supg", "--ex_supg",
                  "supg options in explicit formulation");
   args.AddOption(&itau_, "-itau", "--itau",
                  "tau options in supg.");
   args.AddOption(&visc, "-visc", "--viscosity",
                  "Viscosity coefficient.");
   args.AddOption(&resi, "-resi", "--resistivity",
                  "Resistivity coefficient.");
   args.AddOption(&ALPHA, "-alpha", "--hyperdiff",
                  "Numerical hyprediffusion coefficient.");
   args.AddOption(&beta, "-beta", "--perturb",
                  "Pertubation coefficient in initial conditions.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&local_refine, "-local", "--local-refine", "-no-local","--no-local-refine",
                  "Enable or disable local refinement before unifrom refinement.");
   args.AddOption(&local_refine_levels, "-lr", "--local-refine",
                  "Number of levels to refine locally.");
   args.AddOption(&yrefine, "-yrefine", "--y-region",
                  "Local refinement distance in y.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&part_method, "-part_method", "--partition-method",
                  "Partitioning method: 0-5 (see mfem on partitioning choices).");
   args.AddOption(&smoothOmega, "-smooth", "--smooth-omega", "-no-smooth", "--no-smooth-omega",
                  "Smooth omega in preconditioner.");
   args.AddOption(&usesupg, "-supg", "--implicit-supg", "-no-supg",
                  "--no-implicit-supg",
                  "Use supg in the implicit solvers.");
   args.AddOption(&useStab, "-stab", "--explicit-stab", "-no-stab","--no-explitcit-stab",
                  "Use supg in the explicit solvers.");
   args.AddOption(&maxtau, "-max-tau", "--max-tau", "-no-max-tau", "--no-max-tau",
                  "Use max-tau in supg.");
   args.AddOption(&useFull, "-useFull", "--useFull",
                  "version of Full preconditioner");
   args.AddOption(&usefd, "-fd", "--use-fd", "-no-fd", "--no-fd",
                  "Use fd-fem in the implicit solvers.");
   args.AddOption(&pa, "-pa", "--parallel-assembly", "-no-pa",
                  "--no-parallel-assembly", "Parallel assembly.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview-datafiles", "-no-paraivew",
                  "--no-paraview-datafiles", "Save data files for paraview visualization.");
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
   args.AddOption(&BgradJ, "-BgradJ", "--BgradJ",
                  "BgradJ: 1 - (B.grad J, phi); 2 - (-J, B.grad phi); 3 - (-B J, grad phi).");
   args.AddOption(&slowStart, "-slow", "--slow-start", "-no-slow", "--no-slow-start",
                  "Slow start");
   args.AddOption(&lumpedMass, "-lumpmass", "--lump-mass",  "-no-lumpmass", "--no-lump-mass",
                  "lumped mass for updatej=0");
   args.AddOption(&saveOne, "-saveOne", "--save-One",  "-no-saveOne", "--no-save-One",
                  "Save solution/mesh as one file");
   args.AddOption(&outputdpsi, "-outputdpsi", "--output-dpsi", "-no-outputdpsi",
                  "--no-output-dpsi", "Output dpsidt as Paraview.");
   args.AddOption(&supgoff, "-supgoff", "--supgoff", "-no-supgoff",
                  "--no-supgoff", "Turn on the off diagonal term in supg.");
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

   //+++++Read the mesh from the given mesh file.    
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   //++++Define the ODE solver used for time integration. Several implicit
   //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
   //    backward Euler methods are available.
   PCSolver *ode_solver=NULL;
   ODESolver *ode_solver2=NULL;
   bool explicitSolve=false;
   switch (ode_solver_type)
   {
      //Explicit methods (first-order Predictor-Corrector)
      case 2: ode_solver = new PCSolver; explicitSolve = true; break;
      //Implict L-stable methods 
      case 1: ode_solver2 = new BackwardEulerSolver; break;
      case 3: ode_solver2 = new SDIRK23Solver(2); break;
      case 4: ode_solver2 = new SDIRK33Solver; break;
      // Implicit A-stable methods (not L-stable)
      case 12: ode_solver2 = new ImplicitMidpointSolver; break;
      case 13: ode_solver2 = new SDIRK23Solver; break;
      case 14: ode_solver2 = new SDIRK34Solver; break;
     default:
         if (myid == 0) cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         delete mesh;
         MPI_Finalize();
         return 3;
   }

   //++++++Refine the mesh to increase the resolution.    
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }   

   //it can be performed before or after UniformRefinement
   Array<int> ordering;
   mesh->GetHilbertElementOrdering(ordering);
   mesh->ReorderElements(ordering);
   mesh->EnsureNCMesh();

   //++++++Refine locally first    
   if (local_refine)
   {
      for(int lev=0; lev<local_refine_levels; lev++)
      {

        Vector pt;
        Array<int> marked_elements;
        for (int i = 0; i < mesh->GetNE(); i++)
        {
           // check all nodes of the element
           IsoparametricTransformation T;
           mesh->GetElementTransformation(i, &T);
           for (int j = 0; j < T.GetPointMat().Width(); j++)
           {
              T.GetPointMat().GetColumnReference(j, pt);
              if (true)
              {
                double x0, y0;
                switch (lev)
                {
                    case 0: y0=0.5; break;
                    case 1: y0=0.3; break;
                    case 2: y0=0.2; break;
                    case 3: y0=0.18; x0=.08; break;
                    case 4: y0=0.16; x0=.05; break;
                    case 5: y0=0.15; x0=.04; break;
                    default:
                        if (myid == 0) cout << "Unknown level: " << lev << '\n';
                        delete mesh;
                        MPI_Finalize();
                        return 3;
                }
                if (lev<3){
                    if (yregion(pt, y0))
                    {
                       marked_elements.Append(i);
                       break;
                    }
                }
                else{
                    if (center_region(pt,x0,y0))
                    {
                       marked_elements.Append(i);
                       break;
                    }
                }
              }
              else
              {
                if (region(pt, lev))
                {
                   marked_elements.Append(i);
                   break;
                }
              }
           }
        }
        mesh->GeneralRefinement(marked_elements);
      }
   }

   if (false)
   {
      //***this is the old way to use metis***
      //+++++++here we need to generate a partitioning because the default one is wrong for ncmesh when local_refine is truned on
      int *partitioning = NULL;
      partitioning=mesh->GeneratePartitioning(num_procs, part_method);
      //output partitioning for debugging
      if (myid==0 && false) 
      {
         const char part_file[] = "partitioning.txt";
         ofstream opart(part_file);
         opart << "number_of_elements " << mesh->GetNE() << '\n'
               << "number_of_processors " << num_procs << '\n';
         for (int i = 0; i < mesh->GetNE(); i++)
         {
            opart << partitioning[i] << '\n';
         }
         cout << "Partitioning file: " << part_file << endl;
      }

      pmesh = new ParMesh(MPI_COMM_WORLD, *mesh, partitioning);
      delete partitioning;
   }
   else
   {
      if (part_method!=1)
      {
        if (myid==0) cout<<"======WARNING: custom part_method is not needed any more!======\n";
      }
      pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   }
   
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   //Here rebalancing may create some strange partitioning using certain partitioning methods (this is also old)
   //Note rebalancing is probably not needed for a static adaptive mesh
   if (local_refine && false)
      pmesh->Rebalance();   

   //+++++Define the vector finite element spaces representing  [Psi, Phi, w]
   // in block vector bv, with offsets given by the fe_offset array.
   // All my fespace is 1D but the problem is multi-dimensional
   H1_FECollection fe_coll(order, dim);
   ParFiniteElementSpace fespace(pmesh, &fe_coll); 

   HYPRE_Int global_size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of total scalar unknowns: " << global_size << endl;
   }

   int fe_size = fespace.TrueVSize();
   Array<int> fe_offset(4);
   fe_offset[0] = 0;
   fe_offset[1] = fe_size;
   fe_offset[2] = 2*fe_size;
   fe_offset[3] = 3*fe_size;

   //cout << "TrueVSize is: " << fe_size<<" id = "<<myid << endl;

   BlockVector vx(fe_offset), vxold(fe_offset);
   ParGridFunction psi, phi, w, psiBack(&fespace), psiPer(&fespace);
   phi.MakeTRef(&fespace, vx, fe_offset[0]);
   psi.MakeTRef(&fespace, vx, fe_offset[1]);
     w.MakeTRef(&fespace, vx, fe_offset[2]);

   //+++++Set the initial conditions, and the boundary conditions
   FunctionCoefficient phiInit(InitialPhi);
   phi.ProjectCoefficient(phiInit);
   phi.SetTrueVector();

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
   else if (icase==3 || icase==5)
   {
        FunctionCoefficient psiInit3(InitialPsi3);
        psi.ProjectCoefficient(psiInit3);
   }
   else if (icase==4)
   {
        FunctionCoefficient psiInit4(InitialPsi4);
        psi.ProjectCoefficient(psiInit4);
   }
   else if (icase==6)
   {
        FunctionCoefficient psiInit6(InitialPsi6);
        psi.ProjectCoefficient(psiInit6);
   }
   psi.SetTrueVector();

   FunctionCoefficient wInit(InitialW);
   w.ProjectCoefficient(wInit);
   w.SetTrueVector();
   
   //this step is necessary to make sure unknows are updated!
   phi.SetFromTrueVector(); psi.SetFromTrueVector(); w.SetFromTrueVector();

   //Set the background psi
   if (icase==1)
   {
        FunctionCoefficient psi0(BackPsi);
        psiBack.ProjectCoefficient(psi0);
   }
   else if (icase==2)
   {
        FunctionCoefficient psi02(BackPsi2);
        psiBack.ProjectCoefficient(psi02);
   }
   else if (icase==3 || icase==4 || icase==5 || icase==6)
   {
        FunctionCoefficient psi03(BackPsi3);
        psiBack.ProjectCoefficient(psi03);
   }
   psiBack.SetTrueVector();

   //++++++this is a periodic boundary condition in x and Direchlet in y 
   Array<int> ess_bdr(fespace.GetMesh()->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;  //set attribute 1 to Direchlet boundary fixed
   if(ess_bdr.Size()!=1 || false)
   {
    if (myid==0) cout <<"ess_bdr size should be 1 but it is "<<ess_bdr.Size()<<endl;
    delete ode_solver;
    delete ode_solver2;
    delete pmesh;
    if (use_petsc) { MFEMFinalizePetsc(); }
    MPI_Finalize();
    return 2;
   }

   //++++Initialize the MHD operator, the GLVis visualization    
   ResistiveMHDOperator oper(fespace, ess_bdr, visc, resi, use_petsc, use_factory);
   if (icase==1)
   {
       oper.SetRHSEfield(E0rhs1);
   }
   else if (icase==2)  //add the source term
   {
       oper.SetRHSEfield(E0rhs);
   }
   else if (icase==3 || icase==4 || icase==6)     
   {
       oper.SetRHSEfield(E0rhs3);
   }
   else if (icase==5)
   {
       oper.SetRHSEfield(E0rhs5);
   }

   ParGridFunction j(&fespace);
   //set initial J
   if (icase==1)
   {
        FunctionCoefficient jInit(InitialJ);
        oper.SetInitialJ(jInit);
        j.ProjectCoefficient(jInit);
   }
   else if (icase==2)
   {
        FunctionCoefficient jInit2(InitialJ2);
        oper.SetInitialJ(jInit2);
        j.ProjectCoefficient(jInit2);
   }
   else if (icase==3 || icase==5)
   {
        FunctionCoefficient jInit3(InitialJ3);
        oper.SetInitialJ(jInit3);
        j.ProjectCoefficient(jInit3);
   }
   else if (icase==4)
   {
        FunctionCoefficient jInit4(InitialJ4);
        oper.SetInitialJ(jInit4);
        j.ProjectCoefficient(jInit4);
   }
   else if (icase==6)
   {
        FunctionCoefficient jInit6(InitialJ6);
        oper.SetInitialJ(jInit6);
        j.ProjectCoefficient(jInit6);
   }
   j.SetTrueVector();

   socketstream vis_phi, vis_j, vis_psi, vis_w;
   subtract(psi,psiBack,psiPer);

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
         MPI_Barrier(MPI_COMM_WORLD);//without barrier, glvis may not open

         vis_j.open(vishost, visport);
         vis_j << "parallel " << num_procs << " " << myid << "\n";
         vis_j.precision(8);
         vis_j << "solution\n" << *pmesh << j;
         vis_j << "window_size 800 800\n"<< "window_title '" << "current'" << "keys cm\n";
         vis_j << flush;
         MPI_Barrier(MPI_COMM_WORLD);//without barrier, glvis may not open

         vis_psi.open(vishost, visport);
         vis_psi << "parallel " << num_procs << " " << myid << "\n";
         vis_psi.precision(8);
         vis_psi << "solution\n" << *pmesh << psiPer;
         vis_psi << "window_size 800 800\n"<< "window_title '" << "psi per'" << "keys cm\n";
         vis_psi << flush;
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

   double t = .0, told=.0;
   ConstantCoefficient zero(0.0); 
   double l2norm  = psiPer.ComputeL2Error(zero);
   if (myid==0) cout << "t="<<t<<" ln(|psiPer|_2) ="<<log(l2norm)<<endl;

   oper.SetTime(t);
   if (explicitSolve)
      ode_solver->Init(oper);
   else
      ode_solver2->Init(oper);

   // Create data collection for solution output: either VisItDataCollection for
   // ascii data files, or SidreDataCollection for binary data files.
   DataCollection *dc = NULL;
   if (visit)
   {
      if (icase==1)
      {
        dc = new VisItDataCollection("case1", pmesh);
        dc->RegisterField("psiPer", &psiPer);
      }
      else if (icase==2)
      {
        dc = new VisItDataCollection("case2", pmesh);
        dc->RegisterField("psiPer", &psiPer);
        dc->RegisterField("psi", &psi);
        dc->RegisterField("phi", &phi);
        dc->RegisterField("omega", &w);
      }
      else
      {
        dc = new VisItDataCollection("case3", pmesh);
        dc->RegisterField("current", &j);
        dc->RegisterField("psi", &psi);
        dc->RegisterField("phi", &phi);
        dc->RegisterField("omega", &w);
      }

      bool par_format = false;
      dc->SetFormat(!par_format ?
                      DataCollection::SERIAL_FORMAT :
                      DataCollection::PARALLEL_FORMAT);
      dc->SetPrecision(5);
      dc->SetCycle(0);
      dc->SetTime(t);
      dc->Save();
   }

   ParaViewDataCollection *pd = NULL;
   if (paraview)
   {
      pd = new ParaViewDataCollection("imMHDp", pmesh);
      pd->SetPrefixPath("ParaView");
      pd->RegisterField("psi", &psi);
      pd->RegisterField("phi", &phi);
      pd->RegisterField("omega", &w);
      pd->RegisterField("current", &j);
      pd->SetLevelsOfDetail(order);
      pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetHighOrderOutput(true);
      pd->SetCycle(0);
      pd->SetTime(0.0);
      pd->Save();
   }

   MPI_Barrier(MPI_COMM_WORLD); 
   double start = MPI_Wtime();

   //++++Perform time-integration (looping over the time iterations, ti, with a
   //    time-step dt).
   bool last_step = false;
   if(!useStab) ex_supg=0;
   for (int ti = 1; !last_step; ti++)
   {
      if (t_change>0. && t>=t_change)
      {
        dt=dt/2.;
        if (myid==0) cout << "change time step to "<<dt<<endl;
        t_change=0.;
      }
      double dt_real = min(dt, t_final - t);

      if (explicitSolve)
      {
         //---Predictor stage---
         //assemble the nonlinear terms
         phi.SetFromTrueVector(); 
         psi.SetFromTrueVector(); 
         if (useStab){
            oper.assembleVoper(dt_real, &phi, &psi);
            oper.assembleBoper(dt_real, &phi, &psi);
         }
         else{
            oper.assembleNv(&phi);
            oper.assembleNb(&psi);
         }
         ode_solver->StepP(vx, t, dt_real);

         //---Corrector stage---
         //assemble the nonlinear terms (only psi is updated)
         psi.SetFromTrueVector(); 
         if (useStab)
            oper.assembleBoper(dt_real, &phi, &psi);
         else
            oper.assembleNb(&psi);
         ode_solver->Step(vx, t, dt_real);
         oper.UpdatePhi(vx);
      }
      else
      {
         vxold=vx;
         told=t;

         if (slowStart && ti<2)
         {
           double dt2=dt/2.;
           ode_solver2->Step(vx, t, dt2);
         }
         else
         {
           ode_solver2->Step(vx, t, dt);
         }

         if (!oper.getConverged())
         {
            t=told;
            dt=dt/2.;
            dt_real = min(dt, t_final - t);
            oper.resetConverged();
            if (myid==0) cout << "====== reduced new dt = "<<dt<<endl;

            vx=vxold;
            ode_solver2->Step(vx, t, dt_real);

            if (!oper.getConverged())
                MFEM_ABORT("======ERROR: reduced time step once still failed; checkme!======");
         }
         
      }

      last_step = (t >= t_final - 1e-8*dt);

      if (myid==0) cout << "step " << ti << ", t = " << t <<endl;

      if (last_step || (ti % vis_steps) == 0)
      {

         if (visualization || visit || paraview){
            psi.SetFromTrueVector();
            phi.SetFromTrueVector();
            w.SetFromTrueVector();
            oper.UpdateJ(vx, &j);

            if (icase!=3)
            {
               subtract(psi,psiBack,psiPer);
               double l2norm  = psiPer.ComputeL2Error(zero);
               if (myid==0) cout << "t="<<t<<" ln(|psiPer|_2) ="<<log(l2norm)<<endl;
            }
         }

         //save solution as one file
         if (saveOne && false)
         {
            oper.UpdateJ(vx, &j);
            ostringstream j_name;
            j_name << "jone." << setfill('0') << setw(6) << ti;

            ofstream osolj(j_name.str().c_str());
            osolj.precision(8);
            j.SaveAsOne(osolj);
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
             vis_psi << "parallel " << num_procs << " " << myid << "\n";
             vis_psi << "solution\n" << *pmesh << psiPer << flush;
         }
         
         if(false)
         {
            if(icase!=3)
            {
                vis_phi << "parallel " << num_procs << " " << myid << "\n";
                vis_phi << "solution\n" << *pmesh << psiPer;
            }
            else
            {
                vis_phi << "parallel " << num_procs << " " << myid << "\n";
                vis_phi << "solution\n" << *pmesh << psi;
            }

            if (icase==1) 
            {
                vis_phi << "valuerange -.001 .001\n" << flush;
            }
            else
            {
                vis_phi << flush;
            }
         }

         if (visit)
         {
            dc->SetCycle(ti);
            dc->SetTime(t);
            dc->Save();
         }

         if (paraview)
         {
            pd->SetCycle(ti);
            pd->SetTime(t);
            pd->Save();
         }
      }

   }

   

   MPI_Barrier(MPI_COMM_WORLD); 
   double end = MPI_Wtime();

   //++++++Save the solutions.
   {
      phi.SetFromTrueVector(); 
      psi.SetFromTrueVector(); 
      w.SetFromTrueVector();
      oper.UpdateJ(vx, &j);

      ostringstream mesh_name, mesh_save, phi_name, psi_name, w_name,j_name;

      if (saveOne)
      {
         mesh_name << "full.mesh";
         j_name << "jfull.sol";
         w_name << "wfull.sol";
         psi_name << "psifull.sol";
         phi_name << "phifull.sol";

         ofstream mesh_ofs(mesh_name.str().c_str());
         ofstream osolj(j_name.str().c_str());
         ofstream osolw(w_name.str().c_str());
         ofstream osolpsi(psi_name.str().c_str());
         ofstream osolphi(phi_name.str().c_str());

         mesh_ofs.precision(16);
         osolj.precision(16);
         osolw.precision(16);
         osolpsi.precision(16);
         osolphi.precision(16);

         pmesh->PrintAsOne(mesh_ofs);
         j.SaveAsOne(osolj);
         w.SaveAsOne(osolw);
         psi.SaveAsOne(osolpsi);
         phi.SaveAsOne(osolphi);
      }
      else if (false)
      {
         mesh_name << "mesh." << setfill('0') << setw(6) << myid;
         mesh_save << "ncmesh." << setfill('0') << setw(6) << myid;
         phi_name << "sol_phi." << setfill('0') << setw(6) << myid;
         psi_name << "sol_psi." << setfill('0') << setw(6) << myid;
         w_name << "sol_omega." << setfill('0') << setw(6) << myid;
         j_name << "sol_j." << setfill('0') << setw(6) << myid;

         ofstream ncmesh(mesh_save.str().c_str());
         ncmesh.precision(16);
         pmesh->ParPrint(ncmesh);

         ofstream osol(phi_name.str().c_str());
         osol.precision(16);
         phi.Save(osol);

         ofstream osol3(psi_name.str().c_str());
         osol3.precision(16);
         psi.Save(osol3);

         ofstream osol4(w_name.str().c_str());
         osol4.precision(16);
         w.Save(osol4);

         ofstream osol5(j_name.str().c_str());
         osol5.precision(16);
         j.Save(osol5);
      }

      //those values can be generated in post-processing
      if (!paraview && !visit && false)
      {
        ofstream omesh(mesh_name.str().c_str());
        omesh.precision(8);
        pmesh->Print(omesh);

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
 
   }

   if (icase==1){
      FunctionCoefficient psiExact(exactPsi1); 
      FunctionCoefficient phiExact(exactPhi1); 
      FunctionCoefficient wExact(exactW1); 
      psiExact.SetTime(t_final);
      phiExact.SetTime(t_final);
      wExact.SetTime(t_final);
      double err_psi  = psi.ComputeL2Error(psiExact);
      double err_phi  = phi.ComputeL2Error(phiExact);
      double err_w  = w.ComputeL2Error(wExact);
      if (myid==0)
      {
        cout << "|| psi_h - psi_ex || = " << err_psi << "\n";
        cout << "|| phi_h - phi_ex || = " << err_phi << "\n";
        cout << "||   w_h -   w_ex || = " << err_w << "\n";
      }
   }

   if (myid == 0) 
   { 
       cout <<"######Runtime = "<<end-start<<" ######"<<endl;
   }

   //+++++Free the used memory.
   delete ode_solver;
   delete ode_solver2;
   delete pmesh;
   delete dc;
   delete pd;

   oper.DestroyHypre();

   if (use_petsc) { MFEMFinalizePetsc(); }

   MPI_Finalize();

   return 0;
}



