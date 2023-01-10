//                                MFEM modified from Example 10 and 16
//
// Compile with: make restart
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
double yrefine=0.2;
int icase = 1;

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
   bool local_refine = false;
   bool useStab = false; //use a stabilized formulation (explicit case only)
   int local_refine_levels = 2;
   const char *petscrc_file = "";
   int part_method=1;   //part_method 0 or 1 gives good results for a static adaptive mesh
   beta = 0.001; 
   Lx=3.0;
   lambda=5.0;
   double t = .0, told=.0;

   bool visualization = true;
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
   args.AddOption(&dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&t, "-t0", "--time", "Initial Time (for restart).");
   args.AddOption(&restart_count, "-restart_count", "--restart_count", "number of restarts have been performed");
   args.AddOption(&icase, "-i", "--icase",
                  "Icase: 1 - wave propagation; 2 - Tearing mode.");
   args.AddOption(&itau_, "-itau", "--itau",
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
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&local_refine, "-local", "--local-refine", "-no-local",
                  "--no-local-refine",
                  "Enable or disable local refinement before unifrom refinement.");
   args.AddOption(&local_refine_levels, "-lr", "--local-refine",
                  "Number of levels to refine locally.");
   args.AddOption(&yrefine, "-yrefine", "--y-region",
                  "Local refinement distance in y.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&part_method, "-part_method", "--partition-method",
                  "Partitioning method: 0-5 (see mfem on partitioning choices).");
   args.AddOption(&iUpdateJ, "-updatej", "--update-j",
                  "UpdateJ: 0 - no boundary condition used; 1 - Dirichlet used on J boundary.");
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
   args.AddOption(&usefd, "-fd", "--use-fd", "-no-fd",
                  "--no-fd",
                  "Use fd-fem in the implicit solvers.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&use_petsc, "-usepetsc", "--usepetsc", "-no-petsc",
                  "--no-petsc",
                  "Use or not PETSc to solve the nonlinear system.");
   args.AddOption(&use_factory, "-shell", "--shell", "-no-shell",
                  "--no-shell",
                  "Use user-defined preconditioner factory (PCSHELL).");
   args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                  "PetscOptions file to use.");

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

   if (t<1e-10)
   {
       cout<<"In restart time should be updated!"<<endl;
       MPI_Finalize();
       return 3;
   }

   if (myid == 0) args.PrintOptions(cout);

   if (use_petsc)
   {
      MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);
   }

   //+++++Read the mesh from a mesh file.    
   int dim = 2;
   ParMesh *pmesh;
   {
      ifstream ifs(MakeParFilename("ncmesh.", myid));
      MFEM_VERIFY(ifs.good(), "Mesh file not found.");
      pmesh = new ParMesh(MPI_COMM_WORLD, ifs);
   }

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
         delete pmesh;
         MPI_Finalize();
         return 3;
   }

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

   BlockVector vx(fe_offset);
   BlockVector vxold(fe_offset);

   ParGridFunction *psi, *phi, *w;

   ifstream ifs1(MakeParFilename("sol_psi.", myid));
   MFEM_VERIFY(ifs1.good(), "Solution file not found.");
   psi = new ParGridFunction(pmesh, ifs1);

   ifstream ifs2(MakeParFilename("sol_phi.", myid));
   MFEM_VERIFY(ifs2.good(), "Solution file not found.");
   phi = new ParGridFunction(pmesh, ifs2);

   ifstream ifs3(MakeParFilename("sol_omega.", myid));
   MFEM_VERIFY(ifs3.good(), "Solution file not found.");
   w = new ParGridFunction(pmesh, ifs3);

   // Compute "true" dofs and store them in vx
   phi->GetTrueDofs(vx.GetBlock(0));
   psi->GetTrueDofs(vx.GetBlock(1));
     w->GetTrueDofs(vx.GetBlock(2));

   phi->SetFromTrueDofs(vx.GetBlock(0));
   psi->SetFromTrueDofs(vx.GetBlock(1));
     w->SetFromTrueDofs(vx.GetBlock(2));

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
   if (icase==2)  //add the source term
   {
       oper.SetRHSEfield(E0rhs);
   }
   else if (icase==3 || icase==4 || icase==6)     
   {
       oper.SetRHSEfield(E0rhs3);
   }
   else if (icase==5)
       oper.SetRHSEfield(E0rhs5);

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
   oper.UpdateJ(vx, &j);

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
         vis_phi << "solution\n" << *pmesh << *phi;
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

         vis_w.open(vishost, visport);
         vis_w << "parallel " << num_procs << " " << myid << "\n";
         vis_w.precision(8);
         vis_w << "solution\n" << *pmesh << *w;
         vis_w << "window_size 800 800\n"<< "window_title '" << "omega'" << "keys cm\n";
         vis_w << flush;
         MPI_Barrier(MPI_COMM_WORLD);
      }
   }

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
        dc->RegisterField("psi", psi);
      }
      else if (icase==2)
      {
        dc = new VisItDataCollection("case2", pmesh);
        dc->RegisterField("psi", psi);
        dc->RegisterField("phi", phi);
        dc->RegisterField("omega", w);
      }
      else
      {
        dc = new VisItDataCollection("case3", pmesh);
        dc->RegisterField("current", &j);
        dc->RegisterField("psi", psi);
        dc->RegisterField("phi", phi);
        dc->RegisterField("omega", w);
      }

      bool par_format = false;
      dc->SetFormat(!par_format ?
                      DataCollection::SERIAL_FORMAT :
                      DataCollection::PARALLEL_FORMAT);
      dc->SetPrecision(8);
      dc->SetCycle(0);
      dc->SetTime(t);
      dc->Save();
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
         phi->SetFromTrueDofs(vx.GetBlock(0));
         psi->SetFromTrueDofs(vx.GetBlock(1));
         if (useStab){
            oper.assembleVoper(dt_real, phi, psi);
            oper.assembleBoper(dt_real, phi, psi);
         }
         else{
            oper.assembleNv(phi);
            oper.assembleNb(psi);
         }
         ode_solver->StepP(vx, t, dt_real);

         //---Corrector stage---
         //assemble the nonlinear terms (only psi is updated)
         psi->SetFromTrueDofs(vx.GetBlock(1));
         if (useStab)
            oper.assembleBoper(dt_real, phi, psi);
         else
            oper.assembleNb(psi);
         ode_solver->Step(vx, t, dt_real);
         oper.UpdatePhi(vx);
      }
      else
      {
         vxold=vx;
         told=t;
         ode_solver2->Step(vx, t, dt_real);

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

      if (last_step || (ti % vis_steps) == 0)
      {
         if (myid==0) cout << "step " << ti << ", t = " << t <<endl;
         phi->SetFromTrueDofs(vx.GetBlock(0));
         psi->SetFromTrueDofs(vx.GetBlock(1));
           w->SetFromTrueDofs(vx.GetBlock(2));

         if (visualization)
         {
             oper.UpdateJ(vx, &j);
             vis_phi << "parallel " << num_procs << " " << myid << "\n";
             vis_phi << "solution\n" << *pmesh << *phi;
             if (icase==1) 
                 vis_phi << "valuerange -.001 .001\n" << flush;
             else
                 vis_phi << flush;

             vis_j << "parallel " << num_procs << " " << myid << "\n";
             vis_j << "solution\n" << *pmesh << j << flush;
             vis_w << "parallel " << num_procs << " " << myid << "\n";
             vis_w << "solution\n" << *pmesh << *w << flush;
         }
         
         if(false)
         {
            if(icase!=3)
            {
                vis_phi << "parallel " << num_procs << " " << myid << "\n";
                vis_phi << "solution\n" << *pmesh << *psi;
            }
            else
            {
                vis_phi << "parallel " << num_procs << " " << myid << "\n";
                vis_phi << "solution\n" << *pmesh << *psi;
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
            if(!visualization)
              oper.UpdateJ(vx, &j);
            
            dc->SetCycle(ti);
            dc->SetTime(t);
            dc->Save();
         }
      }

   }

   MPI_Barrier(MPI_COMM_WORLD); 
   double end = MPI_Wtime();

   //++++++Save the solutions.
   if(true)
   {
      phi->SetFromTrueDofs(vx.GetBlock(0));
      psi->SetFromTrueDofs(vx.GetBlock(1));
        w->SetFromTrueDofs(vx.GetBlock(2));
      oper.UpdateJ(vx, &j);

      ostringstream phi_name, psi_name, w_name,j_name;
      phi_name << "sol_phirs." << setfill('0') << setw(6) << myid;
      psi_name << "sol_psirs." << setfill('0') << setw(6) << myid;
      w_name << "sol_omegars." << setfill('0') << setw(6) << myid;
      j_name << "sol_j." << setfill('0') << setw(6) << myid;

      ofstream osol(phi_name.str().c_str());
      osol.precision(16);
      phi->Save(osol);

      ofstream osol3(psi_name.str().c_str());
      osol3.precision(16);
      psi->Save(osol3);

      ofstream osol4(w_name.str().c_str());
      osol4.precision(16);
      w->Save(osol4);

      ofstream osol5(j_name.str().c_str());
      osol5.precision(8);
      j.Save(osol5);

      //output v1 and v2 for a comparision
      ParGridFunction v1(&fespace), v2(&fespace);
      oper.computeV(phi, &v1, &v2);
    
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
      oper.computeV(psi, &b1, &b2);
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
   delete ode_solver2;
   delete pmesh;
   delete dc;
   delete psi;
   delete phi;
   delete w;

   oper.DestroyHypre();

   if (use_petsc) { MFEMFinalizePetsc(); }

   MPI_Finalize();

   return 0;
}



