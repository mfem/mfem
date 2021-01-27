//                                MFEM modified from Example 10 and 16
//
// Compile with: make imScale
//
// Description:  It solves a time dependent ic problem (scaling test only)
// Author: QT
// 
// Example run:
// srun -n 4 imScale -m Meshes/xperiodic-new.mesh -rs 4 -rp 0 -o 2 -tf 1 -dt .1 -usepetsc --petscopts ./petscrc/rc_full -s 3 -shell -resi 1e-4 -visc 1e-4 -no-supg
//
// modify rc_full is important to achieve a good scaling when np is large on fine grids

#include "mfem.hpp"
#include "myCoefficient.hpp"
#include "myIntegrator.hpp"
#include "imScale.hpp"
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
int icase = 3;
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
   int ode_solver_type = 3;
   double t_final = 5.0;
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
   beta = 0.001; 
   Lx=3.0;
   lambda=5.0;

   bool slowStart=false;    //the first step might take longer than usual
   int vis_steps = 1;


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
   args.AddOption(&icase, "-i", "--icase",
                  "Icase: 1 - wave propagation; 2 - Tearing mode.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step","Time step.");
   args.AddOption(&beta, "-perturb", "--perturb","Pertuerbation in the initial condition.");
   args.AddOption(&im_supg, "-im_supg", "--im_supg",
                  "supg options in formulation");
   args.AddOption(&i_supgpre, "-i_supgpre", "--i_supgpre",
                  "supg preconditioner options in formulation");
   args.AddOption(&visc, "-visc", "--viscosity",
                  "Viscosity coefficient.");
   args.AddOption(&resi, "-resi", "--resistivity",
                  "Resistivity coefficient.");
   args.AddOption(&local_refine, "-local", "--local-refine", "-no-local","--no-local-refine",
                  "Enable or disable local refinement before unifrom refinement.");
   args.AddOption(&local_refine_levels, "-lr", "--local-refine",
                  "Number of levels to refine locally.");
   args.AddOption(&yrefine, "-yrefine", "--y-region",
                  "Local refinement distance in y.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&usesupg, "-supg", "--implicit-supg", "-no-supg", "--no-implicit-supg",
                  "Use supg in the implicit solvers.");
   args.AddOption(&paraview, "-paraview", "--paraview-datafiles", "-no-paraivew",
                  "--no-paraview-datafiles", "Save data files for paraview visualization.");
   args.AddOption(&use_petsc, "-usepetsc", "--usepetsc", "-no-petsc",
                  "--no-petsc",
                  "Use or not PETSc to solve the nonlinear system.");
   args.AddOption(&use_factory, "-shell", "--shell", "-no-shell", "--no-shell",
                  "Use user-defined preconditioner factory (PCSHELL).");
   args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                  "PetscOptions file to use.");
   args.AddOption(&iUpdateJ, "-updatej", "--update-j",
                  "UpdateJ: 0 - no boundary condition used; 1 - Dirichlet used on J boundary.");
   args.AddOption(&BgradJ, "-BgradJ", "--BgradJ",
                  "BgradJ: 1 - (B.grad J, phi); 2 - (-J, B.grad phi); 3 - (-B J, grad phi).");
   args.AddOption(&slowStart, "-slow", "--slow-start", "-no-slow", "--no-slow-start",
                  "Slow start");
   args.AddOption(&pa, "-pa", "--patial-assembly", "-no-pa",
                  "--no-partial-assembly", "Parallel assembly.");
   args.AddOption(&debug, "-debug", "--debug", "-no-debug", "--no-debug",
                  "Debug issue.");
   args.AddOption(&bctype, "-bctype", "--bctype","BC 1 - Dirichelt; 2 - weak Dirichelt.");
   args.AddOption(&useFull, "-useFull", "--useFull", "version of Full preconditioner");
   args.AddOption(&lumpedMass, "-lumpmass", "--lump-mass",  "-no-lumpmass", "--no-lump-mass",
                  "lumped mass for updatej=0");


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

   if (icase!=2)
   {
      lambda=.5/M_PI;
   }
   resiG=resi;
  
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

   pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   //Note rebalancing is probably not needed for a static adaptive mesh
   if (local_refine && false)
      pmesh->Rebalance();   


   //+++++Define the vector finite element spaces representing  [Psi, Phi, w]
   // in block vector bv, with offsets given by the fe_offset array.
   // All my fespace is 1D but the problem is multi-dimensional
   H1_FECollection fe_coll(order, dim);
   ParFiniteElementSpace fespace(pmesh, &fe_coll); 

   HYPRE_Int global_size = fespace.GlobalTrueVSize();

   int total_refine_levels = 2+ser_ref_levels+par_ref_levels;
   double gridSize = 2./pow(2., total_refine_levels);
   if (myid == 0)
   {
      cout<<"Number of total scalar unknowns: " << global_size << endl;
      cout<<"Total refinement levels = "<<total_refine_levels
           <<" effective grid size = "<<gridSize<<endl;
   }
   weakPenalty = 100./gridSize;
   //weakPenalty = 0.;

   int fe_size = fespace.TrueVSize();
   Array<int> fe_offset(4);
   fe_offset[0] = 0;
   fe_offset[1] = fe_size;
   fe_offset[2] = 2*fe_size;
   fe_offset[3] = 3*fe_size;

   BlockVector vx(fe_offset);
   ParGridFunction psi, phi, w;
   phi.MakeTRef(&fespace, vx, fe_offset[0]);
   psi.MakeTRef(&fespace, vx, fe_offset[1]);
     w.MakeTRef(&fespace, vx, fe_offset[2]);

   //+++++Set the initial conditions, and the boundary conditions
   FunctionCoefficient phiInit(InitialPhi);
   phi.ProjectCoefficient(phiInit);
   phi.SetTrueVector();

   if (icase==6)
   {
      FunctionCoefficient psiInit6(InitialPsi6);
      psi.ProjectCoefficient(psiInit6);
   }
   else if (icase==2)
   {
      FunctionCoefficient psiInit2(InitialPsi2);
      psi.ProjectCoefficient(psiInit2);
   }
   else
   {
      FunctionCoefficient psiInit3(InitialPsi3);
      psi.ProjectCoefficient(psiInit3);
   }
   psi.SetTrueVector();

   FunctionCoefficient wInit(InitialW);
   w.ProjectCoefficient(wInit);
   w.SetTrueVector();
   
   //this step is necessary to make sure unknows are updated!
   phi.SetFromTrueVector(); psi.SetFromTrueVector(); w.SetFromTrueVector();

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
   if (icase==5)
   {
    oper.SetRHSEfield(E0rhs5);
   }
   else if (icase==2)  //add the source term
   {
    oper.SetRHSEfield(E0rhs);
   }
   else
   {
    oper.SetRHSEfield(E0rhs3);
   }

   //set initial J
   ParGridFunction j(&fespace);
   if(icase==6)
   {
      FunctionCoefficient jInit6(InitialJ6);
      oper.SetInitialJ(jInit6);
      j.ProjectCoefficient(jInit6);
   }
   else if (icase==2)
   {
      FunctionCoefficient jInit2(InitialJ2);
      oper.SetInitialJ(jInit2);
      j.ProjectCoefficient(jInit2);
   }
   else
   {
      FunctionCoefficient jInit3(InitialJ3);
      oper.SetInitialJ(jInit3);
      j.ProjectCoefficient(jInit3);
   }
   j.SetTrueVector();

   double t = 0.0;
   oper.SetTime(t);

   ParaViewDataCollection *pd = NULL;
   if (paraview)
   {
      if (debug)
        pd = new ParaViewDataCollection("debug", pmesh);
      else
        pd = new ParaViewDataCollection("imScale", pmesh);
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

   ode_solver2->Init(oper);

   MPI_Barrier(MPI_COMM_WORLD); 
   double start = MPI_Wtime();

   //++++Perform time-integration (looping over the time iterations, ti, with a
   //    time-step dt).
   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      MPI_Barrier(MPI_COMM_WORLD); 
      double current_time = MPI_Wtime();
      //ignore the first time step
      if (ti==2)
      {
          MPI_Barrier(MPI_COMM_WORLD); 
          start = MPI_Wtime();
      }

      if (slowStart && ti==1)
      {
        double dt2=dt/2.;
        ode_solver2->Step(vx, t, dt2);
      }
      else
      {
        ode_solver2->Step(vx, t, dt);
      }
      last_step = (t >= t_final - 1e-8*dt);

      double comp_time = MPI_Wtime()-current_time;
      if (last_step || (ti % vis_steps) == 0)
      {
         if (paraview){
            if (myid==0) {
             cout << "save paraview solutions" <<endl;
            }
            psi.SetFromTrueVector();
            phi.SetFromTrueVector();
            w.SetFromTrueVector();
            oper.UpdateJ(vx, &j);

            pd->SetCycle(ti);
            pd->SetTime(t);
            pd->Save();
         }

         if (myid==0) {
             cout << "step " << ti << ", t = " << t <<endl;
             cout << "cpu time is " << comp_time <<endl;
         }
      }
   }
   MPI_Barrier(MPI_COMM_WORLD); 
   double end = MPI_Wtime();

   if (myid == 0) 
       cout <<"######Runtime = "<<end-start<<" ######"<<endl;

   //++++++Save the solutions.
   if (false)
   {
      phi.SetFromTrueVector(); psi.SetFromTrueVector(); w.SetFromTrueVector();
      oper.UpdateJ(vx, &j);

      ostringstream mesh_name, phi_name, psi_name, w_name,j_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      //mesh_name << "mesh";
      phi_name << "sol_phi." << setfill('0') << setw(6) << myid;
      psi_name << "sol_psi." << setfill('0') << setw(6) << myid;
      w_name << "sol_omega." << setfill('0') << setw(6) << myid;
      j_name << "sol_j." << setfill('0') << setw(6) << myid;

      ofstream omesh(mesh_name.str().c_str());
      omesh.precision(8);
      pmesh->Print(omesh);

      {
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
      }
   }

   //+++++Free the used memory.
   delete ode_solver;
   delete ode_solver2;
   delete pmesh;

   oper.DestroyHypre();

   if (use_petsc) { MFEMFinalizePetsc(); }

   MPI_Finalize();

   return 0;
}



