//                                MFEM modified from Example 10 and 16
//
// Compile with: make exMHD
//
// Sample runs:
//    exMHD -m xperiodic-square.mesh -r 2 -o 3 -tf 3 -vs 100 -dt .001 -visit
//    exMHD -r 2 -tf 10 -vs 50 -dt .004 -i 2

//
// Description:  it solves a time dependent resistive MHD problem 
// Author: QT

#include "mfem.hpp"
#include "myCoefficient.hpp"
#include "myIntegrator.hpp"
#include "ResistiveMHDOperator.hpp"
#include "PCSolver.hpp"
#include "InitialConditions.hpp"
#include <memory>
#include <iostream>
#include <fstream>

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "./Meshes/xperiodic-square.mesh";
   int ref_levels = 2;
   int order = 2;
   int ode_solver_type = 2;
   double t_final = 5.0;
   double dt = 0.0001;
   double visc = 0.0;
   double resi = 0.0;
   bool visit = false;
   int precision = 8;
   int icase = 1;
   beta = 0.001; 
   Lx=3.0;
   lambda=5.0;

   bool visualization = true;
   int vis_steps = 100;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 2 - Brailovskaya.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&icase, "-i", "--icase",
                  "Icase: 1 - wave propagation; 2 - Tearing mode.");
   args.AddOption(&visc, "-visc", "--viscosity",
                  "Viscosity coefficient.");
   args.AddOption(&resi, "-resi", "--resistivity",
                  "Resistivity coefficient.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (icase==2)
   {
      resi=.001;
      visc=.001;
      resiG=resi;
   }
   else if (icase==3)
   {
      lambda=.5/M_PI;
      resi=.001;
      visc=.001;
      resiG=resi;
   }
   else if (icase!=1)
   {
       cout <<"Unknown icase "<<icase<<endl;
       return 3;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file.    
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Define the ODE solver used for time integration. Several implicit
   //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
   //    explicit Runge-Kutta methods are available.
   PCSolver *ode_solver;
   switch (ode_solver_type)
   {
     // Explicit methods XXX: waring: FE is not stable 
     //case 1: ode_solver = new ForwardEulerSolver; break;
     //case 2: 
     //        ode_solver = new ForwardEulerSolver; 
     //        ode_predictor = new ForwardEulerSolver; 
     //        break; //first order predictor-corrector
     case 2: ode_solver = new PCSolver; break;
     default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         delete mesh;
         return 3;
   }

   // 4. Refine the mesh to increase the resolution.    
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define the vector finite element spaces representing 
   //  [Psi, Phi, w, j]
   // in block vector bv, with offsets given by the
   //    fe_offset array.
   // All my fespace is 1D bu the problem is multi-dimensional!!
   H1_FECollection fe_coll(order, dim);
   FiniteElementSpace fespace(mesh, &fe_coll); 
   if (false)
   {
        cout <<"dim="<<dim<<endl;
        cout <<"fespace dim="<<fespace.GetVDim()<<endl;
   }

   int fe_size = fespace.GetVSize();
   cout << "Number of total scalar unknowns: " << fe_size << endl;
   Array<int> fe_offset(5);
   fe_offset[0] = 0;
   fe_offset[1] = fe_size;
   fe_offset[2] = 2*fe_size;
   fe_offset[3] = 3*fe_size;
   fe_offset[4] = 4*fe_size;

   BlockVector vx(fe_offset);
   GridFunction psi, phi, w, j, psiBack(&fespace), psiPer(&fespace);
   phi.MakeRef(&fespace, vx.GetBlock(0), 0);
   psi.MakeRef(&fespace, vx.GetBlock(1), 0);
     w.MakeRef(&fespace, vx.GetBlock(2), 0);
     j.MakeRef(&fespace, vx.GetBlock(3), 0);

   // 6. Set the initial conditions, and the boundary conditions
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
   else if (icase==3)
   {
        FunctionCoefficient psiInit3(InitialPsi3);
        psi.ProjectCoefficient(psiInit3);
   }
   psi.SetTrueVector();

   FunctionCoefficient wInit(InitialW);
   w.ProjectCoefficient(wInit);
   w.SetTrueVector();

   if (icase==1)
   {
        FunctionCoefficient jInit(InitialJ);
        j.ProjectCoefficient(jInit);
   }
   else if (icase==2)
   {
        FunctionCoefficient jInit2(InitialJ2);
        j.ProjectCoefficient(jInit2);
   }
   else if (icase==3)
   {
        FunctionCoefficient jInit3(InitialJ3);
        j.ProjectCoefficient(jInit3);
   }
   j.SetTrueVector();

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
   else if (icase==3)
   {
        FunctionCoefficient psi03(BackPsi3);
        psiBack.ProjectCoefficient(psi03);
   }
   psiBack.SetTrueVector();

   //this is a periodic boundary condition in x and Direchlet in y 
   Array<int> ess_bdr(fespace.GetMesh()->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;  //set attribute 1 to Direchlet boundary fixed
   if(ess_bdr.Size()!=1)
   {
    cout <<"ess_bdr size should be 1 but it is "<<ess_bdr.Size()<<endl;
    delete ode_solver;
    delete mesh;
    return 2;
   }

   // 7. Initialize the MHD operator, the GLVis visualization    
   ResistiveMHDOperator oper(fespace, ess_bdr, visc, resi);
   if (icase==2)  //add the source term
   {
       FunctionCoefficient e0(E0rhs);
       oper.SetRHSEfield(e0);
   }
   else if (icase==3)
   {
       FunctionCoefficient e0(E0rhs3);
       oper.SetRHSEfield(e0);
   }

   socketstream vis_phi, vis_j;
   subtract(psi,psiBack,psiPer);
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      vis_phi.open(vishost, visport);
      vis_j.open(vishost, visport);
      if (!vis_phi)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
         visualization = false;
         cout << "GLVis visualization disabled.\n";
      }
      else
      {
         vis_phi.precision(8);
         vis_phi << "solution\n" << *mesh << psiPer;
         vis_phi << "window_size 800 800\n"<< "window_title '" << "psi per'" << "keys cm\n";

         if (icase==2)
            vis_phi << "valuerange -.001 .001\n";

         vis_phi << flush;

         vis_j << "solution\n" << *mesh << j;
         vis_j << "window_size 800 800\n"<< "window_title '" << "current'" << "keys cm\n";
         vis_j << flush;
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
        dc = new VisItDataCollection("case1", mesh);
        dc->RegisterField("psiPer", &psiPer);
        dc->RegisterField("current", &j);
      }
      else
      {
        dc = new VisItDataCollection("case2", mesh);
        dc->RegisterField("psiPer", &psiPer);
        dc->RegisterField("psi", &psi);
        dc->RegisterField("current", &j);
        dc->RegisterField("phi", &phi);
        dc->RegisterField("omega", &w);
      }

      dc->SetPrecision(precision);
      dc->SetCycle(0);
      dc->SetTime(t);
      dc->Save();
   }

   clock_t start = clock();
   // 8. Perform time-integration (looping over the time iterations, ti, with a
   //    time-step dt).
   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      double dt_real = min(dt, t_final - t);

      //---Predictor stage---
      //assemble the nonlinear terms
      oper.assembleNv(&phi);
      oper.assembleNb(&psi);
      ode_solver->StepP(vx, t, dt_real);
      oper.UpdateJ(vx);

      //---Corrector stage---
      //assemble the nonlinear terms (only psi is updated)
      oper.assembleNb(&psi);
      ode_solver->Step(vx, t, dt_real);
      oper.UpdateJ(vx); 
      oper.UpdatePhi(vx);

      last_step = (t >= t_final - 1e-8*dt);

      if (last_step || (ti % vis_steps) == 0)
      {
        cout << "step " << ti << ", t = " << t <<endl;
        subtract(psi,psiBack,psiPer);

         if (visualization)
         {
            if(icase!=3)
                vis_phi << "solution\n" << *mesh << psiPer;
            else
                vis_phi << "solution\n" << *mesh << w;

            vis_j << "solution\n" << *mesh << j;
            if (icase==1) 
            {
                vis_phi << "valuerange -.001 .001\n" << flush;
                vis_j << "valuerange -.01425 .01426\n" << flush;
            }
         }

         if (visit)
         {
            dc->SetCycle(ti);
            dc->SetTime(t);
            dc->Save();
         }
      }

   }
   clock_t end = clock();

   // 9. Save the solutions.
   {
      ofstream omesh("refined.mesh");
      omesh.precision(8);
      mesh->Print(omesh);

      ofstream osol("phi.sol");
      osol.precision(8);
      phi.Save(osol);

      ofstream osol2("current.sol");
      osol2.precision(8);
      j.Save(osol2);

      ofstream osol3("psi.sol");
      osol3.precision(8);
      psi.Save(osol3);

      //if (icase!=3)
      {
        subtract(psi,psiBack,psiPer);
        ofstream osol5("psiPer.sol");
        osol5.precision(8);
        psiPer.Save(osol5);
      }

      ofstream osol4("omega.sol");
      osol4.precision(8);
      w.Save(osol4);
   }

   // 10. Free the used memory.
   delete ode_solver;
   delete mesh;
   delete dc;
    
   cout <<"######Runtime = "<<((double)end-start)/CLOCKS_PER_SEC<<" ######"<<endl;

   return 0;
}



