//                                MFEM modified from Example 10 and 16
//
// Compile with: make exAMRMHD
//
// Sample runs:
//    exAMRMHD -m xperiodic-square.mesh -r 2 -o 3 -tf 3 -vs 100 -dt .001 -visit
//    exAMRMHD -r 2 -tf 10 -vs 50 -dt .004 -i 2

//
// Description:  it solves a time dependent resistive MHD problem 
// Author: QT

#include "mfem.hpp"
#include "myCoefficient.hpp"
#include "myIntegrator.hpp"
#include "AMRResistiveMHDOperator.hpp"
#include "BlockZZEstimator.hpp"
#include "PCSolver.hpp"
#include "InitialConditions.hpp"
#include <memory>
#include <iostream>
#include <fstream>

void AMRUpdate(BlockVector &S,
               Array<int> &true_offset,
               GridFunction &phi,
               GridFunction &psi,
               GridFunction &w,
               GridFunction &j);

void AMRUpdate(BlockVector &S, BlockVector &S_tmp,
               Array<int> &true_offset,
               GridFunction &phi,
               GridFunction &psi,
               GridFunction &w,
               GridFunction &j);


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "./Meshes/xperiodic-square.mesh";
   int ref_levels = 2;
   int order = 2;
   int ode_solver_type = 2;
   int amr_levels=0;
   double t_final = 5.0;
   double dt = 0.0001;
   double visc = 0.0;
   double resi = 0.0;
   double ltol_amr=1e-5;
   bool visit = false;
   bool derefine = false;
   bool derefineIt = false;
   int precision = 8;
   int icase = 1;
   int nc_limit = 3;         // maximum level of hanging nodes
   beta = 0.001; 
   Lx=3.0;
   lambda=5.0;
   int ref_steps=1000;

   bool visualization = true;
   int vis_steps = 10;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&amr_levels, "-amrl", "--amr-levels",
                  "AMR refine level.");
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
   args.AddOption(&ltol_amr, "-ltol", "--local-tol",
                  "Local AMR tolerance.");
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
   args.AddOption(&derefine, "-derefine", "--derefine-mesh", "-no-derefine",
                  "--no-derefine-mesh",
                  "Derefine the mesh in AMR.");
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
   //ODESolver *ode_solver, *ode_predictor;
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
   mesh->EnsureNCMesh();
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }
   amr_levels+=ref_levels;

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
   cout << "Number of scalar unknowns: " << fe_size << endl;
   Array<int> fe_offset(5);
   fe_offset[0] = 0;
   fe_offset[1] = fe_size;
   fe_offset[2] = 2*fe_size;
   fe_offset[3] = 3*fe_size;
   fe_offset[4] = 4*fe_size;

   BlockVector vxTmp(fe_offset);
   GridFunction psi, phi, w, j;
   phi.MakeRef(&fespace, vxTmp.GetBlock(0), 0);
   psi.MakeRef(&fespace, vxTmp.GetBlock(1), 0);
     w.MakeRef(&fespace, vxTmp.GetBlock(2), 0);
     j.MakeRef(&fespace, vxTmp.GetBlock(3), 0);
   phi=0.0;
   psi=0.0;
   w=0.0;
   j=0.0;

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

   //-----------------------------------AMR---------------------------------
   int sdim = mesh->SpaceDimension();
   BilinearFormIntegrator *integ = new DiffusionIntegrator;
   FiniteElementSpace flux_fespace1(mesh, &fe_coll, sdim), flux_fespace2(mesh, &fe_coll, sdim);
   BlockZZEstimator estimator(*integ, j, *integ, psi, flux_fespace1, flux_fespace2);
   //ZienkiewiczZhuEstimator estimator(*integ, w, flux_fespace1);
   //ZienkiewiczZhuEstimator estimator(*integ, j, flux_fespace1);

   ThresholdRefiner refiner(estimator);
   //refiner.SetTotalErrorFraction(0.0); // use purely local threshold   
   refiner.SetTotalErrorGoal(1e-7);    // total error goal (stop criterion)
   refiner.SetLocalErrorGoal(1e-7);    // local error goal (stop criterion)
   refiner.SetMaxElements(50000);
   refiner.SetMaximumRefinementLevel(1);
   //refiner.SetYRange(-.75, .75);
   //refiner.PreferNonconformingRefinement();
   refiner.SetNCLimit(nc_limit);

   ThresholdDerefiner derefiner(estimator);
   derefiner.SetThreshold(.15*ltol_amr);
   derefiner.SetNCLimit(nc_limit);
   //-----------------------------------AMR---------------------------------

   //-----------------------------------Generate AMR grid---------------------------------
   AMRResistiveMHDOperator operTmp(fespace, ess_bdr, visc, resi);
   BlockVector vx_old(vxTmp);
   operTmp.assembleProblem(ess_bdr); 

   //psi is needed to get solution started
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

   for (int ref_it = 1; ; ref_it++)
   {
     operTmp.UpdateJ(vxTmp);
     refiner.Apply(*mesh);
     if (refiner.Refined()==false)
     {
         break;
     }
     else
     {
         cout<<"Initial mesh refine..."<<endl;
         //---Update problem---
         AMRUpdate(vxTmp, vx_old, fe_offset, phi, psi, w, j);
         operTmp.UpdateProblem();
         operTmp.assembleProblem(ess_bdr); 
     }
   }
   cout<<"Finish initial mesh refine..."<<endl;
   //-----------------------------------Generate AMR grid---------------------------------


   //-----------------------------------Initial solution on AMR grid---------------------------------
   BlockVector vx(fe_offset);
   phi.MakeRef(&fespace, vx.GetBlock(0), 0);
   psi.MakeRef(&fespace, vx.GetBlock(1), 0);
     w.MakeRef(&fespace, vx.GetBlock(2), 0);
     j.MakeRef(&fespace, vx.GetBlock(3), 0);
 
   FunctionCoefficient phiInit(InitialPhi);
   phi.ProjectCoefficient(phiInit);
   phi.SetTrueVector();

   AMRResistiveMHDOperator oper(fespace, ess_bdr, visc, resi);
   if (icase==2)  //add the source term
   {
       //FunctionCoefficient e0(E0rhs);
       oper.SetRHSEfield(E0rhs);
   }
   else if (icase==3)
   {
       //FunctionCoefficient e0(E0rhs3);
       oper.SetRHSEfield(E0rhs3);
   }

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
   GridFunction psiBack(&fespace);
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
   psiBack.SetTrueVector();

   socketstream vis_phi, vis_j, vis_psi, vis_w;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      vis_psi.open(vishost, visport);
      vis_phi.open(vishost, visport);
      vis_j.open(vishost, visport);
      vis_w.open(vishost, visport);
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
         vis_phi << "solution\n" << *mesh << phi;
         vis_phi << "window_size 800 800\n"<< "window_title '" << "phi'" << "keys cm\n";
         vis_phi << "pause\n";
         vis_phi << flush;
         vis_phi << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";

         vis_j.precision(8);
         vis_j << "solution\n" << *mesh << j;
         vis_j << "window_size 800 800\n"<< "window_title '" << "current'" << "keys cm\n";
         vis_j << flush;

         vis_psi.precision(8);
         vis_psi << "solution\n" << *mesh << psi;
         vis_psi << "window_size 800 800\n"<< "window_title '" << "psi'" << "keys cm\n";
         vis_psi << flush;

         vis_w.precision(8);
         vis_w << "solution\n" << *mesh << w;
         vis_w << "window_size 800 800\n"<< "window_title '" << "omega'" << "keys cm\n";
         vis_w << flush;
      }
   }

   double t = 0.0;
   oper.SetTime(t);
   ode_solver->Init(oper);
   bool last_step = false;

   //reset ltol_amr for full simulation
   refiner.SetTotalErrorGoal(ltol_amr);    // total error goal (stop criterion)
   refiner.SetLocalErrorGoal(ltol_amr);    // local error goal (stop criterion)
   refiner.SetMaximumRefinementLevel(amr_levels);

   // Create data collection for solution output: either VisItDataCollection for
   // ascii data files, or SidreDataCollection for binary data files.
   DataCollection *dc = NULL;
   if (visit)
   {
      if (icase==1)
      {
        dc = new VisItDataCollection("case1", mesh);
        dc->RegisterField("psi", &psi);
        dc->RegisterField("current", &j);
      }
      else
      {
        if (icase==2)
            dc = new VisItDataCollection("case2", mesh);
        else
            dc = new VisItDataCollection("case3", mesh);

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
   bool refineMesh;
   cout<<"Start time stepping..."<<endl;
   //---assemble problem and update boundary condition---
   oper.assembleProblem(ess_bdr); 
   for (int ti = 1; !last_step; ti++)
   {
      double dt_real = min(dt, t_final - t);

      if ((ti % ref_steps) == 0)
          refineMesh=true;
      else
          refineMesh=false;

      if ( derefine && ((ti-ref_steps/2) % ref_steps)==0 && ti >  ref_steps ) 
      {
          derefineIt=true;
          derefiner.Reset();
      }
      else
          derefineIt=false;

      if (refineMesh) refiner.Reset();

      {
        /*
        cout << "Number of unknowns: " << fespace.GetVSize() << endl;
        cout << "Number of elements in mesh: " << mesh->GetNE() << endl;
        */

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
        if ((ti % 20) == 0)
            cout << "step " << ti << ", t = " << t <<endl;

        //----------------------------AMR---------------------------------
        if (refineMesh)  refiner.Apply(*mesh);
        if (refiner.Refined()==false || (!refineMesh))
        {
           if (derefine && derefineIt && derefiner.Apply(*mesh))
           {
              cout << "Derefined mesh..." << endl;

              //---Update problem---
              AMRUpdate(vx, vx_old, fe_offset, phi, psi, w, j);
              oper.UpdateProblem();

              //---assemble problem and update boundary condition---
              oper.assembleProblem(ess_bdr); 
              ode_solver->Init(oper);
           }
           else //mesh is not refined or derefined
           {

                if ( (last_step || (ti % vis_steps) == 0) )
                {

                   if (visualization)
                   {
                        vis_phi << "solution\n" << *mesh << phi;
                        vis_psi << "solution\n" << *mesh << psi;
                        vis_j << "solution\n" << *mesh << j;
                        vis_w << "solution\n" << *mesh << w;

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

                if (last_step)
                    break;
                else
                    continue;
           }
        }
        else
        {
            cout<<"Mesh refine..."<<endl;
            //---Update problem---
            AMRUpdate(vx, vx_old, fe_offset, phi, psi, w, j);
            oper.UpdateProblem();

            //---assemble problem and update boundary condition---
            oper.assembleProblem(ess_bdr); 
            ode_solver->Init(oper);

        }
        //----------------------------AMR---------------------------------
        

        if (visualization)
        {
           vis_phi << "solution\n" << *mesh << phi;
           vis_psi << "solution\n" << *mesh << psi;
           vis_j << "solution\n" << *mesh << j;
           vis_w << "solution\n" << *mesh << w;

           if (icase==1) 
           {
               vis_phi << "valuerange -.001 .001\n" << flush;
               vis_j << "valuerange -.01425 .01426\n" << flush;
           }
        }

      }

      if (visit)
      {
         dc->SetCycle(ti);
         dc->SetTime(t);
         dc->Save();
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

      ofstream osol4("omega.sol");
      osol4.precision(8);
      w.Save(osol4);
   }

   // 10. Free the used memory.
   delete ode_solver;
   delete mesh;
   delete integ;
   delete dc;

   cout <<"######Runtime = "<<((double)end-start)/CLOCKS_PER_SEC<<" ######"<<endl;

   return 0;
}


void AMRUpdate(BlockVector &S, 
               Array<int> &true_offset,
               GridFunction &phi,
               GridFunction &psi,
               GridFunction &w,
               GridFunction &j)
{
   phi.SetFromTrueDofs(S.GetBlock(0));
   psi.SetFromTrueDofs(S.GetBlock(1));
   w.SetFromTrueDofs(S.GetBlock(2));
   j.SetFromTrueDofs(S.GetBlock(3));

   FiniteElementSpace* H1FESpace = phi.FESpace();

   //update fem space
   H1FESpace->Update();

   // Compute new dofs on the new mesh
   phi.Update();
   psi.Update();
   w.Update();
   j.Update();

   int fe_size = H1FESpace->GetTrueVSize();

   //update offset vector
   true_offset[0] = 0;
   true_offset[1] = fe_size;
   true_offset[2] = 2*fe_size;
   true_offset[3] = 3*fe_size;
   true_offset[4] = 4*fe_size;

   // Resize S
   S.Update(true_offset);

   // Compute "true" dofs and store them in S
   phi.GetTrueDofs(S.GetBlock(0));
   psi.GetTrueDofs(S.GetBlock(1));
     w.GetTrueDofs(S.GetBlock(2));
     j.GetTrueDofs(S.GetBlock(3));

   H1FESpace->UpdatesFinished();
}

//this is an function modified based on amr/laghos.cpp
void AMRUpdate(BlockVector &S, BlockVector &S_tmp,
               Array<int> &true_offset,
               GridFunction &phi,
               GridFunction &psi,
               GridFunction &w,
               GridFunction &j)
{
   FiniteElementSpace* H1FESpace = phi.FESpace();

   //update fem space
   H1FESpace->Update();

   int fe_size = H1FESpace->GetVSize();

   //update offset vector
   true_offset[0] = 0;
   true_offset[1] = fe_size;
   true_offset[2] = 2*fe_size;
   true_offset[3] = 3*fe_size;
   true_offset[4] = 4*fe_size;

   S_tmp = S;
   S.Update(true_offset);

   //ofstream myfile0("vxBefore.dat");
   //S_tmp.Print(myfile0, 100);
    
   const Operator* H1Update = H1FESpace->GetUpdateOperator();

   H1Update->Mult(S_tmp.GetBlock(0), S.GetBlock(0));
   H1Update->Mult(S_tmp.GetBlock(1), S.GetBlock(1));
   H1Update->Mult(S_tmp.GetBlock(2), S.GetBlock(2));
   H1Update->Mult(S_tmp.GetBlock(3), S.GetBlock(3));

   phi.MakeRef(H1FESpace, S, true_offset[0]);
   psi.MakeRef(H1FESpace, S, true_offset[1]);
     w.MakeRef(H1FESpace, S, true_offset[2]);
     j.MakeRef(H1FESpace, S, true_offset[3]);

   S_tmp.Update(true_offset);
   H1FESpace->UpdatesFinished();
}

