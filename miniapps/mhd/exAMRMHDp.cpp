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
#include "AMRResistiveMHDOperatorp.hpp"
#include "BlockZZEstimator.hpp"
#include "PCSolver.hpp"
#include "InitialConditions.hpp"
#include <memory>
#include <iostream>
#include <fstream>

void AMRUpdate(BlockVector &S, BlockVector &S_tmp,
               Array<int> &offset,
               ParGridFunction &phi,
               ParGridFunction &psi,
               ParGridFunction &w,
               ParGridFunction &j);


int main(int argc, char *argv[])
{
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 1. Parse command-line options.
   const char *mesh_file = "./Meshes/xperiodic-square.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 0;
   int order = 2;
   int ode_solver_type = 2;
   double t_final = 5.0;
   double dt = 0.0001;
   double visc = 0.0;
   double resi = 0.0;
   bool visit = false;
   //----amr coefficients----
   int amr_levels=0;
   double ltol_amr=1e-5;
   bool derefine = false;
   int precision = 8;
   int nc_limit = 3;         // maximum level of hanging nodes
   int ref_steps=1000;
   //----end of amr----
   int icase = 1;
   beta = 0.001; 
   Lx=3.0;
   lambda=5.0;

   bool visualization = true;
   int vis_steps = 200;

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
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
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
      if (myid == 0) cout <<"Unknown icase "<<icase<<endl;
      MPI_Finalize();
      return 3;
   }
   if (myid == 0) args.PrintOptions(cout);

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
     case 2: ode_solver = new PCSolver; break;
     default:
         if (myid == 0) cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         delete mesh;
         MPI_Finalize();
         return 3;
   }

   //++++Refine the mesh to increase the resolution.    
   mesh->EnsureNCMesh();
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }
   amr_levels+=ser_ref_levels;

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }
   amr_levels+=par_ref_levels;

   // 5. Define the vector finite element spaces representing 
   //  [Psi, Phi, w, j]
   // in block vector bv, with offsets given by the
   //    fe_offset array.
   // All my fespace is 1D bu the problem is multi-dimensional!!
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
    MPI_Finalize();
    return 2;
   }

   int fe_size = fespace.GetVSize();
   Array<int> fe_offset(5);
   fe_offset[0] = 0;
   fe_offset[1] = fe_size;
   fe_offset[2] = 2*fe_size;
   fe_offset[3] = 3*fe_size;
   fe_offset[4] = 4*fe_size;

   BlockVector vxTmp(fe_offset);
   ParGridFunction psi, phi, w, j;
   phi.MakeRef(&fespace, vxTmp.GetBlock(0), 0);
   psi.MakeRef(&fespace, vxTmp.GetBlock(1), 0);
     w.MakeRef(&fespace, vxTmp.GetBlock(2), 0);
     j.MakeRef(&fespace, vxTmp.GetBlock(3), 0);
   phi=0.0;
   psi=0.0;
   w=0.0;
   j=0.0;



   //-----------------------------------AMR---------------------------------
   int sdim = pmesh->SpaceDimension();
   BilinearFormIntegrator *integ = new DiffusionIntegrator;
   ParFiniteElementSpace flux_fespace1(pmesh, &fe_coll, sdim), flux_fespace2(pmesh, &fe_coll, sdim);
   BlockZZEstimator estimator(*integ, j, *integ, psi, flux_fespace1, flux_fespace2);
   //ZienkiewiczZhuEstimator estimator(*integ, w, flux_fespace1);
   //ZienkiewiczZhuEstimator estimator(*integ, j, flux_fespace1);

   ThresholdRefiner refiner(estimator);
   //refiner.SetTotalErrorFraction(0.0); // use purely local threshold   
   refiner.SetTotalErrorGoal(1e-7);    // total error goal (stop criterion)
   refiner.SetLocalErrorGoal(1e-7);    // local error goal (stop criterion)
   refiner.SetMaxElements(50000);
   refiner.SetMaximumRefinementLevel(ser_ref_levels+par_ref_levels+1);
   refiner.SetNCLimit(nc_limit);

   ThresholdDerefiner derefiner(estimator);
   derefiner.SetThreshold(.2*ltol_amr);
   derefiner.SetNCLimit(nc_limit);

   bool derefineMesh = false;
   bool refineMesh = false;
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
     operTmp.UpdateJ(vxTmp, &j);
     refiner.Apply(*pmesh);
     if (refiner.Refined()==false)
     {
         break;
     }
     else
     {
         if (myid == 0) cout<<"Initial mesh refine..."<<endl;
         AMRUpdate(vxTmp, vx_old, fe_offset, phi, psi, w, j);

         pmesh->Rebalance();

         //---Update problem---
         AMRUpdate(vxTmp, vx_old, fe_offset, phi, psi, w, j);
         operTmp.UpdateProblem();
         operTmp.assembleProblem(ess_bdr); 
     }
   }
   if (myid == 0) cout<<"Finish initial mesh refine..."<<endl;
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
       oper.SetRHSEfield(E0rhs);
   }
   else if (icase==3)
   {
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

   socketstream vis_phi, vis_j, vis_psi, vis_w;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      vis_phi.open(vishost, visport);
      if (!vis_phi)
      {
         if (myid == 0) 
            cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl
              << "GLVis visualization disabled.\n";

         visualization = false;
      }
      else
      {
         vis_phi << "parallel " << num_procs << " " << myid << "\n";
         vis_phi.precision(8);
         vis_phi << "solution\n" << *pmesh << phi;
         vis_phi << "window_size 800 800\n"<< "window_title '" << "phi'" << "keys cm\n";
         vis_phi << "pause\n";
         vis_phi << flush;
         if (myid==0)
            cout<< "GLVis visualization paused."
                << " Press space (in the GLVis window) to resume it.\n";
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
         vis_psi << "solution\n" << *pmesh << psi;
         vis_psi << "window_size 800 800\n"<< "window_title '" << "psi'" << "keys cm\n";
         vis_psi << flush;
         MPI_Barrier(MPI_COMM_WORLD);//without barrier, glvis may not open

         vis_w.open(vishost, visport);
         vis_w << "parallel " << num_procs << " " << myid << "\n";
         vis_w.precision(8);
         vis_w << "solution\n" << *pmesh << w;
         vis_w << "window_size 800 800\n"<< "window_title '" << "omega'" << "keys cm\n";
         vis_w << flush;
         MPI_Barrier(MPI_COMM_WORLD);//without barrier, glvis may not open
      }
   }

   double t = 0.0;
   oper.SetTime(t);
   ode_solver->Init(oper);
   bool last_step = false;

   //reset ltol_amr for full simulation
   refiner.SetMaximumRefinementLevel(amr_levels);
   refiner.SetTotalErrorGoal(ltol_amr);    // total error goal (stop criterion)
   refiner.SetLocalErrorGoal(ltol_amr);    // local error goal (stop criterion)

   // Create data collection for solution output: either VisItDataCollection for
   // ascii data files, or SidreDataCollection for binary data files.
   DataCollection *dc = NULL;
   if (visit)
   {
      if (icase==1)
      {
        dc = new VisItDataCollection("case1", pmesh);
        dc->RegisterField("psi", &psi);
        dc->RegisterField("current", &j);
      }
      else
      {
        if (icase==2)
            dc = new VisItDataCollection("case2", pmesh);
        else
            dc = new VisItDataCollection("case3", pmesh);

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

   MPI_Barrier(MPI_COMM_WORLD); 
   double start = MPI_Wtime();

   if (myid == 0) cout<<"Start time stepping..."<<endl;
   //---assemble problem and update boundary condition---
   oper.assembleProblem(ess_bdr); 
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

      //derefine every ref_steps but is lagged by a step of ref_steps/2
      if ( derefine && (ti-ref_steps/2)%ref_steps ==0 && ti >  ref_steps ) 
      {
          derefineMesh=true;
          derefiner.Reset();
      }
      else
          derefineMesh=false;

      {
        //---Predictor stage---
        //assemble the nonlinear terms
        oper.assembleNv(&phi);
        oper.assembleNb(&psi);
        ode_solver->StepP(vx, t, dt_real);
        oper.UpdateJ(vx, &j);

        //---Corrector stage---
        //assemble the nonlinear terms (only psi is updated)
        oper.assembleNb(&psi);
        ode_solver->Step(vx, t, dt_real);
        oper.UpdateJ(vx, &j); 
        oper.UpdatePhi(vx);

        last_step = (t >= t_final - 1e-8*dt);

        if (myid == 0 && (ti % 200)==0)
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

              AMRUpdate(vx, vx_old, fe_offset, phi, psi, w, j);
              pmesh->Rebalance();

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
                      vis_phi << "parallel " << num_procs << " " << myid << "\n";
                      vis_phi << "solution\n" << *pmesh << phi << flush;
                      vis_psi << "parallel " << num_procs << " " << myid << "\n";
                      vis_psi << "solution\n" << *pmesh << psi << flush;
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

            AMRUpdate(vx, vx_old, fe_offset, phi, psi, w, j);

            pmesh->Rebalance();

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
           vis_phi << "parallel " << num_procs << " " << myid << "\n";
           vis_phi << "solution\n" << *pmesh << phi <<flush;
           vis_psi << "parallel " << num_procs << " " << myid << "\n";
           vis_psi << "solution\n" << *pmesh << psi <<flush;
           vis_j << "parallel " << num_procs << " " << myid << "\n";
           vis_j << "solution\n" << *pmesh << j <<flush;
           vis_w << "parallel " << num_procs << " " << myid << "\n";
           vis_w << "solution\n" << *pmesh << w <<flush;
        }

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

   {
      ostringstream mesh_name, phi_name, psi_name, j_name, w_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      phi_name << "sol_phi." << setfill('0') << setw(6) << myid;
      psi_name << "sol_psi." << setfill('0') << setw(6) << myid;
      j_name << "sol_current." << setfill('0') << setw(6) << myid;
      w_name << "sol_omega." << setfill('0') << setw(6) << myid;

      ofstream omesh(mesh_name.str().c_str());
      omesh.precision(8);
      pmesh->Print(omesh);

      ofstream osol(phi_name.str().c_str());
      osol.precision(8);
      phi.Save(osol);

      ofstream osol2(j_name.str().c_str());
      osol2.precision(8);
      j.Save(osol2);

      ofstream osol3(psi_name.str().c_str());
      osol3.precision(8);
      psi.Save(osol3);

      ofstream osol4(w_name.str().c_str());
      osol4.precision(8);
      w.Save(osol4);
   }

   // 10. Free the used memory.
   delete ode_solver;
   delete pmesh;
   delete integ;
   delete dc;

   oper.DestroyHypre();
   MPI_Finalize();

   if (myid == 0) 
       cout <<"######Runtime = "<<end-start<<" ######"<<endl;

   return 0;
}


//this is an function modified based on amr/laghos.cpp
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

