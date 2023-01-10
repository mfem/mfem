//                                MFEM modified from Example 10 and 16
//
// Compile with: make imAMRMHDp
//
// Sample runs:
//
// Description:  this function will only support amr and implicit solvers
// Author: QT

#include "mfem.hpp"
#include "myCoefficient.hpp"
#include "BoundaryGradIntegrator.hpp"
#include "imResistiveMHDOperatorp.hpp"
#include "AMRResistiveMHDOperatorp.hpp"
#include "BlockZZEstimator.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;

double alpha; //a global value of magnetude for the pertubation
double Lx;  //size of x domain
double lambda;
double resiG;

//initial condition
double InitialPhi(const Vector &x)
{
    return cos(M_PI*x(1))*pow(cos(2.0*M_PI/Lx*x(0)), 3);
}

double InitialJ(const Vector &x)
{
   return -M_PI*M_PI*(1.0+4.0/Lx/Lx)*alpha*sin(M_PI*x(1))*cos(2.0*M_PI/Lx*x(0));
}

double InitialPsi(const Vector &x)
{
   return sin(M_PI*x(1))*cos(2.0*M_PI/Lx*x(0));
}

double InitialJ2(const Vector &x)
{
   return lambda/pow(cosh(lambda*(x(1)-.5)),2)
       -M_PI*M_PI*(1.0+4.0/Lx/Lx)*alpha*sin(M_PI*x(1))*cos(2.0*M_PI/Lx*x(0));
}

double InitialPsi2(const Vector &x)
{
   return log(cosh(lambda*(x(1)-.5)))/lambda
       +alpha*sin(M_PI*x(1))*cos(2.0*M_PI/Lx*x(0));
}


double InitialJ3(const Vector &x)
{
   double ep=.2;
   return (ep*ep-1.)/lambda/pow(cosh(x(1)/lambda) +ep*cos(x(0)/lambda), 2)
        -M_PI*M_PI*1.25*alpha*cos(.5*M_PI*x(1))*cos(M_PI*x(0));
}

double InitialPsi3(const Vector &x)
{
   double ep=.2;
   return -lambda*log( cosh(x(1)/lambda) +ep*cos(x(0)/lambda) )
          +alpha*cos(M_PI*.5*x(1))*cos(M_PI*x(0));
}

//this is an AMR update function for VSize (instead of TrueVSize)
//It is only called in the initial stage of AMR to generate an adaptive mesh
void AMRUpdate(BlockVector &S, BlockVector &S_tmp, Array<int> &offset,
               ParGridFunction &phi, ParGridFunction &psi, ParGridFunction &w, ParGridFunction &j,
               BlockVector &STrue, Array<int> &true_offset,
               ParGridFunction &phiTrue, ParGridFunction &psiTrue, ParGridFunction &wTrue)
{
   ParFiniteElementSpace* H1FESpace = phi.ParFESpace();

   //Link GridFunctions with STrue
   phiTrue.SetFromTrueDofs(STrue.GetBlock(0));
   psiTrue.SetFromTrueDofs(STrue.GetBlock(1));
     wTrue.SetFromTrueDofs(STrue.GetBlock(2));

   //update fem space
   H1FESpace->Update();

   //++++Part1: Update GridFunctions of VSize
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

   //++++Part 2: Update the GridFunctions of TrueVSize
   // Compute new dofs on the new mesh
   phiTrue.Update();
   psiTrue.Update();
     wTrue.Update();
 
   int true_size = H1FESpace->GetTrueVSize();

   //update offset vector
   true_offset[0] = 0;
   true_offset[1] = true_size;
   true_offset[2] = 2*true_size;
   true_offset[3] = 3*true_size;

   // Resize S
   STrue.Update(true_offset);

   // Compute "true" dofs and store them in STrue
   phi.GetTrueDofs(STrue.GetBlock(0));
   psi.GetTrueDofs(STrue.GetBlock(1));
     w.GetTrueDofs(STrue.GetBlock(2));

   H1FESpace->UpdatesFinished();
}

//this is an update function for block vector of TureVSize
void AMRUpdateTrue(BlockVector &S, 
               Array<int> &true_offset,
               ParGridFunction &phi,
               ParGridFunction &psi,
               ParGridFunction &w)
{
   FiniteElementSpace* H1FESpace = phi.FESpace();

   //++++an equivalent way:
   //phi.MakeTRef(H1FESpace, S, true_offset[0]);
   //psi.MakeTRef(H1FESpace, S, true_offset[1]);
   //  w.MakeTRef(H1FESpace, S, true_offset[2]);

   //phi.SetFromTrueVector();
   //psi.SetFromTrueVector();
   //  w.SetFromTrueVector();

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
   double dt = 0.0001;
   double visc = 1e-3;
   double resi = 1e-3;
   bool visit = false;
   //----amr coefficients----
   int amr_levels=0;
   double ltol_amr=1e-5;
   bool derefine = false;
   int precision = 8;
   int nc_limit = 3;         // maximum level of hanging nodes
   int ref_steps=2;
   //----end of amr----
   int icase = 1;
   alpha = 0.001; 
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
   else if (icase==3)
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

   //++++Read the mesh from the given mesh file.    
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

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
    delete pmesh;
    MPI_Finalize();
    return 2;
   }

   //-----------------------------------Generate adaptive grid---------------------------------
   //the first part of the code is copied for an explicit code to have a good initial adapative mesh
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
   BlockZZEstimator estimatorTmp(*integ, jTmp, *integ, psiTmp, flux_fespace1, flux_fespace2);

   ThresholdRefiner refinerTmp(estimatorTmp);
   refinerTmp.SetTotalErrorGoal(1e-7);    // total error goal (stop criterion)
   refinerTmp.SetLocalErrorGoal(1e-7);    // local error goal (stop criterion)
   refinerTmp.SetMaxElements(50000);
   refinerTmp.SetMaximumRefinementLevel(ser_ref_levels+par_ref_levels+1);
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
   jTmp.SetTrueVector();

   //--------------debug--------------
   fe_size = fespace.TrueVSize();
   Array<int> fe_offset3(4);
   fe_offset3[0] = 0;
   fe_offset3[1] = fe_size;
   fe_offset3[2] = 2*fe_size;
   fe_offset3[3] = 3*fe_size;

   BlockVector vx(fe_offset3);
   ParGridFunction phiTrue, psiTrue, wTrue; 
   phiTrue.MakeTRef(&fespace, vx, fe_offset3[0]);
   psiTrue.MakeTRef(&fespace, vx, fe_offset3[1]);
     wTrue.MakeTRef(&fespace, vx, fe_offset3[2]);

   //+++++Set the initial conditions, and the boundary conditions
   FunctionCoefficient phiInit(InitialPhi);
   phiTrue.ProjectCoefficient(phiInit);
   phiTrue.SetTrueVector();
   phiTrue.SetFromTrueVector(); 
   
   FunctionCoefficient psiInit3(InitialJ);
   psiTrue.ProjectCoefficient(psiInit3);
   psiTrue.SetTrueVector();
   psiTrue.SetFromTrueVector(); 

   FunctionCoefficient wInit(InitialPsi);
   wTrue.ProjectCoefficient(wInit);
   wTrue.SetTrueVector();
   wTrue.SetFromTrueVector();
   //--------------debug--------------
   
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
         vis_phi << "solution\n" << *pmesh << phiTrue;
         vis_phi << "window_size 800 800\n"<< "window_title '" << "phi'" << "keys cm\n";
         vis_phi << "pause\n"<<flush;

         vis_j.open(vishost, visport);
         vis_j << "parallel " << num_procs << " " << myid << "\n";
         vis_j.precision(8);
         vis_j << "solution\n" << *pmesh << psiTrue;
         vis_j << "window_size 800 800\n"<< "window_title '" << "psi'" << "keys cm\n";
         vis_j << "pause\n"<<flush;
         MPI_Barrier(MPI_COMM_WORLD);//without barrier, glvis may not open

         vis_w.open(vishost, visport);
         vis_w << "parallel " << num_procs << " " << myid << "\n";
         vis_w.precision(8);
         vis_w << "solution\n" << *pmesh << wTrue;
         vis_w << "window_size 800 800\n"<< "window_title '" << "omega'" << "keys cm\n";
         vis_w << "pause\n"<<flush;
         MPI_Barrier(MPI_COMM_WORLD);
      }
   }
   

   for (int ref_it = 1; ; ref_it++)
   {
     exOperator->UpdateJ(*vxTmp);
     refinerTmp.Apply(*pmesh);
     if (refinerTmp.Refined()==false)
     {
         break;
     }
     else
     {
         if (myid == 0) cout<<"Initial mesh refine..."<<endl;

         //AMRUpdate(*vxTmp, *vxTmp_old, fe_offset, phiTmp, psiTmp, wTmp, jTmp,
         //          vx, fe_offset3, phiTrue, psiTrue, wTrue);
         AMRUpdateTrue(vx, fe_offset3, phiTrue, psiTrue, wTrue);

         pmesh->Rebalance();

         //---Update problem---
         AMRUpdateTrue(vx, fe_offset3, phiTrue, psiTrue, wTrue);

         if (visualization)
         {
            phiTrue.SetFromTrueVector(); 
            vis_phi << "parallel " << num_procs << " " << myid << "\n";
            vis_phi << "solution\n" << *pmesh << phiTrue<<"pause\n" << flush;
            vis_j << "parallel " << num_procs << " " << myid << "\n";
            vis_j << "solution\n" << *pmesh << psiTrue<<"pause\n"  << flush;
            vis_w << "parallel " << num_procs << " " << myid << "\n";
            vis_w << "solution\n" << *pmesh << wTrue<<"pause\n" << flush;
         }
         break;

         //exOperator->UpdateProblem();
         //exOperator->assembleProblem(ess_bdr); 
     }
   }
   //-----------------------------------End of generating adaptive grid---------------------------------


   //++++++Save the solutions.
   {
      //phiTrue.SetFromTrueVector(); psiTrue.SetFromTrueVector(); wTrue.SetFromTrueVector();

      ostringstream mesh_name, phi_name, psi_name, w_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      phi_name << "sol_phi." << setfill('0') << setw(6) << myid;
      psi_name << "sol_psi." << setfill('0') << setw(6) << myid;
      w_name << "sol_omega." << setfill('0') << setw(6) << myid;

      ofstream omesh(mesh_name.str().c_str());
      omesh.precision(8);
      pmesh->Print(omesh);

      ofstream osol(phi_name.str().c_str());
      osol.precision(8);
      phiTrue.Save(osol);

      ofstream osol3(psi_name.str().c_str());
      osol3.precision(8);
      psiTrue.Save(osol3);

      ofstream osol4(w_name.str().c_str());
      osol4.precision(8);
      wTrue.Save(osol4);
   }

   //+++++Free the used memory.
   delete vxTmp_old;
   delete vxTmp;
   delete exOperator;
   delete pmesh;
   delete integ;

   MPI_Finalize();

   return 0;
}



