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
#include "AMRupdate.hpp"
#include "checkpoint.hpp"
#include "../navier/ortho_solver.hpp"
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
ParMesh *pmesh;

int main(int argc, char *argv[])
{
   int num_procs, myid, myid_rand;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   srand(myid + 1);
   myid_rand=rand();

   //----Parse command-line options----
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
   bool paraview = false;
   bool use_petsc = false;
   bool use_factory = false;
   bool useStab = false; //use a stabilized formulation (explicit case only)
   bool initial_refine = false;
   bool compute_pressure = false;
   bool compute_tau = false;
   const char *petscrc_file = "";
   double t=0.;

   //----amr coefficients----
   int amr_levels=0;
   double ltol_amr=1e-5;
   bool derefine = false;
   int precision = 8;
   int nc_limit = 1;         // maximum level of hanging nodes
   int ref_steps=4;
   int iestimator=1;
   int derefine_op=1;
   int check_steps=50;
   double err_ratio=.1;
   double err_fraction=.5;
   double derefine_ratio=.2;
   double derefine_fraction=.05;
   int ref_its=1;
   int deref_its=1;
   double t_refs=1e10;
   int    t_refs_steps=4;
   bool     yRange = false; //fix a refinement region along y direction
   double   ytop =.5;       //top of the fixed yrange
   bool     xRange = false; //fix a refinement region along x direction
   double   xright =.5;     //right of the fixed xrange
   int      xlevels=0;
   double error_norm=infinity();
   //----end of amr----
   
   //----problem paramters----
   beta = 0.001; 
   Lx=3.0;
   lambda=5.0;

   bool checkpt=false;
   bool visualization = true;
   int vis_steps = 10;
   int restart_count=0;                 // restart_count for multiple restart

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
   args.AddOption(&t_refs, "-t-refs", "--t-refs",
                  "Time a quick refine/derefine is turned on.");
   args.AddOption(&t_refs_steps, "-t-refs-steps", "--t-refs-steps",
                  "Refine steps for a quick refine/derefine after t_refs.");
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
   args.AddOption(&ltol_amr, "-ltol", "--local-tol",
                  "Local AMR tolerance.");
   args.AddOption(&err_ratio, "-err-ratio", "--err-ratio",
                  "AMR component ratio.");
   args.AddOption(&err_fraction, "-err-fraction", "--err-fraction",
                  "AMR error fraction in estimator.");
   args.AddOption(&derefine, "-derefine", "--derefine-mesh", "-no-derefine",
                  "--no-derefine-mesh","Derefine the mesh in AMR.");
   args.AddOption(&derefine_ratio, "-derefine-ratio", "--derefine-ratio",
                  "AMR derefine error ratio of total_err_goal.");
   args.AddOption(&derefine_fraction, "-derefine-fraction", "--derefine-fraction",
                  "AMR derefine error fraction of total error (derefine if error is less than portion of total error).");
   args.AddOption(&derefine_op, "-derefine-op", "--derefine-op",
                  "AMR Derefine op - 0: minimum of the errors - 1: sum of the errors (default) - 2: maximum of the errors");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&usesupg, "-supg", "--implicit-supg", "-no-supg",
                  "--no-implicit-supg",
                  "Use supg in the implicit solvers.");
   args.AddOption(&useStab, "-stab", "--explicit-stab", "-no-stab","--no-explitcit-stab",
                  "Use supg in the explicit solvers.");
   args.AddOption(&maxtau, "-max-tau", "--max-tau", "-no-max-tau", "--no-max-tau",
                  "Use max-tau in supg.");
   args.AddOption(&dtfactor, "-dtfactor", "--dt-factor",
                  "Tau supg scales like dt/dtfactor.");
   args.AddOption(&factormin, "-factormin", "--factor-min", "Min factor in tau");
   args.AddOption(&useFull, "-useFull", "--useFull",
                  "version of Full preconditioner");
   args.AddOption(&usefd, "-fd", "--use-fd", "-no-fd",
                  "--no-fd","Use fd-fem in the implicit solvers.");
   args.AddOption(&pa, "-pa", "--parallel-assembly", "-no-pa",
                  "--no-parallel-assembly", "Parallel assembly.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles", "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview-datafiles", "-no-paraivew",
                  "--no-paraview-datafiles", "Save data files for paraview visualization.");
   args.AddOption(&error_norm, "-error-norm", "--error-norm","AMR error norm (in both refine and derefine).");
   args.AddOption(&yRange, "-yrange", "--y-refine-range", "-no-yrange", "--no-y-refine-range",
                  "Refine only in the y range of [-ytop, ytop] in AMR.");
   args.AddOption(&ytop, "-ytop", "--y-top",
                  "The top of yrange for AMR refinement.");
   args.AddOption(&xRange, "-xrange", "--x-refine-range", "-no-xrange", "--no-x-refine-range",
                  "Refine only in the x range of [-xright, xright] in AMR.");
   args.AddOption(&xright, "-xright", "--x-right",
                  "The right of xrange for AMR refinement.");
   args.AddOption(&xlevels, "-xlevels", "--x-levels",
                  "The minimal level for xRange being effective. Default is 0");
   args.AddOption(&use_petsc, "-usepetsc", "--usepetsc", "-no-petsc",
                  "--no-petsc",
                  "Use or not PETSc to solve the nonlinear system.");
   args.AddOption(&use_factory, "-shell", "--shell", "-no-shell",
                  "--no-shell",
                  "Use user-defined preconditioner factory (PCSHELL).");
   args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                  "PetscOptions file to use.");
   args.AddOption(&iUpdateJ, "-updatej", "--update-j",
                  "UpdateJ: 0 - no boundary condition used; 1/2 - Dirichlet used on J boundary (2: lumped mass matrix).");
   args.AddOption(&BgradJ, "-BgradJ", "--BgradJ",
                  "BgradJ: 1 - (B.grad J, phi); 2 - (-J, B.grad phi); 3 - (-B J, grad phi).");
   args.AddOption(&t, "-t0", "--time", "Initial Time (for restart).");
   args.AddOption(&restart_count, "-restart_count", "--restart_count", "number of restarts have been performed");
   args.AddOption(&checkpt, "-checkpt", "--check-pt",  "-no-checkpt", "--no-check-pt",
                  "Save check point");
   args.AddOption(&lumpedMass, "-lumpmass", "--lump-mass",  "-no-lumpmass", "--no-lump-mass",
                  "lumped mass for updatej=0");
   args.AddOption(&iestimator, "-iestimator", "--iestimator",
                  "iestimator: 1 - psi and J; 2 - omega and psi.");
   args.AddOption(&compute_tau, "-computetau", "--compute-tau", "-no-computetau",
                  "--no-compute-tau", "Compute tau in the post processing");
   args.AddOption(&compute_pressure, "-computep", "--compute-p", "-no-computep",
                  "--no-compute-p", "Compute pressure in the post processing");
   args.AddOption(&ref_its, "-ref-its", "--ref-its","refinement iterations.");
   args.AddOption(&deref_its, "-deref-its", "--deref-its","refinement iterations.");
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

   if (t<1e-10){
       if (myid == 0) cout<<"In restart time should be updated!"<<endl;
       MPI_Finalize();
       return 3;
   }

   if (use_petsc)
   {
      MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);
   }

   int dim = 2;

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
         if (use_petsc) { MFEMFinalizePetsc(); }
         MPI_Finalize();
         return 3;
   }

   //---- Load the mesh from checkpoint ----
   string mesh_name, phi_name, psi_name, w_name;
   string rs = to_string(restart_count-1);
   ifstream *ifs;
   if (restart_count==0){
       ifs = new ifstream(MakeParFilename("checkpt-mesh.", myid));
   }
   else if (restart_count==1){
       ifs = new ifstream(MakeParFilename("restart-mesh.", myid));
   }
   else{
       string mesh_name = "restart-mesh" + rs + ".";
       ifs = new ifstream(MakeParFilename(mesh_name, myid));
   }
   MFEM_VERIFY(ifs->good(), "Mesh file not found.");
   pmesh = new ParMesh(MPI_COMM_WORLD, *ifs);
   delete ifs;
   
   amr_levels+=par_ref_levels;
   if (xlevels>0) xlevels+=par_ref_levels;

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

   BilinearFormIntegrator *integ = new DiffusionIntegrator;
   int fe_size, sdim = pmesh->SpaceDimension();

   //-----------------------------------Initial solution on adaptive grid---------------------------------
   fe_size = fespace.TrueVSize();
   Array<int> fe_offset3(4);
   fe_offset3[0] = 0;
   fe_offset3[1] = fe_size;
   fe_offset3[2] = 2*fe_size;
   fe_offset3[3] = 3*fe_size;

   BlockVector vx(fe_offset3), vxold(fe_offset3), vk;
   ParGridFunction *phi, *psi, *w, j(&fespace); 

   // load mesh based on restart_count 0: start from an amr solver
   //                                  1: restart from the old restart solver
   //                                 >1: restart from the new restart solver
   ifstream *ifs1, *ifs2, *ifs3;
   if (restart_count==0){
       ifs1 = new ifstream(MakeParFilename("checkpt-psi.", myid));
       ifs2 = new ifstream(MakeParFilename("checkpt-phi.", myid));
       ifs3 = new ifstream(MakeParFilename("checkpt-w."  , myid));
   }
   else if (restart_count==1){
       ifs1 = new ifstream(MakeParFilename("restart-psi.", myid));
       ifs2 = new ifstream(MakeParFilename("restart-phi.", myid));
       ifs3 = new ifstream(MakeParFilename("restart-w."  , myid));
   }
   else{
       string phi_name = "restart-phi" + rs + ".";
       string psi_name = "restart-psi" + rs + ".";
       string   w_name = "restart-w"   + rs + ".";
       ifs1 = new ifstream(MakeParFilename(psi_name, myid));
       ifs2 = new ifstream(MakeParFilename(phi_name, myid));
       ifs3 = new ifstream(MakeParFilename(  w_name, myid));
   }
   MFEM_VERIFY(ifs1->good(), "Solution psi file not found.");
   MFEM_VERIFY(ifs2->good(), "Solution phi file not found.");
   MFEM_VERIFY(ifs3->good(), "Solution w   file not found.");
   psi = new ParGridFunction(pmesh, *ifs1);
   phi = new ParGridFunction(pmesh, *ifs2);
   w   = new ParGridFunction(pmesh, *ifs3);
   delete ifs1;
   delete ifs2;
   delete ifs3;

   // Compute "true" dofs and store them in vx
   phi->GetTrueDofs(vx.GetBlock(0));
   psi->GetTrueDofs(vx.GetBlock(1));
     w->GetTrueDofs(vx.GetBlock(2));

   phi->SetFromTrueDofs(vx.GetBlock(0));
   psi->SetFromTrueDofs(vx.GetBlock(1));
     w->SetFromTrueDofs(vx.GetBlock(2));

   phi->MakeTRef(&fespace, vx, fe_offset3[0]);
   psi->MakeTRef(&fespace, vx, fe_offset3[1]);
     w->MakeTRef(&fespace, vx, fe_offset3[2]);

   //++++Initialize the MHD operator, the GLVis visualization    
   ResistiveMHDOperator oper(fespace, ess_bdr, visc, resi, use_petsc, use_factory);
   //add the source term
   if (icase==2){
       oper.SetRHSEfield(E0rhs);
   }
   else if (icase==3 || icase==4){
       oper.SetRHSEfield(E0rhs3);
   }

   //set initial J
   FunctionCoefficient jInit1(InitialJ), jInit2(InitialJ2), jInit3(InitialJ3), jInit4(InitialJ4), *jptr;
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
   oper.UpdateJ(vx, &j);

   //-----------------------------------AMR for the real computation---------------------------------
   ErrorEstimator *estimator_used;
   BlockZZEstimator *estimator=NULL;
   BlockL2ZZEstimator *L2estimator=NULL;
   ParFiniteElementSpace flux_fespace1(pmesh, &fe_coll, sdim), flux_fespace2(pmesh, &fe_coll, sdim);
   RT_FECollection smooth_flux_fec(order-1, dim);
   ParFiniteElementSpace smooth_flux_fes1(pmesh, &smooth_flux_fec), smooth_flux_fes2(pmesh, &smooth_flux_fec);

   bool regularZZ=true;
   if (regularZZ)
   {
     if (iestimator==1){
        estimator=new BlockZZEstimator(*integ, *psi, *integ, j, flux_fespace1, flux_fespace2);
     }
     else{
        estimator=new BlockZZEstimator(*integ, *w, *integ, *psi, flux_fespace1, flux_fespace2);
     }
     estimator->SetErrorRatio(err_ratio); 
     estimator_used=estimator;
   }
   else{
     L2estimator=new BlockL2ZZEstimator(*integ, *psi, *integ, j, flux_fespace1, flux_fespace2, smooth_flux_fes1,smooth_flux_fes2);
     L2estimator->SetErrorRatio(err_ratio); 
     estimator_used=L2estimator;
   }

   int levels2=par_ref_levels+2, 
       levels3=par_ref_levels+3, 
       levels4=par_ref_levels+4, 
       levels5=par_ref_levels+5,
       levels6=par_ref_levels+6,
       levels7=par_ref_levels+7;
   ThresholdRefiner refiner(*estimator_used);
   refiner.SetTotalErrorFraction(err_fraction);   // here 0.0 means we use local threshold; default is 0.5
   refiner.SetTotalErrorGoal(0.0);       // this error goal is likely not used in the current example
   refiner.SetLocalErrorGoal(ltol_amr);  // local error goal (stop criterion)
   refiner.SetTotalErrorNormP(error_norm);
   refiner.SetMaxElements(10000000);
   if (amr_levels>levels7)
      refiner.SetMaximumRefinementLevel(levels7);
   else
      refiner.SetMaximumRefinementLevel(amr_levels);
   refiner.SetNCLimit(nc_limit);
   if (yRange) refiner.SetYRange(-ytop, ytop);
   if (xRange) refiner.SetXRange(-xright, xright, xlevels);

   ThresholdDerefiner derefiner(*estimator_used);
   derefiner.SetThreshold(derefine_ratio*ltol_amr);
   derefiner.SetNCLimit(nc_limit);
   derefiner.SetTotalErrorNormP(error_norm);
   derefiner.SetOp(derefine_op);
   if (derefine_fraction>=err_fraction && derefine)
   {   
       if (myid==0) cout << "ERROR: derefine_fraction is set to be large than err_fraction!!"<<endl;
       if (use_petsc) { MFEMFinalizePetsc(); }
       delete ode_solver;
       delete pmesh;
       delete integ;
       delete estimator_used;
       MPI_Finalize();
       return 3;
   }
   else
   { derefiner.SetTotalErrorFraction(derefine_fraction); }

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
         vis_phi << "solution\n" << *pmesh << *phi;
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
         vis_w << "solution\n" << *pmesh << *w;
         vis_w << "window_size 800 800\n"<< "window_title '" << "omega'" << "keys cm\n";
         vis_w << flush;
         MPI_Barrier(MPI_COMM_WORLD);
      }
   }

   double told=t;
   double dt0=dt, dt_min=0.0005;
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
        dc->RegisterField("psi", psi);
        dc->RegisterField("phi", phi);
        dc->RegisterField("omega", w);
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

   //save domain decompositino explicitly
   L2_FECollection pw_const_fec(0, dim);
   ParFiniteElementSpace pw_const_fes(pmesh, &pw_const_fec);
   ParGridFunction mpi_rank_gf(&pw_const_fes), tau_value(&pw_const_fes);
   ParLinearForm *computeTau=NULL;
   HypreParVector *tauv=NULL;
   mpi_rank_gf = myid_rand;

   //++++recover pressure and vector fields++++
   ParFiniteElementSpace *vfes;
   ParGridFunction *vel, *mag, *gradP, *BgradB, *gradBP, *gfv, *pre=NULL, *dpsidt=NULL;
   ParMixedBilinearForm *grad, *div;
   ParBilinearForm *a;
   ParNonlinearForm *convect;
   ParLinearForm *zLF, *zLFscalar; 
   ParBilinearForm *Mfull, *Mrot;
   Vector zv, zv2, zscalar, zscalar2;
   HypreParMatrix *MfullMat, *KMat;
   const IntegrationRule &ir = IntRules.Get(fespace.GetFE(0)->GetGeomType(), 3*order);
   CGSolver M_solver(MPI_COMM_WORLD), Mscal_solver;
   HypreSmoother *M_prec;
   HypreBoomerAMG *K_amg;
   CGSolver *K_pcg;
   mfem::navier::OrthoSolver *SpInvOrthoPC;
   Vector vtrue, rhs, vJxB;
   VectorDomainLFIntegrator *domainJxB;
   bool vfes_match=false;

   if(compute_pressure)
   {
      vfes = new ParFiniteElementSpace(pmesh, fespace.FEColl(), 2);
      vel = new ParGridFunction(vfes);
      mag = new ParGridFunction(vfes);
      gradP = new ParGridFunction(vfes);
      BgradB = new ParGridFunction(vfes);
      gradBP = new ParGridFunction(vfes);
      gfv = new ParGridFunction(vfes);
      pre = new ParGridFunction(&fespace);
      dpsidt = new ParGridFunction(&fespace);
      grad = new ParMixedBilinearForm(&fespace, vfes);
      div = new ParMixedBilinearForm(vfes, &fespace);
      convect = new ParNonlinearForm(vfes);
      zLF  = new ParLinearForm(vfes);
      zLFscalar = new ParLinearForm(&fespace);
      Mfull = new ParBilinearForm(vfes);
      Mrot = new ParBilinearForm(vfes);
      Mscal_solver=oper.GetM_solver2();

      int vfes_truevsize = vfes->GetTrueVSize();
      vtrue.SetSize(vfes_truevsize);
      rhs.SetSize(vfes_truevsize);
      vJxB.SetSize(vfes_truevsize);

      DenseMatrix A(2);
      A(0,0) = 0.0; A(0,1) =-1.0;
      A(1,0) = 1.0; A(1,1) = 0.0;
      MatrixConstantCoefficient coeff_curl(A);

      //mass matrix for vector fields
      Mfull->AddDomainIntegrator(new VectorMassIntegrator);
      Mfull->Assemble();
      Mfull->Finalize();
      MfullMat = Mfull->ParallelAssemble();

      M_solver.iterative_mode = false;
      M_solver.SetRelTol(1e-7);
      M_solver.SetAbsTol(0.0);
      M_solver.SetMaxIter(2000);
      M_solver.SetPrintLevel(0);
      M_prec = new HypreSmoother;  
      M_prec->SetType(HypreSmoother::Jacobi);
      M_solver.SetPreconditioner(*M_prec);
      M_solver.SetOperator(*MfullMat);

      //gradient operator from H1 to Vector H1
      grad->AddDomainIntegrator(new GradientIntegrator);
      grad->Assemble(); 

      //nonlinear convection term u.grad u
      convect->AddDomainIntegrator(new VectorConvectionNLFIntegrator);
      convect->Setup();

      //divergence operator from Vector H1 to H1
      div->AddDomainIntegrator(new VectorDivergenceIntegrator);
      div->Assemble();

      //rotation matrix
      Mrot->AddDomainIntegrator(new VectorMassIntegrator(coeff_curl));
      Mrot->Assemble();
      Mrot->Finalize();

       zv.SetSize(vfes->TrueVSize());
      zv2.SetSize(vfes->TrueVSize());
      zscalar.SetSize(fespace.TrueVSize());
      zscalar2.SetSize(fespace.TrueVSize());

      //compute velocity 
      grad->Mult(*phi, *zLF);
      zLF->ParallelAssemble(zv);
      M_solver.Mult(zv, zv2);
      vel->SetFromTrueDofs(zv2);

      //finalize with a rotation
      Mrot->Mult(*vel, *zLF);
      zLF->ParallelAssemble(zv);
      M_solver.Mult(zv, zv2);
      vel->SetFromTrueDofs(zv2);

      //compute B field
      grad->Mult(*psi, *zLF);
      zLF->ParallelAssemble(zv);
      M_solver.Mult(zv, zv2);
      mag->SetFromTrueDofs(zv2);

      //finalize with a rotation
      Mrot->Mult(*mag, *zLF);
      zLF->ParallelAssemble(zv);
      M_solver.Mult(zv, zv2);
      mag->SetFromTrueDofs(zv2);

      //compute -Δp=div(u.grad u - JxB)
      vel->GetTrueDofs(vtrue);
      convect->Mult(vtrue, rhs);  //nonlinear form only works with true dofs?

      JxBCoefficient JxBCoeff(&j, mag);
      domainJxB = new VectorDomainLFIntegrator(JxBCoeff);
      domainJxB->SetIntRule(&ir);
      ParLinearForm zJxB(vfes);
      zJxB.AddDomainIntegrator(domainJxB);
      zJxB.Assemble();
      zJxB.ParallelAssemble(vJxB);
      rhs.Add(-1.0, vJxB);
      
      //compute M^{-1}(u.grad u - JxB)
      M_solver.Mult(rhs, zv2);
      gfv->SetFromTrueDofs(zv2);
      div->Mult(*gfv, *zLFscalar);  //it is a mystery why ParGridFunction fails here

      a = new ParBilinearForm(&fespace);
      a->AddDomainIntegrator(new DiffusionIntegrator);
      a->Assemble();
      a->Finalize();
      KMat=a->ParallelAssemble();

      K_amg = new HypreBoomerAMG(*KMat);
      K_amg->SetPrintLevel(0);
      K_pcg = new CGSolver(MPI_COMM_WORLD);
      SpInvOrthoPC = new mfem::navier::OrthoSolver();
      SpInvOrthoPC->SetOperator(*K_amg);
      K_pcg->SetOperator(*KMat);
      K_pcg->iterative_mode = false;
      K_pcg->SetRelTol(1e-7);
      K_pcg->SetMaxIter(200);
      K_pcg->SetPrintLevel(0);
      K_pcg->SetPreconditioner(*SpInvOrthoPC);

      ParLinearForm b(&fespace);
      b.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(JxBCoeff), ess_bdr);
      b.Assemble();
      b.ParallelAssemble(zscalar);

      zLFscalar->ParallelAssemble(zscalar2);
      zscalar.Add(1.0, zscalar2);
      K_pcg->Mult(zscalar, zscalar2);
      pre->SetFromTrueDofs(zscalar2);
      *dpsidt = 0.0;

      //compute grad P
      zv=0.0;
      grad->TrueAddMult(zscalar2, zv);
      M_solver.Mult(zv, zv2);
      gradP->SetFromTrueDofs(zv2);

      //compute B.gradB
      mag->GetTrueDofs(vtrue);
      convect->Mult(vtrue, zv);  
      M_solver.Mult(zv, zv2);
      BgradB->SetFromTrueDofs(zv2);

      //compute grad magnetic pressure
      B2Coefficient B2Coeff(mag);
      ParLinearForm B2int(&fespace);
      B2int.AddDomainIntegrator(new DomainLFIntegrator(B2Coeff, 2, 0));
      B2int.Assemble();
      B2int.ParallelAssemble(zscalar);
      Mscal_solver.Mult(zscalar, zscalar2);
      zv=0.0;
      grad->TrueAddMult(zscalar2, zv);
      M_solver.Mult(zv, zv2);
      gradBP->SetFromTrueDofs(zv2);

      vfes_match=true;
   }

   ParaViewDataCollection *pd = NULL;
   if (paraview)
   {
      string pd_name = "restart-amr"+rs;
      pd = new ParaViewDataCollection(pd_name, pmesh);
      pd->SetPrefixPath("ParaView");
      pd->RegisterField("psi", psi);
      pd->RegisterField("phi", phi);
      pd->RegisterField("omega", w);
      pd->RegisterField("current", &j);
      pd->RegisterField("MPI rank", &mpi_rank_gf);
      if (compute_pressure){
          pd->RegisterField("V", vel);
          pd->RegisterField("B", mag);
          pd->RegisterField("pre", pre);
          pd->RegisterField("dpsidt", dpsidt);
          pd->RegisterField("grad pre", gradP);
          pd->RegisterField("grad mag pre", gradBP);
          pd->RegisterField("B.gradB", BgradB);
      }
      if(compute_tau){
          //visualize Tau value
          MyCoefficient velocity(phi, 2);
          computeTau = new ParLinearForm(&pw_const_fes);
          //need to multiply a time-step factor for SDIRK(2)!!
          computeTau->AddDomainIntegrator(new CheckTauIntegrator(0.29289321881*dt, resi, velocity, itau_));
          computeTau->Assemble(); 
          tauv=computeTau->ParallelAssemble();
          tau_value.SetFromTrueDofs(*tauv);
          pd->RegisterField("Tau", &tau_value);
      }

      pd->SetLevelsOfDetail(order);
      pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetHighOrderOutput(true);
      pd->SetCycle(0);
      pd->SetTime(t);
      pd->Save();
   }

   MPI_Barrier(MPI_COMM_WORLD); 
   double start = MPI_Wtime();
   bool reduced_step=false, meshChanged=false;
   int  success_step=0;
   cout.precision(16);

   if (myid == 0) cout<<"Start time stepping..."<<endl;

   //++++Perform time-integration (looping over the time iterations, ti, with a time-step dt).
   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      //change time step by user
      if (t_change>0. && t>=t_change)
      {
        dt=dt/2.;
        if (myid==0) cout << "change time step to "<<dt<<endl;
        t_change=0.;
      }

      //adjust refine frequency based on time step
      if (dt<=dt_min*2){
          ref_steps=12;
      }
      else if (dt<=dt_min*4){
          ref_steps=8;
      }
      else {
          ref_steps=4;
      }

      //increase time step when problem becomes nicer
      if (reduced_step){
          success_step++;
          if (success_step>10)
          {
              dt = min(dt*1.1, dt0);
              success_step=0;
              if (myid==0) cout << "increase time step to "<<dt<<endl;
          }
      }

      if (t>=5.2)
      {
          refiner.SetMaximumRefinementLevel(amr_levels);
      }

      double dt_real = min(dt, t_final - t);
      if (meshChanged)
      {
          dt_real*=0.25;
      }

      if ((ti % ref_steps) == 0){
          refineMesh=true;
          refiner.Reset();
          derefineMesh=true;
          derefiner.Reset();
      }
      else{
          refineMesh=false;
          derefineMesh=false;
      }

      vxold=vx;
      told=t;
 
      //---the main solve step---
      ode_solver->Step(vx, t, dt_real);

      //---reduce time step by half if it is too stiff---
      if (!oper.getConverged())
      {
         t=told;
         if (dt<=dt_min)
         {
            if (myid==0) cout << "====== the time step is already <= dt_min, give up for now ======"<<endl;
            break;
         }
         dt=max(dt/2., dt_min);
         dt_real = min(dt, t_final - t);
         if (meshChanged)
         {
            dt_real*=0.25;
         }
         oper.resetConverged();
         if (myid==0) cout << "====== reduced dt: new dt = "<<dt<<" ======"<<endl;

         //reset information for increasing time step
         reduced_step=true;
         success_step=0;

         vx=vxold;
         ode_solver->Step(vx, t, dt_real);

         if (!oper.getConverged()){
             MFEM_ABORT("======ERROR: reduced time step once still failed; checkme!======");
         }
      }

      last_step = (t >= t_final - 1e-8*dt);
      if (last_step){
          refineMesh=false;
          derefineMesh=false;
      }

      //update J and psi as it is needed in the refine or derefine step
      if (refineMesh || derefineMesh){
          phi->SetFromTrueDofs(vx.GetBlock(0));
          psi->SetFromTrueDofs(vx.GetBlock(1));
          w->SetFromTrueDofs(vx.GetBlock(2));
      }

      if (myid == 0){
          global_size = fespace.GlobalTrueVSize();
          cout << "Number of total scalar unknowns: " << global_size << endl;
          cout << "step " << ti << ", t = " << t <<endl;
      }

      //----------------------------AMR---------------------------------
      meshChanged=false;

      //++++++Refine step++++++
      if (refineMesh)  
      {
         if (myid == 0) cout<<"Refine mesh iterations..."<<endl;

         int its;
         for (its=0; its<ref_its; its++)
         {
           oper.UpdateJ(vx, &j);
           if (refiner.Apply(*pmesh)==false)
           {
               if (myid == 0) cout<<"No refined element found. Skip..."<<endl;
               break;
           }
           meshChanged=true;

           // this will update dpsidt through AMRupdate (only needed in the first its)
           if (compute_pressure && its==0){
               // Get dpsi/dt function
               ode_solver->GetStateVector(0, vk);
               int sc = fespace.TrueVSize();
               Vector v_dpsidt(vk.GetData() +  sc, sc);
               dpsidt->SetFromTrueDofs(v_dpsidt);
           }

           AMRUpdateTrue(vx, fe_offset3, *phi, *psi, *w, j, pre, dpsidt);
           oper.UpdateGridFunction();
           if (compute_pressure) {
               vfes->Update();
               //update vector grid function
               vel->Update();
               mag->Update();
               gradP->Update();
               BgradB->Update();
               gradBP->Update();
               gfv->Update();
               vfes->UpdatesFinished();
           }
           if (paraview) 
           {
               pw_const_fes.Update();
               mpi_rank_gf.Update();
               if (compute_tau) tau_value.Update();
           }

           pmesh->Rebalance();

           if (paraview) 
           {
               pw_const_fes.Update();
               mpi_rank_gf.Update();
               if (compute_tau) tau_value.Update();
           }

           //---Update solutions after rebalancing---
           AMRUpdateTrue(vx, fe_offset3, *phi, *psi, *w, j, pre, dpsidt);
           oper.UpdateGridFunction();
           if (compute_pressure) {
               vfes->Update();
               //update vector grid function
               vel->Update();
               mag->Update();
               gradP->Update();
               BgradB->Update();
               gradBP->Update();
               gfv->Update();
               vfes->UpdatesFinished();
           }
           oper.UpdateProblem(ess_bdr); 
           oper.SetInitialJ(*jptr);      //need to reset the current bounary

           if (myid == 0)
           {
             global_size = fespace.GlobalTrueVSize();
             cout << "Number of total scalar unknowns: " << global_size <<"; amr it= "<<its<<endl;
           }
         }

         //upate ode_solver for the next time step
         if (its>0 || refiner.Refined())
         {
            if (myid == 0) cout<<"Refined mesh; initialize ode_solver"<<endl;
            ode_solver->Init(oper);
            if (compute_pressure) {
               if (myid == 0) cout << "Mesh has changed and rebuilding vfes is needed"<<endl;
               vfes_match = false;
            }
         }
      }

      //++++++Derefine step++++++
      if (derefineMesh && derefine)
      {
         if (myid == 0) cout << "Derefined mesh..." << endl;

         int its;
         for (its=0; its<deref_its; its++)
         {
             //only call this at its=0
             if (its==0) oper.UpdateJ(vx, &j);

             if (!derefiner.Apply(*pmesh))
             {
                 if (myid == 0) cout << "No derefine elements found, skip..." << endl;
                 break;
             }
             meshChanged=true;

             // Update dpsidt if it is not updated in refiner 
             // It is needed only when refine is false and derefine is true (very rare)
             if (!(refiner.Refined()) && its==0){
                // Get dpsi/dt function
                ode_solver->GetStateVector(0, vk);
                int sc = fespace.TrueVSize();
                Vector v_dpsidt(vk.GetData() +  sc, sc);
                dpsidt->SetFromTrueDofs(v_dpsidt);
             }

             //---Update solutions first---
             AMRUpdateTrue(vx, fe_offset3, *phi, *psi, *w, j, pre, dpsidt);
             oper.UpdateGridFunction();
             if (compute_pressure) {
                vfes->Update();
                //update vector grid function
                vel->Update();
                mag->Update();
                gradP->Update();
                BgradB->Update();
                gradBP->Update();
                gfv->Update();
                vfes->UpdatesFinished();
             }

             if (paraview) 
             {
                 pw_const_fes.Update();
                 mpi_rank_gf.Update();
                 if (compute_tau) tau_value.Update();
             }

             pmesh->Rebalance();

             if (paraview) 
             {
                 pw_const_fes.Update();
                 mpi_rank_gf.Update();
                 if (compute_tau) tau_value.Update();
             }

             //---Update solutions after rebalancing---
             AMRUpdateTrue(vx, fe_offset3, *phi, *psi, *w, j, pre, dpsidt);
             oper.UpdateGridFunction();
             if (compute_pressure) {
                vfes->Update();
                //update vector grid function
                vel->Update();
                mag->Update();
                gradP->Update();
                BgradB->Update();
                gradBP->Update();
                gfv->Update();
                vfes->UpdatesFinished();
             }

             //---assemble problem and update boundary condition---
             oper.UpdateProblem(ess_bdr); 
             oper.SetInitialJ(*jptr);    //somehow I need to reset the current bounary

             if (myid == 0)
             {
               global_size = fespace.GlobalTrueVSize();
               cout << "Number of total scalar unknowns: " << global_size <<"; amr it= "<<its<<endl;
             }
         }

         if (its>0 || derefiner.Derefined())
         {
            if (myid == 0) cout<<"Derefined mesh; initialize ode_solver"<<endl;
            ode_solver->Init(oper);
            if (compute_pressure) {
               if (myid == 0)  cout << "Mesh has changed and rebuilding vfes is needed"<<endl;
               vfes_match = false;
            }
         }
      }
      //----------------------------AMR---------------------------------

      if ((ti % check_steps) == 0 && checkpt){
         phi->SetFromTrueDofs(vx.GetBlock(0));
         psi->SetFromTrueDofs(vx.GetBlock(1));
         w->SetFromTrueDofs(vx.GetBlock(2));
         checkpoint_rs(myid, t, *pmesh, *phi, *psi, *w, restart_count);
      }

      if ( (last_step || (ti % vis_steps) == 0) )
      {
        if (visualization || visit || paraview)
        {
           phi->SetFromTrueDofs(vx.GetBlock(0));
           psi->SetFromTrueDofs(vx.GetBlock(1));
           w->SetFromTrueDofs(vx.GetBlock(2));
           oper.UpdateJ(vx, &j);

           if(compute_pressure && paraview)
           {
              if (!refineMesh){
                // Get dpsi/dt function
                ode_solver->GetStateVector(0, vk);
                int sc = fespace.TrueVSize();
                Vector v_dpsidt(vk.GetData() +  sc, sc);
                dpsidt->SetFromTrueDofs(v_dpsidt);
              }

              if (!vfes_match){
                delete grad;     
                delete div ;     
                delete convect;  
                delete zLF  ;    
                delete zLFscalar; 
                delete Mfull;    
                delete Mrot ;    
                delete M_prec;
                delete MfullMat;
                delete a;
                delete KMat;
                delete K_amg;
                delete K_pcg;
                delete SpInvOrthoPC;

                grad = new ParMixedBilinearForm(&fespace, vfes);
                div = new ParMixedBilinearForm(vfes, &fespace);
                convect = new ParNonlinearForm(vfes);
                zLF  = new ParLinearForm(vfes);
                zLFscalar = new ParLinearForm(&fespace);
                Mfull = new ParBilinearForm(vfes);
                Mrot = new ParBilinearForm(vfes);
                Mscal_solver=oper.GetM_solver2();

                int vfes_truevsize = vfes->GetTrueVSize();
                vtrue.SetSize(vfes_truevsize);
                rhs.SetSize(vfes_truevsize);
                vJxB.SetSize(vfes_truevsize);

                DenseMatrix A(2);
                A(0,0) = 0.0; A(0,1) =-1.0;
                A(1,0) = 1.0; A(1,1) = 0.0;
                MatrixConstantCoefficient coeff_curl(A);

                //mass matrix for vector fields
                Mfull->AddDomainIntegrator(new VectorMassIntegrator);
                Mfull->Assemble();
                Mfull->Finalize();
                MfullMat = Mfull->ParallelAssemble();

                M_solver.iterative_mode = false;
                M_solver.SetRelTol(1e-7);
                M_solver.SetAbsTol(0.0);
                M_solver.SetMaxIter(2000);
                M_solver.SetPrintLevel(0);
                M_prec = new HypreSmoother;  
                M_prec->SetType(HypreSmoother::Jacobi);
                M_solver.SetPreconditioner(*M_prec);
                M_solver.SetOperator(*MfullMat);

                //gradient operator from H1 to Vector H1
                grad->AddDomainIntegrator(new GradientIntegrator);
                grad->Assemble(); 

                //nonlinear convection term u.grad u
                convect->AddDomainIntegrator(new VectorConvectionNLFIntegrator);
                convect->Setup();

                //divergence operator from Vector H1 to H1
                div->AddDomainIntegrator(new VectorDivergenceIntegrator);
                div->Assemble();

                //rotation matrix
                Mrot->AddDomainIntegrator(new VectorMassIntegrator(coeff_curl));
                Mrot->Assemble();
                Mrot->Finalize();

                 zv.SetSize(vfes->TrueVSize());
                zv2.SetSize(vfes->TrueVSize());
                zscalar.SetSize(fespace.TrueVSize());
                zscalar2.SetSize(fespace.TrueVSize());

                a = new ParBilinearForm(&fespace);
                a->AddDomainIntegrator(new DiffusionIntegrator);
                a->Assemble();
                a->Finalize();
                KMat=a->ParallelAssemble();

                K_amg = new HypreBoomerAMG(*KMat);
                K_amg->SetPrintLevel(0);
                K_pcg = new CGSolver(MPI_COMM_WORLD);
                SpInvOrthoPC = new mfem::navier::OrthoSolver();
                SpInvOrthoPC->SetOperator(*K_amg);
                K_pcg->SetOperator(*KMat);
                K_pcg->iterative_mode = false;
                K_pcg->SetRelTol(1e-7);
                K_pcg->SetMaxIter(200);
                K_pcg->SetPrintLevel(0);
                K_pcg->SetPreconditioner(*SpInvOrthoPC);
               
                vfes_match=true;
              }

              //compute velocity 
              grad->Mult(*phi, *zLF);
              zLF->ParallelAssemble(zv);
              M_solver.Mult(zv, zv2);
              vel->SetFromTrueDofs(zv2);

              //finalize with a rotation
              Mrot->Mult(*vel, *zLF);
              zLF->ParallelAssemble(zv);
              M_solver.Mult(zv, zv2);
              vel->SetFromTrueDofs(zv2);

              //compute B field
              grad->Mult(*psi, *zLF);
              zLF->ParallelAssemble(zv);
              M_solver.Mult(zv, zv2);
              mag->SetFromTrueDofs(zv2);

              //finalize with a rotation
              Mrot->Mult(*mag, *zLF);
              zLF->ParallelAssemble(zv);
              M_solver.Mult(zv, zv2);
              mag->SetFromTrueDofs(zv2);

              //compute -Δp=div(u.grad u - JxB)
              vel->GetTrueDofs(vtrue);
              convect->Mult(vtrue, rhs);  //nonlinear form only works with true dofs?

              JxBCoefficient JxBCoeff(&j, mag);
              domainJxB = new VectorDomainLFIntegrator(JxBCoeff);
              domainJxB->SetIntRule(&ir);
              ParLinearForm zJxB(vfes);
              zJxB.AddDomainIntegrator(domainJxB);
              zJxB.Assemble();
              zJxB.ParallelAssemble(vJxB);
              rhs.Add(-1.0, vJxB);
              
              //compute M^{-1}(u.grad u - JxB)
              M_solver.Mult(rhs, zv2);
              gfv->SetFromTrueDofs(zv2);
              div->Mult(*gfv, *zLFscalar);  //it is a mystery why ParGridFunction fails here

              ParLinearForm b(&fespace);
              b.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(JxBCoeff), ess_bdr);
              b.Assemble();
              b.ParallelAssemble(zscalar);

              zLFscalar->ParallelAssemble(zscalar2);
              zscalar.Add(1.0, zscalar2);
              K_pcg->Mult(zscalar, zscalar2);
              pre->SetFromTrueDofs(zscalar2);

              //compute grad P
              zv=0.0;
              grad->TrueAddMult(zscalar2, zv);
              M_solver.Mult(zv, zv2);
              gradP->SetFromTrueDofs(zv2);

              //compute B.gradB
              mag->GetTrueDofs(vtrue);
              convect->Mult(vtrue, zv);  
              M_solver.Mult(zv, zv2);
              BgradB->SetFromTrueDofs(zv2);

              //compute grad magnetic pressure
              B2Coefficient B2Coeff(mag);
              ParLinearForm B2int(&fespace);
              B2int.AddDomainIntegrator(new DomainLFIntegrator(B2Coeff, 2, 0));
              B2int.Assemble();
              B2int.ParallelAssemble(zscalar);
              Mscal_solver.Mult(zscalar, zscalar2);
              zv=0.0;
              grad->TrueAddMult(zscalar2, zv);
              M_solver.Mult(zv, zv2);
              gradBP->SetFromTrueDofs(zv2);
           }
        }

        if (visualization)
        {
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

        if (visit)
        {
           dc->SetCycle(ti);
           dc->SetTime(t);
           dc->Save();
        }

        if (paraview)
        {
           if(compute_tau){
              MyCoefficient velocity(phi, 2);
              delete computeTau;
              delete tauv;
              computeTau = new ParLinearForm(&pw_const_fes);
              computeTau->AddDomainIntegrator(new CheckTauIntegrator(0.29289321881*dt_real, resi, velocity, itau_));
              computeTau->Assemble(); 
              tauv=computeTau->ParallelAssemble();
              tau_value.SetFromTrueDofs(*tauv);
           }
 
           mpi_rank_gf = myid_rand;
           pd->SetCycle(ti);
           pd->SetTime(t);
           pd->Save();
        }
      }

      if (last_step)
          break;
      else
          continue;
   }

   MPI_Barrier(MPI_COMM_WORLD); 
   double end = MPI_Wtime();

   //++++++Save the solutions.
   if (checkpt)
   {
      phi->SetFromTrueDofs(vx.GetBlock(0));
      psi->SetFromTrueDofs(vx.GetBlock(1));
      w->SetFromTrueDofs(vx.GetBlock(2));
      checkpoint_rs(myid, t, *pmesh, *phi, *psi, *w, restart_count);
   }

   if (myid == 0) 
   { 
       cout <<"######Runtime = "<<end-start<<" ######"<<endl;
   }

   if(compute_pressure)
   {
      delete M_prec;
      delete K_amg;
      delete SpInvOrthoPC;
      delete MfullMat;
      delete KMat;
      delete a;
      delete K_pcg;
      delete vfes;
      delete vel;
      delete mag;
      delete gradP;
      delete gradBP;
      delete BgradB;
      delete gfv;       
      delete zLFscalar; 
      delete pre;      
      delete dpsidt;      
      delete grad;     
      delete div ;     
      delete convect;  
      delete zLF  ;    
      delete Mfull;    
      delete Mrot ;     
   }

   //+++++Free the used memory.
   delete psi;
   delete phi;
   delete w;
   delete ode_solver;
   delete pmesh;
   delete integ;
   delete dc;
   delete pd;
   delete estimator_used;
   if(compute_tau){
      delete tauv;
      delete computeTau;
   }
   oper.DestroyHypre();

   if (use_petsc) { MFEMFinalizePetsc(); }

   MPI_Finalize();

   return 0;
}



