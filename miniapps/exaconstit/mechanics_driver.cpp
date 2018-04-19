//***********************************************************************
//
//   ExaConstit Proxy App for AM Constitutive Properties Applications
//   Author: Steven R. Wopschall
//           wopschall1@llnl.gov
//   Date: August 6, 2017
//
//   Description: The purpose of this proxy app is to determine bulk 
//                constitutive properties for 3D printed material. This 
//                is the first-pass implementation of this app. The app
//                is a quasi-static, implicit solid mechanics code built 
//                on the MFEM library.
//                
//                Currently, only Dirichlet boundary conditions and a 
//                Neo-Hookean hyperelastic material model are 
//                implemented. Neumann (traction) boundary conditions and 
//                a body force are not implemented. This code uses 
//                psuedo-time stepping and Dirichlet boundary conditions 
//                are prescribed as either fully fixed, or along a 
//                prescribed direction. The latter, nonzero Dirichlet 
//                boundary conditions, are hard coded at the moment, and 
//                applied in the negative Z-direction as a function of 
//                the timestep. Until further implementation, the 
//                simulation's "final time" and "time step" size should 
//                be fixed (see running the app comments below). 
//
//                To run this code:
//
//                srun -n 2 ./mechanics_driver -m ../../data/cube-hex.mesh -tf 1.0 -dt 0.2  
//
//                where "np" is number of processors, the mesh is a 
//                simple cube mesh containing 8 elements, "tf" is the 
//                final simulation time, and dt is the timestep. Keep 
//                the time step >= 0.2. This has to do with how the 
//                nonzero Dirichlet BCs are applied.
//
//                Remark:
//                In principle, the mesh may be refined automatically, but 
//                this has not been thoroughly tested and has some problems
//                with -rp or -rs specified as anything but what is shown 
//                in the command line arguments above.
//
//                The finite element order defaults to 1 (linear), but 
//                may be increased by passing the argument at the 
//                command line "-o #" where "#" is the interpolation 
//                order (e.g. -o 2 for quadratic elements).
//
//                The mesh configuration is output for each time step
//                in mesh files (e.g. mesh.000001_1), which is per 
//                timestep, per processor. Visualization is using 
//                glvis. Currently I have not experimented with parallel 
//                mesh generation, so testing and debugging using glvis 
//                has been done in serial. An example call to glvis is
//                as follows
//
//                glvs -m mesh.000001_1
//
//                Lastly, if modifications are made to this file, one 
//                must type "make" in the command line to compile the 
//                code. 
//
//   Future Implemenations Notes:
//                
//                -Visco-plasticity constitutive model
//                -enhanced user control of Dirichlet BCs
//                -debug ability to read different mesh formats
//
//***********************************************************************
#include "mfem.hpp"
#include "mechanics_coefficient.hpp"
#include "mechanics_integrators.hpp"
#include "BCData.hpp"
#include "BCManager.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;

class NonlinearMechOperator : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &fe_space;

   ParNonlinearForm *Hform;
   mutable Operator *Jacobian;
   const Vector *x;

   /// Newton solver for the hyperelastic operator
   NewtonSolver newton_solver;
   /// Solver for the Jacobian solve in the Newton method
   Solver *J_solver;
   /// Preconditioner for the Jacobian
   Solver *J_prec;
   /// nonlinear model 
   NonlinearModel *model;

public:
   NonlinearMechOperator(ParFiniteElementSpace &fes,
                         Array<int> &ess_bdr,
                         double rel_tol,
                         double abs_tol,
                         int iter,
                         bool gmres,
                         bool slu, 
                         bool hyperelastic,
                         bool umat,
                         QuadratureFunction q_matVars0,
                         QuadratureFunction q_matVars1,
                         QuadratureFunction q_sigma0,
                         QuadratureFunction q_sigma1,
                         QuadratureFunction q_matGrad);

   /// Required to use the native newton solver
   virtual Operator &GetGradient(const Vector &x) const;
   virtual void Mult(const Vector &k, Vector &y) const;

   /// Driver for the newton solver
   void Solve(Vector &x) const;

   /// Get essential true dof list, if required
   const Array<int> &GetEssTDofList();

   virtual ~NonlinearMechOperator();
};

void visualize(ostream &out, ParMesh *mesh, ParGridFunction *deformed_nodes,
               ParGridFunction *field, const char *field_name = NULL,
               bool init_vis = false);

// set kinematic functions and boundary condition functions
void ReferenceConfiguration(const Vector &x, Vector &y);
void InitialDeformation(const Vector &x, Vector &y);
void DirBdrFunc(const Vector &x, double t, int attr_id, Vector &y);
void InitGridFunction(const Vector &x, Vector &y);

// material input check routine
bool checkMaterialArgs(bool hyperelastic, bool umat, bool cp, bool g_euler, 
                       bool g_q, bool g_uniform, int ngrains);

// grain data setter routine
void setGrainData(QuadratureFunction *qf, Vector *orient, 
                  ParFiniteElementSpace *fes, ParMesh *pmesh);

// initialize a quadrature function with a single input value, val.
void initQuadFunc(QuadratureFunction *qf, double val, ParFiniteElementSpace *fes);

// set the time step on the boundary condition objects
void setBCTimeStep(double dt, int nDBC);

int main(int argc, char *argv[])
{
   // Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   static Vector grain_uni_vec(0);  // vector to store uniform grain 
                                    // orientation vector 

   // Parse command-line options.
   const char *mesh_file = "../../data/cube-hex-ro.mesh";
   bool cubit = false;

   // serial and parallel refinement levels
   int ser_ref_levels = 0;
   int par_ref_levels = 0;

   // polynomial interpolation order
   int order = 1;

   // final simulation time and time step
   double t_final = 300.0;
   double dt = 3.0;

   // visualization input args
   bool visualization = true;
   int vis_steps = 1;

   // newton input args
   double newton_rel_tol = 1.0e-6;
   double newton_abs_tol = 1.0e-8;
   int newton_iter = 500;
   
   // solver input args
   bool gmres_solver = true;
   bool slu_solver = false;

   // input arg to specify Abaqus UMAT
   bool umat = false;

   // input arg to specify crystal plasticity or hyperelasticity 
   // (for testing)
   bool cp = false;
   bool hyperelastic = true;

   // grain input arguments
   const char *grain_file = "grains.txt";
   int ngrains = 0;
   bool grain_euler = false;
   bool grain_q = false;
   bool grain_uniform = false;

   // boundary condition input args
   Array<int> ess_id;   // essential bc ids for the whole boundary
   Vector     ess_disp; // vector of displacement components for each attribute in ess_id
   Array<int> ess_comp; // component combo (x,y,z = -1, x = 1, y = 2, z = 3, 
                        // xy = 4, yz = 5, xz = 6, free = 0 

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&grain_file, "-g", "--grain",
                  "Grain file to use.");
   args.AddOption(&ngrains, "-ng", "--grain-number",
                  "Number of grains.");
   args.AddOption(&cubit, "-mcub", "--cubit", "-no-mcub", "--no-cubit",
                  "Read in a cubit mesh.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&slu_solver, "-slu", "--superlu", "-no-slu",
                  "--no-superlu", "Use the SuperLU Solver.");
   args.AddOption(&gmres_solver, "-gmres", "--gmres", "-no-gmres", "--no-gmres",
                   "Use gmres, otherwise minimum residual is used.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&newton_rel_tol, "-rel", "--relative-tolerance",
                  "Relative tolerance for the Newton solve.");
   args.AddOption(&newton_abs_tol, "-abs", "--absolute-tolerance",
                  "Absolute tolerance for the Newton solve.");
   args.AddOption(&newton_iter, "-it", "--newton-iterations",
                  "Maximum iterations for the Newton solve.");
   args.AddOption(&hyperelastic, "-hyperel", "--hyperelastic", "-no-hyperel",
                  "--no-hyperelastic", 
                  "Use Neohookean hyperelastic material model.");
   args.AddOption(&umat, "-umat", "--abaqus-umat", "-no-umat",
                  "--no-abaqus-umat", 
                  "Use user-supplied Abaqus UMAT constitutive model.");
   args.AddOption(&cp, "-cp", "--crystal-plasticity", "-no-cp",
                  "--no-crystal-plasticity", 
                  "Use user-supplied Abaqus UMAT crystal plasticity model.");
   args.AddOption(&grain_euler, "-ge", "--euler-grain-orientations", "-no-ge",
                  "--no-euler-grain-orientations", 
                  "Use Euler angles to define grain orientations.");
   args.AddOption(&grain_q, "-gq", "--quaternion-grain-orientations", "-no-gq",
                  "--no-quaternion-grain-orientations", 
                  "Use quaternions to define grain orientations.");
   args.AddOption(&grain_uniform, "-gu", "--uniform-grain-orientations", "-no-gu",
                  "--no-uniform-grain-orientations",
                  "Use uniform grain orientations.");
   args.AddOption(&grain_uni_vec, "-guv", "--uniform-grain-vector",
                  "Vector defining uniform grain orientations.");
   args.AddOption(&ess_id, "-attrid", "--dirichlet-attribute-ids",
                  "Attribute IDs for dirichlet boundary conditions.");
   args.AddOption(&ess_disp, "-disp", "--dirichlet-disp", 
                  "Final (x,y,z) displacement components for each dirichlet BC.");
   args.AddOption(&ess_comp, "-bcid", "--bc-comp-id",
                  "Component ID for essential BCs.");

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
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // Check material model argument input parameters for valid combinations
   bool err = checkMaterialArgs(hyperelastic, umat, cp, grain_euler, grain_q, grain_uniform, ngrains);
   if (!err && myid == 0) 
   {
      cerr << "\nInconsistent material input; check args" << '\n';
   }

   // Open the mesh
   Mesh *mesh;
   if (cubit) {
      //named_ifgzstream imesh(mesh_file);
      mesh = new Mesh(mesh_file, 1, 1);
   }
   if (!cubit) {
      printf("opening mesh file \n");

      ifstream imesh(mesh_file);
      if (!imesh)
      {
         if (myid == 0)
         {
            cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
         }
         MPI_Finalize();
         return 2;
      }
  
      mesh = new Mesh(imesh, 1, 1, true);
      imesh.close();
   }
   ParMesh *pmesh = NULL;
   
   for (int lev = 0; lev < ser_ref_levels; lev++)
      {
         mesh->UniformRefinement();
      }

   pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   for (int lev = 0; lev < par_ref_levels; lev++)
      {
         pmesh->UniformRefinement();
      }

   delete mesh;

   int dim = pmesh->Dimension();

   // Definie the finite element spaces for displacement field
   H1_FECollection fe_coll(order, dim);
   ParFiniteElementSpace fe_space(pmesh, &fe_coll, dim);

   HYPRE_Int glob_size = fe_space.GlobalTrueVSize();

   // Print the mesh statistics
   if (myid == 0)
   {
      std::cout << "***********************************************************\n";
      std::cout << "dim(u) = " << glob_size << "\n";
      std::cout << "***********************************************************\n";
   }

   // determine the type of grain input for crystal plasticity problems
   int grain_offset = 1; // default to 1 so matVars below doesn't have a null/0 size
   if (grain_euler) {
      grain_offset = 3; 
   }
   else if (grain_q) {
      grain_offset = 4;
   }
   else if (grain_uniform) {
      int vdim = grain_uni_vec.Size();
      if (vdim == 0 && myid == 0) {
         cerr << "\nMust specify a uniform grain orientation vector" << '\n' << endl;
      } 
      else {
         grain_offset = 3;
      }
   }

   // Define a quadrature space and material history variable QuadratureFunction.
   // This isn't needed for hyperelastic material models, but for general plasticity 
   // models there will be history variables stored at quadrature points.
   int intOrder = 2*order+1;
   QuadratureSpace qspace (pmesh, intOrder); // 3rd order polynomial for 2x2x2 quadrature
                                             // for first order finite elements
   QuadratureFunction matVars0(&qspace, grain_offset);
   initQuadFunc(&matVars0, 0.0, &fe_space);
   
   // if using a crystal plasticity model then get grain orientation data
   if (cp)
   {
      // define a vector to hold the grain orientation data. The size of the vector is 
      // deined as the grain offset times the number of grains (problem input argument). 
      // The grain offset is the number of values defining a grain orientation. 
      // This could include more things later. Right now the grain id is simply the 
      // index into the grain vector and does not need to be stored or input as a 
      // separate id.
      int gsize = grain_offset * ngrains;
//      printf("grain_offset: %d \n", grain_offset);

      // declare a vector to hold the grain orientation input data.
      Vector g_orient;

      // set the grain orientation vector from either the input grain file or the 
      // input uniform grain vector
      if (!grain_uniform) {
//         printf("processing grain file... \n");
         ifstream igrain(grain_file); 
         if (!igrain && myid == 0)
         {
            cerr << "\nCan not open grain file: " << grain_file << '\n' << endl;
         }
         g_orient.Load(igrain, gsize);
         // close grain input file stream
         igrain.close();
//         printf("after loading the grain file and closing it \n");
      }
      else {
         g_orient = grain_uni_vec;   
      }

      // Set the material variable quadrature function data to the grain orientation 
      // data
      printf("before setGrainData \n");
      setGrainData(&matVars0, &g_orient, &fe_space, pmesh);
   }

   // Define quadrature functions to store a vector representation of the 
   // Cauchy stress, in Voigt notation (s_11, s_22, s_33, s_12, s_13, s_23), for 
   // the beginning of the step and the end of the step (or the end of the increment).
   QuadratureFunction sigma0(&qspace, 6);
   QuadratureFunction sigma1(&qspace, 6);
   initQuadFunc(&sigma0, 0.0, &fe_space);
   initQuadFunc(&sigma1, 0.0, &fe_space);

   // define a quadrature function to store the material tangent stiffness 
   QuadratureFunction matGrad (&qspace, 9);
   initQuadFunc(&matGrad, 0.0, &fe_space);

   // define the end of step (or incrementally updated) material history 
   // variables
   int vdim = matVars0.GetVDim();
   QuadratureFunction matVars1(&qspace, vdim);
   initQuadFunc(&matVars1, 0.0, &fe_space);

   // Define the grid functions for the current configuration, the global reference 
   // configuration, and the global deformed configuration, respectively.
   ParGridFunction x_gf(&fe_space);
   ParGridFunction x_ref(&fe_space);
   ParGridFunction x_def(&fe_space);

   // define a vector function coefficient for the initial deformation 
   // and reference configuration
   VectorFunctionCoefficient deform(dim, InitialDeformation);
   VectorFunctionCoefficient refconfig(dim, ReferenceConfiguration);
  
   // project the vector function coefficients onto current configuration
   // and reference configuration grid functions. The initial deformation 
   // at time t=0 is simply the reference configuration. This function may 
   // be used to project an initial guess or perterbation to the solution 
   // vector if desired.
   x_gf.ProjectCoefficient(deform);
   x_ref.ProjectCoefficient(refconfig);

   // Define grid function for the Dirichlet boundary conditions
   ParGridFunction x_ess(&fe_space);

   // Define grid function for the current configuration grid function
   // WITH Dirichlet BCs
   ParGridFunction x_bar_gf(&fe_space);

   // Define a VectorFunctionCoefficient to initialize a grid function
   VectorFunctionCoefficient init_grid_func(dim, InitGridFunction);

   // initialize grid functions by projecting the VectorFunctionCoefficient
   x_ess.ProjectCoefficient(init_grid_func);
   x_bar_gf.ProjectCoefficient(init_grid_func);
   x_def.ProjectCoefficient(init_grid_func);

   // define a boundary attribute array and initialize to 0
   Array<int> ess_bdr;
   // set the size of the essential boundary conditions attribute array
//   ess_bdr.SetSize(ess_id.Size()+1);
   ess_bdr.SetSize(fe_space.GetMesh()->bdr_attributes.Max());
//   printf("size of ess_bdr: %d \n", ess_bdr.Size());
   ess_bdr = 0;

   // setup inhomogeneous essential boundary conditions using the boundary 
   // condition manager (BCManager) and boundary condition data (BCData) 
   // classes developed for ExaConstit.
   if (ess_disp.Size() != 3*ess_id.Size()) {
      cerr << "\nMust specify three Dirichlet components per essential boundary attribute" << '\n' << endl;
   }
//   printf("ess_id size: %d \n", ess_id.Size());
   int numDirBCs = 0;
   for (int i=0; i<ess_id.Size(); ++i) {
      // set the boundary condition id based on the attribute id
      int bcID = ess_id[i];

      // instantiate a boundary condition manager instance and 
      // create a BCData object
      BCManager & bcManager = BCManager::getInstance();
      BCData & bc = bcManager.CreateBCs( bcID );

      // set the displacement component values
      bc.essDisp[0] = ess_disp[3*i];
      bc.essDisp[1] = ess_disp[3*i+1];
      bc.essDisp[2] = ess_disp[3*i+2];
      bc.compID = ess_comp[i];

      // set the final simulation time 
      bc.tf = t_final;

      // set the boundary condition scales
      bc.setScales();

      // set the active boundary attributes
      if (bc.compID != 0) {
         ess_bdr[i] = 1;
//         printf("active ess_bdr: %d \n", i+1);
//         printf("bc comp id: %d \n", bc.compID);
      }
      ++numDirBCs;
//      printf("bcid, dirDisp: %d %f %f %f \n", bcID, bc.essDisp[0], bc.essDisp[1], bc.essDisp[2]);
   }

   // declare a VectorFunctionRestrictedCoefficient over the boundaries that have attributes
   // associated with a Dirichlet boundary condition (ids provided in input)
   VectorFunctionRestrictedCoefficient ess_bdr_func(dim, DirBdrFunc, ess_bdr);

   // Construct the nonlinear mechanics operator. Note that q_grain0 is
   // being passed as the matVars0 quadarture function. This is the only 
   // history variable considered at this moment. Consider generalizing 
   // this where the grain info is a possible subset only of some 
   // material history variable quadrature function. Also handle the 
   // case where there is no grain data.
   NonlinearMechOperator oper(fe_space, ess_bdr, 
                              newton_rel_tol, newton_abs_tol, 
                              newton_iter, gmres_solver, slu_solver,
                              hyperelastic, umat, matVars0, 
                              matVars1, sigma0, sigma1, matGrad);

   // get the essential true dof list. This may not be used.
   const Array<int> ess_tdof_list = oper.GetEssTDofList();

   // debug print
//   for (int i=0; i<ess_tdof_list.Size(); ++i) {
//      printf("ess_tdof_list[i]: %d %d \n", i, ess_tdof_list[i]);
//   }
   
   // declare solution vector
   Vector x(fe_space.TrueVSize()); // this sizing is correct
   x = 0.0;

   // initialize visualization if requested 
   socketstream vis_u, vis_p;
   if (visualization) {
      char vishost[] = "localhost";
      int  visport   = 19916;
      vis_u.open(vishost, visport);
      vis_u.precision(8);
      visualize(vis_u, pmesh, &x_gf, &x_def, "Deformation", true);
      // Make sure all ranks have sent their 'u' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());
   }

   // initialize/set the time
   double t = 0.0;
   oper.SetTime(t); 

   bool last_step = false;

   // enter the time step loop. This was modeled after example 10p.
   for (int ti = 1; !last_step; ti++)
   {

      // compute time step (this calculation is pulled from ex10p.cpp)
      double dt_real = min(dt, t_final - t);

      // set the time step on the boundary conditions
      setBCTimeStep(dt_real, numDirBCs);

      // compute current time
      t = t + dt_real;

      // set the time for the nonzero Dirichlet BC function evaluation
      //non_zero_ess_func.SetTime(t);
      ess_bdr_func.SetTime(t);

      // register Dirichlet BCs. 
      ess_bdr = 1;

      // overwrite entries in x_bar_gf for dofs with Dirichlet 
      // boundary conditions (note, this routine overwrites, not adds).
      x_ess.ProjectBdrCoefficient(ess_bdr_func); // don't need attr list as input
                                                    // pulled off the 
                                                    // VectorFunctionRestrictedCoefficient

      // sum the current configuration grid function and the Dirichlet BC grid function 
      // into x_bar_gf 
      add(x_ess, x_gf, x_bar_gf);

      // populate the solution vector, x, with the true dofs entries in x_bar_gf
      x_bar_gf.GetTrueDofs(x);

      // Solve the Newton system 
      oper.Solve(x);

      // distribute the solution vector to the current configuration 
      // grid function. Note, the solution vector is the global 
      // current configuration, not the incremental nodal displacements
      x_gf.Distribute(x);

      // Set the end-step deformation with respect to the global reference 
      // configuration at time t=0.
      subtract(x_gf, x_ref, x_def);

      last_step = (t >= t_final - 1e-8*dt);

      if (last_step || (ti % vis_steps) == 0)
      {
         if (myid == 0)
         {
            cout << "step " << ti << ", t = " << t << endl;
         }
      }

      {

      // Save the displaced mesh. These are snapshots of the endstep current 
      // configuration. Later add functionality to not save the mesh at each timestep.

         GridFunction *nodes = &x_gf; // set a nodes grid function to global current configuration
         int owns_nodes = 0;
         pmesh->SwapNodes(nodes, owns_nodes); // pmesh has current configuration nodes

         ostringstream mesh_name, deformed_name;
         mesh_name << "mesh." << setfill('0') << setw(6) << myid << "_" << ti;
         deformed_name << "end_step_def." << setfill('0') << setw(6) << myid << "_" << ti;

         // saving mesh for plotting. pmesh has global current configuration nodal coordinates
         ofstream mesh_ofs(mesh_name.str().c_str());
         mesh_ofs.precision(8);
         pmesh->Print(mesh_ofs); 
       
         // saving incremental nodal displacements. This may not be necessary. May want to 
         // save each current configuration as this data will match the mesh output.
         ofstream deformation_ofs(deformed_name.str().c_str());
         deformation_ofs.precision(8);
         x_def.Save(deformation_ofs); 
      }

   }
      
   // Free the used memory.
   delete pmesh;

   MPI_Finalize();

   return 0;
}

NonlinearMechOperator::NonlinearMechOperator(ParFiniteElementSpace &fes,
                                             Array<int> &ess_bdr,
                                             double rel_tol,
                                             double abs_tol,
                                             int iter,
                                             bool gmres,
                                             bool slu, 
                                             bool hyperelastic,
                                             bool umat,
                                             QuadratureFunction q_matVars0,
                                             QuadratureFunction q_matVars1,
                                             QuadratureFunction q_sigma0,
                                             QuadratureFunction q_sigma1,
                                             QuadratureFunction q_matGrad)
   : TimeDependentOperator(fes.TrueVSize()), fe_space(fes),
     newton_solver(fes.GetComm())
{
   Vector * rhs;
   rhs = NULL;

   // Define the parallel nonlinear form 
   Hform = new ParNonlinearForm(&fes);

   // Set the essential boundary conditions
   Hform->SetEssentialBCPartial(ess_bdr, rhs); 
//   Hform->SetEssentialBC(ess_bdr, rhs);

   if (umat) {
      model = new AbaqusUmatModel(&q_sigma0, &q_sigma1, &q_matGrad, &q_matVars0, &q_matVars1);
      
      // Add the user defined integrator
      Hform->AddDomainIntegrator(new UserDefinedNLFIntegrator(dynamic_cast<AbaqusUmatModel*>(model)));
   }

   else if (hyperelastic) {
      model = new NeoHookeanModel(80.E3, 140.E3);
      // Add the hyperelastic integrator
      Hform->AddDomainIntegrator(new HyperelasticNLFIntegrator(dynamic_cast<HyperelasticModel*>(model)));
   }

   if (gmres) {
      HypreBoomerAMG *prec_amg = new HypreBoomerAMG();
      prec_amg->SetPrintLevel(0);
      prec_amg->SetElasticityOptions(&fe_space);
      J_prec = prec_amg;

      GMRESSolver *J_gmres = new GMRESSolver(fe_space.GetComm());
//      J_gmres->iterative_mode = false;
      J_gmres->SetRelTol(rel_tol);
      J_gmres->SetAbsTol(1e-12);
      J_gmres->SetMaxIter(300);
      J_gmres->SetPrintLevel(0);
      J_gmres->SetPreconditioner(*J_prec);
      J_solver = J_gmres; 

   } 
   // retain super LU solver capabilities
   else if (slu) { 
      SuperLUSolver *superlu = NULL;
      superlu = new SuperLUSolver(MPI_COMM_WORLD);
      superlu->SetPrintStatistics(false);
      superlu->SetSymmetricPattern(false);
      superlu->SetColumnPermutation(superlu::PARMETIS);
      
      J_solver = superlu;
      J_prec = NULL;

   }
   else {
      HypreSmoother *J_hypreSmoother = new HypreSmoother;
      J_hypreSmoother->SetType(HypreSmoother::l1Jacobi);
      J_hypreSmoother->SetPositiveDiagonal(true);
      J_prec = J_hypreSmoother;

      MINRESSolver *J_minres = new MINRESSolver(fe_space.GetComm());
      J_minres->SetRelTol(rel_tol);
      J_minres->SetAbsTol(0.0);
      J_minres->SetMaxIter(300);
      J_minres->SetPrintLevel(-1);
      J_minres->SetPreconditioner(*J_prec);
      J_solver = J_minres;

   }

   // Set the newton solve parameters
   newton_solver.iterative_mode = true;
   newton_solver.SetSolver(*J_solver);
   newton_solver.SetOperator(*this);
   newton_solver.SetPrintLevel(1); 
   newton_solver.SetRelTol(rel_tol);
   newton_solver.SetAbsTol(abs_tol);
   newton_solver.SetMaxIter(iter);
}

const Array<int> &NonlinearMechOperator::GetEssTDofList()
{
   return Hform->GetEssentialTrueDofs();
}

// Solve the Newton system
void NonlinearMechOperator::Solve(Vector &x) const
{
   Vector zero;
   newton_solver.Mult(zero, x);

   MFEM_VERIFY(newton_solver.GetConverged(), "Newton Solver did not converge.");
}

// compute: y = H(x,p)
void NonlinearMechOperator::Mult(const Vector &k, Vector &y) const
{
   // Apply the nonlinear form
   Hform->Mult(k, y);
}

// Compute the Jacobian from the nonlinear form
Operator &NonlinearMechOperator::GetGradient(const Vector &x) const
{
   Jacobian = &Hform->GetGradient(x);
   return *Jacobian;
}

NonlinearMechOperator::~NonlinearMechOperator()
{
   delete J_solver;
   if (J_prec != NULL) {
      delete J_prec;
   }
   delete model;
}

// In line visualization
void visualize(ostream &out, ParMesh *mesh, ParGridFunction *deformed_nodes,
               ParGridFunction *field, const char *field_name, bool init_vis)
{
   if (!out)
   {  
      return;
   }

   GridFunction *nodes = deformed_nodes;
   int owns_nodes = 0;

   mesh->SwapNodes(nodes, owns_nodes);

   out << "parallel " << mesh->GetNRanks() << " " << mesh->GetMyRank() << "\n";
   out << "solution\n" << *mesh << *field;

   mesh->SwapNodes(nodes, owns_nodes);

   if (init_vis)
   {
      out << "window_size 800 800\n";
      out << "window_title '" << field_name << "'\n";
      if (mesh->SpaceDimension() == 2)
      {
         out << "view 0 0\n"; // view from top
         out << "keys jl\n";  // turn off perspective and light
      }
      out << "keys cm\n";         // show colorbar and mesh
      out << "autoscale value\n"; // update value-range; keep mesh-extents fixed
      out << "pause\n";
   }
   out << flush;
}

void ReferenceConfiguration(const Vector &x, Vector &y)
{
   // set the reference, stress free, configuration
   y = x;
}


void InitialDeformation(const Vector &x, Vector &y)
{
   // set initial configuration to be the reference 
   // configuration. Can set some perturbation as long as 
   // the boundary condition degrees of freedom are 
   // overwritten.
   y = x;
}

void DirBdrFunc(const Vector &x, double t, int attr_id, Vector &y)
{
   BCManager & bcManager = BCManager::getInstance();
   BCData & bc = bcManager.GetBCInstance(attr_id);
//   printf("DirBdrFunc attr_id: %d \n", attr_id);

   bc.setDirBCs(x, t, y);

//   if (attr_id == 4) {
//      printf("DirBdrFunc x[2]: %f \n", x[2]);
//   }
}

void InitGridFunction(const Vector &x, Vector &y)
{
   y = 0.;
}

bool checkMaterialArgs(bool hyperelastic, bool umat, bool cp, bool g_euler, 
                       bool g_q, bool g_uniform, int ngrains)
{
   bool err = true;

   if (hyperelastic && cp)
   {

      cerr << "Hyperelastic and cp can't both be true. Choose material model." << "\n";
   }

   // only perform checks, don't set anything here
   if (cp && !g_euler && !g_q && !g_uniform)
   {
      cerr << "\nMust specify grain data type for use with cp input arg." << '\n';
      err = false;
   }

   else if (cp && g_euler && g_q)
   {
      cerr << "\nCannot specify euler and quaternion grain data input args." << '\n';
      err = false;
   }

   else if (cp && g_euler && g_uniform)
   {
      cerr << "\nCannot specify euler and uniform grain data input args." << '\n';
      err = false;
   }

   else if (cp && g_q && g_uniform)
   {
      cerr << "\nCannot specify quaternion and uniform grain data input args." << '\n';
      err = false;
   }

   else if (cp && (ngrains < 1))
   {
      cerr << "\nSpecify number of grains for use with cp input arg." << '\n';
      err = false;
   }

   return err;
}

void setGrainData(QuadratureFunction *qf, Vector *orient, 
                  ParFiniteElementSpace *fes, ParMesh *pmesh)
{

   // put element grain orientation data on the quadrature points. This 
   // should eventually go into a separate subroutine
   const FiniteElement *fe;
   const IntegrationRule *ir;
   double* data = qf->GetData();
   int qf_offset = qf->GetVDim();
   QuadratureSpace* qspace = qf->GetSpace();

   // get the data for the grain orientations
   double * grain_data = orient->GetData();
   int elem_atr;

   for (int i = 0; i < fes->GetNE(); ++i)
   {
      fe = fes->GetFE(i);
      ir = &(qspace->GetElementIntRule(i));
//      ir = &(IntRules.Get(fe->GetGeomType(), 2*fe->GetOrder() + 3));

      int elem_offset = qf_offset * ir->GetNPoints();

      // get the element attribute. Note this assumes that there is an element attribute 
      // for all elements in the mesh corresponding to the grain id to which the element 
      // belongs.
      elem_atr = pmesh->attributes[fes->GetAttribute(i)-1]; 
//      printf("setGrainData: element attr %d \n", elem_atr);

      // loop over quadrature points
      for (int j = 0; j < ir->GetNPoints(); ++j)
      {
         // loop over quadrature point data
         for (int k = 0; k < qf_offset; ++k) 
         {
            // index into the quadrature function with an element stride equal to the length of the 
            // quadrature vector function times the number of quadrature points. Index into 
            // the grain data with a stride equal to the length of the quadrature vector function
            // at the (elem_atr - 1) index position.
            data[(elem_offset * i) + qf_offset * j + k] = grain_data[qf_offset * (elem_atr-1) + k];
//            printf("setGrainData, grain_data %f \n", grain_data[qf_offset * (elem_atr-1) + k]);
         }
      }
   } 
}

void initQuadFunc(QuadratureFunction *qf, double val, ParFiniteElementSpace *fes)
{

   // put element grain orientation data on the quadrature points. This 
   // should eventually go into a separate subroutine
   const FiniteElement *fe;
   const IntegrationRule *ir;
   double* qf_data = qf->GetData();
   int qf_offset = qf->GetVDim();
   QuadratureSpace* qspace = qf->GetSpace();

//   printf("qf data size: %d \n", qf->Size());

   for (int i = 0; i < fes->GetNE(); ++i)
   {
      fe = fes->GetFE(i);
      ir = &(qspace->GetElementIntRule(i));
//      ir = &(IntRules.Get(fe->GetGeomType(), 2*fe->GetOrder() + 3));

      int elem_offset = qf_offset * ir->GetNPoints();
//      printf("element offset %d \n", elem_offset);
//      printf("num ip: %d \n", ir->GetNPoints());
//      printf("elemoffset: %d \n", elem_offset);

      // loop over element data at each quadrature point
      for (int j = 0; j < elem_offset; ++j)
      {
         qf_data[i * elem_offset + j] = val;
      }
   } 
}

void setBCTimeStep(double dt, int nDBC)
{
   for (int i=0; i<nDBC; ++i) {
      BCManager & bcManager = BCManager::getInstance();
      BCData & bc = bcManager.CreateBCs( i+1 );
      bc.dt = dt;
   }

}
