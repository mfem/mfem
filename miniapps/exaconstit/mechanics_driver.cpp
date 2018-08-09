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

class SimVars
{
protected:
   double time;
   double dt;
public:
   double GetTime() { return time; }
   double GetDTime() { return dt; }

   void SetTime(double t) { time = t; }
   void SetDt(double dtime) { dt = dtime; }
}

class NonlinearMechOperator : public TimeDependentOperator
{
public:
   SimVars solVars;
protected:
   ParFiniteElementSpace &fe_space;

   ParNonlinearForm *Hform;
   mutable Operator *Jacobian;
   const Vector *x;

   /// Newton solver for the operator
   ExaNewtonSolver newton_solver;
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
                         QuadratureFunction q_matGrad,
                         QuadratureFunction q_solVars0,
                         Vector matProps, int numProps,
                         int nStateVars);

   /// Required to use the native newton solver
   virtual Operator &GetGradient(const Vector &x) const;
   virtual void Mult(const Vector &k, Vector &y) const;

   /// Driver for the newton solver
   void Solve(Vector &x) const;

   /// Get essential true dof list, if required
   const Array<int> &GetEssTDofList();

   /// Get FE space
   ParFiniteElementSpace GetFESpace() { return fe_space; }

   /// routine to update beginning step model variables with converged end 
   /// step values
   void UpdateModel(const Vector &x);

   void SetTime(const double t);
   void SetDt(const double dt);

   virtual ~NonlinearMechOperator();
};

void visualize(ostream &out, ParMesh *mesh, ParGridFunction *deformed_nodes,
               ParGridFunction *field, const char *field_name = NULL,
               bool init_vis = false);

// set kinematic functions and boundary condition functions
void ReferenceConfiguration(const Vector &x, Vector &y);
void InitialDeformation(const Vector &x, double t,  Vector &y)
void DirBdrFunc(const Vector &x, double t, int attr_id, Vector &y);
void InitGridFunction(const Vector &x, Vector &y);

// material input check routine
bool checkMaterialArgs(bool hyperelastic, bool umat, bool cp, bool g_euler, 
                       bool g_q, bool g_uniform, int ngrains, int numProps,
                       int numStateVars);

// material state variable and grain data setter routine
void setStateVarData(Vector* sVars, Vector* orient, ParFiniteElementSpace *fes, 
                     ParMesh *pmesh, int grainOffset, int stateVarOffset,
                     QuadratureFunction* qf);

// initialize a quadrature function with a single input value, val.
void initQuadFunc(QuadratureFunction *qf, double val, ParFiniteElementSpace *fes);

// compute the beginning step deformation gradient to store on a quadrature
// function
void computeDefGrad(QuadratureFunction *qf, ParFiniteElementSpace *fes, 
                    const Vector &x0);

// set the time step on the boundary condition objects
void setBCTimeStep(double dt, int nDBC);

int main(int argc, char *argv[])
{
   // print the version of the code being run
   printf("MFEM Version: %d \n", GetVersion());

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

   // material properties input arguments
   const char *props_file = "props.txt";
   int nProps = 0;

   // boundary condition input args
   Array<int> ess_id;   // essential bc ids for the whole boundary
   Vector     ess_disp; // vector of displacement components for each attribute in ess_id
   Array<int> ess_comp; // component combo (x,y,z = -1, x = 1, y = 2, z = 3, 
                        // xy = 4, yz = 5, xz = 6, free = 0 

   // specify all input arguments
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&grain_file, "-g", "--grain",
                  "Grain file to use.");
   args.AddOption(&ngrains, "-ng", "--grain-number",
                  "Number of grains.");
   args.AddOption(&props_file, "-props", "--mat-props",
                  "Material properties file to use.");
   args.AddOption(&nProps, "-nprops", "--number-props",
                  "Number of material properties.");
   args.AddOption(&state_file, "-svars", "--state-vars",
                  "State variables file.");
   args.AddOption(&numStateVars, "-nsvars", "--number-state_vars",
                  "Number of state variables.");
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

   // Parse the arguments and check if they are good
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
   bool err = checkMaterialArgs(hyperelastic, umat, cp, grain_euler, grain_q, grain_uniform, 
              ngrains, nProps, numStateVars);
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
   
   // mesh refinement if specified in input
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
   int grain_offset = 0; // note: numMatVars >= 1, no null state vars by construction
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

   // set the offset for the matVars quadrature function. This is the number of 
   // state variables (stored at each integration point) and then the grain offset, 
   // which is the number of variables defining the grain data stored at each 
   // integration point.
   int matVarsOffset = numStateVars + grain_offset;

   // Define a quadrature space and material history variable QuadratureFunction.
   // This isn't needed for hyperelastic material models, but for general plasticity 
   // models there will be history variables stored at quadrature points.
   int intOrder = 2*order+1;
   QuadratureSpace qspace (pmesh, intOrder); // 3rd order polynomial for 2x2x2 quadrature
                                             // for first order finite elements
   QuadratureFunction matVars0(&qspace, matVarsOffset); // was grain_offset prior to stateVars
   initQuadFunc(&matVars0, 0.0, &fe_space);
   
   // read in material properties and state variables files for use with ALL models
   // store input data on Vector object. The material properties vector will be 
   // passed into the Nonlinear mech operator constructor to initialize the material 
   // properties vector on the model and the state variables vector will be used with 
   // the grain data vector (if crystal plasticity) to populate the material state 
   // vector quadrature function. It is assumed that the state variables input file 
   // are initial values for all state variables applied to all quadrature points. 
   // There is not a separate initialization file for each quadrature point
   Vector matProps;
   Vector stateVars;
   { // read in props, material state vars and grains if crystal plasticity
      ifstream iprops(props_file);
      if (!iprops && myid == 0)
      {
         cerr << "\nCannot open material properties file: " << props_file << '\n' << endl;
      }

      // load material properties
      matProps.Load(iprops, nProps);
      iprops.close();
      
      // read in state variables file
      if stream istateVars(state_file);
      if (!istateVars && myid == 0)
      {
         cerr << "\nCannot open state variables file: " << state_file << '\n' << endl;
      }

      // load state variables
      stateVars.Load(istateVars, numStateVars);
      istateVars.close();

      // if using a crystal plasticity model then get grain orientation data
      // declare a vector to hold the grain orientation input data.
      Vector g_orient;
      if (cp)
      {
         // set the grain orientation vector from either the input grain file or the 
         // input uniform grain vector
         if (!grain_uniform) 
         {
            ifstream igrain(grain_file); 
            if (!igrain && myid == 0)
            {
               cerr << "\nCannot open grain file: " << grain_file << '\n' << endl;
            }
            // load separate grain file
            int gsize = grain_offset * ngrains;
            g_orient.Load(igrain, gsize);
            igrain.close();
         }
         else
         { 
            // else assign grain vector input grain for uniform grain orientations
            g_orient = grain_uni_vec;   
         }

         // Set the material variable quadrature function data to the grain orientation 
         // data
         printf("before setGrainData \n");

      } // end if (cp)
     
      // TODO rewrite generic material state variables function to set data 
      // on quadrature function. This will always involve non-null state variables 
      // and grain data if crystal plasticity
      setStateVarData(&stateVars, &g_orient, &fe_space, pmesh, grain_offset, matVarsOffset,
                      &matVars0);

   } // end read of mat props, state vars and grains

   // Declare quadrature functions to store a vector representation of the 
   // Cauchy stress, in Voigt notation (s_11, s_22, s_33, s_23, s_13, s_12), for 
   // the beginning of the step and the end of the step (or the end of the increment).
   QuadratureFunction sigma0(&qspace, 6);
   QuadratureFunction sigma1(&qspace, 6);
   initQuadFunc(&sigma0, 0.0, &fe_space);
   initQuadFunc(&sigma1, 0.0, &fe_space);

   // declare a quadrature function to store the material tangent stiffness 
   QuadratureFunction matGrad (&qspace, 9*9); // TODO allow for symmetry
   initQuadFunc(&matGrad, 0.0, &fe_space);

   // define the end of step (or incrementally updated) material history 
   // variables
   int vdim = matVars0.GetVDim();
   QuadratureFunction matVars1(&qspace, vdim);
   initQuadFunc(&matVars1, 0.0, &fe_space);

   // declare a quadrature function to store the beginning step solution variables 
   // for any incremental kinematics. This is less a UMAT thing and more an MFEM 
   // solution/kinematics convenience thing. This allows us to separate truly user 
   // defined history variables as required by their model from solution variables, 
   // that may be history or state variables, that are carried around out of convenience 
   // to allow MFEM to interface with a user model or implement new kinematics
   int solDim;
   solDim = (umat) ? 9 : 0;
   QuadratureFunction solVars0(&qspace, solDim);
   initQuadFunc(&solVars0, 0.0, &fe_space);

   // Define the grid functions for the beginning step configuration, end step or 
   // current configuration, the global reference configuration, the global 
   // deformed configuration, and the incremental nodal solution
   ParGridFunction x_beg(&fe_space);
   ParGridFunction x_cur(&fe_space);
   ParGridFunction x_ref(&fe_space);
   ParGridFunction x_def(&fe_space);
   ParGridFunction x_hat(&fe_space);

   // define a vector function coefficient for the initial deformation 
   // (based on a velocity projection) and reference configuration
   VectorFunctionCoefficient velProj(dim, InitialDeformation);
   VectorFunctionCoefficient refconfig(dim, ReferenceConfiguration);
  
   // Initialize the reference and current configuration grid functions 
   // with the refconfig vector function coefficient.
   x_cur.ProjectCoefficient(refconfig);
   x_ref.ProjectCoefficient(refconfig);
   x_beg.ProjectCoefficient(refconfig);

   // Define grid function for the Dirichlet boundary conditions
   ParGridFunction x_ess(&fe_space);

   // Define grid function for the incremental nodal solution grid function
   // WITH Dirichlet BCs
   ParGridFunction x_hat_bar(&fe_space);

   // Define a VectorFunctionCoefficient to initialize a grid function
   VectorFunctionCoefficient init_grid_func(dim, InitGridFunction);

   // initialize boundary condition grid functions and deformation and 
   // incremental nodal solution grid functions to zero by projecting the 
   // VectorFunctionCoefficient function onto them
   x_ess.ProjectCoefficient(init_grid_func);
   x_hat_bar.ProjectCoefficient(init_grid_func);
   x_def.ProjectCoefficient(init_grid_func);
   x_hat.ProjectCoefficient(init_grid_func);

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
                              matVars1, sigma0, sigma1, matGrad
                              matProps, solVars0, nProps, nStateVars);

   // get the essential true dof list. This may not be used.
   const Array<int> ess_tdof_list = oper.GetEssTDofList();

   // debug print
//   for (int i=0; i<ess_tdof_list.Size(); ++i) {
//      printf("ess_tdof_list[i]: %d %d \n", i, ess_tdof_list[i]);
//   }
   
   // declare solution vector
   Vector x_hat_sol(fe_space.TrueVSize()); // this sizing is correct
   x_hat_sol = 0.0;

   // initialize visualization if requested 
   socketstream vis_u, vis_p;
   if (visualization) {
      char vishost[] = "localhost";
      int  visport   = 19916;
      vis_u.open(vishost, visport);
      vis_u.precision(8);
      visualize(vis_u, pmesh, &x_cur, &x_def, "Deformation", true);
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

      // set time on the simulation variables and the model through the 
      // nonlinear mechanics operator class
      oper->SetTime(t);
      oper->SetDt(dt_real);

      // set the time for the nonzero Dirichlet BC function evaluation
      //non_zero_ess_func.SetTime(t);
      ess_bdr_func.SetTime(t);

      // register Dirichlet BCs. 
      ess_bdr = 1;

      // Perform velocity projection as initial guess for the Newton solve. 
      // Note that x_hat_bar was set to the previous x_hat solution divided by 
      // the previous time step, so that the velocity projection produces a 
      // guess to the incremental nodal displacement equal to the last 
      // incremental nodal velocity times the current time step
      x_hat_bar.ProjectCoefficient(velProj);

      // overwrite entries in x_hat_bar for dofs with Dirichlet 
      // boundary conditions (note, this routine overwrites, not adds).
      // Note: these are prescribed incremental nodal displacements at 
      // Dirichlet BC dofs.
      x_hat_bar.ProjectBdrCoefficient(ess_bdr_func); // don't need attr list as input
                                                     // pulled off the 
                                                     // VectorFunctionRestrictedCoefficient

      // populate the solution vector, x_hat_sol, with the true dofs entries in x_hat_bar.
      // At this point we initialized x_hat_bar, performed a velocity projection for 
      // all dofs, and then over-wrote the Dirichlet BC dofs with the boundary condition 
      // function.
      x_hat_bar.GetTrueDofs(x_hat_sol);

      // Solve the Newton system 
      oper.Solve(x_hat_sol);

      // distribute the solution vector to the incremental nodal displacement 
      // grid function. 
      x_hat.Distribute(x_hat_sol);

      // set the end step or current configuration
      add(x_beg, x_hat, x_cur);

      // Set the end-step _deformation_ with respect to the global reference 
      // configuration at time t=0.
      subtract(x_cur, x_ref, x_def);

      // initialize x_hat_bar = 0.0 and set x_hat_bar = x_hat / dt, which sets 
      // x_hat_bar to the previous incremental velocity. This is done in order to 
      // compute velocity projection in next time step
      x_hat_bar.ProjectCoefficient(init_grid_func);
      x_hat_bar = x_hat / dt_real; // storing inc. nodal vel here for next time step

      // update the beginning step configuration to the current converged end step 
      // configuration in preparation for the next time step
      x_beg = x_cur; 

      // update the beginning step stress and material state variables 
      // prior to the next time step for all material models other than 
      // MFEM hyperelastic. This also updates the deformation gradient with 
      // the beginning step deformation gradient stored on an Abaqus UMAT model
      oper->UpdateModel(x);

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

         GridFunction *nodes = &x_cur; // set a nodes grid function to global current configuration
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
                                             QuadratureFunction q_matGrad,
                                             QuadratureFunction q_solVars0,
                                             Vector matProps, int numProps,
                                             int nStateVars)
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
      model = new AbaqusUmatModel(&q_sigma0, &q_sigma1, &q_matGrad, &q_matVars0, &q_matVars1,
                                  &q_solVars0, matProps, numProps, nStateVars);
      
      // Add the user defined integrator
      Hform->AddDomainIntegrator(new ExaNLFIntegrator(dynamic_cast<AbaqusUmatModel*>(model)));
   {

   else if (hyperelastic) {
      model = new NeoHookean(80.E3, 140.E3, 1.0, &q_sigma0, &q_sigma1, &q_matGrad, &q_matVars0, 
                             &q_matVars1, &q_solVars0, matProps, numProps, nStateVars);
      // Add the hyperelastic integrator
      Hform->AddDomainIntegrator(new ExaNLFIntegrator(dynamic_cast<NeoHookean*>(model)));
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

void NonlinearMechOperator::UpdateModel(const Vector &x)
{
   const ParFiniteElementSpace *fes = GetFESpace();
   const FiniteElement *fe;
   const IntegrationRule *ir;

   // update state variables on a ExaModel
   for (int i = 0; i < fes->GetNE(); ++i)
   {
      fe = fes->GetFE(i);
      ir = &(IntRules.Get(fe->GetGeomType(), 2*fe->GetOrder() + 3));

      // loop over element quadrature points
      for (int j = 0; j < ir->GetNPoints(); ++j)
      {
         model->UpdateStress(i, j);
         if (model->numStateVars > 0)
         {
           model->UpdateStateVars(i, j);
         }
      }
   } 

   // update the model variables particular to the model class extension
   // NOTE: for an AbaqusUmatModel this updates the beginning step def grad, 
   model->UpdateModelVars(fes, x);

}

void NonlinearMechOperator::SetTime(const double t)
{
   solVars.SetTime(t);
   model->SetModelTime(t);
   return;
}

void NonlinearMechOperator::SetDt(const double dt)
{
   solVars.SetDt(dt);
   model->SetModelDt(dt);
   return;
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


void InitialDeformation(const Vector &x, double t,  Vector &y)
{
   // this performs a velocity projection 
   // TODO implement this routine
   
   // Note: x is coming in as incremental nodal velocity, which is 
   // the previous step's incremental nodal displacement solution 
   // divided by the previous time step

   // get the time step off the boundary condition manager
   // for the first BC, which there has to be at least one of
   BCManager & bcManager = BCManager::getInstance();
   BCData & bc_data = bcManager.GetBCInstance(0);

   double dt = bc_data.dt;

   // velocity projection is the last delta_x solution (x_hat) times 
   // the current timestep.
   y = x * dt;
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
                       bool g_q, bool g_uniform, int ngrains, int numProps,
                       int numStateVars);
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

   if (umat && (numProps < 1))
   {
      cerr << "\nMust specify material properties for umat or cp calculation." << '\n';
      err = false;
   }

   // always input a state variables file with initial values for all models
   if (numStateVars < 1)
   {
      cerr << "\nMust specifiy state variables." << '\n';
   }
   
   return err;
}

void setStateVarData(Vector* sVars, Vector* orient, ParFiniteElementSpace *fes, 
                     ParMesh *pmesh, int grainOffset, int stateVarOffset,
                     QuadratureFunction* qf);
{
   // put element grain orientation data on the quadrature points. This 
   // should eventually go into a separate subroutine
   const FiniteElement *fe;
   const IntegrationRule *ir;
   double* qf_data = qf->GetData();
   int qf_offset = qf->GetVDim();
   QuadratureSpace* qspace = qf->GetSpace();

   // check to make sure the sum of the input offsets matches the offset of 
   // the input quadrature function
   if (qf_offset != (grainOffset + stateVarOffset)
   {
      cerr << "\nsetStateVarData: State variable and grain offsets do not 
                 match quadrature function initialization." << '\n';
   }

   // get the data for the material state variables and grain orientations for 
   // nonzero grainOffset(s), which implies a crystal plasticity calculation
   double* grain_data
   if (grainOffset > 0) grain_data = orient->GetData();

   double* sVars_data = sVars->GetData();
   int elem_atr;

   // loop over elements
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
         // loop over quadrature point material state variable data
         for (int k = 0; k < stateVarOffset; ++k) 
         {
            // index into the quadrature function with stride equal to the element 
            // offset (number of material variables * numIP) plus the stride of the 
            // material variables stored at a single integration point. Index into the 
            // material variables input vector with iterator "k". Data is same for 
            // all integration points for all elements (initial data)
            qf_data[(elem_offset * i) + qf_offset * j + k] = sVar_data[k];
         } // end loop over material state variables

         // loop over quadrature point grain data
         for (int k = stateVarOffset; k < grainOffset; ++k)
         {
            // tack on the grain data at the end, which only happens if 
            // grainOffset > 0. Index into the input grain_data at the 
            // (elem_atr - 1) index position.
            qf_data[(elem_offset * i) + qf_offset * j + k]] = 
             grain_data[grainOffset * (elem_atr-1) + k];
         } // end loop over grain data
      } // end loop over quadrature points
   } // end loop over elements
}

void initQuadFunc(QuadratureFunction *qf, double val, ParFiniteElementSpace *fes)
{
//   const FiniteElement *fe;
   const IntegrationRule *ir;
   double* qf_data = qf->GetData();
   int qf_offset = qf->GetVDim();
   QuadratureSpace* qspace = qf->GetSpace();

//   printf("qf data size: %d \n", qf->Size());

   for (int i = 0; i < fes->GetNE(); ++i)
   {
//      fe = fes->GetFE(i);
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

void computeDefGrad(QuadratureFunction *qf, ParFiniteElementSpace *fes, 
                    const Vector &x0)
{
   const FiniteElement *fe;
   const IntegrationRule *ir;
   double* qf_data = qf->GetData();
   int qf_offset = qf->GetVDim(); // offset at each integration point
   QuadratureSpace* qspace = qf->GetSpace();

   // loop over elements
   for (int i = 0; i < fes->GetNE(); ++i)
   {
      // get element transformation for the ith element
      ElementTransformation* Ttr = GetElementTransformation(i);
      fe = fes->GetFE(i);

      // declare data to store shape function gradients 
      // and element Jacobians
      DenseMatrix Jrt, DSh, DS, PMatI, Jpt, F0, F1;
      int dof = fe->GetDof(), dim = fe->GetDim();

      if (qf_offset != (dim*dim))
      {
         mfem_error("computeDefGrd0 stride input arg not dim*dim");
      }

      DSh.SetSize(dof,dim)
      DS.SetSize(dof,dim);
      Jrt.SetSize(dim);
      Jpt.SetSize(dim);
      F0.SetSize(dim);
      F1.SetSize(dim);

      // get element physical coordinates
      Array<int> vdofs;
      Vector el_x;
      const Vector &px0 = Prolongate(x0);
      fes->GetElementVDofs(i, vdofs);
      px0.GetSubVector(vdofs, el_x);
      PMatI.UseExternalData(el_x.GetData(), dof, dim);
      
      ir = &(qspace->GetElementIntRule(i));
      int elem_offset = qf_offset * ir->GetNPoints();

      // loop over integration points where the quadrature function is 
      // stored
      for (int j = 0; j < ir->GetNPoints(); ++j)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         Ttr.SetIntPoint(&ip);
         CalcInverse(Ttr.Jacobian(), Jrt);
         
         fe->CalcDShape(ip, DSh);
         Mult(DSh, Jrt, DS);
         MultAtB(PMatI, DS, Jpt); 

         // store local beginning step deformation gradient for a given 
         // element and integration point from the quadrature function 
         // input argument. We want to set the new updated beginning 
         // step deformation gradient (prior to next time step) to the current
         // end step deformation gradient associated with the converged 
         // incremental solution. The converged _incremental_ def grad is Jpt 
         // that we just computed above. We compute the updated beginning 
         // step def grad as F1 = Jpt*F0; F0 = F1; We do this because we 
         // are not storing F1.
         int k = 0; 
         for (int m = 0; m < dim; ++m)
         {
            for (int n = 0; n < dim; ++n)
            {
               F0(m,n) = qf_data[i * elem_offset + j * qf_offset + k]
               ++k;
            }
         }

         // compute F1 = Jpt*F0;
         Mult(Jpt, F0, F1);

         // set new F0 = F1
         F0 = F1;
  
         // loop over element Jacobian data and populate 
         // quadrature function with the new F0 in preparation for the next 
         // time step. Note: offset0 should be the 
         // number of true state variables. 
         k = 0; 
         for (int m = 0; m < dim; ++m)
         {
            for (int n = 0; n < dim; ++n)
            {
               qf_data[i * elem_offset + j * qf_offset + k] = 
                  F0(m,n);
               ++k;
            }
         }
      }
   }

   return;
}

void setBCTimeStep(double dt, int nDBC)
{
   for (int i=0; i<nDBC; ++i) {
      BCManager & bcManager = BCManager::getInstance();
      BCData & bc = bcManager.CreateBCs( i+1 );
      bc.dt = dt;
   }

}
