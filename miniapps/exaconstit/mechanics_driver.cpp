//***********************************************************************
//
//   ExaConstit App for AM Constitutive Properties Applications
//   Author: Steven R. Wopschall
//           wopschall1@llnl.gov
//           Robert A. Carson
//           carson16@llnl.gov
//   Date: Aug. 6, 2017
//   Updated: Jan. 15, 2019
//
//   Description: The purpose of this code app is to determine bulk 
//                constitutive properties for 3D printed metal alloys. 
//                This is a nonlinear quasi-static, implicit solid 
//                mechanics code built on the MFEM library.
//                
//                Currently, only Dirichlet boundary conditions 
//                (homogeneous and inhomogeneous by dof component) have been 
//                implemented. Neumann (traction) boundary conditions and a 
//                body force are not implemented. A new ExaModel class allows 
//                one to implement arbitrary constitutive models. Currently, 
//                development efforts have focused on an Abaqus UMAT interface 
//                class extension off ExaModel and porting over the native 
//                MFEM NeoHookean hyperelastic model to the ExaModel form for 
//                the purposes of testing and debugging. Right now, only the 
//                NeoHookean model is functioning. Furthermore, the code uses 
//                psuedo-time stepping and the simulation's "final time" and 
//                "tim step" size are fixed (i.e. no adaptive time stepping).
//
//                To run this code, run the bash script mechanics.bash, found 
//                in the ExaConstit directory. This script contains notes regarding 
//                application of Dirichlet boundary conditions and has the command 
//                line input for an isochoric compression problem on a 1x1x1
//                3D brick.
//
//                where "np" is number of processors, the mesh is a 
//                simple cube mesh containing 8 elements, "tf" is the 
//                final simulation time, and dt is the timestep.
//                This has to do with how the nonzero Dirichlet BCs are applied.
//
//                Remark:
//                In principle, the mesh may be refined automatically, but 
//                this has not been thoroughly tested.
//
//                The finite element order defaults to 1 (linear), but 
//                may be increased by passing the argument at the 
//                command line "-o #" where "#" is the interpolation 
//                order (e.g. -o 2 for quadratic elements).
//
//
//                Note: the grain.txt, props.txt and state.txt files are 
//                expected inputs for CP problems, specifically ones that 
//                use the Abaqus UMAT interface class under the ExaModel.
//
//   Future Implemenations Notes:
//                
//                -Visco-plasticity constitutive model
//                -debug ability to read different mesh formats
//
//***********************************************************************
#include "mfem.hpp"
#include "mechanics_coefficient.hpp"
#include "mechanics_integrators.hpp"
#include "mechanics_solver.hpp"
#include "BCData.hpp"
#include "BCManager.hpp"
#include <memory>
#include <iostream>
#include <fstream>
#include <math.h>

using namespace std;
using namespace mfem;

class SimVars
{
protected:
   double time;
   double dt;
public:
   double GetTime() const { return time; }
   double GetDTime() const { return dt; }

   void SetTime(double t) { time = t; }
   void SetDt(double dtime) { dt = dtime; }
};

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
   ExaModel *model;
   /// Variable telling us if we should use the UMAT specific
   /// stuff
   bool umat_used;
   int newton_iter;

public:
   NonlinearMechOperator(ParFiniteElementSpace &fes,
                         Array<int> &ess_bdr,
                         double rel_tol,
                         double abs_tol,
                         int iter,
                         bool gmres,
			 bool cg,
                         bool slu, 
                         bool hyperelastic,
                         bool umat,
                         QuadratureFunction &q_matVars0,
                         QuadratureFunction &q_matVars1,
                         QuadratureFunction &q_sigma0,
                         QuadratureFunction &q_sigma1,
                         QuadratureFunction &q_matGrad,
                         QuadratureFunction &q_kinVars0,
			 QuadratureFunction &q_vonMises,
			 ParGridFunction &beg_crds,
			 ParGridFunction &end_crds,
			 ParMesh *&pmesh,
                         Vector &matProps, int numProps,
                         int nStateVars);

   /// Required to use the native newton solver
   virtual Operator &GetGradient(const Vector &x) const;
   virtual void Mult(const Vector &k, Vector &y) const;
   //We need the solver to update the end coords after each iteration has been complete
   //We'll also want to have a way to update the coords before we start running the simulations.
   //It might also allow us to set a velocity at every point, so we could test the models almost
   //as if we're doing a MPS.
   void UpdateEndCoords(const Vector& vel) const;  
   /// Driver for the newton solver
   void Solve(Vector &x) const;
   
   /// Solve the Newton system for the 1st time step
   /// It was found that for large meshes a ramp up to our desired applied BC might
   /// be needed. It should be noted that this is no longer a const function since
   /// we modify several values/objects held by our class.
   void SolveInit(Vector &x);

   /// Get essential true dof list, if required
   const Array<int> &GetEssTDofList();

   /// Get FE space
   const ParFiniteElementSpace *GetFESpace() { return &fe_space; }

   /// routine to update beginning step model variables with converged end 
   /// step values
   void UpdateModel(const Vector &x);
   /// Computes a volume average tensor/vector of some quadrature function
   /// it returns the vol avg value.
   void ComputeVolAvgTensor(const ParFiniteElementSpace* fes,
                            const QuadratureFunction* qf,
			    Vector& tensor,
			    int size);
  
   void ProjectModelStress(ParGridFunction &s);
   void ProjectVonMisesStress(ParGridFunction &vm);

   void SetTime(const double t);
   void SetDt(const double dt);
   void SetModelDebugFlg(const bool dbg);

   void DebugPrintModelVars(int procID, double time);
   /// Tests the deformation gradient function
   void testFuncs(const Vector &x0, ParFiniteElementSpace *fes);

   virtual ~NonlinearMechOperator();
};

void visualize(ostream &out, ParMesh *mesh, ParGridFunction *deformed_nodes,
               ParGridFunction *field, const char *field_name = NULL,
               bool init_vis = false);

// set kinematic functions and boundary condition functions
void ReferenceConfiguration(const Vector &x, Vector &y);
// allows you to assign an initial deformation
void InitialDeformation(const Vector &x, double t,  Vector &y);
// computes an approximate velocity based on the coordinates at the beginning
// and end time steps. This function was from when the code was displacement
// based
void Velocity(const Vector &x, double t, Vector &y);
void DirBdrFunc(const Vector &x, double t, int attr_id, Vector &y);
// This initializes some grid function
void InitGridFunction(const Vector &x, Vector &y);

// material input check routine
bool checkMaterialArgs(bool hyperelastic, bool umat, bool cp, bool g_euler, 
                       bool g_q, bool g_custom, int ngrains, int numProps,
                       int numStateVars);

// material state variable and grain data setter routine
void setStateVarData(Vector* sVars, Vector* orient, ParFiniteElementSpace *fes, 
                     ParMesh *pmesh, int grainOffset, int grainIntoStateVarOffset, 
                     int stateVarSize, QuadratureFunction* qf);

// initialize a quadrature function with a single input value, val.
void initQuadFunc(QuadratureFunction *qf, double val, ParFiniteElementSpace *fes);

//initialize a quadrature function that is really a tensor with the identity matrix.
//currently only works for 3x3 tensors.
void initQuadFuncTensorIdentity(QuadratureFunction *qf, ParFiniteElementSpace *fes);
// set the time step on the boundary condition objects
void setBCTimeStep(double dt, int nDBC);

// set the element grain ids from vector data populated from a 
// grain map input text file
void setElementGrainIDs(Mesh *mesh, const Vector grainMap, int ncols, int offset);

// used to reset boundary conditions from MFEM convention using 
// Make3D() called from the mesh constructor to ExaConstit convention
void setBdrConditions(Mesh *mesh);

// reorder mesh elements in MFEM generated mesh using Make3D() in 
// mesh constructor so that the ordering matches the element ordering 
// in the input grain map (e.g. from CA calculation)
void reorderMeshElements(Mesh *mesh, const int nx);

//  provides a constant gradient across a grid function
void test_deformation_field_set(ParGridFunction *gf, Vector *vals, ParFiniteElementSpace *fes);

int main(int argc, char *argv[])
{

   // Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   //Here we start a timer to time everything
   double start = MPI_Wtime();
   // print the version of the code being run
   if(myid == 0) printf("MFEM Version: %d \n", GetVersion());
   
   // Parse command-line options.

   // mesh variables
   const char *mesh_file = "../../data/cube-hex-ro.mesh";
   bool cubit = false;
   bool hex_mesh_gen = false; // TODO test hex_mesh_gen
   double mx = 0.0; // edge dimension (mx = my = mz)
   int  nx = 0; // number of cells on an edge (nx = ny = nz)
 

   // serial and parallel refinement levels
   int ser_ref_levels = 0;
   int par_ref_levels = 0;

   // polynomial interpolation order
   int order = 1;

   // final simulation time and time step (set each to 1.0 for 
   // single step debug)
   double t_final = 1.0;
   double dt = 1.0;
   // We have a custom dt flag
   bool dt_cust = false;
   // Number of time steps to take
   int nsteps = 1;
   // File to read the custom time steps from
   const char *dt_file = "custom_dt.txt";
   // Vector to hold custom time steps if there are any
   Vector cust_dt;

   // visualization input args
   bool visualization = true;
   int vis_steps = 1;
   // visualization variable for visit
   bool visit = true;

   //Where to store the end time step files
   const char *basename = "results/exaconstit";
   
   // newton input args
   double newton_rel_tol = 1.0e-6;
   double newton_abs_tol = 1.0e-8;
   int newton_iter = 25;
   
   // solver input args
   // GMRES is currently set as the default iterative solver
   // until the bug in the PCG solver is found and fixed.
   bool gmres_solver = true;
   bool pcg_solver = false;
   bool slu_solver = false;
   bool grad_debug = false;

   // input arg to specify Abaqus UMAT
   bool umat = false;

   // input arg to specify crystal plasticity or hyperelasticity 
   // (for testing)
   bool cp = false;
   bool hyperelastic = false;

   // grain input arguments
   const char *grain_file = "grains.txt"; // grain orientations (F_p_inv for Curt's UMAT?)
   const char *grain_map = "grain_map.txt"; // map of grain id to element centroid
   int ngrains = 0;
   bool grain_euler = false;
   bool grain_q = false;
   bool grain_custom = false; // for custom grain specification
   int grain_custom_stride = 0; // TODO check that this is used with "grain_custom"
   int grain_statevar_offset = -1; 

   // material properties input arguments
   const char *props_file = "props.txt";
   int nProps = 1; // at least have one dummy property

   // state variables file with constant values used to initialize ALL integration points
   const char *state_file = "state.txt";
   int numStateVars = 1; // at least have one dummy property
  
   // boundary condition input args
   Array<int> ess_id;   // essential bc ids for the whole boundary
   Vector     ess_disp; // vector of displacement components for each attribute in ess_id
   Array<int> ess_comp; // component combo (x,y,z = -1, x = 1, y = 2, z = 3, 
                        // xy = 4, yz = 5, xz = 6, free = 0 

   // specify all input arguments
   // it would be nice if we could just make this all read in from some configuration
   // file format so we don't have to use the command line.
   // .toml files might not be a bad option to look into
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&grain_map, "-gmap", "--grain-map", 
                  "Map of element/cell centroids to grain ids.");
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
   args.AddOption(&hex_mesh_gen, "-hexmesh", "--hex-mesh", "-no-hexmesh",
                  "--no-hex-mesh", "Use an auto-generated parallelepiped hex mesh.");
   args.AddOption(&mx, "-mx", "--mesh-x", 
                  "Auto-gen hex mesh length, mx = my = mz.");
   args.AddOption(&nx, "-nx", "--num-mesh-x", 
                  "Auto-gen hex mesh element number along edge, nx = ny = nz.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt_cust, "-vardt", "--variabledt", "-no-vardt",
		  "--no-variabledt", "Use a variable time step or not.");
   args.AddOption(&nsteps, "-nsteps", "--num_time_steps",
		  "The number of custom time steps.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&dt_file, "-cust-dt","--custom-dt",
		  "Custom time step file that can be used");
   args.AddOption(&slu_solver, "-slu", "--superlu", "-no-slu",
                  "--no-superlu", "Use the SuperLU Solver.");
   args.AddOption(&gmres_solver, "-gmres", "--gmres", "-no-gmres", "--no-gmres",
                   "Use gmres, otherwise minimum residual is used.");
   args.AddOption(&pcg_solver, "-pcg", "--pcg", "-no-pcg", "--no-pcg",
		  "Use pcg, otherwise minimum residual is used.");
   args.AddOption(&grad_debug, "-gdbg", "--grad-debug", "-no-gdbg",
                  "--no-grad-debug",
                  "Use finite difference gradient calculation.");
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
   args.AddOption(&visit, "-visit", "--visit", "-no-visit",
                  "--no-visit",
                  "Output visit data or not.");
   args.AddOption(&grain_euler, "-ge", "--euler-grain-orientations", "-no-ge",
                  "--no-euler-grain-orientations", 
                  "Use Euler angles to define grain orientations.");
   args.AddOption(&grain_q, "-gq", "--quaternion-grain-orientations", "-no-gq",
                  "--no-quaternion-grain-orientations", 
                  "Use quaternions to define grain orientations.");
   args.AddOption(&grain_custom, "-gc", "--custom-grain-orientations", "-no-gc",
                  "--no-custom-grain-orientations",
                  "Use custom grain orientations.");
   args.AddOption(&grain_custom_stride, "-gcstride", "--custom-grain-stride", 
                  "Stride for custom grain orientation data.");
   args.AddOption(&grain_statevar_offset, "-gsvoffset", "--grain-state-var-offset",
                  "Offset for grain orientation data insertion into material state array, if applicable.");
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


   //Check to see if a custom dt file was used
   //if so read that in and if not set the nsteps that we're going to use
   if(dt_cust){
     if(myid == 0) printf("Reading in custom dt file. \n");
      ifstream idt(dt_file);
      if (!idt && myid == 0)
      {
         cerr << "\nCannot open grain map file: " << grain_map << '\n' << endl;
      }                        
      cust_dt.Load(idt, nsteps);
      idt.close();
   }else{
     nsteps = ceil(t_final/dt);
     if(myid==0) printf("number of steps %d \n", nsteps);
   }
   
   // Check material model argument input parameters for valid combinations
   if(myid == 0) printf("after input before checkMaterialArgs. \n");
   bool err = checkMaterialArgs(hyperelastic, umat, cp, grain_euler, grain_q, grain_custom, 
              ngrains, nProps, numStateVars);
   if (!err && myid == 0) 
   {
      cerr << "\nInconsistent material input; check args" << '\n';
   }

   // check mesh input arguments
   if (cubit && hex_mesh_gen)
   {
      cerr << "\nCannot specify a cubit mesh and MFEM auto-gen mesh" << '\n';
   }

   // Open the mesh
   if(myid == 0) printf("before reading the mesh. \n");
   Mesh *mesh;
   Vector g_map;
   if (cubit) 
   {
      //named_ifgzstream imesh(mesh_file);
      mesh = new Mesh(mesh_file, 1, 1);
   }
   else if (hex_mesh_gen)
   {
      if (nx == 0 || mx == 0)
      {
         cerr << "\nMust input mesh geometry/discretization for hex_mesh_gen" << '\n';
      }

      // use constructor to generate a 3D cuboidal mesh with 8 node hexes
      mesh = new Mesh(nx, nx, nx, Element::HEXAHEDRON, 0, mx, mx, mx); 
   }
   else // read in mesh file
   {
      if(myid == 0) printf("opening mesh file \n");

      ifstream imesh(mesh_file);
      if(myid == 0) printf("after declaring imesh \n");
      if (!imesh)
      {
         if (myid == 0)
         {
            cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
         }
         MPI_Finalize();
         return 2;
      }
  
      if(myid == 0) printf("before declaring new mesh \n");
      mesh = new Mesh(imesh, 1, 1, true);
      if(myid == 0) printf("after declaring new mesh \n");
      imesh.close();
      //If we're doing xtal plasticity stuff read in the grain map file
      if(cp){
         ifstream igmap(grain_map);
         if (!igmap && myid == 0)
         {
           cerr << "\nCannot open grain map file: " << grain_map << '\n' << endl;
         }
         //This should here just be the number of elements
         int gmapSize = mesh->GetNE();
         g_map.Load(igmap, gmapSize);
         igmap.close();
         // 1 tells you how many number of possible columns there are and 0 tells
         // us what column we want to use for grain ids
         setElementGrainIDs(mesh, g_map, 1, 0);
      }
   }

   // read in the grain map if using a MFEM auto generated cuboidal mesh
   if (hex_mesh_gen)
   {
      if(myid == 0) printf("using mfem hex mesh generator \n");

      ifstream igmap(grain_map);
      if (!igmap && myid == 0)
      {
         cerr << "\nCannot open grain map file: " << grain_map << '\n' << endl;
      }
      
      int gmapSize = mesh->GetNE();
      g_map.Load(igmap, gmapSize);
      igmap.close();

      // reorder elements to conform to ordering convention in grain map file
      reorderMeshElements(mesh, nx);

      // reset boundary conditions from 
      setBdrConditions(mesh);
 
      // set grain ids as element attributes on the mesh
      //The offset of where the grain index is located is
      //location - 1.
      setElementGrainIDs(mesh, g_map, 1, 0);
   }

   // declare pointer to parallel mesh object
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

   if(myid == 0) printf("after mesh section. \n");

   int dim = pmesh->Dimension();

   // Define the finite element spaces for displacement field
   FiniteElementCollection *fe_coll = NULL;
   fe_coll = new  H1_FECollection(order, dim);
   ParFiniteElementSpace fe_space(pmesh, fe_coll, dim);

   int order_v = 1;
   int order_0 = 0;
   
   //Here we're setting up a discontinuous so that we'll use later to interpolate
   //our quadrature functions from
   L2_FECollection l2_fec(order_0, dim);
   ParFiniteElementSpace l2_fes(pmesh, &l2_fec);
   ParGridFunction vonMises(&l2_fes);
   
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
   if (grain_euler) 
   {
      grain_offset = 3; 
   }
   else if (grain_q) 
   {
      grain_offset = 4;
   }
   else if (grain_custom) 
   {
      if (grain_custom_stride == 0)
      { 
         cerr << "\nMust specify a grain stride for grain_custom input" << '\n';
      }
      grain_offset = grain_custom_stride;
   }

   // set the offset for the matVars quadrature function. This is the number of 
   // state variables (stored at each integration point) and then the grain offset, 
   // which is the number of variables defining the grain data stored at each 
   // integration point. In general, these may come in as different data sets, 
   // even though they will be stored in a single material state variable 
   // quadrature function.
   int matVarsOffset = numStateVars + grain_offset;

   // Define a quadrature space and material history variable QuadratureFunction.
   int intOrder = 2*order+1;
   QuadratureSpace qspace(pmesh, intOrder); // 3rd order polynomial for 2x2x2 quadrature
                                            // for first order finite elements. 
   QuadratureFunction matVars0(&qspace, matVarsOffset); 
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
   if(myid == 0) printf("before reading in matProps and stateVars. \n");
   { // read in props, material state vars and grains if crystal plasticity
      ifstream iprops(props_file);
      if (!iprops && myid == 0)
      {
         cerr << "\nCannot open material properties file: " << props_file << '\n' << endl;
      }

      // load material properties
      matProps.Load(iprops, nProps);
      iprops.close();
      
      if(myid == 0) printf("after loading matProps. \n");
      
      // read in state variables file
      ifstream istateVars(state_file);
      if (!istateVars && myid == 0)
      {
         cerr << "\nCannot open state variables file: " << state_file << '\n' << endl;
      }

      // load state variables
      stateVars.Load(istateVars, numStateVars);
      istateVars.close();
      if(myid == 0) printf("after loading stateVars. \n");

      // if using a crystal plasticity model then get grain orientation data
      // declare a vector to hold the grain orientation input data. This data is per grain 
      // with a stride set previously as grain_offset
      Vector g_orient;
      if(myid == 0) printf("before loading g_orient. \n");
      if (cp)
      {
         // set the grain orientation vector from the input grain file
         ifstream igrain(grain_file); 
         if (!igrain && myid == 0)
         {
            cerr << "\nCannot open grain file: " << grain_file << '\n' << endl;
         }
         // load separate grain file
         int gsize = grain_offset * ngrains;
         g_orient.Load(igrain, gsize);
         igrain.close();
         if(myid == 0) printf("after loading g_orient. \n");

      } // end if (cp)
     
      // set the state var data on the quadrature function
      if(myid == 0) printf("before setStateVarData. \n");
      setStateVarData(&stateVars, &g_orient, &fe_space, pmesh, grain_offset, 
                      grain_statevar_offset, numStateVars, &matVars0);
      if(myid == 0) printf("after setStateVarData. \n");
      
   } // end read of mat props, state vars and grains

   // Declare quadrature functions to store a vector representation of the 
   // Cauchy stress, in Voigt notation (s_11, s_22, s_33, s_23, s_13, s_12), for 
   // the beginning of the step and the end of the step.
   // For hyperelastic formulations that update the PK1 stress directly we 
   // compute the Cauchy stress from the PK1 stress and deformation gradient 
   // and store the 6 symmetric components of Cauchy stress
   int stressOffset = 6;
   QuadratureFunction sigma0(&qspace, stressOffset);
   QuadratureFunction sigma1(&qspace, stressOffset);
   QuadratureFunction q_vonMises(&qspace, 1);
   initQuadFunc(&sigma0, 0.0, &fe_space);
   initQuadFunc(&sigma1, 0.0, &fe_space);
   initQuadFunc(&q_vonMises, 0.0, &fe_space);

   // declare a quadrature function to store the material tangent stiffness.
   // This assumes that a hyperelastic material model will solve directly for the 
   // PK1 stress and that any other model will more traditionally deal with Cauchy 
   // stress. The material tangent stiffness of the PK1 stress is actually the full 
   // element stiffness (24 x 24 for linear hex elements) based on the native MFEM 
   // hyperelastic calculations. The tangent stiffness of the Cauchy stress will 
   // actually be the real material tangent stiffness (4th order tensor) and have 
   // 36 components due to symmetry.
   int matGradOffset = (hyperelastic) ? 12 : 36;
   QuadratureFunction matGrd(&qspace, matGradOffset);
   initQuadFunc(&matGrd, 0.0, &fe_space);

   // define the end of step (or incrementally updated) material history 
   // variables
   int vdim = matVars0.GetVDim();
   QuadratureFunction matVars1(&qspace, vdim);
   initQuadFunc(&matVars1, 0.0, &fe_space);

   // declare a quadrature function to store the beginning step kinematic variables 
   // for any incremental kinematics. Right now this is used to store the beginning 
   // step deformation gradient on the model.
   int kinDim = 9;
   QuadratureFunction kinVars0(&qspace, kinDim);
   initQuadFuncTensorIdentity(&kinVars0, &fe_space);

   // Define a grid function for the global reference configuration, the beginning 
   // step configuration, the global deformation, the current configuration/solution 
   // guess, and the incremental nodal displacements
   ParGridFunction x_ref(&fe_space);
   ParGridFunction x_beg(&fe_space);
   ParGridFunction x_cur(&fe_space);
   //x_diff would be our displacement
   ParGridFunction x_diff(&fe_space);
   ParGridFunction v_cur(&fe_space);
   
   // define a vector function coefficient for the initial deformation 
   // (based on a velocity projection) and reference configuration.
   // Additionally define a vector function coefficient for computing 
   // the grid velocity prior to a velocity projection
   VectorFunctionCoefficient refconfig(dim, ReferenceConfiguration);
  
   // Initialize the reference and beginning step configuration grid functions 
   // with the refconfig vector function coefficient.
   x_beg.ProjectCoefficient(refconfig);
   x_ref.ProjectCoefficient(refconfig);
   x_cur.ProjectCoefficient(refconfig);

   // Define grid function for the velocity solution grid function
   // WITH Dirichlet BCs

   // Define a VectorFunctionCoefficient to initialize a grid function
   VectorFunctionCoefficient init_grid_func(dim, InitGridFunction);

   // initialize boundary condition, velocity, and
   // incremental nodal displacment grid functions by projection the 
   // VectorFunctionCoefficient function onto them
   x_diff.ProjectCoefficient(init_grid_func);
   v_cur.ProjectCoefficient(init_grid_func);
   
   // define a boundary attribute array and initialize to 0
   Array<int> ess_bdr;
   // set the size of the essential boundary conditions attribute array
   ess_bdr.SetSize(fe_space.GetMesh()->bdr_attributes.Max());
   ess_bdr = 0;

   // setup inhomogeneous essential boundary conditions using the boundary 
   // condition manager (BCManager) and boundary condition data (BCData) 
   // classes developed for ExaConstit.
   if (ess_disp.Size() != 3*ess_id.Size()) {
      cerr << "\nMust specify three Dirichlet components per essential boundary attribute" << '\n' << endl;
   }

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
      }
      ++numDirBCs;
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
   if(myid == 0) printf("before NonlinearMechOperator constructor. \n");
   NonlinearMechOperator oper(fe_space, ess_bdr, 
                              newton_rel_tol, newton_abs_tol, 
                              newton_iter, gmres_solver,pcg_solver, slu_solver,
                              hyperelastic, umat, matVars0, 
                              matVars1, sigma0, sigma1, matGrd,
                              kinVars0, q_vonMises, x_beg, x_cur, pmesh,
			      matProps, nProps, matVarsOffset);
   if(myid == 0) printf("after NonlinearMechOperator constructor. \n");

   oper.SetModelDebugFlg(grad_debug);

   if(myid == 0) printf("after SetModelDebugFlg \n");
   
   // get the essential true dof list. This may not be used.
   const Array<int> ess_tdof_list = oper.GetEssTDofList();
   
   // declare incremental nodal displacement solution vector
   Vector v_sol(fe_space.TrueVSize()); // this sizing is correct
   v_sol = 0.0;
   
   // initialize visualization if requested 
   socketstream vis_u, vis_p;
   if (visualization) {
      char vishost[] = "localhost";
      int  visport   = 19916;
      vis_u.open(vishost, visport);
      vis_u.precision(8);
      visualize(vis_u, pmesh, &x_beg, &x_cur, "Deformation", true);
      // Make sure all ranks have sent their 'u' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());
   }

   
   // Save data for VisIt visualization.
   // The below is used to take advantage of mfem's custom Visit plugin
   // It could also allow for restart files later on.
   // If we have large simulations although the current method of printing everything
   // as text will cause issues. The data should really be saved in some binary format.
   // If you don't then you'll often find that the printed data lags behind where
   // the simulation is currently at. This really becomes noticiable if you have
   // a lot of data that you want to output for the user. It might be nice if this
   // was either a netcdf or hdf5 type format instead.
   VisItDataCollection visit_dc(basename, pmesh);
   if (visit)
   {
     visit_dc.RegisterField("Displacement",  &x_diff);
     visit_dc.RegisterQField("Stress", &sigma0);
     visit_dc.RegisterField("Velocity", &v_cur);
     //visit_dc.RegisterQField("State Variables", &matVars0);
     //visit_dc.RegisterQField("DefGrad", &kinVars0);
     //visit_dc.RegisterField("VonMises", &vonMises);
      
     visit_dc.SetCycle(0);
     visit_dc.SetTime(0.0);
     visit_dc.Save();
   }

   if(myid == 0) printf("after visualization if-block \n");

   // initialize/set the time
   double t = 0.0;
   oper.SetTime(t); 

   bool last_step = false;
   {
     GridFunction *nodes = &x_beg; // set a nodes grid function to global current configuration         
     int owns_nodes = 0;
     pmesh->SwapNodes(nodes, owns_nodes); // pmesh has current configuration nodes
     nodes = NULL;
   }

   ess_bdr_func.SetTime(0.0);
   setBCTimeStep(dt, numDirBCs);

   double dt_real;
   
   for (int ti = 1; ti <= nsteps; ti++)
   {

      if(myid == 0) printf("inside timestep loop %d \n", ti);
      //Get out our current delta time step
      if(dt_cust){
         dt_real = cust_dt[ti - 1];
      }else{
         dt_real = min(dt, t_final - t);
      }

      // set the time step on the boundary conditions
      setBCTimeStep(dt_real, numDirBCs);

      // compute current time
      t = t + dt_real;

      // set time on the simulation variables and the model through the 
      // nonlinear mechanics operator class
      oper.SetTime(t);
      oper.SetDt(dt_real);

      // set the time for the nonzero Dirichlet BC function evaluation
      ess_bdr_func.SetTime(t);

      // register Dirichlet BCs. 
      ess_bdr = 1;

      //Now that we're doing velocity based we can just overwrite our data with the ess_bdr_func
      v_cur.ProjectBdrCoefficient(ess_bdr_func); // don't need attr list as input
                                                 // pulled off the 
                                                 // VectorFunctionRestrictedCoefficient

      // populate the solution vector, v_sol, with the true dofs entries in v_cur.
      v_cur.GetTrueDofs(v_sol);
      
      //For the 1st time step, we might need to solve things using a ramp up to
      //our desired applied velocity boundary conditions.
      if(ti == 1){
         oper.SolveInit(v_sol);
      }else{
         oper.Solve(v_sol);
      }
      // distribute the solution vector to v_cur
      v_cur.Distribute(v_sol);

      // find the displacement vector as u = x_cur - x_reference
      subtract(x_cur, x_ref, x_diff);


      // update the beginning step stress and material state variables 
      // prior to the next time step for all Exa material models
      // This also updates the deformation gradient with the beginning step 
      // deformation gradient stored on an Exa model
      oper.UpdateModel(v_sol);

      //Update our beginning time step coords with our end time step coords
      x_beg = x_cur;
      
      last_step = (t >= t_final - 1e-8*dt);

      if (last_step || (ti % vis_steps) == 0)
      {
         if (myid == 0)
         {
            cout << "step " << ti << ", t = " << t << endl;
         }
      }

      { // mesh and stress output. Consider moving this to a separate routine
         //We might not want to update the vonMises stuff
//         oper.ProjectVonMisesStress(vonMises);
         if (visit)
         {
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(t);
            //Our visit data is now saved off
            visit_dc.Save();
         }

      } // end output scope
   } // end loop over time steps

//   This was for a really large simulation so only the last step was desired
//   if (visit)
//   {
//      visit_dc.SetCycle(1);
//      visit_dc.SetTime(t);
//      visit_dc.Save();
//   }

   // Free the used memory.
   delete pmesh;
   //Now find out how long everything took to run roughly
   double end = MPI_Wtime();
   if(myid == 0) printf("The process took %lf seconds to run\n", (end-start));

   MPI_Finalize();

   return 0;
}

NonlinearMechOperator::NonlinearMechOperator(ParFiniteElementSpace &fes,
                                             Array<int> &ess_bdr,
                                             double rel_tol,
                                             double abs_tol,
                                             int iter,
                                             bool gmres,
                                             bool pcg,
                                             bool slu, 
                                             bool hyperelastic,
                                             bool umat,
                                             QuadratureFunction &q_matVars0,
                                             QuadratureFunction &q_matVars1,
                                             QuadratureFunction &q_sigma0,
                                             QuadratureFunction &q_sigma1,
                                             QuadratureFunction &q_matGrad,
                                             QuadratureFunction &q_kinVars0,
                                             QuadratureFunction &q_vonMises,
                                             ParGridFunction &beg_crds,
                                             ParGridFunction &end_crds,
                                             ParMesh *&pmesh,
                                             Vector &matProps, int numProps,
                                             int nStateVars)
   : TimeDependentOperator(fes.TrueVSize()), fe_space(fes),
     newton_solver(fes.GetComm())
{
   Vector * rhs;
   rhs = NULL;

   umat_used = umat;
     
   // Define the parallel nonlinear form 
   Hform = new ParNonlinearForm(&fes);
   
   // Set the essential boundary conditions
   Hform->SetEssentialBCPartial(ess_bdr, rhs);
   
   if (umat) {
      model = new AbaqusUmatModel(&q_sigma0, &q_sigma1, &q_matGrad, &q_matVars0, &q_matVars1,
                                  &q_kinVars0, &beg_crds, &end_crds, pmesh,
                                  &matProps, numProps, nStateVars);

      // Add the user defined integrator
      Hform->AddDomainIntegrator(new ExaNLFIntegrator(dynamic_cast<AbaqusUmatModel*>(model)));
   }else if (hyperelastic) {
      model = new NeoHookean(&q_sigma0, &q_sigma1, &q_matGrad, &q_matVars0, 
                             &q_matVars1, &q_kinVars0, &beg_crds, &end_crds, pmesh,
                             &matProps, numProps, nStateVars,
                             80.E3, 140.E3, 1.0);

      // Add the hyperelastic integrator
      Hform->AddDomainIntegrator(new ExaNLFIntegrator(dynamic_cast<NeoHookean*>(model)));

   }

   model->setVonMisesPtr(&q_vonMises);

   if (gmres) {
      
      HypreBoomerAMG *prec_amg = new HypreBoomerAMG();
      HYPRE_Solver h_amg = (HYPRE_Solver) *prec_amg;
      HYPRE_Real st_val = 0.90;
      HYPRE_Real rt_val = -10.0;
      HYPRE_Real om_val = 1.0;
//      HypreBoomerAMG *prec_amg = new HypreBoomerAMG();
      //
      int ml = HYPRE_BoomerAMGSetMaxLevels(h_amg, 30);
      int ct = HYPRE_BoomerAMGSetCoarsenType(h_amg, 0);
      int mt = HYPRE_BoomerAMGSetMeasureType(h_amg, 0);
      int st = HYPRE_BoomerAMGSetStrongThreshold(h_amg, st_val);
      int ns = HYPRE_BoomerAMGSetNumSweeps(h_amg, 3);
      int rt = HYPRE_BoomerAMGSetRelaxType(h_amg, 8);
      //int rwt = HYPRE_BoomerAMGSetRelaxWt(h_amg, rt_val);
      //int ro = HYPRE_BoomerAMGSetOuterWt(h_amg, om_val);
      //Dimensionality of our problem
      int ss = HYPRE_BoomerAMGSetNumFunctions(h_amg, 3);
      int smt = HYPRE_BoomerAMGSetSmoothType(h_amg, 3);
      int snl = HYPRE_BoomerAMGSetSmoothNumLevels(h_amg, 3);
      int sns = HYPRE_BoomerAMGSetSmoothNumSweeps(h_amg, 3);
      int sv = HYPRE_BoomerAMGSetVariant(h_amg, 0);
      int so = HYPRE_BoomerAMGSetOverlap(h_amg, 0);
      int sdt = HYPRE_BoomerAMGSetDomainType(h_amg, 1);
      int srw = HYPRE_BoomerAMGSetSchwarzRlxWeight(h_amg, rt_val);
      
      prec_amg->SetPrintLevel(0);

//      prec_amg->SetElasticityOptions(&fe_space);
      J_prec = prec_amg;

      GMRESSolver *J_gmres = new GMRESSolver(fe_space.GetComm());
      //These tolerances are currently hard coded while things are being debugged
      //but they should eventually be moved back to being set by the options
//      J_gmres->iterative_mode = false;
      //The relative tolerance should be at this point or smaller
      J_gmres->SetRelTol(1e-10);
      //The absolute tolerance could probably get even smaller then this
      J_gmres->SetAbsTol(1e-30);
      J_gmres->SetMaxIter(150);
      J_gmres->SetPrintLevel(0);
      J_gmres->SetPreconditioner(*J_prec);
      J_solver = J_gmres;

   }else if (pcg){

      HypreBoomerAMG *prec_amg = new HypreBoomerAMG();
      HYPRE_Solver h_amg = (HYPRE_Solver) *prec_amg;
      HYPRE_Real st_val = 0.90;
      HYPRE_Real rt_val = -10.0;
      HYPRE_Real om_val = 1.0;
//      HypreBoomerAMG *prec_amg = new HypreBoomerAMG();
      //
      int ml = HYPRE_BoomerAMGSetMaxLevels(h_amg, 30);
      int ct = HYPRE_BoomerAMGSetCoarsenType(h_amg, 0);
      int mt = HYPRE_BoomerAMGSetMeasureType(h_amg, 0);
      int st = HYPRE_BoomerAMGSetStrongThreshold(h_amg, st_val);
      int ns = HYPRE_BoomerAMGSetNumSweeps(h_amg, 3);
      int rt = HYPRE_BoomerAMGSetRelaxType(h_amg, 8);
      //int rwt = HYPRE_BoomerAMGSetRelaxWt(h_amg, rt_val);
      //int ro = HYPRE_BoomerAMGSetOuterWt(h_amg, om_val);
      //Dimensionality of our problem
      int ss = HYPRE_BoomerAMGSetNumFunctions(h_amg, 3);
      int smt = HYPRE_BoomerAMGSetSmoothType(h_amg, 3);
      int snl = HYPRE_BoomerAMGSetSmoothNumLevels(h_amg, 3);
      int sns = HYPRE_BoomerAMGSetSmoothNumSweeps(h_amg, 3);
      int sv = HYPRE_BoomerAMGSetVariant(h_amg, 0);
      int so = HYPRE_BoomerAMGSetOverlap(h_amg, 0);
      int sdt = HYPRE_BoomerAMGSetDomainType(h_amg, 1);
      int srw = HYPRE_BoomerAMGSetSchwarzRlxWeight(h_amg, rt_val);

      prec_amg->SetPrintLevel(0);
      J_prec = prec_amg;

      CGSolver *J_pcg = new CGSolver(fe_space.GetComm());
      //These tolerances are currently hard coded while things are being debugged
      //but they should eventually be moved back to being set by the options
      //The relative tolerance should be at this point or smaller
      J_pcg->SetRelTol(1e-10);
      //The absolute tolerance could probably get even smaller then this
      J_pcg->SetAbsTol(1e-30);
      J_pcg->SetMaxIter(150);
      J_pcg->SetPrintLevel(0);
      J_pcg->iterative_mode = true;
      J_pcg->SetPreconditioner(*J_prec);
      J_solver = J_pcg;

   }//The SuperLU capabilities were gotten rid of due to the size of our systems
   //no longer making it a viable option to keep 1e6+ dof systems
   //Also, a well tuned PCG should be much faster than SuperLU for systems roughly
   //5e5 and up.
   else {
      printf("using minres solver \n");
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
   //We might want to change our # iterations used in the newton solver
   //for the 1st time step. We'll want to swap back to the old one after this
   //step.
   newton_iter = iter;

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
   //We provide an initial guess for what our current coordinates will look like
   //based on what our last time steps solution was for our velocity field.
   if(!model->GetEndCoordsMesh()){
     model->SwapMeshNodes();
   }
   //The end nodes are updated before the 1st step of the solution here so we're good.
   newton_solver.Mult(zero, x);
   //Just gotta be safe incase something in the solver wasn't playing nice and didn't swap things
   //back to the current configuration...
   if(!model->GetEndCoordsMesh()){
     model->SwapMeshNodes();
   }
   //Once the system has finished solving, our current coordinates configuration are based on what our
   //converged velocity field ended up being equal to.
   MFEM_VERIFY(newton_solver.GetConverged(), "Newton Solver did not converge.");
}

// Solve the Newton system for the 1st time step
// It was found that for large meshes a ramp up to our desired applied BC might
// be needed.
void NonlinearMechOperator::SolveInit(Vector &x)
{
   Vector zero;
   Vector init_x(x);
   //We shouldn't need more than 5 NR to converge to a solution during our
   //initial step in our solution.
   //We'll change this back to the old value at the end of the function.
   newton_solver.SetMaxIter(5);
   //We provide an initial guess for what our current coordinates will look like
   //based on what our last time steps solution was for our velocity field.
   if(!model->GetEndCoordsMesh()){
      model->SwapMeshNodes();
   }
   //The end nodes are updated before the 1st step of the solution here so we're good.
   newton_solver.Mult(zero, x);
   //Just gotta be safe incase something in the solver wasn't playing nice and didn't swap things
   //back to the current configuration...
   if(!model->GetEndCoordsMesh()){
      model->SwapMeshNodes();
   }
   
   //If the step didn't converge we're going to do a ramp up to the applied
   //velocity that we want. The assumption being made here is that our 1st time
   //step should be in the linear elastic regime. Therefore, we should be able
   //to go from our reduced solution to the desired solution. This has been noted
   //to be a problem when really increasing the mesh size.
   if(!newton_solver.GetConverged()){
      //We're going to reset our initial applied BCs to being 1/64 of the original
      mfem::out << "Solution didn't converge. Reducing initial condition to 1/4 original value\n";
      x = init_x;
      x *= 0.25;
      //We're going to keep track of how many cuts we need to make. Hopefully we
      //don't have to reduce it anymore then 3 times total.
      int i = 1;
      
      //We provide an initial guess for what our current coordinates will look like
      //based on what our last time steps solution was for our velocity field.
      if(!model->GetEndCoordsMesh()){
         model->SwapMeshNodes();
      }
      //The end nodes are updated before the 1st step of the solution here so we're good.
      newton_solver.Mult(zero, x);
      //Just gotta be safe incase something in the solver wasn't playing nice and didn't swap things
      //back to the current configuration...
      if(!model->GetEndCoordsMesh()){
         model->SwapMeshNodes();
      }
      
      if(!newton_solver.GetConverged()){
         //We're going to reset our initial applied BCs to being 1/16 of the original
         mfem::out << "Solution didn't converge. Reducing initial condition to 1/16 original value\n";
         x = init_x;
         x *= 0.0625;
         //We're going to keep track of how many cuts we need to make. Hopefully we
         //don't have to reduce it anymore then 3 times total.
         i++;
         
         //We provide an initial guess for what our current coordinates will look like
         //based on what our last time steps solution was for our velocity field.
         if(!model->GetEndCoordsMesh()){
            model->SwapMeshNodes();
         }
         //The end nodes are updated before the 1st step of the solution here so we're good.
         newton_solver.Mult(zero, x);
         //Just gotta be safe incase something in the solver wasn't playing nice and didn't swap things
         //back to the current configuration...
         if(!model->GetEndCoordsMesh()){
            model->SwapMeshNodes();
         }
         
         if(!newton_solver.GetConverged()){
            //We're going to reset our initial applied BCs to being 1/64 of the original
            mfem::out << "Solution didn't converge. Reducing initial condition to 1/64 original value\n";
            x = init_x;
            x *= 0.015625;
            //We're going to keep track of how many cuts we need to make. Hopefully we
            //don't have to reduce it anymore then 3 times total.
            i++;
            
            //We provide an initial guess for what our current coordinates will look like
            //based on what our last time steps solution was for our velocity field.
            if(!model->GetEndCoordsMesh()){
               model->SwapMeshNodes();
            }
            //The end nodes are updated before the 1st step of the solution here so we're good.
            newton_solver.Mult(zero, x);
            //Just gotta be safe incase something in the solver wasn't playing nice and didn't swap things
            //back to the current configuration...
            if(!model->GetEndCoordsMesh()){
               model->SwapMeshNodes();
            }
            
            MFEM_VERIFY(newton_solver.GetConverged(), "Newton Solver did not converge after 1/64 reduction of applied BCs.");
         }// end of 1/64 reduction case
      }// end of 1/16 reduction case
      
      //Here we're upscaling our previous converged solution to the next level.
      //The upscaling should be a good initial guess, since everything we're doing
      //is linear in this first step.
      //We then have the solution try and converge again with our better initial
      //guess of the solution.
      //It might be that this process only needs to occur once and we can directly
      //upscale from the lowest level to our top layer since we're dealing with
      //supposedly a linear elastic type problem here.
      for (int j = 0; j < i; j++) {
         mfem::out << "Upscaling previous solution by factor of 4\n";
         x *= 4.0;
         //We provide an initial guess for what our current coordinates will look like
         //based on what our last time steps solution was for our velocity field.
         if(!model->GetEndCoordsMesh()){
            model->SwapMeshNodes();
         }
         //The end nodes are updated before the 1st step of the solution here so we're good.
         newton_solver.Mult(zero, x);
         //Just gotta be safe incase something in the solver wasn't playing nice and didn't swap things
         //back to the current configuration...
         if(!model->GetEndCoordsMesh()){
            model->SwapMeshNodes();
         }
         
         //Once the system has finished solving, our current coordinates configuration are based on what our
         //converged velocity field ended up being equal to.
         //If the update fails we want to exit.
         MFEM_VERIFY(newton_solver.GetConverged(), "Newton Solver did not converge.");
      }// end of upscaling process
   }// end of 1/4 reduction case
   
   //Reset our max number of iterations to our original desired value.
   newton_solver.SetMaxIter(newton_iter);
}

// compute: y = H(x,p)
void NonlinearMechOperator::Mult(const Vector &k, Vector &y) const
{
   //Wanted to put this in the mechanics_solver.cpp file, but I would have needed to update
   //Solver class to use the NonlinearMechOperator instead of Operator class.
   //We now update our end coordinates based on the solved for velocity.
   UpdateEndCoords(k);
   // Apply the nonlinear form
   if(umat_used){
     
     //I really don't like this. It feels so hacky and
     //potentially dangerous to have these methods just
     //lying around.
     ParGridFunction* end_crds = model->GetEndCoords();
     ParGridFunction* beg_crds = model->GetBegCoords();
     ParMesh* pmesh = model->GetPMesh();
     Vector temp;
     temp.SetSize(k.Size());
     end_crds->GetTrueDofs(temp);
     //Creating a new vector that's going to be used for our
     //UMAT custorm Hform->Mult
     const Vector crd(temp.GetData(), temp.Size());
     //The Mult expects our mesh to have the beg. time step
     //nodes
     //I should probably do the whole
      /*if(!model->GetEndCoordsMesh()){
         model->SwapMeshNodes();
      }*/
     //thing here as well...
     model->SwapMeshNodes();
     Hform->Mult(crd, y, pmesh, end_crds, beg_crds, k);
     //We need to swap back to the current time step nodes
     //here
     model->SwapMeshNodes();
   }else{
     //Without the umat things become nice and simple
     Hform->Mult(k, y);
   }
}
//Update the end coords used in our model
void NonlinearMechOperator::UpdateEndCoords(const Vector& vel) const {
   model->UpdateEndCoords(vel);
}

// Compute the Jacobian from the nonlinear form
Operator &NonlinearMechOperator::GetGradient(const Vector &x) const
{
   Jacobian = &Hform->GetGradient(x);
   return *Jacobian;
}

void NonlinearMechOperator::ComputeVolAvgTensor(const ParFiniteElementSpace* fes,
                                                const QuadratureFunction* qf,
                                                Vector& tensor, int size){

   const FiniteElement *fe;
   const IntegrationRule *ir;
   double* qf_data = qf->GetData();
   int qf_offset = qf->GetVDim(); // offset at each integration point
   QuadratureSpace* qspace = qf->GetSpace();
   
   double el_vol = 0.0;
   double temp_wts = 0.0;
   double incr = 0.0;

   int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   
   // loop over elements
   for (int i = 0; i < fes->GetNE(); ++i)
   {
      // get element transformation for the ith element
      ElementTransformation* Ttr = fes->GetElementTransformation(i);
      fe = fes->GetFE(i);
      ir = &(qspace->GetElementIntRule(i));
      int elem_offset = qf_offset * ir->GetNPoints();
      // loop over element quadrature points
      for (int j = 0; j < ir->GetNPoints(); ++j)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         Ttr->SetIntPoint(&ip);
         //Here we're setting the integration for the average value
         temp_wts = ip.weight * Ttr->Weight();
         //This tells us the element volume
         el_vol += temp_wts;
         incr += 1.0;
         int k = 0;
         for (int m = 0; m < size; ++m)
         {
            tensor[m] += temp_wts * qf_data[i * elem_offset + j * qf_offset + k];
            ++k;
         }
      }
   }

   double data[size];

   for(int i = 0; i < size; i++){
     data[i] = tensor[i];
   }

   MPI_Allreduce(&data, tensor.GetData(), size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   
   double temp = el_vol;

   //Here we find what el_vol should be equal to
   MPI_Allreduce(&temp, &el_vol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

   //We meed to multiple by 1/V by our tensor values to get the appropriate
   //average value for the tensor in the end.
   double inv_vol = 1.0/el_vol;

   for (int m = 0; m < size; m++) {
      tensor[m] *= inv_vol;
   }
   
}

void NonlinearMechOperator::UpdateModel(const Vector &x)
{
   const ParFiniteElementSpace *fes = GetFESpace();
   const FiniteElement *fe;
   const IntegrationRule *ir;

   if(umat_used){
     
     //I really don't like this. It feels so hacky and
     //potentially dangerous to have these methods just
     //lying around.
     ParGridFunction* end_crds = model->GetEndCoords();
     ParGridFunction* beg_crds = model->GetBegCoords();
     ParMesh* pmesh = model->GetPMesh();
     Vector temp;
     temp.SetSize(x.Size());
     end_crds->GetTrueDofs(temp);
     //Creating a new vector that's going to be used for our
     //UMAT custorm Hform->Mult
     const Vector crd(temp.GetData(), temp.Size());
     //As pointed out earlier I should probably check here again that we're
     //doing what we expect here aka swap the nodes to beg time step before
     //swapping back to the end time step coords
     model->SwapMeshNodes();
     model->UpdateModelVars(&fe_space, crd);
     model->SwapMeshNodes();
   }
   else{
     model->UpdateModelVars(&fe_space, x);
   }

   //Everything is the same here no matter if we're using a UMAT
   //or not...
   //update state variables on a ExaModel
   for (int i = 0; i < fes->GetNE(); ++i)
   {
      fe = fes->GetFE(i);
      ir = &(IntRules.Get(fe->GetGeomType(), 2*fe->GetOrder() + 1));

      // loop over element quadrature points
      for (int j = 0; j < ir->GetNPoints(); ++j)
      {
         // update the beginning step stress 
         model->UpdateStress(i, j);

         // compute von Mises stress
         model->ComputeVonMises(i, j);

         // update the beginning step state variables
         if (model->numStateVars > 0)
         {
           model->UpdateStateVars(i, j);
         }
      }
   } 
   

   //Here we're getting the average stress value
   Vector stress;
   int size = 6;
   
   stress.SetSize(size);
   
   stress = 0.0;
   
   QuadratureVectorFunctionCoefficient* qstress = model->GetStress0();
   
   const QuadratureFunction* qf = qstress->GetQuadFunction();
   
   ComputeVolAvgTensor(fes, qf, stress, size);

   cout.setf(ios::fixed);
   cout.setf(ios::showpoint);
   cout.precision(8);
   
   int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   //Now we're going to save off the average stress tensor to a file
   if(my_id == 0){
     std::ofstream file;

     file.open("avg_stress.txt", std::ios_base::app);

     stress.Print(file, 6);
   }
   
   qstress = NULL;
   qf = NULL;
   
   //Here we're computing the average deformation gradient
   Vector defgrad;
   size = 9;

   defgrad.SetSize(size);

   defgrad = 0.0;

   QuadratureVectorFunctionCoefficient* qdefgrad = model->GetDefGrad0();

   const QuadratureFunction* qf1 = qdefgrad->GetQuadFunction();

   ComputeVolAvgTensor(fes, qf1, defgrad, size);

   //We're now saving the average def grad off to a file
   if(my_id == 0){
     std::ofstream file;

     file.open("avg_dgrad.txt", std::ios_base::app);

     defgrad.Print(file, 9);
   }
   
   qf1 = NULL;
   qdefgrad = NULL;
   
   fes = NULL;
   fe = NULL;
   ir = NULL;
   
}

//This is probably wrong and we need to make this more in line with what
//the ProjectVonMisesStress is doing
void NonlinearMechOperator::ProjectModelStress(ParGridFunction &s)
{
   QuadratureVectorFunctionCoefficient *stress;
   stress = model->GetStress0();
   s.ProjectCoefficient(*stress);

   stress = NULL;
   
   return;
}

void NonlinearMechOperator::ProjectVonMisesStress(ParGridFunction &vm)
{
   QuadratureFunctionCoefficient *vonMisesStress;
   vonMisesStress = model->GetVonMises();
   vm.ProjectDiscCoefficient(*vonMisesStress, mfem::GridFunction::ARITHMETIC);

   vonMisesStress = NULL;
   
   return;
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

void NonlinearMechOperator::SetModelDebugFlg(const bool dbg)
{
   model->debug = dbg;
}

void NonlinearMechOperator::DebugPrintModelVars(int procID, double time)
{
   // print material properties vector on the model
   Vector *props = model->GetMatProps();
   ostringstream props_name;
   props_name << "props." << setfill('0') << setw(6) << procID << "_" << time;
   ofstream props_ofs(props_name.str().c_str());
   props_ofs.precision(8);
   props->Print(props_ofs);

   // print the beginning step material state variables quadrature function
   QuadratureVectorFunctionCoefficient *mv0 = model->GetMatVars0();
   ostringstream mv_name;
   mv_name << "matVars." << setfill('0') << setw(6) << procID << "_" << time;
   ofstream mv_ofs(mv_name.str().c_str());
   mv_ofs.precision(8);

   QuadratureFunction *matVars0 = mv0->GetQuadFunction();
   matVars0->Print(mv_ofs);

   matVars0 = NULL;
   props = NULL;
   
   return;
  
}
//A generic test function that we can add whatever unit tests to and then have them be tested
void NonlinearMechOperator::testFuncs(const Vector &x0, ParFiniteElementSpace *fes){
   model->test_def_grad_func(fes, x0);
}

NonlinearMechOperator::~NonlinearMechOperator()
{
   delete J_solver;
   if (J_prec != NULL) {
      delete J_prec;
   }
   delete model;
   delete Hform;
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
   nodes = NULL;
}

void ReferenceConfiguration(const Vector &x, Vector &y)
{
   // set the reference, stress free, configuration
   y = x;
}


void InitialDeformation(const Vector &x, double t,  Vector &y)
{
   // this performs a velocity projection 
   
   // Note: x comes in initialized to 0.0 on the first time step, 
   // otherwise it is coming in as incremental nodal velocity, which is 
   // the previous step's incremental nodal displacement solution 
   // divided by the previous time step

   // get the time step off the boundary condition manager
   // for the first BC, which there has to be at least one of
   BCManager & bcManager = BCManager::getInstance();
   BCData & bc_data = bcManager.GetBCInstance(1);

   double dt = bc_data.dt;

   // velocity projection is the last delta_x solution (x_cur) times 
   // the current timestep.
   Vector temp_x(x);
   temp_x *= dt;
   y = temp_x;
}

void Velocity(const Vector &x, double t, Vector &y)
{
   BCManager & bcManager = BCManager::getInstance();
   BCData & bc_data = bcManager.GetBCInstance(1);
 
   double dt = bc_data.dt;

   // compute the grid velocity by dividing by dt
   Vector temp_x = x;
   temp_x /= dt;
   y = temp_x; 
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
                       bool g_q, bool g_custom, int ngrains, int numProps,
                       int numStateVars)
{
   bool err = true;

   if (hyperelastic && cp)
   {

      cerr << "Hyperelastic and cp can't both be true. Choose material model." << '\n';
   }

   // only perform checks, don't set anything here
   if (cp && !g_euler && !g_q && !g_custom)
   {
      cerr << "\nMust specify grain data type for use with cp input arg." << '\n';
      err = false;
   }

   else if (cp && g_euler && g_q)
   {
      cerr << "\nCannot specify euler and quaternion grain data input args." << '\n';
      err = false;
   }

   else if (cp && g_euler && g_custom)
   {
      cerr << "\nCannot specify euler and uniform grain data input args." << '\n';
      err = false;
   }

   else if (cp && g_q && g_custom)
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
                     ParMesh *pmesh, int grainSize, int grainIntoStateVarOffset, 
                     int stateVarSize, QuadratureFunction* qf)
{
   // put element grain orientation data on the quadrature points. 
   const FiniteElement *fe;
   const IntegrationRule *ir;
   double* qf_data = qf->GetData();
   int qf_offset = qf->GetVDim(); // offset = grainSize + stateVarSize
   QuadratureSpace* qspace = qf->GetSpace();
   
   // check to make sure the sum of the input sizes matches the offset of 
   // the input quadrature function
   if (qf_offset != (grainSize + stateVarSize))
   {
      cerr << "\nsetStateVarData: Input state variable and grain sizes do not "
                 "match quadrature function initialization." << '\n';
   }

   // get the data for the material state variables and grain orientations for 
   // nonzero grainSize(s), which implies a crystal plasticity calculation
   double* grain_data = NULL;
   if (grainSize > 0) grain_data = orient->GetData();

   double* sVars_data = sVars->GetData();
   int elem_atr;

   int offset1;
   int offset2;
   if (grainIntoStateVarOffset < 0) // put grain data at end
   {
      // print warning to screen since this case could arise from a user 
      // simply not setting this parameter
      std::cout << "warning::setStateVarData grain data placed at end of"
                << " state variable array. Check grain_statevar_offset input arg." << "\n";

      offset1 = stateVarSize - 1;
      offset2 = qf_offset; 
   }
   else if (grainIntoStateVarOffset == 0) // put grain data at beginning
   {
      offset1 = -1; 
      offset2 = grainSize;
   }
   else // put grain data somewhere in the middle
   {
      offset1 = grainIntoStateVarOffset - 1;
      offset2 = grainIntoStateVarOffset + grainSize; 
   }

   // loop over elements
   for (int i = 0; i < fes->GetNE(); ++i)
   {
      fe = fes->GetFE(i);
      ir = &(qspace->GetElementIntRule(i));
//      ir = &(IntRules.Get(fe->GetGeomType(), 2*fe->GetOrder() + 3));

      // full history variable offset including grain data
      int elem_offset = qf_offset * ir->GetNPoints();

      // get the element attribute. Note this assumes that there is an element attribute 
      // for all elements in the mesh corresponding to the grain id to which the element 
      // belongs.
      elem_atr = fes->GetAttribute(i) - 1;
      // loop over quadrature points
      for (int j = 0; j < ir->GetNPoints(); ++j)
      {
         // loop over quadrature point material state variable data
         double varData;
         int igrain = 0;
         int istateVar = 0;
         for (int k = 0; k < qf_offset; ++k) 
         {
            // index into either the grain data or the material state variable 
            // data depending on the setting of offset1 and offset2. This handles
            // tacking on the grain data at the beginning of the total material 
            // state variable quadarture function, the end, or somewhere in the 
            // middle, which is dictated by grainIntoStateVarOffset, which is 
            // ultimately a program input. If grainSize == 0 for non-crystal 
            // plasticity problems, we never get into the if-block that gets 
            // data from the grain_data. In fact, grain_data should be a null 
            // pointer
            if (k > offset1 && k < offset2)
            {
               varData = grain_data[grainSize * (elem_atr) + igrain];
               ++igrain;
            }
            else 
            {
               varData = sVars_data[istateVar];
               ++istateVar;
            }

            qf_data[(elem_offset * i) + qf_offset * j + k] = varData;

         } // end loop over material state variables
      } // end loop over quadrature points
   } // end loop over elements

   //Set the pointers to null after using them to hopefully stop any weirdness from happening
   fe = NULL;
   qf_data = NULL;
   ir = NULL;
   qspace = NULL;
   grain_data = NULL;
   sVars_data = NULL;
   
}

void initQuadFunc(QuadratureFunction *qf, double val, ParFiniteElementSpace *fes)
{
//   const FiniteElement *fe;
   const IntegrationRule *ir;
   double* qf_data = qf->GetData();
   int vdim = qf->GetVDim();
   int counter = 0;
   //QuadratureSpace* qspace = qf->GetSpace();
   
   //The below should be exactly the same as what
   //the other for loop is trying to accomplish
   for (int i = 0; i < qf->Size(); ++i){
     qf_data[i] = val;
   }

   qf_data = NULL;
}

void initQuadFuncTensorIdentity(QuadratureFunction *qf, ParFiniteElementSpace *fes){
   
   const FiniteElement *fe;
   const IntegrationRule *ir;
   double* qf_data = qf->GetData();
   int qf_offset = qf->GetVDim(); // offset at each integration point
   QuadratureSpace* qspace = qf->GetSpace();
   
   // loop over elements
   for (int i = 0; i < fes->GetNE(); ++i){
      // get element transformation for the ith element
      ElementTransformation* Ttr = fes->GetElementTransformation(i);
      fe = fes->GetFE(i);
      ir = &(qspace->GetElementIntRule(i));
      int elem_offset = qf_offset * ir->GetNPoints();
      //Hard coded this for now for a 3x3 matrix
      //Fix later if we update
      for (int j = 0; j < ir->GetNPoints(); ++j){
         qf_data[i * elem_offset + j * qf_offset] = 1.0;
         qf_data[i * elem_offset + j * qf_offset + 1] = 0.0;
         qf_data[i * elem_offset + j * qf_offset + 2] =	0.0;
         qf_data[i * elem_offset + j * qf_offset + 3] =	0.0;
         qf_data[i * elem_offset + j * qf_offset + 4] = 1.0;
         qf_data[i * elem_offset + j * qf_offset + 5] =	0.0;
         qf_data[i * elem_offset + j * qf_offset + 6] =	0.0;
         qf_data[i * elem_offset + j * qf_offset + 7] =	0.0;
         qf_data[i * elem_offset + j * qf_offset + 8] = 1.0;
      }
   }
   
   qf_data = NULL;
}

//Routine to test the deformation gradient to make sure we're getting out the right values.
//This applies the following displacement field to the nodal values:
//u_vec = (2x + 3y + 4z)i + (4x + 2y + 3z)j + (3x + 4y + 2z)k
void test_deformation_field_set(ParGridFunction *gf, Vector *vec, ParFiniteElementSpace *fes)
{
   
   const IntegrationRule *ir;
   //HypreParVector* temp = gf->GetTrueDofs();
   //Vector* temp2 = temp->GlobalVector();
   double* temp_vals = gf->GetData();
   double* vals = vec->GetData();

   int dim = gf->Size()/3;
   int dim2 = vec->Size()/3;
   int  myid;

   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   
   printf("gf data size: %d vec data size %d\n", dim, dim2);
   
   for (int i = 0; i < dim; ++i)
   {
      int i1 = i * 3;
      double x1 = temp_vals[i];
      double x2 = temp_vals[i + dim];
      double x3 = temp_vals[i + 2 * dim];
      
      //vals[i] = x1;// + (2 * x1 + 3 * x2 + 4 * x3);
      //vals[i + dim] = x2;// + (4 * x1 + 2 * x2 + 3 * x3);
      //vals[i + 2 * dim] = x3;// + (3 * x1 + 4 * x2 + 2 * x3);
      printf("vertex_num: %d my_id: %d\t x: %f;\t y: %f;\t z: %f\n ",i, myid, x1, x2, x3);
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

void setBdrConditions(Mesh *mesh)
{
   // modify MFEM auto cuboidal hex mesh generation boundary 
   // attributes to correspond to correct ExaConstit boundary conditions. 
   // Look at ../../mesh/mesh.cpp Make3D() to see how boundary attributes 
   // are set and modify according to ExaConstit convention

   // loop over boundary elements
   for (int i=0; i<mesh->GetNBE(); ++i)
   {
      int bdrAttr = mesh->GetBdrAttribute(i);

      switch (bdrAttr)
      {
         // note, srw wrote SetBdrAttribute() in ../../mesh/mesh.hpp
         case 1 : mesh->SetBdrAttribute(i, 1); // bottom
                  break;
         case 2 : mesh->SetBdrAttribute(i, 3); // front
                  break;
         case 3 : mesh->SetBdrAttribute(i, 5); // right
                  break;
         case 4 : mesh->SetBdrAttribute(i, 6); // back
                  break;
         case 5 : mesh->SetBdrAttribute(i, 2); // left
                  break;
         case 6 : mesh->SetBdrAttribute(i, 4); // top
                  break;
      }
   }
   return;
}

void reorderMeshElements(Mesh *mesh, const int nx) 
{
   // reorder mesh elements depending on how the 
   // computational cells are ordered in the grain map file.

   // Right now, the element ordering in the grain map file 
   // starts at (0,0,0) and increments in z, y, then x coordinate 
   // directions. 

   // MFEM Make3D(.) mesh gen increments in x, y, then z.

  Array<int> order(nx*nx*nx);
   int id = 0;
   int k = 0;
   for (int z = 0; z < nx; ++z)
   {
      for (int y = 0; y < nx; ++y)
      {
         for (int x = 0; x < nx; ++x)
         {
            id = (nx * nx) * x + nx * y + z;
            order[k] = id;
            ++k;
         }
      }
   }

   mesh->ReorderElements(order, true);
   
   return;
}

void setElementGrainIDs(Mesh *mesh, const Vector grainMap, int ncols, int offset)
{
   // after a call to reorderMeshElements, the elements in the serial 
   // MFEM mesh should be ordered the same as the input grainMap 
   // vector. Set the element attribute to the grain id. This vector 
   // has stride of 4 with the id in the 3rd position indexing from 0
  
   double* data = grainMap.GetData();

   // loop over elements 
   for (int i=0; i<mesh->GetNE(); ++i)
   {
      mesh->SetAttribute(i, data[ncols*i+offset]);
   }
   return;
}

