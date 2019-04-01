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
//   # Description:
//         The purpose of this code app is to determine bulk constitutive
//         properties for 3D printed metal alloys. This is a nonlinear
//         quasi-static, implicit solid mechanics code built on the MFEM library
//         based on an updated Lagrangian formulation (velocity based).
//
//         Currently, only Dirichlet boundary conditions (homogeneous and
//         inhomogeneous by dof component) have been implemented. Neumann
//         (traction) boundary conditions and a body force are not implemented.
//         A new ExaModel class allows one to implement arbitrary constitutive
//         models. The code currently successfully allows for various UMATs to
//         be interfaced within the code framework. Development work is currently
//         focused on allowing for the mechanical models to run on GPGPUs.
//
//         The code supports either constant time steps or user supplied delta
//         time steps. Boundary conditions are supplied for the velocity field
//         applied on a surface. It supports a number of different preconditioned
//         Krylov iterative solvers (PCG, GMRES, MINRES) for either symmetric or
//         nonsymmetric positive-definite systems.
//
//
//         ## Remark:
//            See the included options.toml to see all of the various different
//            options that are allowable in this code and their default values.
//
//            A TOML parser has been included within this directory, since it
//            has an MIT license. The repository for it can be found at:
//            https://github.com/skystrife/cpptoml .
//
//            Example UMATs maybe obtained from:
//            https://web.njit.edu/~sac3/Software.html . We have not included
//            them due to a question of licensing. The ones that have been run
//            and are known to work are the linear elasticity model and the
//            neo-Hookean material. Although, we might be able to provide an
//            example interface so users can base their interface/build scripts
//            off of what's known to work.
//
//            Note: the grain.txt, props.txt and state.txt files are expected
//            inputs for CP problems, specifically ones that use the Abaqus UMAT
//            interface class under the ExaModel.
//
//   #  Future Implemenations Notes:
//
//      * Visco-plasticity constitutive model
//      * GPGPU material models
//      * A more in-depth README that better covers the different options available.
//      * debug ability to read different mesh formats
//
//
//***********************************************************************
#include "mfem.hpp"
#include "mechanics_coefficient.hpp"
#include "mechanics_integrators.hpp"
#include "mechanics_solver.hpp"
#include "mechanics_operator.hpp"
#include "BCData.hpp"
#include "BCManager.hpp"
#include "option_parser.hpp"
#include <string>
#include <sstream>

using namespace std;
using namespace mfem;

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
bool checkMaterialArgs(bool umat, bool cp, int ngrains, int numProps,
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
   //Here we're going to measure the times of each solve.
   //It'll give us a good idea of strong and weak scaling in
   //comparison to the global value of things.
   //It'll make it easier to point out where some scaling issues might
   //be occurring.
   std::vector<double> times;
   double t1, t2;
   // print the version of the code being run
   if(myid == 0) printf("MFEM Version: %d \n", GetVersion());

   //All of our options are parsed in this file by default
   const char *toml_file = "options.toml";

   //We're going to use the below to allow us to easily swap between different option files
   OptionsParser args(argc, argv);
   args.AddOption(&toml_file, "-opt", "--option", "Option file to use.");
   args.Parse();
   if (!args.Good()){
      if (myid == 0){                                                                                                             
         args.PrintUsage(cout);                                                                                     
      }
      MPI_Finalize();
      return 1;
   }

   ExaOptions toml_opt(toml_file);
   toml_opt.parse_options(myid);

   //Check to see if a custom dt file was used
   //if so read that in and if not set the nsteps that we're going to use
   if(toml_opt.dt_cust){
     if(myid == 0) printf("Reading in custom dt file. \n");
      ifstream idt(toml_opt.dt_file.c_str());
      if (!idt && myid == 0)
      {
         cerr << "\nCannot open grain map file: " << toml_opt.grain_map << '\n' << endl;
      }     
      //Now we're calculating the final time                  
      toml_opt.cust_dt.Load(idt, toml_opt.nsteps);
      toml_opt.t_final = 0.0;
      for(int i = 0; i < toml_opt.nsteps; i++){
         toml_opt.t_final += toml_opt.cust_dt[i]; 
      }

      idt.close();
   }else{
     toml_opt.nsteps = ceil(toml_opt.t_final/toml_opt.dt);
     if(myid==0) printf("number of steps %d \n", toml_opt.nsteps);
   }
   
   times.reserve(toml_opt.nsteps);
   
   // Check material model argument input parameters for valid combinations
   if(myid == 0) printf("after input before checkMaterialArgs. \n");
   bool err = checkMaterialArgs(toml_opt.umat, toml_opt.cp,
              toml_opt.ngrains, toml_opt.nProps, toml_opt.numStateVars);
   if (!err && myid == 0) 
   {
      cerr << "\nInconsistent material input; check args" << '\n';
   }

   // Open the mesh
   if(myid == 0) printf("before reading the mesh. \n");
   Mesh *mesh;
   Vector g_map;
   if (toml_opt.mesh_type == MeshType::CUBIT) 
   {
      //named_ifgzstream imesh(mesh_file);
      mesh = new Mesh(toml_opt.mesh_file.c_str(), 1, 1);
   }
   else if (toml_opt.mesh_type == MeshType::AUTO)
   {
      if (toml_opt.nx == 0 || toml_opt.mx == 0)
      {
         cerr << "\nMust input mesh geometry/discretization for hex_mesh_gen" << '\n';
      }

      // use constructor to generate a 3D cuboidal mesh with 8 node hexes
      mesh = new Mesh(toml_opt.nx, toml_opt.nx, toml_opt.nx, Element::HEXAHEDRON, 0, toml_opt.mx, toml_opt.mx, toml_opt.mx); 
   }
   else // read in mesh file
   {
      if(myid == 0) printf("opening mesh file \n");

      ifstream imesh(toml_opt.mesh_file.c_str());
      if(myid == 0) printf("after declaring imesh \n");
      if (!imesh)
      {
         if (myid == 0)
         {
            cerr << "\nCan not open mesh file: " << toml_opt.mesh_file << '\n' << endl;
         }
         MPI_Finalize();
         return 2;
      }
  
      if(myid == 0) printf("before declaring new mesh \n");
      mesh = new Mesh(imesh, 1, 1, true);
      if(myid == 0) printf("after declaring new mesh \n");
      imesh.close();
      //If we're doing xtal plasticity stuff read in the grain map file
      if(toml_opt.cp){
         ifstream igmap(toml_opt.grain_map.c_str());
         if (!igmap && myid == 0)
         {
           cerr << "\nCannot open grain map file: " << toml_opt.grain_map << '\n' << endl;
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
   if (toml_opt.mesh_type == MeshType::AUTO)
   {
      if(myid == 0) printf("using mfem hex mesh generator \n");

      ifstream igmap(toml_opt.grain_map.c_str());
      if (!igmap && myid == 0)
      {
         cerr << "\nCannot open grain map file: " << toml_opt.grain_map << '\n' << endl;
      }
      
      int gmapSize = mesh->GetNE();
      g_map.Load(igmap, gmapSize);
      igmap.close();

      // reorder elements to conform to ordering convention in grain map file
      reorderMeshElements(mesh, toml_opt.nx);

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
   for (int lev = 0; lev < toml_opt.ser_ref_levels; lev++)
   {
         mesh->UniformRefinement();
   }

   pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   for (int lev = 0; lev < toml_opt.par_ref_levels; lev++)
   {
         pmesh->UniformRefinement();
   }

   delete mesh;

   if(myid == 0) printf("after mesh section. \n");

   int dim = pmesh->Dimension();

   // Define the finite element spaces for displacement field
   FiniteElementCollection *fe_coll = NULL;
   fe_coll = new  H1_FECollection(toml_opt.order, dim);
   ParFiniteElementSpace fe_space(pmesh, fe_coll, dim);

   int order_0 = 0;
   
   //Here we're setting up a discontinuous so that we'll use later to interpolate
   //our quadrature functions from
   L2_FECollection l2_fec(order_0, dim);
   ParFiniteElementSpace l2_fes(pmesh, &l2_fec);
   ParGridFunction vonMises(&l2_fes);
   ParGridFunction hydroStress(&l2_fes);
   
   HYPRE_Int glob_size = fe_space.GlobalTrueVSize();

   // Print the mesh statistics
   if (myid == 0)
   {
      std::cout << "***********************************************************\n";
      std::cout << "dim(u) = " << glob_size << "\n";
      std::cout << "***********************************************************\n";
   }

   // determine the type of grain input for crystal plasticity problems
   int ori_offset = 0; // note: numMatVars >= 1, no null state vars by construction
   if(toml_opt.cp){
      if (toml_opt.ori_type == OriType::EULER) 
      {
         ori_offset = 3; 
      }
      else if (toml_opt.ori_type == OriType::QUAT) 
      {
         ori_offset = 4;
      }
      else if (toml_opt.ori_type == OriType::CUSTOM) 
      {
         if (toml_opt.grain_custom_stride == 0)
         { 
            cerr << "\nMust specify a grain stride for grain_custom input" << '\n';
         }
         ori_offset = toml_opt.grain_custom_stride;
      }
   }

   // set the offset for the matVars quadrature function. This is the number of 
   // state variables (stored at each integration point) and then the grain offset, 
   // which is the number of variables defining the grain data stored at each 
   // integration point. In general, these may come in as different data sets, 
   // even though they will be stored in a single material state variable 
   // quadrature function.
   int matVarsOffset = toml_opt.numStateVars + ori_offset;

   // Define a quadrature space and material history variable QuadratureFunction.
   int intOrder = 2 * toml_opt.order + 1;
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
      ifstream iprops(toml_opt.props_file.c_str());
      if (!iprops && myid == 0)
      {
         cerr << "\nCannot open material properties file: " << toml_opt.props_file << '\n' << endl;
      }

      // load material properties
      matProps.Load(iprops, toml_opt.nProps);
      iprops.close();
      
      if(myid == 0) printf("after loading matProps. \n");
      
      // read in state variables file
      ifstream istateVars(toml_opt.state_file.c_str());
      if (!istateVars && myid == 0)
      {
         cerr << "\nCannot open state variables file: " << toml_opt.state_file << '\n' << endl;
      }

      // load state variables
      stateVars.Load(istateVars, toml_opt.numStateVars);
      istateVars.close();
      if(myid == 0) printf("after loading stateVars. \n");

      // if using a crystal plasticity model then get grain orientation data
      // declare a vector to hold the grain orientation input data. This data is per grain 
      // with a stride set previously as grain_offset
      Vector g_orient;
      if(myid == 0) printf("before loading g_orient. \n");
      if (toml_opt.cp)
      {
         // set the grain orientation vector from the input grain file
         ifstream igrain(toml_opt.ori_file.c_str()); 
         if (!igrain && myid == 0)
         {
            cerr << "\nCannot open orientation file: " << toml_opt.ori_file << '\n' << endl;
         }
         // load separate grain file
         int gsize = ori_offset * toml_opt.ngrains;
         g_orient.Load(igrain, gsize);
         igrain.close();
         if(myid == 0) printf("after loading g_orient. \n");

      } // end if (cp)
     
      // set the state var data on the quadrature function
      if(myid == 0) printf("before setStateVarData. \n");
      setStateVarData(&stateVars, &g_orient, &fe_space, pmesh, ori_offset, 
                      toml_opt.grain_statevar_offset, toml_opt.numStateVars, &matVars0);
      if(myid == 0) printf("after setStateVarData. \n");
      
   } // end read of mat props, state vars and grains

   // Declare quadrature functions to store a vector representation of the 
   // Cauchy stress, in Voigt notation (s_11, s_22, s_33, s_23, s_13, s_12), for 
   // the beginning of the step and the end of the step.
   int stressOffset = 6;
   QuadratureFunction sigma0(&qspace, stressOffset);
   QuadratureFunction sigma1(&qspace, stressOffset);
   QuadratureFunction q_vonMises(&qspace, 1);
   initQuadFunc(&sigma0, 0.0, &fe_space);
   initQuadFunc(&sigma1, 0.0, &fe_space);
   initQuadFunc(&q_vonMises, 0.0, &fe_space);

   // The tangent stiffness of the Cauchy stress will
   // actually be the real material tangent stiffness (4th order tensor) and have 
   // 36 components due to symmetry.
   int matGradOffset = 36;
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
   if (toml_opt.ess_disp.Size() != 3*toml_opt.ess_id.Size()) {
      cerr << "\nMust specify three Dirichlet components per essential boundary attribute" << '\n' << endl;
   }

   int numDirBCs = 0;
   for (int i=0; i<toml_opt.ess_id.Size(); ++i) {
      // set the boundary condition id based on the attribute id
      int bcID = toml_opt.ess_id[i];

      // instantiate a boundary condition manager instance and 
      // create a BCData object
      BCManager & bcManager = BCManager::getInstance();
      BCData & bc = bcManager.CreateBCs( bcID );

      // set the displacement component values
      bc.essDisp[0] = toml_opt.ess_disp[3*i];
      bc.essDisp[1] = toml_opt.ess_disp[3*i+1];
      bc.essDisp[2] = toml_opt.ess_disp[3*i+2];
      bc.compID = toml_opt.ess_comp[i];

      // set the final simulation time 
      bc.tf = toml_opt.t_final;

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
                              toml_opt, matVars0, 
                              matVars1, sigma0, sigma1, matGrd,
                              kinVars0, q_vonMises, x_beg, x_cur, pmesh,
			      matProps, matVarsOffset);
   if(myid == 0) printf("after NonlinearMechOperator constructor. \n");

   oper.SetModelDebugFlg(toml_opt.grad_debug);

   if(myid == 0) printf("after SetModelDebugFlg \n");
   
   // get the essential true dof list. This may not be used.
   const Array<int> ess_tdof_list = oper.GetEssTDofList();
   
   // declare incremental nodal displacement solution vector
   Vector v_sol(fe_space.TrueVSize()); // this sizing is correct
   v_sol = 0.0;

   // Save data for VisIt visualization.
   // The below is used to take advantage of mfem's custom Visit plugin
   // It could also allow for restart files later on.
   // If we have large simulations although the current method of printing everything
   // as text will cause issues. The data should really be saved in some binary format.
   // If you don't then you'll often find that the printed data lags behind where
   // the simulation is currently at. This really becomes noticiable if you have
   // a lot of data that you want to output for the user. It might be nice if this
   // was either a netcdf or hdf5 type format instead.
   VisItDataCollection visit_dc(toml_opt.basename, pmesh);
   if (toml_opt.visit)
   {
     visit_dc.RegisterField("Displacement",  &x_diff);
//     visit_dc.RegisterQField("Stress", &sigma0);
     visit_dc.RegisterField("Velocity", &v_cur);
     //visit_dc.RegisterQField("State Variables", &matVars0);
     //visit_dc.RegisterQField("DefGrad", &kinVars0);
     visit_dc.RegisterField("Von Mises Stress", &vonMises);
     visit_dc.RegisterField("Hydrostatic Stress", &hydroStress);
      
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
   setBCTimeStep(toml_opt.dt, numDirBCs);

   double dt_real;
   
   for (int ti = 1; ti <= toml_opt.nsteps; ti++)
   {

      if(myid == 0) printf("inside timestep loop %d \n", ti);
      //Get out our current delta time step
      if(toml_opt.dt_cust){
         dt_real = toml_opt.cust_dt[ti - 1];
      }else{
         dt_real = min(toml_opt.dt, toml_opt.t_final - t);
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
      t1 = MPI_Wtime();
      if(ti == 1){
         oper.SolveInit(v_sol);
      }else{
         oper.Solve(v_sol);
      }
      t2 = MPI_Wtime();
      times[ti - 1] = t2 - t1;
      
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
      
      last_step = (t >= toml_opt.t_final - 1e-8 * dt_real);

      if (last_step || (ti % toml_opt.vis_steps) == 0)
      {
         if (myid == 0)
         {
            cout << "step " << ti << ", t = " << t << endl;
         }
          // mesh and stress output. Consider moving this to a separate routine
         //We might not want to update the vonMises stuff
         oper.ProjectVonMisesStress(vonMises);
         oper.ProjectHydroStress(hydroStress);
         if (toml_opt.visit)
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
   
   double sim_time = end - start;
   double avg_sim_time;
   
   MPI_Allreduce(&sim_time, &avg_sim_time, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   int world_size;
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);
      
   {
      std::ostringstream oss;
      
      oss << "./time/time_solve." << myid << ".txt";
      std::string file_name = oss.str();
      std::ofstream file;
      file.open(file_name, std::ios::out | std::ios::app);
      
      for(int i = 0; i < toml_opt.nsteps; i++){
         std::ostringstream strs;
         strs << setprecision(8) << times[i] << "\n";
         std::string str = strs.str();
         file << str;
      }
      file.close();
   }
   
   
   if(myid == 0) printf("The process took %lf seconds to run\n", (avg_sim_time / world_size));

   MPI_Finalize();

   return 0;
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

bool checkMaterialArgs(bool umat, bool cp, int ngrains, int numProps,
                       int numStateVars)
{
   bool err = true;

   if (cp && (ngrains < 1))
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
   const IntegrationRule *ir;
   double* qf_data = qf->GetData();
   int qf_offset = qf->GetVDim(); // offset = grainSize + stateVarSize
   QuadratureSpace* qspace = qf->GetSpace();
   
   int myid;
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // check to make sure the sum of the input sizes matches the offset of 
   // the input quadrature function
   if (qf_offset != (grainSize + stateVarSize))
   {
      if(myid == 0) cerr << "\nsetStateVarData: Input state variable and grain sizes do not "
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
      if(myid == 0) std::cout << "warning::setStateVarData grain data placed at end of"
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
      ir = &(qspace->GetElementIntRule(i));

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
   
}

void initQuadFunc(QuadratureFunction *qf, double val, ParFiniteElementSpace *fes)
{
   double* qf_data = qf->GetData();
   
   //The below should be exactly the same as what
   //the other for loop is trying to accomplish
   for (int i = 0; i < qf->Size(); ++i){
     qf_data[i] = val;
   }
}

void initQuadFuncTensorIdentity(QuadratureFunction *qf, ParFiniteElementSpace *fes){
   
   const IntegrationRule *ir;
   double* qf_data = qf->GetData();
   int qf_offset = qf->GetVDim(); // offset at each integration point
   QuadratureSpace* qspace = qf->GetSpace();
   
   // loop over elements
   for (int i = 0; i < fes->GetNE(); ++i){
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
}

//Routine to test the deformation gradient to make sure we're getting out the right values.
//This applies the following displacement field to the nodal values:
//u_vec = (2x + 3y + 4z)i + (4x + 2y + 3z)j + (3x + 4y + 2z)k
void test_deformation_field_set(ParGridFunction *gf, Vector *vec, ParFiniteElementSpace *fes)
{
   
   double* temp_vals = gf->GetData();
//   double* vals = vec->GetData();

   int dim = gf->Size()/3;
   int dim2 = vec->Size()/3;
   int  myid;

   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   
   printf("gf data size: %d vec data size %d\n", dim, dim2);
   
   for (int i = 0; i < dim; ++i)
   {
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

