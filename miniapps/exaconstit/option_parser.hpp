//
//  option_parser.hpp
//  
//
//  Created by Carson, Robert Allen on 1/25/19.
//

#ifndef option_parser_hpp
#define option_parser_hpp

#include <stdio.h>
#include "TOML_Reader/cpptoml.h"
#include <iostream>
#include "mfem.hpp"



//Taking advantage of C++11 to make it much clearer that we're using enums
enum class KrylovSolver {GMRES, PCG, MINRES, NOTYPE};
enum class OriType {EULER, QUAT, CUSTOM, NOTYPE};
enum class MeshType {CUBIT, AUTO, OTHER, NOTYPE};

using namespace std;

namespace mfem{

class ExaOptions{
   
   public:
   
      // mesh variables
      std::string mesh_file;
      MeshType mesh_type;
      double mx; // edge dimension (mx = my = mz)
      int  nx; // number of cells on an edge (nx = ny = nz)
   
   
      // serial and parallel refinement levels
      int ser_ref_levels;
      int par_ref_levels;
   
      // polynomial interpolation order
      int order;
   
      // final simulation time and time step (set each to 1.0 for
      // single step debug)
      double t_final;
      double dt;
      // We have a custom dt flag
      bool dt_cust;
      // Number of time steps to take
      int nsteps;
      // File to read the custom time steps from
      std::string dt_file;
      // Vector to hold custom time steps if there are any
      Vector cust_dt;
   
      // visualization input args
      int vis_steps;
      // visualization variable for visit
      bool visit;
      //Where to store the end time step files
      std::string basename;
   
      // newton input args
      double newton_rel_tol;
      double newton_abs_tol;
      int newton_iter;
   
      // solver input args
      // GMRES is currently set as the default iterative solver
      // until the bug in the PCG solver is found and fixed.
      bool grad_debug;
      double krylov_rel_tol;
      double krylov_abs_tol;
      int krylov_iter;
   
      KrylovSolver solver;
   
      // input arg to specify Abaqus UMAT
      bool umat;
   
      // input arg to specify crystal plasticity or hyperelasticity
      // (for testing)
      bool cp;
      bool hyperelastic;
   
      // grain input arguments
      std::string ori_file; // grain orientations (F_p_inv for Curt's UMAT?)
      std::string grain_map; // map of grain id to element centroid
      int ngrains;
      OriType ori_type;
      int grain_custom_stride; // TODO check that this is used with "grain_custom"
      int grain_statevar_offset;
   
      // material properties input arguments
      std::string props_file;
      int nProps; // at least have one dummy property
   
      // state variables file with constant values used to initialize ALL integration points
      std::string state_file;
      int numStateVars; // at least have one dummy property
   
      // boundary condition input args
      Array<int> ess_id;   // essential bc ids for the whole boundary
      Vector ess_disp; // vector of displacement components for each attribute in ess_id
      Array<int> ess_comp; // component combo (x,y,z = -1, x = 1, y = 2, z = 3,
      // xy = 4, yz = 5, xz = 6, free = 0
   
      //Parse the TOML file for all of the various variables.
      //In other words this is our driver to get all of the values.
      void parse_options(int my_id);
   
   ExaOptions(std::string _floc) : floc{_floc} {
      
      //Matl and State Property related variables
      numStateVars = 1;
      nProps = 1;
      state_file = "state.txt";
      props_file = "props.txt";
      
      //Grain related variables
      grain_statevar_offset = -1;
      grain_custom_stride = 0;
      ori_type = OriType::EULER;
      ngrains = 0;
      grain_map = "grain_map.txt";
      ori_file = "grains.txt";
      
      //Model related parameters
      hyperelastic = false;
      cp = false;
      umat = false;
      
      //Krylov Solver related variables
      //We set the default solver as GMRES in case we accidentally end up dealing
      //with a nonsymmetric matrix for our linearized system of equations.
      solver = KrylovSolver::GMRES;
      krylov_rel_tol = 1.0e-10;
      krylov_abs_tol = 1.0e-30;
      krylov_iter = 200;
      
      //NR parameters
      newton_rel_tol = 1.0e-5;
      newton_abs_tol = 1.0e-10;
      newton_iter = 25;
      grad_debug = false;
      
      //Visualization related parameters
      basename = "results/exaconstit";
      visit = false;
      vis_steps = 1;
      
      //Time step related parameters
      t_final = 1.0;
      dt = 1.0;
      dt_cust = false;
      nsteps = 1;
      dt_file = "custom_dt.txt";
      
      //Mesh related variables
      ser_ref_levels = 0;
      par_ref_levels = 0;
      order = 1;
      mesh_file = "../../data/cube-hex-ro.mesh";
      mesh_type = MeshType::OTHER;
      mx = 1.0;
      nx = 1;
   }//End of ExaOptions constructor
   
   virtual ~ExaOptions() {}
   
   protected:
      std::shared_ptr<cpptoml::table> toml;
      std::string floc;
      //From the toml file it finds all the values related to state and mat'l
      //properties
      void get_properties();
      //From the toml file it finds all the values related to the BCs
      void get_bcs();
      //From the toml file it finds all the values related to the model
      void get_model();
      //From the toml file it finds all the values related to the time
      void get_time_steps();
      //From the toml file it finds all the values related to the visualizations
      void get_visualizations();
      //From the toml file it finds all the values related to the Solvers
      void get_solvers();
      //From the toml file it finds all the values related to the mesh
      void get_mesh();
      //Prints out a list of all the options being used
      void print_options();
};
   
}



#endif /* option_parser_hpp */
