//
//  option_parser.cpp
//  
//
//  Created by Carson, Robert Allen on 1/25/19.
//

#include "option_parser.hpp"
#include "TOML_Reader/cpptoml.h"
#include <iostream>

using namespace std;

namespace mfem{

//my_id corresponds to the processor id.
void ExaOptions::parse_options(int my_id){
   toml = cpptoml::parse_file(floc);
   
   //From the toml file it finds all the values related to state and mat'l
   //properties
   get_properties();
   //From the toml file it finds all the values related to the BCs
   get_bcs();
   //From the toml file it finds all the values related to the model
   get_model();
   //From the toml file it finds all the values related to the time
   get_time_steps();
   //From the toml file it finds all the values related to the visualizations
   get_visualizations();
   //From the toml file it finds all the values related to the Solvers
   get_solvers();
   //From the toml file it finds all the values related to the mesh
   get_mesh();
   //If the processor is set 0 then the options are printed out.
   if(my_id == 0) print_options();
   
}
   
//From the toml file it finds all the values related to state and mat'l
//properties
void ExaOptions::get_properties(){
   
   double _temp_k = toml->get_qualified_as<double>("Properties.temperature").value_or(300.);

   if(_temp_k <= 0.0) {
      MFEM_ABORT("Properties.temperature is given in Kelvins and therefore can't be less than 0");
   }

   temp_k = _temp_k;

   //Material properties are obtained first
   auto prop_table = toml->get_table_qualified("Properties.Matl_Props");
   //Check to see if our table exists
   if(prop_table != nullptr){
     std::string _props_file = prop_table->get_as<std::string>("floc").value_or("props.txt");
      props_file = _props_file;
      nProps = prop_table->get_as<int>("num_props").value_or(1);
   }
   
   //State variable properties are now obtained
   auto state_table = toml->get_table_qualified("Properties.State_Vars");
   //Check to see if our table exists
   if(state_table != nullptr){
      numStateVars = state_table->get_as<int>("num_vars").value_or(1);
      std::string _state_file = state_table->get_as<std::string>("floc").value_or("state.txt");
      state_file = _state_file;
   }
   
   //Grain related properties are now obtained
   auto grain_table = toml->get_table_qualified("Properties.Grain");
   //Check to see if our table exists
   if(grain_table != nullptr){
      grain_statevar_offset = grain_table->get_as<int>("ori_state_var_loc").value_or(-1);
      grain_custom_stride = grain_table->get_as<int>("ori_stride").value_or(0);
      std::string _ori_type = grain_table->get_as<std::string>("ori_type").value_or("euler");
      ngrains = grain_table->get_as<int>("num_grains").value_or(0);
      std::string _ori_file = grain_table->get_as<std::string>("ori_floc").value_or("ori.txt");
      ori_file = _ori_file;
      std::string _grain_map = grain_table->get_as<std::string>("grain_floc").value_or("grain_map.txt");
      grain_map = _grain_map;
      
      //I still can't believe C++ doesn't allow strings to be used in switch statements...
      if((_ori_type == "euler") || _ori_type == "Euler" || (_ori_type == "EULER")) {
         ori_type = OriType::EULER;
      }else if((_ori_type == "quat") || (_ori_type == "Quat") || (_ori_type == "quaternion") || (_ori_type == "Quaternion")){
         ori_type = OriType::QUAT;
      }else if((_ori_type == "custom") || (_ori_type == "Custom") || (_ori_type == "CUSTOM")){
         ori_type = OriType::CUSTOM;
      }else{
	 MFEM_ABORT("Properties.Grain.ori_type was not provided a valid type.");
         ori_type = OriType::NOTYPE;
      }
   }//end of if statement for grain data
}//End of propert parsing
   
//From the toml file it finds all the values related to the BCs
void ExaOptions::get_bcs(){
   
   //Getting out arrays of values isn't always the simplest thing to do using
   //this TOML libary.
   auto ess_ids = toml->get_qualified_array_of<int64_t>("BCs.essential_ids");
   std::vector<int> _essential_ids;
   for(const auto& val: *ess_ids){
      _essential_ids.push_back(val);
   }
   
   if(_essential_ids.empty()) MFEM_ABORT("BCs.essential_ids was not provided any values.");
   
   //Now filling in the essential id array
   ess_id.SetSize(_essential_ids.size());
   for(int i = 0; i < ess_id.Size(); i++){
      ess_id[i] = _essential_ids[i];
   }
   
   
   //Getting out arrays of values isn't always the simplest thing to do using
   //this TOML libary.
   auto ess_comps = toml->get_qualified_array_of<int64_t>("BCs.essential_comps");
   std::vector<int> _essential_comp;
   for(const auto& val: *ess_comps){
      _essential_comp.push_back(val);
   }

   if(_essential_comp.empty()) MFEM_ABORT("BCs.essential_comps was not provided any values.");
   //Now filling in the essential components array
   ess_comp.SetSize(_essential_comp.size());
   for(int i = 0; i < ess_comp.Size(); i++){
      ess_comp[i] = _essential_comp[i];
   }
   
   //Getting out arrays of values isn't always the simplest thing to do using
   //this TOML libary.
   auto ess_vals = toml->get_qualified_array_of<double>("BCs.essential_vals");
   std::vector<double> _essential_vals;
   for(const auto& val: *ess_vals){
      _essential_vals.push_back(val);
   }
   
   if(_essential_vals.empty()) MFEM_ABORT("BCs.essential_vals was not provided any values.");
   
   //Now filling in the essential disp vector
   ess_disp.SetSize(_essential_vals.size());
   for(int i = 0; i < ess_disp.Size(); i++){
      ess_disp[i] = _essential_vals[i];
   }
   
}//end of parsing BCs

//From the toml file it finds all the values related to the model
void ExaOptions::get_model(){
   std::string _mech_type = toml->get_qualified_as<std::string>("Model.mech_type").value_or("");

   //I still can't believe C++ doesn't allow strings to be used in switch statements...
   if((_mech_type == "umat") || (_mech_type == "Umat") || (_mech_type == "UMAT") || (_mech_type == "UMat")) {
      mech_type = MechType::UMAT;
   }else if((_mech_type == "exacmech") || (_mech_type == "Exacmech") || (_mech_type == "ExaCMech") || (_mech_type == "EXACMECH")){
      mech_type = MechType::EXACMECH;
   }else{
      MFEM_ABORT("Model.mech_type was not provided a valid type.");
      mech_type = MechType::NOTYPE;
   }

   cp = toml->get_qualified_as<bool>("Model.cp").value_or(false);

   if (mech_type == MechType::EXACMECH)
   {
      if (!cp) {
         MFEM_ABORT("Model.cp needs to be set to true when using ExaCMech based models.");
      }

      if(ori_type != OriType::QUAT){
         MFEM_ABORT("Properties.Grain.ori_type is not set to quaternion for use with an ExaCMech model.");
         xtal_type = XtalType::NOTYPE;
      }

      grain_statevar_offset = 8;

      auto exacmech_table = toml->get_table_qualified("Model.ExaCMech");
      
      std::string _xtal_type = exacmech_table->get_as<std::string>("xtal_type").value_or("");
      std::string _slip_type = exacmech_table->get_as<std::string>("slip_type").value_or("");

      if((_xtal_type == "fcc") || (_xtal_type == "FCC"))
      {
         xtal_type = XtalType::FCC;

         if(numStateVars != 23){
            MFEM_ABORT("Properties.State_Vars.num_vars needs 23 values for a " 
            "cubic material when using an ExaCMech model. Note: the number of values for a quaternion " 
            "are not included in this count.");
         }

      }else{
         MFEM_ABORT("Model.ExaCMech.xtal_type was not provided a valid type.");
         xtal_type = XtalType::NOTYPE;
      }

      if((_slip_type == "mts") || (_slip_type == "MTS") || (_slip_type == "mtsdd") || (_slip_type == "MTSDD"))
      {
         slip_type = SlipType::MTSDD;
         if(nProps != 24){
            MFEM_ABORT("Properties.Matl_Props.num_props needs 24 values for the MTSDD option");
         }
      }else if ((_slip_type == "powervoce") || (_slip_type == "PowerVoce") || (_slip_type == "POWERVOCE"))
      {
         slip_type = SlipType::POWERVOCE;
         if(nProps != 17){
            MFEM_ABORT("Properties.Matl_Props.num_props needs 17 values for the PowerVoce option");
         }
      }else{
         MFEM_ABORT("Model.ExaCMech.slip_type was not provided a valid type.");
         slip_type = SlipType::NOTYPE;
      }

   } 

}//end of model parsing

//From the toml file it finds all the values related to the time
void ExaOptions::get_time_steps(){
   //First look at the auto time stuff
   auto auto_table = toml->get_table_qualified("Time.Auto");
   //check to see if our table exists
   if(auto_table != nullptr){
      dt_cust = false;
      dt = auto_table->get_as<double>("dt").value_or(1.0);
      t_final = auto_table->get_as<double>("dt").value_or(1.0);
   }
   //Time to look at our custom time table stuff
   auto cust_table = toml->get_table_qualified("Time.Custom");
   //check to see if our table exists
   if(cust_table != nullptr){
      dt_cust = true;
      nsteps = cust_table->get_as<int>("nsteps").value_or(1);
      std::string _dt_file = cust_table->get_as<std::string>("floc").value_or("custom_dt.txt");
      dt_file = _dt_file;
   }
}//end of time step parsing

//From the toml file it finds all the values related to the visualizations
void ExaOptions::get_visualizations(){
   vis_steps = toml->get_qualified_as<int>("Visualizations.steps").value_or(1);
   visit = toml->get_qualified_as<bool>("Visualizations.visit").value_or(false);
   conduit = toml->get_qualified_as<bool>("Visualizations.conduit").value_or(false);
   std::string _basename = toml->get_qualified_as<std::string>("Visualizations.floc").value_or("results/exaconstit");
   basename = _basename;
}//end of visualization parsing

//From the toml file it finds all the values related to the Solvers
void ExaOptions::get_solvers(){
   //Obtaining information related to the newton raphson solver
   auto nr_table = toml->get_table_qualified("Solvers.NR");
   if(nr_table != nullptr){
      newton_iter = nr_table->get_as<int>("iter").value_or(25);
      newton_rel_tol = nr_table->get_as<double>("rel_tol").value_or(1e-5);
      newton_abs_tol = nr_table->get_as<double>("abs_tol").value_or(1e-10);
   }//end of NR info
   //Now getting information about the Krylov solvers used to the linearized
   //system of equations of the nonlinear problem.
   auto iter_table = toml->get_table_qualified("Solvers.Krylov");
   if(iter_table != nullptr){
      krylov_iter = iter_table->get_as<int>("iter").value_or(200);
      krylov_rel_tol = iter_table->get_as<double>("rel_tol").value_or(1e-10);
      krylov_abs_tol = iter_table->get_as<double>("abs_tol").value_or(1e-30);
      std::string _solver = iter_table->get_as<std::string>("solver").value_or("GMRES");
      if((_solver == "GMRES") || (_solver == "gmres")){
         solver = KrylovSolver::GMRES;
      }else if((_solver == "PCG") || (_solver == "pcg")){
         solver = KrylovSolver::PCG;
      }else if ((_solver == "MINRES") || (_solver == "minres")){
         solver = KrylovSolver::MINRES;
      }else{
         MFEM_ABORT("Solvers.Krylov.solver was not provided a valid type.");
         solver = KrylovSolver::NOTYPE;
      }
   }//end of krylov solver info
}//end of solver parsing

//From the toml file it finds all the values related to the mesh
void ExaOptions::get_mesh(){
   //Refinement of the mesh and element order
   ser_ref_levels = toml->get_qualified_as<int>("Mesh.ref_ser").value_or(0);
   par_ref_levels = toml->get_qualified_as<int>("Mesh.ref_par").value_or(0);
   order = toml->get_qualified_as<int>("Mesh.prefinement").value_or(1);
   //file location of the mesh
   std::string _mesh_file = toml->get_qualified_as<std::string>("Mesh.floc").value_or("../../data/cube-hex-ro.mesh");
   mesh_file = _mesh_file;
   //Type of mesh that we're reading/going to generate
   std::string mtype = toml->get_qualified_as<std::string>("Mesh.type").value_or("other");
   if((mtype == "cubit") || (mtype == "Cubit") || (mtype == "CUBIT")){
      mesh_type = MeshType::CUBIT;
   }else if((mtype == "auto") || (mtype == "Auto") || (mtype == "AUTO")){
      mesh_type = MeshType::AUTO;
      auto auto_table = toml->get_table_qualified("Mesh.Auto");

      //Basics to generate at least 1 element of length 1.
      //mx = auto_table->get_as<double>("length").value_or(1.0);
      auto mx = auto_table->get_qualified_array_of<double>("length");
      std::vector<double> _mxyz;
      for(const auto& val: *mx){
	_mxyz.push_back(val);
      }
      if(_mxyz.size() != 3){
	 MFEM_ABORT("Mesh.Auto.length was not provided a valid array of size 3.");
      }
      mxyz[0] = _mxyz[0];
      mxyz[1] = _mxyz[1];
      mxyz[2] = _mxyz[2];

      //nx = auto_table->get_as<int>("ncuts").value_or(1);
      auto nx = auto_table->get_qualified_array_of<int64_t>("ncuts");
      std::vector<int> _nxyz;
      for(const auto& val: *nx){
        _nxyz.push_back(val);
      }
      if(_nxyz.size() != 3){
	MFEM_ABORT("Mesh.Auto.ncuts was not provided a valid array of size 3.");
      } 
      nxyz[0] =_nxyz[0];
      nxyz[1] =_nxyz[1];
      nxyz[2] =_nxyz[2];

   }else if((mtype == "other") || (mtype == "Other") || (mtype == "OTHER")){
      mesh_type = MeshType::OTHER;
   }else{
      MFEM_ABORT("Mesh.type was not provided a valid type.");
      mesh_type = MeshType::NOTYPE;
   }//end of mesh type parsing
}//End of mesh parsing

void ExaOptions::print_options(){
   
   std::cout << "Mesh file location: " << mesh_file << "\n";
   std::cout << "Mesh type: ";
   if (mesh_type == MeshType::OTHER){
      std::cout << "other";
   }else if(mesh_type == MeshType::CUBIT){
      std::cout << "cubit";
   }else{
      std::cout << "auto";
   }
   std::cout << "\n";

   std::cout << "Edge dimensions (mx, my, mz): " << mxyz[0] <<" " << mxyz[1] << " " << mxyz[2] <<"\n";
   std::cout << "Number of cells on an edge (nx, ny, nz): " << nxyz[0] <<" " << nxyz[1] << " " << nxyz[2] <<"\n";
   
   std::cout << "Serial Refinement level: " << ser_ref_levels << "\n";
   std::cout << "Parallel Refinement level: " << par_ref_levels << "\n";
   std::cout << "P-refinement level: " << order << "\n";
   
   std::cout << std::boolalpha;
   std::cout << "Custom dt flag (dt_cust): " << dt_cust << "\n";
   
   if(dt_cust){
      std::cout << "Number of time steps (nsteps): " << nsteps << "\n";
      std::cout << "Custom time file loc (dt_file): " << dt_file << "\n";
   }else{
      std::cout << "Constant time stepping on \n";
      std::cout << "Final time (t_final): " << t_final << "\n";
      std::cout << "Time step (dt): " << dt << "\n";
   }

   std::cout << "Visit flag: " << visit << "\n";
   std::cout << "Conduit flag: " << conduit << "\n";
   std::cout << "Visualization steps: " << vis_steps << "\n";
   std::cout << "Visualization directory: " << basename << "\n";
   
   std::cout << "Newton Raphson rel. tol.: " << newton_rel_tol << "\n";
   std::cout << "Newton Raphson abs. tol.: " << newton_abs_tol << "\n";
   std::cout << "Newton Raphson # of iter.: " << newton_iter << "\n";
   std::cout << "Newton Raphson grad debug: " << grad_debug << "\n";
   
   std::cout << "Krylov solver: ";
   if (solver == KrylovSolver::GMRES){
      std::cout << "GMRES";
   }else if(solver == KrylovSolver::PCG){
      std::cout << "PCG";
   }else{
      std::cout << "MINRES";
   }
   std::cout << "\n";
   
   std::cout << "Krylov solver rel. tol.: " << krylov_rel_tol << "\n";
   std::cout << "Krylov solver abs. tol.: " << krylov_abs_tol << "\n";
   std::cout << "Krylov solver # of iter.: " << krylov_iter << "\n";

   std::cout << "Mechanical model library being used ";

   if (mech_type == MechType::UMAT){
      std::cout << "UMAT\n";
   }else if (mech_type == MechType::EXACMECH){
      std::cout << "ExaCMech\n";
      std::cout << "Crystal symmetry group is ";
      if (xtal_type == XtalType::FCC) {
         std::cout << "FCC\n";
      }

      std::cout << "Slip system and hardening model being used is ";

      if(slip_type == SlipType::MTSDD){
         std::cout << "MTS slip like kinetics with dislocation density based hardening\n";
      } else if (slip_type == SlipType::POWERVOCE) {
         std::cout << "Power law slip kinetics with a Voce hardening law\n";
      }
   }
   
   std::cout << "Xtal Plasticity being used: " << cp << "\n";
   
   std::cout << "Orientation file location: " << ori_file << "\n";
   std::cout << "Grain map file location: " << grain_map << "\n";
   std::cout << "Number of grains: " << ngrains << "\n";
   
   std::cout << "Orientation type: ";
   if(ori_type == OriType::EULER){
      std::cout << "euler";
   }else if(ori_type == OriType::QUAT){
      std::cout << "quaternion";
   }else{
      std::cout << "custom";
   }
   std::cout << "\n";
   
   std::cout << "Custom stride to read grain map file: " << grain_custom_stride << "\n";
   std::cout << "Orientation offset in state variable file: " << grain_statevar_offset << "\n";

   std::cout << "Number of properties: " << nProps << "\n";
   std::cout << "Property file location: " << props_file << "\n";
   
   std::cout << "Number of state variables: " << numStateVars << "\n";
   std::cout << "State variable file location: " << state_file << "\n";
   
   std::cout << "Essential ids are set as: ";
   for(int i = 0; i < ess_id.Size(); i++){
      std::cout << ess_id[i] << " ";
   }
   std::cout << "\n";
   
   std::cout << "Essential components are set as: ";
   for(int i = 0; i < ess_comp.Size(); i++){
      std::cout << ess_comp[i] << " ";
   }
   std::cout << "\n";
   
   std::cout << "Essential boundary values are set as: ";
   for(int i = 0; i < ess_disp.Size(); i++){
      std::cout << ess_disp[i] << " ";
   }
   std::cout << "\n";

}//End of printing out options
   
}

//}
