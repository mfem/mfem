//                                PUMI Example 
//
// Compile with: Add GenerateMFEMMesh.cpp in the CMakeLists.txt
//
// Sample runs:  ./GenerateMFEMMesh -m ../data/pumi/serial/sphere.smb
//                    -p ../data/pumi/geom/sphere.x_t -o 1 
//
// Description:  The purpose of this example is to generate MFEM meshes for 
//               complex geometry. The inputs are a Parasolid model, "*.xmt_txt"
//               and a SCOREC mesh "*.smb". Switch "-o" can be used to increase the 
//               geometric order up to order 6 and consequently write the mesh 
//               in that order.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "../mesh/mesh_pumi.hpp"

#include <SimUtil.h>
#include <apfMDS.h>
#include <gmi_null.h>
#include <gmi_sim.h>
#include <PCU.h>
#include <apfConvert.h>
#include <gmi_mesh.h>
#include <crv.h>

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
    //initilize mpi 
    int num_proc, myId;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &myId);
    
   // 1. Parse command-line options.
   const char *mesh_file = "../data/pumi/serial/sphere.smb";
   const char *model_file = "../data/pumi/geom/sphere.x_t";
   int order = 1;
   bool visualization = 1;  

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&model_file, "-p", "--parasolid",
                  "Parasolid model to use.");   
  
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   
   //Read the SCOREC Mesh 
   PCU_Comm_Init();
   SimUtil_start();
   Sim_readLicenseFile(0);

   gmi_sim_start();
   gmi_register_mesh();
   gmi_register_sim();
   
   apf::Mesh2* pumi_mesh;
   pumi_mesh = apf::loadMdsMesh(model_file, mesh_file);
   
   
   //If it is higher order change shape
   if (order > 1){
       crv::BezierCurver bc(pumi_mesh, order, 2);
       bc.run();
   }   
   pumi_mesh->verify();           

   // 2. Create the MFEM mesh object from the PUMI mesh. We can handle triangular
   //    and tetrahedral meshes. Other inputs are the same as MFEM default 
   //    constructor.
   Mesh *mesh = new PumiMesh(pumi_mesh, 1, 0);   

   //Write mesh in MFEM fromat
   ofstream fout("MFEMformat.mesh");
   fout.precision(8); 
   mesh->Print(fout);
 
   
   delete mesh;

   pumi_mesh->destroyNative();
   apf::destroyMesh(pumi_mesh);
   PCU_Comm_Free();
   gmi_sim_stop();
   
   Sim_unregisterAllKeys();
   SimUtil_stop();
   
   MPI_Finalize();   
   return 0;
}
