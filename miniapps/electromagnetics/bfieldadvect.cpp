#include "bfieldadvect_solver.hpp"

using namespace std;
using namespace mfem;
using namespace mfem::electromagnetics;

void BFieldFunc(const Vector &, Vector&);

// int main(int argc, char *argv[])
// {
//    MPI_Session mpi(argc, argv);

//    // Parse command-line options.
//    const char *mesh_file = "../../data/toroid-hex.mesh";
//    int order = 1;
//    int serial_ref_levels = 0;
//    int parallel_ref_levels = 0;
//    bool visualization = false;
//    bool visit = true;

//    OptionsParser args(argc, argv);
//    args.AddOption(&mesh_file, "-m", "--mesh",
//                   "Mesh file to use.");
//    args.AddOption(&order, "-o", "--order",
//                   "Finite element order (polynomial degree).");
//    args.AddOption(&serial_ref_levels, "-rs", "--serial-ref-levels",
//                   "Number of serial refinement levels.");
//    args.AddOption(&parallel_ref_levels, "-rp", "--parallel-ref-levels",
//                   "Number of parallel refinement levels.");
//    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
//                   "--no-visualization",
//                   "Enable or disable GLVis visualization.");
//    args.AddOption(&visit, "-visit", "--visit", "-no-visit",
//                   "--no-visualization",
//                   "Enable or disable VisIt visualization.");
//    args.Parse();
//    if (!args.Good())
//    {
//       if (mpi.Root())
//       {
//          args.PrintUsage(cout);
//       }
//       return 1;
//    }
//    if (mpi.Root())
//    {
//       args.PrintOptions(cout);
//    }

//    Mesh *mesh;
//    ifstream imesh(mesh_file);
//    if (!imesh)
//    {
//       if (mpi.Root())
//       {
//          cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
//       }
//       return 2;
//    }
//    mesh = new Mesh(20, 20, 5, Element::HEXAHEDRON, false, 4.0, 4.0, 1.0);
//    //mesh = new Mesh(imesh, 1, 1);
//    imesh.close();   

//    // Refine the serial mesh on all processors to increase the resolution. In
//    // this example we do 'ref_levels' of uniform refinement.
//    for (int l = 0; l < serial_ref_levels; l++)
//    {
//       mesh->UniformRefinement();
//    }

//    // Define a parallel mesh by a partitioning of the serial mesh. Refine this
//    // mesh further in parallel to increase the resolution. Once the parallel
//    // mesh is defined, the serial mesh can be deleted.
//    ParMesh pmesh(MPI_COMM_WORLD, *mesh);
//    delete mesh;

//    // Refine this mesh in parallel to increase the resolution.
//    for (int l = 0; l < parallel_ref_levels; l++)
//    {
//       pmesh.UniformRefinement();
//    }

//    //Create a copy of the refined paralle mesh and peturb some inner nodes
//    ParMesh pmesh_new(pmesh);

//    //Set up the pre and post advection fields on the relevant meshes/spaces
//    RT_ParFESpace *HDivFESpaceOld  = new RT_ParFESpace(&pmesh,order,pmesh.Dimension());
//    RT_ParFESpace *HDivFESpaceNew  = new RT_ParFESpace(&pmesh_new,order,pmesh_new.Dimension());
//    ParGridFunction *b = new ParGridFunction(HDivFESpaceOld);
//    ParGridFunction *b_new = new ParGridFunction(HDivFESpaceNew);

//    //Set the initial B value
//    *b = 0.0;
//    *b_new = 0.0;
//    VectorFunctionCoefficient BFieldCoef(3,BFieldFunc);
//    b->ProjectCoefficient(BFieldCoef);


//    BFieldAdvector advector(&pmesh, &pmesh_new, 1);
//    advector.Advect(b, b_new);

//    ParGridFunction *b_recon = advector.GetReconstructedB();
//    ParGridFunction *curl_b = advector.GetCurlB();
//    ParGridFunction *clean_curl_b = advector.GetCleanCurlB();

//    // Handle the visit visualization
//    if (visit)
//    {
//       VisItDataCollection visit_dc("bfield-advect", &pmesh);
//       visit_dc.RegisterField("B", b);
//       visit_dc.RegisterField("Curl_B", curl_b);
//       visit_dc.RegisterField("Clean_Curl_B", clean_curl_b);
//       visit_dc.RegisterField("B_recon", b_recon);
//       visit_dc.SetCycle(0);
//       visit_dc.SetTime(0);
//       visit_dc.Save();
//    }

// }

int main(int argc, char *argv[])
{
   int order = 1;
   MPI_Session mpi(argc, argv);
   Mesh mesh(20, 20, 5, Element::HEXAHEDRON, false, 4.0, 4.0, 1.0);
   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   //Set up the pre and post advection fields on the relevant meshes/spaces
   RT_ParFESpace *HDivFESpace  = new RT_ParFESpace(&pmesh,order,pmesh.Dimension());
   ND_ParFESpace *HCurlFESpace  = new ND_ParFESpace(&pmesh,order,pmesh.Dimension());
   ParGridFunction *b = new ParGridFunction(HDivFESpace);
   ParGridFunction *curl_b = new ParGridFunction(HCurlFESpace);

   //Weak curl operator for taking the curl of B living in Hdiv
   ConstantCoefficient oneCoef(1.0);
   ParMixedBilinearForm *weakCurl = new ParMixedBilinearForm(HDivFESpace, HCurlFESpace);
   weakCurl->AddDomainIntegrator(new VectorFECurlIntegrator(oneCoef));
   weakCurl->Assemble();
   weakCurl->Finalize();   

   //Set the initial B value
   *b = 0.0;
   *curl_b = 0.0;
   VectorFunctionCoefficient BFieldCoef(3,BFieldFunc);
   b->ProjectCoefficient(BFieldCoef);
   weakCurl->Mult(*b, *curl_b);

   // Handle the visit visualization
   VisItDataCollection visit_dc("bfield-advect", &pmesh);
   visit_dc.RegisterField("B", b);
   visit_dc.RegisterField("Curl_B", curl_b);
   visit_dc.SetCycle(0);
   visit_dc.SetTime(0);
   visit_dc.Save();


}


void BFieldFunc(const Vector &x, Vector &B)
{
   B.SetSize(3);
   B[0] =  x[1] - 2.0;
   B[1] = -(x[0] - 2.0);
   B[2] =  0.0;
}