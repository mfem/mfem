#include "mfem.hpp"
#include "boundary_edge_dofs_patch.hpp"
#include "loop_orientation.hpp"
#include "loop_length.hpp"

using namespace std;
using namespace mfem;

void f_exact(const Vector &, Vector &);
int dim;

int main(int argc, char *argv[])
{
    // Initialize MPI
    Mpi::Init(argc, argv);
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    // Parse command-line options
    const char *mesh_file = "../data/nested_cubes.msh";
    bool visualization = true;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                    "--no-visualization", "Enable or disable GLVis visualization.");
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0) { args.PrintUsage(cout); }
        return 1;
    }
    if (myid == 0) { args.PrintOptions(cout); }

    // Read the mesh from the given mesh file
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();
   
    if (myid == 0)
    {
        cout << "Mesh Dimension: " << dim << endl;
        cout << "Number of Elements: " << mesh->GetNE() << endl;
        cout << "Number of Boundary Elements: " << mesh->GetNBE() << endl;
        cout << "Number of Vertices: " << mesh->GetNV() << endl;
      
        // Print boundary attributes
        cout << "Boundary Attributes: ";
        for (int i = 0; i < mesh->bdr_attributes.Size(); i++)
        {
            cout << mesh->bdr_attributes[i] << " ";
        }
        cout << endl;
      
        // Print domain attributes
        cout << "Domain Attributes: ";
        for (int i = 0; i < mesh->attributes.Size(); i++)
        {
            cout << mesh->attributes[i] << " ";
        }
        cout << endl;
    }

    // Create a parallel mesh by a partitioning of the serial mesh
    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;

    // Create edge-based finite element space (Nédélec)
    int order = 1;
    FiniteElementCollection *fec = new ND_FECollection(order, pmesh->Dimension());
    ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);

    // Using the standalone function
    Array<int> ess_tdof_list;
    Array<int> ess_edge_list;
    std::unordered_map<int, int> dof_to_edge, dof_to_orientation;
    std::unordered_set<int> boundary_edge_ldofs;
    std::unordered_map<int, int> dof_to_boundary_element;
   
    // Get boundary edge DoFs for boundary attribute 5
    Array<int> bdr_attr_marker(pmesh->bdr_attributes.Max());
    bdr_attr_marker = 0;
    bdr_attr_marker[10] = 1; // Mark attribute 11 for the loop boundary condition
    

    GetBoundaryEdgeDoFs(fespace, pmesh, bdr_attr_marker, ess_tdof_list, &dof_to_edge, 
                       &dof_to_orientation, &boundary_edge_ldofs, &dof_to_boundary_element, &ess_edge_list);
    //fespace->GetBoundaryEdgeDoFs(bdr_attr_marker, ess_tdof_list, &dof_to_edge, 
    //                   &dof_to_orientation, &boundary_edge_ldofs, &dof_to_boundary_element, &ess_edge_list);
    cout << "Rank " << myid << ", number of edge dofs: " << boundary_edge_ldofs.size() << 
         ", number of essential dofs: " << ess_tdof_list.Size() << endl;

    // Collect the global number of boundary edge DoFs
    int local_final_dofs = boundary_edge_ldofs.size();
    int total_final_dofs;
    MPI_Allreduce(&local_final_dofs, &total_final_dofs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Debug output
    cout << "Rank " << myid << ", final boundary edge DOFs in this rank: " << local_final_dofs << endl;
    if (myid == 0) cout << "  Total final boundary edge DOFs (all ranks): " << total_final_dofs << endl;
    
    // Determine edge orientations relative to counter-clockwise loop direction
    Vector loop_normal(3);
    loop_normal = 0.0;
    //loop_normal[0] = 1.0; // +x direction
    loop_normal[1] = 1.0; // +y direction
    //loop_normal[2] = -1.0; // +z direction

    std::unordered_map<int, int> edge_loop_orientation;
    //ComputeLoopEdgeOrientations(pmesh, dof_to_edge, dof_to_boundary_element,
    //                                        loop_normal, edge_loop_orientation);
    fespace->ComputeLoopEdgeOrientations(dof_to_edge, dof_to_boundary_element,
                                        loop_normal, edge_loop_orientation);
                                        
    Array<HYPRE_BigInt> global_edge_indices;
    pmesh->GetGlobalEdgeIndices(global_edge_indices);
    
    // Compute and print the boundary loop length
    double local_length = ComputeBoundaryLoopLength(pmesh, dof_to_edge);
    double total_length;
    MPI_Allreduce(&local_length, &total_length, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    cout << "Rank " << myid << ", local boundary loop length: " << local_length << endl;
    if (myid == 0) cout << "  Total boundary loop perimeter: " << total_length << endl;
    

    // Mark all other boundaries (1-6) for zero tangential boundary condition
    Array<int> ess_bdr(pmesh->bdr_attributes.Max());
    ess_bdr = 1; // Mark all boundary attributes for essential BC

    // Get essential boundary DOFs for zero tangential field on all boundaries
    Array<int> ess_tdof_list_all;
    fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list_all);

    dim = mesh->Dimension();
    int sdim = mesh->SpaceDimension();
    VectorFunctionCoefficient f(sdim, f_exact);
    ParLinearForm *b = new ParLinearForm(fespace);
    b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
    b->Assemble();

    
    // Create grid function for the solution
    ParGridFunction x(fespace);
    Array<double> bc_values(x.Size());
    x = 0.0;
    bc_values = 0.0;

    // First set the boundary values in the local grid function
    for (int dof : boundary_edge_ldofs)
    {
        int edge = dof_to_edge[dof];
        int orientation = edge_loop_orientation[edge];   
        
        // Set the boundary value with orientation correction
        bc_values[dof] = 1.0 * orientation;
    }

    // Synchronize boundary values across processors
    fespace->SynchronizeBC(bc_values);
    for (int i = 0; i < bc_values.Size(); i++)
    {x[i] = bc_values[i];}
    
    // Set up the bilinear form for the EM diffusion operator: curl curl + sigma I
    Coefficient *muinv = new ConstantCoefficient(1.0);
    Coefficient *sigma = new ConstantCoefficient(1.0);
    ParBilinearForm *a = new ParBilinearForm(fespace);
    a->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
    a->AddDomainIntegrator(new VectorFEMassIntegrator(*sigma));
    
    a->Assemble();

    OperatorPtr A;
    Vector B, X;
    a->FormLinearSystem(ess_tdof_list_all, x, *b, A, X, B);
    
    
    // Create and configure the solver
    if (myid == 0)
    {
        cout << "Size of linear system: "
            << A.As<HypreParMatrix>()->GetGlobalNumRows() << endl;
    }

    ParFiniteElementSpace *prec_fespace =
        (a->StaticCondensationIsEnabled() ? a->SCParFESpace() : fespace);
    HypreAMS ams(*A.As<HypreParMatrix>(), prec_fespace);
    HyprePCG pcg(*A.As<HypreParMatrix>());
    pcg.SetTol(1e-12);
    pcg.SetMaxIter(500);
    pcg.SetPrintLevel(2);
    pcg.SetPreconditioner(ams);
    pcg.Mult(B, X);

    // Recover the solution
    a->RecoverFEMSolution(X, *b, x);
    

    // Save the mesh and solution for visualization
    if (visualization)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;

        // Send the solution with mesh visualization commands
        socketstream sol_sock(vishost, visport);
        sol_sock << "parallel " << num_procs << " " << myid << "\n";
        sol_sock.precision(8);
        sol_sock << "solution\n" << *pmesh << x << flush;
    }


    // Clean up
    delete a;
    delete sigma;
    delete muinv;
    delete fespace;
    delete fec;
    delete pmesh;

    return 0;
}


void f_exact(const Vector &x, Vector &f)
{
   if (dim == 3)
   {
      f(0) = 0.0;
      f(1) = 0.0;
      f(2) = 0.0;
   }
   else
   {
      f(0) = 0.0;
      f(1) = 0.0;
      if (x.Size() == 3) { f(2) = 0.0; }
   }
}