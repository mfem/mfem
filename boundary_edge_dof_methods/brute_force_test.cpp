// Brute force test for GetBoundaryEdgeDoFs with all possible partitionings
#include "mfem.hpp"
#include "test_utils.hpp"
#include <algorithm>
#include <vector>
#include <fstream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
    // Initialize MPI
    Mpi::Init(argc, argv);
    int myid = Mpi::WorldRank();
    int num_procs = Mpi::WorldSize();
    Hypre::Init();

    // Parameters
    const int orientation = 3;
    const int order = 1;
    const int test_num_procs = min(num_procs, 2); // Reduced to 2 for faster testing

    if (myid == 0) {
        cout << "=== Brute Force Test for GetBoundaryEdgeDoFs ===\n";
        cout << "Testing with " << test_num_procs << " processors\n";
    }

    // Create base mesh once
    Mesh *base_mesh = new Mesh(OrientedTriFaceMesh(orientation, true));
    base_mesh->UniformRefinement();
    const int n_elements = base_mesh->GetNE();
    
    if (myid == 0) {
        cout << "Base mesh has " << n_elements << " elements\n";
    }

    // Generate all partitionings on rank 0
    vector<vector<int>> all_partitionings;
    int num_partitionings = 0;
    if (myid == 0) {
        GeneratePartitionings(n_elements, test_num_procs, all_partitionings);
        num_partitionings = all_partitionings.size();
        cout << "Generated " << num_partitionings << " partitionings\n";
    }
    MPI_Bcast(&num_partitionings, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Pre-allocate reusable objects
    vector<int> current_partition(n_elements);
    vector<int> all_results;
    if (myid == 0) all_results.reserve(num_partitionings);
    
    // Create reusable FEC (same for all tests)
    FiniteElementCollection *fec = new ND_FECollection(order, 3);
    
    // Open output file on rank 0
    ofstream outfile;
    if (myid == 0) {
        outfile.open("partitioning_results.txt");
        outfile << "Brute Force Test Results for GetBoundaryEdgeDoFs\n";
        outfile << "Elements: " << n_elements << ", Processors: " << test_num_procs << "\n";
        outfile << "Total Partitionings: " << num_partitionings << "\n\n";
        outfile << "Partitioning_ID\tPartition_Array\tBoundary_Edge_DOFs\n";
    }
    
    // Test each partitioning
    for (int p = 0; p < num_partitionings; p++) 
    {
        // Broadcast current partitioning
        if (myid == 0) current_partition = all_partitionings[p];
        MPI_Bcast(current_partition.data(), n_elements, MPI_INT, 0, MPI_COMM_WORLD);

        // Create parallel mesh with current partitioning
        Mesh *test_mesh = new Mesh(OrientedTriFaceMesh(orientation, true));
        test_mesh->UniformRefinement();
        ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *test_mesh, current_partition.data());
        delete test_mesh;

        // Create finite element space
        ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);

        // Extract boundary edge DoFs
        Array<int> ess_tdof_list;
        std::unordered_set<int> boundary_edge_ldofs;
        Array<int> ldof_marker;
        Array<int> boundary_elements;
        // select the shared face to be the tested boundary
        int bdr_attr = pmesh->bdr_attributes.Max();  
        
        fespace->GetBoundaryElementsByAttribute(bdr_attr, boundary_elements);
        fespace->GetBoundaryEdgeDoFs(boundary_elements, ess_tdof_list, ldof_marker, 
                                    boundary_edge_ldofs);
                                    
        // Collect total boundary edge DoFs
        int local_dofs = boundary_edge_ldofs.size();
        int total_dofs;
        MPI_Allreduce(&local_dofs, &total_dofs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (myid == 0) 
        {
            all_results.push_back(total_dofs);
            cout << "Partitioning " << p << ": Total boundary edge DOFs = " << total_dofs << "\n";
            
            // Write to file
            outfile << p << "\t[";
            for (int i = 0; i < n_elements; i++) 
            {
                outfile << current_partition[i];
                if (i < n_elements - 1) outfile << ",";
            }
            outfile << "]\t" << total_dofs << "\n";
        }
        
        // Cleanup
        delete fespace;
        delete pmesh;
    }
    
    // Cleanup reusable objects
    delete fec;

    // Summary
    if (myid == 0) {
        cout << "\n=== Summary ===" << endl;
        
        // Check if all results are the same
        bool all_same = true;
        int expected = all_results[0];
        for (int result : all_results) 
        {
            if (result != expected) 
            {
                all_same = false;
                break;
            }
        }
        
        // Write summary to file
        outfile << "\n=== SUMMARY ===\n";
        if (all_same) 
        {
            cout << "SUCCESS: All partitionings gave the same result: " << expected << " DOFs" << endl;
            outfile << "RESULT: SUCCESS\n";
            outfile << "All partitionings gave the same result: " << expected << " DOFs\n";
        } 
        else 
        {
            cout << "FAILURE: Different partitionings gave different results!" << endl;
            cout << "Results: ";
            outfile << "RESULT: FAILURE\n";
            outfile << "Different partitionings gave different results!\n";
            outfile << "Results: ";
            for (int result : all_results) 
            {
                cout << result << " ";
                outfile << result << " ";
            }
            cout << endl;
            outfile << "\n";
        }
        
        outfile.close();
        cout << "Results saved to partitioning_results.txt" << endl;
    }
    
    // Cleanup base mesh
    delete base_mesh;

    return 0;
}