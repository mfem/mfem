// Comprehensive brute force test for GetBoundaryEdgeDoFs with all tetrahedral pairs
#include "mfem.hpp"
#include "test_utils.hpp"
#include <algorithm>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <unordered_set>

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
    const int orientation = 1;
    const int order = 1;
    const int test_num_procs = min(num_procs, 2);
    const int n_elements = 16;

    if (myid == 0) {
        cout << "=== Comprehensive Brute Force Test ===\n";
        cout << "Testing 120 tetrahedral pairs with " << test_num_procs << " processors\n";
    }

    // Statistics
    int adjacent_pairs = 0;
    int non_adjacent_pairs = 0;
    vector<int> dof_counts_per_pair;
    
    // Open output file
    ofstream outfile;
    if (myid == 0) {
        outfile.open("comprehensive_test_results.txt");
        outfile << "Comprehensive Brute Force Test Results\n";
        outfile << "Tetrahedral Pairs: 120, Processors: " << test_num_procs << "\n\n";
        outfile << "Pair_ID\tAdjacent\tPartitionings_Tested\tConsistent_Results\tDOF_Count\n";
    }

    // Create reusable FEC
    FiniteElementCollection *fec = new ND_FECollection(order, 3);

    // Test each tetrahedral pair
    for (int pair_idx = 0; pair_idx < 120; pair_idx++) 
    {
        if (myid == 0) 
        {
            cout << "Testing pair " << pair_idx << "/120\n";
        }

        // Create mesh with current pair
        Mesh *test_mesh = RefinedTetPairMesh(orientation, pair_idx);
        
        // Check if pair has internal boundary (attribute 3)
        bool has_internal_boundary = false;
        for (int i = 0; i < test_mesh->GetNBE(); ++i) 
        {
            if (test_mesh->GetBdrElement(i)->GetAttribute() == 3) 
            {
                has_internal_boundary = true;
                break;
            }
        }

        if (has_internal_boundary) {
            adjacent_pairs++;
            
            // Generate all partitionings for this pair
            vector<vector<int>> all_partitionings;
            if (myid == 0) 
            {
                GeneratePartitionings(n_elements, test_num_procs, all_partitionings);
            }
            
            int num_partitionings = all_partitionings.size();
            MPI_Bcast(&num_partitionings, 1, MPI_INT, 0, MPI_COMM_WORLD);
            
            vector<int> current_partition(n_elements);
            vector<int> all_results;
            if (myid == 0) all_results.reserve(num_partitionings);
            
            // Test each partitioning
            for (int p = 0; p < num_partitionings; p++) 
            {
                // Broadcast current partitioning
                if (myid == 0) current_partition = all_partitionings[p];
                MPI_Bcast(current_partition.data(), n_elements, MPI_INT, 0, MPI_COMM_WORLD);

                // Create parallel mesh
                ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *test_mesh, current_partition.data());
                ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);

                // Extract boundary edge DoFs
                Array<int> ess_tdof_list;
                std::unordered_set<int> boundary_edge_ldofs;
                Array<int> ldof_marker;
                Array<int> boundary_elements;
                // select the shared face to be the tested boundary (attribute 3)
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
                }

                // Cleanup
                delete fespace;
                delete pmesh;
            }
            
            // Check consistency for this pair
            bool consistent = true;
            int expected_dofs = 0;
            if (myid == 0) 
            {
                expected_dofs = all_results[0];
                for (int result : all_results) 
                {
                    if (result != expected_dofs) 
                    {
                        consistent = false;
                        break;
                    }
                }
                dof_counts_per_pair.push_back(expected_dofs);
                
                // Write to file
                outfile << pair_idx << "\tYes\t" << num_partitionings << "\t" 
                        << (consistent ? "Yes" : "No") << "\t" << expected_dofs << "\n";
            }
        } 
        else 
        {
            non_adjacent_pairs++;
            if (myid == 0) 
            {
                outfile << pair_idx << "\tNo\t0\tN/A\t0\n";
            }
        }
        
        delete test_mesh;
    }
    
    delete fec;

    // Final statistics
    if (myid == 0) 
    {
        outfile << "\n=== SUMMARY ===\n";
        outfile << "Adjacent pairs: " << adjacent_pairs << "\n";
        outfile << "Non-adjacent pairs: " << non_adjacent_pairs << "\n";
        
        if (!dof_counts_per_pair.empty()) 
        {
            // Analyze DOF count distribution
            sort(dof_counts_per_pair.begin(), dof_counts_per_pair.end());
            int min_dofs = dof_counts_per_pair.front();
            int max_dofs = dof_counts_per_pair.back();
            double avg_dofs = accumulate(dof_counts_per_pair.begin(), dof_counts_per_pair.end(), 0.0) / dof_counts_per_pair.size();
            
            outfile << "DOF count range: " << min_dofs << " - " << max_dofs << "\n";
            outfile << "Average DOF count: " << avg_dofs << "\n";
        }
        
        outfile.close();
        
        cout << "\n=== FINAL SUMMARY ===\n";
        cout << "Adjacent pairs: " << adjacent_pairs << "\n";
        cout << "Non-adjacent pairs: " << non_adjacent_pairs << "\n";
        cout << "Results saved to comprehensive_test_results.txt\n";
    }

    return 0;
}