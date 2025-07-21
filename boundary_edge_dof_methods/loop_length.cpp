#include "mfem.hpp"
#include <unordered_map>
#include <unordered_set>

namespace mfem {

double ComputeBoundaryLoopLength(ParMesh* pmesh, const std::unordered_map<int, int>& dof_to_edge)
{
    // Calculate local boundary loop length
    double local_length = 0.0;
    std::unordered_set<int> processed_edges;
    
    // Process each edge in the dof_to_edge map
    for (const auto& pair : dof_to_edge) 
    {
        int edge_id = pair.second;
        
        // Skip if already processed this edge
        if (!processed_edges.insert(edge_id).second) continue;
        
        // Get edge vertices
        Array<int> edge_verts;
        pmesh->GetEdgeVertices(edge_id, edge_verts);
        
        // Calculate edge length
        const double* v0 = pmesh->GetVertex(edge_verts[0]);
        const double* v1 = pmesh->GetVertex(edge_verts[1]);
        
        double edge_length = 0.0;
        for (int i = 0; i < pmesh->SpaceDimension(); i++) 
        {
            double diff = v1[i] - v0[i];
            edge_length += diff * diff;
        }
        edge_length = sqrt(edge_length);
        
        local_length += edge_length;
    }
    
    return local_length;  // Return local portion of the boundary loop length
}

} // namespace mfem