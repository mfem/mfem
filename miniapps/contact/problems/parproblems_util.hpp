
#include "mfem.hpp"
#include "problems_util.hpp"
#include "../util/mpicomm.hpp"

// Coordinates in xyz are assumed to be ordered as [X, Y, Z]
// where X is the list of x-coordinates for all points and so on.
// conn: connectivity of the target surface elements
// xi: surface reference cooridnates for the cloest point, involves a linear transformation from [0,1] to [-1,1]
void FindPointsInMesh(Mesh & mesh, const Array<int> & gvert, const Vector & xyz, const Array<int> & s_conn, Array<int>& conn,
                      Vector & xyz2, Array<int> & s_conn2, Vector& xi, DenseMatrix & coords);

// somewhat simplified version of the above
void FindPointsInMesh(Mesh & mesh, const Array<int> & gvert, Array<int> & s_conn, const ParGridFunction &x1, Vector & xyz, Array<int>& conn,
                      Vector& xi, DenseMatrix & coords);                   