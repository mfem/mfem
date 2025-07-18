//                                MFEM Example 0
//
// Compile with: make ex0
//
// Sample runs:  ex0
//               ex0 -m ../data/fichera.mesh
//               ex0 -m ../data/square-disc.mesh -o 2
//
// Description: This example code demonstrates the most basic usage of MFEM to
//              define a simple finite element discretization of the Laplace
//              problem -Delta u = 1 with zero Dirichlet boundary conditions.
//              General 2D/3D mesh files and finite element polynomial degrees
//              can be specified by command line options.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "remap.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   Mpi::Init();
   // int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   MPI_Comm comm = MPI_COMM_WORLD;
   if (myid) { out.Disable(); }
   //
   // 1. Parse command line options.
   // int dim = 2;
   int order = 3;
   int qorder = 4;
   int ref_levels = 0;
   int optType = 3;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&qorder, "-qo", "--quad-order", "Quadrature order");
   // args.AddOption(&dim, "-d", "--dim", "Mesh dimension (2 or 3)");
   args.AddOption(&ref_levels, "-r", "--refine", "Mesh refinement levels");
   args.AddOption(&optType, "-opt", "--opt-type",
                  "Type of remap operator:\n"
                  "\t0: eta (QF)\n"
                  "\t1: eta (QF), rho (QF)\n"
                  "\t2: eta (QF), rho (QF), e (L2 GF)\n"
                  "\t3: eta (QF), rho (QF), e (L2 GF), v (H1 GF)\n");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   // Mesh mesh = dim == 2
   //             ? Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL)
   //             : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
   Mesh ser_mesh("../../data/mobius-strip.mesh", 1, 1);
   const int dim = ser_mesh.Dimension();
   for (int i=0; i<ref_levels; i++)
   {
      ser_mesh.UniformRefinement();
   }
   ParMesh mesh(comm, ser_mesh);

   // Get Remap functionals
   std::vector<std::function<real_t(const Vector &)>> f;
   std::vector<std::function<void(const Vector &, Vector &)>> df;
   Array<int> space_idx;
   remap::remap_functionals(optType, dim, f, df, space_idx);

   const int numVars = space_idx.Size();
   const int numConst = f.size();

   QuadratureSpace qspace(&mesh, qorder);
   L2_FECollection l2_fec(order, dim, BasisType::Positive);
   H1_FECollection h1_fec(order, dim);
   ParFiniteElementSpace l2_fespace(&mesh, &l2_fec);
   ParFiniteElementSpace h1_fespace(&mesh, &h1_fec);
   std::vector<ParFiniteElementSpace*> fes(0);
   if (optType >= 2) { fes.push_back(&l2_fespace); }
   if (optType == 3) { fes.push_back(&h1_fespace); }

   // Compute BlockVector offsets (T-Vector)
   Array<int> offsets(0);
   offsets.Append(0);
   for (int vid=0; vid<space_idx.Size(); vid++)
   {
      const int sid = space_idx[vid];
      if (sid < 0) { offsets.Append(qspace.GetSize()); }
      else { offsets.Append(fes[sid]->GetTrueVSize()); }
   }
   offsets.PartialSum();

   // Print info
   const int numDofs = offsets.Last();
   HYPRE_BigInt global_numDofs = offsets.Last();
   MPI_Allreduce(MPI_IN_PLACE, &global_numDofs, 1,
                 MPITypeMap<HYPRE_BigInt>::mpi_type, MPI_SUM, comm);

   out << "Number of quadrature points: " << qspace.GetSize() << "\n";
   out << "Number of L2 unknowns: " << l2_fespace.GetTrueVSize() << "\n";
   out << "Number of H1 unknowns: " << h1_fespace.GetTrueVSize() << "\n";
   out << "Space index: "; space_idx.Print(out, space_idx.Size());
   out << "Total number of unknowns: " << global_numDofs << "\n";
   out << std::endl;

   /// Stack constraints
   StackedFunctional constraints(numDofs);
   std::vector<std::unique_ptr<ComposedFunctional>> C_vec(numConst);
   for (int i=0; i<numConst; i++)
   {
      C_vec[i] = std::make_unique<ComposedFunctional>(
                    f[i], df[i], qspace, fes, space_idx);
      constraints.AddFunctional(*C_vec[i]);
   }

   BlockVector x(offsets);

   Vector constraints_val(numConst);
   constraints.Mult(x, constraints_val);

   DenseMatrix constraint_grads(numDofs, numConst);
   constraints.GetGradientMatrix(x, constraint_grads);

   return 0;
}
