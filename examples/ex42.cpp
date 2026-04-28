//                                MFEM Example 42
//
// Compile with: make ex42
//
// Sample runs:  ex42
//               ex42 -m ../data/beam-quad.mesh -o 2 -r 1
//               ex42 -m ../data/beam-hex.mesh -o 3 -upper
//
// Device sample runs:
//               ex42 -d cuda
//               ex42 -m ../data/beam-hex.mesh -o 2 -d cuda -upper
//
// Description:  This example assembles the L2 mass matrix using the positive
//               (Bernstein) tensor-product basis, stores either the lower or
//               upper triangular portion in a packed container, reconstructs
//               the full element matrices, and checks the result against the
//               existing dense element assembly.

#include "mfem.hpp"
#include <cmath>
#include <iostream>
#include <limits>

using namespace mfem;
using namespace std;

namespace
{

template <TriangularPart PART>
void UnpackTriangularEA(const TriPackMatrix<PART> &tri, Vector &full)
{
   const int ndofs = tri.GetNumRows();
   const int ne = tri.GetNumMatrices();
   const int tri_sz = tri.GetPackedSize();

   full.SetSize(ne*ndofs*ndofs);
   full = 0.0;

   const real_t *src = tri.Data().HostRead();
   real_t *dst = full.HostWrite();
   for (int e = 0; e < ne; ++e)
   {
      const int eoff_tri = e*tri_sz;
      const int eoff_full = e*ndofs*ndofs;
      for (int j = 0; j < ndofs; ++j)
      {
         for (int i = 0; i < ndofs; ++i)
         {
            const int a = (PART == TriangularPart::LOWER) ? max(i, j)
                                                          : min(i, j);
            const int b = (PART == TriangularPart::LOWER) ? min(i, j)
                                                          : max(i, j);
            const int tidx = TriPackMatrix<PART>::Index(a, b, ndofs, PART);
            dst[eoff_full + i + ndofs*j] = src[eoff_tri + tidx];
         }
      }
   }
}

real_t MaxError(const Vector &a, const Vector &b)
{
   MFEM_VERIFY(a.Size() == b.Size(), "Incompatible vector sizes.");
   const real_t *pa = a.HostRead();
   const real_t *pb = b.HostRead();
   real_t err = 0.0;
   for (int i = 0; i < a.Size(); ++i)
   {
      err = max(err, fabs(pa[i] - pb[i]));
   }
   return err;
}

void PackUpperFromFull(const Vector &full, const int ndofs,
                       TriPackMatrix<TriangularPart::UPPER> &upper)
{
   MFEM_VERIFY(full.Size() % (ndofs*ndofs) == 0, "Invalid full EA size.");
   const int ne = full.Size() / (ndofs*ndofs);

   upper.SetSize(ndofs, ne);
   upper.UseDevice(true);

   const real_t *src = full.HostRead();
   real_t *dst = upper.Data().HostWrite();
   const int packed_size = upper.GetPackedSize();
   for (int e = 0; e < ne; ++e)
   {
      const int eoff_full = e*ndofs*ndofs;
      const int eoff_tri = e*packed_size;
      for (int j = 0; j < ndofs; ++j)
      {
         for (int i = 0; i <= j; ++i)
         {
            dst[eoff_tri + TriPackMatrix<TriangularPart::UPPER>::UpperIndex(i, j, ndofs)] =
               src[eoff_full + i + ndofs*j];
         }
      }
   }
}

}

int main(int argc, char *argv[])
{
   const char *mesh_file = "../data/beam-quad.mesh";
   int order = 2;
   int ref_levels = 1;
   bool upper = false;
   const char *device_config = "cpu";
   bool visualization = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of uniform refinements.");
   args.AddOption(&upper, "-upper", "--upper-triangular",
                  "-lower", "--lower-triangular",
                  "Store the upper or lower triangular portion.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable visualization (unused).");
   args.ParseCheck();

   Device device(device_config);
   device.Print();

   Mesh mesh(mesh_file, 1, 1);
   for (int l = 0; l < ref_levels; ++l)
   {
      mesh.UniformRefinement();
   }

   const int dim = mesh.Dimension();
   L2_FECollection fec(order, dim, BasisType::Positive);
   FiniteElementSpace fespace(&mesh, &fec);

   MFEM_VERIFY(UsesTensorBasis(fespace),
               "This example requires a tensor-product finite element space.");
   MFEM_VERIFY(fec.GetBasisType() == BasisType::Positive,
               "This example requires the positive L2 basis.");

   const int ne = mesh.GetNE();
   const int elem_dofs = fespace.GetTypicalFE()->GetDof();

   MassIntegrator mass;

   Vector full_ea(ne*elem_dofs*elem_dofs);
   full_ea.UseDevice(true);
   mass.AssembleEA(fespace, full_ea, false);

   Vector unpacked_ea;
   const real_t tol = 256.0*numeric_limits<real_t>::epsilon();
   bool packed_matches = false;
   real_t err = 0.0;

   TriPackMatrix<TriangularPart::UPPER> upper_ea;
   TriPackMatrix<TriangularPart::LOWER> lower_ea;
   if (upper)
   {
      mass.AssembleEATriangular(fespace, upper_ea, false);
      UnpackTriangularEA(upper_ea, unpacked_ea);
      packed_matches = tripack::CompareWithFull(upper_ea, full_ea, tol);
      err = MaxError(full_ea, unpacked_ea);
   }
   else
   {
      mass.AssembleEATriangular(fespace, lower_ea, false);
      UnpackTriangularEA(lower_ea, unpacked_ea);
      packed_matches = tripack::CompareWithFull(lower_ea, full_ea, tol);
      err = MaxError(full_ea, unpacked_ea);
   }

   PackUpperFromFull(full_ea, elem_dofs, upper_ea);

   Vector rhs(ne*elem_dofs);
   real_t *rhs_data = rhs.HostWrite();
   for (int e = 0; e < ne; ++e)
   {
      for (int i = 0; i < elem_dofs; ++i)
      {
         rhs_data[e*elem_dofs + i] = 1.0 + i + e % 3;
      }
   }

   TriPackMatrix<TriangularPart::UPPER> upper_factor;
   TriPackMatrix<TriangularPart::UPPER> upper_inverse;
   Vector solve_y, inverse_y, dense_y(ne*elem_dofs);

   tripack::ComputeJacobiScaledCholeskyUpper(upper_ea, upper_factor);
   tripack::SolveCholesky(upper_factor, rhs, solve_y);
   tripack::ComputeJacobiScaledCholeskyUpperInverse(upper_ea, upper_inverse);
   tripack::MultUUt(upper_inverse, rhs, inverse_y);

   real_t *dense_y_data = dense_y.HostWrite();
   const real_t *full_data = full_ea.HostRead();
   for (int e = 0; e < ne; ++e)
   {
      DenseMatrix elmat(elem_dofs);
      for (int j = 0; j < elem_dofs; ++j)
      {
         for (int i = 0; i < elem_dofs; ++i)
         {
            elmat(i, j) = full_data[e*elem_dofs*elem_dofs + i + elem_dofs*j];
         }
      }
      DenseMatrixInverse inv(elmat, true);
      inv.Mult(rhs_data + e*elem_dofs, dense_y_data + e*elem_dofs);
   }

   const real_t solve_err = MaxError(solve_y, dense_y);
   const real_t inverse_err = MaxError(inverse_y, dense_y);

   cout << "Number of elements: " << ne << '\n';
   cout << "Element dofs: " << elem_dofs << '\n';
   cout << "Basis: positive L2" << '\n';
   cout << "Triangular part: "
        << (upper ? "upper" : "lower") << '\n';
   cout << "Full entries/element: " << elem_dofs*elem_dofs << '\n';
   cout << "Packed entries/element: " << upper_ea.GetPackedSize() << '\n';
   cout << "Packed storage ratio: "
        << double(upper_ea.GetPackedSize())/(elem_dofs*elem_dofs) << '\n';
   cout << "Packed/full comparison: " << (packed_matches ? "ok" : "failed") << '\n';
   cout << "Max reconstruction error: " << err << '\n';
   cout << "Max Cholesky solve error: " << solve_err << '\n';
   cout << "Max inverse apply error: " << inverse_err << endl;

   MFEM_VERIFY(packed_matches, "Packed triangular EA does not match full EA.");
   MFEM_VERIFY(err <= tol, "Packed triangular EA does not match full EA.");
   MFEM_VERIFY(solve_err <= 1024.0*tol, "Packed Cholesky solve does not match dense inverse.");
   MFEM_VERIFY(inverse_err <= 1024.0*tol, "Packed inverse apply does not match dense inverse.");
   return 0;
}
