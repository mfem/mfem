//                                MFEM Example 42
//
// Compile with: make ex42
//
// Sample runs:  ex42
//               ex42 -m ../data/beam-quad.mesh -o 2 -r 1
//               ex42 -m ../data/beam-hex.mesh -o 3 -upper
//               ex42 -bench -o 3 -brmin 0 -brmax 5 -breps 20
//
// Device sample runs:
//               ex42 -d cuda
//               ex42 -m ../data/beam-hex.mesh -o 2 -d cuda -upper
//               ex42 -bench -d cuda -o 3 -brmin 0 -brmax 5 -breps 20
//               ex42 -magma -d hip -bench -o 3 -brmin 0 -brmax 5 -breps 20
//
// Description:  This example assembles the L2 mass matrix using the positive
//               (Bernstein) tensor-product basis, stores either the lower or
//               upper triangular portion in a packed container, reconstructs
//               the full element matrices, and checks the result against the
//               existing dense element assembly. It can also use MAGMA to
//               compute the Cholesky factorization of the packed lower matrix
//               or benchmark native MFEM against MAGMA over a range of mesh
//               refinement levels.

#include "mfem.hpp"
#ifdef MFEM_USE_MAGMA
#include "linalg/batched/magma.hpp"
#endif
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

using namespace mfem;
using namespace std;

namespace
{

#ifdef MFEM_USE_MAGMA
#ifdef MFEM_USE_SINGLE
#define MFEM_MAGMA_PREFIX(stub) magma_s##stub
#elif defined(MFEM_USE_DOUBLE)
#define MFEM_MAGMA_PREFIX(stub) magma_d##stub
#endif

void ComputeMagmaPackedCholeskyLower(
   const TriPackMatrix<TriangularPart::LOWER> &packed_lower,
   TriPackMatrix<TriangularPart::LOWER> &lower_factor)
{
   const int n = packed_lower.GetNumRows();
   const int batch_size = packed_lower.GetNumMatrices();
   const int packed_size = packed_lower.GetPackedSize();

   lower_factor.SetSize(n, batch_size);
   lower_factor.UseDevice(true);

   if (batch_size == 0) { return; }

   // Ensure MAGMA is initialized before calling the batched factorization.
   magma_queue_t queue = Magma::Queue();

   lower_factor.Data() = packed_lower.Data();

   Array<real_t*> factor_ptrs(batch_size, Device::GetDeviceMemoryType());
   real_t **d_factor_ptrs = factor_ptrs.Write();
   real_t *factor_data = lower_factor.Data().ReadWrite();
   MFEM_MAGMA_PREFIX(set_pointer)(d_factor_ptrs, factor_data, 1, 0, 0,
                                  packed_size, batch_size, queue);

   Array<magma_int_t> info_array(batch_size);
   magma_int_t *d_info = info_array.Write();
   magma_memset(d_info, 0, batch_size*sizeof(magma_int_t));
   const magma_int_t status =
      (n <= 8) ?
      MFEM_MAGMA_PREFIX(pptrf_batched_small)(
         MagmaLower, n, d_factor_ptrs, d_info, batch_size, queue) :
      MFEM_MAGMA_PREFIX(pptf2_batched_small)(
         MagmaLower, n, d_factor_ptrs, d_info, batch_size, queue);
   MFEM_VERIFY(status == MAGMA_SUCCESS,
               "MAGMA packed Cholesky factorization failed.");

   magma_queue_sync(queue);
   const magma_int_t *info = info_array.HostRead();
   for (int e = 0; e < batch_size; ++e)
   {
      MFEM_VERIFY(info[e] == 0,
                  "MAGMA packed Cholesky factorization failed on matrix " << e << '.');
   }
}
#endif

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

void PackLowerFromFull(const Vector &full, const int ndofs,
                       TriPackMatrix<TriangularPart::LOWER> &lower)
{
   MFEM_VERIFY(full.Size() % (ndofs*ndofs) == 0, "Invalid full EA size.");
   const int ne = full.Size() / (ndofs*ndofs);

   lower.SetSize(ndofs, ne);
   lower.UseDevice(true);

   const real_t *src = full.HostRead();
   real_t *dst = lower.Data().HostWrite();
   const int packed_size = lower.GetPackedSize();
   for (int e = 0; e < ne; ++e)
   {
      const int eoff_full = e*ndofs*ndofs;
      const int eoff_tri = e*packed_size;
      for (int j = 0; j < ndofs; ++j)
      {
         for (int i = j; i < ndofs; ++i)
         {
            dst[eoff_tri + TriPackMatrix<TriangularPart::LOWER>::LowerIndex(i, j, ndofs)] =
               src[eoff_full + i + ndofs*j];
         }
      }
   }
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

void BuildUpperDense(const TriPackMatrix<TriangularPart::UPPER> &packed,
                     int e, DenseMatrix &mat)
{
   const int n = packed.GetNumRows();
   mat.SetSize(n);
   mat = 0.0;

   const real_t *data = packed.Data().HostRead() + e*packed.GetPackedSize();
   for (int j = 0; j < n; ++j)
   {
      for (int i = 0; i <= j; ++i)
      {
         mat(i, j) = data[TriPackMatrix<TriangularPart::UPPER>::UpperIndex(i, j, n)];
      }
   }
}

void BuildLowerDense(const TriPackMatrix<TriangularPart::LOWER> &packed,
                     int e, DenseMatrix &mat)
{
   const int n = packed.GetNumRows();
   mat.SetSize(n);
   mat = 0.0;

   const real_t *data = packed.Data().HostRead() + e*packed.GetPackedSize();
   for (int j = 0; j < n; ++j)
   {
      for (int i = j; i < n; ++i)
      {
         mat(i, j) = data[TriPackMatrix<TriangularPart::LOWER>::LowerIndex(i, j, n)];
      }
   }
}

void CheckCholeskyUpperFactorization(const TriPackMatrix<TriangularPart::UPPER> &factor,
                                     const Vector &full_matrix,
                                     real_t tol)
{
   const int n = factor.GetNumRows();
   const int batch_size = factor.GetNumMatrices();
   MFEM_VERIFY(full_matrix.Size() == batch_size*n*n,
               "Full matrix has the wrong size.");

   const real_t *full_data = full_matrix.HostRead();
   for (int e = 0; e < batch_size; ++e)
   {
      DenseMatrix U, recon, orig;
      recon.SetSize(n);
      orig.SetSize(n);
      BuildUpperDense(factor, e, U);
      MultAtB(U, U, recon);

      const int eoff = e*n*n;
      for (int j = 0; j < n; ++j)
      {
         for (int i = 0; i < n; ++i)
         {
            orig(i, j) = full_data[eoff + i + n*j];
         }
      }

      recon -= orig;
      MFEM_VERIFY(recon.MaxMaxNorm() <= tol,
                  "Upper Cholesky factorization check failed for element " << e << '.');
   }
}

void MultLtL(const TriPackMatrix<TriangularPart::LOWER> &packed_lower,
             const Vector &x, Vector &y)
{
   const int n = packed_lower.GetNumRows();
   const int batch_size = packed_lower.GetNumMatrices();
   const int packed_size = packed_lower.GetPackedSize();
   MFEM_VERIFY(x.Size() == batch_size*n, "Input vector has the wrong size.");

   Vector t(batch_size*n);
   t.UseDevice(true);
   y.SetSize(batch_size*n);
   y.UseDevice(true);

   const real_t *L = packed_lower.Data().Read();
   const real_t *X = x.Read();
   real_t *T = t.Write();
   real_t *Y = y.Write();

   mfem::forall(batch_size*n, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int i = idx % n;
      const int e = idx / n;
      const real_t *Le = L + e*packed_size;
      const real_t *Xe = X + e*n;
      real_t sum = 0.0;
      for (int j = 0; j <= i; ++j)
      {
         sum += Le[TriPackMatrix<TriangularPart::LOWER>::LowerIndex(i, j, n)] * Xe[j];
      }
      T[idx] = sum;
   });

   mfem::forall(batch_size*n, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int i = idx % n;
      const int e = idx / n;
      const real_t *Le = L + e*packed_size;
      const real_t *Te = T + e*n;
      real_t sum = 0.0;
      for (int j = i; j < n; ++j)
      {
         sum += Le[TriPackMatrix<TriangularPart::LOWER>::LowerIndex(j, i, n)] * Te[j];
      }
      Y[idx] = sum;
   });
}

void CheckCholeskyLowerFactorization(const TriPackMatrix<TriangularPart::LOWER> &factor,
                                     const Vector &full_matrix,
                                     real_t tol)
{
   const int n = factor.GetNumRows();
   const int batch_size = factor.GetNumMatrices();
   MFEM_VERIFY(full_matrix.Size() == batch_size*n*n,
               "Full matrix has the wrong size.");

   const real_t *full_data = full_matrix.HostRead();
   for (int e = 0; e < batch_size; ++e)
   {
      DenseMatrix L, recon, orig;
      recon.SetSize(n);
      orig.SetSize(n);
      BuildLowerDense(factor, e, L);
      MultABt(L, L, recon);

      const int eoff = e*n*n;
      for (int j = 0; j < n; ++j)
      {
         for (int i = 0; i < n; ++i)
         {
            orig(i, j) = full_data[eoff + i + n*j];
         }
      }

      recon -= orig;
      MFEM_VERIFY(recon.MaxMaxNorm() <= tol,
                  "Cholesky factorization check failed for element " << e << '.');
   }
}

}

int main(int argc, char *argv[])
{
   const char *mesh_file = "../data/beam-quad.mesh";
   int order = 2;
   int ref_levels = 1;
   bool upper = false;
   bool use_magma = false;
   bool benchmark = false;
   int bench_ref_min = 0;
   int bench_ref_max = 5;
   int bench_ref_step = 1;
   int bench_reps = 10;
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
   args.AddOption(&use_magma, "-magma", "--magma-cholesky",
                  "-tripack", "--tripack-cholesky",
                  "Use MAGMA packed batched Cholesky factorization.");
   args.AddOption(&benchmark, "-bench", "--benchmark",
                  "-no-bench", "--no-benchmark",
                  "Benchmark native MFEM and MAGMA Cholesky decompositions.");
   args.AddOption(&bench_ref_min, "-brmin", "--bench-ref-min",
                  "Minimum uniform refinement level for benchmark mode.");
   args.AddOption(&bench_ref_max, "-brmax", "--bench-ref-max",
                  "Maximum uniform refinement level for benchmark mode.");
   args.AddOption(&bench_ref_step, "-brstep", "--bench-ref-step",
                  "Uniform refinement step for benchmark mode.");
   args.AddOption(&bench_reps, "-breps", "--bench-reps",
                  "Number of repetitions per benchmark point.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable visualization (unused).");
   args.ParseCheck();

   Device device(device_config);
   device.Print();

#ifndef MFEM_USE_MAGMA
   MFEM_VERIFY(!use_magma, "MFEM was built without MAGMA support.");
#endif

   if (benchmark)
   {
      MFEM_VERIFY(bench_ref_step > 0, "Benchmark refinement step must be positive.");
      MFEM_VERIFY(bench_ref_min <= bench_ref_max,
                  "Benchmark refinement range is empty.");
      MFEM_VERIFY(bench_reps > 0, "Benchmark repetitions must be positive.");

      StopWatch chrono;
      const real_t tol = 256.0*numeric_limits<real_t>::epsilon();
      cout << fixed << setprecision(3);

      cout << "Benchmarking lower Cholesky factorization and inverse" << '\n';
      cout << "Order: " << order << '\n';
      cout << "Refinement range: " << bench_ref_min << ".." << bench_ref_max
           << " step " << bench_ref_step << '\n';
      cout << "Repetitions: " << bench_reps << '\n';
#ifdef MFEM_USE_MAGMA
      cout << left << setw(8) << "ref"
           << setw(10) << "NE"
           << setw(12) << "dofs/elem"
           << setw(12) << "total dofs"
           << setw(16) << "native(ms)"
           << setw(30) << "JacobiScaledCholeskyUpper(ms)"
           << setw(35) << "JacobiScaledCholeskyLowerInv(ms)"
           << setw(18) << "magma-packed(ms)"
           << setw(28) << "speedup vs JacobiScaledCholeskyUpper" << '\n';
#else
      cout << left << setw(8) << "ref"
           << setw(10) << "NE"
           << setw(12) << "dofs/elem"
           << setw(12) << "total dofs"
           << setw(16) << "native(ms)"
           << setw(30) << "JacobiScaledCholeskyUpper(ms)"
           << setw(35) << "JacobiScaledCholeskyLowerInv(ms)" << '\n';
#endif

      for (int ref = bench_ref_min; ref <= bench_ref_max; ref += bench_ref_step)
      {
         Mesh mesh(mesh_file, 1, 1);
         for (int l = 0; l < ref; ++l)
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

         TriPackMatrix<TriangularPart::LOWER> lower_ea;
         TriPackMatrix<TriangularPart::UPPER> upper_ea;
         PackLowerFromFull(full_ea, elem_dofs, lower_ea);
         PackUpperFromFull(full_ea, elem_dofs, upper_ea);

         TriPackMatrix<TriangularPart::LOWER> lower_factor;
         TriPackMatrix<TriangularPart::UPPER> upper_factor;
         TriPackMatrix<TriangularPart::LOWER> lower_inverse;

         // Time the actual factorization kernels only; matrix assembly and packing
         // are kept outside the timed region.
         chrono.Restart();
         for (int rep = 0; rep < bench_reps; ++rep)
         {
            tripack::ComputeCholeskyLower(lower_ea, lower_factor);
            MFEM_DEVICE_SYNC;
         }
         chrono.Stop();
         const double native_ms = 1000.0*chrono.RealTime()/bench_reps;
         CheckCholeskyLowerFactorization(lower_factor, full_ea, 1024.0*tol);

         chrono.Restart();
         for (int rep = 0; rep < bench_reps; ++rep)
         {
            tripack::ComputeJacobiScaledCholeskyUpper(upper_ea, upper_factor);
            MFEM_DEVICE_SYNC;
         }
         chrono.Stop();
         const double upper_ms = 1000.0*chrono.RealTime()/bench_reps;
         CheckCholeskyUpperFactorization(upper_factor, full_ea, 1024.0*tol);

         chrono.Restart();
         for (int rep = 0; rep < bench_reps; ++rep)
         {
            tripack::ComputeJacobiScaledCholeskyLowerInverse(lower_ea, lower_inverse);
            MFEM_DEVICE_SYNC;
         }
         chrono.Stop();
         const double inverse_ms = 1000.0*chrono.RealTime()/bench_reps;

#ifdef MFEM_USE_MAGMA
         TriPackMatrix<TriangularPart::LOWER> magma_factor;
         chrono.Restart();
         for (int rep = 0; rep < bench_reps; ++rep)
         {
            ComputeMagmaPackedCholeskyLower(lower_ea, magma_factor);
            MFEM_DEVICE_SYNC;
         }
         chrono.Stop();
         const double magma_ms = 1000.0*chrono.RealTime()/bench_reps;
         CheckCholeskyLowerFactorization(magma_factor, full_ea, 1024.0*tol);

         cout << left << setw(8) << ref
              << setw(10) << ne
              << setw(12) << elem_dofs
              << setw(12) << ne*elem_dofs
              << setw(16) << native_ms
              << setw(30) << upper_ms
              << setw(35) << inverse_ms
              << setw(18) << magma_ms
              << setw(28) << (magma_ms > 0.0 ? upper_ms/magma_ms : 0.0)
              << '\n';
#else
         cout << left << setw(8) << ref
              << setw(10) << ne
              << setw(12) << elem_dofs
              << setw(12) << ne*elem_dofs
              << setw(16) << native_ms
              << setw(16) << upper_ms
              << setw(16) << inverse_ms
              << '\n';
#endif
      }
      return 0;
   }

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

   PackLowerFromFull(full_ea, elem_dofs, lower_ea);

   Vector rhs(ne*elem_dofs);
   real_t *rhs_data = rhs.HostWrite();
   for (int e = 0; e < ne; ++e)
   {
      for (int i = 0; i < elem_dofs; ++i)
      {
         rhs_data[e*elem_dofs + i] = 1.0 + i + e % 3;
      }
   }

   TriPackMatrix<TriangularPart::LOWER> lower_factor;
   TriPackMatrix<TriangularPart::LOWER> lower_inverse;
   Vector solve_y, inverse_y, dense_y(ne*elem_dofs);

   if (use_magma)
   {
#ifdef MFEM_USE_MAGMA
      ComputeMagmaPackedCholeskyLower(lower_ea, lower_factor);
#endif
   }
   else
   {
      tripack::ComputeCholeskyLower(lower_ea, lower_factor);
   }
   tripack::SolveCholeskyLower(lower_factor, rhs, solve_y);
   tripack::ComputeJacobiScaledCholeskyLowerInverse(lower_ea, lower_inverse);
   MultLtL(lower_inverse, rhs, inverse_y);

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
   cout << "Cholesky backend: "
        << (use_magma ? "MAGMA packed" : "MFEM tripack") << '\n';
   cout << "Full entries/element: " << elem_dofs*elem_dofs << '\n';
   cout << "Packed entries/element: " << lower_ea.GetPackedSize() << '\n';
   cout << "Packed storage ratio: "
        << double(lower_ea.GetPackedSize())/(elem_dofs*elem_dofs) << '\n';
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
