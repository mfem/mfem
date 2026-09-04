//                    MFEM Batched Mass Benchmark
//
// Compile with: make batched_mass_bench
//
// Sample runs:  batched_mass_bench -m ../../data/beam-hex.mesh -o 2 -r 2 -d hip -reps 100
//               batched_mass_bench -m ../../data/beam-hex.mesh -o 2 -r 3 -d cuda -reps 50
//
// Description:  This example isolates the element mass-matrix
//               inverse paths used by element matrix kernels. It
//               assembles packed lower-triangular L2 element mass matrices,
//               then times repeated mass inverse applications using MFEM's
//               tripack path and MAGMA (packed and full-matrix variants).

#include "mfem.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace mfem;
using namespace std;

namespace
{

struct MethodTiming
{
   std::string label;
   bool available = false;
   double assembly_ms = 0.0;
   double setup_ms = 0.0;
   double apply_ms = 0.0; // ms/apply
   double rel_max = std::numeric_limits<double>::quiet_NaN();
   double rel_l2 = std::numeric_limits<double>::quiet_NaN();
};

static void PrintTimingTable(const double packed_assembly_ms,
                             const double full_assembly_ms,
                             const int reps,
                             const std::vector<MethodTiming> &methods)
{
   cout << '\n';
   cout << "Summary (times in ms; apply totals use reps=" << reps << ")\n";
   cout << "  Assembly packed EA: " << fixed << setprecision(6)
        << packed_assembly_ms << '\n';
   cout << "  Assembly full EA:   " << fixed << setprecision(6)
        << full_assembly_ms << '\n';
   cout << '\n';

   std::size_t max_label_len = 0;
   for (const auto &m : methods)
   {
      max_label_len = std::max(max_label_len, m.label.size());
   }

   const int w_method = std::min(64, std::max(28, int(max_label_len) + 2));
   const int w_num = 14;
   const int w_res = 14;

   cout << left
        << setw(w_method) << "Method"
        << right
        << setw(w_num) << "Setup"
        << setw(w_num) << "Apply"
        << setw(w_num) << "ApplyTot"
        << setw(w_num) << "Total"
        << setw(w_res) << "RelMax"
        << setw(w_res) << "RelL2"
        << '\n';
   cout << string(w_method + 4*w_num + 2*w_res, '-') << '\n';

   auto print_val = [&](const bool ok, const double value)
   {
      if (!ok) { cout << setw(w_num) << "n/a"; }
      else { cout << setw(w_num) << fixed << setprecision(6) << value; }
   };

   auto print_res = [&](const bool ok, const double value)
   {
      if (!ok || std::isnan(value)) { cout << setw(w_res) << "n/a"; }
      else
      {
         std::ios old_state(nullptr);
         old_state.copyfmt(cout);
         cout << setw(w_res) << scientific << setprecision(3) << value;
         cout.copyfmt(old_state);
      }
   };

   double best_total = std::numeric_limits<double>::infinity();
   std::string best_label;
   for (const auto &m : methods)
   {
      if (!m.available) { continue; }
      const double total = m.assembly_ms + m.setup_ms + reps*m.apply_ms;
      if (total < best_total)
      {
         best_total = total;
         best_label = m.label;
      }
   }

   for (const auto &m : methods)
   {
      const bool ok = m.available;
      const double apply_total = reps*m.apply_ms;
      const double total = m.assembly_ms + m.setup_ms + apply_total;

      cout << left << setw(w_method) << m.label << right;
      print_val(ok, m.setup_ms);
      print_val(ok, m.apply_ms);
      print_val(ok, apply_total);
      print_val(ok, total);
      print_res(ok, m.rel_max);
      print_res(ok, m.rel_l2);
      if (ok && m.label == best_label) { cout << "  (best total)"; }
      cout << '\n';
   }
}

void FillRHS(Vector &rhs)
{
   rhs.UseDevice(true);
   real_t *x = rhs.HostWrite();
   for (int i = 0; i < rhs.Size(); ++i)
   {
      x[i] = 1.0 + real_t((13*i + 7) % 29)/real_t(29);
   }
   rhs.Read();
   MFEM_DEVICE_SYNC;
}

void ComputeLowerPackedResidual(
   const TriPackLowerMatrix &lower,
   const Vector &x,
   const Vector &rhs,
   double &l2_residual,
   double &relative_l2_residual,
   real_t &max_residual,
   real_t &relative_max_residual)
{
   const int n = lower.GetNumRows();
   const int batch_size = lower.GetNumMatrices();
   const int packed_size = lower.GetPackedSize();
   MFEM_VERIFY(x.Size() == batch_size*n, "Solution vector has the wrong size.");
   MFEM_VERIFY(rhs.Size() == batch_size*n, "Right-hand side has the wrong size.");

   const real_t *A = lower.Data().HostRead();
   const real_t *X = x.HostRead();
   const real_t *B = rhs.HostRead();

   long double l2_sum = 0.0;
   long double rhs_l2_sum = 0.0;
   real_t max_abs = 0.0;
   real_t rhs_max_abs = 0.0;

   for (int e = 0; e < batch_size; ++e)
   {
      const real_t *Ae = A + e*packed_size;
      const real_t *Xe = X + e*n;
      const real_t *Be = B + e*n;
      for (int i = 0; i < n; ++i)
      {
         long double ax = 0.0;
         for (int j = 0; j < n; ++j)
         {
            const real_t aij =
               (i >= j) ?
               Ae[TriPackLowerMatrix::LowerIndex(i, j, n)] :
               Ae[TriPackLowerMatrix::LowerIndex(j, i, n)];
            ax += (long double)aij * (long double)Xe[j];
         }
         const long double residual = ax - (long double)Be[i];
         l2_sum += residual*residual;
         rhs_l2_sum += (long double)Be[i]*(long double)Be[i];
         max_abs = max(max_abs, (real_t)fabs((double)residual));
         rhs_max_abs = max(rhs_max_abs, fabs(Be[i]));
      }
   }

   l2_residual = sqrt((double)l2_sum);
   const double rhs_l2_norm = sqrt((double)rhs_l2_sum);
   relative_l2_residual =
      (rhs_l2_norm > 0.0) ? l2_residual/rhs_l2_norm : l2_residual;
   max_residual = max_abs;
   relative_max_residual =
      (rhs_max_abs > 0.0) ? max_residual/rhs_max_abs : max_residual;
}

void ApplyLowerInverseInPlace(
   const TriPackLowerMatrix &lower_inverse,
   Vector &x,
   Vector &work)
{
   const int n = lower_inverse.GetNumRows();
   const int batch_size = lower_inverse.GetNumMatrices();
   const int packed_size = lower_inverse.GetPackedSize();
   MFEM_VERIFY(x.Size() == batch_size*n, "Input vector has the wrong size.");

   work.SetSize(batch_size*n);
   work.UseDevice(true);

   const real_t *L = lower_inverse.Data().Read();
   const real_t *X = x.Read();
   real_t *T = work.Write();

   mfem::forall(batch_size*n, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int i = idx % n;
      const int e = idx / n;
      const real_t *Le = L + e*packed_size;
      const real_t *Xe = X + e*n;
      real_t sum = 0.0;
      for (int j = 0; j <= i; ++j)
      {
         sum += Le[TriPackLowerMatrix::LowerIndex(i, j, n)]*Xe[j];
      }
      T[idx] = sum;
   });

   const real_t *T_in = work.Read();
   real_t *Y = x.Write();
   mfem::forall(batch_size*n, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int i = idx % n;
      const int e = idx / n;
      const real_t *Le = L + e*packed_size;
      const real_t *Te = T_in + e*n;
      real_t sum = 0.0;
      for (int j = i; j < n; ++j)
      {
         sum += Le[TriPackLowerMatrix::LowerIndex(j, i, n)]*Te[j];
      }
      Y[idx] = sum;
   });
}

double TimeLowerInverseApply(
   const TriPackLowerMatrix &inverse,
   const Vector &rhs,
   const int reps,
   Vector &x,
   Vector &work)
{
   StopWatch sw;

   // Dry run to remove first-use kernel and workspace allocation costs.
   x = rhs;
   ApplyLowerInverseInPlace(inverse, x, work);
   MFEM_DEVICE_SYNC;
   sw.Start();
   for (int r = 0; r < reps; ++r)
   {
      x = rhs;
      ApplyLowerInverseInPlace(inverse, x, work);
   }
   MFEM_DEVICE_SYNC;
   sw.Stop();
   return 1000.0*sw.RealTime()/reps;
}

#ifdef MFEM_USE_MAGMA
double TimeMagmaSolve(
   const TriPackLowerMatrix &lower_factor,
   const Vector &rhs,
   const int reps,
   Vector &x,
   MagmaPackedLowerCholesky &ws)
{
   StopWatch sw;

   // Dry run to remove first-use MAGMA and RHS pointer-array setup costs.
   x = rhs;
   ws.SolveInPlace(lower_factor, x);
   MFEM_DEVICE_SYNC;
   sw.Start();
   for (int r = 0; r < reps; ++r)
   {
      x = rhs;
      ws.SolveInPlace(lower_factor, x);
   }
   MFEM_DEVICE_SYNC;
   sw.Stop();
   return 1000.0*sw.RealTime()/reps;
}

double TimeMagmaInverseApply(
   const TriPackLowerMatrix &lower_inverse,
   const Vector &rhs,
   const int reps,
   Vector &x,
   MagmaPackedLowerInverse &ws)
{
   StopWatch sw;

   // Dry run to remove first-use MAGMA and RHS pointer-array setup costs.
   x = rhs;
   ws.ApplyInPlace(lower_inverse, x);
   MFEM_DEVICE_SYNC;
   sw.Start();
   for (int r = 0; r < reps; ++r)
   {
      x = rhs;
      ws.ApplyInPlace(lower_inverse, x);
   }
   MFEM_DEVICE_SYNC;
   sw.Stop();
   return 1000.0*sw.RealTime()/reps;
}

double TimeBatchedLinAlgFullLUSolve(
   const DenseTensor &full_factor,
   const Array<int> &pivots,
   const Vector &rhs,
   const int reps,
   Vector &x)
{
   StopWatch sw;

   x = rhs;
   BatchedLinAlg::Get(BatchedLinAlg::MAGMA).LUSolve(full_factor, pivots, x);
   MFEM_DEVICE_SYNC;
   sw.Start();
   for (int r = 0; r < reps; ++r)
   {
      x = rhs;
      BatchedLinAlg::Get(BatchedLinAlg::MAGMA).LUSolve(full_factor, pivots, x);
   }
   MFEM_DEVICE_SYNC;
   sw.Stop();
   return 1000.0*sw.RealTime()/reps;
}

double TimeBatchedLinAlgFullInverseApply(
   const DenseTensor &full_inverse,
   const Vector &rhs,
   const int reps,
   Vector &x)
{
   StopWatch sw;

   // Dry run to remove first-use MAGMA and output allocation costs.
   BatchedLinAlg::Get(BatchedLinAlg::MAGMA).AddMult(full_inverse, rhs, x,
                                                    1.0, 0.0);
   MFEM_DEVICE_SYNC;
   sw.Start();
   for (int r = 0; r < reps; ++r)
   {
      BatchedLinAlg::Get(BatchedLinAlg::MAGMA).AddMult(full_inverse, rhs, x,
                                                       1.0, 0.0);
   }
   MFEM_DEVICE_SYNC;
   sw.Stop();
   return 1000.0*sw.RealTime()/reps;
}
#endif

} // namespace

int main(int argc, char *argv[])
{
   const char *mesh_file = "../../data/beam-hex.mesh";
   int order = 2;
   int ref_levels = 1;
   int reps = 100;
   int setup_reps = 10;
   const char *device_config = "gpu";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of uniform refinements.");
   args.AddOption(&reps, "-reps", "--apply-repetitions",
                  "Number of mass inverse applications to time.");
   args.AddOption(&setup_reps, "-sreps", "--setup-repetitions",
                  "Number of setup repetitions to time.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.ParseCheck();

   MFEM_VERIFY(reps > 0, "The number of apply repetitions must be positive.");
   MFEM_VERIFY(setup_reps > 0,
               "The number of setup repetitions must be positive.");

   Device device(device_config);
   device.Print();

   bool magma_supported_device = false;
#ifdef MFEM_USE_MAGMA
   magma_supported_device = Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK);
#endif

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

   const int ne = mesh.GetNE();
   const int elem_dofs = fespace.GetTypicalFE()->GetDof();

   MassIntegrator mass;
   StopWatch sw;

   // Dry run assembly before timing steady-state work.
   TriPackLowerMatrix packed_ea;
   mass.AssembleEATriangular(fespace, packed_ea, false);
   MFEM_DEVICE_SYNC;

   sw.Clear();
   sw.Start();
   mass.AssembleEATriangular(fespace, packed_ea, false);
   MFEM_DEVICE_SYNC;
   sw.Stop();
   const double assemble_packed_ms = 1000.0*sw.RealTime();

   // Time dense/full element-matrix assembly (needed for the full-LU path).
   DenseTensor full_ea(elem_dofs, elem_dofs, ne, Device::GetDeviceMemoryType());
   Vector full_ea_vec;
   full_ea_vec.NewMemoryAndSize(full_ea.GetMemory(), full_ea.TotalSize(), false);
   full_ea_vec.UseDevice(true);

   // Dry run to remove first-use kernel and workspace allocation costs.
   mass.AssembleEA(fespace, full_ea_vec, false);
   MFEM_DEVICE_SYNC;

   sw.Clear();
   sw.Start();
   mass.AssembleEA(fespace, full_ea_vec, false);
   MFEM_DEVICE_SYNC;
   sw.Stop();
   const double assemble_full_ms = 1000.0*sw.RealTime();

   TriPackLowerMatrix tripack_inverse;
   // Dry run setup before timing steady-state setup work.
   tripack::ComputeCholeskyLowerInverse(packed_ea, tripack_inverse);
   MFEM_DEVICE_SYNC;

   sw.Clear();
   sw.Start();
   for (int r = 0; r < setup_reps; ++r)
   {
      tripack::ComputeCholeskyLowerInverse(packed_ea, tripack_inverse);
   }
   MFEM_DEVICE_SYNC;
   sw.Stop();
   const double tripack_inverse_setup_ms = 1000.0*sw.RealTime()/setup_reps;

#ifdef MFEM_USE_MAGMA
   magma_queue_t magma_queue = nullptr;
   if (magma_supported_device) { magma_queue = Magma::Queue(); }

   TriPackLowerMatrix magma_factor;
   std::unique_ptr<MagmaPackedLowerCholesky> magma_chol_ws;
   double magma_factor_ms = 0.0;

   TriPackLowerMatrix magma_inverse;
   std::unique_ptr<MagmaPackedLowerInverse> magma_inv_ws;
   double magma_inverse_ms = 0.0;
   bool magma_ppinv_enabled = false;
   DenseTensor magma_full_factor;
   Array<int> magma_full_pivots;
   double magma_full_factor_ms = 0.0;
   DenseTensor magma_full_inverse;
   double magma_full_inverse_ms = 0.0;

   if (magma_supported_device)
   {
      magma_chol_ws.reset(new MagmaPackedLowerCholesky());
      magma_chol_ws->SetQueue(magma_queue);

      // Dry run setup before timing steady-state setup work.
      magma_chol_ws->Factor(packed_ea, magma_factor);
      MFEM_DEVICE_SYNC;

      sw.Clear();
      sw.Start();
      for (int r = 0; r < setup_reps; ++r)
      {
         magma_chol_ws->Factor(packed_ea, magma_factor);
      }
      MFEM_DEVICE_SYNC;
      sw.Stop();
      magma_factor_ms = 1000.0*sw.RealTime()/setup_reps;

      // Benchmark packed inverse (ppinv) only for sizes supported by MAGMA's
      // current packed-inverse apply kernel.
      if (elem_dofs <= 64)
      {
         magma_ppinv_enabled = true;
         magma_inv_ws.reset(new MagmaPackedLowerInverse());
         magma_inv_ws->SetQueue(magma_queue);

         // Dry run setup before timing steady-state setup work.
         magma_inv_ws->Compute(packed_ea, magma_inverse);
         MFEM_DEVICE_SYNC;

         sw.Clear();
         sw.Start();
         for (int r = 0; r < setup_reps; ++r)
         {
            magma_inv_ws->Compute(packed_ea, magma_inverse);
         }
         MFEM_DEVICE_SYNC;
         sw.Stop();
         magma_inverse_ms = 1000.0*sw.RealTime()/setup_reps;
      }

      // Full (dense) MFEM BatchedLinAlg LU factorization for comparison.
      magma_full_factor = full_ea;
      BatchedLinAlg::Get(BatchedLinAlg::MAGMA).LUFactor(magma_full_factor,
                                                        magma_full_pivots);
      MFEM_DEVICE_SYNC;

      sw.Clear();
      sw.Start();
      for (int r = 0; r < setup_reps; ++r)
      {
         magma_full_factor = full_ea;
         BatchedLinAlg::Get(BatchedLinAlg::MAGMA).LUFactor(magma_full_factor,
                                                           magma_full_pivots);
      }
      MFEM_DEVICE_SYNC;
      sw.Stop();
      magma_full_factor_ms = 1000.0*sw.RealTime()/setup_reps;

      magma_full_inverse = full_ea;
      BatchedLinAlg::Get(BatchedLinAlg::MAGMA).Invert(magma_full_inverse);
      MFEM_DEVICE_SYNC;

      sw.Clear();
      sw.Start();
      for (int r = 0; r < setup_reps; ++r)
      {
         magma_full_inverse = full_ea;
         BatchedLinAlg::Get(BatchedLinAlg::MAGMA).Invert(magma_full_inverse);
      }
      MFEM_DEVICE_SYNC;
      sw.Stop();
      magma_full_inverse_ms = 1000.0*sw.RealTime()/setup_reps;
   }
#endif

   Vector rhs(ne*elem_dofs);
   FillRHS(rhs);

   Vector tripack_x(rhs.Size()), work;
   tripack_x.UseDevice(true);

   const double tripack_apply_ms =
      TimeLowerInverseApply(tripack_inverse, rhs, reps, tripack_x, work);
   double tripack_res_l2 = 0.0, tripack_rel_res_l2 = 0.0;
   real_t tripack_res_max = 0.0, tripack_rel_res_max = 0.0;
   ComputeLowerPackedResidual(packed_ea, tripack_x, rhs,
                              tripack_res_l2, tripack_rel_res_l2,
                              tripack_res_max, tripack_rel_res_max);

#ifdef MFEM_USE_MAGMA
   double magma_solve_ms = 0.0;
   double magma_full_solve_ms = 0.0;
   double magma_res_l2 = 0.0, magma_rel_res_l2 = 0.0;
   real_t magma_res_max = 0.0, magma_rel_res_max = 0.0;
   double magma_full_res_l2 = 0.0, magma_full_rel_res_l2 = 0.0;
   real_t magma_full_res_max = 0.0, magma_full_rel_res_max = 0.0;
   double magma_full_inv_apply_ms = 0.0;
   double magma_full_inv_res_l2 = 0.0, magma_full_inv_rel_res_l2 = 0.0;
   real_t magma_full_inv_res_max = 0.0, magma_full_inv_rel_res_max = 0.0;
   Vector magma_x;
   Vector magma_full_x;
   Vector magma_full_inv_x;
   if (magma_supported_device)
   {
      magma_x.SetSize(rhs.Size());
      magma_x.UseDevice(true);
      magma_solve_ms =
         TimeMagmaSolve(magma_factor, rhs, reps, magma_x, *magma_chol_ws);
      ComputeLowerPackedResidual(packed_ea, magma_x, rhs,
                                 magma_res_l2, magma_rel_res_l2,
                                 magma_res_max, magma_rel_res_max);

      magma_full_x.SetSize(rhs.Size());
      magma_full_x.UseDevice(true);
      magma_full_solve_ms =
         TimeBatchedLinAlgFullLUSolve(magma_full_factor, magma_full_pivots, rhs,
                                      reps, magma_full_x);
      ComputeLowerPackedResidual(packed_ea, magma_full_x, rhs,
                                 magma_full_res_l2, magma_full_rel_res_l2,
                                 magma_full_res_max, magma_full_rel_res_max);

      magma_full_inv_x.SetSize(rhs.Size());
      magma_full_inv_x.UseDevice(true);
      magma_full_inv_apply_ms =
         TimeBatchedLinAlgFullInverseApply(magma_full_inverse, rhs, reps,
                                           magma_full_inv_x);
      ComputeLowerPackedResidual(packed_ea, magma_full_inv_x, rhs,
                                 magma_full_inv_res_l2,
                                 magma_full_inv_rel_res_l2,
                                 magma_full_inv_res_max,
                                 magma_full_inv_rel_res_max);
   }

   double magma_ppinv_apply_ms = 0.0;
   double magma_ppinv_res_l2 = 0.0, magma_ppinv_rel_res_l2 = 0.0;
   real_t magma_ppinv_res_max = 0.0, magma_ppinv_rel_res_max = 0.0;
   Vector magma_ppinv_x;
   if (magma_supported_device && magma_ppinv_enabled)
   {
      magma_ppinv_x.SetSize(rhs.Size());
      magma_ppinv_x.UseDevice(true);
      magma_ppinv_apply_ms =
         TimeMagmaInverseApply(magma_inverse, rhs, reps, magma_ppinv_x,
                               *magma_inv_ws);
      ComputeLowerPackedResidual(packed_ea, magma_ppinv_x, rhs,
                                 magma_ppinv_res_l2, magma_ppinv_rel_res_l2,
                                 magma_ppinv_res_max, magma_ppinv_rel_res_max);
   }
#endif

   cout << fixed << setprecision(6);
   cout << "Mass matrix inverse microbenchmark" << '\n';
   cout << "Mesh: " << mesh_file << '\n';
   cout << "Dimension: " << dim << '\n';
   cout << "Elements: " << ne << '\n';
   cout << "Element dofs: " << elem_dofs << '\n';
   cout << "Scalar element unknowns: " << ne*elem_dofs << '\n';
   cout << "Apply repetitions: " << reps << '\n';
   cout << "Setup repetitions: " << setup_reps << '\n';
   cout << '\n';

   std::vector<MethodTiming> methods;
   methods.push_back(
   {
      "MFEM tripack inverse", true,
      assemble_packed_ms,
      tripack_inverse_setup_ms, tripack_apply_ms,
      (double)tripack_rel_res_max, tripack_rel_res_l2});
#ifdef MFEM_USE_MAGMA
   if (magma_supported_device)
   {
      methods.push_back(
      {
         "MAGMA packed Cholesky solve", true,
         assemble_packed_ms,
         magma_factor_ms, magma_solve_ms,
         (double)magma_rel_res_max, magma_rel_res_l2});

      if (magma_ppinv_enabled)
      {
         methods.push_back(
         {
            "MAGMA packed inverse (ppinv)", true,
            assemble_packed_ms,
            magma_inverse_ms, magma_ppinv_apply_ms,
            (double)magma_ppinv_rel_res_max, magma_ppinv_rel_res_l2});
      }

      methods.push_back(
      {
         "MAGMA full LU solve", true,
         assemble_full_ms,
         magma_full_factor_ms, magma_full_solve_ms,
         (double)magma_full_rel_res_max, magma_full_rel_res_l2});

      methods.push_back(
      {
         "MAGMA full inverse + apply", true,
         assemble_full_ms,
         magma_full_inverse_ms, magma_full_inv_apply_ms,
         (double)magma_full_inv_rel_res_max, magma_full_inv_rel_res_l2});
   }
#endif
   PrintTimingTable(assemble_packed_ms, assemble_full_ms, reps, methods);

   return 0;
}
