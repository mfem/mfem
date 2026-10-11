// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//            ------------------------------------------------
//            Hologram Miniapp: PNG to phase mask and hologram
//            ------------------------------------------------
//
// This miniapp reads a target PNG and retrieves a complex phase mask.
// The mask is propagated and written as a monochromatic hologram.
//
// Sample runs:  mpirun -np 4 phologram
//               mpirun -np 4 phologram -p 5
//               mpirun -np 4 phologram -p 5 -gem
//               mpirun -np 4 phologram -gem -mi 2 -check

#include "hologram.hpp"
using namespace mfem;

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace holo
{

// True on all ranks when error is empty on rank 0, which prints it otherwise
inline bool RootOk(const std::string &error)
{
   int ok = error.empty() ? 1 : 0;
   MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
   if (!ok && Mpi::Root()) { mfem::err << error << "\n"; }
   return ok == 1;
}

// True when error is empty on all ranks,
// otherwise rank 0 prints the error of the lowest rank with one
inline bool AllOk(const std::string &error)
{
   const int size = Mpi::WorldSize(), rank = Mpi::WorldRank();
   int first = error.empty() ? size : rank;
   MPI_Allreduce(MPI_IN_PLACE, &first, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
   if (first == size) { return true; }
   std::string text = error;
   if (first != 0 && (rank == first || rank == 0))
   {
      int length = static_cast<int>(error.size());
      if (rank == first)
      {
         MPI_Send(&length, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
         MPI_Send(error.data(), length, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
      }
      else
      {
         MPI_Recv(&length, 1, MPI_INT, first, 0, MPI_COMM_WORLD,
                  MPI_STATUS_IGNORE);
         text.resize(static_cast<std::size_t>(length));
         MPI_Recv(&text[0], length, MPI_CHAR, first, 1, MPI_COMM_WORLD,
                  MPI_STATUS_IGNORE);
      }
   }
   if (rank == 0) { mfem::err << text << "\n"; }
   return false;
}

// Number of ranks sharing this rank's node, and of those using its GPU, by
// PCI bus id
inline void NodeRanks(int &nodeRanks, int &deviceRanks)
{
   constexpr int idSize = 32;
   char busId[idSize] = "host";
#if defined(MFEM_USE_CUDA)
   if (Device::Allows(Backend::CUDA_MASK))
   {
      int dev = 0;
      MFEM_GPU_CHECK(cudaGetDevice(&dev));
      MFEM_GPU_CHECK(cudaDeviceGetPCIBusId(busId, idSize, dev));
   }
#elif defined(MFEM_USE_HIP)
   if (Device::Allows(Backend::HIP_MASK))
   {
      int dev = 0;
      MFEM_GPU_CHECK(hipGetDevice(&dev));
      MFEM_GPU_CHECK(hipDeviceGetPCIBusId(busId, idSize, dev));
   }
#endif
   MPI_Comm node;
   MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                       &node);
   MPI_Comm_size(node, &nodeRanks);
   std::vector<char> ids(static_cast<std::size_t>(nodeRanks) * idSize);
   MPI_Allgather(busId, idSize, MPI_CHAR, ids.data(), idSize, MPI_CHAR, node);
   MPI_Comm_free(&node);
   deviceRanks = 0;
   for (int r = 0; r < nodeRanks; ++r)
   {
      const char *id = ids.data() + static_cast<std::size_t>(r) * idSize;
      if (std::strncmp(busId, id, idSize) == 0) { ++deviceRanks; }
   }
}

class ParRowSum : public RowSum
{
   const Array<int> &counts, &displs;
   mutable Vector all;

protected:
   const real_t *AllRows(const Vector &local, int &rows) const override
   {
      if (Mpi::WorldSize() == 1) { return RowSum::AllRows(local, rows); }
      rows = displs.Last() + counts.Last();
      all.SetSize(rows);
      MPI_Allgatherv(const_cast<real_t *>(local.HostRead()), local.Size(),
                     MPITypeMap<real_t>::mpi_type, all.HostWrite(),
                     counts.GetData(), displs.GetData(),
                     MPITypeMap<real_t>::mpi_type, MPI_COMM_WORLD);
      return all.HostRead();
   }

public:
   ParRowSum(int length, const Array<int> &rowCounts,
             const Array<int> &rowDispls)
      : RowSum(length), counts(rowCounts), displs(rowDispls) {}
};

class PixelMesh
{
   L2_FECollection fec, fec0;
   std::unique_ptr<ParMesh> mesh;
   std::unique_ptr<ParFiniteElementSpace> fes, fes0;

   const int order, q1d, nx, ny;
   const int y0, y1, elem0;
   Array<int> counts, displs, rowCounts, rowDispls;
   mutable ParRowSum rowSum;

   static int VerifyOrder(int pixels_x, int pixels_y, int order)
   {
      MFEM_VERIFY(pixels_x >= 1 && pixels_y >= 1,
                  "Pixels: image size must be positive");
      MFEM_VERIFY(order >= 0, "Pixels: order must be >= 0");
      const int q1d = order + 1;
      MFEM_VERIFY(pixels_x % q1d == 0 && pixels_y % q1d == 0,
                  "Pixels: pixel count must be divisible by order+1");
      return order;
   }

   static int RowBegin(int rows, int r, int np)
   {
      return static_cast<int>((static_cast<long long>(rows) * r) / np);
   }

public:
   PixelMesh(PixelMesh &&) = delete;
   PixelMesh(const PixelMesh &) = delete;
   PixelMesh &operator=(PixelMesh &&) = delete;
   PixelMesh &operator=(const PixelMesh &) = delete;

   PixelMesh(int pixels_x, int pixels_y, int order_in,
             real_t extent_x, real_t extent_y)
      : fec(VerifyOrder(pixels_x, pixels_y, order_in), /*dim=*/2,
            BasisType::OpenHalfUniform),
        fec0(0, 2, BasisType::OpenHalfUniform),
        order(order_in),
        q1d(order_in + 1),
        nx(pixels_x / (order_in + 1)),
        ny(pixels_y / (order_in + 1)),
        y0(RowBegin(ny, Mpi::WorldRank(), Mpi::WorldSize())),
        y1(RowBegin(ny, Mpi::WorldRank() + 1, Mpi::WorldSize())),
        elem0(y0 * nx),
        rowSum(nx * q1d * q1d, rowCounts, rowDispls)
   {
      Mesh serial =
         Mesh::MakeCartesian2D(
            nx, ny, Element::QUADRILATERAL,
            /*generate_edges=*/ false,
            extent_x, extent_y,
            /*sfc_ordering=*/ false);
      serial.Transform([&](const Vector &x, Vector &p)
      {
         p(0) = x(0) - 0.5 * extent_x;
         p(1) = x(1) - 0.5 * extent_y;
      });

      const int q2 = q1d * q1d;
      Array<int> partition(serial.GetNE());
      const int np = Mpi::WorldSize();
      counts.SetSize(np);
      displs.SetSize(np);
      rowCounts.SetSize(np);
      rowDispls.SetSize(np);
      int off = 0;
      for (int r = 0; r < np; ++r)
      {
         const int ry0 = RowBegin(ny, r, np);
         const int ry1 = RowBegin(ny, r + 1, np);
         for (int ey = ry0; ey < ry1; ++ey)
            for (int ex = 0; ex < nx; ++ex)
            {
               partition[ex + ey * nx] = r;
            }
         counts[r] = (ry1 - ry0) * nx * q2;
         displs[r] = off;
         off += counts[r];
         rowCounts[r] = ry1 - ry0;
         rowDispls[r] = ry0;
      }
      MFEM_VERIFY(off == NumSamples(), "gather counts");

      mesh = std::make_unique<ParMesh>
             (MPI_COMM_WORLD, serial, partition.GetData());
      fes = std::make_unique<ParFiniteElementSpace>(mesh.get(), &fec);
      fes0 = std::make_unique<ParFiniteElementSpace>(mesh.get(), &fec0);

      const int ne = mesh->GetNE();
      MFEM_VERIFY(ne == (y1 - y0) * nx, "owned element rows");
      MFEM_VERIFY(fes->GetVSize() == ne * q2, "GetVSize sample count");
      MFEM_VERIFY(fes0->GetVSize() == ne, "order-0 element space");
      MFEM_VERIFY((LocalNumSamples() == 0) == (y0 == y1),
                  "LocalNumSamples()==0 exactly when y0==y1");
      MFEM_VERIFY(mesh->GetGlobalElementNum(0) == static_cast<long long>(elem0),
                  "row partition element 0");
      MFEM_VERIFY(ne > 0, "Empty mesh not supported");
      if (Mpi::Root())
      {
         mfem::out << "[hologram] mesh quads=" << nx << "x" << ny
                   << " " << fec.Name()
                   << " dofs=" << NumSamples()
                   << " local=" << LocalNumSamples()
                   << " rows=" << y0 << ":" << y1 << "\n";
      }
   }

   int Order() const { return order; }
   int Q1D() const { return q1d; }
   int Nx() const { return nx; }
   int Ny() const { return ny; }

   int SampleNx() const { return nx * q1d; }
   int SampleNy() const { return ny * q1d; }
   int NumSamples() const { return SampleNx() * SampleNy(); }

   int LocalNumSamples() const { return fes->GetVSize(); }

   // First sample row of rank r
   int SampleRowBegin(int r) const
   {
      return q1d * RowBegin(ny, r, Mpi::WorldSize());
   }
   int FirstSampleRow() const { return y0 * q1d; }
   int LocalSampleRows() const { return (y1 - y0) * q1d; }

   // Sample counts and offsets of all ranks
   const Array<int> &SampleCounts() const { return counts; }
   const Array<int> &SampleDispls() const { return displs; }

   ParFiniteElementSpace &Space() const { return *fes; }
   ParFiniteElementSpace &ElementSpace() const { return *fes0; }

   // Sums over element rows, in global row order
   RowSum &Reducer() const { return rowSum; }

   void GatherDofs(const Vector &local, Vector &global) const
   {
      const int nglob = NumSamples();
      global.UseDevice(true);
      if (global.Size() != nglob) { global.SetSize(nglob); }
      if (Mpi::WorldSize() == 1) { global = local; return; }
      const int nlocal = LocalNumSamples();
      MFEM_VERIFY(local.Size() == nlocal, "GatherDofs size");
      MPI_Allgatherv(const_cast<real_t *>(local.HostRead()), nlocal,
                     MPITypeMap<real_t>::mpi_type,
                     global.HostWrite(), counts.GetData(), displs.GetData(),
                     MPITypeMap<real_t>::mpi_type,
                     MPI_COMM_WORLD);
   }

   void ScatterDofs(const Vector &global, Vector &local) const
   {
      const int nlocal = LocalNumSamples();
      local.UseDevice(true);
      if (local.Size() != nlocal) { local.SetSize(nlocal); }
      if (Mpi::WorldSize() == 1) { local = global; return; }
      MFEM_VERIFY(global.Size() == NumSamples(), "ScatterDofs size");
      const real_t *g =
         global.HostRead() + static_cast<long long>(elem0) * q1d * q1d;
      real_t *l = local.HostWrite();
      for (int i = 0; i < nlocal; ++i) { l[i] = g[i]; }
   }
};

struct Fields
{
   ParComplexGridFunction u, du, s, g;
   Vector dot_v, energy_v, loss_q, phase_v;
   Vector glob_re, glob_im;
   Vector E, E_trial, gE, dE, seed;

   explicit Fields(const PixelMesh &grid)
      : u(&grid.Space()),
        du(&grid.Space()),
        s(&grid.Space()),
        g(&grid.Space())
   {
      const int nloc = grid.LocalNumSamples();
      const int n = grid.NumSamples();
      auto set_size = [](Vector &v, int m)
      {
         v.SetSize(m);
         v.UseDevice(true);
      };
      set_size(dot_v, nloc);
      set_size(energy_v, nloc);
      set_size(loss_q, nloc);
      set_size(phase_v, n);
      set_size(glob_re, n);
      set_size(glob_im, n);

      const int nx = grid.SampleNx(), ny = grid.LocalSampleRows();
      SetInterleavedSize(E, nx, ny);
      SetInterleavedSize(E_trial, nx, ny);
      SetInterleavedSize(gE, nx, ny);
      SetInterleavedSize(dE, nx, ny);
      SetInterleavedSize(seed, nx, ny);

      for (Vector *v :
           {
              &dot_v, &energy_v, &loss_q,
              &phase_v, &glob_re, &glob_im,
              &E, &E_trial, &gE, &dE, &seed
           }) { v->Write(); }
   }
};

struct RasterField
{
   void operator()(const PixelMesh &grid,
                   const Vector &re_s, const Vector &im_s,
                   RealArray2D &image, Fields &data)
   {
      const int n = grid.NumSamples();
      const Vector *re_v = &re_s;
      const Vector *im_v = &im_s;
      if (Mpi::WorldSize() != 1)
      {
         grid.GatherDofs(re_s, data.glob_re);
         grid.GatherDofs(im_s, data.glob_im);
         re_v = &data.glob_re;
         im_v = &data.glob_im;
      }
      const int nx = grid.Nx();
      const int q1d = grid.Q1D();
      const real_t *er = re_v->Read();
      const real_t *ei = im_v->Read();
      real_t *p = data.phase_v.Write();
      mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
      {
         const int k = RasterIndex(i, nx, q1d);
         p[k] = std::atan2(ei[i], er[i]);
      });
      image.SetSize(grid.SampleNy(), grid.SampleNx());
      const real_t *ph = data.phase_v.HostRead();
      real_t *dst = image(0);
      for (int i = 0; i < n; ++i) { dst[i] = ph[i]; }
   }

   static void FromSamples(const PixelMesh &grid, const Vector &re_s,
                           const Vector &im_s, Vector &E)
   {
      SetInterleavedSize(E, grid.SampleNx(), grid.LocalSampleRows());
      Pack(grid.Q1D(), grid.LocalNumSamples(), grid.Nx(), re_s, im_s, E);
   }

   static void ToSamples(const PixelMesh &grid, const Vector &E,
                         Vector &re_s, Vector &im_s)
   {
      Unpack(grid.Q1D(), grid.LocalNumSamples(), grid.Nx(), E, re_s, im_s);
   }

   static void ToSamples(const PixelMesh &grid, const Vector &E,
                         ParComplexGridFunction &field)
   {
      ToSamples(grid, E, field.real(), field.imag());
      field.SyncAlias();
   }
};

// ────────────────────────────────────────────────────────────────────────────
class PhaseMask
{
   PixelMesh &grid;
   const ParGridFunction *src, *tgt;
   ParComplexGridFunction lin;
   static constexpr real_t kLossEps = 1e-24;

public:
   explicit PhaseMask(PixelMesh &grid)
      : grid(grid), src(nullptr), tgt(nullptr), lin(&grid.Space()) {}

   void SetAmplitudes(const ParGridFunction &a_src_in,
                      const ParGridFunction &a_tgt_in)
   {
      src = &a_src_in;
      tgt = &a_tgt_in;
   }

   // Fold the phase into [-π, π)
   void Wrap(Vector &phi) const
   {
      const int n = phi.Size();
      real_t *p = phi.ReadWrite();
      constexpr real_t two_pi = 2.0 * M_PI;
      mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
      {
         p[i] = p[i] - two_pi * std::floor((p[i] + M_PI) / two_pi);
      });
   }

   // Compute A exp(i φ)
   void Modulate(const Vector &phi, ParComplexGridFunction &field) const
   {
      const int n = grid.LocalNumSamples();
      const int q2 = grid.Q1D() * grid.Q1D();
      const real_t *p = phi.Read(), *a = src->Read();
      real_t *re = field.real().Write();
      real_t *im = field.imag().Write();
      mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
      {
         const real_t ai = a[i / q2];
         re[i] = ai * std::cos(p[i]);
         im[i] = ai * std::sin(p[i]);
      });
      field.SyncAlias();
   }

   void Linearize(const Vector &phi) { Modulate(phi, lin); }

   void Derivative(const Vector &dphi, ParComplexGridFunction &dfield) const
   {
      const int n = grid.LocalNumSamples();
      const real_t *er = lin.real().Read();
      const real_t *ei = lin.imag().Read();
      const real_t *dp = dphi.Read();
      real_t *dr = dfield.real().Write();
      real_t *di = dfield.imag().Write();
      mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
      {
         dr[i] = -ei[i] * dp[i];
         di[i] = er[i] * dp[i];
      });
      dfield.SyncAlias();
   }

   // Map a field variation back to the phase
   void Adjoint(const ParComplexGridFunction &dfield, Vector &g_phi) const
   {
      const int n = grid.LocalNumSamples();
      const real_t *er = lin.real().Read();
      const real_t *ei = lin.imag().Read();
      const real_t *gr = dfield.real().Read();
      const real_t *gi = dfield.imag().Read();
      real_t *ww = g_phi.Write();
      mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
      {
         ww[i] = er[i] * gi[i] - ei[i] * gr[i];
      });
   }

   // Compute the mean of (|U| - A_target)² and its gradient w.r.t. the field
   real_t MeanAmplitudeError(const ParComplexGridFunction &field,
                             ParComplexGridFunction &grad, Fields &data) const
   {
      const int n = field.real().Size();
      const real_t inv_n = 1.0 / static_cast<real_t>(grid.NumSamples());
      const real_t *er = field.real().Read();
      const real_t *ei = field.imag().Read();
      const real_t *at = tgt->Read();
      real_t *lq = data.loss_q.Write();
      real_t *gr = grad.real().Write();
      real_t *gi = grad.imag().Write();
      mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
      {
         const real_t mag = std::sqrt(er[i] * er[i] + ei[i] * ei[i] + kLossEps);
         const real_t d = mag - at[i];
         lq[i] = d * d;
         const real_t c = (2.0 * d / mag) * inv_n;
         gr[i] = c * er[i];
         gi[i] = c * ei[i];
      });
      grad.SyncAlias();
      real_t sum = grid.Reducer().Sum(data.loss_q);
      return sum * inv_n;
   }
};

// ────────────────────────────────────────────────────────────────────────────
// Copy between rows x nx local rows and the stage, which holds the local rows
// split by the column block of each of the np ranks, rank by rank
inline void StageTranspose(const Vector &src, Vector &dst, int rows, int nx,
                           int np, bool toColumns)
{
   const int n = rows * nx;
   const real_t *s = src.Read();
   real_t *d = dst.Write();
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int i = idx / nx, c = idx - i * nx;
      const int p = static_cast<int>(
                       ((static_cast<long long>(c) + 1) * np - 1) / nx);
      const int c0 = static_cast<int>((static_cast<long long>(nx) * p) / np);
      const int c1 =
         static_cast<int>((static_cast<long long>(nx) * (p + 1)) / np);
      const int k = rows * c0 + i * (c1 - c0) + (c - c0);
      const int from = toColumns ? idx : k, to = toColumns ? k : idx;
      d[2 * to] = s[2 * from];
      d[2 * to + 1] = s[2 * from + 1];
   });
}

// Fresnel propagation of the local sample rows: the x factor on the local
// rows, the y factor on a block of full columns after an all-to-all transpose
class ParFresnel : public Operator
{
   std::unique_ptr<FresnelOp> op;
   const int nx, ny, np, rows, cols;
   Array<int> rowCounts, rowDispls, colCounts, colDispls;
   mutable Vector stage, block;

   static int ColBegin(int n, int p, int np)
   {
      return static_cast<int>((static_cast<long long>(n) * p) / np);
   }

   void Transpose(const Vector &src, Vector &dst, bool toColumns) const
   {
      StageTranspose(src, dst, rows, nx, np, toColumns);
   }

   void ToColumns(const Vector &y) const
   {
      Transpose(y, stage, true);
      MPI_Alltoallv(const_cast<real_t *>(stage.HostRead()),
                    rowCounts.GetData(), rowDispls.GetData(),
                    MPITypeMap<real_t>::mpi_type,
                    block.HostWrite(), colCounts.GetData(), colDispls.GetData(),
                    MPITypeMap<real_t>::mpi_type, MPI_COMM_WORLD);
   }

   void ToRows(Vector &y) const
   {
      MPI_Alltoallv(const_cast<real_t *>(block.HostRead()),
                    colCounts.GetData(), colDispls.GetData(),
                    MPITypeMap<real_t>::mpi_type,
                    stage.HostWrite(), rowCounts.GetData(), rowDispls.GetData(),
                    MPITypeMap<real_t>::mpi_type, MPI_COMM_WORLD);
      Transpose(stage, y, false);
   }

   void Mult(const Vector &x, Vector &y, bool adjoint) const
   {
      MFEM_VERIFY(x.Size() == Height(), "ParFresnel input size");
      y.UseDevice(true);
      if (y.Size() != Height()) { y.SetSize(Height()); }
      if (&x != &y) { y = x; }
      if (np == 1)
      {
         auto *E = reinterpret_cast<complex_t *>(y.ReadWrite());
         op->ApplyY(E, adjoint);
         op->ApplyX(E, adjoint);
         return;
      }
      ToColumns(y);
      if (cols > 0)
      {
         op->ApplyY(reinterpret_cast<complex_t *>(block.ReadWrite()), adjoint);
      }
      ToRows(y);
      op->ApplyX(reinterpret_cast<complex_t *>(y.ReadWrite()), adjoint);
   }

public:
   ParFresnel(const PixelMesh &grid, bool fft, real_t wavelength,
              real_t extent_x, real_t extent_y, real_t z)
      : Operator(2 * grid.LocalSampleRows() * grid.SampleNx()),
        op(MakeFresnelOp(fft)),
        nx(grid.SampleNx()), ny(grid.SampleNy()),
        np(Mpi::WorldSize()),
        rows(grid.LocalSampleRows()),
        cols(ColBegin(nx, Mpi::WorldRank() + 1, np) -
             ColBegin(nx, Mpi::WorldRank(), np))
   {
      rowCounts.SetSize(np);
      rowDispls.SetSize(np);
      colCounts.SetSize(np);
      colDispls.SetSize(np);
      for (int p = 0; p < np; ++p)
      {
         const int c0 = ColBegin(nx, p, np), c1 = ColBegin(nx, p + 1, np);
         const int r0 = grid.SampleRowBegin(p), r1 = grid.SampleRowBegin(p + 1);
         rowCounts[p] = 2 * rows * (c1 - c0);
         rowDispls[p] = 2 * rows * c0;
         colCounts[p] = 2 * (r1 - r0) * cols;
         colDispls[p] = 2 * r0 * cols;
      }
      stage.UseDevice(true);
      stage.SetSize(2 * rows * nx);
      block.UseDevice(true);
      block.SetSize(2 * ny * cols);
      op->Assemble(wavelength, extent_x, extent_y, z, nx, ny, cols, rows);
   }

   void Mult(const Vector &x, Vector &y) const override { Mult(x, y, false); }
   void MultTranspose(const Vector &x, Vector &y) const override
   {
      Mult(x, y, true);
   }

   void Propagate(Vector &field) const
   {
      MFEM_VERIFY(field.Size() == Height(), "ParFresnel replay size");
      const complex_t phase = op->Phase();
      auto *Z = reinterpret_cast<complex_t *>(field.ReadWrite());
      mfem::forall(Height() / 2, [=] MFEM_HOST_DEVICE (int i)
      {
         Z[i] *= phase;
      });
      Mult(field, field);
   }
};

// ────────────────────────────────────────────────────────────────────────────
class GaussNewtonOp : public Operator
{
   PhaseMask &mask;
   const PixelMesh &grid;
   const ParFresnel &prop;
   const Lens &lens;
   const Vector &E_scaled;
   Fields &data;
   const Vector *phi = nullptr;
   static constexpr real_t kAbsEps = 1e-14;
   static constexpr real_t kLmLambda0 = 1e-2;
   real_t scale = 0.0, lambda = kLmLambda0;

public:
   GaussNewtonOp(PhaseMask &mask, const PixelMesh &grid,
                 const ParFresnel &prop, const Lens &lens,
                 const Vector &E_scaled, Fields &data)
      : Operator(grid.Space().GetVSize()),
        mask(mask), grid(grid), prop(prop),
        lens(lens), E_scaled(E_scaled), data(data) {}

   void SetState(const Vector &phi_in, real_t scale_in)
   {
      phi = &phi_in;
      scale = scale_in;
      mask.Linearize(*phi);
   }

   void SetLambda(real_t lam = kLmLambda0) { lambda = lam; }

   real_t Lambda() const { return lambda; }

   void Mult(const Vector &dphi, Vector &out) const override
   {
      mask.Derivative(dphi, data.du);
      Assemble(out);
      if (lambda != 0.0)
      {
         const real_t inv_n = 1.0 / static_cast<real_t>(grid.NumSamples());
         out.Add(lambda * inv_n, dphi);
      }
   }

   void Assemble(Vector &out) const
   {
      RasterField::FromSamples(grid, data.du.real(), data.du.imag(), data.dE);
      lens.Apply(data.dE, false);
      prop.Mult(data.dE, data.dE);

      const int n = InterleavedSize(data.dE);
      const real_t *Es = E_scaled.Read();
      const real_t *dE = data.dE.Read();
      real_t *seed = data.seed.Write();
      const real_t scale2 = scale * scale;
      mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
      {
         const real_t a = Es[2 * i], b = Es[2 * i + 1];
         const real_t dr = dE[2 * i], di = dE[2 * i + 1];
         const real_t mag = std::hypot(a, b);
         if (mag > kAbsEps)
         {
            const real_t d_abs = (a * dr + b * di) / mag;
            const real_t c = (scale2 * d_abs) / mag;
            seed[2 * i] = c * a;
            seed[2 * i + 1] = c * b;
         }
         else
         {
            seed[2 * i] = 0.0;
            seed[2 * i + 1] = 0.0;
         }
      });
      prop.MultTranspose(data.seed, data.seed);
      lens.Apply(data.seed, true);
      RasterField::ToSamples(grid, data.seed, data.s);
      mask.Adjoint(data.s, out);
      out *= 2.0 / static_cast<real_t>(grid.NumSamples());
   }
};

// ────────────────────────────────────────────────────────────────────────────
class ObjectiveOp : public Operator
{
   PhaseMask &mask;
   PixelMesh &grid;
   const ParFresnel &prop;
   Lens &lens;
   Vector &E;
   Vector &target;
   GaussNewtonOp &op;
   Fields &data;

public:
   ObjectiveOp(PhaseMask &mask, PixelMesh &grid,
               const ParFresnel &prop, Lens &lens,
               Vector &E, Vector &target,
               GaussNewtonOp &op,
               Fields &data)
      : Operator(grid.Space().GetVSize()),
        mask(mask), grid(grid), prop(prop),
        lens(lens), E(E), target(target), op(op),
        data(data) {}

   real_t ForwardLoss(const Vector &phi, Vector &E_out) const
   {
      mask.Modulate(phi, data.u);
      RasterField::FromSamples(grid, data.u.real(), data.u.imag(), E_out);
      lens.Apply(E_out, false);
      prop.Mult(E_out, E_out);
      last_scale = holo::MatchTarget(E_out, target, data.dot_v, data.energy_v,
                                     grid.Reducer());
      RasterField::ToSamples(grid, E_out, data.s);
      last_J = mask.MeanAmplitudeError(data.s, data.g, data);
      return last_J;
   }

   void Mult(const Vector &phi, Vector &F) const override
   {
      F.SetSize(Width());
      ForwardLoss(phi, E);
      data.g.real() *= last_scale;
      data.g.imag() *= last_scale;
      data.g.SyncAlias();
      RasterField::FromSamples(grid, data.g.real(), data.g.imag(), data.gE);
      prop.MultTranspose(data.gE, data.gE);
      lens.Apply(data.gE, true);
      RasterField::ToSamples(grid, data.gE, data.s);
      op.SetState(phi, last_scale);
      mask.Adjoint(data.s, F);
   }

   Operator &GetGradient(const Vector &phi) const override
   {
      op.SetState(phi, last_scale);
      return op;
   }

   real_t LastLoss() const { return last_J; }

   mutable real_t last_scale = 0.0;
   mutable real_t last_J = 0.0;
};

// ────────────────────────────────────────────────────────────────────────────
class GaussNewtonCg : public Solver
{
   CGSolver &cg;
   GaussNewtonOp &op;

public:
   GaussNewtonCg(CGSolver &cg, GaussNewtonOp &op)
      : Solver(op.Height()), cg(cg), op(op) {}

   void SetOperator(const Operator &op_in) override
   {
      MFEM_VERIFY(&op_in == &op, "phase CG expects the Gauss-Newton operator");
      cg.SetOperator(op_in);
   }

   void Mult(const Vector &r, Vector &c) const override
   {
      c = 0.0;
      cg.Mult(r, c);
   }

   int Iterations() const { return cg.GetNumIterations(); }
};

// ────────────────────────────────────────────────────────────────────────────
class PhaseNewton : public NewtonSolver
{
   mutable real_t J_log_prev = -1.0;

public:
   PhaseNewton() : NewtonSolver(MPI_COMM_WORLD) {}

   ObjectiveOp *op = nullptr;
   PhaseMask *phase = nullptr;
   GaussNewtonOp *mask_op = nullptr;
   GaussNewtonCg *step = nullptr;
   Vector *trial_field = nullptr;
   Vector *trial = nullptr;
   real_t alpha0 = 1.0;
   int max_outer = 0;
   mutable int gn_ran = 0;
   mutable bool started = false;
   mutable bool line_search_failed = false;
   // Loss at the first Newton step and at the current iterate
   mutable real_t firstLoss = -1.0, finalLoss = -1.0;

   real_t ComputeScalingFactor(const Vector &x, const Vector &) const override
   {
      const real_t J0 = op->LastLoss();
      char drop[16] = "    n/a", loss[16];
      if (J_log_prev > 0.0)
      {
         const real_t rel = 100.0 * (J_log_prev - J0) / J_log_prev;
         std::snprintf(drop, sizeof(drop), "%6.2f%%", rel);
      }
      std::snprintf(loss, sizeof(loss), "%6.2e", J0);
      if (Mpi::Root())
      {
         out << "[hologram] Newton "
             << (gn_ran + 1) << "/" << max_outer
             << " (" << step->Iterations() << ")"
             << " " << loss << " " << drop << "\n";
      }
      J_log_prev = J0;
      if (firstLoss < 0.0) { firstLoss = J0; }
      finalLoss = J0;
      const real_t scale_at_x = op->last_scale;

      auto try_decrease = [&]() -> real_t
      {
         real_t alpha = alpha0;
         constexpr int kMaxBacktracks = 8;
         for (int bt = 0; bt < kMaxBacktracks; ++bt)
         {
            *trial = x;
            trial->Add(-alpha, c);
            phase->Wrap(*trial);
            const real_t J_new = op->ForwardLoss(*trial, *trial_field);
            if (J_new < J0)
            {
               finalLoss = J_new;
               mask_op->SetLambda(std::max(mask_op->Lambda() * 0.5, 1e-8));
               return alpha;
            }
            alpha *= 0.5;
         }
         return 0.0;
      };

      real_t a = try_decrease();
      if (a > 0.0) { return a; }
      mask_op->SetLambda(std::min(mask_op->Lambda() * 10.0, 1e4));
      mask_op->SetState(x, scale_at_x);
      step->SetOperator(*mask_op);
      step->Mult(r, c);
      a = try_decrease();
      if (a <= 0.0) { line_search_failed = true; }
      return a;
   }

   void ProcessNewState(const Vector &x) const override
   {
      phase->Wrap(const_cast<Vector &>(x));
      if (!started) { started = true; return; }
      ++gn_ran;
   }
};

// ────────────────────────────────────────────────────────────────────────────
class StallController : public IterativeSolverController
{
   real_t J_prev = -1.0;
   int stall = 0;

public:
   ObjectiveOp *op = nullptr;
   real_t rel_tol = 0.0;
   mutable const char *stop_reason = nullptr;

   void Reset() override
   {
      IterativeSolverController::Reset();
      J_prev = -1.0;
      stall = 0;
      stop_reason = nullptr;
   }

   void MonitorResidual(int it, real_t, const Vector &, bool final) override
   {
      if (final || op == nullptr || rel_tol <= 0.0) { return; }
      const real_t J = op->LastLoss();
      if (J_prev < 0.0) { J_prev = J; return; }
      const real_t rel = (J_prev - J) / std::max(J_prev, static_cast<real_t>(1e-30));
      stall = (rel < rel_tol) ? stall + 1 : 0;
      J_prev = J;
      constexpr int kMinNewtonIters = 4, kLossStallNeed = 3;
      if (it >= kMinNewtonIters && stall >= kLossStallNeed)
      {
         converged = true;
         stop_reason = "loss_stall";
      }
   }
};

// ────────────────────────────────────────────────────────────────────────────
class PhaseRetrieval
{
   PixelMesh mesh;
   RealArray2D target;
   const real_t wavelength, extent_x, extent_y, z, focal;
   const bool fft;
   RealArray2D mask_image;
   int acceptedSteps = 0;
   real_t firstLoss = 0.0, finalLoss = 0.0;
   std::unique_ptr<ParFresnel> propagator;

public:
   PhaseRetrieval(RealArray2D target, int order, real_t wavelength,
                  real_t extent_x, real_t extent_y,
                  real_t z, real_t focal,
                  bool fft = false)
      : mesh(target.NumCols(), target.NumRows(), order, extent_x, extent_y),
        target(std::move(target)),
        wavelength(wavelength), extent_x(extent_x), extent_y(extent_y),
        z(z), focal(focal), fft(fft)
   {
      mask_image.SetSize(mesh.SampleNy(), mesh.SampleNx());
      mask_image = 0.0;
   }

   void Solve(int max_iter, real_t stall_rel, real_t learning_rate,
              int cg_max_iter)
   {
      // retrieve the mask
      ParGridFunction phi(&mesh.Space());
      phi = 0.0;
      ParGridFunction a_src(&mesh.ElementSpace());
      a_src = 1.0;
      Vector target_global(mesh.NumSamples());
      target_global.UseDevice(true);
      {
         const real_t *s = target(0);
         real_t *d = target_global.HostWrite();
         for (int i = 0; i < mesh.NumSamples(); ++i) { d[i] = s[i]; }
      }
      // Local sample rows of the target
      const int localSamples = mesh.LocalSampleRows() * mesh.SampleNx();
      Vector target_image(localSamples);
      target_image.UseDevice(true);
      {
         const real_t *s = target(mesh.FirstSampleRow());
         real_t *d = target_image.HostWrite();
         for (int i = 0; i < localSamples; ++i) { d[i] = s[i]; }
      }
      Vector a_tgt_global(mesh.NumSamples());
      a_tgt_global.UseDevice(true);
      {
         const int n = mesh.NumSamples();
         const int nx = mesh.Nx();
         const int q1d = mesh.Q1D();
         const real_t *s = target_global.Read();
         real_t *d = a_tgt_global.Write();
         mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
         {
            d[i] = s[RasterIndex(i, nx, q1d)];
         });
      }
      ParGridFunction a_tgt(&mesh.Space());
      if (Mpi::WorldSize() == 1) { a_tgt = a_tgt_global; }
      else { mesh.ScatterDofs(a_tgt_global, a_tgt); }
      PhaseMask mask(mesh);
      mask.SetAmplitudes(a_src, a_tgt);

      // Create the propagator
      propagator = std::make_unique<ParFresnel>(mesh, fft, wavelength,
                                                extent_x, extent_y, z);

      // Create the lens
      Lens lens;
      lens(mesh.SampleNx(), mesh.SampleNy(),
           extent_x, extent_y, wavelength, focal,
           mesh.FirstSampleRow(), mesh.LocalSampleRows());

      Fields fields(mesh);

      GaussNewtonOp mask_op(mask, mesh, *propagator, lens, fields.E, fields);
      ObjectiveOp op(mask, mesh, *propagator, lens, fields.E,
                     target_image, mask_op, fields);
      op.ForwardLoss(phi, fields.E);

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetInnerProduct(&mesh.Reducer());
      cg.SetRelTol(1e-3);
      cg.SetAbsTol(1e-12);
      cg.SetMaxIter(std::max(1, cg_max_iter));
      cg.SetPrintLevel(-1);

      PhaseNewton newton;
      newton.iterative_mode = true;
      newton.SetRelTol(0.0);
      newton.SetAbsTol(1e-8);
      newton.SetMaxIter(std::max(0, max_iter));
      newton.SetPrintLevel(-1);
      newton.SetInnerProduct(&mesh.Reducer());
      GaussNewtonCg step(cg, mask_op);
      newton.SetOperator(op);
      newton.SetSolver(step);
      newton.op = &op;
      newton.phase = &mask;
      newton.mask_op = &mask_op;
      newton.step = &step;
      newton.trial_field = &fields.E_trial;
      ParGridFunction phi_trial(&mesh.Space());
      newton.trial = &phi_trial;
      newton.alpha0 = (learning_rate > 0.0) ? learning_rate : 1.0;
      newton.max_outer = std::max(0, max_iter);
      StallController stall;
      stall.op = &op;
      stall.rel_tol = stall_rel;
      newton.SetController(stall);

      if (Mpi::Root())
      {
         mfem::out << "[hologram] solve elements=" << mesh.Nx() << "x" << mesh.Ny()
                   << "  order=" << mesh.Order()
                   << "  samples=" << mesh.SampleNx() << "x" << mesh.SampleNy()
                   << "  forward=" << (fft ? "fft" : "gem")
                   << "  max_iter=" << max_iter
                   << "  cg_max_iter=" << std::max(1, cg_max_iter) << "\n";
      }

      Vector b(phi.Size());
      b.UseDevice(true);
      b = 0.0;

      mask_op.SetState(phi, op.last_scale);
      mask_op.SetLambda();
      step.SetOperator(mask_op);

      newton.Mult(b, phi);
      mask.Wrap(phi);
      mask.Modulate(phi, fields.u);

      RasterField aperture;
      aperture(mesh, fields.u.real(), fields.u.imag(),
               mask_image, fields);

      acceptedSteps = newton.gn_ran;
      firstLoss = newton.firstLoss;
      finalLoss = newton.finalLoss;

      const char *why = "max_iter";
      if (newton.line_search_failed) { why = "line_search"; }
      else if (stall.stop_reason) { why = stall.stop_reason; }
      else if (newton.GetConverged()) { why = "abs_tol"; }
      if (Mpi::Root())
      {
         mfem::out << "[hologram] Newton stop reason=" << why << "\n";
      }
   }

   const RealArray2D &MaskImage() const { return mask_image; }
   int AcceptedSteps() const { return acceptedSteps; }
   real_t FirstLoss() const { return firstLoss; }
   real_t FinalLoss() const { return finalLoss; }
   const ParFresnel &Propagator() const { return *propagator; }
   int SampleNx() const { return mesh.SampleNx(); }
   int SampleNy() const { return mesh.SampleNy(); }
   const PixelMesh &Grid() const { return mesh; }
};

} // namespace holo

int main(int argc, char *argv[]) try
{
   Mpi::Init(argc, argv);

   const char *in_file = MFEM_SOURCE_DIR "/doc/logo-small.png";
   const char *holo_file = "hologram.png";
   const char *mask_file = "mask.png";
   const char *device_config = "cpu";
   real_t wavelength_nm = 532.0;
   real_t z_prop = 0.80;
   real_t focal = 0.80;
   int nwt_max_iter = 50;
   int cgs_max_iter = 20;
   int order = 0;
   real_t stall_pct = 1.0;
   real_t gemMaxGib = 16.0;
   bool checkLoss = false;
#if defined(HOLO_USE_CPU_FFT)
   bool use_fft = true;
#else
   bool use_fft = false;
#endif

   OptionsParser args(argc, argv);
   args.AddOption(&in_file, "-i", "--input", "Target PNG.");
   args.AddOption(&holo_file, "-o", "--output", "Hologram sRGB PNG.");
   args.AddOption(&mask_file, "-m", "--mask", "Phase-mask PNG.");
   args.AddOption(&wavelength_nm, "-w", "--wavelength",
                  "Wavelength in nanometres (default 532).");
   args.AddOption(&nwt_max_iter, "-mi", "--max-iterations",
                  "Maximum Newton iterations (default 50).");
   args.AddOption(&cgs_max_iter, "-cg", "--cg-iterations",
                  "Maximum CG iterations per Newton step (default 20).");
   args.AddOption(&order, "-p", "--order",
                  "Pixel block degree. One block covers (order+1)^2 pixels "
                  "(default 0).");
   args.AddOption(&z_prop, "-z", "--z",
                  "Propagation distance in metres (default 0.80).");
   args.AddOption(&focal, "-f", "--focal",
                  "Lens focal length in metres (default 0.80).");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&use_fft, "-fft", "--fft", "-gem", "--gem",
                  "Toeplitz FFT or the dense product (-gem). The default is "
                  "-fft when the library for the device is linked (FFTW on "
                  "cpu, cuFFT or hipFFT on CUDA or HIP).");
   args.AddOption(&stall_pct, "-stall", "--loss-stall",
                  "Newton stall relative loss drop in percent (default 1).");
   args.AddOption(&gemMaxGib, "-gmem", "--gem-max-gib",
                  "Largest memory in GiB for the dense -gem Fresnel matrices "
                  "(default 16).");
   args.AddOption(&checkLoss, "-check", "--check", "-no-check", "--no-check",
                  "Exit with an error unless a Newton step was accepted and "
                  "the final loss is below the first loss (default off).");

   bool chose_propagator = false;
   for (int i = 1; i < argc; ++i)
   {
      const char *arg = argv[i];
      if (std::strcmp(arg, "-fft") == 0 || std::strcmp(arg, "--fft") == 0 ||
          std::strcmp(arg, "-gem") == 0 || std::strcmp(arg, "--gem") == 0)
      {
         chose_propagator = true;
      }
   }
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(out); }
      if (Mpi::IsInitialized()) { Mpi::Finalize(); }
      std::exit(EXIT_FAILURE);
   }

   const std::string optionsError =
      holo::OptionsError(wavelength_nm, z_prop, focal, stall_pct, order,
                         nwt_max_iter, cgs_max_iter, gemMaxGib);
   if (!optionsError.empty())
   {
      if (Mpi::Root()) { mfem::err << optionsError << "\n"; }
      return EXIT_FAILURE;
   }

   const real_t wavelength = wavelength_nm * 1e-9;

   // Configure the device, then pick the default propagator for that device.
   Device device(device_config);
   if (!chose_propagator) { use_fft = holo::FftLinkedForDevice(); }
   if (Mpi::Root())
   {
      args.PrintOptions(out);
      device.Print();
   }

   // Load the target image on rank 0
   holo::GrayImage gray;
   std::string loadError;
   if (Mpi::Root())
   {
      try
      {
         gray = holo::LoadPngGray(in_file);
         if (gray.width < 1 || gray.height < 1)
         {
            loadError = "Empty target PNG";
         }
      }
      catch (const std::exception &exc) { loadError = exc.what(); }
   }
   if (!holo::RootOk(loadError)) { return EXIT_FAILURE; }
   int imgSize[2] = {gray.width, gray.height};
   MPI_Bcast(imgSize, 2, MPI_INT, 0, MPI_COMM_WORLD);

   const int img_w = imgSize[0], img_h = imgSize[1];
   if (Mpi::Root())
   {
      out << "Loaded " << img_w << "x" << img_h
          << " from " << in_file << "\n";
   }

   // Compute mask's extent
   const real_t extent_x = 19.0e-3;
   const real_t extent_y =
      extent_x * (static_cast<real_t>(img_h) /
                  static_cast<real_t>(img_w));

   // Check the image size against the order, the propagator and the ranks,
   // the optics terms and the -gem memory
   int nodeRanks = 1, deviceRanks = 1;
   holo::NodeRanks(nodeRanks, deviceRanks);
   std::string imageError = holo::ImageError(img_w, img_h, order, use_fft);
   if (imageError.empty() && Mpi::WorldSize() > img_h / (order + 1))
   {
      const int elemRows = img_h / (order + 1);
      imageError = "Too many MPI ranks: " + std::to_string(Mpi::WorldSize()) +
                   " ranks for " + std::to_string(elemRows) +
                   " element rows (image height " + std::to_string(img_h) +
                   ", order+1 = " + std::to_string(order + 1) +
                   "), use at most " + std::to_string(elemRows) + " ranks";
   }
   if (imageError.empty())
   {
      imageError = holo::OpticsError(wavelength_nm, z_prop, focal, extent_x,
                                     extent_y, img_w, img_h);
   }
   if (imageError.empty() && !use_fft)
   {
      imageError = holo::GemMemoryError(img_w, img_h, gemMaxGib, nodeRanks,
                                        deviceRanks);
   }
   if (!holo::AllOk(imageError)) { return EXIT_FAILURE; }

   // Broadcast the target image
   gray.width = img_w, gray.height = img_h;
   gray.pixels.resize(static_cast<std::size_t>(img_w) *
                      static_cast<std::size_t>(img_h));
   MPI_Bcast(gray.pixels.data(), img_w * img_h, MPI_FLOAT, 0, MPI_COMM_WORLD);

   // Convert the target image to a 2D array of real values
   auto amplitude = holo::GrayToReal2D(gray);

   // Create the solver
   holo::PhaseRetrieval solver(std::move(amplitude), order, wavelength,
                               extent_x, extent_y, z_prop, focal, use_fft);

   // Solve the phase retrieval problem
   solver.Solve(nwt_max_iter, stall_pct * 0.01, 1.0, cgs_max_iter);

   // Get the phase mask
   const RealArray2D &phase = solver.MaskImage();
   if (phase.NumCols() != solver.SampleNx() ||
       phase.NumRows() != solver.SampleNy())
   {
      throw std::runtime_error("Mask size does not match the sample grid");
   }

   // Write the mask to a PNG file on rank 0
   std::string maskError;
   if (Mpi::Root())
   {
      try
      {
         if (!holo::WritePhasePng(mask_file, phase))
         {
            maskError = std::string("Failed to write ") + mask_file;
         }
      }
      catch (const std::exception &exc) { maskError = exc.what(); }
      if (maskError.empty())
      {
         mfem::out << "Wrote mask " << phase.NumCols() << "x"
                   << phase.NumRows() << " " << mask_file << "\n";
      }
   }
   if (!holo::RootOk(maskError)) { return EXIT_FAILURE; }

   const auto propagator = &solver.Propagator();
   const holo::PixelMesh &grid = solver.Grid();
   const int firstRow = grid.FirstSampleRow(), rows = grid.LocalSampleRows();
   const int nx = phase.NumCols(), ny = phase.NumRows(), n = nx * rows;
   Vector phase_v(n), field(2 * n), Iv(n);
   phase_v.UseDevice(true);
   field.UseDevice(true);
   Iv.UseDevice(true);
   {
      const real_t *src = phase.GetRow(firstRow);
      real_t *h = phase_v.HostWrite();
      for (int k = 0; k < n; ++k) { h[k] = src[k]; }
   }
   const real_t *ph = phase_v.Read();
   real_t *p = field.Write();
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int idx)
   {
      p[2 * idx] = std::cos(ph[idx]);
      p[2 * idx + 1] = std::sin(ph[idx]);
   });
   holo::Lens lens;
   lens(nx, ny, extent_x, extent_y, wavelength, focal, firstRow, rows);
   lens.Apply(field);

   // Propagate the field
   propagator->Propagate(field);

   // Compute the intensity
   const auto *f = field.Read();
   auto *Id = Iv.Write();
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int k)
   {
      const real_t re = f[2 * k], im = f[2 * k + 1];
      Id[k] = re * re + im * im;
   });
   RealArray2D intensity;
   if (Mpi::Root()) { intensity.SetSize(ny, nx); }
   MPI_Gatherv(const_cast<real_t *>(Iv.HostRead()), n,
               MPITypeMap<real_t>::mpi_type,
               Mpi::Root() ? intensity(0) : nullptr,
               grid.SampleCounts().GetData(), grid.SampleDispls().GetData(),
               MPITypeMap<real_t>::mpi_type, 0, MPI_COMM_WORLD);

   // Write the hologram to the output PNG file on rank 0
   std::string holoError;
   if (Mpi::Root())
   {
      std::vector<unsigned char> rgb;
      int w = 0, h = 0;
      try
      {
         const bool encoded =
            holo::EncodeMonoWavelengthRgb(intensity, wavelength_nm, rgb, w, h);
         if (!encoded || !holo::WritePngRgb(holo_file, w, h, rgb))
         {
            holoError = std::string("Failed to write ") + holo_file;
         }
      }
      catch (const std::exception &exc) { holoError = exc.what(); }
      if (holoError.empty())
      {
         mfem::out << "Wrote " << w << "x" << h << " " << holo_file << "\n";
      }
   }
   if (!holo::RootOk(holoError)) { return EXIT_FAILURE; }

   // Check the loss decrease
   if (checkLoss)
   {
      const std::string checkError =
         holo::LossCheckError(solver.AcceptedSteps(), solver.FirstLoss(),
                              solver.FinalLoss());
      if (!holo::AllOk(checkError)) { return EXIT_FAILURE; }
      if (Mpi::Root())
      {
         mfem::out << "Loss check passed: " << solver.AcceptedSteps()
                   << " accepted Newton steps, loss " << solver.FirstLoss()
                   << " -> " << solver.FinalLoss() << "\n";
      }
   }
   return EXIT_SUCCESS;
}
catch (const std::exception &exc)
{
   mfem::err << exc.what() << "\n";
   return EXIT_FAILURE;
}
