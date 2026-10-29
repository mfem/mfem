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
#include <vector>

namespace holo
{

// ----------------------------------------------------------------------------
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
      // Add in global row order. MPI_Allreduce of the local totals would
      // follow an order that changes with the rank count.
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

// ----------------------------------------------------------------------------
class PixelMesh
{
   L2_FECollection fec, fec0;
   std::unique_ptr<ParMesh> mesh;
   std::unique_ptr<ParFiniteElementSpace> fes, fes0;

   const int order, q1d, nx, ny;
   const int y0, y1, elem0;
   Array<int> counts, displs, rowCounts, rowDispls;
   mutable ParRowSum rowSum;

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
      : fec(VerifyPixelOrder(pixels_x, pixels_y, order_in), /*dim=*/2,
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
      Mesh serial = CenteredQuadMesh(nx, ny, extent_x, extent_y);

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
   int PhaseScratch() const { return NumSamples(); }
   bool NeedsPhaseGather() const { return Mpi::WorldSize() != 1; }

   // Sample counts and offsets of all ranks
   const Array<int> &SampleCounts() const { return counts; }
   const Array<int> &SampleDispls() const { return displs; }
   int GlobalDof0() const
   {
      return SampleDispls()[Mpi::WorldRank()];
   }

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

// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
using PhaseRetrieval =
   PhaseRetrievalImpl<PixelMesh, ParGridFunction,
                      ParComplexGridFunction, ParFresnel>;

} // namespace holo

// ----------------------------------------------------------------------------
int main(int argc, char *argv[]) try
{
   Mpi::Init(argc, argv);

   holo::HologramConfig cfg;
   OptionsParser args(argc, argv);
   cfg.AddTo(args);
   cfg.NotePropagator(argc, argv);
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(out); }
      if (Mpi::IsInitialized()) { Mpi::Finalize(); }
      std::exit(EXIT_FAILURE);
   }

   const std::string optionsError =
      holo::OptionsError(cfg.wavelength_nm, cfg.z_prop, cfg.focal,
                         cfg.stall_pct, cfg.order, cfg.nwt_max_iter,
                         cfg.cgs_max_iter, cfg.gemMaxGib);
   if (!optionsError.empty())
   {
      if (Mpi::Root()) { mfem::err << optionsError << "\n"; }
      return EXIT_FAILURE;
   }

   const real_t wavelength = cfg.wavelength_nm * 1e-9;

   // Configure the device, then pick the default propagator for that device.
   Device device(cfg.device_config);
   if (!cfg.chose_propagator) { cfg.use_fft = holo::FftLinkedForDevice(); }
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
         gray = holo::LoadPngGray(cfg.in_file);
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
          << " from " << cfg.in_file << "\n";
   }

   // Compute mask's extent
   real_t extent_x, extent_y;
   holo::MaskExtent(img_w, img_h, extent_x, extent_y);

   // Check the image size against the order, the propagator and the ranks,
   // the optics terms and the -gem memory
   int nodeRanks = 1, deviceRanks = 1;
   holo::NodeRanks(nodeRanks, deviceRanks);
   std::string imageError = holo::ImageError(img_w, img_h, cfg.order,
                                              cfg.use_fft);
   if (imageError.empty() && Mpi::WorldSize() > img_h / (cfg.order + 1))
   {
      const int elemRows = img_h / (cfg.order + 1);
      imageError = "Too many MPI ranks: " + std::to_string(Mpi::WorldSize()) +
                   " ranks for " + std::to_string(elemRows) +
                   " element rows (image height " + std::to_string(img_h) +
                   ", order+1 = " + std::to_string(cfg.order + 1) +
                   "), use at most " + std::to_string(elemRows) + " ranks";
   }
   if (imageError.empty())
   {
      imageError = holo::OpticsError(cfg.wavelength_nm, cfg.z_prop, cfg.focal,
                                     extent_x, extent_y, img_w, img_h);
   }
   if (imageError.empty() && !cfg.use_fft)
   {
      imageError = holo::GemMemoryError(img_w, img_h, cfg.gemMaxGib, nodeRanks,
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
   holo::PhaseRetrieval solver(std::move(amplitude), cfg.order, wavelength,
                               extent_x, extent_y, cfg.z_prop, cfg.focal,
                               cfg.use_fft);

   // Solve the phase retrieval problem
   solver.Solve(cfg.nwt_max_iter, cfg.stall_pct * 0.01, 1.0, cfg.cgs_max_iter,
                Mpi::Root(),
                [&]()
   {
      return std::make_unique<holo::ParFresnel>(solver.Grid(), cfg.use_fft,
                                                wavelength, extent_x,
                                                extent_y, cfg.z_prop);
   },
   [] { return CGSolver(MPI_COMM_WORLD); },
   [] { return holo::PhaseRetrieval::Newton(MPI_COMM_WORLD); });

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
         if (!holo::WritePhasePng(cfg.mask_file, phase))
         {
            maskError = std::string("Failed to write ") + cfg.mask_file;
         }
      }
      catch (const std::exception &exc) { maskError = exc.what(); }
      if (maskError.empty())
      {
         mfem::out << "Wrote mask " << phase.NumCols() << "x"
                   << phase.NumRows() << " " << cfg.mask_file << "\n";
      }
   }
   if (!holo::RootOk(maskError)) { return EXIT_FAILURE; }

   const auto propagator = &solver.Propagator();
   const holo::PixelMesh &grid = solver.Grid();
   const int firstRow = grid.FirstSampleRow(), rows = grid.LocalSampleRows();
   const int nx = phase.NumCols(), ny = phase.NumRows(), n = nx * rows;
   Vector Iv(n);
   holo::ReplayIntensity(*propagator, phase, firstRow, rows,
                         extent_x, extent_y, wavelength, cfg.focal, Iv);
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
            holo::EncodeMonoWavelengthRgb(intensity, cfg.wavelength_nm,
                                          rgb, w, h);
         if (!encoded || !holo::WritePngRgb(cfg.holo_file, w, h, rgb))
         {
            holoError = std::string("Failed to write ") + cfg.holo_file;
         }
      }
      catch (const std::exception &exc) { holoError = exc.what(); }
      if (holoError.empty())
      {
         mfem::out << "Wrote " << w << "x" << h << " " << cfg.holo_file << "\n";
      }
   }
   if (!holo::RootOk(holoError)) { return EXIT_FAILURE; }

   // Check the loss decrease
   if (cfg.checkLoss)
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
   if (Mpi::IsInitialized() && !Mpi::IsFinalized())
   {
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
   }
   return EXIT_FAILURE;
}
