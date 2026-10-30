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
// Sample runs:  hologram
//               hologram -p 5
//               hologram -p 5 -gem
//               hologram -gem -mi 2 -check

#include "hologram.hpp"
using namespace mfem;

#include <cstdlib>

namespace holo
{

// ----------------------------------------------------------------------------
class PixelMesh
{
   L2_FECollection fec, fec0;
   std::unique_ptr<Mesh> mesh;
   std::unique_ptr<FiniteElementSpace> fes, fes0;

   const int order, q1d, nx, ny;
   mutable RowSum rowSum;

public:
   PixelMesh(PixelMesh &&) = delete;
   PixelMesh(const PixelMesh &) = delete;
   PixelMesh &operator=(PixelMesh &&) = delete;
   PixelMesh &operator=(const PixelMesh &) = delete;

   PixelMesh(int pixels_x, int pixels_y, int order_in,
             real_t extent_x, real_t extent_y)
      : fec(VerifyPixelOrder(pixels_x, pixels_y, order_in),
            2, BasisType::OpenHalfUniform),
        fec0(0, 2, BasisType::OpenHalfUniform),
        order(order_in),
        q1d(order_in + 1),
        nx(pixels_x / (order_in + 1)),
        ny(pixels_y / (order_in + 1)),
        rowSum(nx * q1d * q1d)
   {
      Mesh serial = CenteredQuadMesh(nx, ny, extent_x, extent_y);

      mesh = std::make_unique<Mesh>(std::move(serial));
      fes = std::make_unique<FiniteElementSpace>(mesh.get(), &fec);
      fes0 = std::make_unique<FiniteElementSpace>(mesh.get(), &fec0);

      const int ne = mesh->GetNE();
      MFEM_VERIFY(fes->GetVSize() == ne * q1d * q1d, "GetVSize sample count");
      MFEM_VERIFY(fes0->GetVSize() == ne, "order-0 element space");
      MFEM_VERIFY(ne > 0, "Empty mesh not supported");
      mfem::out << "[hologram] mesh quads=" << nx << "x" << ny
                << " " << fec.Name()
                << " dofs=" << NumSamples()
                << " local=" << LocalNumSamples()
                << " rows=" << 0 << ":" << ny << "\n";
   }

   int Order() const { return order; }
   int Q1D() const { return q1d; }
   int Nx() const { return nx; }
   int Ny() const { return ny; }

   int SampleNx() const { return nx * q1d; }
   int SampleNy() const { return ny * q1d; }
   int NumSamples() const { return SampleNx() * SampleNy(); }

   int LocalNumSamples() const { return fes->GetVSize(); }
   int LocalSampleRows() const { return SampleNy(); }
   int FirstSampleRow() const { return 0; }
   int GlobalDof0() const { return 0; }
   int PhaseScratch() const { return 0; }
   bool NeedsPhaseGather() const { return false; }

   FiniteElementSpace &Space() const { return *fes; }
   FiniteElementSpace &ElementSpace() const { return *fes0; }

   RowSum &Reducer() const { return rowSum; }

   void GatherDofs(const Vector &, Vector &) const
   {
      MFEM_ABORT("Serial PixelMesh has no phase gather");
   }
};

} // namespace holo

// ----------------------------------------------------------------------------
int main(int argc, char *argv[]) try
{
   holo::HologramConfig cfg;
   OptionsParser args(argc, argv);
   cfg.AddTo(args);
   cfg.NotePropagator(argc, argv);
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(out);
      std::exit(EXIT_FAILURE);
   }

   const std::string optionsError =
      holo::OptionsError(cfg.wavelength_nm, cfg.z_prop, cfg.focal,
                         cfg.stall_pct, cfg.order, cfg.nwt_max_iter,
                         cfg.cgs_max_iter, cfg.gemMaxGib);
   if (!optionsError.empty())
   {
      mfem::err << optionsError << "\n";
      return EXIT_FAILURE;
   }

   const real_t wavelength = cfg.wavelength_nm * 1e-9;

   // Configure the device, then pick the default propagator for that device
   Device device(cfg.device_config);
   if (!cfg.chose_propagator) { cfg.use_fft = holo::FftLinkedForDevice(); }
   args.PrintOptions(out);
   device.Print();

   // Load the target image
   const auto gray = holo::LoadPngGray(cfg.in_file);
   if (gray.width < 1 || gray.height < 1)
   {
      throw std::runtime_error("Empty target PNG");
   }

   const int img_w = gray.width, img_h = gray.height;
   mfem::out << "Loaded " << img_w << "x" << img_h
             << " from " << cfg.in_file << "\n";

   // Compute mask's extent
   real_t extent_x, extent_y;
   holo::MaskExtent(img_w, img_h, extent_x, extent_y);

   // Check the image size against the order and the propagator
   std::string imageError =
      holo::ImageError(img_w, img_h, cfg.order, cfg.use_fft);
   if (imageError.empty())
   {
      imageError = holo::OpticsError(cfg.wavelength_nm, cfg.z_prop, cfg.focal,
                                     extent_x, extent_y, img_w, img_h);
   }
   if (imageError.empty() && !cfg.use_fft)
   {
      imageError = holo::GemMemoryError(img_w, img_h, cfg.gemMaxGib);
   }
   if (!imageError.empty())
   {
      mfem::err << imageError << "\n";
      return EXIT_FAILURE;
   }

   // Convert the target image to a 2D array of real values
   auto amplitude = holo::GrayToReal2D(gray);

   // Create the solver

   using PhaseRetrieval =
      holo::PhaseRetrievalImpl<holo::PixelMesh, GridFunction,
      ComplexGridFunction, holo::FresnelOp>;
   PhaseRetrieval solver(std::move(amplitude), cfg.order, wavelength,
                         extent_x, extent_y, cfg.z_prop, cfg.focal,
                         cfg.use_fft);

   // Solve the phase retrieval problem
   solver.Solve(cfg.nwt_max_iter, cfg.stall_pct * 0.01, 1.0, cfg.cgs_max_iter,
                /*root=*/true,
                [&]()
   {
      auto prop = holo::MakeFresnelOp(cfg.use_fft);
      prop->Assemble(wavelength, extent_x, extent_y, cfg.z_prop,
                     solver.SampleNx(), solver.SampleNy());
      return prop;
   },
   [] { return CGSolver(); },
   [] { return PhaseRetrieval::Newton(); });

   // Get the phase mask
   const RealArray2D &phase = solver.MaskImage();
   if (phase.NumCols() != solver.SampleNx() ||
       phase.NumRows() != solver.SampleNy())
   {
      throw std::runtime_error("Mask size does not match the sample grid");
   }

   // Write the mask to a PNG file
   if (!holo::WritePhasePng(cfg.mask_file, phase))
   {
      throw std::runtime_error(std::string("Failed to write ") + cfg.mask_file);
   }
   mfem::out << "Wrote mask " << phase.NumCols() << "x"
             << phase.NumRows() << " " << cfg.mask_file << "\n";

   const auto propagator = &solver.Propagator();
   const int nx = phase.NumCols(), ny = phase.NumRows(), n = nx * ny;
   Vector Iv(n);
   holo::ReplayIntensity(*propagator, phase, 0, solver.SampleNy(),
                         extent_x, extent_y, wavelength, cfg.focal, Iv);
   RealArray2D intensity(ny, nx);
   const real_t *Ih = Iv.HostRead();
   real_t *I = intensity(0);
   for (int k = 0; k < n; ++k) { I[k] = Ih[k]; }

   // Write the hologram to the output PNG file
   std::vector<unsigned char> rgb;
   int w = 0, h = 0;
   if (!holo::EncodeMonoWavelengthRgb(intensity, cfg.wavelength_nm,
                                      rgb, w, h) ||
       !holo::WritePngRgb(cfg.holo_file, w, h, rgb))
   {
      throw std::runtime_error(std::string("Failed to write ") + cfg.holo_file);
   }
   mfem::out << "Wrote " << w << "x" << h << " " << cfg.holo_file << "\n";

   // Check the loss decrease
   if (cfg.checkLoss)
   {
      const std::string checkError =
         holo::LossCheckError(solver.AcceptedSteps(), solver.FirstLoss(),
                              solver.FinalLoss());
      if (!checkError.empty())
      {
         mfem::err << checkError << "\n";
         return EXIT_FAILURE;
      }
      mfem::out << "Loss check passed: " << solver.AcceptedSteps()
                << " accepted Newton steps, loss " << solver.FirstLoss()
                << " -> " << solver.FinalLoss() << "\n";
   }
   return EXIT_SUCCESS;
}
catch (const std::exception &exc)
{
   mfem::err << exc.what() << "\n";
   return EXIT_FAILURE;
}
