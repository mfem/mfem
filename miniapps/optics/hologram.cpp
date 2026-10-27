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

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace holo
{

class PixelMesh
{
   L2_FECollection fec, fec0;
   std::unique_ptr<Mesh> mesh;
   std::unique_ptr<FiniteElementSpace> fes, fes0;

   const int order, q1d, nx, ny;
   mutable RowSum rowSum;

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
        rowSum(nx * q1d * q1d)
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

   FiniteElementSpace &Space() const { return *fes; }
   FiniteElementSpace &ElementSpace() const { return *fes0; }

   RowSum &Reducer() const { return rowSum; }
};

struct Fields
{
   ComplexGridFunction u, du, s, g;
   Vector dot_v, energy_v, loss_q, phase_v;
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
      set_size(dot_v, n);
      set_size(energy_v, n);
      set_size(loss_q, nloc);
      set_size(phase_v, n);

      const int nx = grid.SampleNx(), ny = grid.SampleNy();
      SetInterleavedSize(E, nx, ny);
      SetInterleavedSize(E_trial, nx, ny);
      SetInterleavedSize(gE, nx, ny);
      SetInterleavedSize(dE, nx, ny);
      SetInterleavedSize(seed, nx, ny);

      for (Vector *v :
           {
              &dot_v, &energy_v, &loss_q,
              &phase_v,
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
      const int nx = grid.Nx();
      const int q1d = grid.Q1D();
      const real_t *er = re_s.Read();
      const real_t *ei = im_s.Read();
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
      SetInterleavedSize(E, grid.SampleNx(), grid.SampleNy());
      Pack(grid.Q1D(), grid.NumSamples(), grid.Nx(), re_s, im_s, E);
   }

   static void ToSamples(const PixelMesh &grid, const Vector &E,
                         Vector &re_s, Vector &im_s)
   {
      Unpack(grid.Q1D(), grid.NumSamples(), grid.Nx(), E, re_s, im_s);
   }

   static void ToSamples(const PixelMesh &grid, const Vector &E,
                         ComplexGridFunction &field)
   {
      ToSamples(grid, E, field.real(), field.imag());
      field.SyncAlias();
   }
};

// ────────────────────────────────────────────────────────────────────────────
using PhaseMask =
   PhaseMaskImpl<PixelMesh, GridFunction, ComplexGridFunction, Fields>;

using GaussNewtonOp =
   GaussNewtonOpImpl<PixelMesh, PhaseMask, Fields, RasterField>;
using ObjectiveOp =
   ObjectiveOpImpl<PixelMesh, PhaseMask, Fields, RasterField>;
using GaussNewtonCg =
   GaussNewtonCgImpl<PixelMesh, PhaseMask, Fields, RasterField>;
using PhaseNewton =
   PhaseNewtonImpl<PixelMesh, PhaseMask, Fields, RasterField>;
using StallController =
   StallControllerImpl<PixelMesh, PhaseMask, Fields, RasterField>;

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
   std::unique_ptr<FresnelOp> propagator;

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
      GridFunction phi(&mesh.Space());
      phi = 0.0;
      GridFunction a_src(&mesh.ElementSpace());
      a_src = 1.0;
      Vector target_image(mesh.NumSamples());
      target_image.UseDevice(true);
      {
         const real_t *s = target(0);
         real_t *d = target_image.HostWrite();
         for (int i = 0; i < mesh.NumSamples(); ++i) { d[i] = s[i]; }
      }
      Vector a_tgt_global(mesh.NumSamples());
      a_tgt_global.UseDevice(true);
      {
         const int n = mesh.NumSamples();
         const int nx = mesh.Nx();
         const int q1d = mesh.Q1D();
         const real_t *s = target_image.Read();
         real_t *d = a_tgt_global.Write();
         mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
         {
            d[i] = s[RasterIndex(i, nx, q1d)];
         });
      }
      GridFunction a_tgt(&mesh.Space());
      a_tgt = a_tgt_global;
      PhaseMask mask(mesh);
      mask.SetAmplitudes(a_src, a_tgt);

      // Create the propagator
      propagator = MakeFresnelOp(fft);
      propagator->Assemble(wavelength, extent_x, extent_y, z,
                           mesh.SampleNx(), mesh.SampleNy());

      // Create the lens
      Lens lens;
      lens(mesh.SampleNx(), mesh.SampleNy(),
           extent_x, extent_y, wavelength, focal);

      Fields fields(mesh);

      GaussNewtonOp mask_op(mask, mesh, *propagator, lens, fields.E, fields);
      ObjectiveOp op(mask, mesh, *propagator, lens, fields.E,
                     target_image, mask_op, fields);
      op.ForwardLoss(phi, fields.E);

      CGSolver cg;
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
      GridFunction phi_trial(&mesh.Space());
      newton.trial = &phi_trial;
      newton.alpha0 = (learning_rate > 0.0) ? learning_rate : 1.0;
      newton.max_outer = std::max(0, max_iter);
      StallController stall;
      stall.op = &op;
      stall.rel_tol = stall_rel;
      newton.SetController(stall);

      mfem::out << "[hologram] solve elements=" << mesh.Nx() << "x" << mesh.Ny()
                << "  order=" << mesh.Order()
                << "  samples=" << mesh.SampleNx() << "x" << mesh.SampleNy()
                << "  forward=" << (fft ? "fft" : "gem")
                << "  max_iter=" << max_iter
                << "  cg_max_iter=" << std::max(1, cg_max_iter) << "\n";

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
      out << "[hologram] Newton stop reason=" << why << "\n";
   }

   const RealArray2D &MaskImage() const { return mask_image; }
   int AcceptedSteps() const { return acceptedSteps; }
   real_t FirstLoss() const { return firstLoss; }
   real_t FinalLoss() const { return finalLoss; }
   const FresnelOp &Propagator() const { return *propagator; }
   int SampleNx() const { return mesh.SampleNx(); }
   int SampleNy() const { return mesh.SampleNy(); }
};

} // namespace holo

int main(int argc, char *argv[]) try
{
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
                  "Newton stall relative loss drop in percent, 0 disables "
                  "the stall stop (default 1).");
   args.AddOption(&gemMaxGib, "-gmem", "--gem-max-gib",
                  "Largest memory in GiB for the dense -gem Fresnel matrices "
                  "and the work image (default 16).");
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
      args.PrintUsage(out);
      std::exit(EXIT_FAILURE);
   }

   const std::string optionsError =
      holo::OptionsError(wavelength_nm, z_prop, focal, stall_pct, order,
                         nwt_max_iter, cgs_max_iter, gemMaxGib);
   if (!optionsError.empty())
   {
      mfem::err << optionsError << "\n";
      return EXIT_FAILURE;
   }

   const real_t wavelength = wavelength_nm * 1e-9;

   // Configure the device, then pick the default propagator for that device.
   Device device(device_config);
   if (!chose_propagator) { use_fft = holo::FftLinkedForDevice(); }
   args.PrintOptions(out);
   device.Print();

   // Load the target image
   const auto gray = holo::LoadPngGray(in_file);
   if (gray.width < 1 || gray.height < 1)
   {
      throw std::runtime_error("Empty target PNG");
   }

   const int img_w = gray.width, img_h = gray.height;
   out << "Loaded " << img_w << "x" << img_h
       << " from " << in_file << "\n";

   // Compute mask's extent
   const real_t extent_x = 19.0e-3;
   const real_t extent_y =
      extent_x * (static_cast<real_t>(img_h) /
                  static_cast<real_t>(img_w));

   // Check the image size against the order and the propagator, the optics
   // terms and the -gem memory
   std::string imageError = holo::ImageError(img_w, img_h, order, use_fft);
   if (imageError.empty())
   {
      imageError = holo::OpticsError(wavelength_nm, z_prop, focal, extent_x,
                                     extent_y, img_w, img_h);
   }
   if (imageError.empty() && !use_fft)
   {
      imageError = holo::GemMemoryError(img_w, img_h, gemMaxGib);
   }
   if (!imageError.empty())
   {
      mfem::err << imageError << "\n";
      return EXIT_FAILURE;
   }

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

   // Write the mask to a PNG file
   if (!holo::WritePhasePng(mask_file, phase))
   {
      throw std::runtime_error(std::string("Failed to write ") + mask_file);
   }
   mfem::out << "Wrote mask " << phase.NumCols() << "x"
             << phase.NumRows() << " " << mask_file << "\n";

   const auto propagator = &solver.Propagator();
   const int nx = phase.NumCols(), ny = phase.NumRows(), n = nx * ny;
   Vector phase_v(n), field(2 * n), Iv(n);
   phase_v.UseDevice(true);
   field.UseDevice(true);
   Iv.UseDevice(true);
   {
      const real_t *src = phase.GetRow(0);
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
   lens(nx, ny, extent_x, extent_y, wavelength, focal);
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
   RealArray2D intensity(ny, nx);
   const real_t *Ih = Iv.HostRead();
   real_t *I = intensity(0);
   for (int k = 0; k < n; ++k) { I[k] = Ih[k]; }

   // Write the hologram to the output PNG file
   std::vector<unsigned char> rgb;
   int w = 0, h = 0;
   if (!holo::EncodeMonoWavelengthRgb(intensity, wavelength_nm, rgb, w, h) ||
       !holo::WritePngRgb(holo_file, w, h, rgb))
   {
      throw std::runtime_error(std::string("Failed to write ") + holo_file);
   }
   mfem::out << "Wrote " << w << "x" << h << " " << holo_file << "\n";

   // Check the loss decrease
   if (checkLoss)
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
