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
        ny(pixels_y / (order_in + 1))
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
class PhaseMask
{
   PixelMesh &grid;
   const GridFunction *src, *tgt;
   ComplexGridFunction lin;
   static constexpr real_t kLossEps = 1e-24;

public:
   explicit PhaseMask(PixelMesh &grid)
      : grid(grid), src(nullptr), tgt(nullptr), lin(&grid.Space()) {}

   void SetAmplitudes(const GridFunction &a_src_in,
                      const GridFunction &a_tgt_in)
   {
      src = &a_src_in;
      tgt = &a_tgt_in;
   }

   // Fold the phase into (-π, π]
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
   void Modulate(const Vector &phi, ComplexGridFunction &field) const
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

   void Derivative(const Vector &dphi, ComplexGridFunction &dfield) const
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
   void Adjoint(const ComplexGridFunction &dfield, Vector &g_phi) const
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
   real_t MeanAmplitudeError(const ComplexGridFunction &field,
                             ComplexGridFunction &grad, Fields &data) const
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
      real_t sum = data.loss_q.Sum();
      return sum * inv_n;
   }
};

// ────────────────────────────────────────────────────────────────────────────
class GaussNewtonOp : public Operator
{
   PhaseMask &mask;
   const PixelMesh &grid;
   const FresnelOp &prop;
   const Lens &lens;
   const Vector &E_scaled;
   Fields &data;
   const Vector *phi = nullptr;
   static constexpr real_t kAbsEps = 1e-14;
   static constexpr real_t kLmLambda0 = 1e-2;
   real_t scale = 0.0, lambda = kLmLambda0;

public:
   GaussNewtonOp(PhaseMask &mask, const PixelMesh &grid,
                 const FresnelOp &prop, const Lens &lens,
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
         const real_t inv_n =
            1.0 / static_cast<real_t>(InterleavedSize(data.dE));
         out.Add(lambda * inv_n, dphi);
      }
   }

   void Assemble(Vector &out) const
   {
      RasterField::FromSamples(grid, data.du.real(), data.du.imag(),
                               data.dE);
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
      out *= 2.0 / static_cast<real_t>(n);
   }
};

// ────────────────────────────────────────────────────────────────────────────
class ObjectiveOp : public Operator
{
   PhaseMask &mask;
   PixelMesh &grid;
   FresnelOp &prop;
   Lens &lens;
   Vector &E;
   Vector &target;
   GaussNewtonOp &op;
   Fields &data;

public:
   ObjectiveOp(PhaseMask &mask, PixelMesh &grid,
               FresnelOp &prop, Lens &lens,
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
      last_scale = holo::MatchTarget(E_out, target, data.dot_v, data.energy_v);
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
   PhaseNewton() : NewtonSolver() {}

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
      out << "[hologram] Newton "
          << (gn_ran + 1) << "/" << max_outer
          << " (" << step->Iterations() << ")"
          << " " << loss << " " << drop << "\n";
      J_log_prev = J0;
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

      const char *why = "max_iter";
      if (newton.line_search_failed) { why = "line_search"; }
      else if (stall.stop_reason) { why = stall.stop_reason; }
      else if (newton.GetConverged()) { why = "abs_tol"; }
      out << "[hologram] Newton stop reason=" << why << "\n";
   }

   const RealArray2D &MaskImage() const { return mask_image; }
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

   if (!(wavelength_nm > 0.0) || !(z_prop > 0.0) || !(focal > 0.0) ||
       order < 0 || nwt_max_iter < 0 || cgs_max_iter < 1)
   {
      mfem::err << "Invalid wavelength, z, focal, order, or iteration count.\n";
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

   // Convert the target image to a 2D array of real values
   auto amplitude = holo::GrayToReal2D(gray);
   const int img_w = gray.width, img_h = gray.height;
   out << "Loaded " << img_w << "x" << img_h
       << " from " << in_file << "\n";

   // Compute mask's extent
   const real_t extent_x = 19.0e-3;
   const real_t extent_y =
      extent_x * (static_cast<real_t>(img_h) /
                  static_cast<real_t>(img_w));

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
   return EXIT_SUCCESS;
}
catch (const std::exception &exc)
{
   mfem::err << exc.what() << "\n";
   return EXIT_FAILURE;
}
