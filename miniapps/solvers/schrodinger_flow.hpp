// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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
//             ---------------------------------------------
//             Incompressible Schrödinger Flow (ISF) Miniapp
//             ---------------------------------------------
#pragma once

#include "mfem.hpp"
#include "general/forall.hpp"

using namespace mfem;

namespace mfem
{

/// @brief Options for the Incompressible Schrödinger Flow solver.
struct Options: public OptionsParser
{
   const char *device = "cpu";
   int order = 1;
   // Simulation setup
   real_t dt = 0.0;
   real_t hbar = 1e-1;
   int max_steps = 256;
   // Mesh setup
   int dim = 2;
   int nx = 64, ny = 64, nz = 64;
   real_t sx = 4.0, sy = 4.0, sz = 4.0;
   bool periodic = true, set_bc = false;
   // Leapfrog setup
   bool leapfrog = false;
   real_t leapfrog_vx = -0.1, leapfrog_sw = 1.0,
          leapfrog_r1 = 0.4, leapfrog_r2 = 0.26;
   // Jet setup
   bool jet = false;
   real_t jet_vx = 0.6;
   enum class JetGeom : int { Band = 0, Disc = 1, Rect = 2 };
   int jet_geom = 1;
   // Solvers setup
   real_t rtol = 1e-6, atol = 0.0, ftz = 1e-15;
   int max_iters = 1000, print_level = -1;
   // Visualization setup
   enum class VisData : int
   {
      Velocity,      // Velocity norm: jet, leapfrog
      Vorticity,     // Vorticity norm: leapfrog only
      X, Y, Z, Jet,  // (debug: Coordinates and Jet geometry)
      Unknown
   };
   bool visualization = true, paraview = false;
   int vis_steps = 1, vis_width = 1024, vis_height = 1024;
   int vis_data = static_cast<int>(VisData::Vorticity);

   Options(int argc, char *argv[]): OptionsParser(argc, argv)
   {
      AddOption(&device, "-d", "--device",
                "Device configuration string, see Device::Configure().");
      AddOption(&order, "-o", "--order", "Finite element order");
      AddOption(&dt, "-dt", "--dt", "Timestep size");
      AddOption(&hbar, "-hbar", "--hbar", "Planck constant");
      AddOption(&max_steps, "-ms", "--max-steps", "Maximum steps");
      AddOption(&dim, "-dim", "--dim", "Dimension of the problem (2 or 3)");
      AddOption(&nx, "-nx", "--nx", "Number of elements in x direction");
      AddOption(&ny, "-ny", "--ny", "Number of elements in y direction");
      AddOption(&nz, "-nz", "--nz", "Number of elements in z direction");
      AddOption(&sx, "-sx", "--sx", "Size of the domain in x direction");
      AddOption(&sy, "-sy", "--sy", "Size of the domain in y direction");
      AddOption(&sz, "-sz", "--sz", "Size of the domain in z direction");
      AddOption(&periodic, "-per", "--periodic", "-no-per",
                "--no-periodic", "Use a periodic mesh.");
      AddOption(&set_bc, "-bc", "--impose-bc", "-no-bc", "--dont-impose-bc",
                "Impose or not essential boundary conditions.");
      AddOption(&leapfrog, "-lf", "--leapfrog", "-no-lf", "--no-leapfrog",
                "Enable or disable leapfrog.");
      AddOption(&leapfrog_vx, "-lvx", "--leapfrog-vx", "Leapfrog X velocity");
      AddOption(&leapfrog_r1, "-lr1", "--leapfrog-r1", "Leapfrog ring 1 radius");
      AddOption(&leapfrog_r2, "-lr2", "--leapfrog-r2", "Leapfrog ring 2 radius");
      AddOption(&leapfrog_sw, "-lsw", "--leapfrog-sw", "Leapfrog swirling strength");
      AddOption(&jet, "-jet", "--jet", "-no-jet", "--no-jet",
                "Enable or disable jet.");
      AddOption(&jet_vx, "-jvx", "--jet-vx", "Jet X velocity");
      AddOption(&jet_geom, "-jg", "--jet-geom", "0: strip, 1: disc, 2: rect");
      AddOption(&rtol, "-rtol", "--rtol", "Solvers relative tolerance");
      AddOption(&atol, "-atol", "--atol", "Solvers absolute tolerance");
      AddOption(&ftz, "-ftz", "--ftz", "Flush to zero threshold");
      AddOption(&max_iters, "-mi", "--max-iterations", "Solvers max iterations");
      AddOption(&print_level, "-pl", "--print-level", "Solvers print level");
      AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                "--no-visualization", "Enable or not GLVis visualization");
      AddOption(&paraview, "-pv", "--paraview", "-no-pv",
                "--no-paraview", "Enable or not Paraview visualization");
      AddOption(&vis_steps, "-vs", "--vis-steps", "Visualization steps");
      AddOption(&vis_width, "-vw", "--vis-width", "vis width");
      AddOption(&vis_height, "-vh", "--vis-height", "vis height");
      AddOption(&vis_data, "-vd", "--vis-data",
                "Velocity: 0: Vorticity: 1 (leapfrog only)");
      ParseCheck();
      MFEM_VERIFY(jet ^ leapfrog, "'jet' or 'leapfrog' option must be set");
      MFEM_VERIFY(vis_data < static_cast<int>(VisData::Unknown),
                  "Invalid visualization data option.");
      if (dt == 0.0)
      {
         const auto dx = sx / static_cast<real_t>(order * nx);
         dt = (dx*dx) / hbar;
      }
   }
};

} // namespace mfem

/// @brief Complex number type for device.
#if !(defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
#include <cmath>
#include <complex>
#include <utility>
using complex_t = std::complex<real_t>;
#else // CUDA or HIP

#ifdef MFEM_USE_CUDA
#include <cuComplex.h>
#ifdef MFEM_USE_SINGLE
using RealComplex_t = cuFloatComplex;
#else
using RealComplex_t = cuDoubleComplex;
#endif // MFEM_USE_SINGLE
#endif // MFEM_USE_CUDA

#ifdef MFEM_USE_HIP
#include <hip/hip_complex.h>
#ifdef MFEM_USE_SINGLE
using RealComplex_t = hipFloatComplex;
#else
using RealComplex_t = hipDoubleComplex;
#endif // MFEM_USE_SINGLE
#endif // MFEM_USE_HIP

struct Complex : public RealComplex_t
{
   MFEM_HOST_DEVICE Complex() = default;
   MFEM_HOST_DEVICE Complex(real_t r) { x = r, y = 0.0; }
   MFEM_HOST_DEVICE Complex(real_t r, real_t i) { x = r, y = i; }
   MFEM_HOST_DEVICE real_t real() const { return x; }
   MFEM_HOST_DEVICE void real(real_t r) { x = r; }
   MFEM_HOST_DEVICE real_t imag() const { return y; }
   MFEM_HOST_DEVICE void imag(real_t i) { y = i; }

   template <typename U>
   MFEM_HOST_DEVICE inline Complex &operator*=(const U &z)
   {
      return (*this = *this * z, *this);
   }

   template <typename U>
   MFEM_HOST_DEVICE inline Complex &operator/=(const U &z)
   {
      return (*this = *this / z, *this);
   }
};

MFEM_HOST_DEVICE inline Complex operator*(const Complex &x, const real_t &y)
{
   return Complex(x.real() * y, x.imag() * y);
}

MFEM_HOST_DEVICE inline Complex operator+(const Complex &a, const Complex &b)
{
   return Complex(a.real() + b.real(), a.imag() + b.imag());
}

MFEM_HOST_DEVICE inline Complex operator*(const real_t d, const Complex &z)
{
   return Complex(z.real() * d, z.imag() * d);
}

MFEM_HOST_DEVICE inline Complex operator*(const Complex &a, const Complex &b)
{
   return Complex(a.real() * b.real() - a.imag() * b.imag(),
                  a.real() * b.imag() + a.imag() * b.real());
}

MFEM_HOST_DEVICE inline Complex operator/(const Complex &z, const real_t &d)
{
   return Complex(z.real() / d, z.imag() / d);
}

MFEM_HOST_DEVICE inline real_t abs(const Complex &z)
{
   return std::hypot(z.real(), z.imag());
}

MFEM_HOST_DEVICE inline Complex exp(const Complex &q)
{
   real_t s, c, e = std::exp(q.real());
#ifdef MFEM_USE_SINGLE
   sincosf(q.imag(), &s, &c);
#else
   sincos(q.imag(), &s, &c);
#endif
   return Complex(c * e, s * e);
}

MFEM_HOST_DEVICE inline real_t norm(const Complex &z)
{
   return z.real() * z.real() + z.imag() * z.imag();
}

using complex_t = Complex;
#endif // MFEM_USE_CUDA || MFEM_USE_HIP

namespace mfem
{

using real3_t = std::array<real_t, 3>;

/// @brief Base class for Schrodinger solver kernels.
template <typename TMesh,
          typename TFiniteElementSpace,
          typename TComplexGridFunction,
          typename TGridFunction,
          typename TBilinearForm,
          typename TMixedBilinearForm,
          typename TLinearForm>
struct SchrodingerBaseKernels: public Options
{
   std::function<Mesh()> CreateMesh2D, CreateMesh3D;
   Mesh serial_mesh;
   TMesh mesh;
   H1_FECollection h1_fec;
   ND_FECollection nd_fec;
   TFiniteElementSpace h1_fes, nd_fes, nodal_fes;
   TGridFunction nodes;
   const int ne, ndofs;
   ConstantCoefficient one;
   VectorFunctionCoefficient Vx, Vy, Vz;
   TBilinearForm mass_h1, mass_nd, diff_h1;
   TMixedBilinearForm grad_nd, nd_dot_x_h1, nd_dot_y_h1, nd_dot_z_h1;
   Array<int> ess_tdof_list;
   std::function<void()> SetEssentialTrueDofs;
   bool ess_tdof_list_setup, diff_h1_setup;
   OperatorJacobiSmoother Km1_h1_smoother;
   OrthoSolver Km1_ortho;
   CGSolver Mm1_h1, Mm1_nd, Km1_h1;
   TComplexGridFunction psi1, psi2;
   TComplexGridFunction delta_psi1, delta_psi2, gpsi1_nd, gpsi2_nd;
   TComplexGridFunction gpsi1_x, gpsi2_x, gpsi1_y, gpsi2_y, gpsi1_z, gpsi2_z;
   TGridFunction div_u, q, h1, nd;
   TLinearForm rhs;
   GridFunctionCoefficient div_u_coeff;
   OperatorHandle mass_h1_op, mass_nd_op, diff_h1_op;
   OperatorHandle grad_nd_op, nd_dot_x_h1_op, nd_dot_y_h1_op, nd_dot_z_h1_op;

   SchrodingerBaseKernels(Options &config,
                          std::function<TMesh(Mesh&)> CreateMesh,
                          std::function<OrthoSolver()> CreateOrthoSolver,
                          std::function<CGSolver()> CreateCGSolver):
      Options(config),
      CreateMesh2D([&]()
   {
      const auto type = Element::QUADRILATERAL;
      Mesh xy = Mesh::MakeCartesian2D(nx, ny, type, false, sx, sy, false);
      xy.SetCurvature(order);
      if (!periodic) { return xy; }
      std::vector<Vector> Tr2 = { Vector({ sx, 0.0_r }),
                                  Vector({ 0.0_r, sy })
                                };
      return Mesh::MakePeriodic(xy, xy.CreatePeriodicVertexMapping(Tr2));
   }),
   CreateMesh3D([&]()
   {
      const auto type = Element::HEXAHEDRON;
      Mesh xyz = Mesh::MakeCartesian3D(nx, ny, nz, type, sx, sy, sz, false);
      xyz.SetCurvature(order);
      if (!periodic) { return xyz; }
      std::vector<Vector> Tr3 = { Vector({ sx, 0.0_r, 0.0_r }),
                                  Vector({ 0.0_r, sy, 0.0_r }),
                                  Vector({ 0.0_r, 0.0_r, sz })
                                };
      return Mesh::MakePeriodic(xyz, xyz.CreatePeriodicVertexMapping(Tr3));
   }),
   serial_mesh(dim == 3 ? CreateMesh3D() : CreateMesh2D()),
   mesh(CreateMesh(serial_mesh)),
   h1_fec(order, dim),
   nd_fec(order, dim),
   h1_fes(&mesh, &h1_fec),
   nd_fes(&mesh, &nd_fec),
   nodal_fes(&mesh, &h1_fec, dim, Ordering::byVDIM),
   nodes(&nodal_fes),
   ne(mesh.GetNE()),
   ndofs(h1_fes.GetNDofs()),
   one(1.0),
   Vx(dim, [&](const Vector &, Vector &v)
   {
      v.SetSize(dim), v[0] = 1.0, v[1] = 0.0;
      if (dim == 3) { v[2] = 0.0; }
   }),
   Vy(dim, [&](const Vector &, Vector &v)
   {
      v.SetSize(dim), v[0] = 0.0, v[1] = 1.0;
      if (dim == 3) { v[2] = 0.0; }
   }),
   Vz(dim, [&](const Vector &, Vector &v)
   {
      v.SetSize(dim), v[0] = 0.0, v[1] = 0.0;
      if (dim == 3) { v[2] = 1.0; }
   }),
   mass_h1(&h1_fes),
   mass_nd(&nd_fes),
   diff_h1(&h1_fes),
   grad_nd(&h1_fes, &nd_fes),
   nd_dot_x_h1(&nd_fes, &h1_fes),
   nd_dot_y_h1(&nd_fes, &h1_fes),
   nd_dot_z_h1(&nd_fes, &h1_fes),
   ess_tdof_list(),
   SetEssentialTrueDofs([&]()
   {
      if (periodic || mesh.bdr_attributes.Size() == 0) { return; }
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = set_bc ? 1 : 0;
      h1_fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }),
   ess_tdof_list_setup((SetEssentialTrueDofs(), true)),
   diff_h1_setup((diff_h1.AddDomainIntegrator(new DiffusionIntegrator(one)),
                  diff_h1.SetAssemblyLevel(AssemblyLevel::PARTIAL),
                  diff_h1.Assemble(),
                  true)),
   Km1_h1_smoother(diff_h1, ess_tdof_list),
   Km1_ortho(CreateOrthoSolver()),
   Mm1_h1(CreateCGSolver()),
   Mm1_nd(CreateCGSolver()),
   Km1_h1(CreateCGSolver()),
   psi1(&h1_fes),
   psi2(&h1_fes),
   delta_psi1(&h1_fes), delta_psi2(&h1_fes),
   gpsi1_nd(&nd_fes), gpsi2_nd(&nd_fes),
   gpsi1_x(&h1_fes), gpsi2_x(&h1_fes),
   gpsi1_y(&h1_fes), gpsi2_y(&h1_fes),
   gpsi1_z(&h1_fes), gpsi2_z(&h1_fes),
   div_u(&h1_fes),
   q(&h1_fes),
   h1(&h1_fes),
   nd(&nd_fes),
   rhs(&h1_fes),
   div_u_coeff(&div_u)
   {
      mesh.GetNodes(nodes);

      mass_h1.AddDomainIntegrator(new MassIntegrator(one));
      mass_nd.AddDomainIntegrator(new VectorFEMassIntegrator(one));
      grad_nd.AddDomainIntegrator(new MixedVectorGradientIntegrator());
      nd_dot_x_h1.AddDomainIntegrator(new MixedDotProductIntegrator(Vx));
      nd_dot_y_h1.AddDomainIntegrator(new MixedDotProductIntegrator(Vy));
      nd_dot_z_h1.AddDomainIntegrator(new MixedDotProductIntegrator(Vz));

      mass_h1.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      mass_nd.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      grad_nd.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      nd_dot_x_h1.SetAssemblyLevel(AssemblyLevel::LEGACY);
      nd_dot_y_h1.SetAssemblyLevel(AssemblyLevel::LEGACY);
      nd_dot_z_h1.SetAssemblyLevel(AssemblyLevel::LEGACY);

      mass_h1.Assemble();
      mass_nd.Assemble();
      grad_nd.Assemble();
      nd_dot_x_h1.Assemble(), nd_dot_x_h1.Finalize();
      nd_dot_y_h1.Assemble(), nd_dot_y_h1.Finalize();
      if (dim == 3) { nd_dot_z_h1.Assemble(), nd_dot_z_h1.Finalize(); }

      mass_h1.FormSystemMatrix(ess_tdof_list, mass_h1_op);
      mass_nd.FormSystemMatrix(ess_tdof_list, mass_nd_op);
      diff_h1.FormSystemMatrix(ess_tdof_list, diff_h1_op);

      // Only used for velocity computation
      if (visualization && static_cast<VisData>(vis_data) == VisData::Velocity)
      {
         grad_nd.FormRectangularSystemMatrix(ess_tdof_list, ess_tdof_list, grad_nd_op);
         nd_dot_x_h1.FormRectangularSystemMatrix(ess_tdof_list, ess_tdof_list,
                                                 nd_dot_x_h1_op);
         nd_dot_y_h1.FormRectangularSystemMatrix(ess_tdof_list, ess_tdof_list,
                                                 nd_dot_y_h1_op);
         if (dim == 3)
         {
            nd_dot_z_h1.FormRectangularSystemMatrix(ess_tdof_list, ess_tdof_list,
                                                    nd_dot_z_h1_op);
         }
      }

      Mm1_h1.SetRelTol(rtol), Mm1_h1.SetAbsTol(atol), Mm1_h1.SetMaxIter(max_iters);
      Mm1_h1.SetOperator(*mass_h1_op), Mm1_h1.iterative_mode = false;
      Mm1_h1.SetPrintLevel(print_level);

      Mm1_nd.SetRelTol(rtol), Mm1_nd.SetAbsTol(atol), Mm1_nd.SetMaxIter(max_iters);
      Mm1_nd.SetOperator(*mass_nd_op), Mm1_nd.iterative_mode = false;
      Mm1_nd.SetPrintLevel(print_level);

      Km1_h1.SetRelTol(rtol), Km1_h1.SetAbsTol(atol), Km1_h1.SetMaxIter(max_iters);
      Km1_h1.SetOperator(*diff_h1_op), Km1_h1.iterative_mode = false;
      Km1_ortho.SetSolver(Km1_h1_smoother);
      Km1_h1.SetPreconditioner(Km1_ortho);
      Km1_h1.SetPrintLevel(3);

      rhs.AddDomainIntegrator(new DomainLFIntegrator(div_u_coeff));
      rhs.UseFastAssembly(true);
   }

   /// @brief Initialize the wavefunctions psi1 and psi2.
   void Initialize(Vector &phase_r)
   {
      psi1 = 0.0, psi2 = 0.0;
      if (leapfrog && phase_r.Size() > 0)
      {
         const auto phase = phase_r.Read();
         auto psi1_r = Reshape(psi1.real().ReadWrite(), ndofs);
         auto psi1_i = Reshape(psi1.imag().ReadWrite(), ndofs);
         auto psi2_r = Reshape(psi2.real().ReadWrite(), ndofs);
         auto psi2_i = Reshape(psi2.imag().ReadWrite(), ndofs);
         mfem::forall(ndofs, [=] MFEM_HOST_DEVICE(int n)
         {
            const complex_t eps = 0.01, i_phase(0, phase[n]);
            const complex_t z1 = exp(i_phase);
            const complex_t z2 = eps * exp(i_phase);
            psi1_r(n) = z1.real(), psi1_i(n) = z1.imag();
            psi2_r(n) = z2.real(), psi2_i(n) = z2.imag();
         });
      }
      if (jet) { psi1.real() = 1.0, psi2.real() = 0.0; }
   }

   /// @brief Normalize the wavefunctions psi1 and psi2.
   void Normalize()
   {
      auto psi1_r = Reshape(psi1.real().ReadWrite(), ndofs);
      auto psi1_i = Reshape(psi1.imag().ReadWrite(), ndofs);
      auto psi2_r = Reshape(psi2.real().ReadWrite(), ndofs);
      auto psi2_i = Reshape(psi2.imag().ReadWrite(), ndofs);
      mfem::forall(ndofs, [=] MFEM_HOST_DEVICE(int n)
      {
         complex_t psi1(psi1_r(n), psi1_i(n)), psi2(psi2_r(n), psi2_i(n));
         const real_t psi_norm = std::sqrt(norm(psi1) + norm(psi2));
         if (fabs(psi_norm) < 1e-16) { return; }
         psi1_r(n) /= psi_norm, psi1_i(n) /= psi_norm;
         psi2_r(n) /= psi_norm, psi2_i(n) /= psi_norm;
      });
   }

   /// @brief Restrict the wavefunctions psi1 and psi2.
   void Restrict(const real_t t, const TGridFunction &isJet_in,
                 const real_t omega, const TGridFunction &phase_in)
   {
      MFEM_VERIFY(jet, "Jet must be enabled use restrict.");
      const auto isJet = isJet_in.Read();
      const auto phase = phase_in.Read();
      auto psi1_r = Reshape(psi1.real().ReadWrite(), ndofs);
      auto psi1_i = Reshape(psi1.imag().ReadWrite(), ndofs);
      auto psi2_r = Reshape(psi2.real().ReadWrite(), ndofs);
      auto psi2_i = Reshape(psi2.imag().ReadWrite(), ndofs);
      mfem::forall(ndofs, [=] MFEM_HOST_DEVICE(int n)
      {
         if (isJet[n] == 0) { return; }
         const complex_t i_pn_omega_t(0.0, phase[n] - omega * t);
         complex_t psi1(psi1_r(n), psi1_i(n)), psi2(psi2_r(n), psi2_i(n));
         const real_t amp1 = abs(psi1), amp2 = abs(psi2);
         psi1 = amp1 * exp(i_pn_omega_t);
         psi2 = amp2 * exp(i_pn_omega_t);
         psi1_r(n) = psi1.real(), psi1_i(n) = psi1.imag();
         psi2_r(n) = psi2.real(), psi2_i(n) = psi2.imag();
      });
   }

   template<typename Gfn, typename Xfn, typename Yfn, typename Zfn>
   void GradPsi(Gfn &Grad_nd, Xfn &x_dot_Mm1, Yfn &y_dot_Mm1, Zfn &z_dot_Mm1)
   {
      Grad_nd(psi1.real(), gpsi1_nd.real());
      Grad_nd(psi1.imag(), gpsi1_nd.imag());
      Grad_nd(psi2.real(), gpsi2_nd.real());
      Grad_nd(psi2.imag(), gpsi2_nd.imag());

      x_dot_Mm1(gpsi1_nd.real(), gpsi1_x.real());
      x_dot_Mm1(gpsi1_nd.imag(), gpsi1_x.imag());
      x_dot_Mm1(gpsi2_nd.real(), gpsi2_x.real());
      x_dot_Mm1(gpsi2_nd.imag(), gpsi2_x.imag());

      y_dot_Mm1(gpsi1_nd.real(), gpsi1_y.real());
      y_dot_Mm1(gpsi1_nd.imag(), gpsi1_y.imag());
      y_dot_Mm1(gpsi2_nd.real(), gpsi2_y.real());
      y_dot_Mm1(gpsi2_nd.imag(), gpsi2_y.imag());

      if (dim == 3)
      {
         z_dot_Mm1(gpsi1_nd.real(), gpsi1_z.real());
         z_dot_Mm1(gpsi1_nd.imag(), gpsi1_z.imag());
         z_dot_Mm1(gpsi2_nd.real(), gpsi2_z.real());
         z_dot_Mm1(gpsi2_nd.imag(), gpsi2_z.imag());
      }
   }

   // u = ℏRe{−𝑖𝝭ᵀ·∇𝝭} = ħ[𝝭1r.∇𝝭1i - 𝝭1i.∇𝝭1r + 𝝭2r.∇𝝭2i - 𝝭2i.∇𝝭2r]
   void GradPsiVelocity(const real_t hbar, TGridFunction &ux,
                        TGridFunction &uy, TGridFunction &uz)
   {
      const auto psi1r = Reshape(psi1.real().Read(), ndofs);
      const auto psi1i = Reshape(psi1.imag().Read(), ndofs);
      const auto psi2r = Reshape(psi2.real().Read(), ndofs);
      const auto psi2i = Reshape(psi2.imag().Read(), ndofs);

      const auto Gpsi1rx = Reshape(gpsi1_x.real().Read(), ndofs);
      const auto Gpsi1ix = Reshape(gpsi1_x.imag().Read(), ndofs);
      const auto Gpsi1ry = Reshape(gpsi1_y.real().Read(), ndofs);
      const auto Gpsi1iy = Reshape(gpsi1_y.imag().Read(), ndofs);
      const auto Gpsi1rz = Reshape(gpsi1_z.real().Read(), ndofs);
      const auto Gpsi1iz = Reshape(gpsi1_z.imag().Read(), ndofs);

      const auto Gpsi2rx = Reshape(gpsi2_x.real().Read(), ndofs);
      const auto Gpsi2ix = Reshape(gpsi2_x.imag().Read(), ndofs);
      const auto Gpsi2ry = Reshape(gpsi2_y.real().Read(), ndofs);
      const auto Gpsi2iy = Reshape(gpsi2_y.imag().Read(), ndofs);
      const auto Gpsi2rz = Reshape(gpsi2_z.real().Read(), ndofs);
      const auto Gpsi2iz = Reshape(gpsi2_z.imag().Read(), ndofs);

      auto vx = Reshape(ux.Write(), ndofs);
      auto vy = Reshape(uy.Write(), ndofs);
      auto vz = Reshape(uz.Write(), ndofs);
      const real_t FTZ = ftz;
      const int DIM = dim;

      mfem::forall(ndofs, [=] MFEM_HOST_DEVICE(int n)
      {
         vx(n) = +psi1r(n) * Gpsi1ix(n) - psi1i(n) * Gpsi1rx(n);
         vx(n) += psi2r(n) * Gpsi2ix(n) - psi2i(n) * Gpsi2rx(n);
         vx(n) *= (fabs(vx(n)) < FTZ) ? 0.0 : hbar;
         vy(n) = +psi1r(n) * Gpsi1iy(n) - psi1i(n) * Gpsi1ry(n);
         vy(n) += psi2r(n) * Gpsi2iy(n) - psi2i(n) * Gpsi2ry(n);
         vy(n) *= (fabs(vy(n)) < FTZ) ? 0.0 : hbar;
         if (DIM == 2) { return; }
         vz(n) = +psi1r(n) * Gpsi1iz(n) - psi1i(n) * Gpsi1rz(n);
         vz(n) += psi2r(n) * Gpsi2iz(n) - psi2i(n) * Gpsi2rz(n);
         vz(n) *= (fabs(vz(n)) < FTZ) ? 0.0 : hbar;
      });
   }

   // ∇∙u = -ℏ.Re{𝝭ᵀ·𝑖∆𝝭} = -ℏ[𝝭1i.∆𝝭1r - 𝝭1r.∆𝝭1i + 𝝭2i.∆𝝭2r - 𝝭2r.∆𝝭2i]
   void ComputeDivU()
   {
      const auto psi1r = Reshape(psi1.real().Read(), ndofs);
      const auto psi1i = Reshape(psi1.imag().Read(), ndofs);
      const auto psi2r = Reshape(psi2.real().Read(), ndofs);
      const auto psi2i = Reshape(psi2.imag().Read(), ndofs);
      const auto Dpsi1r = Reshape(delta_psi1.real().Read(), ndofs);
      const auto Dpsi1i = Reshape(delta_psi1.imag().Read(), ndofs);
      const auto Dpsi2r = Reshape(delta_psi2.real().Read(), ndofs);
      const auto Dpsi2i = Reshape(delta_psi2.imag().Read(), ndofs);
      auto div_u_w = Reshape(div_u.Write(), ndofs);
      mfem::forall(ndofs, [=] MFEM_HOST_DEVICE(int n)
      {
         div_u_w(n) = +psi1i(n) * Dpsi1r(n) - psi1r(n) * Dpsi1i(n);
         div_u_w(n) += psi2i(n) * Dpsi2r(n) - psi2r(n) * Dpsi2i(n);
         div_u_w(n) *= -1.0;
      });
   }

   // 𝝭ⁿ⁺¹ = exp(−i.q/ħ).𝝭ⁿ
   void GaugeTransform()
   {
      const auto q_r = Reshape(q.Read(), ndofs);
      auto psi1_r = Reshape(psi1.real().ReadWrite(), ndofs);
      auto psi1_i = Reshape(psi1.imag().ReadWrite(), ndofs);
      auto psi2_r = Reshape(psi2.real().ReadWrite(), ndofs);
      auto psi2_i = Reshape(psi2.imag().ReadWrite(), ndofs);
      mfem::forall(ndofs, [=] MFEM_HOST_DEVICE(int n)
      {
         const complex_t minus_i(0, -1.0);
         const complex_t eiq = exp(minus_i * q_r(n));
         complex_t psi1(psi1_r(n), psi1_i(n)), psi2(psi2_r(n), psi2_i(n));
         psi1 *= eiq, psi2 *= eiq;
         psi1_r(n) = psi1.real(), psi1_i(n) = psi1.imag();
         psi2_r(n) = psi2.real(), psi2_i(n) = psi2.imag();
      });
   }

   /// @brief Add a circular vortex to the wavefunctions psi1 and psi2.
   void AddCircularVortex(const real3_t center, const real3_t normal,
                          const real_t radius, const real_t swirling)
   {
      MFEM_VERIFY(swirling > 0.0, "Swirling strength must be positive");
      const auto DIM = dim;
      const real3_t o = { center[0], center[1], center[2] };
      const real_t norm2 = std::sqrt(normal[0] * normal[0] +
                                     normal[1] * normal[1] +
                                     normal[2] * normal[2]);
      const real_t n0 = normal[0] / norm2, n1 = normal[1] / norm2,
                   n2 = normal[2] / norm2;
      const auto &X = Reshape(nodes.Read(), dim, ndofs);
      auto psi1_r = Reshape(psi1.real().ReadWrite(), ndofs);
      auto psi1_i = Reshape(psi1.imag().ReadWrite(), ndofs);
      mfem::forall(ndofs, [=] MFEM_HOST_DEVICE(int n)
      {
         const real_t px = X(0, n), py = X(1, n),
                      pz = DIM == 3 ? X(2, n) : 0.0;
         const real_t rx = px - o[0], ry = py - o[1], rz = pz - o[2];
         const real_t z = rx * n0 + ry * n1 + rz * n2;
         const bool inRange = (rx * rx + ry * ry + rz * rz - z * z) < radius * radius;
         const bool inLayerP = inRange && (z > 0.0  && z <= (+swirling / 2.0));
         const bool inLayerM = inRange && (z <= 0.0 && z >= (-swirling / 2.0));
         real_t alpha = 0.0;
         if (inLayerP) { alpha = -M_PI * (2.0 * z / swirling - 1.0); }
         if (inLayerM) { alpha = -M_PI * (2.0 * z / swirling + 1.0); }
         complex_t psi1(psi1_r(n), psi1_i(n));
         const complex_t alpha_i(0, alpha);
         psi1 *= exp(alpha_i);
         psi1_r(n) = psi1.real(), psi1_i(n) = psi1.imag();
      });
   }
};

/// @brief Crank-Nicolson time solver for the Schrodinger equation.
template<typename TFiniteElementSpace,
         typename TSesquilinearForm,
         typename TComplexGridFunction>
struct CrankNicolsonBaseSolver
{
   ConstantCoefficient one, dthq, mdthq;
   TSesquilinearForm C_form, R_form;
   Array<int> no_bc;
   OperatorHandle C_op, R_op;
   TComplexGridFunction z;
   GMRESSolver Cm1;

   // ∂ₜ𝝭 = ½ℏ𝑖∆𝝭
   // 𝝭ⁿ⁺¹ - 𝝭ⁿ = ¼𝛅t𝑖ℏ(∆𝝭ⁿ⁺¹ + ∆𝝭)
   // 𝝭ⁿ⁺¹ - ¼𝑖ℏ𝛅t∆𝝭ⁿ⁺¹ = 𝝭ⁿ + ¼𝑖ℏ𝛅t∆𝝭ⁿ
   // [M + ¼𝑖ℏ𝛅tA]𝝭ⁿ⁺¹ = [M - ¼𝑖ℏ𝛅tA]𝝭ⁿ
   // C = M + ¼𝑖ℏ𝛅tA, R = M - ¼𝑖ℏ𝛅tA
   CrankNicolsonBaseSolver(TFiniteElementSpace &fes,
                           real_t hbar, real_t dt,
                           std::function<GMRESSolver()> CreateGMRESSolver,
                           real_t rtol, real_t atol, int maxiter,
                           int print_level):
      one(1.0), dthq(dt * hbar / 4.0), mdthq(-dt * hbar / 4.0),
      C_form(&fes), R_form(&fes),
      z(&fes),
      Cm1(CreateGMRESSolver())
   {
      // C = M + ¼𝑖ℏ𝛅tA
      C_form.AddDomainIntegrator(new MassIntegrator(one),
                                 new DiffusionIntegrator(dthq));
      C_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      C_form.Assemble();
      C_form.FormSystemMatrix(no_bc, C_op);

      // R = M - ¼𝑖ℏ𝛅tA
      R_form.AddDomainIntegrator(new MassIntegrator(one),
                                 new DiffusionIntegrator(mdthq));
      R_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      R_form.Assemble();
      R_form.FormSystemMatrix(no_bc, R_op);

      Cm1.SetPrintLevel(print_level);
      Cm1.iterative_mode = false;
      Cm1.SetMaxIter(maxiter);
      Cm1.SetOperator(*C_op);
      Cm1.SetRelTol(rtol);
      Cm1.SetAbsTol(atol);
   }

   // 𝝭ⁿ⁺¹ = C⁻¹ R 𝝭ⁿ
   virtual void Mult(TComplexGridFunction &psi) = 0;
};

/// @brief Class for simulating incompressible Schrodinger flow.
template<typename TSchrodingerSolver, typename TGridFunction>
class IncompressibleBaseFlow : private Options
{
public:
   TSchrodingerSolver &solver;
   const int ndofs;
   real_t omega;
   TGridFunction isJet, phase, vx, vy, vz;

   IncompressibleBaseFlow(Options &config, TSchrodingerSolver &solver):
      Options(config),
      solver(solver),
      ndofs(solver.ndofs),
      omega(0.0),
      isJet(&solver.h1_fes),
      phase(&solver.h1_fes),
      vx(&solver.h1_fes), vy(&solver.h1_fes), vz(&solver.h1_fes)
   {
      isJet = 0.0, phase = 0.0;
      vx = 0.0, vy = 0.0, vz = 0.0;
      Setup();
   }

   /**
    * @brief Setup the solver.
    *
    * This function initializes the solver by setting up the phase,
    * the Jet vectors and normalizing the wave functions.
    * It also adds circular vortex rings if leapfrog is enabled.
    */
   void Setup()
   {
      real_t velocity[3];
      velocity[0] = leapfrog ? leapfrog_vx : jet ? jet_vx : 0.0;
      velocity[1] = velocity[2] = 0.0;

      const real_t kvec0 = velocity[0] / hbar;
      const real_t kvec1 = velocity[1] / hbar;
      const real_t kvec2 = velocity[2] / hbar;

      omega = 0.0;
      for (int i = 0; i < 3; i++) { omega += velocity[i] * velocity[i]; }
      omega /= 2.0 * hbar;

      auto phase_w = phase.Write();
      auto isJet_w = isJet.Write();
      const int DIM = dim, JET_GEOM = jet_geom;
      const real_t SX = sx, SY = sy, SZ = sz;
      const auto X = Reshape(solver.nodes.Read(), dim, ndofs);
      mfem::forall(ndofs, [=] MFEM_HOST_DEVICE(int n)
      {
         const real_t r = SX / 16.0;
         const real_t px = X(0, n), py = X(1, n),
                      pz = DIM == 3 ? X(2, n) : 0.0;
         const real_t dx = px - (SX/8.0), dy = py - (SY/2.0),
                      dz = DIM == 3 ? pz - (SZ/2.0) : 0.0;
         const auto geom = static_cast<Options::JetGeom>(JET_GEOM);
         if (geom == JetGeom::Band) { isJet_w[n] = (fabs(dy*dy) < (r*r)) ? 1 : 0; }
         if (geom == JetGeom::Disc) { isJet_w[n] = (fabs(dx*dx + dy*dy + dz*dz) < (r*r)) ? 1 : 0; }
         if (geom == JetGeom::Rect) { isJet_w[n] = (px > 0.2 && dx < 1.0 && fabs(dy*dy + dz*dz) < (r*r)) ? 1 : 0; }
         phase_w[n] = kvec0 * px + kvec1 * py + kvec2 * pz;
      });

      solver.Initialize(phase);

      if (jet) { ConstrainJetVelocity(); }

      if (leapfrog) // Add vortex rings
      {
         const real_t zh = sx / 2.0, yh = sy / 2.0;
         const real_t z2 = zh * zh, y2 = yh * yh, r = sqrt(z2 + y2);
         const real3_t n = { -1.0, 0.0, 0.0 },
                       o = { sx / 2.0_r, sy / 2.0_r, DIM == 3 ? sz / 2.0_r : 0.0_r };
         solver.AddCircularVortex(o, n, r * leapfrog_r1, leapfrog_sw);
         solver.AddCircularVortex(o, n, r * leapfrog_r2, leapfrog_sw);
         solver.Normalize(), solver.PressureProject();
      }
   }

   /// @brief This function performs a single time step of the solver.
   void Step(const real_t &t)
   {
      solver.Step();
      solver.Normalize(), solver.PressureProject();
      if (jet)
      {
         solver.Restrict(t, isJet, omega, phase);
         solver.Normalize(), solver.PressureProject();
      }
      if (visualization &&
          static_cast<VisData>(vis_data) == VisData::Velocity)
      {
         solver.VelocityOneForm(vx, vy, vz);
      }
   }

   /// @brief Restricts the velocity of the wavefunctions.
   void ConstrainJetVelocity()
   {
      solver.Normalize();
      MFEM_VERIFY(jet, "ConstrainJetVelocity() only for jet geometry");
      for (int i = 0; i < 10; i++)
      {
         solver.Restrict(0.0, isJet, omega, phase);
         solver.PressureProject();
      }
   }
};

/// @brief Base class for visualization.
template <typename TMesh,
          typename TGridFunction,
          typename TFiniteElementSpace,
          typename TSchrodingerSolver,
          typename TIncompressibleFlow>
struct VisualizerBase : private Options
{
   TMesh &mesh;
   socketstream glvis;
   TGridFunction vis_gf;
   std::function<void()> vis_fn;
   std::function<std::string()> vis_prefix;
   const int ndofs;
   const Options::VisData vis_data;
   const TIncompressibleFlow &isf;

#ifndef MFEM_USE_HDF5
   ParaViewDataCollection dc;
#else
   ParaViewHDFDataCollection dc;
#endif

   VisualizerBase(Options &config,
                  TSchrodingerSolver &solver,
                  const TIncompressibleFlow &isf,
                  std::function<std::string()> prefix):
      Options(config),
      mesh(solver.mesh),
      vis_gf(&solver.h1_fes),
      vis_prefix(std::move(prefix)),
      ndofs(solver.h1_fes.GetNDofs()),
      vis_data(static_cast<Options::VisData>(config.vis_data)),
      isf(isf),
      dc("ISF", &mesh)
   {
      vis_gf = 0.0;

      // Coordinates visualization (debug)
      if (vis_data <= Options::VisData::Z)
      {
         vis_fn = [&]()
         {
            const auto X = Reshape(solver.nodes.Read(), dim, ndofs);
            auto viz_h1_w = vis_gf.Write();
            mfem::forall(ndofs, [=] MFEM_HOST_DEVICE(int i)
            {
               viz_h1_w[i] = vis_data == Options::VisData::X ? X(0,i):
                             vis_data == Options::VisData::Y ? X(1,i):
                             vis_data == Options::VisData::Z ? dim == 3 ? X(2,i) : 0.0:
                             0.0;
            });
            vis_gf.HostRead();
         };
      }

      // Jet geometry visualization (debug)
      if (vis_data == Options::VisData::Jet)
      {
         vis_fn = [&]()
         {
            auto isJet_r = isf.isJet.Read();
            auto viz_h1_w = vis_gf.Write();
            mfem::forall(ndofs, [=] MFEM_HOST_DEVICE(int i) { viz_h1_w[i] = isJet_r[i]; });
            vis_gf.HostRead();
         };
      }

      // Velocity norm visualization
      if (vis_data == Options::VisData::Velocity)
      {
         vis_fn = [&]()
         {
            const auto vx_r = isf.vx.Read(), vy_r = isf.vy.Read(), vz_r = isf.vz.Read();
            auto viz_h1_w = vis_gf.Write();
            mfem::forall(ndofs, [=] MFEM_HOST_DEVICE(int i)
            {
               const real_t vx = vx_r[i], vy = vy_r[i], vz = vz_r[i];
               viz_h1_w[i] = std::sqrt(vx*vx + vy*vy + vz*vz);
            });
            vis_gf.HostRead();
         };
      }

      // Vorticity norm visualization
      if (vis_data == Options::VisData::Vorticity)
      {
         vis_fn = [&]()
         {
            const auto psi1_r = solver.psi1.real().Read();
            const auto psi1_i = solver.psi1.imag().Read();
            const auto psi2_r = solver.psi2.real().Read();
            const auto psi2_i = solver.psi2.imag().Read();
            auto viz_h1_w = vis_gf.Write();
            mfem::forall(ndofs, [=] MFEM_HOST_DEVICE(int i)
            {
               const auto psi1 = psi1_r[i] * psi1_r[i] + psi1_i[i] * psi1_i[i];
               const auto psi2 = psi2_r[i] * psi2_r[i] + psi2_i[i] * psi2_i[i];
               viz_h1_w[i] = psi1 * psi1 + psi2 * psi2;
            });
            vis_gf.HostRead();
         };
      }

      if (visualization)
      {
         glvis.open("localhost", 19916);
         if (glvis.is_open())
         {
            glvis.precision(8);
            glvis << vis_prefix().c_str();
            glvis << "solution\n" << mesh << *(this->operator()())
                  << "window_geometry 0 0 " << vis_width << " " << vis_height << "\n"
                  << "keys cgjR\n" << std::flush;
         }
      }

      if (paraview)
      {
         dc.SetDataFormat(VTKFormat::BINARY32);
         dc.SetPrefixPath("ParaView");
         dc.SetHighOrderOutput(true);
         dc.SetLevelsOfDetail(order);
         dc.RegisterField("vis_gf", this->operator()());
         dc.SetCycle(0);
         dc.SetTime(0.0);
         dc.Save();
      }
   }

   ~VisualizerBase() { if (glvis.is_open()) { glvis.close(); } }

   /// @brief Get the visualization data.
   TGridFunction* operator()()
   {
      vis_fn();
      return &vis_gf;
   }

   /// @brief Save the visualization data to a file.
   void Save(int cycle, real_t time)
   {
      if (!paraview) { return; }
      this->operator()();
      dc.SetCycle(cycle);
      dc.SetTime(time);
      dc.Save();
   }

   /// @brief Send the visualization data to GLVis.
   void GLVis()
   {
      if (!glvis.is_open()) { return; }
      glvis << vis_prefix().c_str();
      glvis << "solution\n" << mesh << *(this->operator()()) << std::flush;
   }
};

} // namespace mfem