// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details

// Sanity check for AnisotropicElasticityIntegrator.
//
// For an isotropic material described by the Lame parameters lambda and mu,
// the element matrices produced by ElasticityIntegrator and by
// AnisotropicElasticityIntegrator -- fed the corresponding isotropic Voigt
// stiffness tensor -- must agree to within floating point round-off, since
// they represent the same bilinear form. This miniapp builds that isotropic
// Voigt tensor independently (not by reusing anything internal to either
// integrator) and compares:
//   - the element matrices, over a range of dimensions, element geometries,
//     orders, and (constant and spatially varying) materials;
//   - the recovered stress (ComputeElementFlux), accounting for the
//     different Voigt shear orderings used by the two integrators;
//   - the flux/strain energy (ComputeFluxEnergy), which is ordering
//     independent.
//
// The miniapp exits with a nonzero status if any relative error exceeds the
// requested tolerance.

#include "anisotropic_elasticity_integrator.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>

using namespace std;
using namespace mfem;

namespace
{

/// Builds the isotropic Voigt stiffness tensor corresponding to the
/// constitutive law used internally by ElasticityIntegrator,
///    C_ijkl = lambda * delta_ij * delta_kl
///             + mu * (delta_ik * delta_jl + delta_il * delta_jk),
/// using the ordering xx,yy,zz,yz,xz,xy (3D) / xx,yy,xy (2D).
class IsotropicVoigtCoefficient : public MatrixCoefficient
{
public:
   IsotropicVoigtCoefficient(int dim, Coefficient &lambda_, Coefficient &mu_)
      : MatrixCoefficient(dim*(dim + 1)/2), dim_(dim), lambda(&lambda_),
        mu(&mu_) {}

   void Eval(DenseMatrix &K, ElementTransformation &T,
            const IntegrationPoint &ip) override
   {
      const real_t L = lambda->Eval(T, ip);
      const real_t M = mu->Eval(T, ip);

      K.SetSize(height, width);
      K = 0.0;
      for (int i = 0; i < dim_; i++)
      {
         for (int j = 0; j < dim_; j++)
         {
            K(i, j) += L;
         }
         K(i, i) += 2.0*M;
      }
      for (int s = dim_; s < height; s++)
      {
         K(s, s) = M;
      }
   }

private:
   int dim_;
   Coefficient *lambda, *mu;
};

real_t LambdaField(const Vector &x)
{
   real_t val = 5.0;
   for (int d = 0; d < x.Size(); d++) { val += 0.5*std::sin((d + 1)*x(d)); }
   return val;
}

real_t MuField(const Vector &x)
{
   real_t val = 3.0;
   for (int d = 0; d < x.Size(); d++) { val += 0.25*std::cos((d + 1)*x(d)); }
   return val;
}

/// Maps a Voigt shear index (3, 4, or 5) from ElasticityIntegrator's
/// ordering (xy, xz, yz) to AnisotropicElasticityIntegrator's ordering
/// (yz, xz, xy). Indices 0-2 (normal components) and the full 2D ordering
/// are identical between the two integrators.
int FluxIndexToAniso(int dim, int idx)
{
   if (dim == 2 || idx < 3) { return idx; }
   static const int shear_map[3] = {5, 4, 3}; // xy->5, xz->4, yz->3
   return shear_map[idx - 3];
}

struct Material
{
   const char *name;
   Coefficient *lambda;
   Coefficient *mu;
};

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   int order_max = 3;
   real_t tol = 1e-10;
   bool verbose = false;

   OptionsParser args(argc, argv);
   args.AddOption(&order_max, "-o", "--max-order",
                  "Test finite element orders 1 through this value.");
   args.AddOption(&tol, "-tol", "--tolerance",
                  "Relative-error tolerance for pass/fail.");
   args.AddOption(&verbose, "-v", "--verbose", "-no-v", "--no-verbose",
                  "Print per-case relative errors.");
   args.ParseCheck();

   ConstantCoefficient lambda_const(3.1), mu_const(1.7);
   FunctionCoefficient lambda_fun(LambdaField), mu_fun(MuField);

   Material materials[] =
   {
      {"constant", &lambda_const, &mu_const},
      {"spatially varying", &lambda_fun, &mu_fun}
   };

   real_t worst_elmat_error = 0.0;
   real_t worst_flux_error = 0.0;
   real_t worst_energy_error = 0.0;

   for (int dim = 2; dim <= 3; dim++)
   {
      Array<Element::Type> etypes;
      if (dim == 2)
      {
         etypes.Append(Element::QUADRILATERAL);
         etypes.Append(Element::TRIANGLE);
      }
      else
      {
         etypes.Append(Element::HEXAHEDRON);
         etypes.Append(Element::TETRAHEDRON);
      }

      for (Element::Type etype : etypes)
      {
         unique_ptr<Mesh> mesh;
         if (dim == 2)
         {
            mesh.reset(new Mesh(Mesh::MakeCartesian2D(3, 3, etype, false,
                                                       1.0, 1.0)));
         }
         else
         {
            mesh.reset(new Mesh(Mesh::MakeCartesian3D(2, 2, 2, etype,
                                                       1.0, 1.0, 1.0)));
         }
         const Geometry::Type geom = mesh->GetElementGeometry(0);

         for (int order = 1; order <= order_max; order++)
         {
            H1_FECollection fec(order, dim);
            const FiniteElement *el = fec.FiniteElementForGeometry(geom);
            const int tdim = dim*(dim + 1)/2;

            for (Material &mat : materials)
            {
               IsotropicVoigtCoefficient C_coef(dim, *mat.lambda, *mat.mu);
               ElasticityIntegrator iso_integ(*mat.lambda, *mat.mu);
               AnisotropicElasticityIntegrator aniso_integ(C_coef);

               real_t mat_elmat_err = 0.0;
               for (int e = 0; e < mesh->GetNE(); e++)
               {
                  ElementTransformation *Trans =
                     mesh->GetElementTransformation(e);

                  DenseMatrix elmat_iso, elmat_aniso;
                  iso_integ.AssembleElementMatrix(*el, *Trans, elmat_iso);
                  aniso_integ.AssembleElementMatrix(*el, *Trans, elmat_aniso);

                  DenseMatrix diff(elmat_iso);
                  diff -= elmat_aniso;
                  const real_t ref = std::max(elmat_iso.FNorm(),
                                              (real_t) 1e-300);
                  const real_t rel_err = diff.FNorm()/ref;
                  mat_elmat_err = std::max(mat_elmat_err, rel_err);
               }
               worst_elmat_error = std::max(worst_elmat_error, mat_elmat_err);

               // Stress and strain-energy comparison, on the first element.
               ElementTransformation *Trans =
                  mesh->GetElementTransformation(0);
               const int dof = el->GetDof();
               Vector u(dof*dim);
               for (int i = 0; i < u.Size(); i++)
               {
                  u(i) = std::sin(0.7*(i + 1)) - std::cos(0.3*(i + 1));
               }

               Vector flux_iso, flux_aniso;
               iso_integ.ComputeElementFlux(*el, *Trans, u, *el, flux_iso);
               aniso_integ.ComputeElementFlux(*el, *Trans, u, *el, flux_aniso);

               const int fnd = flux_iso.Size()/tdim;
               real_t flux_num = 0.0, flux_den = 0.0;
               for (int i = 0; i < fnd; i++)
               {
                  for (int d = 0; d < tdim; d++)
                  {
                     const int d2 = FluxIndexToAniso(dim, d);
                     const real_t diff_val =
                        flux_iso(i + fnd*d) - flux_aniso(i + fnd*d2);
                     flux_num += diff_val*diff_val;
                     flux_den += flux_iso(i + fnd*d)*flux_iso(i + fnd*d);
                  }
               }
               const real_t flux_rel_err =
                  std::sqrt(flux_num)/std::max(std::sqrt(flux_den),
                                               (real_t) 1e-300);
               worst_flux_error = std::max(worst_flux_error, flux_rel_err);

               const real_t energy_iso =
                  iso_integ.ComputeFluxEnergy(*el, *Trans, flux_iso);
               const real_t energy_aniso =
                  aniso_integ.ComputeFluxEnergy(*el, *Trans, flux_aniso);
               const real_t energy_rel_err =
                  std::abs(energy_iso - energy_aniso) /
                  std::max(std::abs(energy_iso), (real_t) 1e-300);
               worst_energy_error =
                  std::max(worst_energy_error, energy_rel_err);

               if (verbose && Mpi::Root())
               {
                  cout << "dim=" << dim
                       << " geom=" << Geometry::Name[geom]
                       << " order=" << order
                       << " mat=" << mat.name
                       << " | elmat rel. err=" << scientific
                       << setprecision(3) << mat_elmat_err
                       << " flux rel. err=" << flux_rel_err
                       << " energy rel. err=" << energy_rel_err << "\n";
               }
            }
         }
      }
   }

   const bool passed = (worst_elmat_error < tol) &&
                       (worst_flux_error < tol) &&
                       (worst_energy_error < tol);

   if (Mpi::Root())
   {
      cout << "\nAnisotropic vs. isotropic elasticity integrator check\n"
           << scientific << setprecision(4)
           << "  worst element matrix relative error : "
           << worst_elmat_error << "\n"
           << "  worst flux relative error           : "
           << worst_flux_error << "\n"
           << "  worst flux energy relative error     : "
           << worst_energy_error << "\n"
           << "  tolerance                            : " << tol << "\n"
           << (passed ? "PASSED\n" : "FAILED\n");
   }

   return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
