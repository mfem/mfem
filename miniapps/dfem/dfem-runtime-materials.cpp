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
//      --------------------------------------------------------------------
//      dFEM: mixed-material mass operator with sparse material densities
//      --------------------------------------------------------------------
//
// Compile with: make dfem-runtime-materials
//
// Sample runs:  mpirun -np 4 dfem-runtime-materials
//               mpirun -np 4 dfem-runtime-materials -nm 13
//               mpirun -np 4 dfem-runtime-materials -nm 40 -r 3 -o 3
//
// Description:  Mass operator on a mesh of M elements with n materials, n
//               chosen at runtime (-nm). Material m occupies the vertical
//               stripe m/n <= x < (m+1)/n, so elements cut by a stripe
//               boundary contain several materials (mixed elements):
//
//                  R(u, rho)_i = \int sigma(x) u phi_i dx,
//
//                  1 / sigma|_e = \sum_{m in e} 1 / rho_{m,e},
//
//               where the sum runs only over the materials present in element
//               e and rho_{m,e} is the density of material m in e. The mixing
//               rule is nonlinear and couples all materials present in an
//               element.
//
//               Material data is stored sparsely. A CSR MaterialMap lists, for
//               every element, the materials present in it, giving nnz
//               (element, material) pairs with nnz << n M. The densities live on
//               a SparseMaterialParameterSpace: one value per (element,
//               material) pair, nnz values in total, local to each rank.
//
//               The space gathers to K slots per quadrature point, where K is
//               the maximum number of materials in any local element,
//               determined at runtime. Unused slots are zero, so the gather B
//               is linear and also maps derivative directions correctly. The
//               scatter (Bt) sums over the quadrature points of each (element,
//               material) pair and adds the result to its entry.
//
//               A MaterialMaskSpace, which stores nothing, gathers 1 to used
//               and 0 to unused slots. The quadrature function skips the
//               unused slots, where 1 / rho would be singular.
//
//               Neither n nor K appears in a template argument. The quadrature
//               function takes the material data as tensor_ndarray<., 2>, a
//               (slot, point) array whose slot count the global backend derives
//               from the size of the field at runtime.
//
//               The example checks, for dR/du and dR/drho:
//
//               - the JVP (Mult) against central finite differences, and
//               - the VJP (MultTranspose) against the JVP, <lambda, J d> =
//                 <J^T lambda, d>.
//
//               The primal residual is checked against a MassIntegrator with an
//               equivalent hand-written Coefficient.

#include "mfem.hpp"
#include "../../fem/dfem/doperator.hpp"
#include "../../linalg/tensor_arrays.hpp"

#include <array>
#include <iomanip>

using namespace mfem;
using namespace mfem::future;

#ifdef MFEM_USE_ENZYME
using dreal_t = real_t;
#else
using dreal_t = dual<real_t, real_t>;
#endif

/// Sparse element -> material incidence in CSR format.
///
/// The materials of element e are mats[offsets[e]] ... mats[offsets[e+1]-1].
/// Entry i in [0, nnz) is one (element, material) pair; entry_elem[i] is its
/// element.
struct MaterialMap
{
   int nmat = 0;
   Array<int> offsets;    // size M + 1
   Array<int> mats;       // size nnz
   Array<int> entry_elem; // size nnz

   int NumElements() const { return offsets.Size() - 1; }
   int NumEntries() const { return mats.Size(); }
   int MaxPerElement() const
   {
      int k = 0;
      for (int e = 0; e < NumElements(); e++)
      {
         k = std::max(k, offsets[e + 1] - offsets[e]);
      }
      return k;
   }
};

/// Common gather/scatter for material data.
///
/// Q-vector layout: K slots per quadrature point with nparam values each,
/// c + nparam * (s + K * (q + nqp * e)). Slot s < (number of materials in
/// e) holds the data of the s-th material of e, the remaining slots are zero.
/// The zero fill keeps B linear: dFEM applies it to derivative directions as
/// well as to the state, so a nonzero fill would put spurious tangents into
/// unused slots.
///
/// T-vector layout: nparam values per CSR entry, i.e. per (element, material)
/// pair, in CSR order.
class MaterialSpaceBase : public ParameterSpace
{
public:
   /// Number of material slots per quadrature point.
   int GetNumSlots() const { return K; }

   int GetTrueVSize() const override { return tsize * nparam; }

   int GetVSize() const override { return tsize * nparam; }

   const Operator *GetB() const override
   {
      if (!B) { B = std::make_unique<Gather>(*this); }
      return B.get();
   }

   const Operator *GetBt() const override
   {
      if (!Bt) { Bt = std::make_unique<Scatter>(*this); }
      return Bt.get();
   }

protected:
   /// The number of slots K is the largest number of materials in any local
   /// element, so the Q-vector is only as wide as this rank needs.
   MaterialSpaceBase(const MaterialMap &map, const IntegrationRule &ir,
                     int nparam, int tsize) :
      ParameterSpace(std::max(1, map.MaxPerElement()) * nparam),
      map(map), K(std::max(1, map.MaxPerElement())), nparam(nparam),
      nqp(ir.GetNPoints()), tsize(tsize) {}

   const MaterialMap &map;
   const int K, nparam, nqp, tsize;

private:
   /// B: T-entries -> K slots at every quadrature point
   class Gather : public Operator
   {
   public:
      Gather(const MaterialSpaceBase &s) :
         Operator(s.K * s.nparam * s.nqp * s.map.NumElements(),
                  s.tsize * s.nparam), s(s) {}

      void Mult(const Vector &x, Vector &y) const override
      {
         const int K = s.K, np = s.nparam, nqp = s.nqp;
         const auto off = s.map.offsets.Read();
         const auto xr = x.Read();
         auto yw = y.Write();
         mfem::forall(nqp * s.map.NumElements(), [=] MFEM_HOST_DEVICE(int qe)
         {
            const int e = qe / nqp;
            const int begin = off[e], count = off[e + 1] - begin;
            for (int slot = 0; slot < K; slot++)
            {
               for (int c = 0; c < np; c++)
               {
                  yw[c + np * (slot + K * qe)] =
                     (slot < count) ? xr[c + np * (begin + slot)] : 0.0;
               }
            }
         });
      }

   private:
      const MaterialSpaceBase &s;
   };

   /// Bt: K slots at every quadrature point -> T-entries. One thread per
   /// (element, material) pair sums over the element's quadrature points and
   /// writes its own entry, so no atomics are needed.
   class Scatter : public Operator
   {
   public:
      Scatter(const MaterialSpaceBase &s) :
         Operator(s.tsize * s.nparam,
                  s.K * s.nparam * s.nqp * s.map.NumElements()), s(s) {}

      void Mult(const Vector &x, Vector &y) const override
      {
         const int K = s.K, np = s.nparam, nqp = s.nqp;
         const auto off = s.map.offsets.Read();
         const auto elem = s.map.entry_elem.Read();
         const auto xr = x.Read();
         auto yw = y.Write();
         mfem::forall(s.map.NumEntries(), [=] MFEM_HOST_DEVICE(int i)
         {
            const int e = elem[i], slot = i - off[e];
            for (int c = 0; c < np; c++)
            {
               real_t sum = 0.0;
               for (int q = 0; q < nqp; q++)
               {
                  sum += xr[c + np * (slot + K * (q + nqp * e))];
               }
               yw[c + np * i] = sum;
            }
         });
      }

   private:
      const MaterialSpaceBase &s;
   };
};

/// Per-(element, material) data: nnz T-entries, owned by the rank that owns
/// the element. Prolongation and element restriction are identities.
class SparseMaterialParameterSpace : public MaterialSpaceBase
{
public:
   SparseMaterialParameterSpace(const MaterialMap &map,
                                const IntegrationRule &ir, int nparam) :
      MaterialSpaceBase(map, ir, nparam, map.NumEntries()) {}
};

/// Slot mask: 1 in slots that hold a material, 0 in unused slots. Stores
/// nothing (T- and L-vector size 0); its gather ignores its input and writes the
/// mask from the material map. Use it as a non-differentiated input to let a
/// quadrature function skip unused slots without relying on data values.
class MaterialMaskSpace : public MaterialSpaceBase
{
public:
   MaterialMaskSpace(const MaterialMap &map, const IntegrationRule &ir) :
      MaterialSpaceBase(map, ir, 1, 0) {}

   const Operator *GetB() const override
   {
      if (!B) { B = std::make_unique<MaskGather>(*this); }
      return B.get();
   }

   const Operator *GetBt() const override
   {
      if (!Bt) { Bt = std::make_unique<ZeroScatter>(*this); }
      return Bt.get();
   }

private:
   class MaskGather : public Operator
   {
   public:
      MaskGather(const MaterialMaskSpace &s) :
         Operator(s.K * s.nqp * s.map.NumElements(), 0), s(s) {}

      void Mult(const Vector &, Vector &y) const override
      {
         const int K = s.K, nqp = s.nqp;
         const auto off = s.map.offsets.Read();
         auto yw = y.Write();
         mfem::forall(nqp * s.map.NumElements(), [=] MFEM_HOST_DEVICE(int qe)
         {
            const int e = qe / nqp, count = off[e + 1] - off[e];
            for (int slot = 0; slot < K; slot++)
            {
               yw[slot + K * qe] = (slot < count) ? 1.0 : 0.0;
            }
         });
      }

   private:
      const MaterialMaskSpace &s;
   };

   /// The mask has no T-entries, so there is nothing to scatter to.
   class ZeroScatter : public Operator
   {
   public:
      ZeroScatter(const MaterialMaskSpace &s) :
         Operator(0, s.K * s.nqp * s.map.NumElements()) {}

      void Mult(const Vector &, Vector &) const override {}
   };
};

/// Material m occupies the stripe m/n <= x < (m+1)/n. Builds the element ->
/// material map of the local elements: element e contains material m if their
/// x-ranges overlap.
void BuildStripeMaterials(ParMesh &mesh, int nmat, MaterialMap &map)
{
   const int ne = mesh.GetNE();
   map.nmat = nmat;
   map.offsets.SetSize(ne + 1);
   map.offsets[0] = 0;
   map.mats.SetSize(0);
   map.entry_elem.SetSize(0);

   Array<int> verts;
   for (int e = 0; e < ne; e++)
   {
      mesh.GetElementVertices(e, verts);
      real_t x0 = mesh.GetVertex(verts[0])[0], x1 = x0;
      for (int v : verts)
      {
         x0 = std::min(x0, mesh.GetVertex(v)[0]);
         x1 = std::max(x1, mesh.GetVertex(v)[0]);
      }
      for (int m = 0; m < nmat; m++)
      {
         const real_t overlap = std::min(x1, real_t(m + 1) / nmat) -
                                std::max(x0, real_t(m) / nmat);
         if (overlap > 1e-12 * (x1 - x0))
         {
            map.mats.Append(m);
            map.entry_elem.Append(e);
         }
      }
      map.offsets[e + 1] = map.mats.Size();
   }
}

/// sigma u v with 1/sigma = sum_s 1 / rho_s over the material slots of the
/// point. mask and rho are (slot, point) arrays; the number of slots is a
/// runtime size. mask is 1 in slots that hold a material and 0 in unused ones,
/// which are skipped: rho is zero there and 1 / rho singular.
template <int DIM>
struct MixtureMassQF
{
   void operator()(tensor_array<const dreal_t> &u,
                   tensor_ndarray<const real_t, 2> &mask,
                   tensor_ndarray<const dreal_t, 2> &rho,
                   tensor_array<const real_t, DIM, DIM> &J,
                   tensor_array<const real_t> &weights,
                   tensor_array<dreal_t> &v) const
   {
      const int nslots = static_cast<int>(rho.size(0));
      mfem::forall(weights.size(), [=] MFEM_HOST_DEVICE(int q)
      {
         dreal_t inv_sigma{};
         for (int s = 0; s < nslots; s++)
         {
            const real_t used = mask(s, q);
            if (used == 0.0) { continue; } // unused slot
            const dreal_t rs = rho(s, q);
            inv_sigma += 1.0 / rs;
         }
         v(q) = u(q) / inv_sigma * (det(J(q)) * weights(q));
      });
   }
};

/// Reference coefficient for the primal check: sigma on element e, computed
/// directly from the CSR data.
class MixtureCoefficient : public Coefficient
{
public:
   MixtureCoefficient(const MaterialMap &map, const Vector &rho) :
      map(map), rho(rho) {}

   real_t Eval(ElementTransformation &T, const IntegrationPoint &) override
   {
      const int e = T.ElementNo;
      real_t inv_sigma = 0.0;
      for (int i = map.offsets[e]; i < map.offsets[e + 1]; i++)
      {
         inv_sigma += 1.0 / rho(i);
      }
      return 1.0 / inv_sigma;
   }

private:
   const MaterialMap &map;
   const Vector &rho;
};

int main(int argc, char *argv[])
{
   Mpi::Init();
   const int myid = Mpi::WorldRank();
   Hypre::Init();

   int nmat = 5;
   int order = 2;
   int refinements = 2;
   real_t fd_eps = 1e-6;

   OptionsParser args(argc, argv);
   args.AddOption(&nmat, "-nm", "--num-materials", "Number of materials.");
   args.AddOption(&order, "-o", "--order", "Finite element order.");
   args.AddOption(&refinements, "-r", "--refine",
                  "Number of uniform serial refinements.");
   args.AddOption(&fd_eps, "-eps", "--fd-eps", "Finite difference step.");
   args.ParseCheck();
   MFEM_VERIFY(nmat > 0, "need at least one material");

   constexpr int DIM = 2;

   Mesh smesh = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL);
   for (int l = 0; l < refinements; l++) { smesh.UniformRefinement(); }
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   smesh.Clear();
   pmesh.EnsureNodes();
   const MPI_Comm comm = pmesh.GetComm();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   ParFiniteElementSpace &nfes = *nodes->ParFESpace();

   H1_FECollection fec(order, DIM);
   ParFiniteElementSpace u_fes(&pmesh, &fec);
   const IntegrationRule &ir =
      IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * order + 2);

   // Sparse material layout of the local elements.
   MaterialMap map;
   BuildStripeMaterials(pmesh, nmat, map);

   SparseMaterialParameterSpace rho_space(map, ir, 1);
   MaterialMaskSpace mask_space(map, ir);

   {
      long long loc[3] = {pmesh.GetNE(), map.NumEntries(), 0}, glob[3];
      for (int e = 0; e < pmesh.GetNE(); e++)
      {
         loc[2] += (map.offsets[e + 1] - map.offsets[e] > 1);
      }
      MPI_Allreduce(loc, glob, 3, MPI_LONG_LONG, MPI_SUM, comm);
      int kmax = map.MaxPerElement(), kmax_glob;
      MPI_Allreduce(&kmax, &kmax_glob, 1, MPI_INT, MPI_MAX, comm);
      if (myid == 0)
      {
         std::cout << "Number of unknowns (u):   " << u_fes.GlobalTrueVSize()
                   << "\nElements M:               " << glob[0]
                   << " (" << glob[2] << " mixed)"
                   << "\nMaterials n:              " << nmat
                   << "\n(element, material) pairs: " << glob[1]
                   << " (dense would be n M = " << glob[0] * nmat << ")"
                   << "\nMax materials per element: " << kmax_glob
                   << " (runtime slot count)\n";
      }
   }

   // Field IDs. U and RHO double as indices into the std::array states below.
   static constexpr int U = 0, RHO = 1, X = 2, MASK = 3;

   const std::vector<FieldDescriptor> in_fds =
   {
      {U, &u_fes}, {RHO, &rho_space}, {X, &nfes}, {MASK, &mask_space}
   };
   const std::vector<FieldDescriptor> out_fds = {{U, &u_fes}};
   DifferentiableOperator R(in_fds, out_fds, pmesh);

   Array<int> all_attr(pmesh.attributes.Max());
   all_attr = 1;

   constexpr auto kernels =
      DerivativeKernels::Action | DerivativeKernels::ApplyTranspose;
   MixtureMassQF<DIM> qf;
   R.AddDomainIntegrator<GlobalQFBackend, kernels>(
      qf,
      Inputs<Value<U>, Value<MASK>, Value<RHO>, Gradient<X>, Weight> {},
      Outputs<Value<U>> {},
      ir, all_attr, Derivatives<U, RHO> {});

   // State. rho differs per material and varies from element to element.
   ParGridFunction u_gf(&u_fes);
   FunctionCoefficient u_coef([](const Vector &x)
   {
      return std::sin(M_PI * x(0)) * std::cos(0.5 * M_PI * x(1)) + x(0) * x(1);
   });
   u_gf.ProjectCoefficient(u_coef);
   Vector u(u_fes.GetTrueVSize()), x(nfes.GetTrueVSize());
   Vector mask(mask_space.GetTrueVSize()); // size 0, the mask stores nothing
   u_gf.GetTrueDofs(u);
   nodes->GetTrueDofs(x);

   ParameterFunction rho(rho_space);
   for (int i = 0; i < map.NumEntries(); i++)
   {
      Vector c;
      pmesh.GetElementCenter(map.entry_elem[i], c);
      rho(i) = (1.0 + 0.25 * map.mats[i]) *
               (1.0 + 0.3 * std::sin(2.0 * M_PI * c(0)) * std::cos(M_PI * c(1)));
   }

   const int n = u_fes.GetTrueVSize();
   bool pass = true;
   auto report = [&](const std::string &what, real_t err, real_t tol)
   {
      const bool ok = err <= tol;
      pass = pass && ok;
      if (myid == 0)
      {
         std::cout << std::left << std::setw(44) << what << std::right
                   << std::scientific << std::setprecision(3) << err
                   << (ok ? "  ok" : "  FAIL") << std::defaultfloat << "\n";
      }
   };

   // Residual for the differentiable inputs (u, rho).
   using state_t = std::array<Vector, 2>;
   auto residual = [&](const state_t &st, Vector &r)
   {
      MultiVector out{r};
      R.Mult(MultiVector{st[U], st[RHO], x, mask}, out);
   };
   const state_t state{u, rho};

   // 1) Primal residual against MassIntegrator with the same sigma.
   {
      Vector r(n), r_ref(n);
      residual(state, r);

      MixtureCoefficient sigma(map, rho);
      ParBilinearForm a(&u_fes);
      a.AddDomainIntegrator(new MassIntegrator(sigma, &ir));
      a.Assemble();
      a.Finalize();
      std::unique_ptr<HypreParMatrix> A(a.ParallelAssemble());
      A->Mult(u, r_ref);

      r -= r_ref;
      report("|R - R_ref| / |R_ref|",
             std::sqrt(InnerProduct(comm, r, r) /
                       InnerProduct(comm, r_ref, r_ref)), 1e-12);
   }

   // 2) JVP vs FD and VJP vs JVP for each derivative.
   MultiVector state_mv{u, rho, x, mask};
   Vector lambda(n);
   lambda.Randomize(1234 + myid);

   const std::array<const char *, 2> names{"u", "rho"};
   for (int id : {U, RHO})
   {
      auto dR = R.GetDerivative(id, state_mv);
      const int width = state[id].Size();
      MFEM_VERIFY(dR->Width() == width, "unexpected derivative width");

      Vector d(width);
      d.Randomize(77 + myid);

      // JVP
      Vector Jd(n);
      {
         MultiVector Jd_mv{Jd};
         dR->Mult(d, Jd_mv);
      }

      // Central finite differences of the residual
      state_t plus = state, minus = state;
      plus[id].Add(fd_eps, d);
      minus[id].Add(-fd_eps, d);
      Vector rp(n), rm(n), fd(n);
      residual(plus, rp);
      residual(minus, rm);
      subtract(1.0 / (2.0 * fd_eps), rp, rm, fd);

      Vector diff(Jd);
      diff -= fd;
      report(std::string("|dR/d") + names[id] + " d - FD| / |FD|",
             std::sqrt(InnerProduct(comm, diff, diff) /
                       InnerProduct(comm, fd, fd)), 1e-6);

      // VJP
      Vector JTl(width);
      {
         MultiVector l_mv{lambda}, JTl_mv{JTl};
         dR->MultTranspose(l_mv, JTl_mv);
      }
      const real_t lhs = InnerProduct(comm, lambda, Jd);
      const real_t rhs = InnerProduct(comm, JTl, d);
      report(std::string("|<l, dR/d") + names[id] + " d> - <dR/d" +
             names[id] + "^T l, d>| / |.|",
             std::abs(lhs - rhs) / std::abs(lhs), 1e-12);
   }

   if (myid == 0) { std::cout << (pass ? "\nPASSED" : "\nFAILED") << std::endl; }
   return pass ? 0 : 1;
}
