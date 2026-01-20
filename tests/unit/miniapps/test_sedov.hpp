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

#define CATCH_CONFIG_RUNNER
#include "mfem.hpp"
#include "run_unit_tests.hpp"

using namespace mfem;

namespace mfem
{

template <class TMesh,
          class TGridFunction,
          class TBilinearForm,
          class TFiniteElementSpace>
struct Types
{
   using Mesh = TMesh;
   using GridFunction = TGridFunction;
   using BilinearForm = TBilinearForm;
   using FiniteElementSpace = TFiniteElementSpace;
};

#if defined(MFEM_USE_MPI) && (defined(MFEM_SEDOV_PA_MPI) || defined(MFEM_SEDOV_DFEM_MPI))
using T = Types<ParMesh, ParGridFunction, ParBilinearForm, ParFiniteElementSpace>;
static auto ParMeshFromSeq = [](Mesh &mesh) { return ParMesh(MPI_COMM_WORLD, mesh); };
static auto TrueVSize = [](auto &fes) { return fes.GlobalTrueVSize(); };
static auto GetCGSolver = []() { return CGSolver(MPI_COMM_WORLD); };
template <typename T, typename OP>
static void Reduce(const T *src, T *dst, OP op)
{
   MPI_Allreduce(src, dst, 1, MPITypeMap<T>::mpi_type, op, MPI_COMM_WORLD);
}
template <typename T> void SumReduce(const T *src, T *dst) { Reduce(src, dst, MPI_SUM); }
template <typename T> void MinReduce(const T *src, T *dst) { Reduce(src, dst, MPI_MIN); }
template <typename T> void MaxReduce(const T *src, T *dst) { Reduce(src, dst, MPI_MAX); }
#else
using T = Types<Mesh, GridFunction, BilinearForm, FiniteElementSpace>;
static auto ParMeshFromSeq = [](Mesh &mesh) { return Mesh(mesh); };
static auto TrueVSize = [](auto &fes) { return fes.GetVSize(); };
static auto GetCGSolver = []() { return CGSolver(); };
template <typename T> void Reduce(const T *src, T *dst) { *dst = *src; }
template <typename T> void SumReduce(const T *src, T *dst) { Reduce(src, dst); }
template <typename T> void MinReduce(const T *src, T *dst) { Reduce(src, dst); }
template <typename T> void MaxReduce(const T *src, T *dst) { Reduce(src, dst); }
#endif

static void v0_fn(const Vector&, Vector &v) { v = 0.0; }
static real_t rho0_fn(const Vector&) { return 1.0; }
static real_t gamma_fn(const Vector&) { return 1.4; }

template <int DIM>
class LagrangianHydro : public TimeDependentOperator
{
   typename T::FiniteElementSpace &H1, &L2;
   mutable typename T::FiniteElementSpace H1c;
   const int H1Vsize, H1TVSize, H1cTVSize, L2Vsize, L2TVSize;
   Array<int> block_offsets;
   mutable typename T::GridFunction x_gf;
   const Array<int> &ess_tdofs;
   const int nzones, l2dofs_cnt, h1dofs_cnt, source_type;
   const real_t cfl;
   const bool use_viscosity;
   const real_t cg_rel_tol;
   const int cg_max_iter;
   const real_t ftz_tol;
   const Coefficient &material_pcf;
   const IntegrationRule &ir;
   //    mutable QuadratureData quad_data;
   mutable bool quad_data_is_current;
   //    PAForceOperator<typename T::FiniteElementSpace> force;
   //    PAMassOperator<typename T::FiniteElementSpace, typename T::BilinearForm>
   //    VMassPA, EMassPA;
   CGSolver CG_VMass, CG_EMass;
   const real_t gamma;
   //    mutable QUpdate<DIM, typename T::FiniteElementSpace> Q;
   mutable Vector X, B, one, rhs, e_rhs;
   mutable typename T::GridFunction rhs_c_gf, dvc_gf;
   mutable Array<int> c_tdofs[3];

   void UpdateQuadratureData(const Vector &S) const = 0;
   //    {
   //       return Q.UpdateQuadratureData(S, quad_data_is_current, quad_data);
   //    }

public:
   LagrangianHydro(Coefficient &rho_coeff, const int size,
                   typename T::FiniteElementSpace &h1,
                   typename T::FiniteElementSpace &l2,
                   typename T::Mesh &pmesh,
                   const Array<int> &essential_tdofs,
                   typename T::GridFunction &rho0,
                   const int source_type,
                   const real_t cfl_,
                   const Coefficient &material,
                   const bool visc,
                   const real_t cgt,
                   const int cgiter,
                   real_t ftz,
                   const int order_q,
                   const real_t gm,
                   int h1_basis_type):
      TimeDependentOperator(size),
      H1(h1), L2(l2),
      H1c(&pmesh, h1.FEColl(), 1),
      H1Vsize(H1.GetVSize()),
      H1TVSize(H1.GetTrueVSize()),
      H1cTVSize(H1c.GetTrueVSize()),
      L2Vsize(L2.GetVSize()),
      L2TVSize(L2.GetTrueVSize()),
      block_offsets(4),
      x_gf(&H1),
      ess_tdofs(essential_tdofs),
      nzones(h1.GetMesh()->GetNE()),
      l2dofs_cnt(l2.GetTypicalFE()->GetDof()),
      h1dofs_cnt(h1.GetTypicalFE()->GetDof()),
      source_type(source_type), cfl(cfl_),
      use_viscosity(visc),
      cg_rel_tol(cgt), cg_max_iter(cgiter),ftz_tol(ftz),
      material_pcf(material),
      ir(IntRules.Get(h1.GetMesh()->GetTypicalElementGeometry(),
                      (order_q > 0) ? order_q :
                      3*h1.GetElementOrder(0) + l2.GetElementOrder(0) - 1)),
      //   quad_data(DIM, nzones, ir.GetNPoints()),
      quad_data_is_current(false),
      //   force(quad_data, h1,l2, ir),
      //   VMassPA(rho_coeff, H1c, ir),
      //   EMassPA(rho_coeff, L2, ir),
      CG_VMass(GetCGSolver()),
      CG_EMass(GetCGSolver()),
      gamma(gm),
      //   Q(DIM, nzones, use_viscosity, cfl, gamma, ir, H1, L2),
      X(H1c.GetTrueVSize()),
      B(H1c.GetTrueVSize()),
      one(L2Vsize),
      rhs(H1Vsize),
      e_rhs(L2Vsize),
      rhs_c_gf(&H1c),
      dvc_gf(&H1c)
   {
      block_offsets[0] = 0;
      block_offsets[1] = block_offsets[0] + H1Vsize;
      block_offsets[2] = block_offsets[1] + H1Vsize;
      block_offsets[3] = block_offsets[2] + L2Vsize;
      one.UseDevice(true);
      one = 1.0;
      H1.GetMesh()->GetNodes()->ReadWrite();
      const int bdr_attr_max = H1.GetMesh()->bdr_attributes.Max();
      Array<int> ess_bdr(bdr_attr_max);
      for (int c = 0; c < DIM; c++)
      {
         ess_bdr = 0; ess_bdr[c] = 1;
         H1c.GetEssentialTrueDofs(ess_bdr, c_tdofs[c]);
         c_tdofs[c].Read();
      }
      X.UseDevice(true);
      B.UseDevice(true);
      rhs.UseDevice(true);
      e_rhs.UseDevice(true);
      GridFunctionCoefficient rho_coeff_gf(&rho0);
      //   real_t loc_area = 0.0, glob_area;
      //   int loc_z_cnt = nzones, glob_z_cnt;
      //   auto *pm = H1.GetMesh();
      //   ComputeRho0DetJ0AndVolume<DIM>(nzones, ir,
      //                                  H1.GetMesh(),
      //                                  l2, rho0, quad_data, loc_area);
      //   SumReduce(&loc_area, &glob_area);
      //   SumReduce(&loc_z_cnt, &glob_z_cnt);
      //   switch (pm->GetTypicalElementGeometry())
      //   {
      //      case Geometry::SQUARE: quad_data.h0 = sqrt(glob_area / glob_z_cnt); break;
      //      case Geometry::CUBE: quad_data.h0 = pow(glob_area / glob_z_cnt, 1.0/3.0); break;
      //      default: MFEM_ABORT("Unknown zone type!");
      //   }
      //   quad_data.h0 /= (real_t) H1.GetElementOrder(0);

      //   CG_VMass.SetOperator(VMassPA);
      CG_VMass.SetRelTol(cg_rel_tol);
      CG_VMass.SetAbsTol(0.0);
      CG_VMass.SetMaxIter(cg_max_iter);
      CG_VMass.SetPrintLevel(0);

      //   CG_EMass.SetOperator(EMassPA);
      CG_EMass.iterative_mode = false;
      CG_EMass.SetRelTol(1e-8);
      CG_EMass.SetAbsTol(1e-8 * std::numeric_limits<real_t>::epsilon());
      CG_EMass.SetMaxIter(200);
      CG_EMass.SetPrintLevel(-1);
   }

   void Mult(const Vector &S, Vector &dS_dt) const override
   {
      UpdateMesh(S);
      auto *sptr = const_cast<Vector*>(&S);
      typename T::GridFunction v, dx;
      const int VsizeH1 = H1.GetVSize();
      v.MakeRef(&H1, *sptr, VsizeH1);
      dx.MakeRef(&H1, dS_dt, 0);
      dx = v;
      SolveVelocity(S, dS_dt);
      SolveEnergy(S, v, dS_dt);
      quad_data_is_current = false;
   }

   MemoryClass GetMemoryClass() const override  { return Device::GetDeviceMemoryClass(); }

   void SolveVelocity(const Vector &S, Vector &dS_dt) const
   {
      UpdateQuadratureData(S);
      typename T::GridFunction dv(&H1, dS_dt, H1Vsize);
      dv = 0.0;
      //   force.Mult(one, rhs);
      rhs.Neg();
      const int size = H1c.GetVSize();
      const Operator *Pconf = H1c.GetProlongationMatrix();
      const Operator *Rconf = H1c.GetRestrictionMatrix();
      for (int c = 0; c < DIM; c++)
      {
         dvc_gf.MakeRef(&H1c, dS_dt, H1Vsize + c*size);
         rhs_c_gf.MakeRef(&H1c, rhs, c*size);
         if (Pconf) { Pconf->MultTranspose(rhs_c_gf, B); }
         else { B = rhs_c_gf; }
         if (Rconf) { Rconf->Mult(dvc_gf, X); }
         else { X = dvc_gf; }
         //  VMassPA.SetEssentialTrueDofs(c_tdofs[c]);
         //  VMassPA.EliminateRHS(B);
         CG_VMass.Mult(B, X);
         if (Pconf) { Pconf->Mult(X, dvc_gf); }
         else { dvc_gf = X; }
         dvc_gf.GetMemory().SyncAlias(dS_dt.GetMemory(), dvc_gf.Size());
      }
   }

   void SolveEnergy(const Vector &S, const Vector &v, Vector &dS_dt) const
   {
      UpdateQuadratureData(S);
      typename T::GridFunction de;
      de.MakeRef(&L2, dS_dt, H1Vsize*2);
      de = 0.0;
      //   force.MultTranspose(v, e_rhs);
      CG_EMass.Mult(e_rhs, de);
      de.GetMemory().SyncAlias(dS_dt.GetMemory(), de.Size());
   }

   void UpdateMesh(const Vector &const_S) const
   {
      auto *S = const_cast<Vector*>(&const_S);
      x_gf.MakeRef(&H1, *S, 0);
      H1.GetMesh()->NewNodes(x_gf, false);
   }

   real_t GetTimeStepEstimate(const Vector &S) const
   {
      UpdateMesh(S);
      UpdateQuadratureData(S);
      real_t glob_dt_est;
      //   MinReduce(&quad_data.dt_est, &glob_dt_est);
      return glob_dt_est;
   }

   void ResetTimeStepEstimate() const
   {
      //   quad_data.dt_est = std::numeric_limits<real_t>::infinity();
   }

   void ResetQuadratureData() const { quad_data_is_current = false; }
};

} // namespace mfem

template <int DIM, template<int> typename TLagrangianHydroOperator>
int sedov(int myid, int argc, char *argv[])
{
   int rs_levels = 0;
   int max_tsteps = -1;
   constexpr int rp_levels = 0;
   constexpr int order_v = 2;
   constexpr int order_e = 1;
   constexpr int order_q = -1;
   constexpr real_t t_final = 0.6;
   constexpr real_t cfl = 0.5;
   constexpr real_t cg_tol = 1e-14;
   constexpr real_t ftz_tol = 0.0;
   constexpr int cg_max_iter = 300;
   constexpr int vis_steps = 5;
   constexpr real_t blast_energy = 0.25;
   constexpr real_t blast_position[] = {0.0, 0.0, 0.0};
   constexpr int source = 0;
   constexpr bool visc = true;

   OptionsParser args(argc, argv);
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&max_tsteps, "-ms", "--max-steps",
                  "Maximum number of steps (negative means no restriction).");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(mfem::out); }
      return -1;
   }

   Mesh mesh;
   if constexpr (DIM == 2)
   {
      constexpr Element::Type QUAD = Element::QUADRILATERAL;
      mesh = Mesh::MakeCartesian2D(2, 2, QUAD, true);
      const int NBE = mesh.GetNBE();
      for (int b = 0; b < NBE; b++)
      {
         Element *bel = mesh.GetBdrElement(b);
         MFEM_ASSERT(bel->GetType() == Element::SEGMENT, "");
         const int attr = (b < NBE/2) ? 2 : 1;
         bel->SetAttribute(attr);
      }
   }

   if constexpr (DIM == 3)
   {
      mesh = Mesh::MakeCartesian3D(2, 2, 2,Element::HEXAHEDRON);
      const int NBE = mesh.GetNBE();
      MFEM_ASSERT(NBE == 24,"");
      for (int b = 0; b < NBE; b++)
      {
         Element *bel = mesh.GetBdrElement(b);
         MFEM_ASSERT(bel->GetType() == Element::QUADRILATERAL, "");
         const int attr = (b < NBE/3) ? 3 : (b < 2*NBE/3) ? 1 : 2;
         bel->SetAttribute(attr);
      }
   }

   for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }

   auto pmesh = ParMeshFromSeq(mesh);
   mesh.Clear();
   for (int lev = 0; lev < rp_levels; lev++) { pmesh.UniformRefinement(); }

   int nzones = pmesh.GetNE(), nzones_min, nzones_max;
   MinReduce(&nzones, &nzones_min);
   MaxReduce(&nzones, &nzones_max);
   if (nzones_min == 0)
   {
      if (myid == 0) { mfem::out << "Some MPI ranks have no zones. Exiting." << std::endl; }
      return EXIT_SUCCESS;
   }

   L2_FECollection L2fec(order_e, DIM, BasisType::Positive);
   H1_FECollection H1fec(order_v, DIM);
   typename T::FiniteElementSpace L2(&pmesh, &L2fec);
   typename T::FiniteElementSpace H1(&pmesh, &H1fec, pmesh.Dimension());
   Array<int> ess_tdofs;
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max()), tdofs1d;
      for (int d = 0; d < pmesh.Dimension(); d++)
      {
         ess_bdr = 0; ess_bdr[d] = 1;
         H1.GetEssentialTrueDofs(ess_bdr, tdofs1d, d);
         ess_tdofs.Append(tdofs1d);
      }
   }

   RK4Solver ode_solver;

   const auto H1TVSize = TrueVSize(H1), L2TVSize = TrueVSize(L2);
   const auto H1Vsize = H1.GetVSize(), L2Vsize = L2.GetVSize();
   if (myid == 0)
   {
      mfem::out << "Number of local/global kinematic (position, velocity) dofs: "
                << H1Vsize << "/" << H1TVSize << std::endl;
      mfem::out << "Number of local/global specific internal energy dofs: "
                << L2Vsize << "/" << L2TVSize << std::endl;
   }

   Array<int> true_offset(4);
   true_offset[0] = 0;
   true_offset[1] = true_offset[0] + H1Vsize;
   true_offset[2] = true_offset[1] + H1Vsize;
   true_offset[3] = true_offset[2] + L2Vsize;
   BlockVector S(true_offset, Device::GetDeviceMemoryType());
   S.UseDevice(true);

   typename T::GridFunction x_gf, v_gf, e_gf;
   x_gf.MakeRef(&H1, S, true_offset[0]);
   v_gf.MakeRef(&H1, S, true_offset[1]);
   e_gf.MakeRef(&L2, S, true_offset[2]);
   pmesh.SetNodalGridFunction(&x_gf);
   x_gf.SyncAliasMemory(S);

   VectorFunctionCoefficient v_coeff(pmesh.Dimension(), v0_fn);
   v_gf.ProjectCoefficient(v_coeff);
   v_gf.SyncAliasMemory(S);

   typename T::GridFunction rho(&L2);
   FunctionCoefficient rho_fct_coeff(rho0_fn);
   ConstantCoefficient rho_coeff(1.0);
   L2_FECollection l2_fec(order_e, pmesh.Dimension());
   typename T::FiniteElementSpace l2_fes(&pmesh, &l2_fec);
   typename T::GridFunction l2_rho(&l2_fes), l2_e(&l2_fes);
   l2_rho.ProjectCoefficient(rho_fct_coeff);
   rho.ProjectGridFunction(l2_rho);

   DeltaCoefficient e_coeff(blast_position[0], blast_position[1],
                            blast_position[2], blast_energy);
   e_coeff.SetWeight(new ConstantCoefficient(1.0));
   l2_e.ProjectCoefficient(e_coeff);
   e_gf.ProjectGridFunction(l2_e);
   e_gf.SyncAliasMemory(S);

   L2_FECollection mat_fec(0, pmesh.Dimension());
   typename T::FiniteElementSpace mat_fes(&pmesh, &mat_fec);
   typename T::GridFunction mat_gf(&mat_fes);
   FunctionCoefficient mat_coeff(gamma_fn);
   mat_gf.ProjectCoefficient(mat_coeff);
   GridFunctionCoefficient mat_gf_coeff(&mat_gf);

   TLagrangianHydroOperator<DIM> oper(rho_coeff, S.Size(),
                                      H1, L2, pmesh, ess_tdofs,
                                      rho, source, cfl, mat_gf_coeff, visc,
                                      cg_tol, cg_max_iter, ftz_tol,
                                      order_q, gamma_fn(S), H1fec.GetBasisType());

   ode_solver.Init(oper);
   oper.ResetTimeStepEstimate();
   real_t t = 0.0, t_old, dt = oper.GetTimeStepEstimate(S);
   bool last_step = false;
   int steps = 0;
   BlockVector S_old(S);
   int checks = 0;
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final)
      {
         dt = t_final - t;
         last_step = true;
      }
      if (steps == max_tsteps) { last_step = true; }
      S_old = S;
      t_old = t;
      oper.ResetTimeStepEstimate();
      ode_solver.Step(S, t, dt);
      steps++;
      const real_t dt_est = oper.GetTimeStepEstimate(S);
      if (dt_est < dt)
      {
         dt *= 0.85;
         if (dt < std::numeric_limits<real_t>::epsilon())
         { MFEM_ABORT("The time step crashed!"); }
         t = t_old;
         S = S_old;
         oper.ResetQuadratureData();
         if (myid == 0) { mfem::out << "Repeating step " << ti << std::endl; }
         if (steps < max_tsteps) { last_step = false; }
         ti--; continue;
      }
      else if (dt_est > 1.25 * dt) { dt *= 1.02; }
      x_gf.SyncAliasMemory(S);
      v_gf.SyncAliasMemory(S);
      e_gf.SyncAliasMemory(S);
      pmesh.NewNodes(x_gf, false);
      if (last_step || (ti % vis_steps) == 0)
      {
         real_t loc_norm = e_gf * e_gf, tot_norm;
         SumReduce(&loc_norm, &tot_norm);
         if (myid == 0)
         {
            const real_t sqrt_tot_norm = sqrt(tot_norm);
            mfem::out << std::fixed;
            mfem::out << "step " << std::setw(5) << ti
                      << ",\tt = " << std::setw(5) << std::setprecision(4) << t
                      << ",\tdt = " << std::setw(5) << std::setprecision(6) << dt
                      << ",\t|e| = " << std::setprecision(10)
                      << sqrt_tot_norm;
            mfem::out << std::endl;
         }
      }
      real_t loc_norm = e_gf * e_gf, tot_norm;
      SumReduce(&loc_norm, &tot_norm);
      const real_t stm = sqrt(tot_norm);
      REQUIRE((rs_levels == 0 || rs_levels == 1));
      if constexpr (DIM == 2)
      {
         constexpr real_t p1_05[2] = { 3.508254945225794e+00, 1.403249766367977e+01 };
         constexpr real_t p1_15[2] = { 2.756444596823211e+00, 1.104093401469385e+01 };
         if (ti==5) {checks++; REQUIRE(stm == MFEM_Approx(p1_05[rs_levels]));}
         if (ti==15) {checks++; REQUIRE(stm == MFEM_Approx(p1_15[rs_levels]));}
      }
      if constexpr (DIM == 3)
      {
         constexpr real_t p1_05[2] = { 1.339163718592567e+01, 1.071277540097426e+02 };
         constexpr real_t p1_28[2] = { 7.521073677398005e+00, 5.985720905709158e+01 };
         if (ti==5) {checks++; REQUIRE(stm == MFEM_Approx(p1_05[rs_levels]));}
         if (ti==28) {checks++; REQUIRE(stm == MFEM_Approx(p1_28[rs_levels]));}
      }
   }
   REQUIRE(checks == 2);
   steps *= 4;
   return EXIT_SUCCESS;
}

static inline int argn(const char *argv[], int argc = 0)
{
   while (argv[argc]) { argc+=1; }
   return argc;
}

template <template<int> typename T>
static void sedov_tests(int rank)
{
   const char *argv2D[]= { "sedov<2>", nullptr };
   REQUIRE(sedov<2, T>(rank, argn(argv2D), const_cast<char**>(argv2D)) == 0);

   const char *argv2Drs1[]= { "sedov<2>", "-rs", "1", "-ms", "20", nullptr };
   REQUIRE(sedov<2, T>(rank, argn(argv2Drs1), const_cast<char**>(argv2Drs1)) == 0);

   const char *argv3D[]= { "sedov<3>", nullptr };
   REQUIRE(sedov<3, T>(rank, argn(argv3D), const_cast<char**>(argv3D)) == 0);

   //    const char *argv3Drs1[]= { "sedov<3>", "-rs", "1", "-ms", "28", nullptr };
   //    REQUIRE(sedov<3, T>(rank, argn(argv3Drs1), const_cast<char**>(argv3Drs1)) == 0);
}