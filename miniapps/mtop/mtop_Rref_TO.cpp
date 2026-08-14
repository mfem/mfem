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
// Sample runs:
//    mpirun -np 4 mtop_test_iso_elasticity -tri -o 2
//    mpirun -np 4 mtop_test_iso_elasticity -tri -o 2 -pa
//    mpirun -np 4 mtop_test_iso_elasticity -tri -o 2 -dfem
//    mpirun -np 4 mtop_test_iso_elasticity -tri -o 3 -dfem
//
//    mpirun -np 4 mtop_test_iso_elasticity -quad -o 2
//    mpirun -np 4 mtop_test_iso_elasticity -quad -o 2 -pa
//    mpirun -np 4 mtop_test_iso_elasticity -quad -o 2 -dfem
//    mpirun -np 4 mtop_test_iso_elasticity -quad -o 3 -dfem -prl 2
//
// Device sample runs:
//    mpirun -np 4 mtop_test_iso_elasticity -d gpu -quad -o 2
//    mpirun -np 4 mtop_test_iso_elasticity -d gpu -quad -o 2 -pa
//    mpirun -np 4 mtop_test_iso_elasticity -d gpu -quad -o 2 -dfem
//    mpirun -np 4 mtop_test_iso_elasticity -d gpu -quad -o 3 -dfem

#include "mtop_solvers.hpp"
#include "tmop_ad_err.hpp"
#include "mfem.hpp"

using namespace std;
using namespace mfem;

constexpr auto MESH_TRI = MFEM_SOURCE_DIR "/miniapps/mtop/sq_2D_9_tri.mesh";
constexpr auto MESH_QUAD = MFEM_SOURCE_DIR "/miniapps/mtop/sq_2D_9_quad.mesh";

   real_t function_(const Vector &x)
   {
      if (x[0] <=2.5 && x[0] >=1.0 && x[1] <=0.8 && x[1] >=0.2)
      {
        return 10.0;
        //return 1.0 + 100* sin((x[0]-0.5)*M_PI/2.0) * sin((x[1]-0.2)*M_PI/0.6);
      }
   return 1.0;
   }

///////////////////////////////////////////////////////////////////////////////
  template <int DIM>
  struct InternalComplianceQF
  {
     using matd_t = mfem::future::tensor<real_t, DIM, DIM>;

     MFEM_HOST_DEVICE inline void operator()(const matd_t &dudxi,
                                             const real_t &L,
                                             const real_t &M,
                                             const matd_t &J,
                                             const real_t &w,
                                             real_t &q) const
     {
        const auto Jinv = mfem::future::inv(J);
        constexpr auto I = mfem::future::IsotropicIdentity<DIM>();

        const auto eps = mfem::future::sym(dudxi * Jinv);
        const auto sigma = L * tr(eps) * I + 2.0 * M * eps;

        q = mfem::future::inner(eps, sigma) * det(J) * w;
     }
  };

template <int DIM>
  struct AdjointResidualFunctionalQF
  {
     using matd_t = mfem::future::tensor<real_t, DIM, DIM>;

     MFEM_HOST_DEVICE inline void operator()(const matd_t &dudxi,
                                             const matd_t &dadjdxi,
                                             const real_t &rho,
                                             const real_t &L,
                                             const real_t &M,
                                             const matd_t &J,
                                             const real_t &w,
                                             real_t &q) const
     {
        constexpr real_t exponent = 3.0;
        constexpr real_t rho_min = 1e-3;

        const auto Jinv = mfem::future::inv(J);
        constexpr auto I = mfem::future::IsotropicIdentity<DIM>();

        const auto eps_u = mfem::future::sym(dudxi * Jinv);
        const auto eps_adj = mfem::future::sym(dadjdxi * Jinv);
        const auto sigma_u = L * tr(eps_u) * I + 2.0 * M * eps_u;

        const real_t simp = rho_min + mfem::future::pow(rho, exponent) *
                                        (1.0 - rho_min);

        q = -simp * mfem::future::inner(eps_adj, sigma_u) * det(J) * w;
     }
  };

  template <int DIM>
  struct AdjointElasticEnergyQF
  {
     using matd_t = mfem::future::tensor<real_t, DIM, DIM>;

     MFEM_HOST_DEVICE inline void operator()(const matd_t &dudxi,
                                             const matd_t &dadjdxi,
                                             const real_t &L,
                                             const real_t &M,
                                             const matd_t &J,
                                             const real_t &w,
                                             real_t &q) const
     {
        const auto Jinv = mfem::future::inv(J);
        constexpr auto I = mfem::future::IsotropicIdentity<DIM>();

        const auto eps_u = mfem::future::sym(dudxi * Jinv);
        const auto eps_adj = mfem::future::sym(dadjdxi * Jinv);
        const auto sigma_u = L * tr(eps_u) * I + 2.0 * M * eps_u;

        q = mfem::future::inner(eps_adj, sigma_u) * det(J) * w;
     }
  };

   template <int DIM>
  struct TMOPMetric002QF
  {
     using matd_t = mfem::future::tensor<real_t, DIM, DIM>;

     MFEM_HOST_DEVICE inline void operator()(const matd_t &Jpr,
                                             const matd_t &Winv,
                                             const real_t &detW,
                                             const real_t &w,
                                             real_t &q) const
     {
        const auto Jpt = Jpr * Winv;
        q = (0.5 * mfem::future::inner(Jpt, Jpt) / det(Jpt) - 1.0)
            * detW * w;
     }
  };

   void GetQuadPointsPositions(const mfem::ParMesh & pmesh_init, const QuadratureSpace &qspace,
      const Vector &pos_mesh,  Vector &pos_quads)
{
   const int NE  = qspace.GetMesh()->GetNE(), dim = pmesh_init.Dimension();
   const int nsp = qspace.GetElementIntRule(0).GetNPoints();

   pos_quads.SetSize(nsp * NE * dim);
   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir = qspace.GetElementIntRule(e);

      // Transformation of the element with the pos_mesh coordinates.
      IsoparametricTransformation Tr;
      pmesh_init.GetElementTransformation(e, pos_mesh, &Tr);

      // Node positions of pfes for pos_mesh.
      DenseMatrix pos_quads_e;
      Tr.Transform(ir, pos_quads_e);
      Vector rowx(pos_quads.GetData() + e*nsp, nsp),
             rowy(pos_quads.GetData() + e*nsp + NE*nsp, nsp), rowz;
      if (dim == 3)
      {
         rowz.SetDataAndSize(pos_quads.GetData() + e*nsp + 2*NE*nsp, nsp);
      }
      pos_quads_e.GetRow(0, rowx);
      pos_quads_e.GetRow(1, rowy);
      if (dim == 3) { pos_quads_e.GetRow(2, rowz); }
   }
}

/// @brief Inverse sigmoid function
real_t inv_sigmoid(real_t x)
{
   real_t tol = 1e-12;
   x = std::min(std::max(tol,x), real_t(1.0)-tol);
   return std::log(x/(1.0-x));
}

/// @brief Sigmoid function
real_t sigmoid(real_t x)
{
   if (x >= 0)
   {
      return 1.0/(1.0+std::exp(-x));
   }
   else
   {
      return std::exp(x)/(1.0+std::exp(x));
   }
}

/// @brief Derivative of sigmoid function
real_t der_sigmoid(real_t x)
{
   real_t tmp = sigmoid(-x);
   return tmp - std::pow(tmp,2);
}

/// @brief Returns f(u(x)) where u is a scalar GridFunction and f:R → R
class MappedGridFunctionCoefficient : public GridFunctionCoefficient
{
protected:
   std::function<real_t(const real_t)> fun; // f:R → R
public:
   MappedGridFunctionCoefficient()
      :GridFunctionCoefficient(),
       fun([](real_t x) {return x;}) {}
   MappedGridFunctionCoefficient(const GridFunction *gf,
                                 std::function<real_t(const real_t)> fun_,
                                 int comp=1)
      :GridFunctionCoefficient(gf, comp),
       fun(fun_) {}


   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      return fun(GridFunctionCoefficient::Eval(T, ip));
   }
   void SetFunction(std::function<real_t(const real_t)> fun_) { fun = fun_; }
};


/// @brief Returns f(u(x)) - f(v(x)) where u, v are scalar GridFunctions and f:R → R
class DiffMappedGridFunctionCoefficient : public GridFunctionCoefficient
{
protected:
   const GridFunction *OtherGridF;
   GridFunctionCoefficient OtherGridF_cf;
   std::function<real_t(const real_t)> fun; // f:R → R
public:
   DiffMappedGridFunctionCoefficient()
      :GridFunctionCoefficient(),
       OtherGridF(nullptr),
       OtherGridF_cf(),
       fun([](real_t x) {return x;}) {}
   DiffMappedGridFunctionCoefficient(const GridFunction *gf,
                                     const GridFunction *other_gf,
                                     std::function<real_t(const real_t)> fun_,
                                     int comp=1)
      :GridFunctionCoefficient(gf, comp),
       OtherGridF(other_gf),
       OtherGridF_cf(OtherGridF),
       fun(fun_) {}

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      const real_t value1 = fun(GridFunctionCoefficient::Eval(T, ip));
      const real_t value2 = fun(OtherGridF_cf.Eval(T, ip));
      return value1 - value2;
   }
   void SetFunction(std::function<real_t(const real_t)> fun_) { fun = fun_; }
};

/// @brief Solid isotropic material penalization (SIMP) coefficient
class SIMPInterpolationCoefficient : public Coefficient
{
protected:
   GridFunction *rho_filter;
   real_t min_val;
   real_t max_val;
   real_t exponent;

public:
   SIMPInterpolationCoefficient(GridFunction *rho_filter_, real_t min_val_= 1e-6,
                                real_t max_val_ = 1.0, real_t exponent_ = 3)
      : rho_filter(rho_filter_), min_val(min_val_), max_val(max_val_),
        exponent(exponent_) { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      real_t val = rho_filter->GetValue(T, ip);
      real_t coeff = min_val + pow(val,exponent)*(max_val-min_val);
      return coeff;
   }
};

class SIMPInterpolationCoefficientUsingCoeff : public Coefficient
{
protected:
   Coefficient *rho_filter;
   real_t min_val;
   real_t max_val;
   real_t exponent;

public:
   SIMPInterpolationCoefficientUsingCoeff(Coefficient *rho_filter_, real_t min_val_= 1e-6,
                                real_t max_val_ = 1.0, real_t exponent_ = 3)
      : rho_filter(rho_filter_), min_val(min_val_), max_val(max_val_),
        exponent(exponent_) { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      real_t val = rho_filter->Eval(T, ip);
      real_t coeff = min_val + pow(val,exponent)*(max_val-min_val);
      return coeff;
   }
};

class StrainEnergyDensityCoefficient : public Coefficient
{
protected:
   Coefficient * lambda=nullptr;
   Coefficient * mu=nullptr;
   GridFunction *u = nullptr; // displacement
   GridFunction *rho_filter = nullptr; // filter density
   DenseMatrix grad; // auxiliary matrix, used in Eval
      Vector vec; // auxiliary matrix, used in Eval
   real_t exponent;
   real_t rho_min;

public:
   StrainEnergyDensityCoefficient(Coefficient *lambda_, Coefficient *mu_,
                                  GridFunction * u_, GridFunction * rho_filter_, real_t rho_min_=1e-6,
                                  real_t exponent_ = 3.0)
      : lambda(lambda_), mu(mu_),  u(u_), rho_filter(rho_filter_),
        exponent(exponent_), rho_min(rho_min_)
   {
      MFEM_ASSERT(rho_min_ >= 0.0, "rho_min must be >= 0");
      MFEM_ASSERT(rho_min_ < 1.0,  "rho_min must be > 1");
      MFEM_ASSERT(u, "displacement field is not set");
      MFEM_ASSERT(rho_filter, "density field is not set");
   }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      real_t L = lambda->Eval(T, ip);
      real_t M = mu->Eval(T, ip);
      u->GetVectorGradient(T, grad);
      u->GetVectorValue(T, ip, vec);
      real_t div_u = grad.Trace();
      real_t density = L*div_u*div_u;
      int dim = T.GetSpaceDim();
      for (int i=0; i<dim; i++)
      {
         for (int j=0; j<dim; j++)
         {
            density += M*grad(i,j)*(grad(i,j)+grad(j,i));
         }
      }
      real_t val = rho_filter->GetValue(T,ip);

      return -exponent * pow(val, exponent-1.0) * (1-rho_min) * density;
   }
};


real_t proj(ParGridFunction &psi, real_t target_volume, real_t tol=1e-12,
            int max_its=10)
{
   MappedGridFunctionCoefficient sigmoid_psi(&psi, sigmoid);
   MappedGridFunctionCoefficient der_sigmoid_psi(&psi, der_sigmoid);

   ParLinearForm int_sigmoid_psi(psi.ParFESpace());
   int_sigmoid_psi.AddDomainIntegrator(new DomainLFIntegrator(sigmoid_psi));
   ParLinearForm int_der_sigmoid_psi(psi.ParFESpace());
   int_der_sigmoid_psi.AddDomainIntegrator(new DomainLFIntegrator(
                                              der_sigmoid_psi));
   bool done = false;
   for (int k=0; k<max_its; k++) // Newton iteration
   {
      int_sigmoid_psi.Assemble(); // Recompute f(c) with updated ψ
      real_t f = int_sigmoid_psi.Sum();
      MPI_Allreduce(MPI_IN_PLACE, &f, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, MPI_COMM_WORLD);
      f -= target_volume;

      int_der_sigmoid_psi.Assemble(); // Recompute df(c) with updated ψ
      real_t df = int_der_sigmoid_psi.Sum();
      MPI_Allreduce(MPI_IN_PLACE, &df, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, MPI_COMM_WORLD);

      const real_t dc = -f/df;
      psi += dc;
      if (abs(dc) < tol) { done = true; break; }
   }
   if (!done)
   {
      mfem_warning("Projection reached maximum iteration without converging. "
                   "Result may not be accurate.");
   }
   int_sigmoid_psi.Assemble();
   real_t material_volume = int_sigmoid_psi.Sum();
   MPI_Allreduce(MPI_IN_PLACE, &material_volume, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
   return material_volume;
}


int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init();
   Hypre::Init();

   // Parse command-line options.
   const char *mesh_file = MESH_QUAD;
   const char *device_config = "cpu";
   int order = 1;
   bool pa = true;
   bool dfem = false;
   bool mesh_tri = false;
   bool mesh_quad = false;
   int par_ref_levels = 0;
   bool paraview = true;
   bool visualization = true;
   real_t epsilon = 0.02;       //dens filterrad
   real_t alpha = 1.0;
   real_t vol_fraction = 0.5;
   real_t rho_min = 1e-3;
   real_t lambda = 1.0;
   real_t mu = 1.0;
   // real_t lambda = 0.5769230769;
   // real_t mu = 1.0/2.6;
   const int ref_levels = 5;

   int neumannBCIndex = 2;  // 1 based
   double neumannLoad = -5.0e-1;

   double weight_1 = -1e03;
   double weight_tmop = 1e-2;

   double filterRadius = 0.2;

     double max_ch=0.02;       //opt solver step size

   bool dQduFD =false;
   bool dQdxFD =false;
   bool dQdxFD_global =false;
   bool BreakAfterFirstIt = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&dfem, "-dfem", "--dFEM", "-no-dfem", "--no-dFEM",
                  "Enable or not dFEM.");
   args.AddOption(&mesh_tri, "-tri", "--triangular", "-no-tri",
                  "--no-triangular", "Enable or not triangular mesh.");
   args.AddOption(&mesh_quad, "-quad", "--quadrilateral", "-no-quad",
                  "--no-quadrilateral", "Enable or not quadrilateral mesh.");
   args.AddOption(&par_ref_levels, "-prl", "--par-ref-levels",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable or not Paraview visualization");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();
   MFEM_VERIFY(!(pa && dfem), "pa and dfem cannot be both set");

   // Enable hardware devices such as GPUs, and programming models such as
   // CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   real_t Nx = 3;
   real_t Ny = 1;

   real_t Lx = 3.0;
   real_t Ly = 1.0;

   Mesh mesh = Mesh::MakeCartesian2D(Nx, Ny, mfem::Element::Type::QUADRILATERAL,
                                     true, Lx, Ly);
   const int dim = mesh.Dimension();
   constexpr int DIM = 2;

   for (int l = 0; l < ref_levels; l++) { mesh.UniformRefinement(); }

   if (Mpi::Root())
   {
      std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
   }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   for (int l = 0; l < par_ref_levels; l++) { pmesh.UniformRefinement(); }

   H1_FECollection fec(order, dim);
   ParFiniteElementSpace state_fes(&pmesh, &fec,dim, Ordering::byNODES);
   ParFiniteElementSpace coord_fes(&pmesh, &fec,dim, Ordering::byNODES);
   ParFiniteElementSpace filter_fes(&pmesh, &fec);
   ParFiniteElementSpace control_fes(&pmesh, &fec);

   pmesh.SetNodalFESpace(&coord_fes);
   mfem::ParGridFunction coords_(&coord_fes);
   pmesh.SetNodalGridFunction(&coords_);
   ParGridFunction x0(&coord_fes);
   x0 = coords_;

   // 5. Set the initial guess for ρ.
   ParGridFunction u(&state_fes);
   ParGridFunction psi(&control_fes);
   ParGridFunction psi_old(&control_fes);
   ParGridFunction rho_filter(&filter_fes);
   u = 0.0;
   rho_filter = vol_fraction;
   psi = inv_sigmoid(vol_fraction);
   psi_old = inv_sigmoid(vol_fraction);
   ConstantCoefficient one(1.0);
   ConstantCoefficient lambda_cf(lambda);
   ConstantCoefficient mu_cf(mu);

   // 8. Define the Lagrange multiplier and gradient functions.
   ParGridFunction grad(&control_fes);
   ParGridFunction w_filter(&filter_fes);

   // 9. Define some tools for later.
   ConstantCoefficient zero(0.0);
   ParGridFunction onegf(&control_fes);
   onegf = 1.0;
   ParGridFunction zerogf(&control_fes);
   zerogf = 0.0;
   ParLinearForm vol_form(&control_fes);
   vol_form.AddDomainIntegrator(new DomainLFIntegrator(one));
   vol_form.Assemble();
   real_t domain_volume = vol_form(onegf);
   const real_t target_volume = domain_volume * vol_fraction;

   // ρ = sigmoid(ψ)
   MappedGridFunctionCoefficient rho(&psi, sigmoid);
   // Interpolation of ρ = sigmoid(ψ) in control fes (for ParaView output)
   ParGridFunction rho_gf(&control_fes);
   rho_gf.ProjectCoefficient(rho);
   // ρ - ρ_old = sigmoid(ψ) - sigmoid(ψ_old)
   DiffMappedGridFunctionCoefficient succ_diff_rho(&psi, &psi_old, sigmoid);

   ParBilinearForm mass(&control_fes);
   mass.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator(one)));
   mass.Assemble();
   HypreParMatrix M;
   Array<int> empty;
   mass.FormSystemMatrix(empty,M);

   bool isConverged = true;

   ParaViewDataCollection paraview_dc("isoel", &pmesh);
   paraview_dc.SetPrefixPath("ParaView_Rref_TO");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetCycle(0);
   paraview_dc.SetTime(0.0);
   paraview_dc.RegisterField("disp", &u);
   paraview_dc.RegisterField("design", &rho_gf);
   paraview_dc.RegisterField("filtered_design", &rho_filter);   
   paraview_dc.Save();

   // set esing variable bounds
   int numOptVars = state_fes.GetTrueVSize();
   Vector objgrad(numOptVars); objgrad=0.0;
   Vector volgrad(numOptVars); volgrad=1.0;
   Vector xxmax(numOptVars);   xxmax=  0.001;
   Vector xxmin(numOptVars);   xxmin= -0.001;

   VectorArrayCoefficient tractionLoad(dim);
   tractionLoad.Set(0, new ConstantCoefficient( 0.0));
   tractionLoad.Set(1, new ConstantCoefficient(neumannLoad));

   while(isConverged)
   {
      // ==========================================================================================
      //                       Topology Optimization
      // ==========================================================================================

      // 1. Set-up the filter solver.
      ConstantCoefficient eps2_cf(epsilon*epsilon);
      DiffusionSolver * FilterSolver = new DiffusionSolver();
      FilterSolver->SetMesh(&pmesh);
      FilterSolver->SetOrder(fec.GetOrder());
      FilterSolver->SetDiffusionCoefficient(&eps2_cf);
      FilterSolver->SetMassCoefficient(&one);
      Array<int> ess_bdr_filter;
      if (pmesh.bdr_attributes.Size())
      {
         ess_bdr_filter.SetSize(pmesh.bdr_attributes.Max());
         ess_bdr_filter = 0;
      }
      FilterSolver->SetEssentialBoundary(ess_bdr_filter);
      FilterSolver->SetupFEM();

      // Create the solver
      IsoLinElasticSolver elsolver(&pmesh, order, pa, dfem);
      if (Mpi::Root())
      {
         std::cout << "Number of unknowns: "
                  << elsolver.GetSolutionVector().Size() << std::endl;
      }

      const auto ir = IntRules.Get(state_fes.GetFE(0)->GetGeomType(), state_fes.GetFE(0)->GetOrder() + state_fes.GetFE(0)->GetOrder() + state_fes.GetFE(0)->GetDim() - 1);

      elsolver.AddDispBC(4, -1, 0.0);
      elsolver.AddSurfLoad(neumannBCIndex, 0.0, neumannLoad);
      elsolver.SetLinearSolver(1e-6,1e-8,1000);

      int numTOit = 40;
      for( int ik = 0; ik < numTOit; ik++)
      {
         if (ik > 1) { alpha *= ((real_t) ik) / ((real_t) ik-1); }

         FilterSolver->SetRHSCoefficient(&rho);
         FilterSolver->Solve();
         rho_filter = *FilterSolver->GetFEMSolution();

         SIMPInterpolationCoefficient SIMP_cf(&rho_filter,rho_min, 1.0, 3.0);
         ProductCoefficient lambda_SIMP_cf(lambda_cf,SIMP_cf);
         ProductCoefficient mu_SIMP_cf(mu_cf,SIMP_cf);

         elsolver.SetMaterialLame(lambda_SIMP_cf, mu_SIMP_cf);
         elsolver.Assemble();
         elsolver.FSolve();
         elsolver.GetSol(u);

         mfem::Coefficient *rhs_cf = nullptr;
         mfem::ParGridFunction adjTimesdRdrho(&filter_fes); adjTimesdRdrho = 0.0;
         
      {
         Array<int> all_domain_attr;
         if (pmesh.attributes.Size() > 0)
         {
            all_domain_attr.SetSize(pmesh.attributes.Max());
            all_domain_attr = 1;
         }

         mfem::QuadratureSpace scalar_qs(pmesh, ir);
         mfem::VectorQuadratureSpace Lambda_scalar_qs(scalar_qs, 1);
         mfem::VectorQuadratureSpace Mu_scalar_qs(scalar_qs, 1);
         mfem::VectorQuadratureSpace q_scalar_qs(scalar_qs, 1);

         CoefficientVector Lambda_scalar_cv(lambda_cf, scalar_qs);
         CoefficientVector Mu_scalar_cv(mu_cf, scalar_qs);
         QuadratureFunction scalar_q(q_scalar_qs);

         static constexpr int uN = 0, adjN = 1, rhoN = 2;
         static constexpr int coordsN = 3, lambdaN = 4, muN = 5, qN = 6;

         mfem::future::DifferentiableOperator scalar_dop(
            std::vector<mfem::future::FieldDescriptor>
         {
            { uN, &state_fes },
            { adjN, &state_fes },
            { rhoN, &filter_fes },
            { coordsN, &coord_fes },
            { lambdaN, &Lambda_scalar_qs },
            { muN, &Mu_scalar_qs }
         },
         std::vector<mfem::future::FieldDescriptor>
         {
            { qN, &q_scalar_qs }
         },
         pmesh);

         const auto scalar_inputs = mfem::future::tuple
         {
            mfem::future::Gradient<uN>{},
            mfem::future::Gradient<adjN>{},
            mfem::future::Value<rhoN>{},
            mfem::future::Identity<lambdaN>{},
            mfem::future::Identity<muN>{},
            mfem::future::Gradient<coordsN>{},
            mfem::future::Weight{}
         };
         const auto scalar_outputs = mfem::future::tuple
         {
            mfem::future::FunctionalValue<qN>{}
         };
         const auto scalar_derivatives = std::integer_sequence<size_t, rhoN>{};

         if (2 == DIM)
         {
            AdjointResidualFunctionalQF<2> qf;
            scalar_dop.AddDomainIntegrator<mfem::future::LocalQFBackend>(
               qf, scalar_inputs, scalar_outputs, ir, all_domain_attr,
               scalar_derivatives);
         }
         else if (3 == DIM)
         {
            AdjointResidualFunctionalQF<3> qf;
            scalar_dop.AddDomainIntegrator<mfem::future::LocalQFBackend>(
               qf, scalar_inputs, scalar_outputs, ir, all_domain_attr,
               scalar_derivatives);
         }

         mfem::MultiVector scalar_x
         {
            u, u, rho_filter, coords_, Lambda_scalar_cv, Mu_scalar_cv
         };
         mfem::MultiVector scalar_y { scalar_q };

         scalar_dop.Mult(scalar_x, scalar_y);

         Vector scalar_dRdrho_true(filter_fes.GetTrueVSize());
         scalar_dop.GetDerivative(rhoN, scalar_x)->Assemble(scalar_dRdrho_true);

         adjTimesdRdrho.SetFromTrueDofs(scalar_dRdrho_true);

         rhs_cf = new mfem::GridFunctionCoefficient(&adjTimesdRdrho);
      }

         FilterSolver->SetRHSCoefficient(rhs_cf);
         FilterSolver->Solve();
         w_filter = *FilterSolver`->GetFEMSolution();


         delete rhs_cf;

         // Solve G = M⁻¹w̃
         GridFunctionCoefficient w_cf(&w_filter);
         ParLinearForm w_rhs(&control_fes);
         w_rhs.AddDomainIntegrator(new DomainLFIntegrator(w_cf));
         w_rhs.Assemble();
         M.Mult(w_rhs,grad);

         // Step 5 - Update design variable ψ ← proj(ψ - αG)
         psi.Add(-alpha, grad);
         const real_t material_volume = proj(psi, target_volume);

         // Compute ||ρ - ρ_old|| in control fes.
         real_t norm_increment = zerogf.ComputeL1Error(succ_diff_rho);
         real_t norm_reduced_gradient = norm_increment/alpha;
         psi_old = psi;

         rho_gf.ProjectCoefficient(rho);
         paraview_dc.SetCycle(ik+1);
         paraview_dc.SetTime(ik+1);
         paraview_dc.Save();
      }

      // ==========================================================================================
      //                       R - Refinement Optimization
      // ==========================================================================================

      ParGridFunction gridfuncOptVar(&coord_fes);
      gridfuncOptVar = 0.0;
      gridfuncOptVar.SetTrueVector();
      Vector & trueOptvar = gridfuncOptVar.GetTrueVector();

      Array<int> neumannBdr(pmesh.bdr_attributes.Max());
      std::cout<<"bdr_attributes: "<<pmesh.bdr_attributes.Max()<<std::endl;
      neumannBdr = 0; neumannBdr[neumannBCIndex-1] = 1;

      std::vector<std::pair<int, int>> essentialBCfilter(pmesh.bdr_attributes.Max());
      essentialBCfilter[0] = {1, 1};
      essentialBCfilter[1] = {2, 0};
      essentialBCfilter[2] = {3, 1};
      essentialBCfilter[3] = {4, 0};

      ConstantCoefficient filterRadiusCoeff(filterRadius);
      VectorHelmholtz  filterSolver(&pmesh, essentialBCfilter, filterRadius, order, order);

      int cycle_count = 1;
      double final_strain_energy = 0.0;

      Vector X0_;
      pmesh.GetNodes(X0_);

      ParGridFunction u_morph(&state_fes); u_morph = 0.0;
      ParGridFunction mesh_disp(&coord_fes); mesh_disp = 0.0;

      FunctionCoefficient desing_func(function_);
      ParGridFunction test_design_filter(&filter_fes);
    	test_design_filter.ProjectCoefficient (desing_func);

      QuadratureSpace constrolQuadSpace(pmesh, ir);
      QuadratureFunction dens_interp(&constrolQuadSpace);

      MMA* mma=new MMA(MPI_COMM_WORLD, trueOptvar.Size(), 0, trueOptvar);

      ParaViewDataCollection paraview_dc_morph("isoel_morph", &pmesh);
      paraview_dc_morph.SetPrefixPath("ParaView_Rref_TOaa");
      paraview_dc_morph.SetLevelsOfDetail(order);
      paraview_dc_morph.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_morph.SetHighOrderOutput(true);
      paraview_dc_morph.SetCycle(0);
      paraview_dc_morph.SetTime(0.0);
      paraview_dc_morph.RegisterField("disp", &u_morph);
      paraview_dc_morph.RegisterField("mesh_disp", &mesh_disp);
      paraview_dc_morph.RegisterField("design", &rho_filter);
      paraview_dc_morph.RegisterQField("designQuadrature", &dens_interp);


      paraview_dc_morph.Save();

      ParGridFunction gridfuncBoundIndicator(&coord_fes);
      ParGridFunction gridfuncBoundfunc_Min(&coord_fes);
      ParGridFunction gridfuncBoundfunc_Max(&coord_fes);
      gridfuncBoundIndicator = 0.0;
      gridfuncBoundfunc_Min = 0.0;
      gridfuncBoundfunc_Max = 0.0;
      Array<int> vdofs;
      int numNodes   = gridfuncBoundfunc_Min.Size() / DIM;
      mfem::Vector locationVector(DIM);

      int nodesinX = Nx*ref_levels+1;
      int nodesinY = Ny*ref_levels+1;

      int eleinX = Nx*ref_levels;
      int eleinY = Ny*ref_levels;

      real_t initialEleEdgeLenghtY = Ly / eleinY;
      real_t initialEleEdgeLenghtX = Lx / eleinX;

      real_t initialEleEdgeLenghtX_noroundoff = std::round(initialEleEdgeLenghtX * 10e5) / 10e5;
      real_t initialEleEdgeLenghtY_noroundoff = std::round(initialEleEdgeLenghtY * 10e5) / 10e5;
      
    for (int i = 0; i < pmesh.GetNBE(); i++)
    {
      Element * tEle = pmesh.GetBdrElement(i);
      int attribute = tEle->GetAttribute();
      coord_fes.GetBdrElementVDofs(i, vdofs);
      const int nd = coord_fes.GetBE(i)->GetDof();

      if (attribute == 1 || attribute == 3) // zero out motion in y
      {
        for (int j = 0; j < nd; j++)
        {
          gridfuncBoundIndicator[ vdofs[j+nd] ] = 1.0;
        }
      }
      else if (attribute == 2 || attribute == 4) // zero out in x
      {
        for (int j = 0; j < nd; j++)
        {
          gridfuncBoundIndicator[ vdofs[j] ] = 1.0;
        }
      }
    }

    for ( int Ik = 0; Ik<numNodes; Ik++)
    {
       pmesh.GetNode(Ik, &locationVector[0]);
       const double * pCoords(locationVector.GetData());

       int ijk_X = pCoords[0] / initialEleEdgeLenghtX_noroundoff;
       int ijk_Y = pCoords[1] / initialEleEdgeLenghtY_noroundoff;

       gridfuncBoundfunc_Max[Ik] = Lx - pCoords[0] - (nodesinX - ijk_X)*1e-6;
       gridfuncBoundfunc_Min[Ik] = -(pCoords[0] -  ijk_X*1e-6);

       gridfuncBoundfunc_Max[Ik + numNodes] = Ly - pCoords[1] - (nodesinY - ijk_Y)*1e-6;
       gridfuncBoundfunc_Min[Ik + numNodes] = -(pCoords[1] -  ijk_Y*1e-6);
    }

    gridfuncBoundIndicator.SetTrueVector();
    gridfuncBoundfunc_Max.SetTrueVector();
    gridfuncBoundfunc_Min.SetTrueVector();

      for(int i=1;i<100;i++)
      {
         filterSolver.setLoadGridFunction(gridfuncOptVar);
         filterSolver.FSolve();
         ParGridFunction & filteredNodePos = filterSolver.GetSolution();

         mesh_disp = filteredNodePos;

         Vector Xi = X0_;
         Xi += filteredNodePos;

         // ----------------------   gslib densities

         Vector pos_quad_final;
         GetQuadPointsPositions(pmesh, constrolQuadSpace, Xi, pos_quad_final);

         FindPointsGSLIB finder(pmesh.GetComm());
         finder.SetL2AvgType(FindPointsGSLIB::NONE);
         finder.Setup(pmesh);
         finder.Interpolate(pos_quad_final, rho_filter, dens_interp); 

         // ----------------------- compute and set material 

         // update mesh coordinates
         pmesh.SetNodes(Xi);
         pmesh.DeleteGeometricFactors();

         QuadratureFunctionCoefficient newDensityCoeff(dens_interp);         
         SIMPInterpolationCoefficientUsingCoeff SIMP_cf_morph(&newDensityCoeff,rho_min, 1.0, 1.0);
         //GridFunctionCoefficient tGFCoeff(&rho_filter);
         //SIMPInterpolationCoefficientUsingCoeff SIMP_cf_morph(&tGFCoeff,rho_min, 1.0, 3.0);
         ProductCoefficient lambda_SIMP_cf_morph(lambda_cf,SIMP_cf_morph);
         ProductCoefficient mu_SIMP_cf_morph(mu_cf,SIMP_cf_morph);

         u_morph = 0.0;
         IsoLinElasticSolver elsolver_morph(&pmesh, order, pa, dfem);
         elsolver_morph.AddDispBC(4, -1, 0.0);
         elsolver_morph.AddSurfLoad(neumannBCIndex, 0.0, neumannLoad);
         elsolver_morph.SetLinearSolver(1e-6,1e-8,1000);
         elsolver_morph.SetMaterialLame(lambda_SIMP_cf_morph, mu_SIMP_cf_morph);
         elsolver_morph.Assemble();
         elsolver_morph.FSolve();
         elsolver_morph.GetSol(u_morph);
    
         real_t ObjVal = 0.0;
         real_t dfem_mesh_quality = 0.0;
         Vector dfem_dQdu(state_fes.GetTrueVSize());
         Vector dfem_dQdx(coord_fes.GetTrueVSize());

         ParLinearForm dMeshQdxExpl(&coord_fes); dMeshQdxExpl = 0.0;
         dfem_dQdu = 0.0;
         dfem_dQdx = 0.0;

         {
            static constexpr int uDFEM = 0, coordsDFEM = 1;
            static constexpr int lambdaDFEM = 2, muDFEM = 3, qDFEM = 4;

            Array<int> all_domain_attr;
            if (pmesh.attributes.Size() > 0)
            {
               all_domain_attr.SetSize(pmesh.attributes.Max());
               all_domain_attr = 1;
            }

            mfem::QuadratureSpace compliance_qspace(pmesh, ir);
            mfem::VectorQuadratureSpace lambda_qspace(compliance_qspace, 1);
            mfem::VectorQuadratureSpace mu_qspace(compliance_qspace, 1);
            mfem::VectorQuadratureSpace q_qspace(compliance_qspace, 1);

            CoefficientVector lambda_cv(lambda_SIMP_cf_morph, compliance_qspace);
            CoefficientVector mu_cv(mu_SIMP_cf_morph, compliance_qspace);
            QuadratureFunction compliance_q(q_qspace);

            ParGridFunction current_coords(&coord_fes);
            Vector current_nodes;
            pmesh.GetNodes(current_nodes);
            current_coords = current_nodes;

            mfem::future::DifferentiableOperator compliance_dop(
               std::vector<mfem::future::FieldDescriptor>
            {
               { uDFEM, &state_fes },
               { lambdaDFEM, &lambda_qspace },
               { muDFEM, &mu_qspace },
               { coordsDFEM, &coord_fes }
            },
            std::vector<mfem::future::FieldDescriptor>
            {
               { qDFEM, &q_qspace }
            },
            pmesh);

            const auto compliance_inputs = mfem::future::tuple
            {
               mfem::future::Gradient<uDFEM>{},
               mfem::future::Identity<lambdaDFEM>{},
               mfem::future::Identity<muDFEM>{},
               mfem::future::Gradient<coordsDFEM>{},
               mfem::future::Weight{}
            };
            const auto compliance_outputs = mfem::future::tuple
            {
               mfem::future::FunctionalValue<qDFEM>{}
            };
            const auto compliance_derivatives =
               std::integer_sequence<size_t, uDFEM, coordsDFEM>{};

            if (2 == DIM)
            {
               InternalComplianceQF<2> qf;
               compliance_dop.AddDomainIntegrator<mfem::future::LocalQFBackend>(
                  qf, compliance_inputs, compliance_outputs, ir,
                  all_domain_attr, compliance_derivatives);
            }
            else if (3 == DIM)
            {
               InternalComplianceQF<3> qf;
               compliance_dop.AddDomainIntegrator<mfem::future::LocalQFBackend>(
                  qf, compliance_inputs, compliance_outputs, ir,
                  all_domain_attr, compliance_derivatives);
            }
            else { MFEM_ABORT("Space dimension not supported"); }

            mfem::MultiVector compliance_x
            {
               u_morph, lambda_cv, mu_cv, current_coords
            };
            mfem::MultiVector compliance_y { compliance_q };

            compliance_dop.Mult(compliance_x, compliance_y);
            ObjVal = compliance_q.Sum();
            MPI_Allreduce(MPI_IN_PLACE, &ObjVal, 1,
                          MPITypeMap<real_t>::mpi_type, MPI_SUM,
                          pmesh.GetComm());


            compliance_dop.GetDerivative(uDFEM, compliance_x)->Assemble(dfem_dQdu);
            compliance_dop.GetDerivative(coordsDFEM, compliance_x)->Assemble(dfem_dQdx);
         }

         {
            static constexpr int coordsT = 0, winvT = 1, detWT = 2, qT = 3;

            Array<int> all_domain_attr;
            if (pmesh.attributes.Size() > 0)
            {
               all_domain_attr.SetSize(pmesh.attributes.Max());
               all_domain_attr = 1;
            }

            mfem::QuadratureSpace tmop_qspace(pmesh, ir);
            mfem::VectorQuadratureSpace winv_qspace(tmop_qspace, DIM * DIM);
            mfem::VectorQuadratureSpace detW_qspace(tmop_qspace, 1);
            mfem::VectorQuadratureSpace q_qspace(tmop_qspace, 1);

            QuadratureFunction winv_q(winv_qspace);
            QuadratureFunction detW_q(detW_qspace);
            QuadratureFunction tmop_q(q_qspace);

            winv_q = 0.0;
            detW_q = 0.0;
            for (int e = 0; e < tmop_qspace.GetNE(); e++)
            {
               const DenseMatrix &Wideal =
                  Geometries.GetGeomToPerfGeomJac(pmesh.GetElementBaseGeometry(e));
               DenseMatrix Winv(DIM);
               CalcInverse(Wideal, Winv);
               const real_t detW = Wideal.Det();
               const IntegrationRule &el_ir = tmop_qspace.GetElementIntRule(e);
               const int offset = tmop_qspace.Offset(e);

               for (int q = 0; q < el_ir.GetNPoints(); q++)
               {
                  const int qidx = offset + q;
                  detW_q[qidx] = detW;
                  for (int i = 0; i < DIM; i++)
                  {
                     for (int j = 0; j < DIM; j++)
                     {
                        winv_q[qidx * DIM * DIM + i * DIM + j] = Winv(i, j);
                     }
                  }
               }
            }

            ParGridFunction current_coords(&coord_fes);
            Vector current_nodes;
            pmesh.GetNodes(current_nodes);
            current_coords = current_nodes;

            mfem::future::DifferentiableOperator tmop_dop(
               std::vector<mfem::future::FieldDescriptor>
            {
               { coordsT, &coord_fes },
               { winvT, &winv_qspace },
               { detWT, &detW_qspace }
            },
            std::vector<mfem::future::FieldDescriptor>
            {
               { qT, &q_qspace }
            },
            pmesh);

            const auto tmop_inputs = mfem::future::tuple
            {
               mfem::future::Gradient<coordsT>{},
               mfem::future::Identity<winvT>{},
               mfem::future::Identity<detWT>{},
               mfem::future::Weight{}
            };
            const auto tmop_outputs = mfem::future::tuple
            {
               mfem::future::FunctionalValue<qT>{}
            };
            const auto tmop_derivatives =
               std::integer_sequence<size_t, coordsT>{};

            if (2 == DIM)
            {
               TMOPMetric002QF<2> qf;
               tmop_dop.AddDomainIntegrator<mfem::future::LocalQFBackend>(
                  qf, tmop_inputs, tmop_outputs, ir, all_domain_attr,
                  tmop_derivatives);
            }
            else { MFEM_ABORT("TMOP_Metric_002 is implemented here only in 2D"); }

            mfem::MultiVector tmop_x { current_coords, winv_q, detW_q };
            mfem::MultiVector tmop_y { tmop_q };

            tmop_dop.Mult(tmop_x, tmop_y);
            dfem_mesh_quality = tmop_q.Sum();
            MPI_Allreduce(MPI_IN_PLACE, &dfem_mesh_quality, 1,
                          MPITypeMap<real_t>::mpi_type, MPI_SUM,
                          pmesh.GetComm());

            Vector true_dMeshQdx_dfem(coord_fes.GetTrueVSize());
            true_dMeshQdx_dfem = 0.0;
            tmop_dop.GetDerivative(coordsT, tmop_x)->Assemble(true_dMeshQdx_dfem);

            ParLinearForm dMeshQdx_dfem(&coord_fes);
            dMeshQdx_dfem = 0.0;
            coord_fes.GetRestrictionTransposeOperator()->Mult(true_dMeshQdx_dfem, dMeshQdxExpl);
         }

         double val = weight_1 * ObjVal+ weight_tmop * dfem_mesh_quality;

         elsolver_morph.ASolve( dfem_dQdu );
         mfem::ParGridFunction adj_sol_dfem = elsolver_morph.GetADisplacements();

         ParLinearForm dQdxImpl_dfem(&coord_fes); dQdxImpl_dfem = 0.0;

         {
            static constexpr int uA = 0, adjA = 1, lambdaA = 2;
            static constexpr int muA = 3, coordsA = 4, qA = 5;

            Array<int> all_domain_attr;
            if (pmesh.attributes.Size() > 0)
            {
               all_domain_attr.SetSize(pmesh.attributes.Max());
               all_domain_attr = 1;
            }

            mfem::QuadratureSpace energy_qspace(pmesh, ir);
            mfem::VectorQuadratureSpace lambda_qspace(energy_qspace, 1);
            mfem::VectorQuadratureSpace mu_qspace(energy_qspace, 1);
            mfem::VectorQuadratureSpace q_qspace(energy_qspace, 1);

            CoefficientVector lambda_cv(lambda_SIMP_cf_morph, energy_qspace);
            CoefficientVector mu_cv(mu_SIMP_cf_morph, energy_qspace);
            QuadratureFunction energy_q(q_qspace);

            ParGridFunction current_coords(&coord_fes);
            Vector current_nodes;
            pmesh.GetNodes(current_nodes);
            current_coords = current_nodes;

            mfem::future::DifferentiableOperator energy_dop(
               std::vector<mfem::future::FieldDescriptor>
            {
               { uA, &state_fes },
               { adjA, &state_fes },
               { lambdaA, &lambda_qspace },
               { muA, &mu_qspace },
               { coordsA, &coord_fes }
            },
            std::vector<mfem::future::FieldDescriptor>
            {
               { qA, &q_qspace }
            },
            pmesh);

            const auto energy_inputs = mfem::future::tuple
            {
               mfem::future::Gradient<uA>{},
               mfem::future::Gradient<adjA>{},
               mfem::future::Identity<lambdaA>{},
               mfem::future::Identity<muA>{},
               mfem::future::Gradient<coordsA>{},
               mfem::future::Weight{}
            };
            const auto energy_outputs = mfem::future::tuple
            {
               mfem::future::FunctionalValue<qA>{}
            };
            const auto energy_derivatives =
               std::integer_sequence<size_t, coordsA>{};

            if (2 == DIM)
            {
               AdjointElasticEnergyQF<2> qf;
               energy_dop.AddDomainIntegrator<mfem::future::LocalQFBackend>(
                  qf, energy_inputs, energy_outputs, ir, all_domain_attr,
                  energy_derivatives);
            }
            else if (3 == DIM)
            {
               AdjointElasticEnergyQF<3> qf;
               energy_dop.AddDomainIntegrator<mfem::future::LocalQFBackend>(
                  qf, energy_inputs, energy_outputs, ir, all_domain_attr,
                  energy_derivatives);
            }
            else { MFEM_ABORT("Space dimension not supported"); }

            mfem::MultiVector energy_x
            {
               u_morph, adj_sol_dfem, lambda_cv, mu_cv, current_coords
            };
            mfem::MultiVector energy_y { energy_q };

            energy_dop.Mult(energy_x, energy_y);

            Vector dA_dx(coord_fes.GetTrueVSize());
            dA_dx = 0.0;
            energy_dop.GetDerivative(coordsA, energy_x)->Assemble(dA_dx);

            dQdxImpl_dfem = 0.0;
            dQdxImpl_dfem.Add(-1.0, dA_dx);
         }


         if(i==60){
            mfem::mfem_error("aaaaaaaaaaaaaaaaaaaaaa");
         }

         ParLinearForm dQdx_filtered(&coord_fes); dQdx_filtered = 0.0;

         dQdx_filtered.Add(weight_1, dfem_dQdx);
         dQdx_filtered.Add(weight_1, dQdxImpl_dfem);
         dQdx_filtered.Add(weight_tmop, dMeshQdxExpl);

         paraview_dc_morph.SetCycle(i+1);
         paraview_dc_morph.SetTime(i+1);
         paraview_dc_morph.Save();

         MFEM_VERIFY( !static_cast<bool>(dQdx_filtered.CheckFinite()), "dQdx before filter is NAN.");

         filterSolver.ASolve(dQdx_filtered, true);
         ParLinearForm * dQdxImplfilter = filterSolver.GetImplicitDqDx();

         ParLinearForm dQdx(&coord_fes); dQdx = 0.0;
         dQdx.Add(1.0, *dQdxImplfilter);

         HypreParVector *truedQdx = dQdx.ParallelAssemble();
         objgrad = *truedQdx;
         objgrad *= 1e-0;

      double epsilon = 1e-8;

      if(dQduFD)
      {
      }

      if(dQdxFD)
      {
      }

      if(dQdxFD_global)
      {     
      }

      if( BreakAfterFirstIt )
      {
        mfem_error("break before update");
      }
      //----------------------------------------------------------------------------------------------------------
      Vector & trueBounds = gridfuncBoundIndicator.GetTrueVector();
      Vector & trueBounds_Max = gridfuncBoundfunc_Max.GetTrueVector();
      Vector & trueBounds_Min = gridfuncBoundfunc_Min.GetTrueVector();

      // impose desing variable bounds - set xxmin and xxmax
      xxmin=trueOptvar; xxmin-=max_ch;
      xxmax=trueOptvar; xxmax+=max_ch;
      for(int li=0;li<xxmin.Size();li++){
         if(xxmin[li] <= trueBounds_Min[li])
         {
            xxmin[li] = trueBounds_Min[li];
         }
         if(xxmax[li] >= trueBounds_Max[li])
         {
            xxmax[li] = trueBounds_Max[li];
         }

        if( trueBounds[li] ==1.0)
        {
          xxmin[li] = -1e-10;
          xxmax[li] =  1e-10;
        }
      }

      double localGradNormSquared = std::pow(objgrad.Norml2(), 2);
      double globGradNorm;
#ifdef MFEM_USE_MPI
      MPI_Allreduce(&localGradNormSquared, &globGradNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      globGradNorm = std::sqrt(globGradNorm);

      std::cout<<"grad norm = "<<globGradNorm<<" obj val: "<<ObjVal <<" | meshQualityVal: "<<dfem_mesh_quality<<" | totalObj: "<<val<<std::endl;


      mfem::Vector conDummy(1);  conDummy= -0.1;
      mma->Update( objgrad, conDummy, volgrad, xxmin,xxmax, trueOptvar);

      gridfuncOptVar.SetFromTrueVector();

      }
//       isConverged = false;

   }

   return EXIT_SUCCESS;
}
