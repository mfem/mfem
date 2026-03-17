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
        return 1.0;
        //return 1.0 + 100* sin((x[0]-0.5)*M_PI/2.0) * sin((x[1]-0.2)*M_PI/0.6);
      }
   return 1.0;
   }

///////////////////////////////////////////////////////////////////////////////
/// \brief The QFunction struct defining the linear elasticity operator at
/// integration points which is valid in 2D and 3D
template <int DIM> struct QFunction
{
   using matd_t = mfem::future::tensor<real_t, DIM, DIM>;
   using vecd_t = mfem::future::tensor<real_t, DIM>;

   struct Elasticity
   {
      MFEM_HOST_DEVICE inline auto operator()(const matd_t &dudxi,
                                              const real_t &L, const real_t &M,
                                              const matd_t &J,
                                              const real_t &w) const
      {
         const matd_t JxW = transpose(inv(J)) * det(J) * w;
         constexpr auto I = mfem::future::IsotropicIdentity<DIM>();
         const auto eps = mfem::future::sym(dudxi * mfem::future::inv(J));
         return mfem::future::tuple{(L * tr(eps) * I + 2.0 * M * eps) * JxW};
      }
   };

   struct Elasticity_dDdrho
   {
      MFEM_HOST_DEVICE inline auto operator()(const matd_t &dudxi, const vecd_t &u, const real_t &rho,
                                              const real_t &L, const real_t &M,
                                              const matd_t &J,
                                              const real_t &w) const
      {
         constexpr real_t exponent = 3.0;
         constexpr real_t rho_min = 1e-6;

         const auto val = -exponent * pow(rho, exponent-1.0) * (1-rho_min);
 
         const matd_t JxW = transpose(inv(J)) * det(J) * w;
         constexpr auto I = mfem::future::IsotropicIdentity<DIM>();
         const auto eps = mfem::future::sym(dudxi * mfem::future::inv(J));
         const auto scalarVal = u*(L * tr(eps) * I + 2.0 * M * eps);



         return mfem::future::tuple{ val* scalarVal * JxW};
      }
   };

      struct Elasticity_dDdrho1
   {
      MFEM_HOST_DEVICE inline auto operator()(const matd_t &dudxi, const matd_t &dadjdxi, const real_t &rho,
                                              const real_t &L, const real_t &M,
                                              const matd_t &J,
                                              const real_t &w) const
      {
         constexpr real_t exponent = 3.0;
         constexpr real_t rho_min = 1e-6;

         const auto JxW =  det(J) * w;
         const auto Jinv =  mfem::future::inv(J);
         constexpr auto I = mfem::future::IsotropicIdentity<DIM>();
         const auto eps = mfem::future::sym(dudxi * Jinv);
         const auto dadjdx = dudxi ;
         // const auto scalarVal =L * tr(eps) * tr(dadjdxi);
         // const auto scalarVal1 = mfem::future::inner(mfem::future::transpose(dadjdx),(2.0 * M * eps));
         const auto density = mfem::future::inner(mfem::future::transpose(dadjdx),(L * tr(eps) * I + 2.0 * M * eps));
         const auto val = -exponent * mfem::future::pow(rho, exponent-1.0) * (1-rho_min);

         return mfem::future::tuple{ val * density* JxW} ;
      }
   };



   //    real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   // {
   //    real_t L = lambda->Eval(T, ip);
   //    real_t M = mu->Eval(T, ip);
   //    u->GetVectorGradient(T, grad);
   //    real_t div_u = grad.Trace();
   //    real_t density = L*div_u*div_u;
   //    int dim = T.GetSpaceDim();
   //    for (int i=0; i<dim; i++)
   //    {
   //       for (int j=0; j<dim; j++)
   //       {
   //          density += M*grad(i,j)*(grad(i,j)+grad(j,i));
   //       }
   //    }
   //    real_t val = rho_filter->GetValue(T,ip);

   //    return -exponent * pow(val, exponent-1.0) * (1-rho_min) * density;
   // }

   // struct Elasticity_dDdrho
   // {
   //    MFEM_HOST_DEVICE inline auto operator()(const matd_t &dudxi, const real_t &rho,
   //                                            const real_t &L, const real_t &M,
   //                                            const matd_t &J,
   //                                            const real_t &w) const
   //    {
   //       constexpr real_t exponent = 3.0;
   //       constexpr real_t rho_min = 1e-9;
   //       const matd_t JxW = transpose(inv(J)) * det(J) * w;
   //       const auto eps = mfem::future::sym(dudxi * mfem::future::inv(J));   //dudx
   //       const auto val = -exponent * pow(rho, exponent-1.0) * (1-rho_min);
   //       return tuple{(L * tr(eps) * tr(eps) + 2.0 * M * eps * dudxi) * JxW * val};
   //    }
   // };
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

//(-r'(ρ̃) ( λ |∇⋅u|² + 2 μ |ε(u)|²),v)
// template <int DIM> struct QFunction_dDdrho
// {
//    using matd_t = mfem::future::tensor<real_t, DIM, DIM>;

//    struct Elasticity_dDdrho
//    {
//       MFEM_HOST_DEVICE inline auto operator()(const matd_t &dudxi, const real_t &rho,
//                                               const real_t &L, const real_t &M,
//                                               const matd_t &J,
//                                               const real_t &w) const
//       {
//          constexpr real_t exponent = 3.0;
//          constexpr real_t rho_min = 1e-9;
//          const matd_t JxW = transpose(inv(J)) * det(J) * w;
//          const auto eps = mfem::future::sym(dudxi * mfem::future::inv(J));
//          const auto val = -exponent * pow(rho, exponent-1.0) * (1-rho_min);
//          return tuple{(L * tr(eps) * tr(eps) + M * eps * dudxi) * JxW * val};
//       }
//    };
// };

   //  auto lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));  // Lame's first parameter
   //  auto mu = E / (2.0 * (1.0 + nu));                        // Lame's second parameter
   //  const auto I = ::smith::Identity<dim>();
   //  auto strain = ::smith::sym(du_dX);                            // small strain tensor
   //  return lambda * ::smith::tr(strain) * I + 2.0 * mu * strain;  // Cauchy stress

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
   bool pa = false;
   bool dfem = false;
   bool mesh_tri = false;
   bool mesh_quad = false;
   int par_ref_levels = 0;
   bool paraview = true;
   bool visualization = true;
   real_t epsilon = 0.02;
   real_t alpha = 1.0;
   real_t vol_fraction = 0.5;
   real_t rho_min = 1e-6;
   real_t lambda = 0.5769230769;
   real_t mu = 1.0/2.6;

   int neumannBCIndex = 3;  // 1 based
   double neumannLoad = -1.0e-0;

   double weight_1 = -1e01;
   double weight_tmop = 1e-2;

   double filterRadius = 0.4;

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

   Mesh mesh = Mesh::MakeCartesian2D(3, 1, mfem::Element::Type::QUADRILATERAL,
                                     true, 3.0, 1.0);
   const int dim = mesh.Dimension();
   constexpr int DIM = 2;

   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement. We choose
   // 'ref_levels' to be the largest number that gives a final mesh with no
   // more than 1000 elements.
   {
      const int ref_levels = 4;
         //(int)floor(log(1000. / mesh.GetNE()) / log(2.) / dim);
      for (int l = 0; l < ref_levels; l++) { mesh.UniformRefinement(); }
   }
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
   // mfem::ParGridFunction * nodes((pmesh.EnsureNodes(),
   //        static_cast<mfem::ParGridFunction *>(pmesh.GetNodes())));
   // mfem::ParFiniteElementSpace *nodes_fes(nodes->ParFESpace());

   //nodes->Print();


   H1_FECollection fec(order, dim);
   ParFiniteElementSpace state_fes(&pmesh, &fec,dim);
   ParFiniteElementSpace coord_fes(&pmesh, &fec,dim);
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
   paraview_dc.SetPrefixPath("ParaView");
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
  double max_ch=0.1;

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

      // set boundary conditions
      elsolver.AddDispBC(4, -1, 0.0);

      VectorArrayCoefficient tractionLoad(dim);
      tractionLoad.Set(0, new ConstantCoefficient( 0.0));
      tractionLoad.Set(1, new ConstantCoefficient(neumannLoad));

      // set surface load
      elsolver.AddSurfLoad(neumannBCIndex, 0.0, neumannLoad);

      // set convergence tolerances and max iterations
      elsolver.SetLinearSolver(1e-6,1e-8,100);

      int numTOit = 0;
      for( int ik = 0; ik < numTOit; ik++)
      {
         if (ik > 1) { alpha *= ((real_t) ik) / ((real_t) ik-1); }

         FilterSolver->SetRHSCoefficient(&rho);
         FilterSolver->Solve();
         rho_filter = *FilterSolver->GetFEMSolution();

                  std::cout<<"filter solve done"<<std::endl;

         SIMPInterpolationCoefficient SIMP_cf(&rho_filter,rho_min, 1.0, 3.0);
         ProductCoefficient lambda_SIMP_cf(lambda_cf,SIMP_cf);
         ProductCoefficient mu_SIMP_cf(mu_cf,SIMP_cf);

         // set material properties
         elsolver.SetMaterialLame(lambda_SIMP_cf, mu_SIMP_cf);

         // assemble the discrete system
         elsolver.Assemble();

         // solve the system
         elsolver.FSolve();

         // extract the solution
        elsolver.GetSol(u);

         // Solve (ϵ² ∇ w̃, ∇ v) + (w̃ ,v) = (-r'(ρ̃) ( λ |∇⋅u|² + 2 μ |ε(u)|²),v)

         StrainEnergyDensityCoefficient rhs_cf(&lambda_cf,&mu_cf,&u, &rho_filter,
                                           rho_min);

         Array<int> all_domain_attr;
         if (pmesh.attributes.Size() > 0)
         {
            all_domain_attr.SetSize(pmesh.attributes.Max());
            all_domain_attr = 1;
         }

         IsoLinElasticSolver::NqptUniformParameterSpace Lambda_ps(pmesh, ir, 1);
         IsoLinElasticSolver::NqptUniformParameterSpace Mu_ps(pmesh, ir, 1);
         mfem::QuadratureSpace qs(pmesh, ir);

         // sample lambda on the integration points
         CoefficientVector Lambda_cv(lambda_cf, qs);
         CoefficientVector Mu_cv(mu_cf, qs);

         static constexpr int uN = 0, adjN = 2, rhoN = 3, coordsN = 1, lambdaN = 4, muN = 5;
         std::unique_ptr<mfem::future::DifferentiableOperator> dopdRds = std::make_unique<mfem::future::DifferentiableOperator>(
         std::vector<mfem::future::FieldDescriptor> {{ uN, &state_fes }},
         std::vector<mfem::future::FieldDescriptor> {{adjN, &state_fes}, { rhoN, &filter_fes}, { lambdaN, &Lambda_ps}, { muN, &Mu_ps}, { coordsN, &coord_fes } },
         pmesh);

         const auto inputs = mfem::future::tuple{ mfem::future::Gradient<uN>{},
                                                  mfem::future::Gradient<adjN>{},
                                                  mfem::future::Value<rhoN>{},
                                                  mfem::future::Identity<lambdaN>{},
                                                  mfem::future::Identity<muN>{},
                                                  mfem::future::Gradient<coordsN>{},
                                                  mfem::future::Weight{} };
         const auto output = mfem::future::tuple{ mfem::future::Value<rhoN>{} };

         std::cout<<"running qFunc"<<std::endl;

         if (2 == DIM)
         {
            typename QFunction<2>::Elasticity_dDdrho1 e2qf;
            dopdRds->AddDomainIntegrator(e2qf, inputs, output, ir, all_domain_attr);           
         }
         else if (3 == DIM)
         {
             typename QFunction<3>::Elasticity_dDdrho e3qf;
            dopdRds->AddDomainIntegrator(e3qf, inputs, output, ir, all_domain_attr);
         }
         mfem::ParGridFunction adjTimesdRdrho(&filter_fes); adjTimesdRdrho = 0.0;
         dopdRds->SetParameters({ &u, &rho_filter, &Lambda_cv, &Mu_cv, &coords_ });
         dopdRds->Mult( u, adjTimesdRdrho);

         //adjTimesdRdrho.Print();
                 // u.Print();

         mfem::GridFunctionCoefficient rhs_cf1(&adjTimesdRdrho);


        // std::cout<<"========================================================================"<<std::endl;
         // mfem::ParGridFunction outputAA(&filter_fes);
         // outputAA.ProjectCoefficient(rhs_cf);
         // outputAA.Print();




         FilterSolver->SetRHSCoefficient(&rhs_cf);
         FilterSolver->Solve();
         w_filter = *FilterSolver->GetFEMSolution();

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

      // real_t compliance = (*(ElasticitySolver->GetLinearForm()))(u);
      // MPI_Allreduce(MPI_IN_PLACE, &compliance, 1, MPITypeMap<real_t>::mpi_type,
      //               MPI_SUM, MPI_COMM_WORLD);

        // mfem::mfem_error("end of forward solve");

         //====================== QoI ==================================

//          ParGridFunction one(&fes); one = 1.0;
//          Vector one_trueV(fes.GetTrueVSize()); one_trueV = 1.0;
//          one.SetFromTrueDofs(one_trueV);

//          static constexpr int U = 0, Coords = 1;
//          const auto vol_in = std::vector{ mfem::future::FieldDescriptor{ U, &fes } };
//          mfem::future::DifferentiableOperator dop_volume(vol_in, {{Coords, nodes->ParFESpace()}}, pmesh);


//          Array<int> all_domain_attr;
//          if (pmesh.attributes.Size() > 0)
//          {
//             all_domain_attr.SetSize(pmesh.attributes.Max());
//             all_domain_attr = 1;
//          }

//          auto derivatives = std::integer_sequence<size_t, U> {};
//          dop_volume.AddDomainIntegrator(vol_functional_qf,
//                            tuple{ mfem::future::Value<U>{}, mfem::future::Gradient<Coords>{}, mfem::future::Weight{} },
//                            tuple{ mfem::future::Sum<U>{} },
//                            *ir, all_domain_attr, derivatives);
//          dop_volume.SetParameters({ nodes });

//          fes.GetRestrictionMatrix()->Mult(one, one_trueV);
//          mfem::Vector sum(1);
//          dop_volume.Mult(one_trueV, sum);

         // ========= adjoint solve ===================

         // perfom adjoint solver
         // elsolver.ASolve( dQdu );

         // // extract the adjoint solution
         // ParGridFunction &adj = elsolver.GetADisplacements();


         // ========= postmultiplication ===============


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

      QuantityOfInterest QoIEvaluator(&pmesh, QoIType::STRUC_COMPLIANCE, order, 
                              order, neumannBdr, dim);
      QoIEvaluator.setTractionCoeff(&tractionLoad);

      TMOP_QualityMetric *metric = new TMOP_Metric_001;
      TargetConstructor *target_c2 = new TargetConstructor(TargetConstructor::IDEAL_SHAPE_UNIT_SIZE, MPI_COMM_WORLD);
      NodeAwareTMOPQuality MeshQualityEvaluator(&pmesh, order, metric, target_c2);

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

      ParGridFunction sensitivity_GF(&coord_fes); sensitivity_GF = 0.0;

      MMA* mma=new MMA(MPI_COMM_WORLD, trueOptvar.Size(), 0, trueOptvar);

      ParaViewDataCollection paraview_dc_morph("isoel_morph", &pmesh);
      paraview_dc_morph.SetPrefixPath("ParaView");
      paraview_dc_morph.SetLevelsOfDetail(order);
      paraview_dc_morph.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_morph.SetHighOrderOutput(true);
      paraview_dc_morph.SetCycle(0);
      paraview_dc_morph.SetTime(0.0);
      paraview_dc_morph.RegisterField("disp", &u_morph);
      paraview_dc_morph.RegisterField("mesh_disp", &mesh_disp);
      paraview_dc_morph.RegisterField("design", &test_design_filter);
      paraview_dc_morph.RegisterQField("designQuadrature", &dens_interp);
      paraview_dc_morph.RegisterField("sensitivity", &sensitivity_GF);

      paraview_dc_morph.Save();

      ParGridFunction gridfuncBoundIndicator(&coord_fes);
      gridfuncBoundIndicator = 0.0;
      Array<int> vdofs;

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
    gridfuncBoundIndicator.SetTrueVector();

    std::vector<std::pair<int, double>> essentialBC(pmesh.bdr_attributes.Max());
    essentialBC.resize(1);   essentialBC[0] = {4, 0};
    Elasticity_Solver * elasticitysolver_old = new Elasticity_Solver(&pmesh, essentialBC, neumannBdr, order);
    elasticitysolver_old->SetLoad(&tractionLoad);


      for(int i=1;i<40;i++)
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
         finder.Interpolate(pos_quad_final, test_design_filter, dens_interp);      //TODO filtered density
         //finder.Interpolate(pos_quad_final, rho_filter, dens_interp);      //TODO filtered density

         // ----------------------- compute and set material 


         
         // update mesh coordinates

         pmesh.SetNodes(Xi);
         pmesh.DeleteGeometricFactors();

         QuadratureFunctionCoefficient newDensityCoeff(dens_interp);         
         //SIMPInterpolationCoefficientUsingCoeff SIMP_cf_morph(&newDensityCoeff,rho_min, 1.0, 1.0);
         GridFunctionCoefficient tGFCoeff(&test_design_filter);
         SIMPInterpolationCoefficientUsingCoeff SIMP_cf_morph(&desing_func,rho_min, 1.0, 1.0);
         ProductCoefficient lambda_SIMP_cf_morph(lambda_cf,SIMP_cf_morph);
         ProductCoefficient mu_SIMP_cf_morph(mu_cf,SIMP_cf_morph);



         // set material properties
         elsolver.SetMaterialLame(lambda_SIMP_cf_morph, mu_SIMP_cf_morph);

         // assemble the discrete system
         elsolver.Assemble();

         // solve the system
         elsolver.FSolve();

//==============================
            elasticitysolver_old->setMaterial( &lambda_SIMP_cf_morph, &mu_SIMP_cf_morph);
            elasticitysolver_old->SetDesign( filteredNodePos );
            elasticitysolver_old->FSolve();
            ParGridFunction & u_morphold = elasticitysolver_old->GetSolution();
   


         std::cout<<"f solve"<<std::endl;

         // extract the solution
         u_morph = 0.0;
         elsolver.GetSol(u_morph);



         MeshQualityEvaluator.SetDesign( filteredNodePos );
         QoIEvaluator.SetDesign( filteredNodePos );
         QoIEvaluator.SetDiscreteSol( u_morphold );                       //fix
         //QoIEvaluator.SetIntegrationRules(&IntRulesLo, quad_order);

         double ObjVal = QoIEvaluator.EvalQoI();
         double meshQualityVal = MeshQualityEvaluator.EvalQoI();


         double val = weight_1 * ObjVal+ weight_tmop * meshQualityVal;

         QoIEvaluator.EvalQoIGrad();
         MeshQualityEvaluator.EvalQoIGrad();

         ParLinearForm * dQdu = QoIEvaluator.GetDQDu();
         ParLinearForm * dQdxExpl = QoIEvaluator.GetDQDx();
         ParLinearForm * dMeshQdxExpl = MeshQualityEvaluator.GetDQDx();

         elsolver.ASolve( *dQdu );

//====================================
            elasticitysolver_old->ASolve( *dQdu );
            ParLinearForm * dQdxImplold = elasticitysolver_old->GetImplicitDqDx();


         std::cout<<"a solve"<<std::endl;

         mfem::ParGridFunction & adj_sol = elsolver.GetADisplacements();

         // get adjoint

                  std::cout<<"+++++++++++++++++++++++++++++"<<std::endl;
                  //u_morph.Print();

         const IntegrationRule &ir = constrolQuadSpace.GetIntRule(0);
         ParLinearForm LHS_sensitivity(&coord_fes);
         LinearFormIntegrator *lfi = new ElasticityStiffnessShapeSensitivityIntegrator(
                                            lambda_SIMP_cf_morph, mu_SIMP_cf_morph, u_morph, adj_sol);
         lfi->SetIntRule(&ir);
         LHS_sensitivity.AddDomainIntegrator(lfi);

         LHS_sensitivity.Assemble();
         //LHS_sensitivity.Print();

         MFEM_VERIFY( !static_cast<bool>(LHS_sensitivity.CheckFinite()), "LHS_sensitivity before filter is NAN.");

         // ParLinearForm RHS_sensitivity(coord_fes);
         // RHS_sensitivity.AddBoundaryIntegrator(new ElasticityTractionShapeSensitivityIntegrator(*QCoef_, adj_sol, 12,12), bdr);
         // RHS_sensitivity.Assemble();

         ParLinearForm dQdxImpl(&coord_fes); dQdxImpl = 0.0;
         //dQdxImpl.Add(-1.0, LHS_sensitivity);
         dQdxImpl.Add(1.0, *dQdxImplold);
         //dQdx_->Add( 1.0, RHS_sensitivity);


//          ParLinearForm dQdx_physics(pfespace); dQdx_physics = 0.0;
         ParLinearForm dQdx_filtered(&coord_fes); dQdx_filtered = 0.0;
         // dQdx_physics.Add(1.0, *dQdxExpl);
         // dQdx_physics.Add(1.0, *dQdxImpl);
         MFEM_VERIFY( !static_cast<bool>(dQdxExpl->CheckFinite()), "dQdxExpl before filter is NAN.");
         MFEM_VERIFY( !static_cast<bool>(dQdxImpl.CheckFinite()), "dQdxImpl before filter is NAN.");

         HypreParVector *dQdxExpl_H = dQdxExpl->ParallelAssemble();
         HypreParVector *dQdxImpl_H = dQdxImpl.ParallelAssemble();
         sensitivity_GF = 0.0;
         sensitivity_GF += *dQdxExpl_H;
         sensitivity_GF += *dQdxImpl_H;
         //dQdxImpl.Print();

         dQdx_filtered.Add(weight_1, *dQdxExpl);
         dQdx_filtered.Add(weight_1, dQdxImpl);
         dQdx_filtered.Add(weight_tmop, *dMeshQdxExpl);

         paraview_dc_morph.SetCycle(i+1);
         paraview_dc_morph.SetTime(i+1);
         paraview_dc_morph.Save();

         MFEM_VERIFY( !static_cast<bool>(dQdx_filtered.CheckFinite()), "dQdx before filter is NAN.");


         filterSolver.ASolve(dQdx_filtered, true);
                           std::cout<<"a filter"<<std::endl;
         ParLinearForm * dQdxImplfilter = filterSolver.GetImplicitDqDx();

             //dQdxImplfilter->Print();


         ParLinearForm dQdx(&coord_fes); dQdx = 0.0;
         dQdx.Add(1.0, *dQdxImplfilter);
         //dQdx.Add(1.0, dQdx_filtered);

         HypreParVector *truedQdx = dQdx.ParallelAssemble();

         objgrad = *truedQdx;

         objgrad *= 1e-0;

          //objgrad.Print();







































      double epsilon = 1e-8;



       if(dQduFD)
      {
         QuantityOfInterest QoIEvaluator_FD1(&pmesh, QoIType::STRUC_COMPLIANCE, order, order, neumannBdr, dim);
         QuantityOfInterest QoIEvaluator_FD2(&pmesh, QoIType::STRUC_COMPLIANCE, order, order, neumannBdr, dim);
         QoIEvaluator_FD1.setTractionCoeff(&tractionLoad);
         QoIEvaluator_FD2.setTractionCoeff(&tractionLoad);

        ParGridFunction tFD_sens(&state_fes); tFD_sens = 0.0;
        for( int Ia = 0; Ia<u_morphold.Size(); Ia++)
        {
         //  if (myid == 0)
         //  {
           // std::cout<<"iter: "<< Ia<< " out of: "<<u_morphold.Size() <<std::endl;
         // }
          u_morphold[Ia] +=epsilon;

          QoIEvaluator_FD1.SetDesign( filteredNodePos );
          QoIEvaluator_FD1.SetDiscreteSol( u_morphold );
          QoIEvaluator_FD1.SetNodes(x0);
          //QoIEvaluator_FD1.SetIntegrationRules(&IntRulesLo, quad_order);

          double ObjVal_FD1 = QoIEvaluator_FD1.EvalQoI();

          u_morphold[Ia] -=2.0*epsilon;

          QoIEvaluator_FD2.SetDesign( filteredNodePos );
          QoIEvaluator_FD2.SetDiscreteSol( u_morphold );
          QoIEvaluator_FD2.SetNodes(x0);
          //QoIEvaluator_FD2.SetIntegrationRules(&IntRulesLo, quad_order);

          double ObjVal_FD2 = QoIEvaluator_FD2.EvalQoI();

          u_morphold[Ia] +=epsilon;

          tFD_sens[Ia] = (ObjVal_FD1-ObjVal_FD2)/(2.0*epsilon);
        }
        dQdu->Print();
        std::cout<<"  ----------  FD Diff ------------"<<std::endl;
        tFD_sens.Print();

        std::cout<<"  ---------- dQdu Analytic - FD Diff ------------"<<std::endl;
        ParGridFunction tFD_diff(&state_fes); tFD_diff = 0.0;
        tFD_diff = *dQdu;
        tFD_diff -=tFD_sens;
        //tFD_diff.Print();
        std::cout<<"norm: "<<tFD_diff.Norml2()<<std::endl;
      }

      if(dQdxFD)
      {
        // nodes are p
        // det(J) is order d*p-1

        ParGridFunction tFD_sens(&coord_fes); tFD_sens = 0.0;
      //   Array<double> GLLVec;
      //   int nqpts;
      //   {
      //     const IntegrationRule *ir = &IntRulesLo.Get(Geometry::SQUARE, 8);
      //     nqpts = ir->GetNPoints();
      //     // std::cout << nqpts << " k10c\n";
      //     for (int e = 0; e < PMesh->GetNE(); e++)
      //     {
      //       ElementTransformation *T = PMesh->GetElementTransformation(e);
      //       for (int q = 0; q < ir->GetNPoints(); q++)
      //       {
      //         const IntegrationPoint &ip = ir->IntPoint(q);
      //         T->SetIntPoint(&ip);
      //         double disc_val = discreteSol.GetValue(e, ip);
      //         double exact_val = trueSolution->Eval( *T, ip );
      //         GLLVec.Append(disc_val-exact_val);
      //       }
      //     }
      //   }
      //   std::cout << nqpts << " " << GLLVec.Size() << " k10c\n";
        // MFEM_ABORT(" ");

        for( int Ia = 0; Ia<filteredNodePos.Size(); Ia++)
        {
          if(gridfuncBoundIndicator[Ia] == 1.0)
          {
            (*dQdxExpl)[Ia] = 0.0;

            continue;
          }

          //std::cout<<"iter: "<< Ia<< " out of: "<<gridfuncOptVar.Size() <<std::endl;
          double fac = 1.0-gridfuncBoundIndicator[Ia];
          filteredNodePos[Ia] +=(fac)*epsilon;

          QuantityOfInterest QoIEvaluator_FD1(&pmesh, QoIType::STRUC_COMPLIANCE, order, order, neumannBdr, dim);
          QoIEvaluator_FD1.setTractionCoeff(&tractionLoad);
          QoIEvaluator_FD1.SetDesign( filteredNodePos );
          QoIEvaluator_FD1.SetDiscreteSol( u_morphold );
          QoIEvaluator_FD1.SetNodes(x0);

          double ObjVal_FD1 = QoIEvaluator_FD1.EvalQoI();

          filteredNodePos[Ia] -=(fac)*2.0*epsilon;

          QuantityOfInterest QoIEvaluator_FD2(&pmesh, QoIType::STRUC_COMPLIANCE, order, order, neumannBdr, dim);
          QoIEvaluator_FD2.setTractionCoeff(&tractionLoad);
          QoIEvaluator_FD2.SetDesign( filteredNodePos );
          QoIEvaluator_FD2.SetDiscreteSol( u_morphold );
          QoIEvaluator_FD2.SetNodes(x0);

          double ObjVal_FD2 = QoIEvaluator_FD2.EvalQoI();

          filteredNodePos[Ia] +=(fac)*epsilon;

          tFD_sens[Ia] = (ObjVal_FD1-ObjVal_FD2)/(2.0*epsilon);
        }

        dQdxExpl->Print();
        std::cout<<"  ----------  FD Diff ------------"<<std::endl;
        tFD_sens.Print();

        std::cout<<"  ---------- dQdx Analytic - FD Diff ------------"<<std::endl;
        ParGridFunction tFD_diff(&coord_fes); tFD_diff = 0.0;
        tFD_diff = *dQdxExpl;
        tFD_diff -=tFD_sens;
        tFD_diff.Print();
        std::cout<<"norm: "<<tFD_diff.Norml2()<<std::endl;
        for( int Ia = 0; Ia<filteredNodePos.Size(); Ia++)
        {
          tFD_diff[Ia] *= (1.0-gridfuncBoundIndicator[Ia]);
        }
        // tFD_diff.Print();
        std::cout<<"norm: "<<tFD_diff.Norml2()<<std::endl;
      }

      if(dQdxFD_global)
      {
      //   ParGridFunction tFD_sens(&coord_fes); tFD_sens = 0.0;
      //   for( int Ia = 0; Ia<filteredNodePos.Size(); Ia++)
      //   {
      //     if(gridfuncBoundIndicator[Ia] == 1.0)
      //     {
      //       dQdx_physics[Ia] = 0.0;
      //       dQdx[Ia] = 0.0;

      //       continue;
      //     }
      //     std::cout<<"iter: "<< Ia<< " out of: "<<gridfuncOptVar.Size() <<std::endl;
      //     double fac = 1.0-gridfuncBoundIndicator[Ia];
      //     gridfuncOptVar[Ia] +=fac*epsilon;

      //     IsoLinElasticSolver elsolver_1(&pmesh, order, pa, dfem);
      //     elsolver_1.AddDispBC(4, -1, 0.0);
      //     elsolver_1.AddSurfLoad(neumannBCIndex, 0.0, neumannLoad);
      //     elsolver_1.SetLinearSolver(1e-6,1e-8,100);

      //    Vector Xi = X0_;
      //    Xi += filteredNodePos;

      //    // ----------------------   gslib densities

      //    Vector pos_quad_final;
      //    GetQuadPointsPositions(pmesh, constrolQuadSpace, Xi, pos_quad_final);

      //    FindPointsGSLIB finder(pmesh.GetComm());
      //    finder.SetL2AvgType(FindPointsGSLIB::NONE);
      //    finder.Setup(pmesh);
      //    finder.Interpolate(pos_quad_final, test_design_filter, dens_interp);      //TODO filtered density

      //    QuadratureFunctionCoefficient newDensityCoeff(dens_interp);         
      //    SIMPInterpolationCoefficientUsingCoeff SIMP_cf_morph(&newDensityCoeff,rho_min, 1.0, 3.0);
      //    ProductCoefficient lambda_SIMP_cf_morph(lambda_cf,SIMP_cf_morph);
      //    ProductCoefficient mu_SIMP_cf_morph(mu_cf,SIMP_cf_morph);
         
      //    pmesh.SetNodes(Xi);
      //    pmesh.DeleteGeometricFactors();

      //    elsolver.SetMaterialLame(lambda_SIMP_cf_morph, mu_SIMP_cf_morph);
      //    elsolver.Assemble();
      //    elsolver.FSolve();



      //     solver_FD1.SetDesign( gridfuncOptVar );
      //     solver_FD1.FSolve();
      //     ParGridFunction & discreteSol_1 = solver_FD1.GetSolution();

      //     QoIEvaluator_FD1.setTrueSolCoeff(  trueSolution );
      //     if(qoiType == QoIType::ENERGY){QoIEvaluator_FD1.setTrueSolCoeff( QCoef );}
      //     QoIEvaluator_FD1.setTrueSolGradCoeff(trueSolutionGrad);
      //     QoIEvaluator_FD1.SetDesign( gridfuncOptVar );
      //     QoIEvaluator_FD1.SetDiscreteSol( discreteSol_1 );
      //     QoIEvaluator_FD1.SetNodes(x0);
      //     QoIEvaluator_FD1.SetIntegrationRules(&IntRulesLo, quad_order);

      //     double ObjVal_FD1 = QoIEvaluator_FD1.EvalQoI();

      //     gridfuncOptVar[Ia] -=fac*2.0*epsilon;

      //     solver_FD2.SetDesign( gridfuncOptVar );
      //     solver_FD2.FSolve();
      //     ParGridFunction & discreteSol_2 = solver_FD2.GetSolution();

      //     QoIEvaluator_FD2.setTrueSolCoeff(  trueSolution );
      //     if(qoiType == QoIType::ENERGY){QoIEvaluator_FD2.setTrueSolCoeff( QCoef );}
      //     QoIEvaluator_FD2.setTrueSolGradCoeff(trueSolutionGrad);
      //     QoIEvaluator_FD2.SetDesign( gridfuncOptVar );
      //     QoIEvaluator_FD2.SetDiscreteSol( discreteSol_2 );
      //     QoIEvaluator_FD2.SetNodes(x0);
      //     QoIEvaluator_FD2.SetIntegrationRules(&IntRulesLo, quad_order);

      //     double ObjVal_FD2 = QoIEvaluator_FD2.EvalQoI();

      //     gridfuncOptVar[Ia] +=fac*epsilon;

      //     tFD_sens[Ia] = (ObjVal_FD1-ObjVal_FD2)/(2.0*epsilon);
      //   }

      //   dQdx.Print();
      //   std::cout<<"  ----------  FD Diff - Global ------------"<<std::endl;
      //   tFD_sens.Print();

      //   std::cout<<"  ---------- dQdx Analytic - FD Diff ------------"<<std::endl;
      //   ParGridFunction tFD_diff(pfespace); tFD_diff = 0.0;
      //   tFD_diff = dQdx;
      //   tFD_diff -=tFD_sens;
      //   tFD_diff.Print();
      //   std::cout<<"norm: "<<tFD_diff.Norml2()<<std::endl;
      //   for( int Ia = 0; Ia<gridfuncOptVar.Size(); Ia++)
      //   {
      //     tFD_diff[Ia] *= (1.0-gridfuncLSBoundIndicator[Ia]);
      //   }
      //   // tFD_diff.Print();
      //   std::cout<<"norm: "<<tFD_diff.Norml2()<<std::endl;

      //   paraview_dc.SetCycle(i);
      //   paraview_dc.SetTime(i*1.0);
      //   //paraview_dc.RegisterField("ObjGrad",&objGradGF);
      //   paraview_dc.RegisterField("Solution",&x_gf);
      //   paraview_dc.RegisterField("SolutionD",&discreteSol   );
      //   paraview_dc.RegisterField("Sensitivity",&dQdx_physicsGF);
      //   paraview_dc.RegisterField("SensitivityFD",&tFD_sens);
      //   paraview_dc.RegisterField("SensitivityDiff",&tFD_diff);
      //   paraview_dc.RegisterField("SensitivityExpl",&dQdx_ExplGF);
      //   paraview_dc.RegisterField("SensitivityImpl",&dQdx_ImplGF);
      //   paraview_dc.Save();

      //   std::cout<<"expl: "<<dQdxExpl->Norml2()<<std::endl;
      //   std::cout<<"impl: "<<dQdxImpl->Norml2()<<std::endl;
      }

      if( BreakAfterFirstIt )
      {
        mfem_error("break before update");
      }
























//       //----------------------------------------------------------------------------------------------------------
//       gridfuncOptVar.SetTrueVector();
      Vector & trueBounds = gridfuncBoundIndicator.GetTrueVector();

      // impose desing variable bounds - set xxmin and xxmax
      xxmin=trueOptvar; xxmin-=max_ch;
      xxmax=trueOptvar; xxmax+=max_ch;
      for(int li=0;li<xxmin.Size();li++){
        if( trueBounds[li] ==1.0)
        {
          xxmin[li] = -1e-10;
          xxmax[li] =  1e-10;
        }
      }


      ParaViewDataCollection paraview_dc_morph1("isoel_bdr_mor11ph", &pmesh);
      paraview_dc_morph1.SetPrefixPath("ParaView");
      paraview_dc_morph1.SetLevelsOfDetail(1);
      paraview_dc_morph1.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_morph1.SetHighOrderOutput(true);
      paraview_dc_morph1.SetCycle(0);
      paraview_dc_morph1.SetTime(0.0);
      paraview_dc_morph1.RegisterField("bdr", &gridfuncBoundIndicator);
      paraview_dc_morph1.Save();

//       Vector Xi = x0;
//       Xi += filteredDesign;
//       PMesh->SetNodes(Xi);
//       PMesh->DeleteGeometricFactors();

      double localGradNormSquared = std::pow(objgrad.Norml2(), 2);
      double globGradNorm;
#ifdef MFEM_USE_MPI
      MPI_Allreduce(&localGradNormSquared, &globGradNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      globGradNorm = std::sqrt(globGradNorm);

      std::cout<<"grad norm = "<<globGradNorm<<" obj val: "<<ObjVal <<" | meshQualityVal: "<<meshQualityVal<<" | totalObj: "<<val<<std::endl;


      mfem:Vector conDummy(1);  conDummy= -0.1;
      mma->Update( objgrad, conDummy, volgrad, xxmin,xxmax, trueOptvar);

      //trueOptvar.Print();

      gridfuncOptVar.SetFromTrueVector();

      // mfem::mfem_error("aaa");



      }
//       isConverged = false;

   }

   return EXIT_SUCCESS;
}
