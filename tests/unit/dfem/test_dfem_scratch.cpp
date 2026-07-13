// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.

// Multi-kernel scratch regression for the weak residual
//
//   F(u)_i = int_Omega phi_i c u^3 dx,
//
// evaluated as the scratch chain s = u, s = s*u, y = c*s*u.  The directional
// derivative is
//
//   DF(u)[du]_i = int_Omega phi_i 3 c u^2 du dx.
//
// If the tangent stored in the qfunction scratch shadow is lost between the
// split scratch updates, the final product only sees the direct derivative of
// the last factor and produces int_Omega phi_i c u^2 du dx instead.  This test
// checks both the direct DerivativeAction path and the cached
// DerivativeSetup+DerivativeApply path against the factor-of-3 result.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Test for the dFEM global qfunction with split computation and scratch space. 

#include "../unit_tests.hpp"
#include "mfem.hpp"

#ifdef MFEM_USE_MPI

#include "../../../fem/dfem/doperator.hpp"
#include "../../../fem/dfem/backends/global_qf/prelude.hpp"      

#include "../../../linalg/tensor_arrays.hpp"

#include <type_traits>

using namespace std;
using namespace mfem;
using namespace mfem::future;

using dscalar_t = real_t;

///<--- Q-functions
constexpr int U = 1;
constexpr int Y = 2;
constexpr int COEF = 3;
constexpr int COORDINATES = 4;

// Scratch storage for global qfunctions. The bank supports two scratch kinds:
// - quadrature-point scratch: real_t buffers sized as NQ * components_per_qp,
// - global scratch: one tuple of qfunction-local temporaries, independent of
//   NQ, used for values such as flags, scalars, or small Vector workspaces.
template <typename... GlobalScratchTypes>
struct ScratchBank
{
    //=================================
    ///<--- Global scratch utilities.
    //=================================

    using GlobalScratchTuple = mfem::future::tuple<GlobalScratchTypes...>;

    template <typename T>
    static T MakeGlobalScratchShadow(const T &)
    {
        return T {};
    }

    static Vector MakeGlobalScratchShadow(const Vector &primal)
    {
        Vector shadow(primal.Size());
        shadow.UseDevice(true);
        shadow = 0.0;
        return shadow;
    }

    template <typename Tuple, size_t... Is>
    static auto MakeGlobalScratchShadowTuple(const Tuple &primal,
                                             std::index_sequence<Is...>)
    {
        return mfem::future::make_tuple(
            MakeGlobalScratchShadow(mfem::future::get<Is>(primal))...);
    }

    template <typename Tuple>
    static auto MakeGlobalScratchShadowTuple(const Tuple &primal)
    {
        return MakeGlobalScratchShadowTuple(
            primal, std::make_index_sequence<mfem::future::tuple_size<Tuple>::value>{});
    }
    
    
    //===========================
    ///<--- Scratch objects
    //===========================

    // Global scratch layout (not NQ sized, used for flags, scalars, or small Vector workspaces)
    mutable GlobalScratchTuple global;

    // Quadrature-point scratch layout (NQ x components_per_qp sized)
    int nq = 0;
    std::vector<int> components;
    std::vector<int> sizes;
    std::vector<std::shared_ptr<Vector>> owned;
    std::vector<real_t *> ptrs;


    //===========================
    ///<--- Setter methods
    //===========================

    void SetScratch(const int nq_, std::initializer_list<int> components_per_qp = {1})
    {
        SetScratch(nq_, std::vector<int>(components_per_qp));
    }

    void SetScratch(const int nq_, const std::vector<int> &components_per_qp)
    {
        nq = nq_;
        components.clear();
        sizes.clear();
        owned.clear();
        ptrs.clear();
        for (int component_count : components_per_qp)
        {
            AddScratch(component_count);
        }
    }

    // Optional add for additional scratch vectors after the initial SetScratch call
    void AddScratch(const int components_per_qp = 1)
    {
        MFEM_VERIFY(nq > 0, "SetScratch must be called before AddScratch");
        MFEM_VERIFY(components_per_qp > 0,
                    "scratch components per quadrature point must be positive");
        owned.push_back(std::make_shared<Vector>());
        Vector &scratch = *owned.back();
        const int size = components_per_qp * nq;
        scratch.SetSize(size); scratch.UseDevice(true); scratch = 0.0;
        components.push_back(components_per_qp);
        sizes.push_back(scratch.Size());
        ptrs.push_back(scratch.ReadWrite());
    }

    void SetGlobalScratch(const GlobalScratchTuple &global_)
    {
        global = global_;
    }

    template <int I>
    auto &Global() const
    {
        return mfem::future::get<I>(global);
    }

    void CloneScratchLayoutTo(ScratchBank &shadow) const
    {
        shadow.SetScratch(nq, components);
        shadow.SetGlobalScratch(MakeGlobalScratchShadowTuple(global));
    }

    real_t *operator[](const int i) const { return ptrs[i]; }
    int Size() const { return static_cast<int>(ptrs.size()); }
};

// Shared QF scratch base.
// This owns the scratch bank and the shadow cloning behavior.
template <typename... GlobalScratchTypes>
struct QFWithScratch
{
    using GlobalScratchTuple = mfem::future::tuple<GlobalScratchTypes...>;

    int nq = 0;
    ScratchBank<GlobalScratchTypes...> scratch;

    // Exposes to the user the internal scratchbank methods
    void SetScratch(const int nq_, std::initializer_list<int> components_per_qp = {1})
    {
        nq = nq_;
        scratch.SetScratch(nq, components_per_qp);
    }

    void SetScratch(const int nq_, const std::vector<int> &components_per_qp)
    {
        nq = nq_;
        scratch.SetScratch(nq, components_per_qp);
    }

    void SetScratch(const int nq_, const int num_scratch_elem, const int components_per_qp = 1)
    {
        // This is a convenience method for the common case of multiple scratch vectors of the same size (at qp)
        nq = nq_;
        scratch.SetScratch(nq, std::vector<int>(num_scratch_elem, components_per_qp));
    }

    void SetGlobalScratch(const GlobalScratchTuple &global_scratch_)
    {
        scratch.SetGlobalScratch(global_scratch_);
    }

    template <int I>
    auto &GlobalScratch() const
    {
        return scratch.template Global<I>();
    }

    void CloneScratchLayoutTo(QFWithScratch &shadow) const
    {
        shadow.nq = nq;
        scratch.CloneScratchLayoutTo(shadow.scratch);
    }
};




// Global qf with splitting and scratch space.
// The user only writes operator(); the shared base handles scratch setup.
struct CubicQFWithScratch : QFWithScratch<>
{
    CubicQFWithScratch CreateShadow() const
    {
        CubicQFWithScratch shadow;
        CloneScratchLayoutTo(shadow);
        return shadow;
    }

    void operator()(tensor_array<const dscalar_t> &x,
                    tensor_array<const dscalar_t> &coef,
                    tensor_array<const real_t, 2, 2> &J,
                    tensor_array<const real_t> &w,
                    tensor_array<dscalar_t> &y) const
    {
        const int NQ = nq;
        MFEM_ASSERT(NQ == static_cast<int>(x.size()),
                    "unexpected number of quadrature points");

        // Unpack the scratch vectors from the scratch bank
        // In this case it is only one
        auto scratch_q = make_tensor_array<>(scratch[0], NQ);

        // Actual computation, using the scratch vector for intermediate steps
        for (int q = 0; q < NQ; ++q)
        {
            scratch_q(q) = x(q);
        }

        for (int q = 0; q < NQ; ++q)
        {
            scratch_q(q) = scratch_q(q) * x(q);
        }

        for (int q = 0; q < NQ; ++q)
        {
            y(q) = coef(q) * scratch_q(q) * x(q) * det(J(q)) * w(q);
        }

        /*
        // enzyme-aware forall for Cuda run
        mfem::forall<UseEnzyme>(NQ, [=] MFEM_HOST_DEVICE(int q)
        {
            scratch_q(q) = x(q);
        });
        
        mfem::forall<UseEnzyme>(NQ, [=] MFEM_HOST_DEVICE(int q)
        {
            scratch_q(q) = scratch_q(q) * x(q);
        });

        mfem::forall<UseEnzyme>(NQ, [=] MFEM_HOST_DEVICE(int q)
        {
            y(q) = coef(q) * scratch_q(q) * x(q) * det(J(q)) * w(q);
        });*/
    }
};

template <int DIM>
struct CubicQFWithScratchMultipleSizes : QFWithScratch<>
{
    CubicQFWithScratchMultipleSizes CreateShadow() const
    {
        CubicQFWithScratchMultipleSizes shadow;
        this->CloneScratchLayoutTo(shadow);
        return shadow;
    }

   void operator()(tensor_array<const dscalar_t> &x,
                    tensor_array<const dscalar_t> &coef,
                    tensor_array<const real_t, 2, 2> &J,
                    tensor_array<const real_t> &w,
                    tensor_array<dscalar_t> &y) const
    {
        const int NQ = nq;
        MFEM_ASSERT(NQ == static_cast<int>(x.size()),
                    "unexpected number of quadrature points");
 
        // scratch[0]: scalar scratch (1 component per qp) -> will hold x^3
        // @note: need to add a check on the vector/tensor size as it is templated on DIM,
        // but size of scratch[1] is specified in SetScratch
        auto scratch_scalar = make_tensor_array<>(scratch[0], NQ);
        auto scratch_vector = make_tensor_array<DIM>(scratch[1], NQ);
 
        // Step 1: fill the vector scratch with {x, x^2}
        for (int q = 0; q < NQ; ++q)
        {
            scratch_vector(q)(0) = x(q);
            scratch_vector(q)(1) = x(q) * x(q);
        }
 
        // Step 2: combine the two vector components into the scalar scratch -> x^3
        for (int q = 0; q < NQ; ++q)
        {
            scratch_scalar(q) = scratch_vector(q)(0) * scratch_vector(q)(1);
        }
 
        // Step 3: final output, same analytic form as CubicQFWithScratch
        for (int q = 0; q < NQ; ++q)
        {
            y(q) = coef(q) * scratch_scalar(q) * det(J(q)) * w(q);
        }
    }

};

struct CubicQFWithGlobalScratch : QFWithScratch<bool, real_t, Vector>
{
    CubicQFWithGlobalScratch CreateShadow() const
    {
        CubicQFWithGlobalScratch shadow;
        CloneScratchLayoutTo(shadow);
        return shadow;
    }

    void operator()(tensor_array<const dscalar_t> &x,
                    tensor_array<const dscalar_t> &coef,
                    tensor_array<const real_t, 2, 2> &J,
                    tensor_array<const real_t> &w,
                    tensor_array<dscalar_t> &y) const
    {
        const int NQ = nq;
        MFEM_ASSERT(NQ == static_cast<int>(x.size()),
                    "unexpected number of quadrature points");

        // Unpack the scratch vectors from the scratch bank
        auto scratch_q = make_tensor_array<>(scratch[0], NQ);

        // Unpack global scratch
        auto &has_scale = mfem::future::get<0>(scratch.global);  // Or with the convenience scratch.template Global<0>();
        const auto scale = mfem::future::get<1>(scratch.global);
        auto &global_vector = mfem::future::get<2>(scratch.global);

        has_scale = global_vector.Size() > 0;  // If the global vector is non-empty, we will use it to scale the output
        if (has_scale)
        {
            global_vector(0) = 1.0;
        }

        for (int q = 0; q < NQ; ++q)
        {
            scratch_q(q) = x(q) * x(q);
        }

        for (int q = 0; q < NQ; ++q)
        {
            const real_t global_scale = has_scale ? scale * global_vector(0) : 0.0;
            y(q) = global_scale * coef(q) * scratch_q(q) * x(q) * det(J(q)) * w(q);
        }
    }
};

struct CubicQFWithScratchThreeKernels : QFWithScratch<>
{
    CubicQFWithScratchThreeKernels CreateShadow() const
    {
        CubicQFWithScratchThreeKernels shadow;
        CloneScratchLayoutTo(shadow);
        return shadow;
    }

    void operator()(tensor_array<const dscalar_t> &x,
                    tensor_array<const dscalar_t> &coef,
                    tensor_array<const real_t, 2, 2> &J,
                    tensor_array<const real_t> &w,
                    tensor_array<dscalar_t> &y) const
    {
        const int NQ = nq;
        MFEM_ASSERT(NQ == static_cast<int>(x.size()),
                    "unexpected number of quadrature points");

        auto scratch_q = make_tensor_array<>(scratch[0], NQ);

        for (int q = 0; q < NQ; ++q)
        {
            scratch_q(q) = x(q);
        }

        for (int q = 0; q < NQ; ++q)
        {
            scratch_q(q) = scratch_q(q) * x(q);
        }

        for (int q = 0; q < NQ; ++q)
        {
            y(q) = coef(q) * scratch_q(q) * x(q) * det(J(q)) * w(q);
        }
    }
};

///<--- Utils

/// @param fes
/// @param x
void FillInput(ParFiniteElementSpace &fes, Coefficient &input_coeff, Vector &x)
{
    GridFunction x_gf(&fes);
    x_gf.ProjectCoefficient(input_coeff);
    x_gf.GetTrueDofs(x);
}

void FillQData(FiniteElementSpace &fes, const IntegrationRule &ir,
               Coefficient &coeff_fc, QuadratureFunction &coef)
{
    QuadratureSpace qspace(*fes.GetMesh(), ir);
    QuadratureFunction coef_qf(qspace);
    coeff_fc.Project(coef_qf);
    coef = coef_qf;
}

void CheckResults(ParFiniteElementSpace &fes, const IntegrationRule &ir,
           Vector &y, Vector &dy)
{
    FunctionCoefficient expected_coeff([](const Vector &p)
    {
        const real_t input = 1.0 + p(0) + 0.25 * (p.Size() > 1 ? p(1) : 0.0);
        const real_t coeff = 0.5 + p(0) + 0.125 * (p.Size() > 1 ? p(1) : 0.0);
        return coeff * input * input * input;
    });

    ParLinearForm expected_lf(&fes);
    expected_lf.AddDomainIntegrator(new DomainLFIntegrator(expected_coeff, &ir));
    expected_lf.Assemble();

    Vector expected_y(fes.GetTrueVSize());
    fes.GetProlongationMatrix()->MultTranspose(expected_lf, expected_y);

    y -= expected_y;
    const real_t local_err = y.Normlinf();
    real_t global_err = 0.0;
    MPI_Allreduce(&local_err, &global_err, 1, MPITypeMap<real_t>::mpi_type,
                  MPI_MAX, MPI_COMM_WORLD);

    FunctionCoefficient expected_deriv_coeff([](const Vector &p)
    {
        const real_t input = 1.0 + p(0) + 0.25 * (p.Size() > 1 ? p(1) : 0.0);
        const real_t coeff = 0.5 + p(0) + 0.125 * (p.Size() > 1 ? p(1) : 0.0);
        return 3.0 * coeff * input * input;
    });

    ParLinearForm expected_deriv_lf(&fes);
    expected_deriv_lf.AddDomainIntegrator(new DomainLFIntegrator(expected_deriv_coeff, &ir));
    expected_deriv_lf.Assemble();

    Vector expected_dy(fes.GetTrueVSize());
    fes.GetProlongationMatrix()->MultTranspose(expected_deriv_lf, expected_dy);

    dy -= expected_dy;
    const real_t local_deriv_err = dy.Normlinf();
    real_t global_deriv_err = 0.0;
    MPI_Allreduce(&local_deriv_err, &global_deriv_err, 1, MPITypeMap<real_t>::mpi_type,
                  MPI_MAX, MPI_COMM_WORLD);

    if (Mpi::Root())
    {
        cout << "Primal output max error: " << global_err << endl;
        cout << "Derivative output max error: "
             << global_deriv_err << endl;
    }

    REQUIRE(global_err == MFEM_Approx(0.0));
    REQUIRE(global_deriv_err == MFEM_Approx(0.0));
}

///<--- Test
TEST_CASE("dFEM Scratch scalar", "[Parallel][dFEM][Scratch-Scalar]")
{
    int order = 2;
    int ref_levels = 1;

    ///<--- Mesh and finite element space setup
    Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, true, 1.0,
                                      1.0);
    for (int l = 0; l < ref_levels; l++)
    {
        mesh.UniformRefinement();
    }

    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    pmesh.EnsureNodes();
    H1_FECollection fec(order, pmesh.Dimension());
    ParFiniteElementSpace fes(&pmesh, &fec);
    auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
    ParFiniteElementSpace *nodes_fes = nodes->ParFESpace();
    Vector nodes_tvec;
    nodes->GetTrueDofs(nodes_tvec);

    ///<--- dFEM setup
    const IntegrationRule &ir = IntRules.Get(Geometry::SQUARE, 2 * order + 1);
    QuadratureSpace qspace(pmesh, ir);
    VectorQuadratureSpace coef_qspace(qspace, 1);
    QuadratureFunction coef(coef_qspace);
    coef.UseDevice(true);
    FunctionCoefficient coeff_fc([](const Vector &p)
                                   { return 0.5 + p(0) + 0.125 * (p.Size() > 1 ? p(1) : 0.0); });
    FillQData(fes, ir, coeff_fc, coef);

    Array<int> all_domain_attr(pmesh.attributes.Max());
    all_domain_attr = 1;

    const std::vector<FieldDescriptor> inputs{
        {U, &fes},
        {COEF, &coef_qspace},
        {COORDINATES, nodes_fes}};
    const std::vector<FieldDescriptor> outputs{
        {Y, &fes}};
    DifferentiableOperator dop(inputs, outputs, pmesh);

    // Define the cubic qfunction with scratch space
    // Requesting one scalar scratch vector
    // Equivalent to 
    // cubic_qf.SetScratch(pmesh.GetNE() * ir.GetNPoints(), 1, 1);

    CubicQFWithScratch cubic_qf;
    cubic_qf.SetScratch(pmesh.GetNE() * ir.GetNPoints(), {1});
    dop.AddDomainIntegrator<GlobalQFBackend>(
        cubic_qf,
        Inputs<Value<U>, Identity<COEF>, Gradient<COORDINATES>, Weight>{},
        Outputs<Value<Y>>{},
        ir, all_domain_attr,
        Derivatives<U>{});
        

    Vector x(fes.GetTrueVSize()), y(fes.GetTrueVSize()), dx(fes.GetTrueVSize()), dy(fes.GetTrueVSize());
    x.UseDevice(true);
    y.UseDevice(true);
    dx.UseDevice(true);
    dy.UseDevice(true);
    FunctionCoefficient input_coeff([](const Vector &p)
                                    { return 1.0 + p(0) + 0.25 * (p.Size() > 1 ? p(1) : 0.0); });
    FillInput(fes, input_coeff, x);
    ConstantCoefficient direction_coeff(1.0);
    FillInput(fes, direction_coeff, dx);
    y = 0.0;
    dy = 0.0;

    ///<--- Apply the operator
    MultiVector X{x, coef, nodes_tvec};
    MultiVector Y{y};
    dop.Mult(X, Y);

    //<--- Apply derivative operator
    auto dop_deriv = dop.GetDerivative(U, X);
    MultiVector dY{dy};
    dop_deriv->Mult(dx, dY);

    ///<--- Check the result against the expected output
    CheckResults(fes, ir, y, dy);
}

TEST_CASE("dFEM Scratch multi-kernel persists tangents",
          "[Parallel][dFEM][Scratch-MultiKernel]")
{
    int order = 2;
    int ref_levels = 1;

    Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, true, 1.0,
                                      1.0);
    for (int l = 0; l < ref_levels; l++)
    {
        mesh.UniformRefinement();
    }

    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    pmesh.EnsureNodes();
    H1_FECollection fec(order, pmesh.Dimension());
    ParFiniteElementSpace fes(&pmesh, &fec);
    auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
    ParFiniteElementSpace *nodes_fes = nodes->ParFESpace();
    Vector nodes_tvec;
    nodes->GetTrueDofs(nodes_tvec);

    const IntegrationRule &ir = IntRules.Get(Geometry::SQUARE, 2 * order + 1);
    QuadratureSpace qspace(pmesh, ir);
    VectorQuadratureSpace coef_qspace(qspace, 1);
    QuadratureFunction coef(coef_qspace);
    coef.UseDevice(true);
    FunctionCoefficient coeff_fc([](const Vector &p)
                                   { return 0.5 + p(0) + 0.125 * (p.Size() > 1 ? p(1) : 0.0); });
    FillQData(fes, ir, coeff_fc, coef);

    Array<int> all_domain_attr(pmesh.attributes.Max());
    all_domain_attr = 1;

    const std::vector<FieldDescriptor> inputs{
        {U, &fes},
        {COEF, &coef_qspace},
        {COORDINATES, nodes_fes}};
    const std::vector<FieldDescriptor> outputs{
        {Y, &fes}};
    DifferentiableOperator dop(inputs, outputs, pmesh);

    CubicQFWithScratchThreeKernels cubic_qf;
    cubic_qf.SetScratch(pmesh.GetNE() * ir.GetNPoints(), {1});
    dop.AddDomainIntegrator<GlobalQFBackend>(
        cubic_qf,
        Inputs<Value<U>, Identity<COEF>, Gradient<COORDINATES>, Weight>{},
        Outputs<Value<Y>>{},
        ir, all_domain_attr,
        Derivatives<U>{});

    Vector x(fes.GetTrueVSize()), y(fes.GetTrueVSize()), dx(fes.GetTrueVSize());
    Vector dy_action(fes.GetTrueVSize()), dy_cached(fes.GetTrueVSize());
    x.UseDevice(true);
    y.UseDevice(true);
    dx.UseDevice(true);
    dy_action.UseDevice(true);
    dy_cached.UseDevice(true);
    FunctionCoefficient input_coeff([](const Vector &p)
                                    { return 1.0 + p(0) + 0.25 * (p.Size() > 1 ? p(1) : 0.0); });
    FillInput(fes, input_coeff, x);
    ConstantCoefficient direction_coeff(1.0);
    FillInput(fes, direction_coeff, dx);
    y = 0.0;
    dy_action = 0.0;
    dy_cached = 0.0;

    MultiVector X{x, coef, nodes_tvec};
    MultiVector Y{y};
    dop.Mult(X, Y);

    auto dop_deriv_action = dop.GetDerivative(U, X, false);
    MultiVector dY_action{dy_action};
    dop_deriv_action->Mult(dx, dY_action);
    Vector y_action_check(y);
    CheckResults(fes, ir, y_action_check, dy_action);

    auto dop_deriv_cached = dop.GetDerivative(U, X, true);
    MultiVector dY_cached{dy_cached};
    dop_deriv_cached->Mult(dx, dY_cached);

    Vector y_cached_check(y);
    CheckResults(fes, ir, y_cached_check, dy_cached);
}


TEST_CASE("dFEM Scratch multiple sizes", "[Parallel][dFEM][Scratch-Multiple-Sizes]")
{
    int order = 2;
    int ref_levels = 1;
    int DIM = 2;

    ///<--- Mesh and finite element space setup
    Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, true, 1.0,
                                      1.0);
    for (int l = 0; l < ref_levels; l++)
    {
        mesh.UniformRefinement();
    }

    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    pmesh.EnsureNodes();
    H1_FECollection fec(order, pmesh.Dimension());
    ParFiniteElementSpace fes(&pmesh, &fec);
    auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
    ParFiniteElementSpace *nodes_fes = nodes->ParFESpace();
    Vector nodes_tvec;
    nodes->GetTrueDofs(nodes_tvec);

    ///<--- dFEM setup
    const IntegrationRule &ir = IntRules.Get(Geometry::SQUARE, 2 * order + 1);
    QuadratureSpace qspace(pmesh, ir);
    VectorQuadratureSpace coef_qspace(qspace, 1);
    QuadratureFunction coef(coef_qspace);
    coef.UseDevice(true);
    FunctionCoefficient coeff_fc([](const Vector &p)
                                   { return 0.5 + p(0) + 0.125 * (p.Size() > 1 ? p(1) : 0.0); });
    FillQData(fes, ir, coeff_fc, coef);

    Array<int> all_domain_attr(pmesh.attributes.Max());
    all_domain_attr = 1;

    const std::vector<FieldDescriptor> inputs{
        {U, &fes},
        {COEF, &coef_qspace},
        {COORDINATES, nodes_fes}};
    const std::vector<FieldDescriptor> outputs{
        {Y, &fes}};
    DifferentiableOperator dop(inputs, outputs, pmesh);

    // Define the cubic qfunction with scratch space
    // Requesting one scalar scratch vector per dimension
    CubicQFWithScratch cubic_qf;
    cubic_qf.SetScratch(pmesh.GetNE() * ir.GetNPoints(), {1, DIM});
    dop.AddDomainIntegrator<GlobalQFBackend>(
        cubic_qf,
        Inputs<Value<U>, Identity<COEF>, Gradient<COORDINATES>, Weight>{},
        Outputs<Value<Y>>{},
        ir, all_domain_attr,
        Derivatives<U>{});
        

    Vector x(fes.GetTrueVSize()), y(fes.GetTrueVSize()), dx(fes.GetTrueVSize()), dy(fes.GetTrueVSize());
    x.UseDevice(true);
    y.UseDevice(true);
    dx.UseDevice(true);
    dy.UseDevice(true);
    FunctionCoefficient input_coeff([](const Vector &p)
                                    { return 1.0 + p(0) + 0.25 * (p.Size() > 1 ? p(1) : 0.0); });
    FillInput(fes, input_coeff, x);
    ConstantCoefficient direction_coeff(1.0);
    FillInput(fes, direction_coeff, dx);
    y = 0.0;
    dy = 0.0;

    ///<--- Apply the operator
    MultiVector X{x, coef, nodes_tvec};
    MultiVector Y{y};
    dop.Mult(X, Y);

    //<--- Apply derivative operator
    auto dop_deriv = dop.GetDerivative(U, X);
    MultiVector dY{dy};
    dop_deriv->Mult(dx, dY);

    ///<--- Check the result against the expected output
    CheckResults(fes, ir, y, dy);
}

TEST_CASE("dFEM Global Scratch with tuple objects", "[Parallel][dFEM][GlobalScratch]")
{
    int order = 2;
    int ref_levels = 1;

    ///<--- Mesh and finite element space setup
    Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, true, 1.0,
                                      1.0);
    for (int l = 0; l < ref_levels; l++)
    {
        mesh.UniformRefinement();
    }

    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    pmesh.EnsureNodes();
    H1_FECollection fec(order, pmesh.Dimension());
    ParFiniteElementSpace fes(&pmesh, &fec);
    auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
    ParFiniteElementSpace *nodes_fes = nodes->ParFESpace();
    Vector nodes_tvec;
    nodes->GetTrueDofs(nodes_tvec);

    ///<--- dFEM setup
    const IntegrationRule &ir = IntRules.Get(Geometry::SQUARE, 2 * order + 1);
    QuadratureSpace qspace(pmesh, ir);
    VectorQuadratureSpace coef_qspace(qspace, 1);
    QuadratureFunction coef(coef_qspace);
    coef.UseDevice(true);
    FunctionCoefficient coeff_fc([](const Vector &p)
                                   { return 0.5 + p(0) + 0.125 * (p.Size() > 1 ? p(1) : 0.0); });
    FillQData(fes, ir, coeff_fc, coef);

    Array<int> all_domain_attr(pmesh.attributes.Max());
    all_domain_attr = 1;

    const std::vector<FieldDescriptor> inputs{
        {U, &fes},
        {COEF, &coef_qspace},
        {COORDINATES, nodes_fes}};
    const std::vector<FieldDescriptor> outputs{
        {Y, &fes}};
    DifferentiableOperator dop(inputs, outputs, pmesh);

    Vector global_vec(1);
    global_vec.UseDevice(true);
    global_vec = 0.0;
    real_t global_scalar = 1.0;
    bool global_flag;

    CubicQFWithGlobalScratch cubic_qf;
    cubic_qf.SetScratch(pmesh.GetNE() * ir.GetNPoints(), {1});
    cubic_qf.SetGlobalScratch(
        mfem::future::make_tuple(global_flag, global_scalar, global_vec));
    dop.AddDomainIntegrator<GlobalQFBackend>(
        cubic_qf,
        Inputs<Value<U>, Identity<COEF>, Gradient<COORDINATES>, Weight>{},
        Outputs<Value<Y>>{},
        ir, all_domain_attr,
        Derivatives<U>{});

    Vector x(fes.GetTrueVSize()), y(fes.GetTrueVSize()), dx(fes.GetTrueVSize()), dy(fes.GetTrueVSize());
    x.UseDevice(true);
    y.UseDevice(true);
    dx.UseDevice(true);
    dy.UseDevice(true);
    FunctionCoefficient input_coeff([](const Vector &p)
                                    { return 1.0 + p(0) + 0.25 * (p.Size() > 1 ? p(1) : 0.0); });
    FillInput(fes, input_coeff, x);
    ConstantCoefficient direction_coeff(1.0);
    FillInput(fes, direction_coeff, dx);
    y = 0.0;
    dy = 0.0;

    ///<--- Apply the operator
    MultiVector X{x, coef, nodes_tvec};
    MultiVector Y{y};
    dop.Mult(X, Y);

    //<--- Apply derivative operator
    auto dop_deriv = dop.GetDerivative(U, X);
    MultiVector dY{dy};
    dop_deriv->Mult(dx, dY);

    ///<--- Check the result against the expected output
    CheckResults(fes, ir, y, dy);
}

#endif // MFEM_USE_MPI