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

// Test for the dFEM global qfunction with split computation and scratch space. 

#include "../unit_tests.hpp"
#include "mfem.hpp"

#ifdef MFEM_USE_MPI

#include "../../../fem/dfem/doperator.hpp"
#include "../../../fem/dfem/backends/global_qf/prelude.hpp"      

#include "../../../linalg/tensor_arrays.hpp"

using namespace std;
using namespace mfem;
using namespace mfem::future;

using dscalar_t = real_t;

///<--- Q-functions
constexpr int U = 1;
constexpr int Y = 2;
constexpr int COEF = 3;
constexpr int COORDINATES = 4;


// Simple container for scratch vectors that can be used in a global qfunction. 
// The user can add scratch vectors to the bank, and the qfunction can access them by index.
struct ScratchBank
{
    int nq = 0;
    std::vector<int> components;
    std::vector<int> sizes;
    std::vector<std::shared_ptr<Vector>> owned;
    std::vector<real_t *> ptrs;

    // Setter methods
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

    real_t *operator[](const int i) const { return ptrs[i]; }
    int Size() const { return static_cast<int>(ptrs.size()); }
};

// Global qf with splitting and scratch space
// @note: We might create a class for this,  and the user only takes care about the operator() method.
// @note: Otherwise the user might instantiate a ScratchBank object directly and pass it to the qfunction
struct CubicQFWithScratch
{
    int nq = 0;
    ScratchBank scratch;

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

    CubicQFWithScratch CreateShadow() const
    {
        CubicQFWithScratch shadow;
        shadow.SetScratch(nq, scratch.components);
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
TEST_CASE("dFEM Global Scratch", "[Parallel][dFEM]")
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

#endif // MFEM_USE_MPI