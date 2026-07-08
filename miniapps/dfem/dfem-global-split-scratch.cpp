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

//                   -----------------------------------------
//                   Minimal dFEM Global QFunction Split Demo
//                   -----------------------------------------
//
// Compile with: make dfem-global-split-scratch
//
// Sample runs:  mpirun -np 1 dfem-global-split-scratch
//               mpirun -np 1 dfem-global-split-scratch -d cuda
//
// Description:  This miniapp mirrors the dop cubic qfunction from the
//               small Enzyme tests through dFEM instead of calling Enzyme
//               directly. It applies y = coef * x^3 with a global qfunction.

#include "mfem.hpp"
#include "../../fem/dfem/doperator.hpp"
#include "../../linalg/tensor_arrays.hpp"

using namespace std;
using namespace mfem;
using namespace mfem::future;

using dscalar_t = real_t;

///<--- Q-functions
constexpr int U = 1;
constexpr int Y = 2;
constexpr int COEF = 3;
constexpr int COORDINATES = 4;

// Global qf with splitting and scratch space
struct CubicQFWithScratch
{
    int nq = 0;
    Vector scratch;
    real_t* scratch_d = nullptr;

    void SetScratch(const int nq_)
    {
        nq = nq_;
        scratch.SetSize(nq);
        scratch.UseDevice(true);
        scratch = 0.0;
        scratch_d = scratch.ReadWrite();
    }

    CubicQFWithScratch CreateShadow() const
    {
        CubicQFWithScratch shadow;
        shadow.SetScratch(nq);
        return shadow;
    }

    inline MFEM_HOST_DEVICE
    void operator()(tensor_array<const dscalar_t> &x,
                    tensor_array<const dscalar_t> &coef,
                    tensor_array<const real_t, 2, 2> &J,
                    tensor_array<const real_t> &w,
                    tensor_array<dscalar_t> &y) const
    {
        const int NQ = nq;
        MFEM_ASSERT(NQ == static_cast<int>(x.size()),
                    "unexpected number of quadrature points");

        auto scratch_q = make_tensor_array<>(scratch_d, NQ);

        for (size_t q = 0; q < x.size(); q++)
        {
            scratch_q(q) = x(q);
        }

        for (size_t q = 0; q < x.size(); q++)
        {
            scratch_q(q) = scratch_q(q) * x(q);
        }

        for (size_t q = 0; q < x.size(); q++)
        {
            y(q) = coef(q) * scratch_q(q) * x(q) * det(J(q)) * w(q);
        }
        
        /*mfem::forall<UseEnzyme>(NQ, [=] MFEM_HOST_DEVICE(int q)
                                { qf_scratch[q] = x(q); });

        mfem::forall<UseEnzyme>(NQ, [=] MFEM_HOST_DEVICE(int q)
                                { qf_scratch[q] = qf_scratch[q] * x(q); });

        mfem::forall<UseEnzyme>(NQ, [=] MFEM_HOST_DEVICE(int q)
                                { y(q) = coef(q) * qf_scratch[q] * x(q) * det(J(q)) * w(q); });*/
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
}

///<--- Main
int main(int argc, char *argv[])
{
    Mpi::Init(argc, argv);
    Hypre::Init();

    const char *device_config = "cpu";
    int order = 2;
    int ref_levels = 1;

    OptionsParser args(argc, argv);
    args.AddOption(&device_config, "-d", "--device",
                   "Device configuration string, see Device::Configure().");
    args.AddOption(&order, "-o", "--order", "Finite element order.");
    args.AddOption(&ref_levels, "-r", "--refine", "Serial refinement levels.");
    args.Parse();
    if (!args.Good())
    {
        if (Mpi::Root())
        {
            args.PrintUsage(cout);
        }
        return 1;
    }
    if (Mpi::Root())
    {
        args.PrintOptions(cout);
    }

    ///<--- Device configuration
    Device device(device_config);
    if (Mpi::Root())
    {
        device.Print();
    }

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
    //Vector scratch(qspace.GetSize());
    coef.UseDevice(true);
    //scratch.UseDevice(true);
    FunctionCoefficient coeff_fc([](const Vector &p)
                                   { return 0.5 + p(0) + 0.125 * (p.Size() > 1 ? p(1) : 0.0); });
    FillQData(fes, ir, coeff_fc, coef);
    //scratch = -1.0;

    Array<int> all_domain_attr(pmesh.attributes.Max());
    all_domain_attr = 1;

    const std::vector<FieldDescriptor> inputs{
        {U, &fes},
        {COEF, &coef_qspace},
        {COORDINATES, nodes_fes}};
    const std::vector<FieldDescriptor> outputs{
        {Y, &fes}};
    DifferentiableOperator dop(inputs, outputs, pmesh);
    dop.SetQLayouts({{Value<U>{}, {1, 0}}}, {{Value<Y>{}, {1, 0}}});
    CubicQFWithScratch cubic_qf;
    cubic_qf.SetScratch(pmesh.GetNE() * ir.GetNPoints());
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

    return 0;
}