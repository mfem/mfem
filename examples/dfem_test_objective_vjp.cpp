#include "dfem/dfem_refactor.hpp"

#include <fstream>

using namespace mfem;
using mfem::internal::tensor;

static constexpr int dim = 2;

struct ObjectiveQFunction
{
    MFEM_HOST_DEVICE inline
    auto operator()(const real_t& u, const tensor<real_t, dim, dim>& dXdxi, const double& w) const
    {
        auto dV = det(dXdxi)*w;
        return mfem::tuple{u*dV};
    }
};

int main()
{
    Mpi::Init();
    out << std::setprecision(8);

    int polynomial_order = 1;
    int ir_order = 2;

    Mesh mesh_serial = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL, false, 1.0, 1.0);
    mesh_serial.EnsureNodes();
    auto mesh = ParMesh(MPI_COMM_WORLD, mesh_serial);

    auto mesh_nodes = static_cast<ParGridFunction*>(mesh.GetNodes());
    ParFiniteElementSpace& mesh_fes = *mesh_nodes->ParFESpace();

    out << "#el: " << mesh.GetNE() << "\n";

    H1_FECollection fec(polynomial_order, dim);
    ParFiniteElementSpace fes(&mesh, &fec);
    const IntegrationRule &ir =
      IntRules.Get(fes.GetFE(0)->GetGeomType(),
                   2 * ir_order + fes.GetFE(0)->GetOrder());

    ParGridFunction u(&fes);
    //u = 0.0;
    FunctionCoefficient fc([](const Vector& X) { return 2.0*X(0) - X(1); });
    u.ProjectCoefficient(fc);
    out << "u = ";
    u.Print();

    constexpr int Potential = 0;
    constexpr int Coordinates = 1;

    std::vector<FieldDescriptor> solutions{FieldDescriptor{Potential, &fes}};
    std::vector<FieldDescriptor> parameters{FieldDescriptor{Coordinates, &mesh_fes}};
    DifferentiableOperator op(solutions, parameters, mesh);
    op.DisableTensorProductStructure();

    // set up differentiable operator
    ObjectiveQFunction qf;
    mfem::tuple inputs{Value<Potential>{}, Gradient<Coordinates>{}, Weight{}};
    // mfem::tuple outputs{One<Potential>{}};
    mfem::tuple outputs{Value<Potential>{}};
    Array<int> solid_domain_attr(mesh.attributes.Max());
    solid_domain_attr[0] = 1;
    auto derivatives = std::integer_sequence<size_t, Potential>{};
    op.AddDomainIntegrator(qf, inputs, outputs, ir, solid_domain_attr, derivatives);
    op.SetParameters({mesh_nodes});

    Vector z(1);
    op.Mult(u, z);
    // z should be singleton, debug needed
    out << "z = ";
    z.Print();

    // z_bar should be a singelton, need to debug
    Vector z_bar(u.Size());
    z_bar = 0.0;
    z_bar(0) = 1.0;
    out << "z_bar = ";
    z_bar.Print();
    Vector u_bar(u.Size());
    auto jac = op.GetDerivative(Potential, {&u}, {mesh_nodes});
    // jac->MultTranspose(z_bar, u_bar);
    // out << "u_bar = ";
    // u_bar.Print();

    for (int i = 0; i < z_bar.Size(); i++) {
        z_bar = 0.0;
        z_bar(i) = 1.0;
        jac->MultTranspose(z_bar, u_bar);
        out << "jac col " << i << " = ";
        u_bar.Print();
    }

    out << std::endl;
    
    Vector u_dot(u.Size());
    Vector z_dot(u.Size());
    for (int i = 0; i < z_bar.Size(); i++) {
        u_dot = 0.0;
        u_dot(i) = 1.0;
        jac->Mult(u_dot, z_dot);
        out << "jac row " << i << " = ";
        z_dot.Print();
    }
    return 0;
}