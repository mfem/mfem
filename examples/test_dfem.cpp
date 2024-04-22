#include "dfem/dfem_refactor.hpp"
#include "examples/dfem/dfem_fieldoperator.hpp"
#include "examples/dfem/dfem_util.hpp"
#include "fem/bilininteg.hpp"
#include "fem/coefficient.hpp"
#include "fem/pbilinearform.hpp"
#include "general/error.hpp"
#include "general/globals.hpp"
#include <linalg/tensor.hpp>

using namespace mfem;
using mfem::internal::tensor;

// int test_interpolate_linear_scalar(std::string mesh_file,
//                                    int refinements,
//                                    int polynomial_order)
// {
//    Mesh mesh_serial = Mesh(mesh_file);
//    for (int i = 0; i < refinements; i++)
//    {
//       mesh_serial.UniformRefinement();
//    }
//    ParMesh mesh(MPI_COMM_WORLD, mesh_serial);

//    mesh.SetCurvature(1);
//    const int dim = mesh.Dimension();
//    mesh_serial.Clear();

//    ParGridFunction* mesh_nodes = static_cast<ParGridFunction *>(mesh.GetNodes());
//    ParFiniteElementSpace &mesh_fes = *mesh_nodes->ParFESpace();

//    H1_FECollection h1fec(polynomial_order, dim);
//    ParFiniteElementSpace h1fes(&mesh, &h1fec);

//    const IntegrationRule &ir =
//       IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder() + 1);

//    QuadratureSpace qspace(mesh, ir);
//    QuadratureFunction qf(&qspace);

//    ParGridFunction f1_g(&h1fes);

//    auto kernel = [](const double &u, const tensor<double, 2, 2> &J,
//                     const double &w)
//    {
//       return u;
//    };

//    std::tuple argument_operators = {Value{"potential"}, Gradient{"coordinates"}, Weight{"integration_weights"}};
//    std::tuple output_operator = {Value{"potential"}};

//    ElementOperator eop{kernel, argument_operators, output_operator};
//    auto ops = std::tuple{eop};

//    auto solutions = std::array{FieldDescriptor{&h1fes, "potential"}};
//    auto parameters = std::array{FieldDescriptor{&mesh_fes, "coordinates"}};

//    DifferentiableOperator dop(solutions, parameters, ops, mesh, ir);

//    auto f1 = [](const Vector &coords)
//    {
//       const double x = coords(0);
//       const double y = coords(1);
//       return 2.345 + x + y;
//    };

//    FunctionCoefficient f1_c(f1);
//    f1_g.ProjectCoefficient(f1_c);

//    Vector x(*f1_g.GetTrueDofs()), y(h1fes.TrueVSize());
//    dop.SetParameters({mesh_nodes});
//    dop.Mult(x, y);

//    Vector f_test(h1fes.GetElementRestriction(
//                     ElementDofOrdering::NATIVE)->Height());
//    for (int e = 0; e < mesh.GetNE(); e++)
//    {
//       ElementTransformation *T = mesh.GetElementTransformation(e);
//       for (int qp = 0; qp < ir.GetNPoints(); qp++)
//       {
//          const IntegrationPoint &ip = ir.IntPoint(qp);
//          T->SetIntPoint(&ip);

//          f_test((e * ir.GetNPoints()) + qp) = f1_c.Eval(*T, ip);
//       }
//    }

//    f_test -= dop.GetResidualQpMemory();
//    if (f_test.Norml2() > 1e-10)
//    {
//       return 1;
//    }

//    return 0;
// }

// int test_interpolate_gradient_scalar(std::string mesh_file,
//                                      int refinements,
//                                      int polynomial_order)
// {
//    Mesh mesh_serial = Mesh(mesh_file);
//    for (int i = 0; i < refinements; i++)
//    {
//       mesh_serial.UniformRefinement();
//    }
//    ParMesh mesh(MPI_COMM_WORLD, mesh_serial);
//    mesh.SetCurvature(1);
//    const int dim = mesh.Dimension();
//    mesh_serial.Clear();

//    ParGridFunction* mesh_nodes = static_cast<ParGridFunction *>(mesh.GetNodes());
//    ParFiniteElementSpace &mesh_fes = *mesh_nodes->ParFESpace();

//    H1_FECollection h1fec(polynomial_order, dim);
//    ParFiniteElementSpace h1fes(&mesh, &h1fec);

//    const IntegrationRule &ir =
//       IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder() + 1);

//    QuadratureSpace qspace(mesh, ir);
//    QuadratureFunction qf(&qspace, dim);

//    ParGridFunction f1_g(&h1fes);

//    auto kernel = [](const double &u, const tensor<double, 2> &grad_u,
//                     const tensor<double, 2, 2> &J,
//                     const double &w)
//    {
//       return grad_u * inv(J);
//    };

//    std::tuple argument_operators = {Value{"potential"}, Gradient{"potential"}, Gradient{"coordinates"}, Weight{"integration_weights"}};
//    std::tuple output_operator = {Gradient{"potential"}};

//    ElementOperator eop{kernel, argument_operators, output_operator};
//    auto ops = std::tuple{eop};

//    auto solutions = std::array{FieldDescriptor{&h1fes, "potential"}};
//    auto parameters = std::array{FieldDescriptor{&mesh_fes, "coordinates"}};

//    DifferentiableOperator dop(solutions, parameters, ops, mesh, ir);

//    auto f1 = [](const Vector &coords)
//    {
//       const double x = coords(0);
//       const double y = coords(1);
//       return 2.345 + x*y + y;
//    };

//    FunctionCoefficient f1_c(f1);
//    f1_g.ProjectCoefficient(f1_c);

//    Vector x(f1_g), y(f1_g.Size());
//    dop.SetParameters({mesh_nodes});
//    dop.Mult(x, y);

//    Vector f_test(qf.Size());
//    for (int e = 0; e < mesh.GetNE(); e++)
//    {
//       ElementTransformation *T = mesh.GetElementTransformation(e);
//       for (int qp = 0; qp < ir.GetNPoints(); qp++)
//       {
//          const IntegrationPoint &ip = ir.IntPoint(qp);
//          T->SetIntPoint(&ip);

//          Vector g(dim);
//          f1_g.GetGradient(*T, g);
//          for (int d = 0; d < dim; d++)
//          {
//             int qpo = qp * dim;
//             int eo = e * (ir.GetNPoints() * dim);
//             f_test(d + qpo + eo) = g(d);
//          }
//       }
//    }

//    f_test -= dop.GetResidualQpMemory();
//    if (f_test.Norml2() > 1e-10)
//    {
//       return 1;
//    }
//    return 0;
// }

// int test_interpolate_linear_vector(std::string mesh_file, int refinements,
//                                    int polynomial_order)
// {
//    constexpr int vdim = 2;
//    Mesh mesh_serial = Mesh(mesh_file);
//    for (int i = 0; i < refinements; i++)
//    {
//       mesh_serial.UniformRefinement();
//    }

//    ParMesh mesh(MPI_COMM_WORLD, mesh_serial);
//    mesh.SetCurvature(1);
//    const int dim = mesh.Dimension();
//    mesh_serial.Clear();

//    ParGridFunction* mesh_nodes = static_cast<ParGridFunction *>(mesh.GetNodes());
//    ParFiniteElementSpace &mesh_fes = *mesh_nodes->ParFESpace();

//    H1_FECollection h1fec(polynomial_order, dim);
//    ParFiniteElementSpace h1fes(&mesh, &h1fec, vdim);

//    const IntegrationRule &ir =
//       IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder() + 1);

//    QuadratureSpace qspace(mesh, ir);
//    QuadratureFunction qf(&qspace, vdim);

//    ParGridFunction f1_g(&h1fes);

//    auto kernel = [](const tensor<double, 2> &u, const tensor<double, 2, 2> &grad_u,
//                     const tensor<double, 2, 2> &J,
//                     const double &w)
//    {
//       return u;
//    };

//    std::tuple argument_operators = {Value{"potential"}, Gradient{"potential"}, Gradient{"coordinates"}, Weight{"integration_weights"}};
//    std::tuple output_operator = {Value{"potential"}};

//    ElementOperator eop{kernel, argument_operators, output_operator};
//    auto ops = std::tuple{eop};

//    auto solutions = std::array{FieldDescriptor{&h1fes, "potential"}};
//    auto parameters = std::array{FieldDescriptor{&mesh_fes, "coordinates"}};

//    DifferentiableOperator dop(solutions, parameters, ops, mesh, ir);

//    auto f1 = [](const Vector &coords, Vector &u)
//    {
//       const double x = coords(0);
//       const double y = coords(1);
//       u(0) = 2.345 + x + y;
//       u(1) = 12.345 + x + y;
//    };

//    VectorFunctionCoefficient f1_c(vdim, f1);
//    f1_g.ProjectCoefficient(f1_c);

//    Vector x(f1_g), y(f1_g.Size());
//    dop.SetParameters({mesh_nodes});
//    dop.Mult(x, y);

//    Vector f_test(qf.Size());
//    for (int e = 0; e < mesh.GetNE(); e++)
//    {
//       ElementTransformation *T = mesh.GetElementTransformation(e);
//       for (int qp = 0; qp < ir.GetNPoints(); qp++)
//       {
//          const IntegrationPoint &ip = ir.IntPoint(qp);
//          T->SetIntPoint(&ip);

//          Vector f(vdim);
//          f1_g.GetVectorValue(*T, ip, f);
//          for (int d = 0; d < vdim; d++)
//          {
//             int qpo = qp * vdim;
//             int eo = e * (ir.GetNPoints() * vdim);
//             f_test(d + qpo + eo) = f(d);
//          }
//       }
//    }

//    f_test -= dop.GetResidualQpMemory();
//    if (f_test.Norml2() > 1e-10)
//    {
//       return 1;
//    }
//    return 0;
// }

// int test_interpolate_gradient_vector(std::string mesh_file,
//                                      int refinements,
//                                      int polynomial_order)
// {
//    constexpr int vdim = 2;
//    Mesh mesh_serial = Mesh(mesh_file, 1, 1);
//    for (int i = 0; i < refinements; i++)
//    {
//       mesh_serial.UniformRefinement();
//    }
//    ParMesh mesh(MPI_COMM_WORLD, mesh_serial);
//    mesh.SetCurvature(1);
//    const int dim = mesh.Dimension();
//    mesh_serial.Clear();

//    ParGridFunction* mesh_nodes = static_cast<ParGridFunction *>(mesh.GetNodes());
//    ParFiniteElementSpace &mesh_fes = *mesh_nodes->ParFESpace();

//    H1_FECollection h1fec(polynomial_order, dim);
//    ParFiniteElementSpace h1fes(&mesh, &h1fec, vdim);

//    const IntegrationRule &ir =
//       IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder() + 1);

//    QuadratureSpace qspace(mesh, ir);
//    QuadratureFunction qf(&qspace, dim * vdim);

//    ParGridFunction f1_g(&h1fes);

//    auto kernel = [](const tensor<double, vdim, 2> &grad_u,
//                     const tensor<double, 2, 2> &J,
//                     const double &w)
//    {
//       return grad_u * inv(J);
//    };

//    std::tuple argument_operators = {Gradient{"potential"}, Gradient{"coordinates"}, Weight{"integration_weights"}};
//    std::tuple output_operator = {Gradient{"potential"}};

//    ElementOperator eop{kernel, argument_operators, output_operator};
//    auto ops = std::tuple{eop};

//    auto solutions = std::array{FieldDescriptor{&h1fes, "potential"}};
//    auto parameters = std::array{FieldDescriptor{&mesh_fes, "coordinates"}};

//    DifferentiableOperator dop(solutions, parameters, ops, mesh, ir);

//    auto f1 = [](const Vector &coords, Vector &u)
//    {
//       const double x = coords(0);
//       const double y = coords(1);
//       u(0) = x + y;
//       u(1) = x + 0.5*y;
//    };

//    VectorFunctionCoefficient f1_c(vdim, f1);
//    f1_g.ProjectCoefficient(f1_c);

//    Vector x(f1_g), y(f1_g.Size());
//    dop.SetParameters({mesh_nodes});
//    dop.Mult(x, y);

//    Vector f_test(qf.Size());
//    for (int e = 0; e < mesh.GetNE(); e++)
//    {
//       ElementTransformation *T = mesh.GetElementTransformation(e);
//       for (int qp = 0; qp < ir.GetNPoints(); qp++)
//       {
//          const IntegrationPoint &ip = ir.IntPoint(qp);
//          T->SetIntPoint(&ip);

//          DenseMatrix g(vdim, dim);
//          f1_g.GetVectorGradient(*T, g);
//          for (int i = 0; i < vdim; i++)
//          {
//             for (int j = 0; j < dim; j++)
//             {
//                int eo = e * (ir.GetNPoints() * dim * vdim);
//                int qpo = qp * dim * vdim;
//                int idx = (j + (i * dim) + qpo + eo);
//                f_test(idx) = g(i, j);
//             }
//          }
//       }
//    }

//    f_test -= dop.GetResidualQpMemory();
//    if (f_test.Norml2() > 1e-10)
//    {
//       out << "||u - u_ex||_l2 = " << f_test.Norml2() << "\n";
//       return 1;
//    }
//    return 0;
// }

// int test_domain_lf_integrator(std::string mesh_file,
//                               int refinements,
//                               int polynomial_order)
// {
//    Mesh mesh_serial = Mesh(mesh_file);
//    for (int i = 0; i < refinements; i++)
//    {
//       mesh_serial.UniformRefinement();
//    }
//    ParMesh mesh(MPI_COMM_WORLD, mesh_serial);

//    mesh.SetCurvature(1);
//    const int dim = mesh.Dimension();
//    mesh_serial.Clear();

//    ParGridFunction* mesh_nodes = static_cast<ParGridFunction *>(mesh.GetNodes());
//    ParFiniteElementSpace &mesh_fes = *mesh_nodes->ParFESpace();

//    H1_FECollection h1fec(polynomial_order, dim);
//    ParFiniteElementSpace h1fes(&mesh, &h1fec);

//    const IntegrationRule &ir =
//       IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder());

//    ParGridFunction f1_g(&h1fes);

//    auto kernel = [](const double &u, const tensor<double, 2, 2> &J,
//                     const double &w)
//    {
//       return u * det(J) * w;
//    };

//    std::tuple argument_operators = {Value{"potential"}, Gradient{"coordinates"}, Weight{"integration_weights"}};
//    std::tuple output_operator = {Value{"potential"}};

//    ElementOperator eop{kernel, argument_operators, output_operator};
//    auto ops = std::tuple{eop};

//    auto solutions = std::array{FieldDescriptor{&h1fes, "potential"}};
//    auto parameters = std::array{FieldDescriptor{&mesh_fes, "coordinates"}};

//    DifferentiableOperator dop(solutions, parameters, ops, mesh, ir);

//    auto f1 = [](const Vector &coords)
//    {
//       const double x = coords(0);
//       const double y = coords(1);
//       return 2.345 + x + y;
//    };

//    FunctionCoefficient f1_c(f1);
//    f1_g.ProjectCoefficient(f1_c);

//    Vector x(f1_g), y(h1fes.TrueVSize());
//    dop.SetParameters({mesh_nodes});
//    dop.Mult(x, y);

//    ParLinearForm b(&h1fes);
//    b.AddDomainIntegrator(new DomainLFIntegrator(f1_c));
//    b.Assemble();

//    b -= y;
//    if (b.Norml2() > 1e-10)
//    {
//       out << "||u - u_ex||_l2 = " << b.Norml2() << "\n";
//       return 1;
//    }

//    return 0;
// }

// int test_diffusion_integrator(std::string mesh_file,
//                               int refinements,
//                               int polynomial_order)
// {
//    Mesh mesh_serial = Mesh(mesh_file);
//    for (int i = 0; i < refinements; i++)
//    {
//       mesh_serial.UniformRefinement();
//    }
//    ParMesh mesh(MPI_COMM_WORLD, mesh_serial);

//    mesh.SetCurvature(1);
//    const int dim = mesh.Dimension();
//    mesh_serial.Clear();

//    ParGridFunction* mesh_nodes = static_cast<ParGridFunction *>(mesh.GetNodes());
//    ParFiniteElementSpace &mesh_fes = *mesh_nodes->ParFESpace();

//    H1_FECollection h1fec(polynomial_order, dim);
//    ParFiniteElementSpace h1fes(&mesh, &h1fec);

//    const IntegrationRule &ir =
//       IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder());

//    ParGridFunction f1_g(&h1fes);
//    ParGridFunction rho_g(&h1fes);

//    auto rho_f = [](const Vector &coords)
//    {
//       const double x = coords(0);
//       const double y = coords(1);
//       return x + y;
//    };

//    FunctionCoefficient rho_c(rho_f);
//    rho_g.ProjectCoefficient(rho_c);

//    auto kernel = [](const tensor<double, 2> &grad_u, const double &rho,
//                     const tensor<double, 2, 2> &J,
//                     const double &w)
//    {
//       auto invJ = inv(J);
//       return rho*rho * grad_u * invJ * transpose(invJ) * det(J) * w;
//    };

//    std::tuple argument_operators = {Gradient{"potential"}, Value{"density"}, Gradient{"coordinates"}, Weight{"integration_weights"}};
//    std::tuple output_operator = {Gradient{"potential"}};

//    ElementOperator eop = {kernel, argument_operators, output_operator};
//    auto ops = std::tuple{eop};

//    auto solutions = std::array{FieldDescriptor{&h1fes, "potential"}};
//    auto parameters = std::array
//    {
//       FieldDescriptor{&h1fes, "density"},
//       FieldDescriptor{&mesh_fes, "coordinates"}
//    };

//    DifferentiableOperator dop(solutions, parameters, ops, mesh, ir);

//    auto f1 = [](const Vector &coords)
//    {
//       const double x = coords(0);
//       const double y = coords(1);
//       return 2.345 + 0.25 * x*x*y + y*y*x;
//    };

//    FunctionCoefficient f1_c(f1);
//    f1_g.ProjectCoefficient(f1_c);

//    Vector x(f1_g), y(h1fes.TrueVSize());
//    dop.SetParameters({&rho_g, mesh_nodes});
//    dop.Mult(x, y);

//    ParBilinearForm a(&h1fes);
//    TransformedCoefficient rho_c2(&rho_c, [](double c) {return c*c;});
//    a.AddDomainIntegrator(new DiffusionIntegrator(rho_c2));
//    a.Assemble();
//    a.Finalize();

//    Vector y2(h1fes.TrueVSize());
//    a.Mult(x, y2);
//    y2 -= y;
//    if (y2.Norml2() > 1e-10)
//    {
//       out << "||F(u) - ex||_l2 = " << y2.Norml2() << "\n";
//       return 1;
//    }

//    // Test linearization here as well
//    auto dFdu = dop.GetDerivativeWrt<0>({&f1_g}, {&rho_g, mesh_nodes});

//    if (dFdu->Height() != h1fes.GetTrueVSize())
//    {
//       out << "dFdu unexpected height of " << dFdu->Height() << "\n";
//       return 1;
//    }

//    dFdu->Mult(x, y);
//    a.Mult(x, y2);
//    y2 -= y;
//    if (y2.Norml2() > 1e-10)
//    {
//       out << "||dFdu u^* - ex||_l2 = " << y2.Norml2() << "\n";
//       return 1;
//    }

//    // fd jacobian test
//    {
//       double eps = 1.0e-6;
//       Vector v(x), xpv(x), xmv(x), fxpv(x.Size()), fxmv(x.Size());
//       v *= eps;
//       xpv += v;
//       xmv -= v;
//       dop.Mult(xpv, fxpv);
//       dop.Mult(xmv, fxmv);
//       fxpv -= fxmv;
//       fxpv /= (2.0*eps);

//       fxpv -= y;
//       if (fxpv.Norml2() > eps)
//       {
//          out << "||dFdu_FD u^* - ex||_l2 = " << fxpv.Norml2() << "\n";
//          return 1;
//       }
//    }

//    HypreParMatrix A;
//    dFdu->Assemble(A);


//    f1_g.ProjectCoefficient(f1_c);
//    rho_g.ProjectCoefficient(rho_c);
//    auto dFdrho = dop.GetDerivativeWrt<1>({&f1_g}, {&rho_g, mesh_nodes});
//    if (dFdrho->Height() != h1fes.GetTrueVSize())
//    {
//       out << "dFdrho unexpected height of " << dFdrho->Height() << "\n";
//       return 1;
//    }

//    dFdrho->Mult(rho_g, y);

//    // fd test
//    {
//       double eps = 1.0e-6;
//       Vector v(rho_g), rhopv(rho_g), rhomv(rho_g), frhopv(x.Size()), frhomv(x.Size());
//       v *= eps;
//       rhopv += v;
//       rhomv -= v;
//       dop.SetParameters({&rhopv, mesh_nodes});
//       dop.Mult(x, frhopv);
//       dop.SetParameters({&rhomv, mesh_nodes});
//       dop.Mult(x, frhomv);
//       frhopv -= frhomv;
//       frhopv /= (2.0*eps);

//       frhopv -= y;
//       if (frhopv.Norml2() > eps)
//       {
//          out << "||dFdu_FD u^* - ex||_l2 = " << frhopv.Norml2() << "\n";
//          return 1;
//       }
//    }

//    return 0;
// }

// int test_qoi(std::string mesh_file,
//              int refinements,
//              int polynomial_order)
// {
//    Mesh mesh_serial = Mesh(mesh_file);
//    for (int i = 0; i < refinements; i++)
//    {
//       mesh_serial.UniformRefinement();
//    }
//    ParMesh mesh(MPI_COMM_WORLD, mesh_serial);

//    mesh.SetCurvature(1);
//    const int dim = mesh.Dimension();
//    mesh_serial.Clear();

//    ParGridFunction* mesh_nodes = static_cast<ParGridFunction *>(mesh.GetNodes());
//    ParFiniteElementSpace &mesh_fes = *mesh_nodes->ParFESpace();

//    H1_FECollection h1fec(polynomial_order, dim);
//    ParFiniteElementSpace h1fes(&mesh, &h1fec);

//    const IntegrationRule &ir =
//       IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder());

//    ParGridFunction rho_g(&h1fes);

//    auto rho_f = [](const Vector &coords)
//    {
//       const double x = coords(0);
//       const double y = coords(1);
//       return x + y;
//    };

//    FunctionCoefficient rho_c(rho_f);
//    rho_g.ProjectCoefficient(rho_c);

//    auto kernel = [](const double &rho,
//                     const tensor<double, 2, 2> &J,
//                     const double &w)
//    {
//       const double eps = 1.2345;
//       return 0.5 * eps * rho*rho * det(J) * w;
//    };

//    std::tuple argument_operators = {Value{"density"}, Gradient{"coordinates"}, Weight{"integration_weights"}};
//    std::tuple output_operator = {One{"density"}};

//    ElementOperator eop = {kernel, argument_operators, output_operator};
//    auto ops = std::tuple{eop};

//    auto solutions = std::array{FieldDescriptor{&h1fes, "density"}};
//    auto parameters = std::array{FieldDescriptor{&mesh_fes, "coordinates"}};

//    DifferentiableOperator dop(solutions, parameters, ops, mesh, ir);

//    Vector x(rho_g), y(1);
//    dop.SetParameters({mesh_nodes});
//    dop.Mult(x, y);

//    out << "#el: " << mesh.GetNE() << " #qp: " << ir.GetNPoints() << "\n";
//    // print_vector(y);

//    auto dFdrho = dop.GetDerivativeWrt<0>({&rho_g}, {mesh_nodes});

//    Vector dx(rho_g.Size());
//    dx = 1.0;
//    dFdrho->Mult(dx, y);

//    print_vector(y);

//    Vector dFdrho_vec;
//    dFdrho->Assemble(dFdrho_vec);

//    // print_vector(dFdrho_vec);

//    double acc = dFdrho_vec.Sum();
//    out << acc << "\n";

//    acc -= y(0);
//    if (sqrt(acc*acc) > 1e-12)
//    {
//       out << "||dFdu_action  - assembled(dFdu)||_l2 = " << sqrt(acc*acc) << "\n";
//       return 1;
//    }

//    // fd test
//    {
//       double eps = 1.0e-6;
//       Vector v(rho_g.Size()), rhopv(rho_g), rhomv(rho_g), frhopv(1), frhomv(1);
//       v = eps;
//       rhopv += v;
//       rhomv -= v;
//       dop.Mult(rhopv, frhopv);
//       dop.Mult(rhomv, frhomv);
//       frhopv -= frhomv;
//       frhopv /= (2.0*eps);

//       frhopv -= y;
//       if (frhopv.Norml2() > eps)
//       {
//          out << "||dFdu_FD u^* - ex||_l2 = " << frhopv.Norml2() << "\n";
//          return 1;
//       }
//    }

//    return 0;
// }

int test_assemble_mass_hypreparmatrix(std::string mesh_file,
                                      int refinements,
                                      int polynomial_order,
                                      int ir_order)
{
   Mesh mesh_serial = Mesh(mesh_file);
   for (int i = 0; i < refinements; i++)
   {
      mesh_serial.UniformRefinement();
   }
   ParMesh mesh(MPI_COMM_WORLD, mesh_serial);

   mesh.SetCurvature(1);
   const int dim = mesh.Dimension();
   mesh_serial.Clear();

   ParGridFunction* mesh_nodes = static_cast<ParGridFunction *>(mesh.GetNodes());
   ParFiniteElementSpace &mesh_fes = *mesh_nodes->ParFESpace();

   H1_FECollection h1fec(polynomial_order, dim);
   ParFiniteElementSpace h1fes(&mesh, &h1fec);

   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   Array<int> ess_tdof;
   ess_bdr = 1;
   h1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof);

   const IntegrationRule &ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(), ir_order * h1fec.GetOrder());

   ParGridFunction u(&h1fes);

   // ParBilinearForm m_form(&h1fes);
   // auto m_integ = new MassIntegrator;
   // m_integ->SetIntegrationRule(ir);
   // m_form.AddDomainIntegrator(m_integ);
   // m_form.Assemble();
   // m_form.Finalize();
   // auto M_mat = m_form.ParallelAssemble();
   ParBilinearForm m_form(&h1fes);
   auto m_integ = new DiffusionIntegrator;
   m_integ->SetIntegrationRule(ir);
   m_form.AddDomainIntegrator(m_integ);
   m_form.Assemble();
   m_form.Finalize();
   auto M_mat = m_form.ParallelAssemble();

   // auto mass_kernel = [](const double &u, const tensor<double, 2, 2> &J,
   //                       const double &w)
   // {
   //    return std::tuple{u * det(J) * w};
   // };

   // std::tuple argument_operators{Value{"potential"}, Gradient{"coordinates"}, Weight{"integration_weights"}};
   // std::tuple output_operator{Value{"potential"}};

   // ElementOperator op{mass_kernel, argument_operators, output_operator};

   // std::array solutions{FieldDescriptor{&h1fes, "potential"}};
   // std::array parameters{FieldDescriptor{&mesh_fes, "coordinates"}};

   // DifferentiableOperator dop{solutions, parameters, std::tuple{op}, mesh, ir};

   // Vector x(h1fes.GetTrueVSize()), y1(h1fes.GetTrueVSize()),
   //        y2(h1fes.GetTrueVSize());

   // x = 1.0;
   // M_mat->Mult(x, y1);

   // dop.SetParameters({mesh_nodes});
   // dop.Mult(x, y2);

   // y2 -= y1;
   // if (y2.Norml2() > 1e-12)
   // {
   //    return 1;
   // }

   // M_mat->PrintMatlab(out);

   auto diffusion_kernel = [](tensor<double, 2> &dudxi,
                              const tensor<double, 2, 2> &J,
                              const double &w)
   {
      return dudxi * det(J) * w * inv(J) * transpose(inv(J));
   };

   std::tuple argument_operators{Gradient{"potential"}, Gradient{"coordinates"}, Weight{"integration_weights"}};
   std::tuple output_operator{Gradient{"potential"}};

   ElementOperator op{diffusion_kernel, argument_operators, output_operator};

   std::array solutions{FieldDescriptor{&h1fes, "potential"}};
   std::array parameters{FieldDescriptor{&mesh_fes, "coordinates"}};

   DifferentiableOperator dop{solutions, parameters, std::tuple{op}, mesh, ir};
   // dop.SetEssentialTrueDofs(ess_tdof);

   auto u_f = [](const Vector &coords)
   {
      const double x = coords(0);
      const double y = coords(1);
      return x + y;
   };

   auto u_coef = FunctionCoefficient(u_f);

   u.ProjectCoefficient(u_coef);
   auto dFdU = dop.GetDerivativeWrt<0>({&u}, {mesh_nodes});

   HypreParMatrix M_mat_dop;
   dFdU->Assemble(M_mat_dop);

   // M_mat_dop.PrintMatlab(out);

   return 0;
}

int main(int argc, char *argv[])
{
   Mpi::Init();

   std::cout << std::setprecision(9);

   const char *mesh_file = "../data/star.mesh";
   int polynomial_order = 1;
   int ir_order = 2;
   int refinements = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&polynomial_order, "-o", "--order", "");
   args.AddOption(&refinements, "-r", "--r", "");
   args.AddOption(&ir_order, "-iro", "--iro", "");
   args.ParseCheck();

   out << std::setprecision(12);

   int ret;

   // ret = test_interpolate_linear_scalar(mesh_file, refinements, polynomial_order);
   // out << "test_interpolate_linear_scalar";
   // ret ? out << " FAILURE\n" : out << " OK\n";

   // ret = test_interpolate_gradient_scalar(mesh_file, refinements,
   //                                        polynomial_order);
   // out << "test_interpolate_gradient_scalar";
   // ret ? out << " FAILURE\n" : out << " OK\n";

   // ret = test_interpolate_linear_vector(mesh_file, refinements, polynomial_order);
   // out << "test_interpolate_linear_vector";
   // ret ? out << " FAILURE\n" : out << " OK\n";

   // ret = test_interpolate_gradient_vector(mesh_file,
   //                                        refinements,
   //                                        polynomial_order);
   // out << "test_interpolate_gradient_vector";
   // ret ? out << " FAILURE\n" : out << " OK\n";

   // ret = test_domain_lf_integrator(mesh_file,
   //                                 refinements,
   //                                 polynomial_order);
   // out << "test_domain_lf_integrator";
   // ret ? out << " FAILURE\n" : out << " OK\n";

   // ret = test_diffusion_integrator(mesh_file,
   //                                 refinements,
   //                                 polynomial_order);
   // out << "test_diffusion_integrator";
   // ret ? out << " FAILURE\n" : out << " OK\n";

   // ret = test_qoi(mesh_file, refinements, polynomial_order);
   // out << "test_qoi";
   // ret ? out << " FAILURE\n" : out << " OK\n";

   ret = test_assemble_mass_hypreparmatrix(mesh_file, refinements,
                                           polynomial_order, ir_order);
   out << "test_assemble_mass_hypreparmatrix";
   ret ? out << " FAILURE\n" : out << " OK\n";

   return 0;
}
