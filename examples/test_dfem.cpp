#include "dfem/dfem.hpp"
#include "examples/dfem/dfem_util.hpp"
#include "fem/bilinearform.hpp"
#include "fem/coefficient.hpp"
#include "fem/lininteg.hpp"

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
//       // out << "u: " << u << "\n";
//       // out << "J: " << J << "\n";
//       return mfem::tuple{u * det(J) * w};
//    };

//    mfem::tuple argument_operators = {Value{"potential"}, Gradient{"coordinates"}, Weight{}};
//    mfem::tuple output_operator = {Value{"potential"}};

//    ElementOperator eop{kernel, argument_operators, output_operator};
//    auto ops = mfem::tuple{eop};

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

// int test_boundary_lf_integrator(std::string mesh_file,
//                                 int refinements,
//                                 int polynomial_order)
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
//    ParFiniteElementSpace h1vfes(&mesh, &h1fec, dim);

//    const IntegrationRule &ir_face =
//       IntRules.Get(h1fes.GetBE(0)->GetGeomType(), 2 * h1fec.GetOrder());

//    ParGridFunction f1_g(&h1fes);
//    ParGridFunction f1_vg(&h1vfes);

//    // {
//    //    auto kernel = [](const double &u,
//    //                     const tensor<double, 2> &x,
//    //                     const tensor<double, 2, 1> &J,
//    //                     const double &w)
//    //    {
//    //       // tensor<double, 2, 1> normal = ortho(J);
//    //       return std::tuple{u * norm(J) * w};
//    //    };

//    //    std::tuple argument_operators = {Value{"potential"}, Value{"coordinates"}, Gradient{"coordinates"}, Weight{}};
//    //    std::tuple output_operator = {Value{"potential"}};

//    //    BoundaryElementOperator eop{kernel, argument_operators, output_operator};
//    //    auto ops = std::tuple{eop};

//    //    auto solutions = std::array{FieldDescriptor{&h1fes, "potential"}};
//    //    auto parameters = std::array{FieldDescriptor{&mesh_fes, "coordinates"}};

//    //    DifferentiableOperator dop(solutions, parameters, ops, mesh, ir_face);

//    //    auto f1 = [](const Vector &coords)
//    //    {
//    //       const double x = coords(0);
//    //       const double y = coords(1);
//    //       return 2.345 + x*x + x*y*y;
//    //    };

//    //    FunctionCoefficient f1_c(f1);
//    //    f1_g.ProjectCoefficient(f1_c);

//    //    GridFunctionCoefficient f1_gfc(&f1_g);

//    //    Vector x(f1_g), y(h1fes.TrueVSize());
//    //    dop.SetParameters({mesh_nodes});
//    //    dop.Mult(x, y);

//    //    ParLinearForm b(&h1fes);
//    //    auto integ = new BoundaryLFIntegrator(f1_gfc);
//    //    integ->SetIntRule(&ir_face);
//    //    b.AddBoundaryIntegrator(integ);
//    //    b.Assemble();

//    //    b -= y;
//    //    if (b.Norml2() > 1e-10)
//    //    {
//    //       out << "||u - u_ex||_l2 = " << b.Norml2() << "\n";
//    //       return 1;
//    //    }
//    // }

//    {
//       auto kernel = [](const double &u,
//                        const tensor<double, 2> &g,
//                        const tensor<double, 2> &coords,
//                        const tensor<double, 2, 1> &J,
//                        const tensor<double, 2> &n,
//                        const double &w)
//       {
//          const double x = coords(0);
//          const double y = coords(1);
//          return std::tuple{dot(g, n) * norm(J) * w};
//       };

//       std::tuple argument_operators =
//       {
//          Value{"potential"},
//          Value{"field"},
//          Value{"coordinates"},
//          Gradient{"coordinates"},
//          FaceNormal{"coordinates"},
//          Weight{}
//       };
//       std::tuple output_operator = {Value{"potential"}};

//       BoundaryElementOperator eop{kernel, argument_operators, output_operator};
//       auto ops = std::tuple{eop};

//       auto solutions = std::array{FieldDescriptor{&h1fes, "potential"}};
//       auto parameters = std::array{FieldDescriptor{&mesh_fes, "coordinates"}, FieldDescriptor{&h1vfes, "field"}};

//       DifferentiableOperator dop(solutions, parameters, ops, mesh, ir_face);

//       auto f1 = [](const Vector &coords, Vector &u)
//       {
//          const double x = coords(0);
//          const double y = coords(1);
//          u(0) = 2.345 + x*x + x*y*y;
//          u(1) = 2.345 + x*x + x*y*y;
//       };

//       VectorFunctionCoefficient f1_c(dim, f1);
//       f1_vg.ProjectCoefficient(f1_c);

//       VectorGridFunctionCoefficient f1_gfc(&f1_vg);

//       Vector x(f1_g), y(h1fes.TrueVSize());
//       dop.SetParameters({mesh_nodes, &f1_vg});
//       dop.Mult(x, y);

//       ParLinearForm b(&h1fes);
//       auto integ = new BoundaryNormalLFIntegrator(f1_gfc);
//       integ->SetIntRule(&ir_face);
//       b.AddBoundaryIntegrator(integ);
//       b.Assemble();

//       b -= y;
//       if (b.Norml2() > 1e-10)
//       {
//          out << "BoundaryNormalLFIntegrator\n";
//          out << "||u - u_ex||_l2 = " << b.Norml2() << "\n";
//          return 1;
//       }
//    }

//    return 0;
// }

int test_diffusion_integrator(std::string mesh_file,
                              int refinements,
                              int polynomial_order)
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

   out << "#el: " << mesh.GetNE() << "\n";

   ParGridFunction* mesh_nodes = static_cast<ParGridFunction *>(mesh.GetNodes());
   ParFiniteElementSpace &mesh_fes = *mesh_nodes->ParFESpace();

   H1_FECollection h1fec(polynomial_order, dim);
   ParFiniteElementSpace h1fes(&mesh, &h1fec);

   out << "#dofs " << h1fes.GetTrueVSize() << "\n";

   const IntegrationRule &ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder());

   out << "#qp: " << ir.GetNPoints() << "\n";

   ParGridFunction f1_g(&h1fes);
   ParGridFunction rho_g(&h1fes);

   auto rho_f = [](const Vector &coords)
   {
      const double x = coords(0);
      const double y = coords(1);
      return x + y;
   };

   FunctionCoefficient rho_c(rho_f);
   rho_g.ProjectCoefficient(rho_c);

   auto kernel = [] MFEM_HOST_DEVICE
                 (const tensor<double, 2, 2> &J,
                  const double &w,
                  const tensor<double, 2> &dudxi)
   {
      auto invJ = inv(J);
      return mfem::tuple{dudxi * invJ * transpose(invJ) * det(J) * w};
   };

   mfem::tuple argument_operators = {Gradient{"coordinates"}, Weight{}, Gradient{"potential"}};
   mfem::tuple output_operator = {Gradient{"potential"}};

   ElementOperator eop = {kernel, argument_operators, output_operator};
   auto ops = mfem::tuple{eop};

   auto solutions = std::array{FieldDescriptor{&h1fes, "potential"}};
   auto parameters = std::array
   {
      FieldDescriptor{&mesh_fes, "coordinates"}
   };

   DifferentiableOperator dop(solutions, parameters, ops, mesh, ir);

   auto f1 = [](const Vector &coords)
   {
      const double x = coords(0);
      const double y = coords(1);
      return 2.345 + 0.25 * x*x*y + y*y*x;
   };

   FunctionCoefficient f1_c(f1);
   f1_g.ProjectCoefficient(f1_c);

   Vector x(f1_g), y(h1fes.TrueVSize());
   dop.SetParameters({mesh_nodes});
   dop.Mult(x, y);
   y.HostRead();
   print_vector(y);

   ParBilinearForm a(&h1fes);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a.Assemble();
   a.Finalize();

   Vector y2(h1fes.TrueVSize());
   a.Mult(x, y2);
   y2.HostRead();
   print_vector(y2);
   y2 -= y;
   if (y2.Norml2() > 1e-10)
   {
      out << "||F(u) - ex||_l2 = " << y2.Norml2() << "\n";
      return 1;
   }

   // Test linearization here as well
   auto dFdu = dop.GetDerivativeWrt<0>({&f1_g}, {mesh_nodes});

   if (dFdu->Height() != h1fes.GetTrueVSize())
   {
      out << "dFdu unexpected height of " << dFdu->Height() << "\n";
      return 1;
   }

   dFdu->Mult(x, y);
   a.Mult(x, y2);
   y2 -= y;
   out << "||dFdu x - A x||_l2 = " << y2.Norml2() << "\n";
   if (y2.Norml2() > 1e-10)
   {
      out << "||dFdu u^* - ex||_l2 = " << y2.Norml2() << "\n";
      return 1;
   }

   // // fd jacobian test
   // {
   //    double eps = 1.0e-6;
   //    Vector v(x), xpv(x), xmv(x), fxpv(x.Size()), fxmv(x.Size());
   //    v *= eps;
   //    xpv += v;
   //    xmv -= v;
   //    dop.Mult(xpv, fxpv);
   //    dop.Mult(xmv, fxmv);
   //    fxpv -= fxmv;
   //    fxpv /= (2.0*eps);

   //    fxpv -= y;
   //    if (fxpv.Norml2() > eps)
   //    {
   //       out << "||dFdu_FD u^* - ex||_l2 = " << fxpv.Norml2() << "\n";
   //       return 1;
   //    }
   // }

   // f1_g.ProjectCoefficient(f1_c);
   // rho_g.ProjectCoefficient(rho_c);
   // auto dFdrho = dop.GetDerivativeWrt<1>({&f1_g}, {&rho_g, mesh_nodes});
   // if (dFdrho->Height() != h1fes.GetTrueVSize())
   // {
   //    out << "dFdrho unexpected height of " << dFdrho->Height() << "\n";
   //    return 1;
   // }

   // dFdrho->Mult(rho_g, y);

   // // fd test
   // {
   //    double eps = 1.0e-6;
   //    Vector v(rho_g), rhopv(rho_g), rhomv(rho_g), frhopv(x.Size()), frhomv(x.Size());
   //    v *= eps;
   //    rhopv += v;
   //    rhomv -= v;
   //    dop.SetParameters({&rhopv, mesh_nodes});
   //    dop.Mult(x, frhopv);
   //    dop.SetParameters({&rhomv, mesh_nodes});
   //    dop.Mult(x, frhomv);
   //    frhopv -= frhomv;
   //    frhopv /= (2.0*eps);

   //    frhopv -= y;
   //    if (frhopv.Norml2() > eps)
   //    {
   //       out << "||dFdu_FD u^* - ex||_l2 = " << frhopv.Norml2() << "\n";
   //       return 1;
   //    }
   // }

   return 0;
}

// int test_cuda(std::string mesh_file,
//               int refinements,
//               int polynomial_order)
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

//    out << "#el: " << mesh.GetNE() << "\n";

//    ParGridFunction *mesh_nodes = static_cast<ParGridFunction *>(mesh.GetNodes());
//    ParFiniteElementSpace &mesh_fes = *mesh_nodes->ParFESpace();

//    H1_FECollection h1fec(polynomial_order, dim);
//    ParFiniteElementSpace h1fes(&mesh, &h1fec);

//    out << "#dofs " << h1fes.GetTrueVSize() << "\n";

//    const IntegrationRule &ir =
//        IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder());

//    out << "#qp: " << ir.GetNPoints() << "\n";

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

//    auto kernel = [] MFEM_HOST_DEVICE(
//       const double &u,
//       const tensor<double, 2, 2> &J,
//       const double &w)
//    {
//       return mfem::tuple{u * det(J) * w};
//    };
//    mfem::tuple argument_operators = {Value{"potential"}, Gradient{"coordinates"}, Weight{}};
//    mfem::tuple output_operator = {Value{"potential"}};

//    ElementOperator eop = {kernel, argument_operators, output_operator};
//    auto ops = mfem::tuple{eop};

//    auto solutions = std::array{FieldDescriptor{&h1fes, "potential"}};
//    // std::array<FieldDescriptor, 0> parameters;
//    auto parameters = std::array{FieldDescriptor{&mesh_fes, "coordinates"}};

//    DifferentiableOperator dop(solutions, parameters, ops, mesh, ir);

//    auto f1 = [](const Vector &coords)
//    {
//       const double x = coords(0);
//       const double y = coords(1);
//       return 2.345 + 0.25 * x * x * y + y * y * x;
//    };

//    FunctionCoefficient f1_c(f1);
//    f1_g.ProjectCoefficient(f1_c);

//    Vector x(f1_g), y(h1fes.TrueVSize());
//    dop.SetParameters({mesh_nodes});
//    dop.Mult(x, y);

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
//    ParFiniteElementSpace h1fes(&mesh, &h1fec, dim);

//    const IntegrationRule &ir =
//       IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder());

//    ParGridFunction rho_g(&h1fes);

//    auto rho_f = [](const Vector &coords, Vector &u)
//    {
//       const double x = coords(0);
//       const double y = coords(1);
//       u(0) = x + y;
//       u(1) = x + y;
//    };

//    VectorFunctionCoefficient rho_c(dim, rho_f);
//    rho_g.ProjectCoefficient(rho_c);

//    auto kernel = [](const tensor<double, 2> &rho,
//                     const tensor<double, 2, 2> &drhodxi,
//                     const tensor<double, 2, 2> &J,
//                     const double &w)
//    {
//       const double eps = 1.2345;
//       const auto drhodx = drhodxi * inv(J);
//       return std::tuple{(0.5 * eps * dot(rho, rho) + ddot(drhodx, drhodx)) * det(J) * w};
//    };

//    std::tuple argument_operators = {Value{"density"}, Gradient{"density"}, Gradient{"coordinates"}, Weight{}};
//    std::tuple output_operator = {One{"density"}};

//    ElementOperator eop = {kernel, argument_operators, output_operator};
//    auto ops = std::tuple{eop};

//    auto solutions = std::array{FieldDescriptor{&h1fes, "density"}};
//    auto parameters = std::array{FieldDescriptor{&mesh_fes, "coordinates"}};

//    DifferentiableOperator dop(solutions, parameters, ops, mesh, ir);

//    Vector x(rho_g), y(1);
//    dop.SetParameters({mesh_nodes});
//    dop.Mult(x, y);

//    auto dFdrho = dop.GetDerivativeWrt<0>({&rho_g}, {mesh_nodes});
//    Vector dFdrho_vec;
//    dFdrho->Assemble(dFdrho_vec);

//    // // fd jacobian test
//    // {
//    //    double eps = 1.0e-8;
//    //    Vector v(x), fxpv(1), fxmv(1), dfdx(x.Size());
//    //    for (int i = 0; i < x.Size(); i++)
//    //    {
//    //       v(i) += eps;
//    //       dop.Mult(v, fxpv);
//    //       v(i) -= 2.0 * eps;
//    //       dop.Mult(v, fxmv);
//    //       fxpv -= fxmv;
//    //       fxpv /= (2.0*eps);
//    //       dfdx(i) = fxpv(0);
//    //    }

//    //    // print_vector(dfdx);
//    //    dfdx -= dFdrho_vec;
//    //    if (dfdx.Norml2() > 1e-6)
//    //    {
//    //       out << "||dFdu_FD u^* - ex||_l2 = " << dfdx.Norml2() << "\n";
//    //       return 1;
//    //    }
//    // }

//    return 0;
// }

// int test_assemble_mass_hypreparmatrix(std::string mesh_file,
//                                       int refinements,
//                                       int polynomial_order,
//                                       int ir_order)
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

//    Array<int> ess_bdr(mesh.bdr_attributes.Max());
//    Array<int> ess_tdof;
//    ess_bdr = 1;
//    h1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof);

//    const IntegrationRule &ir =
//       IntRules.Get(h1fes.GetFE(0)->GetGeomType(), ir_order * h1fec.GetOrder());

//    ParGridFunction u(&h1fes);

//    ParBilinearForm m_form(&h1fes);
//    auto m_integ = new MassIntegrator;
//    m_integ->SetIntegrationRule(ir);
//    m_form.AddDomainIntegrator(m_integ);
//    m_form.Assemble();
//    m_form.Finalize();
//    auto M_mat = m_form.ParallelAssemble();
//    // ParBilinearForm m_form(&h1fes);
//    // auto m_integ = new DiffusionIntegrator;
//    // m_integ->SetIntegrationRule(ir);
//    // m_form.AddDomainIntegrator(m_integ);
//    // m_form.Assemble();
//    // m_form.Finalize();
//    // auto M_mat = m_form.ParallelAssemble();

//    auto mass_kernel = [](const double &u, const tensor<double, 2, 2> &J,
//                          const double &w)
//    {
//       return u * det(J) * w;
//    };

//    std::tuple argument_operators{Value{"potential"}, Gradient{"coordinates"}, Weight{"integration_weights"}};
//    std::tuple output_operator{Value{"potential"}};

//    ElementOperator op{mass_kernel, argument_operators, output_operator};

//    std::array solutions{FieldDescriptor{&h1fes, "potential"}};
//    std::array parameters{FieldDescriptor{&mesh_fes, "coordinates"}};

//    DifferentiableOperator dop{solutions, parameters, std::tuple{op}, mesh, ir};

//    Vector x(h1fes.GetTrueVSize()), y1(h1fes.GetTrueVSize()),
//           y2(h1fes.GetTrueVSize());

//    x = 1.0;
//    M_mat->Mult(x, y1);

//    dop.SetParameters({mesh_nodes});
//    dop.Mult(x, y2);

//    y2 -= y1;
//    if (y2.Norml2() > 1e-12)
//    {
//       return 1;
//    }

//    // M_mat->PrintMatlab(out);

//    // auto diffusion_kernel = [](tensor<double, 2> &dudxi,
//    //                            const tensor<double, 2, 2> &J,
//    //                            const double &w)
//    // {
//    //    return dudxi * det(J) * w * inv(J) * transpose(inv(J));
//    // };

//    // std::tuple argument_operators{Gradient{"potential"}, Gradient{"coordinates"}, Weight{"integration_weights"}};
//    // std::tuple output_operator{Gradient{"potential"}};

//    // ElementOperator op{diffusion_kernel, argument_operators, output_operator};

//    // std::array solutions{FieldDescriptor{&h1fes, "potential"}};
//    // std::array parameters{FieldDescriptor{&mesh_fes, "coordinates"}};

//    // DifferentiableOperator dop{solutions, parameters, std::tuple{op}, mesh, ir};
//    // // dop.SetEssentialTrueDofs(ess_tdof);

//    // auto u_f = [](const Vector &coords)
//    // {
//    //    const double x = coords(0);
//    //    const double y = coords(1);
//    //    return x + y;
//    // };

//    // auto u_coef = FunctionCoefficient(u_f);

//    // u.ProjectCoefficient(u_coef);
//    u = 0.0;
//    auto dFdU = dop.GetDerivativeWrt<0>({&u}, {mesh_nodes});

//    HypreParMatrix M_mat_dop;
//    dFdU->Assemble(M_mat_dop);

//    // M_mat_dop.PrintMatlab(out);

//    auto res = Add(1.0, *M_mat, -1.0, M_mat_dop);
//    SparseMatrix diag;
//    res->GetDiag(diag);
//    if (diag.MaxNorm() > 1e-12)
//    {
//       res->PrintMatlab(out);
//       out << "mfem assembled hypreparmatrix != dfem assembled hypreparmatrix" <<
//           std::endl;
//       return 1;
//    }

//    return 0;
// }

// int test_assemble_vector_mass_hypreparmatrix(std::string mesh_file,
//                                              int refinements,
//                                              int polynomial_order,
//                                              int ir_order)
// {
//    Mesh mesh_serial = Mesh(mesh_file);
//    for (int i = 0; i < refinements; i++)
//    {
//       mesh_serial.UniformRefinement();
//    }
//    ParMesh mesh(MPI_COMM_WORLD, mesh_serial);

//    mesh.SetCurvature(1);
//    const int dim = mesh.Dimension();
//    const int vdim = dim;
//    mesh_serial.Clear();

//    ParGridFunction* mesh_nodes = static_cast<ParGridFunction *>(mesh.GetNodes());
//    ParFiniteElementSpace &mesh_fes = *mesh_nodes->ParFESpace();

//    H1_FECollection h1fec(polynomial_order, dim);
//    ParFiniteElementSpace h1fes(&mesh, &h1fec, vdim);

//    Array<int> ess_bdr(mesh.bdr_attributes.Max());
//    Array<int> ess_tdof;
//    ess_bdr = 1;
//    h1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof);

//    const IntegrationRule &ir =
//       IntRules.Get(h1fes.GetFE(0)->GetGeomType(), ir_order * h1fec.GetOrder());

//    ParGridFunction u(&h1fes);

//    ParBilinearForm m_form(&h1fes);
//    auto m_integ = new VectorMassIntegrator;
//    m_integ->SetVDim(vdim);
//    m_integ->SetIntegrationRule(ir);
//    m_form.AddDomainIntegrator(m_integ);
//    m_form.Assemble();
//    m_form.Finalize();
//    auto M_mat = m_form.ParallelAssemble();

//    auto mass_kernel = [](const tensor<double, 2> &u, const tensor<double, 2, 2> &J,
//                          const double &w)
//    {
//       return std::tuple{u * det(J) * w};
//    };

//    std::tuple argument_operators{Value{"potential"}, Gradient{"coordinates"}, Weight{}};
//    std::tuple output_operator{Value{"potential"}};

//    ElementOperator op{mass_kernel, argument_operators, output_operator};

//    std::array solutions{FieldDescriptor{&h1fes, "potential"}};
//    std::array parameters{FieldDescriptor{&mesh_fes, "coordinates"}};

//    DifferentiableOperator dop{solutions, parameters, std::tuple{op}, mesh, ir};

//    Vector x(h1fes.GetTrueVSize()), y1(h1fes.GetTrueVSize()),
//           y2(h1fes.GetTrueVSize());

//    x = 1.0;
//    M_mat->Mult(x, y1);

//    dop.SetParameters({mesh_nodes});
//    dop.Mult(x, y2);

//    y2 -= y1;
//    if (y2.Norml2() > 1e-12)
//    {
//       return 1;
//    }

//    // M_mat->PrintMatlab(out);
//    // std::ofstream mmatofs("mfem_mat.dat");
//    // M_mat->PrintMatlab(mmatofs);

//    u = 0.0;
//    auto dFdU = dop.GetDerivativeWrt<0>({&u}, {mesh_nodes});

//    HypreParMatrix M_mat_dop;
//    dFdU->Assemble(M_mat_dop);

//    // M_mat_dop.PrintMatlab(out);
//    // std::ofstream mmatdopofs("dfem_mat.dat");
//    // M_mat_dop.PrintMatlab(mmatdopofs);

//    auto res = Add(1.0, *M_mat, -1.0, M_mat_dop);
//    SparseMatrix diag;
//    res->GetDiag(diag);
//    if (diag.MaxNorm() > 1e-12)
//    {
//       // res->PrintMatlab(out);
//       out << "mfem assembled hypreparmatrix != dfem assembled hypreparmatrix" <<
//           std::endl;
//       return 1;
//    }

//    return 0;
// }

// int test_assemble_vector_diffusion_hypreparmatrix(std::string mesh_file,
//                                                   int refinements,
//                                                   int polynomial_order,
//                                                   int ir_order)
// {
//    Mesh mesh_serial = Mesh(mesh_file);
//    for (int i = 0; i < refinements; i++)
//    {
//       mesh_serial.UniformRefinement();
//    }
//    ParMesh mesh(MPI_COMM_WORLD, mesh_serial);

//    mesh.SetCurvature(1);
//    const int dim = mesh.Dimension();
//    const int vdim = dim;
//    mesh_serial.Clear();

//    ParGridFunction* mesh_nodes = static_cast<ParGridFunction *>(mesh.GetNodes());
//    ParFiniteElementSpace &mesh_fes = *mesh_nodes->ParFESpace();

//    H1_FECollection h1fec(polynomial_order, dim);
//    ParFiniteElementSpace h1fes(&mesh, &h1fec, vdim);

//    Array<int> ess_bdr(mesh.bdr_attributes.Max());
//    Array<int> ess_tdof;
//    ess_bdr = 1;
//    h1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof);

//    const IntegrationRule &ir =
//       IntRules.Get(h1fes.GetFE(0)->GetGeomType(), ir_order * h1fec.GetOrder());

//    ParGridFunction u(&h1fes);

//    ParBilinearForm A_form(&h1fes);
//    auto A_integ = new VectorDiffusionIntegrator(vdim);
//    A_integ->SetIntegrationRule(ir);
//    A_form.AddDomainIntegrator(A_integ);
//    A_form.Assemble();
//    A_form.Finalize();
//    auto A_mat = A_form.ParallelAssemble();

//    auto vector_diffusion_kernel = [](const tensor<double, 2, 2> &dudxi,
//                                      const tensor<double, 2, 2> &J,
//                                      const double &w)
//    {
//       return std::tuple{transpose((dudxi * inv(J)) * det(J) * w * transpose(inv(J)))};
//    };

//    std::tuple argument_operators{Gradient{"potential"}, Gradient{"coordinates"}, Weight{}};
//    std::tuple output_operator{Gradient{"potential"}};

//    ElementOperator op{vector_diffusion_kernel, argument_operators, output_operator};

//    std::array solutions{FieldDescriptor{&h1fes, "potential"}};
//    std::array parameters{FieldDescriptor{&mesh_fes, "coordinates"}};

//    DifferentiableOperator dop{solutions, parameters, std::tuple{op}, mesh, ir};

//    Vector x(h1fes.GetTrueVSize()), y1(h1fes.GetTrueVSize()),
//           y2(h1fes.GetTrueVSize());

//    // A_mat->PrintMatlab(out);
//    // std::ofstream amatofs("mfem_mat.dat");
//    // A_mat->PrintMatlab(amatofs);
//    // amatofs.close();

//    u = 0.0;
//    auto dFdU = dop.GetDerivativeWrt<0>({&u}, {mesh_nodes});

//    HypreParMatrix A_mat_dop;
//    dFdU->Assemble(A_mat_dop);

//    // A_mat_dop.PrintMatlab(out);
//    // std::ofstream mmatdopofs("dfem_mat.dat");
//    // A_mat_dop.PrintMatlab(mmatdopofs);

//    auto res = Add(1.0, *A_mat, -1.0, A_mat_dop);
//    SparseMatrix diag;
//    res->GetDiag(diag);
//    if (diag.MaxNorm() > 1e-12)
//    {
//       // res->PrintMatlab(out);
//       out << "mfem assembled hypreparmatrix != dfem assembled hypreparmatrix" <<
//           std::endl;
//       return 1;
//    }

//    return 0;
// }

// int test_assemble_elasticity_hypreparmatrix(std::string mesh_file,
//                                             int refinements,
//                                             int polynomial_order,
//                                             int ir_order)
// {
//    Mesh mesh_serial = Mesh(mesh_file);
//    for (int i = 0; i < refinements; i++)
//    {
//       mesh_serial.UniformRefinement();
//    }
//    ParMesh mesh(MPI_COMM_WORLD, mesh_serial);

//    mesh.SetCurvature(1);
//    const int dim = mesh.Dimension();
//    const int vdim = dim;
//    mesh_serial.Clear();

//    ParGridFunction* mesh_nodes = static_cast<ParGridFunction *>(mesh.GetNodes());
//    ParFiniteElementSpace &mesh_fes = *mesh_nodes->ParFESpace();

//    H1_FECollection h1fec(polynomial_order, dim);
//    ParFiniteElementSpace h1fes(&mesh, &h1fec, vdim);

//    Array<int> ess_bdr(mesh.bdr_attributes.Max());
//    Array<int> ess_tdof;
//    ess_bdr = 1;
//    h1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof);

//    const IntegrationRule &ir =
//       IntRules.Get(h1fes.GetFE(0)->GetGeomType(), ir_order * h1fec.GetOrder());

//    ParGridFunction u(&h1fes);

//    ConstantCoefficient l_coeff(1.0), m_coeff(1.0);

//    ParBilinearForm A_form(&h1fes);
//    auto A_integ = new ElasticityIntegrator(l_coeff, m_coeff);
//    A_integ->SetIntegrationRule(ir);
//    A_form.AddDomainIntegrator(A_integ);
//    A_form.Assemble();
//    A_form.Finalize();
//    auto A_mat = A_form.ParallelAssemble();

//    // A_mat->PrintMatlab(out);
//    std::ofstream amatofs("mfem_mat.dat");
//    A_mat->PrintMatlab(amatofs);
//    amatofs.close();

//    auto elasticity_kernel = [](const tensor<double, 2, 2> &dudxi,
//                                const tensor<double, 2, 2> &J,
//                                const double &w)
//    {
//       constexpr double lambda = 1.0;
//       constexpr double mu = 1.0;
//       static constexpr auto I = mfem::internal::IsotropicIdentity<2>();
//       auto eps = sym(dudxi * inv(J));
//       return std::tuple{transpose((lambda * tr(eps) * I + 2.0 * mu * eps) * det(J) * w * transpose(inv(J)))};
//    };

//    std::tuple argument_operators{Gradient{"displacement"}, Gradient{"coordinates"}, Weight{}};
//    std::tuple output_operator{Gradient{"displacement"}};

//    ElementOperator op{elasticity_kernel, argument_operators, output_operator};

//    std::array solutions{FieldDescriptor{&h1fes, "displacement"}};
//    std::array parameters{FieldDescriptor{&mesh_fes, "coordinates"}};

//    DifferentiableOperator dop{solutions, parameters, std::tuple{op}, mesh, ir};

//    Vector x(h1fes.GetTrueVSize()), y1(h1fes.GetTrueVSize()),
//           y2(h1fes.GetTrueVSize());

//    u = 1.0;
//    auto dFdU = dop.GetDerivativeWrt<0>({&u}, {mesh_nodes});

//    HypreParMatrix A_mat_dop;
//    dFdU->Assemble(A_mat_dop);

//    // A_mat_dop.PrintMatlab(out);
//    std::ofstream amatdopofs("dfem_mat.dat");
//    A_mat_dop.PrintMatlab(amatdopofs);
//    amatdopofs.close();

//    auto res = Add(1.0, *A_mat, -1.0, A_mat_dop);
//    SparseMatrix diag;
//    res->GetDiag(diag);
//    if (diag.MaxNorm() > 1e-12)
//    {
//       // res->PrintMatlab(out);
//       out << "mfem assembled hypreparmatrix != dfem assembled hypreparmatrix" <<
//           std::endl;
//       return 1;
//    }

//    return 0;
// }

// int test_elasticity_integrator(std::string mesh_file,
//                                int refinements,
//                                int polynomial_order,
//                                int ir_order)
// {
//    Mesh mesh_serial = Mesh(mesh_file);
//    for (int i = 0; i < refinements; i++)
//    {
//       mesh_serial.UniformRefinement();
//    }
//    ParMesh mesh(MPI_COMM_WORLD, mesh_serial);

//    mesh.SetCurvature(1);
//    const int dim = mesh.Dimension();
//    const int vdim = dim;
//    mesh_serial.Clear();

//    ParGridFunction* mesh_nodes = static_cast<ParGridFunction *>(mesh.GetNodes());
//    ParFiniteElementSpace &mesh_fes = *mesh_nodes->ParFESpace();

//    H1_FECollection h1fec(polynomial_order, dim);
//    ParFiniteElementSpace h1fes(&mesh, &h1fec, vdim);

//    Array<int> ess_bdr(mesh.bdr_attributes.Max());
//    Array<int> ess_tdof;
//    ess_bdr = 1;
//    h1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof);

//    const IntegrationRule &ir =
//       IntRules.Get(h1fes.GetFE(0)->GetGeomType(), ir_order * h1fec.GetOrder());

//    out << "#qp: " << ir.GetNPoints() << "\n";
//    out << "#dof_el: " << h1fes.GetRestrictionMatrix()->Height() / mesh.GetNE() <<
//        "\n";

//    ParGridFunction u(&h1fes);

//    ConstantCoefficient l_coeff(0.5), m_coeff(0.25);

//    ParBilinearForm A_form(&h1fes);
//    auto A_integ = new ElasticityIntegrator(l_coeff, m_coeff);
//    A_integ->SetIntegrationRule(ir);
//    A_form.AddDomainIntegrator(A_integ);
//    A_form.Assemble();
//    A_form.Finalize();
//    auto A_mat = A_form.ParallelAssemble();

//    // A_mat->PrintMatlab(out);
//    // std::ofstream amatofs("mfem_mat.dat");
//    // A_mat->PrintMatlab(amatofs);
//    // amatofs.close();

//    auto elasticity_kernel = [](const tensor<double, 2, 2> &dudxi,
//                                const tensor<double, 2, 2> &J,
//                                const double &w)
//    {
//       constexpr double lambda = 0.5;
//       constexpr double mu = 0.25;
//       static constexpr auto I = mfem::internal::IsotropicIdentity<2>();
//       auto invJ = inv(J);
//       auto eps = sym(dudxi * invJ);
//       return mfem::tuple{transpose((lambda * tr(eps) * I + 2.0 * mu * eps) * det(J) * w * transpose(invJ))};
//    };

//    mfem::tuple argument_operators{Gradient{"displacement"}, Gradient{"coordinates"}, Weight{}};
//    mfem::tuple output_operator{Gradient{"displacement"}};

//    ElementOperator op{elasticity_kernel, argument_operators, output_operator};

//    std::array solutions{FieldDescriptor{&h1fes, "displacement"}};
//    std::array parameters{FieldDescriptor{&mesh_fes, "coordinates"}};

//    DifferentiableOperator dop{solutions, parameters, mfem::tuple{op}, mesh, ir};

//    Vector x(h1fes.GetTrueVSize()), y1(h1fes.GetTrueVSize()),
//           y2(h1fes.GetTrueVSize());

//    x.Randomize(12345);
//    A_mat->Mult(x, y1);

//    // out << ">>> y1\n";
//    // print_vector(y1);

//    dop.SetParameters({mesh_nodes});
//    dop.Mult(x, y2);

//    // out << ">>> y2\n";
//    // print_vector(y2);

//    y2 -= y1;
//    if (y2.Norml2() > 1e-10)
//    {
//       out << "||u - u_ex||_l2 = " << y2.Norml2() << "\n";
//       return 1;
//    }

//    return 0;
// }


// int test_assemble_mixed_gradient_hypreparmatrix(std::string mesh_file,
//                                                 int refinements,
//                                                 int polynomial_order,
//                                                 int ir_order)
// {
//    Mesh mesh_serial = Mesh(mesh_file);
//    for (int i = 0; i < refinements; i++)
//    {
//       mesh_serial.UniformRefinement();
//    }
//    ParMesh mesh(MPI_COMM_WORLD, mesh_serial);

//    mesh.SetCurvature(1);
//    const int dim = mesh.Dimension();
//    const int vdim = dim;
//    mesh_serial.Clear();

//    ParGridFunction* mesh_nodes = static_cast<ParGridFunction *>(mesh.GetNodes());
//    ParFiniteElementSpace &mesh_fes = *mesh_nodes->ParFESpace();

//    H1_FECollection h1fec(polynomial_order, dim);
//    ParFiniteElementSpace h1fes_vdim(&mesh, &h1fec, vdim);
//    ParFiniteElementSpace h1fes_scalar(&mesh, &h1fec);

//    const IntegrationRule &ir =
//       IntRules.Get(h1fes_vdim.GetFE(0)->GetGeomType(), ir_order * h1fec.GetOrder());

//    ParGridFunction u(&h1fes_vdim);
//    ParGridFunction p(&h1fes_scalar);

//    ParMixedBilinearForm A_form(&h1fes_scalar, &h1fes_vdim);
//    auto A_integ = new GradientIntegrator;
//    A_integ->SetIntegrationRule(ir);
//    A_form.AddDomainIntegrator(A_integ);
//    A_form.Assemble();
//    A_form.Finalize();
//    auto A_mat = A_form.ParallelAssemble();

//    // A_mat->PrintMatlab(out);
//    std::ofstream amatofs("mfem_mat.dat");
//    A_mat->PrintMatlab(amatofs);
//    amatofs.close();

//    auto gradient_kernel = [](const tensor<double, 2> &dpdxi,
//                              const tensor<double, 2, 2> &J,
//                              const double &w)
//    {
//       return dpdxi * inv(J) * det(J) * w;
//    };

//    std::tuple argument_operators{Gradient{"pressure"}, Gradient{"coordinates"}, Weight{"integration_weights"}};
//    std::tuple output_operator{Value{"velocity"}};

//    ElementOperator op{gradient_kernel, argument_operators, output_operator};

//    std::array solutions{FieldDescriptor{&h1fes_scalar, "pressure"}};
//    std::array parameters{FieldDescriptor{&h1fes_vdim, "velocity"}, FieldDescriptor{&mesh_fes, "coordinates"}};

//    DifferentiableOperator dop{solutions, parameters, std::tuple{op}, mesh, ir};

//    auto dFdU = dop.GetDerivativeWrt<0>({&p}, {&u, mesh_nodes});

//    HypreParMatrix A_mat_dop;
//    dFdU->Assemble(A_mat_dop);

//    // A_mat_dop.PrintMatlab(out);
//    std::ofstream amatdopofs("dfem_mat.dat");
//    A_mat_dop.PrintMatlab(amatdopofs);
//    amatdopofs.close();

//    // auto res = Add(1.0, *A_mat, -1.0, A_mat_dop);
//    // SparseMatrix diag;
//    // res->GetDiag(diag);
//    // if (diag.MaxNorm() > 1e-12)
//    // {
//    //    res->PrintMatlab(out);
//    //    out << "mfem assembled hypreparmatrix != dfem assembled hypreparmatrix" <<
//    //        std::endl;
//    //    return 1;
//    // }

//    return 0;
// }

// int test_assemble_nonlinear_diffusion_hypreparmatrix(std::string mesh_file,
//                                                      int refinements,
//                                                      int polynomial_order,
//                                                      int ir_order)
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
//       IntRules.Get(h1fes.GetFE(0)->GetGeomType(), ir_order * h1fec.GetOrder());

//    ParGridFunction u(&h1fes);

//    auto u_f = [](const Vector &coords)
//    {
//       const double x = coords(0);
//       const double y = coords(1);
//       return x + y;
//    };
//    auto u_coef = FunctionCoefficient(u_f);

//    ParBilinearForm m_form(&h1fes);
//    auto m_integ = new DiffusionIntegrator(u_coef);
//    m_integ->SetIntegrationRule(ir);
//    m_form.AddDomainIntegrator(m_integ);
//    m_form.Assemble();
//    m_form.Finalize();
//    auto M_mat = m_form.ParallelAssemble();

//    M_mat->PrintMatlab(out);
//    out << "\n\n";
//    std::ofstream mmatofs("dfem_mat.dat");
//    M_mat->PrintMatlab(mmatofs);
//    mmatofs.close();

//    auto nonlinear_diffusion = [](const double &u,
//                                  const tensor<double, 2> &dudxi,
//                                  const tensor<double, 2, 2> &J,
//                                  const double &w)
//    {
//       auto invJ = inv(J);
//       auto dudx = dudxi * invJ;
//       return u * dudx * det(J) * w * transpose(invJ);
//    };

//    std::tuple argument_operators{Value{"potential"}, Gradient{"potential"}, Gradient{"coordinates"}, Weight{"integration_weights"}};
//    std::tuple output_operator{Gradient{"potential"}};

//    ElementOperator op{nonlinear_diffusion, argument_operators, output_operator};

//    std::array solutions{FieldDescriptor{&h1fes, "potential"}};
//    std::array parameters{FieldDescriptor{&mesh_fes, "coordinates"}};

//    DifferentiableOperator dop{solutions, parameters, std::tuple{op}, mesh, ir};

//    Vector x(h1fes.GetTrueVSize()), y1(h1fes.GetTrueVSize()),
//           y2(h1fes.GetTrueVSize());

//    // u.ProjectCoefficient(u_coef);
//    u = 1.0;
//    auto dFdU = dop.GetDerivativeWrt<0>({&u}, {mesh_nodes});

//    HypreParMatrix M_mat_dop;
//    dFdU->Assemble(M_mat_dop);

//    M_mat_dop.PrintMatlab(out);
//    std::ofstream mmatdopofs("dfem_mat.dat");
//    M_mat_dop.PrintMatlab(mmatofs);
//    mmatofs.close();

//    return 0;
// }

// int test_assemble_mixed_scalar_curl_hypreparmatrix(std::string mesh_file,
//                                                    int refinements,
//                                                    int polynomial_order,
//                                                    int ir_order)
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

//    ND_FECollection ndfec(polynomial_order, dim);
//    ParFiniteElementSpace ndfes(&mesh, &ndfec);

//    const IntegrationRule &ir =
//       IntRules.Get(h1fes.GetFE(0)->GetGeomType(), ir_order * h1fec.GetOrder());

//    ParGridFunction u(&ndfes);
//    ParGridFunction v(&h1fes);

//    auto u_f = [](const Vector &coords, Vector &u)
//    {
//       const double x = coords(0);
//       const double y = coords(1);
//       u(0) = x + y * y;
//       u(1) = y - x;
//    };
//    auto u_coef = VectorFunctionCoefficient(dim, u_f);

//    u.ProjectCoefficient(u_coef);

//    ParMixedBilinearForm blf(&ndfes, &h1fes);
//    auto integ = new MixedScalarCurlIntegrator();
//    integ->SetIntRule(&ir);
//    blf.AddDomainIntegrator(integ);
//    blf.Assemble();
//    blf.Finalize();
//    auto A_mat = blf.ParallelAssemble();

//    A_mat->PrintMatlab(out);

//    auto mixed_scalar_curl = [](const double &curl_u,
//                                const tensor<double, 2, 2> &J,
//                                const double &w)
//    {
//       return std::tuple{curl_u / det(J) * det(J) * w};
//    };

//    std::tuple argument_operators{Curl{"potential_vector"}, Gradient{"coordinates"}, Weight{"integration_weights"}};
//    std::tuple output_operator{Value{"potential_scalar"}};

//    ElementOperator op{mixed_scalar_curl, argument_operators, output_operator};

//    std::array solutions{FieldDescriptor{&ndfes, "potential_vector"}};
//    std::array parameters{FieldDescriptor{&h1fes, "potential_scalar"}, FieldDescriptor{&mesh_fes, "coordinates"}};

//    DifferentiableOperator dop{solutions, parameters, std::tuple{op}, mesh, ir};

//    Vector x(h1fes.GetTrueVSize()), y1(h1fes.GetTrueVSize()),
//           y2(h1fes.GetTrueVSize());

//    auto dFdU = dop.GetDerivativeWrt<0>({&u}, {&v, mesh_nodes});

//    HypreParMatrix A_mat_dop;
//    dFdU->Assemble(A_mat_dop);

//    A_mat_dop.PrintMatlab(out);

//    return 0;
// }

int main(int argc, char *argv[])
{
   Mpi::Init();

   std::cout << std::setprecision(9);

   const char *device_config = "cpu";
   const char *mesh_file = "../data/ref-square.mesh";
   int polynomial_order = 1;
   int ir_order = 2;
   int refinements = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&polynomial_order, "-o", "--order", "");
   args.AddOption(&refinements, "-r", "--r", "");
   args.AddOption(&ir_order, "-iro", "--iro", "");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.ParseCheck();

   Device device(device_config);
   if (Mpi::Root() == 0) { device.Print(); }

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

   // ret = test_boundary_lf_integrator(mesh_file,
   //                                   refinements,
   //                                   polynomial_order);
   // out << "test_boundary_lf_integrator";
   // ret ? out << " FAILURE\n" : out << " OK\n";

   ret = test_diffusion_integrator(mesh_file,
                                   refinements,
                                   polynomial_order);
   out << "test_diffusion_integrator";
   ret ? out << " FAILURE\n" : out << " OK\n";

   // ret = test_cuda(mesh_file,
   //                 refinements,
   //                 polynomial_order);
   // out << "test_cuda";
   // ret ? out << " FAILURE\n" : out << " OK\n";

   // ret = test_elasticity_integrator(mesh_file, refinements,
   //                                  polynomial_order, ir_order);
   // out << "test_elasticity_integrator";
   // ret ? out << " FAILURE\n" : out << " OK\n";

   // ret = test_qoi(mesh_file, refinements, polynomial_order);
   // out << "test_qoi";
   // ret ? out << " FAILURE\n" : out << " OK\n";

   // ret = test_assemble_mass_hypreparmatrix(mesh_file, refinements,
   //                                         polynomial_order, ir_order);
   // out << "test_assemble_mass_hypreparmatrix";
   // ret ? out << " FAILURE\n" : out << " OK\n";

   // ret = test_assemble_vector_mass_hypreparmatrix(mesh_file, refinements,
   //                                                polynomial_order, ir_order);
   // out << "test_assemble_vector_mass_hypreparmatrix";
   // ret ? out << " FAILURE\n" : out << " OK\n";

   // ret = test_assemble_vector_diffusion_hypreparmatrix(mesh_file, refinements,
   //                                                     polynomial_order, ir_order);
   // out << "test_assemble_vector_diffusion_hypreparmatrix";
   // ret ? out << " FAILURE\n" : out << " OK\n";

   // ret = test_assemble_elasticity_hypreparmatrix(mesh_file, refinements,
   //                                               polynomial_order, ir_order);
   // out << "test_assemble_elasticity_hypreparmatrix";
   // ret ? out << " FAILURE\n" : out << " OK\n";

   // ret = test_assemble_mixed_gradient_hypreparmatrix(mesh_file, refinements,
   //                                                   polynomial_order, ir_order);
   // out << "test_assemble_elasticity_hypreparmatrix";
   // ret ? out << " FAILURE\n" : out << " OK\n";

   // ret = test_assemble_nonlinear_diffusion_hypreparmatrix(mesh_file, refinements,
   //                                                        polynomial_order, ir_order);
   // out << "test_assemble_nonlinear_diffusion_hypreparmatrix";
   // ret ? out << " FAILURE\n" : out << " OK\n";

   // ret = test_assemble_mixed_scalar_curl_hypreparmatrix(mesh_file, refinements,
   //                                                      polynomial_order, ir_order);
   // out << "test_assemble_mixed_scalar_curl_hypreparmatrix";
   // ret ? out << " FAILURE\n" : out << " OK\n";

   return 0;
}
