#include <functional>
#include <iostream>
#include <variant>
#include <vector>

#include "dfem.hpp"

using namespace mfem;
using mfem::internal::dual;
using mfem::internal::tensor;

using namespace std;

int test_integrate_boundary()
{
   int polynomial_order = 1;

   Mesh mesh = Mesh::MakeCartesian2D(10, 2, Element::QUADRILATERAL, false, 0.0,
                                     2.0 * M_PI);
   mesh.EnsureNodes();
   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   H1_FECollection h1_fec(polynomial_order);
   ParFiniteElementSpace h1_fes(&pmesh, &h1_fec);

   auto ir_face = const_cast<IntegrationRule *>(
                     &IntRules.Get(mesh.GetBdrElementGeometry(0),
                                   3 * mesh.GetNodes()->FESpace()->GetElementOrder(0) + 1));

   auto h1_prolongation = h1_fes.GetProlongationMatrix();

   ParGridFunction boundary_load(&h1_fes);
   boundary_load = 0.0;

   VectorFunctionCoefficient boundary_load_coeff(2, [](const Vector &x, Vector &u)
   {
      u(0) = 0.0;
      u(1) = 1.0;
   });

   {
      Array<int> boundary_load_attr(pmesh.bdr_attributes.Max());
      boundary_load_attr = 0;
      boundary_load_attr[2] = 1;
      boundary_load.ProjectBdrCoefficient(boundary_load_coeff, boundary_load_attr);
   }

   Vector boundary_load_qp;
   interpolate_boundary(boundary_load, *ir_face, boundary_load_qp);

   auto foo = Reshape(boundary_load_qp.Read(), h1_fes.GetVDim(),
                      ir_face->GetNPoints(), pmesh.GetNBE());

   Vector vec(h1_fes.GetVDim());
   for (int e = 0; e < pmesh.GetNBE(); e++)
   {
      auto Tr = pmesh.GetBdrElementTransformation(e);

      for (int qp = 0; qp < ir_face->GetNPoints(); qp++)
      {
         const IntegrationPoint &ip = ir_face->IntPoint(qp);

         Tr->SetIntPoint(&ip);

         boundary_load_coeff.Eval(vec, *Tr, ip);

         out << "(" << ip.x << "," << "y)" << " = " << vec(0) << " " << vec(1) << "\n";

      }
   }

   return 0;
}

void compute_element_jacobian_inverse(Mesh &mesh, IntegrationRule *ir,
                                      Vector &element_jacobian_inverse)
{
   const int dim = mesh.Dimension();
   const int num_el = mesh.GetNE();
   const int num_qp = ir->GetNPoints();

   element_jacobian_inverse.SetSize(num_qp * dim * dim * num_el);

   // Cache inverse Jacobian on each quadrature point
   const GeometricFactors *geom = mesh.GetGeometricFactors(
                                     *ir, GeometricFactors::JACOBIANS);
   auto J = Reshape(geom->J.Read(), num_qp, dim, dim, num_el);
   auto Jinv = Reshape(element_jacobian_inverse.Write(), num_qp, dim, dim, num_el);
   DenseMatrix Jqp(dim, dim), JqpInv(dim, dim);
   for (int e = 0; e < num_el; e++)
   {
      for (int qp = 0; qp < num_qp; qp++)
      {
         for (int i = 0; i < dim; i++)
         {
            for (int j = 0; j < dim; j++)
            {
               Jqp(i, j) = J(qp, i, j, e);
            }
         }

         CalcInverse(Jqp, JqpInv);

         for (int i = 0; i < dim; i++)
         {
            for (int j = 0; j < dim; j++)
            {
               Jinv(qp, i, j, e) = JqpInv(i, j);
            }
         }
      }
   }
}

inline
std::string check_result(double norm, double rtol = 1e-12)
{
   if (norm < rtol)
   {
      return "✅";
   }
   return "❌";
}

int main(int argc, char *argv[])
{
   using namespace std;

   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   int dimension = 2;
   int polynomial_order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&polynomial_order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.ParseCheck();

   std::cout << "Polynomial order = " << polynomial_order << "\n";

   FunctionCoefficient linear_scalar_coeff([&](const Vector &c)
   {
      double x = c(0), y = c(1);
      return 2.0 * x + x * y;
   });

   VectorFunctionCoefficient dlinear_scalardx_coeff(dimension, [&](const Vector &c,
                                                                   Vector &u)
   {
      double x = c(0), y = c(1);
      u(0) = 2.0 + c(1);
      u(1) = c(0);
   });

   FunctionCoefficient quadratic_coeff([&](const Vector &c)
   {
      double x = c(0), y = c(1);
      return 2.0*x*x + x*y*y;
   });

   VectorFunctionCoefficient dquadraticdx_coeff(dimension, [&](const Vector &c,
                                                               Vector &u)
   {
      double x = c(0), y = c(1);
      u(0) = 4.0*x+y*y,
      u(1) = 2.0*x*y;
   });

   {
      Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL, false, 1.0,
                                        1.0);
      mesh.EnsureNodes();
      ParMesh pmesh(MPI_COMM_WORLD, mesh);

      H1_FECollection h1_fec(polynomial_order);
      ParFiniteElementSpace h1_fes(&pmesh, &h1_fec);
      ParFiniteElementSpace h1_vfes(&pmesh, &h1_fec, dimension, Ordering::byVDIM);

      cout << "#dofs: " << h1_fes.GetVSize() << "\n\n";

      auto ir = const_cast<IntegrationRule *>(
                   &IntRules.Get(mesh.GetElementGeometry(0),
                                 2 * mesh.GetNodes()->FESpace()->GetElementOrder(0)));

      Vector element_jacobian_inverse;
      compute_element_jacobian_inverse(mesh, ir, element_jacobian_inverse);

      auto h1v_prolongation = h1_vfes.GetProlongationMatrix();

      ParGridFunction u(&h1_fes), du(&h1_vfes), uv(&h1_vfes);
      u = 0.0, du = 0.0, uv = 0.0;

      {
         cout << "scalar interpolation\n";
         Vector u_qp;
         u.ProjectCoefficient(linear_scalar_coeff);
         interpolate(u, *ir, u_qp);
         integrate_basis(u_qp, h1_fes, *ir, u);
         double integral = 0.0;
         for (int dof = 0; dof < h1_fes.GetVSize(); dof++)
         {
            integral += u(dof);
         }
         cout << "|I[u]dx - I[u_ex]dx| = " << abs(integral - 5.0/4.0) << "\n";
         cout << endl;
      }

      {
         cout << "weak gradient of scalar\n";
         Vector dudx_qp;
         u.ProjectCoefficient(linear_scalar_coeff);
         gradient_wrt_x(u, *ir, dudx_qp);
         integrate_basis(dudx_qp, h1_vfes, *ir, du);

         Vector integral(2);
         for (int d = 0; d < du.FESpace()->GetVDim(); d++)
         {
            integral(d) = 0.0;
            for (int i = 0; i < du.FESpace()->GetNDofs(); i++)
            {
               int idx = Ordering::Map<Ordering::byVDIM>(
                            du.FESpace()->GetNDofs(),
                            du.FESpace()->GetVDim(),
                            i,
                            d);
               integral(d) += du(idx);
            }
         }
         cout << "|I[du]dx - I[du_ex]dx| = " << abs(integral(0) - 5.0/2.0) << "\n"
              << "|I[du]dy - I[du_ex]dy| = " << abs(integral(1) - 1.0/2.0) << "\n";

         ParLinearForm l(&h1_vfes);
         auto integrator = new VectorDomainLFIntegrator(dlinear_scalardx_coeff);
         integrator->SetIntRule(ir);
         l.AddDomainIntegrator(integrator);
         l.Assemble();
         du -= *l.ParallelAssemble();
         cout << "|du - du_form|_l2 = " << du.Norml2()
              << check_result(du.Norml2()) << "\n";

         cout << endl;
      }

      {
         cout << "scalar diffusion, linear u\n";
         Vector dudx_qp, ru(h1_fes.GetVSize());
         u.ProjectCoefficient(linear_scalar_coeff);
         gradient_wrt_x(u, *ir, dudx_qp);
         integrate_basis_gradient(dudx_qp, h1_fes, *ir, ru,
                                  element_jacobian_inverse);

         ParBilinearForm b(&h1_fes);
         auto integrator = new DiffusionIntegrator;
         integrator->SetIntRule(ir);
         b.AddDomainIntegrator(integrator);
         b.Assemble();
         b.Finalize();

         ParGridFunction y(&h1_fes);
         b.Mult(u, y);

         y -= ru;
         cout << "|r(u) - r(u)_form|_l2 = " << y.Norml2()
              << check_result(y.Norml2()) << "\n";

         cout << endl;
      }

      {
         cout << "scalar diffusion, quadratic u\n";
         Vector dudx_qp, ru(h1_fes.GetVSize());
         u.ProjectCoefficient(quadratic_coeff);
         gradient_wrt_x(u, *ir, dudx_qp);
         integrate_basis_gradient(dudx_qp, h1_fes, *ir, ru,
                                  element_jacobian_inverse);

         ParBilinearForm b(&h1_fes);
         auto integrator = new DiffusionIntegrator;
         integrator->SetIntRule(ir);
         b.AddDomainIntegrator(integrator);
         b.Assemble();
         b.Finalize();

         ParGridFunction y(&h1_fes);
         b.Mult(u, y);

         y -= ru;
         cout << "|r(u) - r(u)_form|_l2 = " << y.Norml2()
              << check_result(y.Norml2()) << "\n";

         cout << endl;
      }

      {
         cout << "vector diffusion, linear u\n";
         Vector duvdx_qp, ru(h1_vfes.GetVSize());
         uv.ProjectCoefficient(dlinear_scalardx_coeff);
         gradient_wrt_x(uv, *ir, duvdx_qp);
         integrate_basis_gradient(duvdx_qp, h1_vfes, *ir, ru,
                                  element_jacobian_inverse);

         ParBilinearForm b(&h1_vfes);
         auto integrator = new VectorDiffusionIntegrator;
         integrator->SetIntRule(ir);
         b.AddDomainIntegrator(integrator);
         b.Assemble();
         b.Finalize();

         ParGridFunction y(&h1_vfes);
         b.Mult(uv, y);

         y -= ru;
         cout << "|r(u) - r(u)_form|_l2 = " << y.Norml2()
              << check_result(y.Norml2()) << "\n";

         cout << endl;
      }

      {
         cout << "vector diffusion, quadratic u\n";
         Vector duvdx_qp, ru(h1_vfes.GetVSize());
         uv.ProjectCoefficient(dquadraticdx_coeff);
         gradient_wrt_x(uv, *ir, duvdx_qp);
         integrate_basis_gradient(duvdx_qp, h1_vfes, *ir, ru,
                                  element_jacobian_inverse);

         ParBilinearForm b(&h1_vfes);
         auto integrator = new VectorDiffusionIntegrator;
         integrator->SetIntRule(ir);
         b.AddDomainIntegrator(integrator);
         b.Assemble();
         b.Finalize();

         ParGridFunction y(&h1_vfes);
         b.Mult(uv, y);

         y -= ru;
         cout << "|r(u) - r(u)_form|_l2 = " << y.Norml2()
              << check_result(y.Norml2()) << "\n";

         cout << endl;
      }

   }

   return 0;
}