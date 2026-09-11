//                                MFEM Example 44
//
// Compile with: make ex44
//
// Sample runs:  ex44
//               ex44 -r 5
//
// Description: Solve the clamped biharmonic problem Delta^2 u = f on the unit
// square with the cubic Hsieh--Clough--Tocher element. The manufactured smooth
// solution is used to report L2, H1, and H2 convergence rates.

#include "mfem.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>

using namespace mfem;
using namespace std;

namespace
{

real_t ExactSolution(const Vector &x)
{
   return std::pow(std::sin(M_PI*x(0))*std::sin(M_PI*x(1)), 2);
}

void ExactGradient(const Vector &x, Vector &gradient)
{
   const real_t ax = std::pow(std::sin(M_PI*x(0)), 2);
   const real_t ay = std::pow(std::sin(M_PI*x(1)), 2);
   gradient.SetSize(2);
   gradient(0) = M_PI*std::sin(2.0*M_PI*x(0))*ay;
   gradient(1) = M_PI*std::sin(2.0*M_PI*x(1))*ax;
}

void ExactHessian(const Vector &x, DenseMatrix &hessian)
{
   const real_t ax = std::pow(std::sin(M_PI*x(0)), 2);
   const real_t ay = std::pow(std::sin(M_PI*x(1)), 2);
   const real_t axp = M_PI*std::sin(2.0*M_PI*x(0));
   const real_t ayp = M_PI*std::sin(2.0*M_PI*x(1));
   const real_t axpp = 2.0*M_PI*M_PI*std::cos(2.0*M_PI*x(0));
   const real_t aypp = 2.0*M_PI*M_PI*std::cos(2.0*M_PI*x(1));
   hessian.SetSize(2);
   hessian(0,0) = axpp*ay;
   hessian(0,1) = hessian(1,0) = axp*ayp;
   hessian(1,1) = ax*aypp;
}

real_t RightHandSide(const Vector &x)
{
   const real_t ax = std::pow(std::sin(M_PI*x(0)), 2);
   const real_t ay = std::pow(std::sin(M_PI*x(1)), 2);
   const real_t cx = std::cos(2.0*M_PI*x(0));
   const real_t cy = std::cos(2.0*M_PI*x(1));
   return 8.0*std::pow(M_PI,4)*(cx*cy - cx*ay - ax*cy);
}

void ComputeErrors(const GridFunction &solution, real_t &l2_error,
                   real_t &h1_error, real_t &h2_error)
{
   const FiniteElementSpace *fes = solution.FESpace();
   Mesh *mesh = fes->GetMesh();
   const IntegrationRule &base = IntRules.Get(Geometry::TRIANGLE, 10);
   unique_ptr<IntegrationRule> rule(base.ApplyToTriangleAlfeldSplit());
   real_t l2_sq = 0.0;
   real_t h1_sq = 0.0;
   real_t h2_sq = 0.0;
   Array<int> dofs;
   Vector local, shape, point(2), gradient(2), exact_gradient(2);
   DenseMatrix dshape, hshape, exact_hessian(2);

   for (int element = 0; element < mesh->GetNE(); element++)
   {
      const FiniteElement *fe = fes->GetFE(element);
      ElementTransformation *T = mesh->GetElementTransformation(element);
      fes->GetElementDofs(element, dofs);
      solution.GetSubVector(dofs, local);
      shape.SetSize(fe->GetDof());
      dshape.SetSize(fe->GetDof(), 2);
      hshape.SetSize(fe->GetDof(), 3);
      for (int q = 0; q < rule->GetNPoints(); q++)
      {
         const IntegrationPoint &ip = rule->IntPoint(q);
         T->SetIntPoint(&ip);
         fe->CalcPhysShape(*T, shape);
         fe->CalcPhysDShape(*T, dshape);
         fe->CalcPhysHessian(*T, hshape);
         T->Transform(ip, point);
         ExactGradient(point, exact_gradient);
         ExactHessian(point, exact_hessian);

         const real_t value_error = shape*local - ExactSolution(point);
         dshape.MultTranspose(local, gradient);
         const real_t gx = gradient(0) - exact_gradient(0);
         const real_t gy = gradient(1) - exact_gradient(1);
         real_t hxx = -exact_hessian(0,0);
         real_t hxy = -exact_hessian(0,1);
         real_t hyy = -exact_hessian(1,1);
         for (int i = 0; i < fe->GetDof(); i++)
         {
            hxx += local(i)*hshape(i,0);
            hxy += local(i)*hshape(i,1);
            hyy += local(i)*hshape(i,2);
         }
         const real_t weight = ip.weight*T->Weight();
         l2_sq += weight*value_error*value_error;
         h1_sq += weight*(gx*gx + gy*gy);
         h2_sq += weight*(hxx*hxx + 2.0*hxy*hxy + hyy*hyy);
      }
   }
   l2_error = std::sqrt(l2_sq);
   h1_error = std::sqrt(h1_sq);
   h2_error = std::sqrt(h2_sq);
}

} // namespace

int main(int argc, char *argv[])
{
   int refinements = 5;
   OptionsParser args(argc, argv);
   args.AddOption(&refinements, "-r", "--refinements",
                  "Number of convergence levels.");
   args.ParseCheck();

   FunctionCoefficient rhs(RightHandSide);
   const IntegrationRule &base = IntRules.Get(Geometry::TRIANGLE, 10);
   unique_ptr<IntegrationRule> load_rule(base.ApplyToTriangleAlfeldSplit());
   real_t old_l2 = 0.0, old_h1 = 0.0, old_h2 = 0.0;
   real_t l2_rate = 0.0, h1_rate = 0.0, h2_rate = 0.0;

   cout << "\n  n       dofs          L2 error    rate"
        << "          H1 error    rate          H2 error    rate\n";
   for (int level = 0; level < refinements; level++)
   {
      const int n = 2 << level;
      Mesh mesh = Mesh::MakeCartesian2D(n, n, Element::TRIANGLE, true,
                                        1.0, 1.0);
      HCT_FECollection fec;
      FiniteElementSpace fes(&mesh, &fec);

      LinearForm b(&fes);
      b.AddDomainIntegrator(new DomainLFIntegrator(rhs, load_rule.get()));
      b.Assemble();

      BilinearForm a(&fes);
      a.AddDomainIntegrator(new HessianIntegrator);
      a.Assemble();

      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      Array<int> ess_tdof_list;
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      GridFunction solution(&fes);
      solution = 0.0;
      OperatorPtr A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, solution, b, A, X, B);
      GSSmoother smoother(static_cast<SparseMatrix &>(*A));
      PCG(*A, smoother, B, X, 0, 20000, 1e-12, 0.0);
      a.RecoverFEMSolution(X, b, solution);

      real_t l2_error, h1_error, h2_error;
      ComputeErrors(solution, l2_error, h1_error, h2_error);
      if (level > 0)
      {
         l2_rate = std::log(old_l2/l2_error)/std::log(2.0);
         h1_rate = std::log(old_h1/h1_error)/std::log(2.0);
         h2_rate = std::log(old_h2/h2_error)/std::log(2.0);
      }
      cout << setw(3) << n << setw(11) << fes.GetTrueVSize()
           << scientific << setprecision(6) << setw(19) << l2_error
           << fixed << setprecision(2) << setw(8) << l2_rate
           << scientific << setprecision(6) << setw(19) << h1_error
           << fixed << setprecision(2) << setw(8) << h1_rate
           << scientific << setprecision(6) << setw(19) << h2_error
           << fixed << setprecision(2) << setw(8) << h2_rate << '\n';
      old_l2 = l2_error;
      old_h1 = h1_error;
      old_h2 = h2_error;
   }

   if (refinements >= 5 &&
       (l2_rate < 3.5 || h1_rate < 2.8 || h2_rate < 1.8))
   {
      cerr << "The observed convergence rates are below the expected orders.\n";
      return 2;
   }
   return 0;
}
