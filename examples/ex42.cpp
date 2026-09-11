//                                MFEM Example 42
//
// Compile with: make ex42
//
// Sample runs:  ex42
//               ex42 -r 5
//
// Description:  This example solves div-div plus mass for a symmetric matrix
//               field with the lowest-order 2D Johnson--Mercier element. It
//               reports matrix L2, row-divergence L2, and graph-norm errors
//               under uniform refinement.

#include "mfem.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>

using namespace mfem;
using namespace std;

namespace
{

void ExactMatrix(const Vector &x, DenseMatrix &s)
{
   const real_t a = x(0)*(1.0 - x(0));
   const real_t b = x(1)*(1.0 - x(1));
   s.SetSize(2);
   s(0,0) = a;
   s(0,1) = s(1,0) = a*b;
   s(1,1) = b;
}

void ExactDiv(const Vector &x, Vector &d)
{
   const real_t a = x(0)*(1.0 - x(0));
   const real_t b = x(1)*(1.0 - x(1));
   const real_t ap = 1.0 - 2.0*x(0);
   const real_t bp = 1.0 - 2.0*x(1);
   d.SetSize(2);
   d(0) = ap + a*bp;
   d(1) = ap*b + bp;
}

void RightHandSide(const Vector &x, DenseMatrix &f)
{
   const real_t a = x(0)*(1.0 - x(0));
   const real_t b = x(1)*(1.0 - x(1));
   const real_t ap = 1.0 - 2.0*x(0);
   const real_t bp = 1.0 - 2.0*x(1);
   f.SetSize(2);
   f(0,0) = a + 2.0 - ap*bp;
   f(0,1) = f(1,0) = a*b + a + b;
   f(1,1) = b + 2.0 - ap*bp;
}

void ComputeErrors(const GridFunction &solution, real_t &matrix_error,
                   real_t &div_error)
{
   const FiniteElementSpace *fes = solution.FESpace();
   Mesh *mesh = fes->GetMesh();
   const IntegrationRule &base = IntRules.Get(Geometry::TRIANGLE, 6);
   unique_ptr<IntegrationRule> ir(base.ApplyToTriangleAlfeldSplit());
   real_t matrix_error_sq = 0.0;
   real_t div_error_sq = 0.0;
   Array<int> dofs;
   DofTransformation doftrans;
   Vector local, point(2), exact_div(2);
   DenseMatrix exact(2);

   for (int e = 0; e < mesh->GetNE(); e++)
   {
      const FiniteElement *fe = fes->GetFE(e);
      ElementTransformation *T = mesh->GetElementTransformation(e);
      fes->GetElementDofs(e, dofs, doftrans);
      solution.GetSubVector(dofs, local);
      doftrans.InvTransformPrimal(local);
      DenseTensor shape(2, 2, fe->GetDof());
      DenseMatrix divshape(fe->GetDof(), 2);

      for (int q = 0; q < ir->GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir->IntPoint(q);
         T->SetIntPoint(&ip);
         fe->CalcMShape(*T, shape);
         fe->CalcPhysDivShape(*T, divshape);
         T->Transform(ip, point);
         ExactMatrix(point, exact);
         ExactDiv(point, exact_div);

         real_t matrix_diff_sq = 0.0;
         for (int i = 0; i < 2; i++)
         {
            for (int j = 0; j < 2; j++)
            {
               real_t value = 0.0;
               for (int k = 0; k < fe->GetDof(); k++)
               {
                  value += local(k)*shape(i,j,k);
               }
               matrix_diff_sq += (value - exact(i,j))*(value - exact(i,j));
            }
         }
         real_t div_diff_sq = 0.0;
         for (int i = 0; i < 2; i++)
         {
            real_t value = 0.0;
            for (int k = 0; k < fe->GetDof(); k++)
            {
               value += local(k)*divshape(k,i);
            }
            div_diff_sq += (value - exact_div(i))*(value - exact_div(i));
         }
         const real_t w = ip.weight*T->Weight();
         matrix_error_sq += w*matrix_diff_sq;
         div_error_sq += w*div_diff_sq;
      }
   }
   matrix_error = std::sqrt(matrix_error_sq);
   div_error = std::sqrt(div_error_sq);
}

} // namespace

int main(int argc, char *argv[])
{
   int refinements = 4;
   bool visualization = false;
   OptionsParser args(argc, argv);
   args.AddOption(&refinements, "-r", "--refinements",
                  "Number of refinement levels.");
   args.AddOption(&visualization, "-vis", "--visualization",
                  "-no-vis", "--no-visualization",
                  "Enable or disable visualization (accepted for consistency).");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   MatrixFunctionCoefficient rhs(2, RightHandSide);
   const IntegrationRule &load_rule = IntRules.Get(Geometry::TRIANGLE, 6);
   real_t previous_matrix_error = 0.0;
   real_t previous_div_error = 0.0;
   real_t previous_graph_error = 0.0;
   real_t matrix_rate = 0.0;
   real_t div_rate = 0.0;
   real_t graph_rate = 0.0;

   cout << "\n  n       dofs       ||sigma-sigma_h||"
        << "    rate       ||div(sigma-sigma_h)||    rate"
        << "       graph error    rate\n";
   for (int level = 0; level < refinements; level++)
   {
      const int n = 2 << level;
      Mesh mesh = Mesh::MakeCartesian2D(n, n, Element::TRIANGLE, true,
                                        1.0, 1.0);
      JohnsonMercierFECollection fec;
      FiniteElementSpace fes(&mesh, &fec);

      LinearForm b(&fes);
      b.AddDomainIntegrator(new MatrixFEDomainLFIntegrator(rhs, &load_rule));
      b.Assemble();

      BilinearForm a(&fes);
      a.AddDomainIntegrator(new MatrixDivDivIntegrator);
      a.AddDomainIntegrator(new MatrixFEMassIntegrator);
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
      GSSmoother prec(static_cast<SparseMatrix &>(*A));
      PCG(*A, prec, B, X, 0, 5000, 1e-12, 0.0);
      a.RecoverFEMSolution(X, b, solution);

      real_t matrix_error, div_error;
      ComputeErrors(solution, matrix_error, div_error);
      const real_t graph_error = std::hypot(matrix_error, div_error);
      matrix_rate = level == 0 ? 0.0 :
                    std::log(previous_matrix_error/matrix_error)/std::log(2.0);
      div_rate = level == 0 ? 0.0 :
                 std::log(previous_div_error/div_error)/std::log(2.0);
      graph_rate = level == 0 ? 0.0 :
                   std::log(previous_graph_error/graph_error)/std::log(2.0);
      cout << setw(3) << n << setw(11) << fes.GetTrueVSize()
           << scientific << setprecision(6) << setw(23) << matrix_error
           << fixed << setprecision(2) << setw(8) << matrix_rate
           << scientific << setprecision(6) << setw(27) << div_error
           << fixed << setprecision(2) << setw(8) << div_rate
           << scientific << setprecision(6) << setw(20) << graph_error
           << fixed << setprecision(2) << setw(8) << graph_rate << '\n';
      previous_matrix_error = matrix_error;
      previous_div_error = div_error;
      previous_graph_error = graph_error;
   }
   if (refinements > 1 &&
       (matrix_rate < 1.8 || div_rate < 0.9 || graph_rate < 0.9))
   {
      cerr << "The observed convergence rates are below the expected orders.\n";
      return 2;
   }
   return 0;
}
