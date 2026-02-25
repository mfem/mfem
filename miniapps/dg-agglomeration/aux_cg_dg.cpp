#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "mg_agglom.hpp"

using namespace std;
using namespace mfem;

struct OswaldOperator : Operator
{
   const FiniteElementSpace &fes_aux;
   const FiniteElementSpace &fes;
   const Array<int> &ess_dofs;
   Vector multiplicity;
   mutable Vector z;
   OswaldOperator(const FiniteElementSpace &fes_aux_,
                  const FiniteElementSpace &fes_,
                  const Array<int> &ess_dofs_)
      : Operator(fes_.GetTrueVSize(), fes_aux_.GetTrueVSize()),
        fes_aux(fes_aux_),
        fes(fes_),
        ess_dofs(ess_dofs_)
   {
      const auto ordering = ElementDofOrdering::LEXICOGRAPHIC;
      const Operator *restr_op = fes.GetElementRestriction(ordering);
      const auto *restr = dynamic_cast<const ElementRestriction*>(restr_op);
      MFEM_VERIFY(restr, "");

      multiplicity.SetSize(restr->Width());
      {
         Vector ones(restr->Height());
         ones = 1.0;
         restr->MultTransposeUnsigned(ones, multiplicity);
      }
   }

   SparseMatrix Assemble() const
   {
      SparseMatrix R(fes.GetTrueVSize(), fes_aux.GetTrueVSize());

      for (int e = 0; e < fes_aux.GetNE(); ++e)
      {
         DenseMatrix I;
         {
            const auto T = fes.GetElementTransformation(e);
            fes.GetFE(e)->Project(*fes_aux.GetFE(e), *T, I);
         }

         Array<int> vdofs, vdofs_aux;
         fes_aux.GetElementVDofs(e, vdofs_aux);
         fes.GetElementVDofs(e, vdofs);

         R.AddSubMatrix(vdofs, vdofs_aux, I);
      }

      Vector m = multiplicity;
      m.Reciprocal();
      R.ScaleRows(m);

      R.Finalize();

      for (int i : ess_dofs)
      {
         R.EliminateRow(i);
      }

      return R;
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      const int ne = fes_aux.GetNE();
      y = 0.0;

      for (int e = 0; e < ne; ++e)
      {
         DenseMatrix I;
         {
            const auto T = fes.GetElementTransformation(e);
            fes.GetFE(e)->Project(*fes_aux.GetFE(e), *T, I);
         }

         Array<int> vdofs, vdofs_aux;
         fes_aux.GetElementVDofs(e, vdofs_aux);
         fes.GetElementVDofs(e, vdofs);
         Vector x_e(vdofs_aux.Size()), y_e(vdofs.Size());
         x.GetSubVector(vdofs_aux, x_e);

         I.Mult(x_e, y_e);

         y.AddElementVector(vdofs, y_e);
      }
      for (int i = 0; i < y.Size(); ++i)
      {
         y[i] /= multiplicity[i];
      }
      for (int i : ess_dofs)
      {
         y[i] = 0.0;
      }
   }

   void MultTranspose(const Vector &x, Vector &y) const override
   {
      const int ne = fes.GetNE();
      y = 0.0;
      z = x;
      for (int i : ess_dofs)
      {
         z[i] = 0.0;
      }
      for (int i = 0; i < z.Size(); ++i)
      {
         z[i] /= multiplicity[i];
      }

      for (int e = 0; e < ne; ++e)
      {
         DenseMatrix I;
         {
            const auto T = fes.GetElementTransformation(e);
            fes.GetFE(e)->Project(*fes_aux.GetFE(e), *T, I);
         }

         Array<int> vdofs, vdofs_aux;
         fes.GetElementVDofs(e, vdofs);
         fes_aux.GetElementVDofs(e, vdofs_aux);

         Vector x_e(vdofs.Size()), y_e(vdofs_aux.Size());
         z.GetSubVector(vdofs, x_e);

         I.MultTranspose(x_e, y_e);

         y.AddElementVector(vdofs_aux, y_e);
      }
   }
};

struct AuxiliarySolver : Solver
{
   const Solver &A_hat_inv;
   const Operator &R;
   const Array<int> ess_dofs;
   const Solver *D;
   mutable Vector z1, z2, z3;
   mutable Vector z;

   AuxiliarySolver(const Solver &A_hat_inv_, const Operator &R_,
                   const Array<int> &ess_dofs_,
                   const Solver *D_)
      : Solver(R_.Height()),
        A_hat_inv(A_hat_inv_),
        R(R_),
        ess_dofs(ess_dofs_),
        D(D_)
   { }

   void SetOperator(const Operator &op) { }

   void Mult(const Vector &b, Vector &x) const
   {
      z1.SetSize(R.Width());
      z2.SetSize(R.Width());

      R.MultTranspose(b, z1);
      A_hat_inv.Mult(z1, z2);
      R.Mult(z2, x);

      if (D)
      {
         z.SetSize(x.Size());
         D->Mult(b, z);
         x += z;
      }

      for (int i : ess_dofs)
      {
         x[i] = b[i];
      }
   }
};

class CompositeAuxiliaryAgglomerationSolver : public Solver
{
   AgglomerationMultigrid a_mg;
   TruncatedMultigrid t_mg;
   OswaldOperator oswald;
   unique_ptr<SparseMatrix> RP;
   GSSmoother S;
   AuxiliarySolver aux;
public:
   CompositeAuxiliaryAgglomerationSolver(
      FiniteElementSpace &fes,
      SparseMatrix &A,
      FiniteElementSpace &fes_aux,
      SparseMatrix &A_aux,
      Array<int> &ess_dofs,
      int ncoarse,
      int num_levels,
      int smoother_choice)
      : a_mg(fes_aux, A_aux, ncoarse, num_levels, smoother_choice, false),
        t_mg(a_mg),
        oswald(fes_aux, fes, ess_dofs),
        RP(mfem::Mult(oswald.Assemble(), a_mg.GetFinestProlongation())),
        S(A),
        aux(t_mg, *RP, ess_dofs, &S)
   { }

   void SetOperator(const Operator &op) { }

   void Mult(const Vector &b, Vector &x) const
   {
      aux.Mult(b, x);
   }
};


int main(int argc, char *argv[])
{
   // 1. Parse command line options.
   string mesh_file = "../../data/star.mesh";
   int order = 1;
   int ref = 0;
   real_t kappa_0 = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree.");
   args.AddOption(&ref, "-r", "--refine", "Number of refinements.");
   args.AddOption(&kappa_0, "-k", "--kappa", "Penalty factor.");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh mesh(mesh_file);
   for (int i = 0; i < ref; ++i) { mesh.UniformRefinement(); }
   int dim = mesh.Dimension();

   // 3. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   H1_FECollection h1_fec(order, mesh.Dimension());
   FiniteElementSpace h1_fes(&mesh, &h1_fec);

   DG_FECollection dg_fec(order, mesh.Dimension(), BasisType::GaussLobatto);
   FiniteElementSpace dg_fes(&mesh, &dg_fec);

   cout << "Number of H1 unknowns: " << h1_fes.GetTrueVSize() << endl;
   cout << "Number of DG unknowns: " << dg_fes.GetTrueVSize() << endl;

   // 4. Extract the list of all the boundary DOFs. These will be marked as
   //    Dirichlet in order to enforce zero boundary conditions.
   Array<int> ess_dofs;
   h1_fes.GetBoundaryTrueDofs(ess_dofs);

   // 5. Define the solution x as a finite element grid function in h1_fes. Set
   //    the initial guess to zero, which also sets the boundary conditions.
   GridFunction x(&h1_fes);
   x = 0.0;

   // 6. Set up the linear form b(.) corresponding to the right-hand side.
   ConstantCoefficient one(1.0);
   LinearForm b(&h1_fes);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // 7. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
   BilinearForm a(&h1_fes);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.AddDomainIntegrator(new MassIntegrator);
   a.Assemble();

   // 7. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
   const real_t sigma = -1.0;
   const real_t kappa = kappa_0*(order+1)*(order+dim)/dim;
   BilinearForm a_aux(&dg_fes);
   a_aux.AddDomainIntegrator(new DiffusionIntegrator);
   a_aux.AddDomainIntegrator(new MassIntegrator);
   a_aux.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
   a_aux.AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
   a_aux.Assemble();
   a_aux.Finalize();

   SparseMatrix &A_cg= a.SpMat();
   SparseMatrix &A_dg= a_aux.SpMat();
   // OswaldOperator R_op(dg_fes, h1_fes, ess_dofs);
   // SparseMatrix R = R_op.Assemble();
   // AgglomerationMultigrid A_hat_inv(dg_fes, A_cg, 4, 3, 0, false, R);
   // Vector diag(dg_fes.GetTrueVSize());
   // a_aux.AssembleDiagonal(diag);
   // Solver* smoother = new OperatorChebyshevSmoother(A_dg, diag, ess_dofs, 2);
   // AuxiliarySolver prec(A_hat_inv, R_op, ess_dofs, smoother);

   CompositeAuxiliaryAgglomerationSolver prec(h1_fes, A_cg, dg_fes, A_dg, ess_dofs,
                                              4, 3, 0);

   // 8. Form the linear system A X = B. This includes eliminating boundary
   //    conditions, applying AMR constraints, and other transformations.
   SparseMatrix A;
   Vector B, X;
   a.FormLinearSystem(ess_dofs, x, b, A, X, B);

   {
      ofstream f("A.txt");
      A.PrintMatlab(f);
   }
   {
      ofstream f("prec.txt");
      prec.PrintMatlab(f);
   }

   CGSolver cg;
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetPreconditioner(prec);
   cg.SetOperator(A);
   cg.Mult(B, X);

   // 10. Recover the solution x as a grid function and save to file. The output
   //     can be viewed using GLVis as follows: "glvis -m mesh.mesh -g sol.gf"
   a.RecoverFEMSolution(X, b, x);
   x.Save("sol.gf");
   mesh.Save("mesh.mesh");

   return 0;
}
