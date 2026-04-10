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
         int nodes_per_dim = vdofs_aux.Size()/fes.GetVDim();
         fes.GetElementVDofs(e, vdofs);
         for (int vd=0; vd < fes.GetVDim(); vd++)
         {
            Array<int> sub_vdofs, sub_vdofs_aux;
            vdofs.GetSubArray(vd*nodes_per_dim, nodes_per_dim, sub_vdofs);
            vdofs_aux.GetSubArray(vd*nodes_per_dim, nodes_per_dim, sub_vdofs_aux);
            R.AddSubMatrix(sub_vdofs, sub_vdofs_aux, I);
         }
      }

      Vector m = multiplicity;
      m.Reciprocal();
      R.ScaleRows(m);

      R.Finalize();

      for (int i : ess_dofs)
      {
         R.EliminateRow(i);
      }
      std::cout << "num rows R = " << R.NumRows() << std::endl;
      std::cout << "num cols R = " << R.NumCols() << std::endl;
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
   // mutable Vector z;

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
      // A_hat_inv.Mult(b, x);

      if (D)
      {
         z3.SetSize(x.Size());
         D->Mult(b, z3);
         x += z3;
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
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/beam-tri.mesh";
   int order = 1;
   bool static_cond = false;
   bool visualization = 0;
   real_t kappa_0 = 10.0;
   int ref_levels = 4;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&ref_levels, "-r", "--ref_levels",
                  "Number of times to refine mesh");
   args.AddOption(&kappa_0, "-k", "--kappa_0",
                  "DG Penalty Param");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral or hexahedral elements with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   if (mesh->attributes.Max() < 2 || mesh->bdr_attributes.Max() < 2)
   {
      cerr << "\nInput mesh should have at least two materials and "
           << "two boundary attributes! (See schematic in ex2.cpp)\n"
           << endl;
      return 3;
   }

   // 3. Select the order of the finite element discretization space. For NURBS
   //    meshes, we increase the order by degree elevation.
   if (mesh->NURBSext)
   {
      mesh->DegreeElevate(order, order);
   }

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 5,000
   //    elements.
   // int ref_levels_old = (int)floor(log(5000./mesh->GetNE())/log(2.)/dim);
   // std::cout << "num ref old = " << ref_levels_old << std::endl;
   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define a finite element space on the mesh. Here we use vector finite
   //    elements, i.e. dim copies of a scalar finite element space. The vector
   //    dimension is specified by the last argument of the FiniteElementSpace
   //    constructor. For NURBS meshes, we use the (degree elevated) NURBS space
   //    associated with the mesh nodes.
   FiniteElementCollection *fec;
   FiniteElementSpace *fespace;
   FiniteElementCollection *dg_fec;
   FiniteElementSpace *dg_fes;
   if (mesh->NURBSext)
   {
      fec = NULL;
      fespace = mesh->GetNodes()->FESpace();
   }
   else
   {
      fec = new H1_FECollection(order, dim);
      fespace = new FiniteElementSpace(mesh, fec, dim);
      dg_fec = new DG_FECollection(order, dim, BasisType::GaussLobatto);
      dg_fes = new FiniteElementSpace(mesh, dg_fec, dim);
   }
   std::cout << "num elements \n" << mesh->GetNE() << std::endl;
   cout << "Number of finite element unknowns: " << fespace->GetTrueVSize()
        << endl << "Assembling: " << flush;

   std::cout << "num dg unknowns \n" << dg_fes->GetTrueVSize() << std::endl;
   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking only
   //    boundary attribute 1 from the mesh as essential and converting it to a
   //    list of true dofs.
   // Array<int> ess_tdof_list;
   // fespace->GetBoundaryTrueDofs(ess_tdof_list);
   Array<int> ess_tdof_list, ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system. In this case, b_i equals the boundary integral
   //    of f*phi_i where f represents a "pull down" force on the Neumann part
   //    of the boundary and phi_i are the basis functions in the finite element
   //    fespace. The force is defined by the VectorArrayCoefficient object f,
   //    which is a vector of Coefficient objects. The fact that f is non-zero
   //    on boundary attribute 2 is indicated by the use of piece-wise constants
   //    coefficient for its last component.
   VectorArrayCoefficient f(dim);
   for (int i = 0; i < dim-1; i++)
   {
      f.Set(i, new ConstantCoefficient(0.0));
   }
   {
      Vector pull_force(mesh->bdr_attributes.Max());
      pull_force = 0.0;
      pull_force(1) = -1.0e-2;
      f.Set(dim-1, new PWConstCoefficient(pull_force));
   }

   LinearForm b(fespace);
   b.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   cout << "r.h.s. ... " << flush;
   b.Assemble();
   // b->Finalize();

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the linear elasticity integrator with piece-wise
   //    constants coefficient lambda and mu.
   Vector lambda(mesh->attributes.Max());
   lambda = 1.0;
   lambda(0) = lambda(1)*50;
   PWConstCoefficient lambda_func(lambda);
   Vector mu(mesh->attributes.Max());
   mu = 1.0;
   mu(0) = mu(1)*50;
   PWConstCoefficient mu_func(mu);

   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func,mu_func));
   cout << "matrix ... " << flush;
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();
   // a->Finalize();

   // const real_t sigma = -1.0;
   const real_t kappa = kappa_0*(order+1)*(order+dim)/dim;
   BilinearForm *a_aux = new BilinearForm(dg_fes);
   a_aux->AddDomainIntegrator(new ElasticityIntegrator(lambda_func, mu_func));
   a_aux->AddInteriorFaceIntegrator(
      new DGElasticityIntegrator(lambda_func, mu_func, -1.0, kappa));
   a_aux->AddBdrFaceIntegrator(
      new DGElasticityIntegrator(lambda_func, mu_func, -1.0, kappa), ess_bdr);
   a_aux->Assemble();
   a_aux->Finalize();

   SparseMatrix &A_cg= a->SpMat();
   SparseMatrix &A_dg= a_aux->SpMat();
   std::cout << "num cols Acg = " << A_cg.NumCols() << std::endl;
   std::cout << "num cols dg = " << A_dg.NumCols() << std::endl;
   OswaldOperator R_op(*dg_fes, *fespace, ess_tdof_list);
   SparseMatrix R_mat = R_op.Assemble();
   UMFPackSolver A_dg_inv(A_dg);
   //AgglomerationMultigrid mg(fespace, A, ncoarse, num_levels, smoother, paraview_vis);
   AuxiliarySolver prec(A_dg_inv, R_mat, ess_tdof_list, nullptr);
   // CompositeAuxiliaryAgglomerationSolver prec(*fespace, A_cg, *dg_fes, A_dg, ess_tdof_list, 4, 2, 0);

      // 8. Form the linear system A X = B. This includes eliminating boundary
   //    conditions, applying AMR constraints, and other transformations.
   SparseMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   // {
   //    ofstream f("A.txt");
   //    A.PrintMatlab(f);
   // }
   // {
   //    ofstream f("prec.txt");
   //    prec.PrintMatlab(f);
   // }


   CGSolver cg;
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetPreconditioner(prec);
   cg.SetOperator(A);
   cg.Mult(B, X);

   // 10. Recover the solution x as a grid function and save to file. The output
   //     can be viewed using GLVis as follows: "glvis -m mesh.mesh -g sol.gf"
   a->RecoverFEMSolution(X, b, x);
   
   // 13. For non-NURBS meshes, make the mesh curved based on the finite element
   //     space. This means that we define the mesh elements through a fespace
   //     based transformation of the reference element. This allows us to save
   //     the displaced mesh as a curved mesh when using high-order finite
   //     element displacement field. We assume that the initial mesh (read from
   //     the file) is not higher order curved mesh compared to the chosen FE
   //     space.
   if (!mesh->NURBSext)
   {
      mesh->SetNodalFESpace(fespace);
   }

   // 14. Save the displaced mesh and the inverted solution (which gives the
   //     backward displacements to the original grid). This output can be
   //     viewed later using GLVis: "glvis -m displaced.mesh -g sol.gf".
   {
      GridFunction *nodes = mesh->GetNodes();
      *nodes += x;
      x *= -1;
      ofstream mesh_ofs("displaced.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 15. Send the above data by socket to a GLVis server. Use the "n" and "b"
   //     keys in GLVis to visualize the displacements.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   // 16. Free the used memory.
   delete a;
   // delete b;
   if (fec)
   {
      delete fespace;
      delete fec;
   }
   delete mesh;

   return 0;
}
