#include "mfem.hpp"
#include <fstream>
#include <iostream>
using namespace std;
using namespace mfem;

void E_exact(const Vector &x, Vector &f)
{
   f = 0.0;
   double kappa = 3.14;
   f(0) = sin(kappa * x(0));
   // f = 1.;
}

int main(int argc, char *argv[])
{
   int order = 2;
   Mesh *mesh = new Mesh("../data/inline-tet.mesh", 1, 1);
   int dim = mesh->Dimension();

   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   FiniteElementSpace *fes = new FiniteElementSpace(mesh, fec);

   GridFunction x(fes);
   GridFunction x2(fes);
   VectorFunctionCoefficient E(dim, E_exact);
   x.ProjectCoefficient(E);

   FiniteElementSpaceHierarchy *h =
      new FiniteElementSpaceHierarchy(mesh, fes, true, true);


   // 1. Uniform geometric refinement
   h->AddUniformlyRefinedLevel();
   FiniteElementSpace *fes_href = & h -> GetFESpaceAtLevel(1);
   Mesh *mesh_href = fes_href->GetMesh();


   // 2. Uniform order refinement
   h->AddOrderRefinedLevel(new ND_FECollection(order+1,dim));
   FiniteElementSpace *fes_pref = & h -> GetFESpaceAtLevel(2);



   GridFunction x_href(fes_href); x_href = 0.;
   GridFunction x_href2(fes_href); x_href2 = 0.;
   GridFunction x_pref(fes_pref); x_pref = 0.;

   TransferOperator *P_href = new TransferOperator(*fes, *fes_href);
   TransferOperator *P_pref = new TransferOperator(*fes_href, *fes_pref);

   P_href -> Mult(x, x_href);
   P_pref -> Mult(x_href, x_pref);


   char vishost[] = "localhost";
   int  visport   = 19916;

   string keys = "keys macF\n";

   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);
   sol_sock << "solution\n" << *mesh << x << keys
            << "window_title 'Original GridFunction x'" << flush;

   socketstream sol_sock_href(vishost, visport);
   sol_sock_href.precision(8);
   sol_sock_href << "solution\n" << *mesh_href << x_href << keys
                 << "window_title 'GridFunction on h-refined mesh P_h * x'" << flush;

   socketstream sol_sock_pref(vishost, visport);
   sol_sock_pref.precision(8);
   sol_sock_pref << "solution\n" << *mesh_href << x_pref << keys
                 << "window_title 'GridFunction on p-refined mesh P_p * P_h * x '" << flush;

   BilinearForm a_pref(fes_pref);
   a_pref.AddDomainIntegrator(new VectorFEMassIntegrator());
   a_pref.Assemble();
   Array<int> empty;
   SparseMatrix M_pref;
   a_pref.FormSystemMatrix(empty, M_pref);

   Vector B_pref(fes_pref->GetTrueVSize());
   Vector B_href(fes_href->GetTrueVSize());
   Vector B(fes->GetTrueVSize());

   M_pref.Mult(x_pref,B_pref);

   P_pref -> MultTranspose(B_pref, B_href);

   BilinearForm a_href(fes_href);
   a_href.AddDomainIntegrator(new VectorFEMassIntegrator());
   a_href.Assemble();
   SparseMatrix M_href;
   a_href.FormSystemMatrix(empty, M_href);

   {
      GSSmoother prec(M_href);
      PCG(M_href, prec, B_href, x_href2, -1, 2000, 1e-24, 0.0);
   }

   P_href -> MultTranspose(B_href, B);


   x_href -= x_href2;
   cout << "||x_href - x_href2|| = " << x_href.Norml2() << endl;

   BilinearForm a(fes);
   a.AddDomainIntegrator(new VectorFEMassIntegrator());
   a.Assemble();
   SparseMatrix M;
   a.FormSystemMatrix(empty, M);

   {
      GSSmoother prec(M);
      PCG(M, prec, B, x2, -1, 2000, 1e-24, 0.0);
   }


   x -= x2;
   cout << "||x - x_2|| = " << x.Norml2() << endl;

   socketstream sol_sock_href2(vishost, visport);
   sol_sock_href2.precision(8);
   sol_sock_href2 << "solution\n" << *mesh_href << x_href2 <<  keys
                  << "window_title 'GridFunction on h-refined mesh M_h^-1 * P_p^T * M_p * P_p * P_h * x '"
                  << flush;

   socketstream sol_sock2(vishost, visport);
   sol_sock2.precision(8);
   sol_sock2 << "solution\n" << *mesh << x2 << keys
             << "window_title 'GridFunction on h-refined mesh M^-1 * P_h^T * P_p^T * M_p * P_p * P_h * x '"
             << flush;

}