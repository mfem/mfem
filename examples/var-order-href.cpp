

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;


double u(const Vector & x);
GridFunction* TransferToMaxOrder(const GridFunction *x);
int RandomPRefinement(FiniteElementSpace & fes);

int main()
{
   char vishost[] = "localhost";
   int  visport   = 19916;

   int order = 2;
   int dim = 2;

   Mesh mesh = Mesh::MakeCartesian2D(4,4,mfem::Element::QUADRILATERAL,1.,1.);
   mesh.EnsureNCMesh();
   mesh.RandomRefinement(0.5);

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);

   RandomPRefinement(fes);

   ConstantCoefficient one(1.);
   FunctionCoefficient f(u);
   GridFunction gf(&fes);
   gf.ProjectCoefficient(f);

   L2_FECollection ordersfec(0, dim);
   FiniteElementSpace ordersfes(&mesh, &ordersfec);
   GridFunction orders0(&ordersfes);
   for (int i = 0; i<mesh.GetNE(); i++)
   {
      orders0(i) = fes.GetElementOrder(i);
   }

   // visualize initial mesh and solution
   {
      socketstream gf0_sock(vishost, visport);
      gf0_sock.precision(8);
      GridFunction * pgf = TransferToMaxOrder(&gf);
      gf0_sock << "solution\n" << mesh << *pgf
               << "window_title 'GridFunction on the coarse mesh'" << flush;

      socketstream mesh0_sock(vishost, visport);
      mesh0_sock.precision(8);
      mesh0_sock << "solution\n" << mesh << orders0
                 << "window_title 'Coarse mesh orders'" << flush;
      delete pgf;
   }

   // copy the mesh to check transfer operator
   Mesh cmesh(mesh);
   FiniteElementSpace cfes(&cmesh, &fec);
   for (int i = 0; i<cmesh.GetNE(); i++)
   {
      cfes.SetElementOrder(i,fes.GetElementOrder(i));
   }
   cfes.Update(false);
   mesh.RandomRefinement(0.5);
   fes.Update();
   gf.Update();


   // visualize initial mesh and solution
   {
      FiniteElementSpace orders1fes(&mesh, &ordersfec);
      GridFunction orders1(&orders1fes);
      for (int i = 0; i<mesh.GetNE(); i++)
      {
         orders1(i) = fes.GetElementOrder(i);
      }

      socketstream gf1_sock(vishost, visport);
      gf1_sock.precision(8);
      GridFunction * pgf1 = TransferToMaxOrder(&gf);
      gf1_sock << "solution\n" << mesh << *pgf1
               << "window_title 'GridFunction on the fine mesh after transfer'" << flush;

      socketstream mesh1_sock(vishost, visport);
      mesh1_sock.precision(8);
      mesh1_sock << "solution\n" << mesh << orders1
                 << "window_title 'Fine mesh orders (after h-refinement)'" << flush;
      delete pgf1;
   }

   // Transfer operator
   OperatorHandle T;
   T.SetType(Operator::MFEM_SPARSEMAT);
   fes.GetTransferOperator(cfes,T);

   GridFunction gft(&cfes);
   Vector trueX(fes.GetTrueVSize());
   Vector trueY(fes.GetTrueVSize());
   Vector Y(fes.GetVSize());

   {
      BilinearForm a(&fes);
      a.AddDomainIntegrator(new MassIntegrator());
      a.Assemble();
      a.Finalize();
      SparseMatrix M;
      Array<int> empty;
      a.FormSystemMatrix(empty,M);

      const SparseMatrix * R = fes.GetHpRestrictionMatrix();

      if (R)
      {
         R->Mult(gf,trueX);
      }
      else
      {
         trueX.MakeRef(gf,0);
      }
      M.Mult(trueX,trueY);
      if (R)
      {
         R->MultTranspose(trueY,Y);
      }
      else
      {
         Y.MakeRef(trueY,0);
      }
   }

   // map to course space
   Vector cY(cfes.GetVSize());
   T->MultTranspose(Y,cY);

   Vector truecY(cfes.GetTrueVSize());
   Vector truecX(cfes.GetTrueVSize());
   GridFunction cX(&cfes);

   {
      const SparseMatrix * P = cfes.GetConformingProlongation();
      if (P)
      {
         P->MultTranspose(cY,truecY);
      }
      else
      {
         truecY.MakeRef(cY,0);
      }

      BilinearForm ac(&cfes);
      ac.AddDomainIntegrator(new MassIntegrator());
      ac.Assemble();
      ac.Finalize();
      SparseMatrix Mc;
      Array<int> empty;
      ac.FormSystemMatrix(empty,Mc);
      UMFPackSolver invMc(Mc);
      invMc.Mult(truecY,truecX);

      if (P)
      {
         P->Mult(truecX,cX);
      }
      else
      {
         cX.MakeRef(truecX,0);
      }
   }


   {
      socketstream gf2_sock(vishost, visport);
      gf2_sock.precision(8);
      GridFunction * pgf2 = TransferToMaxOrder(&cX);
      gf2_sock << "solution\n" << cmesh << *pgf2
               << "window_title 'GridFunction on the coarse mesh after transfer (transpose)'"
               << flush;
   }

   return 0;
}



double u(const Vector & x)
{
   return sin(M_PI*(x.Sum()));
}

GridFunction* TransferToMaxOrder(const GridFunction *x)
{
   const FiniteElementSpace *fespace = x->FESpace();
   Mesh *mesh = fespace->GetMesh();
   const FiniteElementCollection *fec = fespace->FEColl();

   // find the max order in the space
   int max_order = 1;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      max_order = std::max(fespace->GetElementOrder(i), max_order);
   }

   // create a visualization space of max order for all elements
   FiniteElementCollection *l2fec =
      new L2_FECollection(max_order, mesh->Dimension(), BasisType::GaussLobatto);
   FiniteElementSpace *l2space = new FiniteElementSpace(mesh, l2fec);

   IsoparametricTransformation T;
   DenseMatrix I;

   GridFunction *prolonged_x = new GridFunction(l2space);

   // interpolate solution vector in the larger space
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      Geometry::Type geom = mesh->GetElementGeometry(i);
      T.SetIdentityTransformation(geom);

      Array<int> dofs;
      fespace->GetElementDofs(i, dofs);
      Vector elemvect, l2vect;
      x->GetSubVector(dofs, elemvect);

      const auto *fe = fec->GetFE(geom, fespace->GetElementOrder(i));
      const auto *l2fe = l2fec->GetFE(geom, max_order);

      l2fe->GetTransferMatrix(*fe, T, I);
      l2space->GetElementDofs(i, dofs);
      l2vect.SetSize(dofs.Size());

      I.Mult(elemvect, l2vect);
      prolonged_x->SetSubVector(dofs, l2vect);
   }

   prolonged_x->MakeOwner(l2fec);
   return prolonged_x;
}

int RandomPRefinement(FiniteElementSpace & fes)
{
   Mesh *mesh = fes.GetMesh();
   int maxorder = 0;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      const int order = fes.GetElementOrder(i);
      maxorder = std::max(maxorder,order);
      if ((double) rand() / RAND_MAX < 0.5)
      {
         fes.SetElementOrder(i,order+1);
         maxorder = std::max(maxorder,order+1);
      }
   }
   fes.Update(false);
   return maxorder;
}