#include "mfem.hpp"
#include "exAdvection.hpp"
using namespace std;
using namespace mfem;
double u_exact(const Vector &);
double f_exact(const Vector &);
// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

int main(int argc, char *argv[])
{
   // 1. mesh to be used
   const char *mesh_file = "../data/periodic-segment.mesh";
   int ref_levels = -1;
   int order = 1;
   bool visualization = 1;
   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   //    NURBS meshes are projected to second order meshes.
   //Mesh *mesh = new Mesh(mesh_file, 1, 2);
   Mesh *mesh = new Mesh(5, 1);
   int dim = mesh->Dimension();
   cout << "number of elements " << mesh->GetNE() << endl;
   ofstream sol_ofv("square_disc_mesh.vtk");
   sol_ofv.precision(14);
   mesh->PrintVTK(sol_ofv, 1);
   // 4. Define a finite element space on the mesh. Here we use discontinuous
   //    finite elements of the specified order >= 0.
   FiniteElementCollection *fec = new DG_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "Number of unknowns: " << fespace->GetVSize() << endl;
   // 5. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system.
   LinearForm *b = new LinearForm(fespace);
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   FunctionCoefficient f(f_exact);
   FunctionCoefficient u(u_exact);
   VectorFunctionCoefficient velocity(dim, velocity_function);
   b->AddDomainIntegrator(new DomainLFIntegrator(f, 1.0));
   b->AddBdrFaceIntegrator(new BoundaryAdvectIntegrator(zero, velocity, -1.0, -0.5));
   b->Assemble();
   GridFunction x(fespace);
   x = 0.0;
   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   a->AddInteriorFaceIntegrator(new DGFaceIntegrator(velocity, 1.0, -0.5));
   a->AddBdrFaceIntegrator(new DGFaceIntegrator(velocity, 1.0, -0.5));
   a->Assemble();
   a->Finalize();
   const SparseMatrix &A = a->SpMat();
   A.Print();
   ofstream write("stiffmat.txt");
   A.PrintMatlab(write);
   write.close();
#ifndef MFEM_USE_SUITESPARSE
   // 8. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //    solve the system Ax=b with PCG in the symmetric case, and GMRES in the
   //    non-symmetric one.
   GSSmoother M(A);
   //PCG(A, M, *b, x, 1, 1000, 1e-12, 0.0);
   // else
   // {
   GMRES(A, M, *b, x, 1, 1000, 10, 1e-12, 0.0);
   // }
#else
   // 8. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   umf_solver.SetOperator(A);
   umf_solver.Mult(*b, x);
#endif
   ofstream adj_ofs("dgAdvlap.vtk");
   adj_ofs.precision(14);
   mesh->PrintVTK(adj_ofs, 1);
   x.SaveVTK(adj_ofs, "dgAdvSolution", 1);
   adj_ofs.close();
   cout << "solution norm: " << endl;
   cout << x.ComputeL2Error(u) << endl;
   // 11. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   delete fec;
   delete mesh;
   return 0;
}

double u_exact(const Vector &x)
{
   //return exp(x(0));
   return 2 * x(0);
}
double f_exact(const Vector &x)
{
   //return -exp(x(0));
   return -2.0;
}
// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();
   switch (dim)
   {
   case 1:
      v(0) = 1.0;
      break;
   case 2:
      v(0) = sqrt(2. / 3.);
      v(1) = sqrt(1. / 3.);
      break;
   case 3:
      v(0) = sqrt(3. / 6.);
      v(1) = sqrt(2. / 6.);
      v(2) = sqrt(1. / 6.);
      break;
   }
}
// assemble the elmat for interior and boundary faces
void DGFaceIntegrator::AssembleFaceMatrix(const FiniteElement &el1,
                                          const FiniteElement &el2,
                                          FaceElementTransformations &Trans,
                                          DenseMatrix &elmat)
{
   int dim, ndof1, ndof2;
   double un, a, b, w;
   dim = el1.GetDim();
   ndof1 = el1.GetDof();
   Vector vu(dim), nor(dim);
   if (Trans.Elem2No >= 0)
   {
      ndof2 = el2.GetDof();
   }
   else
   {
      ndof2 = 0;
   }
   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);
   elmat.SetSize(ndof1 + ndof2);
   elmat = 0.0;
   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (Trans.Elem2No >= 0)
         order = (min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
                  2 * max(el1.GetOrder(), el2.GetOrder()));
      else
      {
         order = Trans.Elem1->OrderW() + 2 * el1.GetOrder();
      }
      if (el1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.FaceGeom, order);
   }
   // cout << "face is " << Trans.Face->ElementNo << " elements are " <<  Trans.Elem1No << " , " << Trans.Elem2No << endl;
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip1, eip2;
      Trans.Loc1.Transform(ip, eip1);
      if (ndof2)
      {
         Trans.Loc2.Transform(ip, eip2);
      }
      el1.CalcShape(eip1, shape1);
      Trans.Face->SetIntPoint(&ip);
      Trans.Elem1->SetIntPoint(&eip1);
      u->Eval(vu, *Trans.Elem1, eip1);
      nor(0) = 2 * eip1.x - 1.0;
      un = vu * nor;
      a = 0.5 * alpha * un;
      b = beta * fabs(un);
      w = ip.weight * (a + b);
      if (w != 0.0)
      {
         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < ndof1; j++)
            {
               elmat(j, i) += w * shape1(i) * shape1(j);
            }
      }
      if (ndof2)
      {
         el2.CalcShape(eip2, shape2);

         if (w != 0.0)
            for (int i = 0; i < ndof2; i++)
               for (int j = 0; j < ndof1; j++)
               {
                  elmat(j, ndof1 + i) -= w * shape2(i) * shape1(j);
               }

         w = ip.weight * (b - a);
         if (w != 0.0)
         {
            for (int i = 0; i < ndof2; i++)
               for (int j = 0; j < ndof2; j++)
               {
                  elmat(ndof1 + j, ndof1 + i) += w * shape2(i) * shape2(j);
               }

            for (int i = 0; i < ndof1; i++)
               for (int j = 0; j < ndof2; j++)
               {
                  elmat(ndof1 + j, i) -= w * shape1(i) * shape2(j);
               }
         }
      }
   }
}

void BoundaryAdvectIntegrator::AssembleRHSElementVect(
    const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   int dim, ndof, order;
   double un, w, vu_data[3], nor_data[3];
   dim = el.GetDim();
   ndof = el.GetDof();
   elvect.SetSize(ndof);
   if (Tr.Face->ElementNo == 5)
   {
      elvect = 0.0;
   }
   else
   {
      Vector vu(vu_data, dim), nor(nor_data, dim);
      cout << "face is " << Tr.Face->ElementNo << endl;

      shape.SetSize(ndof);
      elvect = 0.0;
      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
      {
         // Assuming order(u)==order(mesh)
         order = Tr.Elem1->OrderW() + 2 * el.GetOrder();
         if (el.Space() == FunctionSpace::Pk)
         {
            order++;
         }
         ir = &IntRules.Get(Tr.FaceGeom, order);
      }
      for (int p = 0; p < ir->GetNPoints(); p++)
      {
         const IntegrationPoint &ip = ir->IntPoint(p);
         IntegrationPoint eip;
         Tr.Loc1.Transform(ip, eip);
         el.CalcShape(eip, shape);
         Tr.Face->SetIntPoint(&ip);
         u->Eval(vu, *Tr.Elem1, eip);
         nor(0) = 2 * eip.x - 1.0;
         un = vu * nor;
         w = 0.5 * alpha * un - beta * fabs(un);
         w *= ip.weight * uD->Eval(*Tr.Elem1, eip);
         elvect.Add(w, shape);
      }
   }
}

void BoundaryAdvectIntegrator::AssembleRHSElementVect(
    const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("BoundaryFlowIntegrator::AssembleRHSElementVect\n"
              "  is not implemented as boundary integrator!\n"
              "  Use LinearForm::AddBdrFaceIntegrator instead of\n"
              "  LinearForm::AddBoundaryIntegrator.");
}
