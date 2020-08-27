#include "mfem.hpp"
#include "cutgridfunc.hpp"
#include "exAdvection_cut.hpp"
#include "cellMerging.hpp"
#include <fstream>
#include <iostream>
using namespace std;
using namespace mfem;
void exact_function(const Vector &x, Vector &v);
double u_exact(const Vector &);
double f_exact(const Vector &);
// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

int main(int argc, char *argv[])
{
   // mesh to be used
   //const char *mesh_file = "../data/periodic-segment.mesh";
   int ref_levels = -1;
   int order = 1;
   double cutsize = 1.0;
   int N = 20;
   bool visualization = 1;
   double scale;
   //Read the mesh from the given mesh file.
   //Mesh *mesh = new Mesh(mesh_file, 1, 2);
   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&cutsize, "-s", "--cutsize",
                  "scale of the cut finite elements.");
   args.AddOption(&N, "-n", "--#elements",
                  "number of mesh elements.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   int deg = 4;
   int np[5] = {20, 50, 100, 200, 300};
   DenseMatrix Norm_err(5, 4);
   for (int d = 0; d < deg; ++d)
   {
      order = d + 1;
      for (int n = 0; n < 5; ++n)
      {
         N = np[n];
         // create mesh using mfem
         Mesh *mesh = new Mesh(N, 1);
         // dimension
         int dim = mesh->Dimension();
         // # mesh elements
         int nels = mesh->GetNE();
         // set the size of cut element
         scale = 1.0 / nels;
         scale = scale * cutsize;

         //Define a finite element space on the mesh. Here we use discontinuous
         //  finite elements of the specified order >= 0.
         FiniteElementCollection *fec = new DG_FECollection(order, dim);
         FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
         cout << "Number of unknowns: " << fespace->GetTrueVSize() << endl;

         // total number of dofs
         int ndofs = fespace->GetTrueVSize();
         // get element dofs
         const FiniteElement *fe = fec->FiniteElementForGeometry(Geometry::SEGMENT);
         const int num_dofs = fe->GetDof();

         GridFunction x(fespace);
         ConstantCoefficient one(1.0);
         ConstantCoefficient zero(0.0);
         // source term
         FunctionCoefficient f(f_exact);
         // exact solution
         FunctionCoefficient u(u_exact);
         // advection velocity
         VectorFunctionCoefficient velocity(dim, velocity_function);
         // linear form
         LinearForm *b = new LinearForm(fespace);
         b->AddDomainIntegrator(new CutDomainLFIntegrator(f, scale, nels));
         b->AddBdrFaceIntegrator(new BoundaryAdvectIntegrator(u, velocity, nels, scale));
         b->Assemble();
         // blinear form
         BilinearForm *a = new BilinearForm(fespace);
         a->AddDomainIntegrator(new TransposeIntegrator(new AdvectionIntegrator(velocity, scale, nels, -1.0)));
         a->AddInteriorFaceIntegrator(new TransposeIntegrator(new DGFaceIntegrator(velocity, scale, nels)));
         a->AddBdrFaceIntegrator(new TransposeIntegrator(new DGFaceIntegrator(velocity, scale, nels)));
         a->Assemble();
         a->Finalize();
         // stiffness matrix
         SparseMatrix &Aold = a->SpMat();
         // ofstream write("stiffmat_cutmerge_old.txt");
         // Aold.PrintMatlab(write);
         // write.close();
         // SparseMatrix P(fespace->GetVSize(), (nels * num_dofs) );
         SparseMatrix P;
         // initialize the cellmerging class
         CellMerging Prolongate(mesh, dim, mesh->GetNE(), fec, fespace, 1, order, 1.0 / cutsize);
         // get the prolongation matrix
         P = Prolongate.getProlongationOperator();
         cout << "Aold size " << Aold.Height() << " x " << Aold.Width() << endl;
         cout << "P size " << P.Height() << " x " << P.Width() << endl;
         SparseMatrix *pm = RAP(P, Aold, P);
         SparseMatrix &A = *pm;
         ofstream write("stiffmat_cutmerge.txt");
         A.PrintMatlab(write);
         write.close();
         cout << "size of A " << A.Height() << " x " << A.Width() << endl;
         // vector that stores the results at nodes (except for the last two elements in case of merged ones)
         Vector y(A.Width());
         for (int k = 0; k < y.Size(); ++k)
         {
            y(k) = 0.0;
         }
         // get P^T b
         Vector bnew(A.Width());
         P.MultTranspose(*b, bnew);
         cout << "size of bnew " << bnew.Size() << endl;
#ifndef MFEM_USE_SUITESPARSE
         // 8. Define a simple symmetric Gauss-Seidel preconditioner and use it to
         //    solve the system Ax=b with PCG in the symmetric case, and GMRES in the
         //    non-symmetric one.
         GSSmoother M(A);
         GMRES(A, M, bnew, y, 1, 1000, 200, 1e-60, 1e-60);
#else
         // 8. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
         UMFPackSolver umf_solver;
         umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
         umf_solver.SetOperator(A);
         umf_solver.Mult(bnew, y);
#endif
         cout << "size of y " << y.Size() << endl;
         cout << "----------------------------- " << endl;
         cout << "solution at nodes with merged cell obtained: " << endl;
         y.Print();
         cout << "----------------------------- " << endl;
         // get x = P y
         P.Mult(y, x);
         // save the solution
         ofstream adj_ofs("dgAdvection.vtk");
         adj_ofs.precision(14);
         mesh->PrintVTK(adj_ofs, 1);
         x.SaveVTK(adj_ofs, "dgAdvSolution", 1);
         adj_ofs.close();
         // cout << x.ComputeL2Error(u) << endl;
         // check the error
         double norm = CutComputeL2Error(x, fespace, u, scale);
         cout << "solution at nodes is: " << endl;
         x.Print();
         cout << "########################################## " << endl;
         cout << "mesh size, h = " << 1.0 / mesh->GetNE() << endl;
         cout << "solution norm: " << norm << endl;
         cout << "########################################## " << endl;
         // 11. Free the used memory.
         Norm_err(n, d) = norm;
         // 11. Free the used memory.
         delete a;
         delete b;
         delete fespace;
         delete fec;
         delete mesh;
      }
   }
   Norm_err.PrintMatlab();
   return 0;
}

void exact_function(const Vector &x, Vector &v)
{
   int dim = x.Size();
   //v(0) = x(0)*x(0)*x(0);
   //v(0) = 2.0;
   v(0) = exp(x(0));
   //v(0) = cos(M_PI * x(0));
   //v(0) = x(0);
}

double u_exact(const Vector &x)
{
   return exp(x(0));
   //return 2.0;
   //return cos(M_PI * x(0));
   //return x(0)*x(0)*x(0);
   //return x(0);
}
double f_exact(const Vector &x)
{
   //return 3*x(0)*x(0);
   return -exp(x(0));
   //return 0.0;
   //return -M_PI*sin(M_PI * x(0));
   //return 1.0;
   // return 2.0*x(0);
}
// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();
   switch (dim)
   {
   case 1:
      v(0) = -1.0;
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

/// function to compute l2 error for cut domain
double CutComputeL2Error(GridFunction &x, FiniteElementSpace *fes,
                         Coefficient &exsol, double scale)
{
   double error = 0.0;
   const FiniteElement *fe;
   ElementTransformation *T;
   Vector vals;
   int p = 2;
   for (int i = 0; i < fes->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      const IntegrationRule *ir;
      int intorder = 2 * fe->GetOrder() + 1; // <----------
      ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      // }
      T = fes->GetElementTransformation(i);
      if (T->ElementNo == fes->GetNE() - 1)
      {
         IntegrationRule *cutir;
         cutir = new IntegrationRule(ir->Size());
         for (int k = 0; k < cutir->GetNPoints(); k++)
         {
            IntegrationPoint &cutip = cutir->IntPoint(k);
            const IntegrationPoint &ip = ir->IntPoint(k);
            cutip.x = (scale * ip.x) / T->Weight();
            cutip.weight = ip.weight;
         }
         x.GetValues(i, *cutir, vals);
         for (int j = 0; j < cutir->GetNPoints(); j++)
         {
            IntegrationPoint &ip = cutir->IntPoint(j);
            T->SetIntPoint(&ip);
            cout << " x is " << fabs(vals(j)) << endl;
            cout << " u_exact is " << exsol.Eval(*T, ip) << endl;
            double err = fabs(vals(j) - exsol.Eval(*T, ip));
            if (p < infinity())
            {
               err = pow(err, p);
               error += ip.weight * scale * err;
            }
            else
            {
               error = std::max(error, err);
            }
         }
      }
      else
      {
         x.GetValues(i, *ir, vals);
         for (int j = 0; j < ir->GetNPoints(); j++)
         {
            const IntegrationPoint &ip = ir->IntPoint(j);
            T->SetIntPoint(&ip);
            double err = fabs(vals(j) - exsol.Eval(*T, ip));
            if (p < infinity())
            {
               err = pow(err, p);
               error += ip.weight * T->Weight() * err;
            }
            else
            {
               error = std::max(error, err);
            }
         }
      }
   }
   if (p < infinity())
   {
      // negative quadrature weights may cause the error to be negative
      if (error < 0.)
      {
         error = -pow(-error, 1. / p);
      }
      else
      {
         error = pow(error, 1. / p);
      }
   }

   return error;
}

void CutDomainLFIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                                   ElementTransformation &Tr,
                                                   Vector &elvect)
{
   int dof = el.GetDof();
   double sf;
   shape.SetSize(dof); // vector of size dof
   elvect.SetSize(dof);
   elvect = 0.0;
   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);
   }
   IntegrationRule *cutir;
   cutir = new IntegrationRule(ir->Size());
   if (Tr.ElementNo == nels - 1)
   {
      for (int k = 0; k < cutir->GetNPoints(); k++)
      {
         IntegrationPoint &cutip = cutir->IntPoint(k);
         const IntegrationPoint &ip = ir->IntPoint(k);
         cutip.x = (scale * ip.x) / Tr.Weight();
         cutip.weight = ip.weight;
      }
   }
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      IntegrationPoint &cutip = cutir->IntPoint(i);
      Tr.SetIntPoint(&ip);
      double val = Tr.Weight() * Q.Eval(Tr, ip);
      el.CalcShape(ip, shape);
      if (Tr.ElementNo == nels - 1)
      {
         Tr.SetIntPoint(&cutip);
         val = scale * Q.Eval(Tr, cutip);
         el.CalcShape(cutip, shape);
      }
      add(elvect, ip.weight * val, shape, elvect);
   }
}

void CutDomainLFIntegrator::AssembleDeltaElementVect(
    const FiniteElement &fe, ElementTransformation &Trans, Vector &elvect)
{
   MFEM_ASSERT(delta != NULL, "coefficient must be DeltaCoefficient");
   elvect.SetSize(fe.GetDof());
   fe.CalcPhysShape(Trans, elvect);
   elvect *= delta->EvalDelta(Trans, Trans.GetIntPoint());
}

void AdvectionIntegrator::AssembleElementMatrix(
    const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   int nd = el.GetDof();
   int dim = el.GetDim();
   double w;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape, adjJ, Q_ir;
   Vector shape, vec2, BdFidxT;
#endif
   elmat.SetSize(nd);
   dshape.SetSize(nd, dim);
   adjJ.SetSize(dim);
   shape.SetSize(nd);
   vec2.SetSize(dim);
   BdFidxT.SetSize(nd);
   Vector vec1;
   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = Trans.OrderGrad(&el) + Trans.Order() + el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), order);
   }
   if (Trans.ElementNo == nels - 1)
   {
      IntegrationRule *cutir;
      cutir = new IntegrationRule(ir->Size());
      for (int k = 0; k < cutir->GetNPoints(); k++)
      {
         IntegrationPoint &cutip = cutir->IntPoint(k);
         const IntegrationPoint &ip = ir->IntPoint(k);
         cutip.x = (scale * ip.x) / Trans.Weight();
         cutip.weight = ip.weight;
      }
      Q->Eval(Q_ir, Trans, *cutir);
      elmat = 0.0;
      for (int i = 0; i < cutir->GetNPoints(); i++)
      {
         IntegrationPoint &ip = cutir->IntPoint(i);
         el.CalcDShape(ip, dshape);
         el.CalcShape(ip, shape);
         Trans.SetIntPoint(&ip);
         CalcAdjugate(Trans.Jacobian(), adjJ);
         adjJ *= scale / Trans.Weight();
         Q_ir.GetColumnReference(i, vec1);
         vec1 *= alpha * ip.weight;
         adjJ.Mult(vec1, vec2);
         dshape.Mult(vec2, BdFidxT);
         AddMultVWt(shape, BdFidxT, elmat);
      }
   }
   else
   {
      Q->Eval(Q_ir, Trans, *ir);
      elmat = 0.0;
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         el.CalcDShape(ip, dshape);
         el.CalcShape(ip, shape);
         Trans.SetIntPoint(&ip);
         CalcAdjugate(Trans.Jacobian(), adjJ);
         Q_ir.GetColumnReference(i, vec1);
         vec1 *= alpha * ip.weight;
         adjJ.Mult(vec1, vec2);
         dshape.Mult(vec2, BdFidxT);
         AddMultVWt(shape, BdFidxT, elmat);
      }
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
      shape2.SetSize(ndof2);
   }
   else
   {
      ndof2 = 0;
   }
   shape1.SetSize(ndof1);
   elmat.SetSize(ndof1 + ndof2);
   elmat = 0.0;
   const IntegrationRule *ir = IntRule;
   // get integration rule
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
   // get the elmat matrix
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip1, eip2;
      Trans.Loc1.Transform(ip, eip1);
      if (Trans.Elem1No == nels - 1)
      {
         eip1.x = (scale * eip1.x) / Trans.Elem1->Weight();
      }
      if (ndof2)
      {
         Trans.Loc2.Transform(ip, eip2);
         if (Trans.Elem2No == nels - 1)
         {
            eip2.x = (scale * eip2.x) / Trans.Elem2->Weight();
         }
      }

      el1.CalcShape(eip1, shape1);
      Trans.Face->SetIntPoint(&ip);
      Trans.Elem1->SetIntPoint(&eip1);
      // evaluate the velocity coefficient
      u->Eval(vu, *Trans.Elem1, eip1);
      nor(0) = 2 * eip1.x - 1.0;
      // normal velocity
      if (Trans.Elem1No == nels - 1)
      {
         nor(0) = 1.0;
      }
      un = vu * nor;
      a = un;
      b = fabs(un);
      w = 0.5 * ip.weight * (a + b);
      if (w != 0.0)
      {
         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < ndof1; j++)
            {
               elmat(i, j) += w * shape1(i) * shape1(j);
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

         w = 0.5 * ip.weight * (a - b);
         if (w != 0.0)
         {
            for (int i = 0; i < ndof2; i++)
               for (int j = 0; j < ndof2; j++)
               {
                  elmat(ndof1 + i, ndof1 + j) -= w * shape2(i) * shape2(j);
               }

            for (int i = 0; i < ndof1; i++)
               for (int j = 0; j < ndof2; j++)
               {
                  elmat(ndof1 + j, i) += w * shape1(i) * shape2(j);
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
   Vector vu(vu_data, dim), nor(nor_data, dim);
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
      if (Tr.Elem1No == nels - 1)
      {
         eip.x = (scale * eip.x) / Tr.Elem1->Weight();
      }
      el.CalcShape(eip, shape);
      Tr.Face->SetIntPoint(&ip);
      u->Eval(vu, *Tr.Elem1, eip);
      nor(0) = 2 * eip.x - 1.0;
      if (Tr.Elem1No == nels - 1)
      {
         nor(0) = 1;
      }
      un = vu * nor;
      w = -0.5 * (un - fabs(un));
      w *= ip.weight * uD->Eval(*Tr.Elem1, eip);
      elvect.Add(w, shape);
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

/// functions for `CellMerging` class
mfem::SparseMatrix CellMerging::getProlongationOperator() const
{
   const FiniteElement *fe = fec->FiniteElementForGeometry(Geometry::SEGMENT);
   const int num_dofs = fe->GetDof();

   cout << "#dofs " << num_dofs << endl;
   if (scale == 0.0)
   {
      mfem::SparseMatrix P(fespace->GetVSize(), fespace->GetVSize());
      cout << "cP size " << P.Height() << " x " << P.Width() << endl;
      int mat_size = P.Height();
      mfem::DenseMatrix local_id_mat(mat_size);
      for (int i = 0; i < mat_size; ++i)
      {
         local_id_mat(i, i) = 1.0;
      }
      Array<int> col_index(mat_size);
      Array<int> row_index(mat_size);
      for (int k = 0; k < mat_size; ++k)
      {
         col_index[k] = k;
         row_index[k] = k;
      }
      P.SetSubMatrix(col_index, row_index, local_id_mat, 1);
      P.Finalize();
      cout << "Check cP size: " << P.Height() << " x " << P.Width() << '\n';
      ofstream write("cp.txt");
      P.PrintMatlab(write);
      write.close();
      return P;
   }
   else
   {
      // Prolongation matrix
      mfem::SparseMatrix P(fespace->GetVSize(), (nEle * num_dofs) - num_dofs);
      cout << "cP size " << P.Height() << " x " << P.Width() << endl;
      int mat_size = P.Height() - (num_dofs);
      // it should be identity for all elements but last
      mfem::DenseMatrix local_id_mat(mat_size);
      for (int i = 0; i < mat_size; ++i)
      {
         local_id_mat(i, i) = 1.0;
      }
      Array<int> col_index(mat_size);
      Array<int> row_index(mat_size);
      for (int k = 0; k < mat_size; ++k)
      {
         col_index[k] = k;
         row_index[k] = k;
      }
      P.SetSubMatrix(col_index, row_index, local_id_mat, 1);
      // vector that contains element id (resize to zero )
      mfem::Array<int> elmt_id;
      mfem::DenseMatrix cut_mat, quad_mat, local_mat;
      Array<int> c_index(num_dofs);
      Array<int> r_index(num_dofs);
      r_index.LoseData();
      c_index.LoseData();
      // last and second last elements are merged
      elmt_id.Append(nEle - 1);
      elmt_id.Append(nEle - 2);
      // get the quadrature point matrices
      BuildNeighbourMat(elmt_id, cut_mat, quad_mat);
      cout << "Neighbour mat is built " << endl;
      // Get the `x` coordinate to shift the polynomials
      Array<int> vertices(dim + 1);
      mesh->GetElement(nEle - 2)->GetVertices(vertices);
      cout << "vertices are " << endl;
      vertices.Print();
      double *xshift1, *xshift;
      mfem::Vector shift_coord(dim);
      double l = 0.0;
      xshift = &l;
      for (int j = 0; j < vertices.Size(); ++j)
      {
         cout << "here goes " << endl;
         xshift1 = mesh->GetVertex(vertices[j]);
         if (xshift1[0] > xshift[0])
         {
            shift_coord[0] = xshift1[0];
         }
         xshift[0] = xshift1[0];
         cout << xshift[0] << endl;
      }
      // GetElementCenter(nEle - 2, shift_coord);
      shift_coord.Print();
      //  3. build the local reconstruction matrix
      buildLSInterpolation(dim, degree, shift_coord, cut_mat, quad_mat, local_mat);
      cout << "Local reconstruction matrix built " << endl;
      //cout << "Local reconstruction matrix R:\n";
      // local_mat.Print(cout, local_mat.Width());
      // get the row and column indices to put the local
      //interpolation matrix in the prolongation matrix
      Array<int> el_dofs;
      fespace->GetElementVDofs(elmt_id[0], el_dofs);
      int v = 0;
      el_dofs.GetSubArray(v * num_dofs, num_dofs, r_index);
      cout << "r_index " << endl;
      r_index.Print();
      int ind = P.Width() - num_dofs;
      for (int k = 0; k < num_dofs; ++k)
      {
         c_index.Append(ind);
         ++ind;
      }
      cout << "c_index " << endl;
      c_index.Print();
      P.SetSubMatrix(r_index, c_index, local_mat, 1);
      P.Finalize();
      cout << "Check cP size: " << P.Height() << " x " << P.Width() << '\n';
      ofstream write("cp.txt");
      P.PrintMatlab(write);
      write.close();
      return P;
   }
}

void CellMerging::BuildNeighbourMat(const mfem::Array<int> &elmt_id,
                                    mfem::DenseMatrix &cut_quad,
                                    mfem::DenseMatrix &mat_quad) const
{
   // resize the DenseMatrices and clean the data
   int num_el = elmt_id.Size();
   cut_quad.Clear();
   cout << "elements are " << endl;
   elmt_id.Print();
   const FiniteElement *fe = fec->FiniteElementForGeometry(Geometry::SEGMENT);
   const int num_dofs = fe->GetDof();
   // vectors that hold coordinates of quadrature points
   // used for duplication tests
   vector<double> quad_data;
   vector<double> cut_quad_data;
   Vector quad_coord(dim); // used to store quadrature coordinate temperally
   Vector cut_quad_coord(dim);
   ElementTransformation *eltransf;
   for (int j = 0; j < num_el; j++)
   {
      // deal with quadrature points
      eltransf = mesh->GetElementTransformation(elmt_id[j]);
      if (eltransf->ElementNo == nEle - 1)
      {
         for (int k = 0; k < num_dofs; k++)
         {
            const IntegrationPoint &eip = fe->GetNodes().IntPoint(k);
            eltransf->Transform(eip, cut_quad_coord);
            cout << "cut element " << elmt_id[j] << " int rule: " << endl;
            cut_quad_coord.Print();
            for (int di = 0; di < dim; di++)
            {
               cut_quad_data.push_back(cut_quad_coord(di));
            }
         }
      }
      else
      {
         for (int k = 0; k < num_dofs; k++)
         {
            const IntegrationPoint &eip = fe->GetNodes().IntPoint(k);
            eltransf->Transform(eip, quad_coord);
            cout << "element " << elmt_id[j] << " int rule: " << endl;
            quad_coord.Print();
            for (int di = 0; di < dim; di++)
            {
               quad_data.push_back(quad_coord(di));
            }
         }
      }
   }

   // reset the quadrature point matrix
   mat_quad.Clear();
   int num_col = quad_data.size() / dim;
   mat_quad.SetSize(dim, num_col);
   for (int i = 0; i < num_col; i++)
   {
      for (int j = 0; j < dim; j++)
      {
         mat_quad(j, i) = quad_data[i * dim + j];
      }
   }
   num_col = cut_quad_data.size() / dim;
   cut_quad.SetSize(dim, num_col);
   for (int i = 0; i < num_col; i++)
   {
      for (int j = 0; j < dim; j++)
      {
         cut_quad(j, i) = cut_quad_data[i * dim + j];
      }
   }
}

void CellMerging::GetElementCenter(int id, mfem::Vector &cent) const
{
   cent.SetSize(mesh->Dimension());
   int geom = mesh->GetElement(id)->GetGeometryType();
   ElementTransformation *eltransf = mesh->GetElementTransformation(id);
   eltransf->Transform(Geometries.GetCenter(geom), cent);
}

void buildLSInterpolation(int dim, int degree, const Vector &x_center, const DenseMatrix &x_cut_quad,
                          const DenseMatrix &x_quad, DenseMatrix &interp)
{
   // get the number of quadrature points and elements.
   int num_quad = x_quad.Width();
   int num_int = x_cut_quad.Width();
   int ndofs = degree + 1;
   cout << " --- --- --- ---  --- --- --- --- " << endl;
   cout << "ndofs in buildLS " << ndofs << endl;
   cout << "cut quad mat " << endl;
   x_cut_quad.Print();
   cout << "quad mat " << endl;
   x_quad.Print();
   // number of total polynomial basis functions
   int num_basis = -1;
   num_basis = degree + 1;
   cout << "num_quad " << num_quad << endl;
   // build Vandemonde matrix
   mfem::DenseMatrix V(num_quad, num_basis);
   for (int n = 0; n < num_quad; ++n)
   {
      double dx = x_quad(0, n) - x_center(0);
      for (int p = 0; p <= degree; ++p)
      {
         V(n, p) = pow(dx, p);
      }
   }
   cout << "Vandermonde matrix is: " << endl;
   V.PrintMatlab();
   // Set the RHS for the LS problem (it's the identity matrix)
   // This will store the solution, that is, the basis coefficients, hence
   // the name `coeff`
   mfem::DenseMatrix coeff(num_quad, num_quad);
   coeff = 0.0;
   for (int i = 0; i < num_quad; ++i)
   {
      coeff(i, i) = 1.0;
   }
   cout << "RHS is " << endl;
   coeff.PrintMatlab();
   // Set-up and solve the least-squares problem using LAPACK's dgels
   char TRANS = 'N';
   int info;
   int lwork = 2 * num_quad * num_basis;
   //int lwork = (num_int * num_basis) + (3 * num_basis) + 1;
   double work[lwork];
   int rank;
   Array<int> jpvt;
   jpvt.SetSize(num_basis);
   jpvt = 0;
   double rcond = 1e-16;
   dgels_(&TRANS, &num_quad, &num_basis, &num_quad, V.GetData(), &num_quad,
          coeff.GetData(), &num_quad, work, &lwork, &info);
   // dgelsy_(&num_int, &num_basis, &num_int, V.GetData(), &num_int, coeff.GetData(),
   //         &num_int, jpvt.GetData(), &rcond, &rank, work, &lwork, &info);
   cout << "info is " << info << endl;
   MFEM_ASSERT(info == 0, "Fail to solve the underdetermined system.\n");
   cout << "coeff mat: "
        << "\n";
   coeff.PrintMatlab();
   // Perform matrix-matrix multiplication between basis functions evalauted at
   // quadrature nodes and basis function coefficients.
   interp.SetSize(num_quad, num_quad);
   interp = 0.0;
   // loop over quadrature points
   for (int j = 0; j < num_quad; ++j)
   {
      double dx = x_cut_quad(0, j) - x_center(0);
      // loop over the cut element quad points
      for (int i = 0; i < num_quad; ++i)
      {
         for (int p = 0; p <= degree; ++p)
         {
            interp(j, i) += pow(dx, p) * coeff(p, i);
         }
      }
   }
   cout << "interpolation matrix: " << endl;
   interp.PrintMatlab();
   // loop over quadrature points
   for (int j = 0; j < num_quad; ++j)
   {
      for (int p = 0; p <= degree; ++p)
      {
         // loop over the element centers
         double poly_at_quad = 0.0;
         double dx = x_cut_quad(0, j) - x_center(0);
         for (int i = 0; i < num_int; ++i)
         {
            poly_at_quad += interp(j, i) * pow(dx, p);
         }
         double exact = ((p == 0)) ? 1.0 : 0.0;
         mfem::out << "polynomial interpolation error (" << p << ","
                   << ") = " << fabs(exact - poly_at_quad) << endl;
         // MFEM_ASSERT(fabs(exact - poly_at_quad) <= 1e-12,
         //             "Interpolation operator does not interpolate exactly!\n");
      }
   }
}

///functions related to CutGridFunction class
CutGridFunction::CutGridFunction(FiniteElementSpace *f)
{
   SetSize(f->GetVDim() * f->GetNE());
   fes = f;
   fec = NULL;
   sequence = f->GetSequence();
   UseDevice(true);
}

void CutGridFunction::ProjectCoefficient(VectorCoefficient &coeff)
{
   int vdim = fes->GetVDim();
   Array<int> vdofs(vdim);
   Vector vals;

   int geom = fes->GetMesh()->GetElement(0)->GetGeometryType();
   const IntegrationPoint &cent = Geometries.GetCenter(geom);
   const FiniteElement *fe;
   ElementTransformation *eltransf;
   for (int i = 0; i < fes->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      // Get the indices of dofs
      for (int j = 0; j < vdim; j++)
      {
         vdofs[j] = i * vdim + j;
      }

      eltransf = fes->GetElementTransformation(i);
      eltransf->SetIntPoint(&cent);
      vals.SetSize(vdofs.Size());
      coeff.Eval(vals, *eltransf, cent);

      if (fe->GetMapType() == 1)
      {
         vals(i) *= eltransf->Weight();
      }
      SetSubVector(vdofs, vals);
   }
}

CutGridFunction &CutGridFunction::operator=(const Vector &v)
{
   std::cout << "cent = is called.\n";
   MFEM_ASSERT(fes && v.Size() == fes->GetTrueVSize(), "");
   Vector::operator=(v);
   return *this;
}

CutGridFunction &CutGridFunction::operator=(double value)
{
   Vector::operator=(value);
   return *this;
}
