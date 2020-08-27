#include "mfem.hpp"
#include "cutgridfunc.hpp"
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
   // 1. mesh to be used
   //const char *mesh_file = "../data/periodic-segment.mesh";
   int ref_levels = -1;
   int order = 1;
   int cutsize = 1;
   int N = 20;
   bool visualization = 1;
   double scale;
   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   //    NURBS meshes are projected to second order meshes.
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
   Mesh *mesh = new Mesh(N, 1);
   int dim = mesh->Dimension();
   cout << "number of elements " << mesh->GetNE() << endl;
   ofstream sol_ofv("square_disc_mesh.vtk");
   sol_ofv.precision(14);
   mesh->PrintVTK(sol_ofv, 1);
   int nels = mesh->GetNE();
   // 4. Define a finite element space on the mesh. Here we use discontinuous
   //    finite elements of the specified order >= 0.
   FiniteElementCollection *fec = new DG_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "Number of unknowns: " << fespace->GetTrueVSize() << endl;
   // 5. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system.
   /// cut cell finite element space
   FiniteElementSpace *fes = new CellMerging(mesh, dim, mesh->GetNE(), fec, 1, Ordering::byVDIM, order);
   // // cout << "Number of unknowns: " << fes->GetTrueVSize() << endl;
   // // cout << "#dofs " << fes->GetNDofs() << endl;
   // // 5. Set up the linear form b(.) which corresponds to the right-hand side of
   // //    the FEM linear system.
   // GridFunction y(fes);
   // y = 0.0;
   // cout << "cut grid function created " << endl;
   // VectorFunctionCoefficient exact(dim, exact_function);
   // y.ProjectCoefficient(exact);
   // cout << "exact solution is " << endl;
   // y.Print();
   // cout << "check if the prolongation matrix is correct " << endl;
   GridFunction x(fespace);
   // CellMerging interpolate(mesh, dim, mesh->GetNE(), fec, 1, Ordering::byVDIM, order);
   // mfem::DenseMatrix cent_mat, quad_mat, local_mat;
   // // vector that contains element id (resize to zero )
   // mfem::Array<int> elmt_id;
   // cout << "flag 1 " << endl;
   // elmt_id.Append(nels-1);
   // elmt_id.Append(nels-2);
   // cout << "flag 2 " << endl;
   // elmt_id.Print();
   // interpolate.BuildNeighbourMat(elmt_id, cent_mat, quad_mat);
   // buildLSInterpolation(dim, order, cent_mat, quad_mat, local_mat);
   //    // 4. assemble them back to prolongation matrix
   // interpolate.AssembleProlongationMatrix(elmt_id, local_mat);
   int ndofs= fespace->GetTrueVSize();
   Vector y(ndofs-(2*order));
   for (int k=0; k < (ndofs-(2*order)); ++k)
   {
      y(k) = 2.0;
   }
   fes->GetProlongationMatrix()->Mult(y, x);
   x.Print();
   // int nels = mesh->GetNE();
   // set the size of cut element
   scale = 1.0 / nels;
   scale = scale / cutsize;
   delete fespace;
   delete fec;
   delete mesh;
   return 0;
}

void exact_function(const Vector &x, Vector &v)
{
   int dim = x.Size();
   //v(0) = x(0)*x(0)*x(0);
   v(0) = 2.0;
   //v(0) = exp(x(0));
   //v(0) = cos(M_PI * x(0));
   //v(0) = x(0);
}

double u_exact(const Vector &x)
{
   //return exp(x(0));
   return 2.0;
   //return cos(M_PI * x(0));
   //return x(0)*x(0)*x(0);
   //return x(0);
}
double f_exact(const Vector &x)
{
   //return 3*x(0)*x(0);
   //return -exp(x(0));
   return 0.0;
   //return -M_PI*sin(M_PI * x(0));
   //return -1.0;
   // return 2.0*x(0);
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

/// functions for `CellMerging` class

void CellMerging::BuildProlongation() const
{
   const FiniteElement *fe = fec->FiniteElementForGeometry(Geometry::SEGMENT);
   const int num_dofs = fe->GetDof();
   cout << "#dofs " << num_dofs << endl;
   // allocate the space for the prolongation matrix
   // this step should be done in the constructor (probably)
   // should it be GetTrueVSize() ? or GetVSize()?
   // need a new method that directly construct a CSR format sparsematrix ？
   cP = new mfem::SparseMatrix(GetVSize(), (vdim * nEle * num_dofs) - 2);
   // // loop over all the element:
   // // 1. build the patch for each element,
   // // 2. construct the local reconstruction operator
   // // 3. assemble local reconstruction operator
   mfem::DenseMatrix local_id_mat(cP->Height() - (2 * num_dofs));
   int mat_size = cP->Height() - (2 * num_dofs);
   for (int i = 0; i < (cP->Height() - (2 * num_dofs)); ++i)
   {
      local_id_mat(i, i) = 1.0;
   }
   local_id_mat.PrintMatlab();
   // Array<int> col_index(mat_size);
   // Array<int> row_index(mat_size);
   Array<int> col_index(mat_size);
   Array<int> row_index(mat_size);
   for (int k=0; k<mat_size; ++k)
   {
       col_index[k] = k;
       row_index[k] = k;
   }
   // col_index[0] = 0;
   // row_index[0] = 0;
   // col_index[1] = 1;
   // row_index[1] = 1;
   cP->SetSubMatrix(col_index, row_index, local_id_mat, 1);
   // // vector that contains element id (resize to zero )
   mfem::Array<int> elmt_id;
   mfem::Array<int> merge_els;
   mfem::DenseMatrix cent_mat, quad_mat, local_mat;
   merge_els.Append(nEle - 2);
   merge_els.Append(nEle - 1);

   cout << "The size of the prolongation matrix is " << cP->Height() << " x " << cP->Width() << '\n';
   //int degree_actual;
   for (int i = 0; i < merge_els.Size(); i++)
   {
      //GetNeighbourSet(i, nelmt, elmt_id);
      // cout << "neighbours are set " << endl;
      // 2. build the quadrature and barycenter coordinate matrices
      elmt_id.LoseData();
      if (merge_els[i] == nEle - 1)
      {
         elmt_id.Append(nEle - 1);
         elmt_id.Append(nEle - 2);
      }
      else
      {
         elmt_id.Append(nEle - 2);
         elmt_id.Append(nEle - 1);
      }
      elmt_id.Print();
      BuildNeighbourMat(elmt_id, cent_mat, quad_mat);
      //    cout << "Neighbour mat is built " << endl;
      //    // 3. buil the loacl reconstruction matrix
      buildLSInterpolation(dim, degree, cent_mat, quad_mat, local_mat);
      //    // cout << "Local reconstruction matrix built " << endl;
      //    //cout << "Local reconstruction matrix R:\n";
      //    // local_mat.Print(cout, local_mat.Width());

      // 4. assemble them back to prolongation matrix
      AssembleProlongationMatrix(elmt_id, local_mat);
      // cout << "assembly done " << endl;
      // local_mat.PrintMatlab();
      //}
   }
   cP->Finalize();
   cP_is_set = true;
   cout << "Check cP size: " << cP->Height() << " x " << cP->Width() << '\n';
   ofstream write("cp.txt");
   cP->PrintMatlab(write);
   write.close();
}
void CellMerging::BuildNeighbourMat(const mfem::Array<int> &elmt_id,
                                    mfem::DenseMatrix &merge_quad,
                                    mfem::DenseMatrix &mat_quad) const
{
   // resize the DenseMatrices and clean the data
   int num_el = elmt_id.Size();
   merge_quad.Clear();
   // mat_cent.Clear();
   // mat_cent.SetSize(dim, num_el);
   const FiniteElement *fe = fec->FiniteElementForGeometry(Geometry::SEGMENT);
   const int num_dofs = fe->GetDof();
   // vectors that hold coordinates of quadrature points
   // used for duplication tests
   vector<double> quad_data;
   vector<double> merge_quad_data;
   Vector quad_coord(dim); // used to store quadrature coordinate temperally
   Vector merge_quad_coord(dim);
   ElementTransformation *eltransf;
   cout << "flag 3 " << endl;
   for (int j = 0; j < num_el; j++)
   {
      // Get and store the element center
      mfem::Vector cent_coord(dim);
      GetElementCenter(elmt_id[j], cent_coord);
      // for (int i = 0; i < dim; i++)
      // {
      //    mat_cent(i, j) = cent_coord(i);
      // }
      cout << "flag 4 " << endl;
      // deal with quadrature points
      eltransf = mesh->GetElementTransformation(elmt_id[j]);
      for (int k = 0; k < num_dofs; k++)
      {
         const IntegrationPoint &eip = fe->GetNodes().IntPoint(k);
         //eip.x= (scale * eip.x) / eltransf->Weight();
         eltransf->Transform(eip, quad_coord);
         cout << "element " << elmt_id[j] << " int rule: " << endl;
         quad_coord.Print();
         for (int di = 0; di < dim; di++)
         {
            quad_data.push_back(quad_coord(di));
         }
      }
      if (eltransf->ElementNo == nEle - 2)
      {
         for (int k = 0; k < num_dofs; k++)
         {
            const IntegrationPoint &eip = fe->GetNodes().IntPoint(k);
            //eip.x= (scale * eip.x) / eltransf->Weight();
            eltransf->Transform(eip, merge_quad_coord);
            double scale = 2;
            merge_quad_coord(0) = (eip.x * eltransf->Weight() * scale) +
                                  (eltransf->ElementNo * eltransf->Weight());

            cout << "element " << elmt_id[j] << " int rule: " << endl;
            merge_quad_coord.Print();
            for (int di = 0; di < dim; di++)
            {
               merge_quad_data.push_back(merge_quad_coord(di));
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
   num_col = merge_quad_data.size() / dim;
   merge_quad.SetSize(dim, num_col);
   for (int i = 0; i < num_col; i++)
   {
      for (int j = 0; j < dim; j++)
      {
         merge_quad(j, i) = merge_quad_data[i * dim + j];
      }
   }
}

void CellMerging::GetNeighbourSet(int id, int req_n,
                                  mfem::Array<int> &nels) const
{
   // using mfem mesh object to construct the element patch
   // initialize the patch list
   nels.LoseData();
   nels.Append(id);
   int ne = mesh->GetNE();
   // if (id == ne-1)
   // {
   //    nels.Append(id-1);
   // }
   // if (id == ne-2)
   // {
   //    nels.Append(id+1);
   // }
   cout << "element is " << id << endl;
   cout << "neighbours are " << endl;
   for (int k = 0; k < nels.Size(); ++k)
   {
      cout << nels[k] << endl;
   }
}

void CellMerging::GetElementCenter(int id, mfem::Vector &cent) const
{
   cent.SetSize(mesh->Dimension());
   int geom = mesh->GetElement(id)->GetGeometryType();
   ElementTransformation *eltransf = mesh->GetElementTransformation(id);
   eltransf->Transform(Geometries.GetCenter(geom), cent);
}

void CellMerging::AssembleProlongationMatrix(const mfem::Array<int> &id,
                                             const DenseMatrix &local_mat) const
{
   // element id coresponds to the column indices
   // dofs id coresponds to the row indices
   // the local reconstruction matrix needs to be assembled `vdim` times
   const FiniteElement *fe = fec->FiniteElementForGeometry(Geometry::SEGMENT);
   const int num_dofs = fe->GetDof();

   int nel = id.Size();

   Array<int> el_dofs;
   Array<int> col_index;
   Array<int> row_index(num_dofs);
   Array<Array<int>> dofs_mat(vdim);

   // Get the id of the element want to assemble in
   int el_id = id[0];

   GetElementVDofs(el_id, el_dofs);
   col_index.SetSize(nel);

   for (int e = 0; e < nel; e++)
   {
      col_index[e] = vdim * id[e];
   }

   for (int v = 0; v < vdim; v++)
   {
      el_dofs.GetSubArray(v * num_dofs, num_dofs, row_index);
      cout << "flag 5 " << endl;
      // cout << "local mat will be assembled into: ";
      // row_index.Print(cout, num_dofs);
      // cout << endl;
      cP->SetSubMatrix(row_index, col_index, local_mat, 1);
      row_index.LoseData();
      // elements id also need to be shift accordingly
      col_index.SetSize(nel);
      for (int e = 0; e < nel; e++)
      {
         col_index[e]++;
      }
   }
}

void buildLSInterpolation(int dim, int degree, const DenseMatrix &x_merge_quad,
                          const DenseMatrix &x_quad, DenseMatrix &interp)
{
   // get the number of quadrature points and elements.
   int num_quad = x_quad.Width();
   int num_int = x_merge_quad.Width();
   int ndofs = degree + 1;
   cout << "ndofs in buildLS " << ndofs << endl;
   x_merge_quad.Print();
   // number of total polynomial basis functions
   int num_basis = -1;
   num_basis = degree + 1;
   // Construct the generalized Vandermonde matrix
   // mfem::DenseMatrix V(num_elem, num_basis);
   // for (int i = 0; i < num_elem; ++i)
   // {
   //    double dx = x_center(0, i) - x_center(0, 0);
   //    for (int p = 0; p <= degree; ++p)
   //    {
   //       V(i, p) = pow(dx, p);
   //    }
   // }
   cout << "num_quad " << num_quad << endl;
   mfem::DenseMatrix V(num_int, num_basis);
   // for (int i = 0; i < num_quad - 1; i += ndofs)
   // {
   for (int n = 0; n < ndofs; ++n)
   {
      //double dx = x_quad(0, i + n) - x_quad(0, n);
      double dx = x_quad(0, n) - x_merge_quad(0, n);
      //double dx = x_quad(0, i + n) - x_center(0, 0);
      for (int p = 0; p <= degree; ++p)
      {
         V(n, p) = pow(dx, p);
      }
   }
   //  }

   cout << "Vandermonde matrix is: " << endl;
   V.PrintMatlab();
   // Set the RHS for the LS problem (it's the identity matrix)
   // This will store the solution, that is, the basis coefficients, hence
   // the name `coeff`
   // mfem::DenseMatrix coeff(num_quad, num_quad);
   mfem::DenseMatrix coeff(num_int, num_int);
   coeff = 0.0;
   // for (int i = 0; i < num_quad; ++i)
   // {
   //    coeff(i, i) = 1.0;
   // }
   for (int i = 0; i < num_int; ++i)
   {
      coeff(i, i) = 1.0;
   }
   cout << "RHS is " << endl;
   coeff.PrintMatlab();
   // Set-up and solve the least-squares problem using LAPACK's dgels
   char TRANS = 'N';
   int info;
   // int lwork = 2 * num_elem * num_basis;
   //int lwork = 2 * num_quad * num_basis;
   // int lwork = (num_quad * num_basis) + (3 * num_basis) + 1;
   int lwork = (num_int * num_basis) + (3 * num_basis) + 1;
   double work[lwork];
   int rank;
   Array<int> jpvt;
   jpvt.SetSize(num_basis);
   jpvt = 0;
   double rcond = 1e-16;
   // dgels_(&TRANS, &num_elem, &num_basis, &num_elem, V.GetData(), &num_elem,
   //        coeff.GetData(), &num_elem, work, &lwork, &info);
   dgels_(&TRANS, &num_int, &num_basis, &num_int, V.GetData(), &num_int,
          coeff.GetData(), &num_int, work, &lwork, &info);
   // dgelsy_(&num_quad, &num_basis, &num_quad, V.GetData(), &num_quad, coeff.GetData(),
   //         &num_quad, jpvt.GetData(), &rcond, &rank, work, &lwork, &info);
   cout << "info is " << info << endl;
   MFEM_ASSERT(info == 0, "Fail to solve the underdetermined system.\n");
   cout << "coeff mat: "
        << "\n";
   coeff.PrintMatlab();
   // Perform matrix-matrix multiplication between basis functions evalauted at
   // quadrature nodes and basis function coefficients.
   // interp.SetSize(num_quad, num_elem);
   // interp = 0.0;
   // // loop over quadrature points
   // for (int j = 0; j < num_quad; ++j)
   // {
   //    double dx = x_quad(0, j) - x_center(0, 0);
   //    // loop over the element centers
   //    for (int i = 0; i < num_elem; ++i)
   //    {
   //       for (int p = 0; p <= degree; ++p)
   //       {
   //          interp(j, i) += pow(dx, p) * coeff(p, i);
   //       }
   //    }
   // }
   // interp.SetSize(num_quad, num_quad);
   interp.SetSize(num_quad, num_int);
   interp = 0.0;
   // loop over quadrature points
   for (int j = 0; j < num_quad - 1; j += ndofs)
   {
      for (int n = 0; n < ndofs; ++n)
      {
         //double dx = x_quad(0, j + n)- x_quad(0, n);
         double dx = x_quad(0, j + n) - x_merge_quad(0, n);
         // loop over the element quadrature points instead of centers
         for (int i = 0; i < num_int; ++i)
         {
            for (int p = 0; p <= degree; ++p)
            {
               interp(j + n, i) += pow(dx, p) * coeff(p, i);
            }
         }
      }
   }
   cout << "interpolation matrix: " << endl;
   interp.PrintMatlab();
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
