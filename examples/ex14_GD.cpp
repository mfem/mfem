//                                MFEM Example 14
//
// Compile with: make ex14
//
// Sample runs:  ex14 -m ../data/inline-quad.mesh -o 0
//               ex14 -m ../data/star.mesh -r 4 -o 2
//               ex14 -m ../data/star-mixed.mesh -r 4 -o 2
//               ex14 -m ../data/escher.mesh -s 1
//               ex14 -m ../data/fichera.mesh -s 1 -k 1
//               ex14 -m ../data/fichera-mixed.mesh -s 1 -k 1
//               ex14 -m ../data/square-disc-p2.vtk -r 3 -o 2
//               ex14 -m ../data/square-disc-p3.mesh -r 2 -o 3
//               ex14 -m ../data/square-disc-nurbs.mesh -o 1
//               ex14 -m ../data/disc-nurbs.mesh -r 3 -o 2 -s 1 -k 0
//               ex14 -m ../data/pipe-nurbs.mesh -o 1
//               ex14 -m ../data/inline-segment.mesh -r 5
//               ex14 -m ../data/amr-quad.mesh -r 3
//               ex14 -m ../data/amr-hex.mesh
//               ex14 -m ../data/fichera-amr.mesh
//
// Description:  This example code demonstrates the use of MFEM to define a
//               discontinuous Galerkin (DG) finite element discretization of
//               the Laplace problem -Delta u = 1 with homogeneous Dirichlet
//               boundary conditions. Finite element spaces of any order,
//               including zero on regular grids, are supported. The example
//               highlights the use of discontinuous spaces and DG-specific face
//               integrators.
//
//               We recommend viewing examples 1 and 9 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "exGD.hpp"
#include "centgridfunc.hpp"
using namespace std;
using namespace mfem;
double u_exact(const Vector &);
double f_exact(const Vector &);
void exact_function(const Vector &x, Vector &v);
int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
  // const char *mesh_file = "/home/sharan/Spring_2020/quadrature_rules_generation/mfem_install_lapack/data/periodic-square.mesh";
   const char *mesh_file = "../data/square-disc.mesh";
   // const char *mesh_file = "../data/periodic-square.mesh";
   //const char *mesh_file = "../data/inline-tri.mesh";
   int ref_levels = 1;
   int order = 1;
   int N = 5;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&N, "-n", "--#elements",
                  "number of mesh elements.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // mesh to be used
   // Mesh *mesh = new Mesh(mesh_file, 1, 1);
   // Mesh *mesh = new Mesh(N, N, Element::QUADRILATERAL, true,
   //                       1, 1, true);
   Mesh *mesh = new Mesh(N, N, Element::TRIANGLE, true,
                         1, 1, true);
   int dim = mesh->Dimension();
   //  {
   //    if (ref_levels < 0)
   //    {
   //       ref_levels = (int)floor(log(50000./mesh->GetNE())/log(2.)/dim);
   //    }
   //    for (int l = 0; l < ref_levels; l++)
   //    {
   //       mesh->UniformRefinement();
   //    }
   // }

   cout << "number of elements " << mesh->GetNE() << endl;
   // Define a finite element space on the mesh. Here we use discontinuous
   //    finite elements of the specified order >= 0.
   FiniteElementCollection *fec = new L2_FECollection(order, dim);
   FiniteElementSpace *fes = new GalerkinDifference(mesh, dim, mesh->GetNE(), fec,
                                                    1, 1, Ordering::byVDIM, order);
   FunctionCoefficient u(u_exact);
   GridFunction x_exact(fes);
   x_exact.ProjectCoefficient(u);

   GridFunction x(fes);
   CentGridFunction y(fes);
   VectorFunctionCoefficient exact(1, exact_function);
   cout << "exact sol created " << endl;
   // project exact solution at center
   y.ProjectCoefficient(exact);
   // get the solution at nodes
   fes->GetProlongationMatrix()->Mult(y, x);
   // compute the norm
   double norm =  x.ComputeL2Error(u);
   // write solution 
   ofstream adj_ofs("dgsolexact.vtk");
   adj_ofs.precision(14);
   mesh->PrintVTK(adj_ofs, 1);
   x.SaveVTK(adj_ofs, "dgSolution", 1);
   adj_ofs.close();
   cout << " solution at nodes " << endl;
   x.Print();
   x -= x_exact;
   // cout << " projection error " << endl;
   // x.Print();
   cout << "----------------------------- " << endl;
   cout << "projection error: " << x.Norml2() << '\n';
   cout << "mesh size, h = " << 1.0 / N << endl;
   cout << "solution norm: " << norm << endl;
   cout << "----------------------------- " << endl;
   delete fes;
   delete fec;
   delete mesh;
   return 0;
}

void exact_function(const Vector &x, Vector &v)
{
   // v(0) = sin(M_PI* x(0));
   v(0) = 2.0;
   //v(0) = x(0)+ x(1);
   //v(0) = sin(M_PI * x(0)) * sin(M_PI * x(1));
}

double u_exact(const Vector &x)
{
   //return x(0)+ x(1);
   return 2.0;
   // return exp(x(0)); //*x(0);
   //return (x(0) * x(0) ) + (x(1) * x(1));
   //return sin(M_PI* x(0));
   //return sin(M_PI * x(0)) * sin(M_PI * x(1));
   //return (2*x(0)) - (2*x(1));
}

/// functions for `GalerkinDifference` class

void GalerkinDifference::BuildNeighbourMat(const mfem::Array<int> &elmt_id,
                                           mfem::DenseMatrix &mat_cent,
                                           mfem::DenseMatrix &mat_quad) const
{
   // resize the DenseMatrices and clean the data
   int num_el = elmt_id.Size();
   mat_cent.Clear();
   mat_cent.SetSize(dim, num_el);

   //const FiniteElement *fe = fec->FiniteElementForGeometry(Geometry::SQUARE);
   const FiniteElement *fe = fec->FiniteElementForGeometry(Geometry::TRIANGLE);
   const int num_dofs = fe->GetDof();
   // vectors that hold coordinates of quadrature points
   // used for duplication tests
   vector<double> quad_data;
   Vector quad_coord(dim); // used to store quadrature coordinate temperally
   ElementTransformation *eltransf;
   for (int j = 0; j < num_el; j++)
   {
      // Get and store the element center
      mfem::Vector cent_coord(dim);
      GetElementCenter(elmt_id[j], cent_coord);
      for (int i = 0; i < dim; i++)
      {
         mat_cent(i, j) = cent_coord(i);
      }

      // deal with quadrature points
      eltransf = mesh->GetElementTransformation(elmt_id[j]);
      for (int k = 0; k < num_dofs; k++)
      {
         eltransf->Transform(fe->GetNodes().IntPoint(k), quad_coord);
         for (int di = 0; di < dim; di++)
         {
            quad_data.push_back(quad_coord(di));
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
}

void GalerkinDifference::GetNeighbourSet(int id, int req_n,
                                         mfem::Array<int> &nels) const
{
   // using mfem mesh object to construct the element patch
   // initialize the patch list
   nels.LoseData();
   nels.Append(id);
   // Creat the adjacent array and fill it with the first layer of adj
   // adjcant element list, candidates neighbors, candidates neighbors' adj
   Array<int> adj, cand, cand_adj, cand_next;
   mesh->ElementToElementTable().GetRow(id, adj);
   cand.Append(adj);
   //cout << "List is initialized as: ";
   //nels.Print(cout, nels.Size());
   //cout << "Initial candidates: ";
   //cand.Print(cout, cand.Size());
   while (nels.Size() < req_n)
   {
      for (int i = 0; i < adj.Size(); i++)
      {
         if (-1 == nels.Find(adj[i]))
         {
            nels.Append(adj[i]);
         }
      }
      //cout << "List now is: ";
      //nels.Print(cout, nels.Size());
      adj.LoseData();
      for (int i = 0; i < cand.Size(); i++)
      {
         //cout << "deal with cand " << cand[i];
         mesh->ElementToElementTable().GetRow(cand[i], cand_adj);
         //cout << "'s adj are ";
         //cand_adj.Print(cout, cand_adj.Size());
         for (int j = 0; j < cand_adj.Size(); j++)
         {
            if (-1 == nels.Find(cand_adj[j]))
            {
               //cout << cand_adj[j] << " is not found in nels. add to adj and cand_next.\n";
               adj.Append(cand_adj[j]);
               cand_next.Append(cand_adj[j]);
            }
         }
         cand_adj.LoseData();
      }
      cand.LoseData();
      cand = cand_next;
      //cout << "cand copy from next: ";
      //cand.Print(cout, cand.Size());
      cand_next.LoseData();
   }
}

void GalerkinDifference::GetElementCenter(int id, mfem::Vector &cent) const
{
   cent.SetSize(mesh->Dimension());
   int geom = mesh->GetElement(id)->GetGeometryType();
   ElementTransformation *eltransf = mesh->GetElementTransformation(id);
   eltransf->Transform(Geometries.GetCenter(geom), cent);
}

void GalerkinDifference::BuildGDProlongation() const
{
    //const FiniteElement *fe = fec->FiniteElementForGeometry(Geometry::SQUARE);
   const FiniteElement *fe = fec->FiniteElementForGeometry(Geometry::TRIANGLE);
   const int num_dofs = fe->GetDof();
   // allocate the space for the prolongation matrix
   // this step should be done in the constructor (probably)
   // should it be GetTrueVSize() ? or GetVSize()?
   // need a new method that directly construct a CSR format sparsematrix ？
   cP = new mfem::SparseMatrix(GetVSize(), vdim * nEle);
   // determine the minimum # of element in each patch
   int nelmt;
   switch (dim)
   {
   case 1:
      nelmt = degree + 1;
      break;
   case 2:
      nelmt = (degree + 1) * (degree + 2) / 2;
      break;
   default:;
   }
   cout << "Number of required element: " << nelmt << '\n';
   // loop over all the element:
   // 1. build the patch for each element,
   // 2. construct the local reconstruction operator
   // 3. assemble local reconstruction operator

   // vector that contains element id (resize to zero )
   mfem::Array<int> elmt_id;
   mfem::DenseMatrix cent_mat, quad_mat, local_mat;
   cout << "The size of the prolongation matrix is " << cP->Height() << " x " << cP->Width() << '\n';
   //int degree_actual;
   for (int i = 0; i < nEle; i++)
   {

      GetNeighbourSet(i, nelmt, elmt_id);
      cout << "element " << "( " << i  << ") " << " #neighbours = " << elmt_id.Size() << endl;
      // cout << "Elements id(s) in patch: ";
      elmt_id.Print(cout, elmt_id.Size());
      cout << " ----------------------- "  << endl;
      // 2. build the quadrature and barycenter coordinate matrices
      BuildNeighbourMat(elmt_id, cent_mat, quad_mat);
      // cout << "The element center matrix:\n";
      // cent_mat.Print(cout, cent_mat.Width());
      // cout << endl;
      // cout << "Quadrature points id matrix:\n";
      // quad_mat.Print(cout, quad_mat.Width());
      // cout << endl;

      // 3. buil the loacl reconstruction matrix
      buildLSInterpolation(dim, degree, cent_mat, quad_mat, local_mat);
      // cout << "Local reconstruction matrix R:\n";
      // local_mat.Print(cout, local_mat.Width());

      // 4. assemble them back to prolongation matrix
      AssembleProlongationMatrix(elmt_id, local_mat);
   }
   cP->Finalize();
   cP_is_set = true;
   cout << "Check cP size: " << cP->Height() << " x " << cP->Width() << '\n';
   // ofstream cp_save("cP.txt");
   // cP->PrintMatlab(cp_save);
   // cp_save.close();
}
void GalerkinDifference::AssembleProlongationMatrix(const mfem::Array<int> &id,
                                                    const DenseMatrix &local_mat) const
{
   // element id coresponds to the column indices
   // dofs id coresponds to the row indices
   // the local reconstruction matrix needs to be assembled `vdim` times
    //const FiniteElement *fe = fec->FiniteElementForGeometry(Geometry::SQUARE);
   const FiniteElement *fe = fec->FiniteElementForGeometry(Geometry::TRIANGLE);
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

void buildLSInterpolation(int dim, int degree, const DenseMatrix &x_center,
                          const DenseMatrix &x_quad, DenseMatrix &interp)
{
   // get the number of quadrature points and elements.
   int num_quad = x_quad.Width();
   int num_elem = x_center.Width();
   // number of total polynomial basis functions
   int num_basis = -1;
   if (1 == dim)
   {
      num_basis = degree + 1;
   }
   else if (2 == dim)
   {
      num_basis = (degree + 1) * (degree + 2) / 2;
   }
   else if (3 == dim)
   {
      num_basis = (degree + 1) * (degree + 2) * (degree + 3) / 6;
   }
   else
   {
      cout << "not implemented " << endl;
   }

   // Construct the generalized Vandermonde matrix
   mfem::DenseMatrix V(num_elem, num_basis);
   if (1 == dim)
   {
      for (int i = 0; i < num_elem; ++i)
      {
         double dx = x_center(0, i) - x_center(0, 0);
         for (int p = 0; p <= degree; ++p)
         {
            V(i, p) = pow(dx, p);
         }
      }
   }
   else if (2 == dim)
   {
      for (int i = 0; i < num_elem; ++i)
      {
         double dx = x_center(0, i) - x_center(0, 0);
         double dy = x_center(1, i) - x_center(1, 0);
         int col = 0;
         for (int p = 0; p <= degree; ++p)
         {
            for (int q = 0; q <= p; ++q)
            {
               V(i, col) = pow(dx, p - q) * pow(dy, q);
               ++col;
            }
         }
      }
   }
   else if (3 == dim)
   {
      for (int i = 0; i < num_elem; ++i)
      {
         double dx = x_center(0, i) - x_center(0, 0);
         double dy = x_center(1, i) - x_center(1, 0);
         double dz = x_center(2, i) - x_center(2, 0);
         int col = 0;
         for (int p = 0; p <= degree; ++p)
         {
            for (int q = 0; q <= p; ++q)
            {
               for (int r = 0; r <= p - q; ++r)
               {
                  V(i, col) = pow(dx, p - q - r) * pow(dy, r) * pow(dz, q);
                  ++col;
               }
            }
         }
      }
   }

   // Set the RHS for the LS problem (it's the identity matrix)
   // This will store the solution, that is, the basis coefficients, hence
   // the name `coeff`
   mfem::DenseMatrix coeff(num_elem, num_elem);
   coeff = 0.0;
   for (int i = 0; i < num_elem; ++i)
   {
      coeff(i, i) = 1.0;
   }

   // Set-up and solve the least-squares problem using LAPACK's dgels
   char TRANS = 'N';
   int info;
   int lwork = 2 * num_elem * num_basis;
   double work[lwork];
   dgels_(&TRANS, &num_elem, &num_basis, &num_elem, V.GetData(), &num_elem,
          coeff.GetData(), &num_elem, work, &lwork, &info);
   MFEM_ASSERT(info == 0, "Fail to solve the underdetermined system.\n");

   // Perform matrix-matrix multiplication between basis functions evalauted at
   // quadrature nodes and basis function coefficients.
   interp.SetSize(num_quad, num_elem);
   interp = 0.0;
   if (1 == dim)
   {
      // loop over quadrature points
      for (int j = 0; j < num_quad; ++j)
      {
         double dx = x_quad(0, j) - x_center(0, 0);
         // loop over the element centers
         for (int i = 0; i < num_elem; ++i)
         {
            for (int p = 0; p <= degree; ++p)
            {
               interp(j, i) += pow(dx, p) * coeff(p, i);
            }
         }
      }
   }
   else if (2 == dim)
   {
      // loop over quadrature points
      for (int j = 0; j < num_quad; ++j)
      {
         double dx = x_quad(0, j) - x_center(0, 0);
         double dy = x_quad(1, j) - x_center(1, 0);
         // loop over the element centers
         for (int i = 0; i < num_elem; ++i)
         {
            int col = 0;
            for (int p = 0; p <= degree; ++p)
            {
               for (int q = 0; q <= p; ++q)
               {
                  interp(j, i) += pow(dx, p - q) * pow(dy, q) * coeff(col, i);
                  ++col;
               }
            }
         }
      }
   }
   else if (dim == 3)
   {
      // loop over quadrature points
      for (int j = 0; j < num_quad; ++j)
      {
         double dx = x_quad(0, j) - x_center(0, 0);
         double dy = x_quad(1, j) - x_center(1, 0);
         double dz = x_quad(2, j) - x_center(2, 0);
         // loop over the element centers
         for (int i = 0; i < num_elem; ++i)
         {
            int col = 0;
            for (int p = 0; p <= degree; ++p)
            {
               for (int q = 0; q <= p; ++q)
               {
                  for (int r = 0; r <= p - q; ++r)
                  {
                     interp(j, i) += pow(dx, p - q - r) * pow(dy, r) * pow(dz, q) * coeff(col, i);
                     ++col;
                  }
               }
            }
         }
      }
   }
}

///functions related to CentGridFunction class
CentGridFunction::CentGridFunction(FiniteElementSpace *f)
{
   SetSize(f->GetVDim() * f->GetNE());
   fes = f;
   fec = NULL;
   sequence = f->GetSequence();
   UseDevice(true);
}

void CentGridFunction::ProjectCoefficient(VectorCoefficient &coeff)
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

CentGridFunction &CentGridFunction::operator=(const Vector &v)
{
   std::cout << "cent = is called.\n";
   MFEM_ASSERT(fes && v.Size() == fes->GetTrueVSize(), " not true ");
   Vector::operator=(v);
   return *this;
}

CentGridFunction &CentGridFunction::operator=(double value)
{
   Vector::operator=(value);
   return *this;
}
