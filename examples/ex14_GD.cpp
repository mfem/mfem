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
   const char *mesh_file = "../data/square-disc.mesh";
   //const char *mesh_file = "../data/periodic-square.mesh";
   int ref_levels = -1;
   int order = 1;
   int N = 5;
   double sigma = -1.0;
   double kappa = 100.0;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&N, "-n", "--#elements",
                  "number of mesh elements.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&sigma, "-s", "--sigma",
                  "One of the two DG penalty parameters, typically +1/-1."
                  " See the documentation of class DGDiffusionIntegrator.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "One of the two DG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (kappa < 0)
   {
      kappa = (order+1)*(order+1);
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   //    NURBS meshes are projected to second order meshes.
  // Mesh *mesh = new Mesh(mesh_file, 1, 1);
   // Mesh *mesh = new Mesh(N, N, Element::TRIANGLE, true,
   //                       1, 1, true);
   Mesh *mesh = new Mesh(N, N, Element::QUADRILATERAL, true,
                         1, 1, true);
   int dim = mesh->Dimension();
   cout << "number of elements " << mesh->GetNE() << endl;
   ofstream sol_ofv("square_disc_mesh.vtk");
   sol_ofv.precision(14);
   mesh->PrintVTK(sol_ofv, 1);
   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. By default, or if ref_levels < 0,
   //    we choose it to be the largest number that gives a final mesh with no
   //    more than 50,000 elements.
   // {
   //    if (ref_levels < 0)
   //    {
   //       ref_levels = (int)floor(log(50000./mesh->GetNE())/log(2.)/dim);
   //    }
   //    for (int l = 0; l < ref_levels; l++)
   //    {
   //       mesh->UniformRefinement();
   //    }
   // }
   // if (mesh->NURBSext)
   // {
   //    mesh->SetCurvature(max(order, 1));
   // }

   // 4. Define a finite element space on the mesh. Here we use discontinuous
   //    finite elements of the specified order >= 0.
   FiniteElementCollection *fec = new DG_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "Number of unknowns: " << fespace->GetVSize() << endl;
   /// GD finite element space
   FiniteElementSpace *fes = new GalerkinDifference(mesh, dim, mesh->GetNE(), fec, 1, Ordering::byVDIM, order);
   cout << "Number of GD unknowns: " << fes->GetTrueVSize() << endl;
   cout << "#dofs " << fes->GetNDofs() << endl;
//    // 5. Set up the linear form b(.) which corresponds to the right-hand side of
//    //    the FEM linear system.
//    LinearForm *b = new LinearForm(fespace);
//    ConstantCoefficient one(1.0);
//    ConstantCoefficient zero(0.0);
//    FunctionCoefficient f(f_exact);
//    FunctionCoefficient u(u_exact);
//    VectorFunctionCoefficient exact(1, exact_function);
//    b->AddDomainIntegrator(new DomainLFIntegrator(f));
//    b->AddBdrFaceIntegrator(
//       new DGDirichletLFIntegrator(u, one, sigma, kappa));
//    b->Assemble();
//    // GD grid function
//    CentGridFunction y(fes);
//    y.ProjectCoefficient(exact);
//    // cout << "exact solution " << endl;
//    // y.Print();
//    // cout << "center grid function created " << endl;
//    // VectorFunctionCoefficient exact(dim, exact_function);
//    // y.ProjectCoefficient(exact);
//    // cout << "solution at center is " << endl;
//    // y.Print();
//    // cout << "check if the prolongation matrix is correct " << endl;
//    // GridFunction x(fespace);
//    // fes->GetProlongationMatrix()->Mult(y, x);
//    // x.Print();
// //    cout << "rhs is " << endl;
// //   b->Print();
//    // 6. Define the solution vector x as a finite element grid function
//    //    corresponding to fespace. Initialize x with initial guess of zero.
//    GridFunction x(fespace);
//    // 7. Set up the bilinear form a(.,.) on the finite element space
//    //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
//    //    domain integrator and the interior and boundary DG face integrators.
//    //    Note that boundary conditions are imposed weakly in the form, so there
//    //    is no need for dof elimination. After assembly and finalizing we
//    //    extract the corresponding sparse matrix A.
//    BilinearForm *a = new BilinearForm(fespace);
//    a->AddDomainIntegrator(new DiffusionIntegrator(one));
//    a->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
//    a->AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
//    a->Assemble();
//    a->Finalize();
//     // stiffness matrix
//    SparseMatrix &Aold = a->SpMat();
//    SparseMatrix *cp = dynamic_cast<GalerkinDifference *>(fes)->GetCP();
//    SparseMatrix *p = RAP(*cp, Aold, *cp);
//    SparseMatrix &A = *p;
//    ofstream write("stiffmat_ex14GD.txt");
//    A.PrintMatlab(write);
//    write.close();
//    // get P^T b
//    Vector bnew(A.Width());
//    fes->GetProlongationMatrix()->MultTranspose(*b, bnew);
//    // write stiffness matrix to file
//    //cout << "bilinear form size " << a->Size() << endl;
//    //A.Print();
//    //cout << x.Size() << endl;
// //#ifndef MFEM_USE_SUITESPARSE
//    // 8. Define a simple symmetric Gauss-Seidel preconditioner and use it to
//    //    solve the system Ax=b with PCG in the symmetric case, and GMRES in the
//    //    non-symmetric one.
//    GSSmoother M(A);
//    if (sigma == -1.0)
//    {
//       PCG(A, M, bnew, y, 1, 2000, 1e-40, 0.0);
//    }
//    else
//    {
//       GMRES(A, M, bnew, y, 1, 500, 10, 1e-16, 0.0);
//    }
// // #else
// //    // 8. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
// //    UMFPackSolver umf_solver;
// //    umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
// //    umf_solver.SetOperator(A);
// //    umf_solver.Mult(*b, x);
// // #endif

//    // 9. Save the refined mesh and the solution. This output can be viewed later
//    //    using GLVis: "glvis -m refined.mesh -g sol.gf".
//    // cout << "----------------------------- "<< endl;
//    // cout << "solution at center obtained: "<< endl;
//    // y.Print();
//    // cout << "----------------------------- "<< endl;
//    // get x = P y
//    fes->GetProlongationMatrix()->Mult(y, x);
//    // cout << "solution at nodes " << endl;
//    // x.Print();
//    // cout << "************************" << endl;
//    ofstream mesh_ofs("refined.mesh");
//    mesh_ofs.precision(8);
//    mesh->Print(mesh_ofs);
//    ofstream sol_ofs("sol.gf");
//    sol_ofs.precision(8);
//    x.Save(sol_ofs);

//    // ofstream adj_ofs("dgsoldisc.vtk");
//    // adj_ofs.precision(14);
//    // mesh->PrintVTK(adj_ofs, 1);
//    // x.SaveVTK(adj_ofs, "dgSolution", 1);
//    // adj_ofs.close();
//    // 10. Send the solution by socket to a GLVis server.
//    if (visualization)
//    {
//       char vishost[] = "localhost";
//       int  visport   = 19916;
//       socketstream sol_sock(vishost, visport);
//       sol_sock.precision(8);
//       sol_sock << "solution\n" << *mesh << x << flush;
//    }
//   double norm = x.ComputeL2Error(u);
//   cout << "----------------------------- " << endl;
//   cout << "mesh size, h = " << 1.0 / N << endl;
//   cout << "solution norm: " << norm << endl;
//   // 11. Free the used memory.
//   delete a;
//   delete b;
  delete fespace;
  delete fec;
  delete mesh;
  return 0;
}
double u_exact(const Vector &x)
{
   //return 2.0;
   //return exp(x(0));
   //return exp(x(0)+x(1)); 
   //return (x(0) * x(0) ) + (x(1) * x(1));
   //return x(0)*x(0);
   return sin(M_PI* x(0))*sin(M_PI*x(1));
   //return (2*x(0)) - (2*x(1));
}
double f_exact(const Vector &x)
{
   //return 0.0;
   //return -2*exp(x(0)+x(1)); 
   //return -exp(x(0));
   //return -4.0;
   return 2*M_PI * M_PI* sin(M_PI*x(0)) * sin(M_PI* x(1));
}


void exact_function(const Vector &x, Vector &v)
{
   //v(0) = 2.0;
   // v(0) = sin(M_PI* x(0));
   //v(0) = exp(x(0)+x(1)); 
   //v(0) = x(0)+ x(1);
   v(0) = sin(M_PI * x(0)) * sin(M_PI * x(1));
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

   const FiniteElement *fe = fec->FiniteElementForGeometry(Geometry::SQUARE);
   //const FiniteElement *fe = fec->FiniteElementForGeometry(Geometry::TRIANGLE);
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
   const FiniteElement *fe = fec->FiniteElementForGeometry(Geometry::SQUARE);
   //const FiniteElement *fe = fec->FiniteElementForGeometry(Geometry::TRIANGLE);
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
      //nelmt = nelmt + 1;
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
      //cout << "element " << "( " << i  << ") " << " #neighbours = " << elmt_id.Size() << endl;
      // cout << "Elements id(s) in patch: ";
      //elmt_id.Print(cout, elmt_id.Size());
      //cout << " ----------------------- "  << endl;
      // 2. build the quadrature and barycenter coordinate matrices
      BuildNeighbourMat(elmt_id, cent_mat, quad_mat);
      // cout << "The element center matrix:\n";
      // cent_mat.Print(cout, cent_mat.Width());
      // cout << endl;
      // cout << "Quadrature points id matrix:\n";
      // quad_mat.Print(cout, quad_mat.Width());
      // cout << endl;

      // 3. buil the loacl reconstruction matrix
     // cout << "element is " << i << endl;
      buildLSInterpolation(dim, degree, cent_mat, quad_mat, local_mat);
      //cout << " ######################### " << endl;
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
   const FiniteElement *fe = fec->FiniteElementForGeometry(Geometry::SQUARE);
   //const FiniteElement *fe = fec->FiniteElementForGeometry(Geometry::TRIANGLE);
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
   //cout << num_elem << " x " << num_basis << endl;
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
   mfem::DenseMatrix rhs(num_elem, num_elem);
   rhs = coeff;
   // Set-up and solve the least-squares problem using LAPACK's dgels
   char TRANS = 'N';
   int info;
   //int lwork = 2 * num_elem * num_basis;
   int lwork = (num_elem * num_basis) + (3* num_basis) + 1; 
   double work[lwork];
   int rank;
   Array<int> jpvt;
   jpvt.SetSize(num_basis);
   jpvt = 0;
   double rcond= 1e-16;
   // cout << "right hand side " << endl;
   // coeff.PrintMatlab();
  // cout << "A is  " << endl;
   ofstream write("V_mat.txt");
   write.precision(16);
   V.PrintMatlab(write);
   write.close();
   //V.PrintMatlab();
  // cout << "rank is " << V.Rank(1e-12) << endl;
   dgels_(&TRANS, &num_elem, &num_basis, &num_elem, V.GetData(), &num_elem,
          coeff.GetData(), &num_elem, work, &lwork, &info);
   // dgelsy_(&num_elem, &num_basis, &num_elem,  V.GetData(), &num_elem, coeff.GetData(),
   //       &num_elem, jpvt.GetData(),  &rcond , &rank,  work, &lwork, &info);
   //cout<< "info is " << info << endl;

   MFEM_ASSERT(info == 0, "Fail to solve the underdetermined system.\n");

   // mfem::DenseMatrix res(num_elem, num_elem);
   // for (int i = 0; i < num_elem; ++i)
   // {
   //    int coln = 0;
   //    for (int k = 0; k < num_elem; ++k)
   //    {
   //       for (int p = 0; p <= num_basis; ++p)
   //       {
   //          res(i, k) += V(i, p ) * coeff(coln, i);
   //          ++ coln;
   //       }
   //    }
   // }
   // res -=rhs;
   // res.Print();
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
      // loop over quadrature points
      for (int j = 0; j < num_quad; ++j)
      {
         for (int p = 0; p <= degree; ++p)
         {
            for (int q = 0; q <= p; ++q)
            {
         // int p = 0;
         // int q = 0;
         // loop over the element centers
         double poly_at_quad = 0.0;
         for (int i = 0; i < num_elem; ++i)
         {
            double dx = x_quad(0, j) - x_center(0, i);
            double dy = x_quad(1, j) - x_center(1, i);
            poly_at_quad += interp(j, i) * pow(dx, p - q) * pow(dy, q);
         }
         double exact = ((p == 0) && (q == 0)) ? 1.0 : 0.0;
         // mfem::out << "polynomial interpolation error (" << p - q << ","
         //           << q << ") = " << fabs(exact - poly_at_quad) << endl;
         // if ((p == 0) && (q == 0))
         // {
         MFEM_ASSERT(fabs(exact - poly_at_quad) <= 1e-12,
                     "Interpolation operator does not interpolate exactly!\n");
         // }
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
