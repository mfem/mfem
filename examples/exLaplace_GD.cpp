#include "exLaplace.hpp"
#include "exGD_cut.hpp"
#include "centgridfunc.hpp"
#include <chrono>
using namespace std::chrono;
using namespace std;
using namespace mfem;
double u_exact(const Vector &);
double f_exact(const Vector &);
void exact_function(const Vector &x, Vector &v);
void u_neumann(const Vector &, Vector &);
double CutComputeL2Error(GridFunction &x, FiniteElementSpace *fes,
                         Coefficient &exsol, const std::vector<bool> &EmbeddedElems,
                         std::map<int, IntegrationRule *> &CutSquareIntRules);
template <int N>
struct circle
{
   double xscale;
   double yscale;
   double xmin;
   double ymin;
   double radius;
   template <typename T>
   T operator()(const blitz::TinyVector<T, N> &x) const
   {
      // level-set function to work in physical space
      // return -1 * (((x[0] - 5) * (x[0] - 5)) +
      //               ((x[1]- 5) * (x[1] - 5)) - (0.5 * 0.5));
      // level-set function for reference elements
      return -1 * ((((x[0] * xscale) + xmin - 0.5) * ((x[0] * xscale) + xmin - 0.5)) +
                   (((x[1] * yscale) + ymin - 0.5) * ((x[1] * yscale) + ymin - 0.5)) - (radius * radius));
   }
   template <typename T>
   blitz::TinyVector<T, N> grad(const blitz::TinyVector<T, N> &x) const
   {
      // return blitz::TinyVector<T, N>(-1 * (2.0 * (x(0) - 5)), -1 * (2.0 * (x(1) - 5)));
      return blitz::TinyVector<T, N>(-1 * (2.0 * xscale * ((x(0) * xscale) + xmin - 0.5)),
                                     -1 * (2.0 * yscale * ((x(1) * yscale) + ymin - 0.5)));
   }
};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int order = 1;
   int N = 5;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   double sigma = -1.0;
   double kappa = 100.0;
   double cutsize;
   double radius = 0.2;
   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&N, "-n", "--#elements",
                  "number of mesh elements.");
   args.AddOption(&radius, "-r", "--radius",
                  "radius of circle.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (kappa < 0)
   {
      kappa = (order + 1) * (order + 1);
   }
   args.PrintOptions(cout);

   Mesh *mesh = new Mesh(N, N, Element::QUADRILATERAL, true,
                         1, 1, true);

   // Mesh *mesh = new Mesh(N, N, Element::TRIANGLE, true,
   //                       1, 1, true);
   ofstream sol_ofv("square_mesh.vtk");
   sol_ofv.precision(14);
   mesh->PrintVTK(sol_ofv, 0);
   std::map<int, IntegrationRule *> CutSquareIntRules;
   std::map<int, IntegrationRule *> cutSegmentIntRules;
   std::map<int, IntegrationRule *> cutInteriorFaceIntRules;
   //find the elements cut by boundary
   vector<int> cutelems;
   vector<int> innerelems;
   vector<int> cutinteriorFaces;
   for (int i = 0; i < mesh->GetNE(); ++i)
   {
      if (cutByCircle(mesh, radius, i) == true)
      {
         cutelems.push_back(i);
      }
      if (insideBoundary(mesh, radius, i) == true)
      {
         innerelems.push_back(i);
      }
   }
   cout << "elements cut by circle:  " << endl;
   for (int i = 0; i < cutelems.size(); ++i)
   {
      cout << cutelems.at(i) << endl;
   }
   cout << "elements completely inside circle:  " << endl;
   for (int i = 0; i < innerelems.size(); ++i)
   {
      cout << innerelems.at(i) << endl;
   }
   cout << "#elements completely inside circle:  " << innerelems.size() << endl;
   int dim = mesh->Dimension();
   for (int i = 0; i < mesh->GetNumFaces(); ++i)
   {
      FaceElementTransformations *tr;
      tr = mesh->GetInteriorFaceTransformations(i);
      if (tr != NULL)
      {
         if ((find(cutelems.begin(), cutelems.end(), tr->Elem1No) != cutelems.end()) &&
             (find(cutelems.begin(), cutelems.end(), tr->Elem2No) != cutelems.end()))
         {
            cutinteriorFaces.push_back(tr->Face->ElementNo);
         }
      }
   }
   cout << "faces cut by circle:  " << endl;
   for (int i = 0; i < cutinteriorFaces.size(); ++i)
   {
      cout << cutinteriorFaces.at(i) << endl;
   }
   cout << "dimension is " << dim << endl;
   std::cout << "Number of elements: " << mesh->GetNE() << '\n';
   int deg = order + 1;
   // define map for integration rule for cut elements
   GetCutElementIntRule<2>(mesh, cutelems, deg, radius, CutSquareIntRules);
   GetCutSegmentIntRule<2>(mesh, cutelems, cutinteriorFaces, deg, radius, cutSegmentIntRules,
                           cutInteriorFaceIntRules);
   GetCutsize(mesh, cutelems, CutSquareIntRules, cutsize);
   std::vector<bool> EmbeddedElems;
   for (int i = 0; i < mesh->GetNE(); ++i)
   {
      if (insideBoundary(mesh, radius, i) == true)
      {
         EmbeddedElems.push_back(true);
      }
      else
      {
         EmbeddedElems.push_back(false);
      }
   }
   std::map<int, bool> immersedFaces;
   for (int i = 0; i < mesh->GetNumFaces(); ++i)
   {
      FaceElementTransformations *tr;
      tr = mesh->GetInteriorFaceTransformations(i);
      if (tr != NULL)
      {
         if ((EmbeddedElems.at(tr->Elem1No) == true) && (EmbeddedElems.at(tr->Elem2No)) == false)
         {
            immersedFaces[tr->Face->ElementNo] = true;
         }
         if ((EmbeddedElems.at(tr->Elem2No) == true) && (EmbeddedElems.at(tr->Elem1No)) == false)
         {
            immersedFaces[tr->Face->ElementNo] = true;
         }
      }
   }
   // 4. Define a finite element space on the mesh. Here we use discontinuous
   //    finite elements of the specified order >= 0.
   FiniteElementCollection *fec = new DG_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec, 1);
   /// GD finite element space
   // auto start = high_resolution_clock::now();
   FiniteElementSpace *fes = new GalerkinDifference(mesh, dim, mesh->GetNE(), fec, EmbeddedElems, 1, Ordering::byVDIM, order);
   // auto stop = high_resolution_clock::now();
   // auto duration = duration_cast<microseconds>(stop - start);
   // cout << "time taken for prolongation: " << duration.count()*1e-06 << endl;
   cout << "fes created " << endl;
   cout << "Number of unknowns in GD: " << fes->GetTrueVSize() << endl;
   cout << "Number of unknowns: " << fespace->GetVSize() << endl;
   cout << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;
   LinearForm *b = new LinearForm(fespace);
   FunctionCoefficient f(f_exact);
   FunctionCoefficient u(u_exact);
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   ConstantCoefficient two(2.0);
   ConstantCoefficient zerop(0.01);
   VectorFunctionCoefficient uN(dim, u_neumann);
   //linear form
   b->AddDomainIntegrator(new CutDomainLFIntegrator(f, CutSquareIntRules, EmbeddedElems));
   // b->AddDomainIntegrator(new CutDGDirichletLFIntegrator(u, one, sigma, kappa,
   //                                                       cutSegmentIntRules));
   b->AddDomainIntegrator(new CutDGNeumannLFIntegrator(uN,
                                                       cutSegmentIntRules));
   b->AddBdrFaceIntegrator(
       new DGDirichletLFIntegrator(u, one, sigma, kappa));
   b->Assemble();
   // cout << "RHS: " << endl;
   // b->Print();
   GridFunction x(fespace);
   CentGridFunction y(fes);
   VectorFunctionCoefficient exact(1, exact_function);
   GridFunction xexact(fespace);
   xexact.ProjectCoefficient(exact);
   // cout << "exact sol created " << endl;
   // xexact.Print();
   // cout << "prolongated sol " << endl;
   y.ProjectCoefficient(exact);
   fes->GetProlongationMatrix()->Mult(y, x);
   // x.Print();
   // cout << "check prolongation operator " << endl;
   // cout << x.ComputeL2Error(u) << endl;
   // bilinear form
   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new CutDiffusionIntegrator(one, CutSquareIntRules, EmbeddedElems));
   //a->AddDomainIntegrator(new CutBoundaryFaceIntegrator(one, sigma, kappa, cutSegmentIntRules));
   a->AddInteriorFaceIntegrator(new CutDGDiffusionIntegrator(one, sigma, kappa,
                                                             immersedFaces, cutinteriorFaces,
                                                             cutInteriorFaceIntRules));
   a->AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
   a->Assemble();
   a->Finalize();
   //const SparseMatrix &A = a->SpMat();
   SparseMatrix &Aold = a->SpMat();
   SparseMatrix *cp = dynamic_cast<GalerkinDifference *>(fes)->GetCP();
   SparseMatrix *p = RAP(*cp, Aold, *cp);
   SparseMatrix &A = *p;
   ofstream write("stiffmat_lap_cut_gd.txt");
   A.PrintMatlab(write);
   write.close();
   // calculate condition number
   // DenseMatrix Ad;
   // A.ToDenseMatrix(Ad);
   // Vector si;
   // Ad.SingularValues(si);
   // // cout << "singular values " << endl;
   // // si.Print();
   // double cond;
   // cond = si(0) / si(si.Size() - 1);
   // cout << "cond# " << endl;
   // cout << cond << endl;
   Vector bnew(A.Width());
   fes->GetProlongationMatrix()->MultTranspose(*b, bnew);
   // Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //    solve the system Ax=b with PCG in the symmetric case, and GMRES in the
   //    non-symmetric one.
   GSSmoother M(A);
   if (sigma == -1.0)
   {
      PCG(A, M, bnew, y, 1, 10000, 1e-40, 0.0);
   }
   else
   {
      GMRES(A, M, bnew, y, 1, 1000, 10, 1e-12, 0.0);
   }
   //x.Print();
   // #else
   //    // 8. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
   //    UMFPackSolver umf_solver;
   //    umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   //    umf_solver.SetOperator(A);
   //    umf_solver.Mult(bnew, y);
   // #endif

   fes->GetProlongationMatrix()->Mult(y, x);

   ofstream adj_ofs("dgSolcirclelap_gd.vtk");
   adj_ofs.precision(14);
   mesh->PrintVTK(adj_ofs, 1);
   x.SaveVTK(adj_ofs, "Solution", 1);
   adj_ofs.close();
   //double norm = x.ComputeL2Error(u);
   double norm = CutComputeL2Error(x, fespace, u, EmbeddedElems, CutSquareIntRules);
   cout << "----------------------------- " << endl;
   cout << "mesh size, h = " << 1.0 / N << endl;
   cout << "solution norm: " << norm << endl;
   // x.Print();
   // 11. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   delete fec;
   delete mesh;
   return 0;
}

void exact_function(const Vector &x, Vector &v)
{
   //int dim = x.Size();
   // v(0) = x(0)*x(0);
   //v(0) = exp(x(0));
   // v(0) = sin(M_PI*x(0));
   //v(0) = 2.0;
   v(0) = sin(M_PI * x(0)) * sin(M_PI * x(1));
   // v(0) = x(0);
}

double u_exact(const Vector &x)
{
   return sin(M_PI * x(0)) * sin(M_PI * x(1));
   //return 2.0;
   // return x(0);
   //return (2*x(0)) - (2*x(1));
}
double f_exact(const Vector &x)
{
   return 2 * M_PI * M_PI * sin(M_PI * x(0)) * sin(M_PI * x(1));
   //return 0.0;
}

void u_neumann(const Vector &x, Vector &u)
{
   u(0) = M_PI * cos(M_PI * x(0)) * sin(M_PI * x(1));
   u(1) = M_PI * sin(M_PI * x(0)) * cos(M_PI * x(1));
   // u(0) = 0.0;
   // u(1) = 0.0;
}

double CutComputeL2Error(GridFunction &x, FiniteElementSpace *fes,
                         Coefficient &exsol, const std::vector<bool> &EmbeddedElems,
                         std::map<int, IntegrationRule *> &CutSquareIntRules)
{
   double error = 0.0;
   const FiniteElement *fe;
   ElementTransformation *T;
   Vector vals;
   int p = 2;
   for (int i = 0; i < fes->GetNE(); i++)
   {
      if (EmbeddedElems.at(i) == true)
      {
         error += 0.0;
      }
      else
      {
         fe = fes->GetFE(i);
         const IntegrationRule *ir;
         ir = CutSquareIntRules[i];
         if (ir == NULL)
         {
            int intorder = 2 * fe->GetOrder() + 1; // <----------
            ir = &(IntRules.Get(fe->GetGeomType(), intorder));
         }
         x.GetValues(i, *ir, vals);
         T = fes->GetElementTransformation(i);
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

void GetCutsize(Mesh *mesh, vector<int> cutelems, std::map<int, IntegrationRule *> &CutSquareIntRules,
                double &cutsize)
{
   cutsize = 1.0;
   for (int k = 0; k < cutelems.size(); ++k)
   {
      int id = cutelems.at(k);
      ElementTransformation *Trans = mesh->GetElementTransformation(id);
      const IntegrationRule *ir;
      ir = CutSquareIntRules[Trans->ElementNo];
      double area = 0.0;
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         Trans->SetIntPoint(&ip);
         area += ip.weight * Trans->Weight();
      }
      // cout << "normal element area is " << Trans->Weight() << endl;
      // cout << "area of cut element " << id << " : " << area << endl;
      double cs = area / Trans->Weight();
      //cout << "ratio: " << cs << endl;
      if (cs < cutsize)
      {
         cutsize = cs;
      }
   }
   // cout << "cutsize is " << cutsize << endl;
}

template <int N>
void GetCutElementIntRule(Mesh *mesh, vector<int> cutelems, int order, double r,
                          std::map<int, IntegrationRule *> &CutSquareIntRules)
{
   double tol = 1e-16;
   for (int k = 0; k < cutelems.size(); ++k)
   {
      IntegrationRule *ir;
      blitz::TinyVector<double, N> xmin;
      blitz::TinyVector<double, N> xmax;
      blitz::TinyVector<double, N> xupper;
      blitz::TinyVector<double, N> xlower;
      // standard reference element
      xlower = {0, 0};
      xupper = {1, 1};
      int dir = -1;
      int side = -1;
      int elemid = cutelems.at(k);
      findBoundingBox<N>(mesh, elemid, xmin, xmax);
      circle<N> phi;
      phi.xscale = xmax[0] - xmin[0];
      phi.yscale = xmax[1] - xmin[1];
      phi.xmin = xmin[0];
      phi.ymin = xmin[1];
      phi.radius = r;
      auto q = Algoim::quadGen<N>(phi, Algoim::BoundingBox<double, N>(xlower, xupper), dir, side, order);
      int i = 0;
      ir = new IntegrationRule(q.nodes.size());
      for (const auto &pt : q.nodes)
      {
         IntegrationPoint &ip = ir->IntPoint(i);
         ip.x = pt.x[0];
         ip.y = pt.x[1];
         ip.weight = pt.w;
         i = i + 1;
         MFEM_ASSERT(ip.weight > 0, "integration point weight is negative in domain integration from Saye's method");
         MFEM_ASSERT((phi(pt.x) < tol), " phi = " << phi(pt.x) << " : "
                                                  << " levelset function positive at the quadrature point domain integration (Saye's method)");
      }
      CutSquareIntRules[elemid] = ir;
      //  cout << "int size for element " << elemid << "   " << ir->Size() << endl;
   }
}

template <int N>
void GetCutSegmentIntRule(Mesh *mesh, vector<int> cutelems, vector<int> cutinteriorFaces,
                          int order, double r, std::map<int, IntegrationRule *> &cutSegmentIntRules,
                          std::map<int, IntegrationRule *> &cutInteriorFaceIntRules)
{
   for (int k = 0; k < cutelems.size(); ++k)
   {
      IntegrationRule *ir;
      blitz::TinyVector<double, N> xmin;
      blitz::TinyVector<double, N> xmax;
      blitz::TinyVector<double, N> xupper;
      blitz::TinyVector<double, N> xlower;
      int side;
      int dir;
      double tol = 1e-16;
      // standard reference element
      xlower = {0, 0};
      xupper = {1, 1};
      int elemid = cutelems.at(k);
      findBoundingBox<N>(mesh, elemid, xmin, xmax);
      circle<N> phi;
      phi.xscale = xmax[0] - xmin[0];
      phi.yscale = xmax[1] - xmin[1];
      phi.xmin = xmin[0];
      phi.ymin = xmin[1];
      phi.radius = r;
      dir = N;
      side = -1;
      auto q = Algoim::quadGen<N>(phi, Algoim::BoundingBox<double, N>(xlower, xupper), dir, side, order);
      int i = 0;
      ir = new IntegrationRule(q.nodes.size());
      for (const auto &pt : q.nodes)
      {
         IntegrationPoint &ip = ir->IntPoint(i);
         ip.x = pt.x[0];
         ip.y = pt.x[1];
         ip.weight = pt.w;
         i = i + 1;
         // cout << "elem " << elemid << " , " << ip.weight << endl;
         double xqp = (pt.x[0] * phi.xscale) + phi.xmin;
         double yqp = (pt.x[1] * phi.yscale) + phi.ymin;
         MFEM_ASSERT(ip.weight > 0, "integration point weight is negative in curved surface int rule from Saye's method");
      }
      cutSegmentIntRules[elemid] = ir;
      Array<int> orient;
      Array<int> fids;
      mesh->GetElementEdges(elemid, fids, orient);
      int fid;
      for (int c = 0; c < fids.Size(); ++c)
      {
         fid = fids[c];
         if (find(cutinteriorFaces.begin(), cutinteriorFaces.end(), fid) != cutinteriorFaces.end())
         {
            if (cutInteriorFaceIntRules[fid] == NULL)
            {
               Array<int> v;
               mesh->GetEdgeVertices(fid, v);
               double *v1coord, *v2coord;
               v1coord = mesh->GetVertex(v[0]);
               v2coord = mesh->GetVertex(v[1]);
               if (v1coord[0] == v2coord[0])
               {
                  dir = 0;
                  if (v1coord[0] < xmax[0])
                  {
                     side = 0;
                  }
                  else
                  {
                     side = 1;
                  }
               }
               else
               {
                  dir = 1;
                  if (v1coord[1] < xmax[1])
                  {
                     side = 0;
                  }
                  else
                  {
                     side = 1;
                  }
               }
               auto q = Algoim::quadGen<N>(phi, Algoim::BoundingBox<double, N>(xlower, xupper), dir, side, order);
               int i = 0;
               ir = new IntegrationRule(q.nodes.size());
               for (const auto &pt : q.nodes)
               {
                  IntegrationPoint &ip = ir->IntPoint(i);
                  ip.y = 0.0;
                  if (dir == 0)
                  {
                     if (-1 == orient[c])
                     {
                        ip.x = 1 - pt.x[1];
                     }
                     else
                     {
                        ip.x = pt.x[1];
                     }
                  }
                  else if (dir == 1)
                  {
                     if (-1 == orient[c])
                     {
                        ip.x = 1 - pt.x[0];
                     }
                     else
                     {
                        ip.x = pt.x[0];
                     }
                  }
                  ip.weight = pt.w;
                  i = i + 1;

                  // scaled to original element space
                  double xq = (pt.x[0] * phi.xscale) + phi.xmin;
                  double yq = (pt.x[1] * phi.yscale) + phi.ymin;
                  MFEM_ASSERT(ip.weight > 0, "integration point weight is negative from Saye's method");
                  MFEM_ASSERT((phi(pt.x) < tol), " phi = " << phi(pt.x) << " : "
                                                           << "levelset function positive at the quadrature point (Saye's method)");
                  MFEM_ASSERT((xq <= (max(v1coord[0], v2coord[0]))) && (xq >= (min(v1coord[0], v2coord[0]))),
                              "integration point (xcoord) not on element face (Saye's rule)");
                  MFEM_ASSERT((yq <= (max(v1coord[1], v2coord[1]))) && (yq >= (min(v1coord[1], v2coord[1]))),
                              "integration point (ycoord) not on element face (Saye's rule)");
               }
               cutInteriorFaceIntRules[fid] = ir;
            }
         }
      }
   }
}

bool insideBoundary(Mesh *mesh, double r, int &elemid)
{
   Element *el = mesh->GetElement(elemid);
   Array<int> v;
   el->GetVertices(v);
   int k;
   k = 0;
   double xc = 0.5;
   double yc = 0.5;
   for (int i = 0; i < v.Size(); ++i)
   {
      double *coord = mesh->GetVertex(v[i]);
      Vector lvsval(v.Size());
      lvsval(i) = ((coord[0] - xc) * (coord[0] - xc)) + ((coord[1] - yc) * (coord[1] - yc)) - (r * r);
      if ((lvsval(i) < 0) || (lvsval(i) == 0))
      {
         k = k + 1;
      }
   }
   if (k == v.Size())
   {
      return true;
   }
   else
   {
      return false;
   }
}
// function to get element center
void GetElementCenter(Mesh *mesh, int id, mfem::Vector &cent)
{
   cent.SetSize(mesh->Dimension());
   int geom = mesh->GetElement(id)->GetGeometryType();
   ElementTransformation *eltransf = mesh->GetElementTransformation(id);
   eltransf->Transform(Geometries.GetCenter(geom), cent);
}

// find bounding box for a given cut element
template <int N>
void findBoundingBox(Mesh *mesh, int id, blitz::TinyVector<double, N> &xmin, blitz::TinyVector<double, N> &xmax)
{
   Element *el = mesh->GetElement(id);
   Array<int> v;
   Vector min, max;
   min.SetSize(N);
   max.SetSize(N);
   for (int d = 0; d < N; d++)
   {
      min(d) = infinity();
      max(d) = -infinity();
   }
   el->GetVertices(v);
   for (int iv = 0; iv < v.Size(); ++iv)
   {
      double *coord = mesh->GetVertex(v[iv]);
      for (int d = 0; d < N; d++)
      {
         if (coord[d] < min(d))
         {
            min(d) = coord[d];
         }
         if (coord[d] > max(d))
         {
            max(d) = coord[d];
         }
      }
   }
   xmin = {min[0], min[1]};
   xmax = {max[0], max[1]};
}

void CutDomainLFIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                                   ElementTransformation &Tr,
                                                   Vector &elvect)
{
   int dof = el.GetDof();
   shape.SetSize(dof); // vector of size dof
   elvect.SetSize(dof);
   elvect = 0.0;
   const IntegrationRule *ir;
   ir = CutIntRules[Tr.ElementNo];
   if (ir == NULL)
   {
      // ir = &IntRules.Get(el.GetGeomType(),
      //                    oa * el.GetOrder() + ob + Tr.OrderW());
      ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);
   }
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint(&ip);
      double val = Tr.Weight() * Q.Eval(Tr, ip);

      el.CalcShape(ip, shape);

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

void CutDomainIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                                 ElementTransformation &Tr,
                                                 Vector &elvect)
{
   int dof = el.GetDof();
   shape.SetSize(dof); // vector of size dof
   elvect.SetSize(dof);
   elvect = 0.0;
   const IntegrationRule *ir;
   ir = CutIntRules[Tr.ElementNo];
   if (ir == NULL)
   {
      ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);
   }
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint(&ip);
      double val = Tr.Weight() * Q.Eval(Tr, ip);

      el.CalcShape(ip, shape);

      add(elvect, ip.weight * val, shape, elvect);
   }
}

void CutDomainIntegrator::AssembleDeltaElementVect(
    const FiniteElement &fe, ElementTransformation &Trans, Vector &elvect)
{
   MFEM_ASSERT(delta != NULL, "coefficient must be DeltaCoefficient");
   elvect.SetSize(fe.GetDof());
   fe.CalcPhysShape(Trans, elvect);
   elvect *= delta->EvalDelta(Trans, Trans.GetIntPoint());
}

// function to see if an element is cut-element
bool cutByCircle(Mesh *mesh, double r, int &elemid)
{
   Element *el = mesh->GetElement(elemid);
   Array<int> v;
   el->GetVertices(v);
   int k, l, n;
   k = 0;
   l = 0;
   n = 0;
   double xc = 0.5;
   double yc = 0.5;
   for (int i = 0; i < v.Size(); ++i)
   {
      double *coord = mesh->GetVertex(v[i]);
      Vector lvsval(v.Size());
      lvsval(i) = ((coord[0] - xc) * (coord[0] - xc)) + ((coord[1] - yc) * (coord[1] - yc)) - (r * r);
      if ((lvsval(i) < 0) && (abs(lvsval(i)) > 1e-16))
      {
         k = k + 1;
      }
      if ((lvsval(i) > 0))
      {
         l = l + 1;
      }
      if ((lvsval(i) == 0) || (abs(lvsval(i)) < 1e-16))
      {
         n = n + 1;
      }
   }
   if ((k == v.Size()) || (l == v.Size()))
   {
      return false;
   }
   if (((k == 3) || (l == 3)) && (n == 1))
   {
      return false;
   }
   else
   {
      return true;
   }
}

void CutDiffusionIntegrator::AssembleElementMatrix(const FiniteElement &el, ElementTransformation &Trans,
                                                   DenseMatrix &elmat)
{
   int nd = el.GetDof();
   int dim = el.GetDim();
   int spaceDim = Trans.GetSpaceDim();
   bool square = (dim == spaceDim);
   double w;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape(nd, dim), dshapedxt(nd, spaceDim), invdfdx(dim, spaceDim);
#else
   dshape.SetSize(nd, dim);
   dshapedxt.SetSize(nd, spaceDim);
   invdfdx.SetSize(dim, spaceDim);
#endif
   elmat.SetSize(nd);
   elmat = 0.0;
   // elmat is identity for embedded elements
   if (EmbeddedElements.at(Trans.ElementNo) == true)
   {
      for (int k = 0; k < elmat.Size(); ++k)
      {
         elmat(k, k) = 1.0;
      }
   }
   else
   {
      // use Saye's quadrature rule for elements cut by boundary
      const IntegrationRule *ir;
      ir = CutIntRules[Trans.ElementNo];
      if (ir == NULL)
      {
         ir = IntRule ? IntRule : &GetRule(el, el);
      }
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         el.CalcDShape(ip, dshape);
         Trans.SetIntPoint(&ip);
         w = Trans.Weight();
         w = ip.weight / (square ? w : w * w * w);
         // AdjugateJacobian = / adj(J),         if J is square
         //                    \ adj(J^t.J).J^t, otherwise
         Mult(dshape, Trans.AdjugateJacobian(), dshapedxt);
         if (!MQ)
         {
            if (Q)
            {
               w *= Q->Eval(Trans, ip);
            }
            AddMult_a_AAt(w, dshapedxt, elmat);
         }
         else
         {
            MQ->Eval(invdfdx, Trans, ip);
            invdfdx *= w;
            Mult(dshapedxt, invdfdx, dshape);
            AddMultABt(dshape, dshapedxt, elmat);
         }
      }
   }
}

const IntegrationRule &CutDiffusionIntegrator::GetRule(
    const FiniteElement &trial_fe, const FiniteElement &test_fe)
{
   int order;
   if (trial_fe.Space() == FunctionSpace::Pk)
   {
      order = trial_fe.GetOrder() + test_fe.GetOrder() - 2;
   }
   else
   {
      // order = 2*el.GetOrder() - 2;  // <-- this seems to work fine too
      order = trial_fe.GetOrder() + test_fe.GetOrder() + trial_fe.GetDim() - 1;
   }

   if (trial_fe.Space() == FunctionSpace::rQk)
   {
      return RefinedIntRules.Get(trial_fe.GetGeomType(), order);
   }
   return IntRules.Get(trial_fe.GetGeomType(), order);
}

void CutBoundaryFaceIntegrator::AssembleElementMatrix(const FiniteElement &el,
                                                      ElementTransformation &Trans,
                                                      DenseMatrix &elmat)
{
   int dim, ndof1, ndofs;
   bool kappa_is_nonzero = (kappa != 0.);
   double w, wq = 0.0;
   dim = el.GetDim();
   ndof1 = el.GetDof();
   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   adjJ.SetSize(dim);
   if (MQ)
   {
      mq.SetSize(dim);
   }
   shape1.SetSize(ndof1);
   dshape1.SetSize(ndof1, dim);
   dshape1dn.SetSize(ndof1);
   ndofs = ndof1;
   elmat.SetSize(ndofs);
   elmat = 0.0;
   if (kappa_is_nonzero)
   {
      jmat.SetSize(ndofs);
      jmat = 0.;
   }
   const IntegrationRule *ir;
   ir = cutSegmentIntRules[Trans.ElementNo];
   if (ir == NULL)
   {
      // cout << "element is " << Trans.ElementNo << endl;
      elmat = 0.0;
   }
   // assemble: < {(Q \nabla u).n},[v] >      --> elmat
   //           kappa < {h^{-1} Q} [u],[v] >  --> jmat
   else
   {
      for (int p = 0; p < ir->GetNPoints(); p++)
      {
         const IntegrationPoint &ip = ir->IntPoint(p);
         IntegrationPoint eip1;
         eip1 = ip;
         Trans.SetIntPoint(&ip);
         double ds;
         if (dim == 1)
         {
            nor(0) = 2 * eip1.x - 1.0;
         }
         else
         {
            //CalcOrtho(Trans.Jacobian(), nor);
            // double ds = sqrt((eip1.x*eip1.x) + (eip1.y*eip1.y));
            Vector v(dim);
            Trans.Transform(eip1, v);
            double nx = 2 * (v(0) - 0.5);
            double ny = 2 * (v(1) - 0.5);
            ds = sqrt((nx * nx) + (ny * ny));
            nor(0) = -nx / ds;
            nor(1) = -ny / ds;
         }
         el.CalcShape(eip1, shape1);
         el.CalcDShape(eip1, dshape1);
         Trans.SetIntPoint(&eip1);
         w = ip.weight / Trans.Weight();
         if (!MQ)
         {
            if (Q)
            {
               w *= Q->Eval(Trans, eip1);
            }
            ni.Set(w, nor);
         }
         else
         {
            nh.Set(w, nor);
            MQ->Eval(mq, Trans, eip1);
            mq.MultTranspose(nh, ni);
         }
         CalcAdjugate(Trans.Jacobian(), adjJ);
         adjJ.Mult(ni, nh);
         if (kappa_is_nonzero)
         {
            wq = ni * nor;
            //wq = ip.weight*nor.Norml2();
         }
         // std::cout << "normal is " << nor(0) << " , " << nor(1) << endl;
         // std::cout << "norm is " << nor * nor << endl;
         // std::cout << "ds is " << ds << std::endl;
         // std::cout << "wq/ip.weight " << wq / ip.weight << std::endl;
         // Note: in the jump term, we use 1/h1 = |nor|/det(J1) which is
         // independent of Loc1 and always gives the size of element 1 in
         // direction perpendicular to the face. Indeed, for linear transformation
         //     |nor|=measure(face)/measure(ref. face),
         //   det(J1)=measure(element)/measure(ref. element),
         // and the ratios measure(ref. element)/measure(ref. face) are
         // compatible for all element/face pairs.
         // For example: meas(ref. tetrahedron)/meas(ref. triangle) = 1/3, and
         // for any tetrahedron vol(tet)=(1/3)*height*area(base).
         // For interior faces: q_e/h_e=(q1/h1+q2/h2)/2.
         dshape1.Mult(nh, dshape1dn);
         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < ndof1; j++)
            {
               elmat(i, j) += shape1(i) * dshape1dn(j);
            }

         if (kappa_is_nonzero)
         {
            // only assemble the lower triangular part of jmat
            wq *= kappa;
            for (int i = 0; i < ndof1; i++)
            {
               const double wsi = wq * shape1(i);
               for (int j = 0; j <= i; j++)
               {
                  jmat(i, j) += wsi * shape1(j);
               }
            }
         }
      }

      // elmat := -elmat + sigma*elmat^t + jmat
      if (kappa_is_nonzero)
      {
         for (int i = 0; i < ndofs; i++)
         {
            for (int j = 0; j < i; j++)
            {
               double aij = elmat(i, j), aji = elmat(j, i), mij = jmat(i, j);
               elmat(i, j) = sigma * aji - aij + mij;
               elmat(j, i) = sigma * aij - aji + mij;
            }
            elmat(i, i) = (sigma - 1.) * elmat(i, i) + jmat(i, i);
         }
      }
      else
      {
         for (int i = 0; i < ndofs; i++)
         {
            for (int j = 0; j < i; j++)
            {
               double aij = elmat(i, j), aji = elmat(j, i);
               elmat(i, j) = sigma * aji - aij;
               elmat(j, i) = sigma * aij - aji;
            }
            elmat(i, i) *= (sigma - 1.);
         }
      }
   }
}

void CutDGDiffusionIntegrator::AssembleFaceMatrix(
    const FiniteElement &el1, const FiniteElement &el2,
    FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   int dim, ndof1, ndof2, ndofs;
   bool kappa_is_nonzero = (kappa != 0.);
   double w, wq = 0.0;
   dim = el1.GetDim();
   ndof1 = el1.GetDof();
   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   adjJ.SetSize(dim);
   if (MQ)
   {
      mq.SetSize(dim);
   }
   shape1.SetSize(ndof1);
   dshape1.SetSize(ndof1, dim);
   dshape1dn.SetSize(ndof1);
   if (Trans.Elem2No >= 0)
   {
      ndof2 = el2.GetDof();
      shape2.SetSize(ndof2);
      dshape2.SetSize(ndof2, dim);
      dshape2dn.SetSize(ndof2);
   }
   else
   {
      ndof2 = 0;
   }
   ndofs = ndof1 + ndof2;
   elmat.SetSize(ndofs);
   elmat = 0.0;
   if (immersedFaces[Trans.Face->ElementNo] == true)
   {
      elmat = 0.0;
   }
   else
   {
      if (kappa_is_nonzero)
      {
         jmat.SetSize(ndofs);
         jmat = 0.;
      }
      const IntegrationRule *ir;
      if (find(cutinteriorFaces.begin(), cutinteriorFaces.end(), Trans.Face->ElementNo) != cutinteriorFaces.end())
      {
         ir = cutInteriorFaceIntRules[Trans.Face->ElementNo];
      }
      else
      {
         ir = IntRule;
      }
      if (ir == NULL)
      {
         // a simple choice for the integration order; is this OK?
         int order;
         if (ndof2)
         {
            order = 2 * max(el1.GetOrder(), el2.GetOrder());
         }
         else
         {

            order = 2 * el1.GetOrder();
         }
         ir = &IntRules.Get(Trans.FaceGeom, order);
      }
      // assemble: < {(Q \nabla u).n},[v] >      --> elmat
      //           kappa < {h^{-1} Q} [u],[v] >  --> jmat
      // if (find(cutinteriorFaces.begin(), cutinteriorFaces.end(), Trans.Face->ElementNo) != cutinteriorFaces.end())
      // {
      //    std::cout << "face is " << Trans.Face->ElementNo << " elements are " << Trans.Elem1No << " , " << Trans.Elem2No << std::endl;
      // }
      for (int p = 0; p < ir->GetNPoints(); p++)
      {
         const IntegrationPoint &ip = ir->IntPoint(p);
         IntegrationPoint eip1, eip2;
         Trans.Loc1.Transform(ip, eip1);
         Trans.Face->SetIntPoint(&ip);
         if (dim == 1)
         {
            nor(0) = 2 * eip1.x - 1.0;
         }
         else
         {
            CalcOrtho(Trans.Face->Jacobian(), nor);
         }
         el1.CalcShape(eip1, shape1);
         el1.CalcDShape(eip1, dshape1);
         Trans.Elem1->SetIntPoint(&eip1);
         w = ip.weight / Trans.Elem1->Weight();
         if (ndof2)
         {
            w /= 2;
         }
         if (!MQ)
         {
            if (Q)
            {
               w *= Q->Eval(*Trans.Elem1, eip1);
            }
            ni.Set(w, nor);
         }
         else
         {
            nh.Set(w, nor);
            MQ->Eval(mq, *Trans.Elem1, eip1);
            mq.MultTranspose(nh, ni);
         }
         CalcAdjugate(Trans.Elem1->Jacobian(), adjJ);
         adjJ.Mult(ni, nh);
         if (kappa_is_nonzero)
         {
            wq = ni * nor;
         }
         // Note: in the jump term, we use 1/h1 = |nor|/det(J1) which is
         // independent of Loc1 and always gives the size of element 1 in
         // direction perpendicular to the face. Indeed, for linear transformation
         //     |nor|=measure(face)/measure(ref. face),
         //   det(J1)=measure(element)/measure(ref. element),
         // and the ratios measure(ref. element)/measure(ref. face) are
         // compatible for all element/face pairs.
         // For example: meas(ref. tetrahedron)/meas(ref. triangle) = 1/3, and
         // for any tetrahedron vol(tet)=(1/3)*height*area(base).
         // For interior faces: q_e/h_e=(q1/h1+q2/h2)/2.

         dshape1.Mult(nh, dshape1dn);
         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < ndof1; j++)
            {
               elmat(i, j) += shape1(i) * dshape1dn(j);
            }

         if (ndof2)
         {
            Trans.Loc2.Transform(ip, eip2);
            Vector v(dim);
            Trans.Elem2->Transform(eip2, v);
            el2.CalcShape(eip2, shape2);
            el2.CalcDShape(eip2, dshape2);
            Trans.Elem2->SetIntPoint(&eip2);
            w = ip.weight / 2 / Trans.Elem2->Weight();
            if (!MQ)
            {
               if (Q)
               {
                  w *= Q->Eval(*Trans.Elem2, eip2);
               }
               ni.Set(w, nor);
            }
            else
            {
               nh.Set(w, nor);
               MQ->Eval(mq, *Trans.Elem2, eip2);
               mq.MultTranspose(nh, ni);
            }
            CalcAdjugate(Trans.Elem2->Jacobian(), adjJ);
            adjJ.Mult(ni, nh);
            if (kappa_is_nonzero)
            {
               wq += ni * nor;
            }

            dshape2.Mult(nh, dshape2dn);

            for (int i = 0; i < ndof1; i++)
               for (int j = 0; j < ndof2; j++)
               {
                  elmat(i, ndof1 + j) += shape1(i) * dshape2dn(j);
               }

            for (int i = 0; i < ndof2; i++)
               for (int j = 0; j < ndof1; j++)
               {
                  elmat(ndof1 + i, j) -= shape2(i) * dshape1dn(j);
               }

            for (int i = 0; i < ndof2; i++)
               for (int j = 0; j < ndof2; j++)
               {
                  elmat(ndof1 + i, ndof1 + j) -= shape2(i) * dshape2dn(j);
               }
         }

         if (kappa_is_nonzero)
         {
            // only assemble the lower triangular part of jmat
            wq *= kappa;
            for (int i = 0; i < ndof1; i++)
            {
               const double wsi = wq * shape1(i);
               for (int j = 0; j <= i; j++)
               {
                  jmat(i, j) += wsi * shape1(j);
               }
            }
            if (ndof2)
            {
               for (int i = 0; i < ndof2; i++)
               {
                  const int i2 = ndof1 + i;
                  const double wsi = wq * shape2(i);
                  for (int j = 0; j < ndof1; j++)
                  {
                     jmat(i2, j) -= wsi * shape1(j);
                  }
                  for (int j = 0; j <= i; j++)
                  {
                     jmat(i2, ndof1 + j) += wsi * shape2(j);
                  }
               }
            }
         }
      }

      // elmat := -elmat + sigma*elmat^t + jmat
      if (kappa_is_nonzero)
      {
         for (int i = 0; i < ndofs; i++)
         {
            for (int j = 0; j < i; j++)
            {
               double aij = elmat(i, j), aji = elmat(j, i), mij = jmat(i, j);
               elmat(i, j) = sigma * aji - aij + mij;
               elmat(j, i) = sigma * aij - aji + mij;
            }
            elmat(i, i) = (sigma - 1.) * elmat(i, i) + jmat(i, i);
         }
      }
      else
      {
         for (int i = 0; i < ndofs; i++)
         {
            for (int j = 0; j < i; j++)
            {
               double aij = elmat(i, j), aji = elmat(j, i);
               elmat(i, j) = sigma * aji - aij;
               elmat(j, i) = sigma * aij - aji;
            }
            elmat(i, i) *= (sigma - 1.);
         }
      }
   }
}

void CutDGDirichletLFIntegrator::AssembleRHSElementVect(
    const FiniteElement &el, ElementTransformation &Trans,
    Vector &elvect)
{
   int dim, ndof;
   bool kappa_is_nonzero = (kappa != 0.);
   double w;
   dim = el.GetDim();
   ndof = el.GetDof();
   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   adjJ.SetSize(dim);
   if (MQ)
   {
      mq.SetSize(dim);
   }
   shape.SetSize(ndof);
   dshape.SetSize(ndof, dim);
   dshape_dn.SetSize(ndof);

   elvect.SetSize(ndof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   ir = cutSegmentIntRules[Trans.ElementNo];
   if (ir == NULL)
   {
      elvect = 0.0;
   }
   else
   {
      for (int p = 0; p < ir->GetNPoints(); p++)
      {
         const IntegrationPoint &ip = ir->IntPoint(p);
         IntegrationPoint eip;
         eip = ip;
         if (dim == 1)
         {
            nor(0) = 2 * eip.x - 1.0;
         }
         else
         {
            Vector v(dim);
            Trans.Transform(eip, v);
            double nx = 2 * (v(0) - 0.5);
            double ny = 2 * (v(1) - 0.5);
            double ds = sqrt((nx * nx) + (ny * ny));
            nor(0) = -nx / ds;
            nor(1) = -ny / ds;
         }
         el.CalcShape(eip, shape);
         el.CalcDShape(eip, dshape);
         Trans.SetIntPoint(&eip);
         // compute uD through the face transformation
         w = ip.weight * uD->Eval(Trans, ip) / Trans.Weight();
         if (!MQ)
         {
            if (Q)
            {
               w *= Q->Eval(Trans, eip);
            }
            ni.Set(w, nor);
         }
         else
         {
            nh.Set(w, nor);
            MQ->Eval(mq, Trans, eip);
            mq.MultTranspose(nh, ni);
         }
         CalcAdjugate(Trans.Jacobian(), adjJ);
         adjJ.Mult(ni, nh);

         dshape.Mult(nh, dshape_dn);
         elvect.Add(sigma, dshape_dn);

         if (kappa_is_nonzero)
         {
            elvect.Add(kappa * (ni * nor), shape);
         }
      }
   }
}

void CutDGDirichletLFIntegrator::AssembleDeltaElementVect(
    const FiniteElement &fe, ElementTransformation &Trans, Vector &elvect)
{
   MFEM_ASSERT(delta != NULL, "coefficient must be DeltaCoefficient");
   elvect.SetSize(fe.GetDof());
   fe.CalcPhysShape(Trans, elvect);
   elvect *= delta->EvalDelta(Trans, Trans.GetIntPoint());
}

void CutDGNeumannLFIntegrator::AssembleRHSElementVect(
    const FiniteElement &el, ElementTransformation &Trans,
    Vector &elvect)
{
   int dim, ndof;
   double w;
   Vector Qvec;
   dim = el.GetDim();
   ndof = el.GetDof();
   shape.SetSize(ndof);
   elvect.SetSize(ndof);
   elvect = 0.0;
   nor.SetSize(dim);
   const IntegrationRule *ir = IntRule;
   ir = cutSegmentIntRules[Trans.ElementNo];
   // elvect is zero for elements other than cut
   if (ir == NULL)
   {
      elvect = 0.0;
   }
   else
   {
      for (int p = 0; p < ir->GetNPoints(); p++)
      {
         const IntegrationPoint &ip = ir->IntPoint(p);
         el.CalcShape(ip, shape);
         Trans.SetIntPoint(&ip);
         // this evaluates the coefficient for the
         // integration points in physical space
         QN.Eval(Qvec, Trans, ip);
         Vector v(dim);
         // transform the integration point to original element
         Trans.Transform(ip, v);
         double nx = 2 * (v(0) - 0.5);
         double ny = 2 * (v(1) - 0.5);
         double ds = sqrt((nx * nx) + (ny * ny));
         nor(0) = -nx / ds;
         nor(1) = -ny / ds;
         elvect.Add(ip.weight * sqrt(Trans.Weight()) * (Qvec * nor), shape);
      }
   }
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
         // cout << "int rule for element " << elmt_id[j] << endl;
         // quad_coord.Print();
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
void GalerkinDifference::GetNeighbourSet(int id, int req_n, Array<int> &nels_x, Array<int> &nels_y) const
{}
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
   // Get and store the element center
   mfem::Vector cent_coord(dim);
   GetElementCenter(id, cent_coord);
   // cout << "List is initialized as: ";
   // nels.Print(cout, nels.Size());
   // cout << "Initial candidates: ";
   // cand.Print(cout, cand.Size());
   while (nels.Size() < req_n)
   {
      for (int i = 0; i < adj.Size(); i++)
      {
         if (-1 == nels.Find(adj[i]))
         {
            // Get and store the element center
            mfem::Vector cent(dim);
            GetElementCenter(adj[i], cent);
            if ((cent(0) == cent_coord(0)) || (cent(1) == cent_coord(1)))
            {
               if (EmbeddedElements.at(adj[i]) == false)
               {
                  nels.Append(adj[i]);
               }
            }
         }
      }
      // cout << "List now is: ";
      // nels.Print(cout, nels.Size());
      adj.LoseData();
      for (int i = 0; i < cand.Size(); i++)
      {
         //cout << "deal with cand " << cand[i];
         if (EmbeddedElements.at(cand[i]) == false)
         {
            mesh->ElementToElementTable().GetRow(cand[i], cand_adj);
            // cout << "'s adj are ";
            // cand_adj.Print(cout, cand_adj.Size());
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
      if (EmbeddedElements.at(i) == false)
      {
         cout << " element is " << i << endl;
         // 1. get the elements in patch
         GetNeighbourSet(i, nelmt, elmt_id);
         cout << "element "
              << "( " << i << ") "
              << " #neighbours = " << elmt_id.Size() << endl;
         cout << "Elements id(s) in patch: ";
         elmt_id.Print(cout, elmt_id.Size());
         cout << " ----------------------- " << endl;

         // 2. build the quadrature and barycenter coordinate matrices
         BuildNeighbourMat(elmt_id, cent_mat, quad_mat);

         // 3. buil the local reconstruction matrix
         //cout << "element is " << i << endl;
         buildLSInterpolation(dim, degree, cent_mat, quad_mat, local_mat);
         //cout << " ######################### " << endl;
         // cout << "Local reconstruction matrix R:\n";
         // local_mat.Print(cout, local_mat.Width());

         // 4. assemble them back to prolongation matrix
         AssembleProlongationMatrix(elmt_id, local_mat);
      }
      else
      {
         elmt_id.LoseData();
         elmt_id.Append(i);

         local_mat.SetSize(num_dofs, 1);

         for (int k = 0; k < num_dofs; ++k)
         {
            local_mat(k, 0) = 1.0;
         }

         AssembleProlongationMatrix(elmt_id, local_mat);
      }
   }
   cP->Finalize();
   cP_is_set = true;
   cout << "Check cP size: " << cP->Height() << " x " << cP->Width() << '\n';
   ofstream cp_save("cp.txt");
   cP->PrintMatlab(cp_save);
   cp_save.close();
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
   int lwork = (num_elem * num_basis) + (3 * num_basis) + 1;
   double work[lwork];
   int rank;
   Array<int> jpvt;
   jpvt.SetSize(num_basis);
   jpvt = 0;
   double rcond = 1e-16;
   ofstream write("V_mat.txt");
   write.precision(16);
   V.PrintMatlab(write);
   write.close();
   //V.PrintMatlab();
   // cout << "rank is " << V.Rank(1e-12) << endl;
   // dgels_(&TRANS, &num_elem, &num_basis, &num_elem, V.GetData(), &num_elem,
   //        coeff.GetData(), &num_elem, work, &lwork, &info);
   dgelsy_(&num_elem, &num_basis, &num_elem, V.GetData(), &num_elem, coeff.GetData(),
           &num_elem, jpvt.GetData(), &rcond, &rank, work, &lwork, &info);
   //cout<< "info is " << info << endl;

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
      // loop over quadrature points
      for (int j = 0; j < num_quad; ++j)
      {
         for (int p = 0; p <= degree; ++p)
         {
            for (int q = 0; q <= p; ++q)
            {
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
               if ((p == 0) && (q == 0))
               {
                  MFEM_ASSERT(fabs(exact - poly_at_quad) <= 1e-12, " p = " << p << " , q = " << q << " : "
                                                                           << "Interpolation operator does not interpolate exactly!\n");
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
