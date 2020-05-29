#include "exCircle.hpp"
using namespace std;
using namespace mfem;
double u_exact(const Vector &);
double f_exact(const Vector &);
void u_neumann(const Vector &, Vector &);
double CutComputeL2Error( GridFunction &x, FiniteElementSpace *fes,
    Coefficient &exsol, const std::vector<bool> &EmbeddedElems, 
    std::map<int, IntegrationRule *> &CutSquareIntRules);
template <int N>
struct circle
{
   double xscale;
   double yscale;
   double xmin;
   double ymin;
   template <typename T>
   T operator()(const blitz::TinyVector<T, N> &x) const
   {
      // level-set function to work in physical space
      // return -1 * (((x[0] - 5) * (x[0] - 5)) +
      //               ((x[1]- 5) * (x[1] - 5)) - (0.5 * 0.5));
      // level-set function for reference elements
      return -1 * ((((x[0] * xscale) + xmin - 0.5) * ((x[0] * xscale) + xmin - 0.5)) +
                   (((x[1] * yscale) + ymin - 0.5) * ((x[1] * yscale) + ymin - 0.5)) - (0.04));
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
   int order = 2;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   double sigma = -1.0;
   double kappa = 100.0;
   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
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

   Mesh *mesh = new Mesh(15, 15, Element::QUADRILATERAL, true,
                         1, 1, true);
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
      if (cutByCircle(mesh, i) == true)
      {
         cutelems.push_back(i);
      }
      if (insideBoundary(mesh, i) == true)
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
   // define map for integration rule for cut elements
   GetCutElementIntRule<2>(mesh, cutelems, CutSquareIntRules);
   GetCutSegmentIntRule<2>(mesh, cutelems, cutinteriorFaces, cutSegmentIntRules,
                           cutInteriorFaceIntRules);
   std::vector<bool> EmbeddedElems;
   for (int i = 0; i < mesh->GetNE(); ++i)
   {
      if (insideBoundary(mesh, i) == true)
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
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
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
   x = 0.0;
   //x.ProjectCoefficient(u);
   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new CutDiffusionIntegrator(one, CutSquareIntRules, EmbeddedElems));
   //a->AddDomainIntegrator(new CutBoundaryFaceIntegrator(one, sigma, kappa, cutSegmentIntRules));
   a->AddInteriorFaceIntegrator(new CutDGDiffusionIntegrator(one, sigma, kappa,
                                                             immersedFaces, cutinteriorFaces, 
                                                             cutInteriorFaceIntRules));
   a->AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
   a->Assemble();
   a->Finalize();
   const SparseMatrix &A = a->SpMat();
   //cout << "bilinear form size " << a->Size() << endl;
   //a->PrintMatlab();
   //A.PrintMatlab("stiffmat.txt");
   ofstream write("stiffmat.txt");
   A.PrintMatlab(write);
   write.close();
#ifndef MFEM_USE_SUITESPARSE
   // 8. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //    solve the system Ax=b with PCG in the symmetric case, and GMRES in the
   //    non-symmetric one.
   GSSmoother M(A);
   if (sigma == -1.0)
   {
      PCG(A, M, *b, x, 1, 1000, 1e-12, 0.0);
   }
   else
   {
      GMRES(A, M, *b, x, 1, 1000, 10, 1e-12, 0.0);
   }
#else
   // 8. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   umf_solver.SetOperator(A);
   umf_solver.Mult(*b, x);
#endif

   ofstream adj_ofs("dgSolcirclelap.vtk");
   adj_ofs.precision(14);
   mesh->PrintVTK(adj_ofs, 1);
   x.SaveVTK(adj_ofs, "dgSolution", 1);
   adj_ofs.close();
   // 10. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << *mesh << x << flush;
   }
   cout << "h" << " " << "solution norm: "  << endl;
   cout << 1.0/(sqrt(mesh->GetNE())) << " " << 
   CutComputeL2Error(x, fespace, u, EmbeddedElems, CutSquareIntRules) << endl;
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
   return sin(M_PI* x(0))*sin(M_PI*x(1));
   //return (2*x(0)) - (2*x(1));
}
double f_exact(const Vector &x)
{
   return 2*M_PI * M_PI* sin(M_PI*x(0)) * sin(M_PI* x(1));
}

void u_neumann(const Vector &x, Vector &u)
{
   u(0) = M_PI * cos(M_PI* x(0))*sin(M_PI*x(1));
   u(1) = M_PI * sin(M_PI* x(0))*cos(M_PI*x(1));
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
template <int N>
void GetCutElementIntRule(Mesh *mesh, vector<int> cutelems,
                          std::map<int, IntegrationRule *> &CutSquareIntRules)
{
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
      int order = 4;
      int elemid = cutelems.at(k);
      findBoundingBox<N>(mesh, elemid, xmin, xmax);
      circle<N> phi;
      phi.xscale = xmax[0] - xmin[0];
      phi.yscale = xmax[1] - xmin[1];
      phi.xmin = xmin[0];
      phi.ymin = xmin[1];
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
         MFEM_ASSERT(!(phi(pt.x) > 0), "levelset function positive at the quadrature point domain integration (Saye's method)");
      }
      CutSquareIntRules[elemid] = ir;
   }
}

template <int N>
void GetCutSegmentIntRule(Mesh *mesh, vector<int> cutelems, vector<int> cutinteriorFaces,
                          std::map<int, IntegrationRule *> &cutSegmentIntRules,
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
      int order;
      order = 4;
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
                  MFEM_ASSERT(!(phi(pt.x) > 0), "levelset function positive at the quadrature point (Saye's method)");
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

bool insideBoundary(Mesh *mesh, int &elemid)
{
   Element *el = mesh->GetElement(elemid);
   Array<int> v;
   el->GetVertices(v);
   int k;
   k = 0;
   double xc = 0.5;
   double yc = 0.5;
   double r = 0.2;
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
bool cutByCircle(Mesh *mesh, int &elemid)
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
   double r = 0.2;
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
            double nx = 2*(v(0) - 0.5);
            double ny = 2*(v(1) - 0.5);
            ds = sqrt((nx*nx) + (ny*ny));
            nor(0) = -nx/ds;
            nor(1) = -ny/ds;
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
         std::cout << "normal is " << nor(0) << " , " << nor(1) << endl;
         std::cout << "norm is " << nor*nor << endl;
         std::cout << "ds is " << ds << std::endl;
         std::cout << "wq/ip.weight " << wq/ip.weight << std::endl;
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
      if (find(cutinteriorFaces.begin(), cutinteriorFaces.end(), Trans.Face->ElementNo) != cutinteriorFaces.end())
      {
         std::cout << "face is " << Trans.Face->ElementNo << " elements are " << Trans.Elem1No << " , " << Trans.Elem2No << std::endl;
      }
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
            double nx = 2*(v(0) - 0.5);
            double ny = 2*(v(1) - 0.5);
            double ds = sqrt((nx*nx) + (ny*ny));
            nor(0) = -nx/ds;
            nor(1) = -ny/ds;
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
         double nx = 2*(v(0) - 0.5);
         double ny = 2*(v(1) - 0.5);
         double ds = sqrt((nx*nx) + (ny*ny));
         nor(0) = -nx/ds;
         nor(1) = -ny/ds;
         elvect.Add(ip.weight*sqrt(Trans.Weight())*(Qvec*nor), shape);
      }
   }
}
