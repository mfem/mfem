#include "exCircle.hpp"
using namespace std;
using namespace mfem;

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
      // return -1 * ((((x[0]*xscale) + xmin - 5) * ((x[0]*xscale) + xmin - 5)) +
      //               (((x[1]*yscale) + ymin- 5) * ((x[1]*yscale) + ymin - 5)) - (1));
      return -1 * ((((x[0] * xscale) + xmin - 0.5) * ((x[0] * xscale) + xmin - 0.5)) +
                   (((x[1] * yscale) + ymin - 0.5) * ((x[1] * yscale) + ymin - 0.5)) - (0.04));
   }
   template <typename T>
   blitz::TinyVector<T, N> grad(const blitz::TinyVector<T, N> &x) const
   {
      // return blitz::TinyVector<T, N>(-1 * (2.0 * (x(0) - 5)), -1 * (2.0 * (x(1) - 5)));
      // return blitz::TinyVector<T, N>(-1 * (2.0 * xscale* ((x(0) *xscale) + xmin- 5)),
      //                                  -1 * (2.0 * yscale* ((x(1) * yscale) + ymin- 5)));
      return blitz::TinyVector<T, N>(-1 * (2.0 * xscale * ((x(0) * xscale) + xmin - 0.5)),
                                     -1 * (2.0 * yscale * ((x(1) * yscale) + ymin - 0.5)));
   }
};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;
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
   args.PrintOptions(cout);

   Mesh *mesh = new Mesh(5, 5, Element::QUADRILATERAL, true,
                         1, 1, true);
   ofstream sol_ofv("square_mesh_new.vtk");
   sol_ofv.precision(14);
   mesh->PrintVTK(sol_ofv, 0);
   std::map<int, IntegrationRule *> CutSquareIntRules;
   std::map<int, IntegrationRule *> CutSegIntRules;
   //find the elements cut by boundary
   vector<int> cutelems;
   vector<int> innerelems;
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
   cout << "dimension is " << dim << endl;
   std::cout << "Number of elements: " << mesh->GetNE() << '\n';
   FiniteElementCollection *fec;
   fec = new H1_FECollection(1, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;
   GridFunction x(fespace);
   NonlinearForm *b = new NonlinearForm(fespace);
   NonlinearForm *a = new NonlinearForm(fespace);
   //ConstantCoefficient one(1.0);
   // define map for integration rule for cut elements

   GetCutElementIntRule<2>(mesh, cutelems, CutSquareIntRules);
   GetCutSegIntRule<2>(mesh, cutelems, CutSegIntRules);
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
   b->AddBdrFaceIntegrator(new CutBoundaryNLFIntegrator());
   b->AddDomainIntegrator(new CutImmersedBoundaryNLFIntegrator(CutSegIntRules, EmbeddedElems));
   a->AddDomainIntegrator(new CutDomainNLFIntegrator(CutSquareIntRules, EmbeddedElems));
   double area;
   area = a->GetEnergy(x);
   double parameter;
   parameter = b->GetEnergy(x);
   cout << "area of desired domain is " << area << endl;
   cout << "parameter of desired domain is " << parameter << endl;
   //b->Assemble();
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
      int elemid = cutelems.at(k);
      findBoundingBox<N>(mesh, elemid, xmin, xmax);
      circle<N> phi;
      phi.xscale = xmax[0] - xmin[0];
      phi.yscale = xmax[1] - xmin[1];
      phi.xmin = xmin[0];
      phi.ymin = xmin[1];
      auto q = Algoim::quadGen<N>(phi, Algoim::BoundingBox<double, N>(xlower, xupper), -1, -1, 4);
      //auto q = Algoim::quadGen<N>(phi, Algoim::BoundingBox<double,N>(xmin, xmax), -1, -1, 1);
      //cout << "number of quadrature nodes: " << q.nodes.size() << endl;
      int i = 0;
      ir = new IntegrationRule(q.nodes.size());
      // cout << "quadrature rule for mapped elements " << endl;
      for (const auto &pt : q.nodes)
      {
         IntegrationPoint &ip = ir->IntPoint(i);
         ip.x = pt.x[0];
         ip.y = pt.x[1];
         ip.weight = pt.w;
         i = i + 1;
         //cout << pt.x[0] << " , " << pt.x[1] << endl;
         // cout << "mapped back " << endl;
         //cout << (pt.x[0] * phi.xscale) + phi.xmin << " , " << (pt.x[1] * phi.yscale) + phi.ymin  << endl;
      }
      //  cout << "element id for cut element " << elemid << endl;
      CutSquareIntRules[elemid] = ir;
   }
}
template <int N>
void GetCutSegIntRule(Mesh *mesh, vector<int> cutelems,
                      std::map<int, IntegrationRule *> &CutSegIntRules)
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
      int elemid = cutelems.at(k);
      findBoundingBox<N>(mesh, elemid, xmin, xmax);
      circle<N> phi;
      phi.xscale = xmax[0] - xmin[0];
      phi.yscale = xmax[1] - xmin[1];
      phi.xmin = xmin[0];
      phi.ymin = xmin[1];
      auto q = Algoim::quadGen<N>(phi, Algoim::BoundingBox<double, N>(xlower, xupper), N, -1, 4);
      //auto q = Algoim::quadGen<N>(phi, Algoim::BoundingBox<double,N>(xmin, xmax), -1, -1, 1);
      //cout << "number of quadrature nodes: " << q.nodes.size() << endl;
      int i = 0;
      ir = new IntegrationRule(q.nodes.size());
      // cout << "quadrature rule for mapped elements " << endl;
      //cout << "element is " << elemid << endl;
      for (const auto &pt : q.nodes)
      {
         IntegrationPoint &ip = ir->IntPoint(i);
         ip.x = pt.x[0];
         ip.y = pt.x[1];
         ip.weight = pt.w;
         i = i + 1;
         // cout << "level set is " << phi(pt.x) << endl;
         // cout << pt.x[0] << " , " << pt.x[1] << endl;
         // cout << "mapped back " << endl;
         //cout << (pt.x[0] * phi.xscale) + phi.xmin << " , " << (pt.x[1] * phi.yscale) + phi.ymin  << endl;
      }
      //  cout << "element id for cut element " << elemid << endl;
      CutSegIntRules[elemid] = ir;
   }
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
   double xc = 0.50;
   double yc = 0.50;
   double r = 0.20;
   for (int i = 0; i < v.Size(); ++i)
   {
      double *coord = mesh->GetVertex(v[i]);
      Vector lvsval(v.Size());
      //cout << x[1] << endl;
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
      if (elemid == 52)
      {
         cout << "level set value is " << lvsval(i) << endl;
         cout << "coordinates are " << coord[0] << " , " << coord[1] << endl;
         cout << "k, l, n values are " << k << ", " << l << " , " << n << endl;
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
   //cout << min[0] << " , " << max[0] << endl;
   xmin = {min[0], min[1]};
   xmax = {max[0], max[1]};
}

double CutDomainNLFIntegrator::GetElementEnergy(const FiniteElement &el,
                                                ElementTransformation &Ttr,
                                                const Vector &elfun)
{
   if (EmbeddedElements.at(Ttr.ElementNo) == true)
   {
      return 0;
   }
   int dof = el.GetDof(), dim = el.GetDim();
   double energy;
   DSh.SetSize(dof, dim); // this is not required
   const IntegrationRule *ir;
   ir = CutIntRules[Ttr.ElementNo];
   //ir=NULL;
   if (!ir)
   {
      ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 3)); // <---
   }
   energy = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Ttr.SetIntPoint(&ip);
      energy += ip.weight * Ttr.Weight();
   }
   return energy;
}

double CutBoundaryNLFIntegrator::GetElementEnergy(const FiniteElement &el,
                                                  ElementTransformation &Ttr,
                                                  const Vector &elfun)
{
   double energy;
   const IntegrationRule *ir;
   ir = NULL;
   // cout << Ttr.GetGeometryType() << endl;
   if (!ir)
   {
      ir = &(IntRules.Get(Ttr.GetGeometryType(), 2)); // <---
   }
   // cout << "element is " << Ttr.ElementNo << endl;
   // cout << "before trans weight is " <<  Ttr.Weight() << endl;
   energy = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Ttr.SetIntPoint(&ip);
      energy += ip.weight * Ttr.Weight();
   }
   return energy;
}

double CutImmersedBoundaryNLFIntegrator::GetElementEnergy(const FiniteElement &el,
                                                          ElementTransformation &Ttr,
                                                          const Vector &elfun)
{
   double energy;
   const IntegrationRule *ir;
   ir = CutSegIntRules[Ttr.ElementNo];
   if (!(ir))
   {
      return 0;
   }
   energy = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Ttr.SetIntPoint(&ip);
      //cout << "trans weight now is " << Ttr.Weight() << endl;
      double area =  sqrt(Ttr.Weight());
      energy += ip.weight * area;
   }
   return -energy;
}