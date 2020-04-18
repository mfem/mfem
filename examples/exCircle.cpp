#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "algoim_quad.hpp"

using namespace std;
using namespace mfem;
// function that checks if an element is `cut` by `embedded circle` or  not
bool cutByCircle(Mesh *mesh, int &elemid);
// function to get element center
void GetElementCenter(Mesh *mesh, int id, mfem::Vector &cent);
// find bounding box for a given cut element
template <int N>
void findBoundingBox(Mesh *mesh, int id, blitz::TinyVector<double,N> &xmin, blitz::TinyVector<double,N> &xmax);
template <int N>
struct circle
{
   template <typename T>
   T operator()(const blitz::TinyVector<T, N> &x) const
   {
      return -1 * (((x[0] - 5) * (x[0] - 5)) + ((x[1] - 5) * (x[1] - 5)) - (0.5 * 0.5));
   }
   template <typename T>
   blitz::TinyVector<T, N> grad(const blitz::TinyVector<T, N> &x) const
   {
      return blitz::TinyVector<T, N>(-1 * (2.0 * (x(0) - 5)), -1 * (2.0 * (x(1) - 5)));
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

   Mesh *mesh = new Mesh(10, 10, Element::QUADRILATERAL, true,
                         10, 10, true);
   int dim = mesh->Dimension();
   cout << dim << endl;
   std::cout << "Number of elements " << mesh->GetNE() << '\n';

   /* mesh on a circle*/
   Mesh *mesh2 = new Mesh(20, 20, Element::QUADRILATERAL, true,
                          2, 2 * M_PI, true);
   H1_FECollection *fec = new H1_FECollection(2, 2 /* = dim */);
   FiniteElementSpace *fes = new FiniteElementSpace(mesh2, fec, 2,
                                                    Ordering::byVDIM);

   // This lambda function transforms from (r,\theta) space to (x,y) space
   auto xy_fun = [](const Vector &rt, Vector &xy) {
      xy(0) = 0.25 * (rt(0)) * cos(rt(1)); // need + 1.0 to shift r away from origin
      xy(1) = 0.25 * (rt(0)) * sin(rt(1));
   };
   VectorFunctionCoefficient xy_coeff(2, xy_fun);
   GridFunction *xy = new GridFunction(fes);
   xy->MakeOwner(fec);
   xy->ProjectCoefficient(xy_coeff);
   mesh2->NewNodes(*xy, true);
   ofstream sol_ofss("/users/kaurs3/Sharan/Research/Spring_2020/quadrature_rule/test_quadrature/circle_mesh.vtk");
   sol_ofss.precision(14);
   mesh2->PrintVTK(sol_ofss, 0);
   /* mesh for circle ends here */
   // center of circle
   double xc = 5.0;
   double yc = 5.0;
   // find near-body elements to refine
   for (int k = 0; k < 0; ++k)
   {
      Array<int> nearbody_elements;
      for (int i = 0; i < mesh->GetNE(); ++i)
      {
         Vector center(dim);
         GetElementCenter(mesh, i, center);
         if ((center(0) > xc - 2) && (center(1) > yc - 2) && (center(0) < xc + 2) && (center(1) < yc + 2))
         {
            nearbody_elements.Append(i);
         }
      }
      mesh->GeneralRefinement(nearbody_elements, 1);
   }
   // get elements `cut` by circle
   cout << "elements cut by circle before refinement " << endl;
   // find boundary elements to refine
   for (int k = 0; k <1; ++k)
   {
      Array<int> marked_elements;
      for (int i = 0; i < mesh->GetNE(); ++i)
      {
         if (cutByCircle(mesh, i) == true)
         {
            cout << i << endl;
            marked_elements.Append(i);
         }
      }
      //mesh->GeneralRefinement(marked_elements, 1);
   }
   ofstream sol_ofs("/users/kaurs3/Sharan/Research/Spring_2020/quadrature_rule/test_quadrature/square_mesh.vtk");
   sol_ofs.precision(14);
   mesh->PrintVTK(sol_ofs, 0);
   // find the elements cut by boundary after refinement
   vector<int> cutelems;
   for (int i = 0; i < mesh->GetNE(); ++i)
   {
      if (cutByCircle(mesh, i) == true)
      {
         cutelems.push_back(i);
      }
   }
   //find the quadrature rule for cut elements
   std::cout << "the quadrature rule for cut elements:\n";
   {
      for (int i = 0; i < cutelems.size(); ++i)
      {
         int elemid = cutelems.at(i);
         blitz::TinyVector<double,2> xmin;
         blitz::TinyVector<double,2> xmax;
         findBoundingBox(mesh, elemid, xmin, xmax);
         circle<2> phi;
         auto q = Algoim::quadGen<2>(phi, Algoim::BoundingBox<double,2>(xmin, xmax), -1, -1, 4);
         for (const auto &pt : q.nodes)
         {
            //cout << phi(pt.x) << endl;
         }
      }
   }
   cout << "elements cut by circle after refinement " << endl;
   for (int k = 0; k < cutelems.size(); ++k)
   {
      cout << cutelems.at(k) << endl;
   }
}
// function to see if an element is cut-element
bool cutByCircle(Mesh *mesh, int &elemid)
{
   Array<Element *> elements;
   Element *el = mesh->GetElement(elemid);
   Array<int> v;
   el->GetVertices(v);
   int k, l;
   k = 0;
   l = 0;
   for (int i = 0; i < v.Size(); ++i)
   {
      double *coord = mesh->GetVertex(v[i]);
      Vector lvsval(v.Size());
      //cout << x[1] << endl;
      lvsval(i) = ((coord[0] - 5) * (coord[0] - 5)) + ((coord[1] - 5) * (coord[1] - 5)) - (0.5 * 0.5);
      if ((lvsval(i) < 0))
      {
         k = k + 1;
      }
      if ((lvsval(i) > 0))
      {
         l = l + 1;
      }
   }
   if ((k == v.Size()) || (l == v.Size()))
   {
      return false;
   }
   else
   {
      return true;
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
         if (coord[d] < min(d)) { min(d) = coord[d]; }
         if (coord[d] > max(d)) { max(d) = coord[d]; }
      }
   }
   cout << min[0] << " , " << max[0] << endl;   
   xmin={5, 5};
   xmax={5.5, 5.5};
}