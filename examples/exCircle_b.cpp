#include "exCircle.hpp"
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

   Mesh *mesh2 = new Mesh(10, 10, Element::QUADRILATERAL, true,
                         10, 10, true);
   int dim = mesh2->Dimension();
   cout << dim << endl;
   std::cout << "Number of elements " << mesh2->GetNE() << '\n';

   // std::cout<< mesh->GetBdrElement(0)->GetNVertices() << endl;
   // std::cout << "Number of boundary elements " << mesh->GetNBE() << endl;
  // std::cout <<"boundary element id " << mesh->boundary[0] << endl;
   std::cout << "Attributes for all elements " << endl;
   for (int i = 0; i < mesh2->GetNE(); ++i)
   {
      cout << mesh2->GetElement(i)->GetAttribute()<< endl;
   }
   cout << "element type is " << endl;                                                 
   cout << mesh2->GetElement(1)->GetGeometryType() << endl;
   mesh2->GetElement(21)->SetAttribute(5);
   cout << "element attribute is " << endl;
   cout << mesh2->GetElement(21)->GetAttribute()<< endl;
   // Geometry::Type CUTSQUARE;
   // Element *el;
   // //el[21] = mesh->NewElement(7);
   // mesh->GetElement(21)->SetGeometryType(CUTSQUARE);
   cout << "element geometry type now is " << endl;                                                 
   cout << mesh2->GetElement(21)->GetGeometryType() << endl;
   cout << "element type now is " << endl;                                                 
   cout << mesh2->GetElement(21)->GetType() << endl;
   Array<int> v1;
   mesh2->GetElement(21)->GetVertices(v1);
   cout << mesh2->GetElement(21)->GetNVertices() << endl;
   cout << v1[0] << ", " << v1[1] << ", " <<v1[2] << ", " << v1[3] <<endl;
   ofstream sol_ofs("/users/kaurs3/Sharan/Research/Spring_2020/quadrature_rule/test_quadrature/square_mesh.mesh");
   sol_ofs.precision(14);
   mesh2->Print(sol_ofs);
   const char *mesh_file = "/users/kaurs3/Sharan/Research/Spring_2020/quadrature_rule/test_quadrature/square_mesh.mesh";
   FiniteElementCollection *fec;
   fec = new H1_FECollection(1, dim);
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
   
   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm *b = new LinearForm(fespace);
   ConstantCoefficient one(1.0);
   std::map<int, IntegrationRule *> CutSquareIntRules;
   IntegrationRule *ir;
   blitz::TinyVector<double,2> xmin;
   blitz::TinyVector<double,2> xmax;
   findBoundingBox(mesh,21, xmin, xmax);
   auto q = Algoim::quadGen<2>(phi, Algoim::BoundingBox<double,2>(xmin, xmax), 2, -1, 2);
   cout << "number of quadrature nodes: " << q.nodes.size() << endl;
   for (const auto &pt : q.nodes)
    {
       // cout << pt.x << endl;
       // cout << pt.w <<endl;
       // cout << phi(pt.x) << endl;
    }
   // b->AddDomainIntegrator(new CutDomainIntegrator(one, CutSquareIntRules));
   // b->Assemble();
   cout << "element geometry type for cut element is " << endl;                                                 
   cout << mesh->GetElement(21)->GetGeometryType() << endl;
   cout << "check GetGeomType() for cut element  " << endl;                                                 
   //cout << mesh->GetElement(21)->GetGeomType() << endl;
   cout << "element type for cut element is " << endl;                                                 
   cout << mesh->GetElement(21)->GetType() << endl;
   Array<int> v;
   mesh->GetElement(21)->GetVertices(v);
   cout << v[0] << ", " << v[1] << ", " << v[2]  << ", " << v[3] << endl;
   ofstream sol_ofv("/users/kaurs3/Sharan/Research/Spring_2020/quadrature_rule/test_quadrature/square_mesh.vtk");
   sol_ofv.precision(14);
   mesh->PrintVTK(sol_ofv,2);
   Array<int> nearbody_elements;
   nearbody_elements.Append(21);
   mesh->GeneralRefinement(nearbody_elements, 1);
   ofstream sol_ofr("/users/kaurs3/Sharan/Research/Spring_2020/quadrature_rule/test_quadrature/square_mesh_ref.mesh");
   sol_ofr.precision(14);
   mesh->Print(sol_ofr);
   //Array<IntegrationRule *> CutSquareIntRules;

   
   //CutSquareIntRules[3] = 
    // center of circle
   // double xc = 5.0;
   // double yc = 5.0;
   // // find near-body elements to refine
   // for (int k = 0; k < 0; ++k)
   // {
   //    Array<int> nearbody_elements;
   //    for (int i = 0; i < mesh->GetNE(); ++i)
   //    {
   //       Vector center(dim);
   //       GetElementCenter(mesh, i, center);
   //       if ((center(0) > xc - 2) && (center(1) > yc - 2) && (center(0) < xc + 2) && (center(1) < yc + 2))
   //       {
   //          nearbody_elements.Append(i);
   //       }
   //    }
   //    mesh->GeneralRefinement(nearbody_elements, 1);
   // }
   // get elements `cut` by circle
  // cout << "elements cut by circle before refinement " << endl;
   // find boundary elements to refine
   // for (int k = 0; k <1; ++k)
   // {
   //    Array<int> marked_elements;
   //    for (int i = 0; i < mesh->GetNE(); ++i)
   //    {
   //       if (cutByCircle(mesh, i) == true)
   //       {
   //          //cout << i << endl;
   //          marked_elements.Append(i);
   //       }
   //    }
   //    //mesh->GeneralRefinement(marked_elements, 1);
   // }
   // ofstream sol_ofs("/users/kaurs3/Sharan/Research/Spring_2020/quadrature_rule/test_quadrature/square_mesh.vtk");
   // sol_ofs.precision(14);
   // mesh->PrintVTK(sol_ofs, 0);
   // find the elements cut by boundary after refinement
   // vector<int> cutelems;
   // for (int i = 0; i < mesh->GetNE(); ++i)
   // {
   //    if (cutByCircle(mesh, i) == true)
   //    {
   //       cutelems.push_back(i);
   //    }
   // }
   //find the quadrature rule for cut elements
  // std::cout << "under the quadrature rule for cut elements:\n";
   // {
   //    for (int i = 0; i < cutelems.size(); ++i)
   //    {
   //       int elemid = cutelems.at(i);
   //       blitz::TinyVector<double,2> xmin;
   //       blitz::TinyVector<double,2> xmax;
   //       findBoundingBox(mesh, elemid, xmin, xmax);
   //       circle<2> phi;
   //     //  cout << "----------------------- " << endl;
   //      // cout << "cut element id is " << elemid << endl;
   //      // cout << "Bounding box: " << xmin << " , " << xmax << endl;
   //       auto q = Algoim::quadGen<2>(phi, Algoim::BoundingBox<double,2>(xmin, xmax), 2, -1, 2);
   //       // cout << "number of quadrature nodes: " << q.nodes.size() << endl;
   //       // cout << "Node coordinates and weight: " << endl;
   //       for (const auto &pt : q.nodes)
   //       {
   //          // cout << pt.x << endl;
   //          // cout << pt.w <<endl;
   //          // cout << phi(pt.x) << endl;
   //       }
   //    }
   // }
  // cout << "elements cut by circle after refinement " << endl;
   // for (int k = 0; k < cutelems.size(); ++k)
   // {
   //   // cout << cutelems.at(k) << endl;
   // }
}
void CutDomainIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                                ElementTransformation &Tr,
                                                Vector &elvect)
{
   int dof = el.GetDof();

   shape.SetSize(dof);       // vector of size dof
   elvect.SetSize(dof);
   elvect = 0.0;
   //cout << Tr.ElementNo << endl;
   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // ir = &IntRules.Get(el.GetGeomType(),
      //                    oa * el.GetOrder() + ob + Tr.OrderW());
      ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint (&ip);
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
   //cout << min[0] << " , " << max[0] << endl;   
   xmin={min[0], min[1]};
   xmax={max[0], max[1]};
}
