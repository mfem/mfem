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
      return -1 * ((((x[0] * xscale) + xmin - 5) * ((x[0] * xscale) + xmin - 5)) +
                   (((x[1] * yscale) + ymin - 5) * ((x[1] * yscale) + ymin - 5)) - (1));
   }
   template <typename T>
   blitz::TinyVector<T, N> grad(const blitz::TinyVector<T, N> &x) const
   {
      // return blitz::TinyVector<T, N>(-1 * (2.0 * (x(0) - 5)), -1 * (2.0 * (x(1) - 5)));
      return blitz::TinyVector<T, N>(-1 * (2.0 * xscale * ((x(0) * xscale) + xmin - 5)),
                                     -1 * (2.0 * yscale * ((x(1) * yscale) + ymin - 5)));
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
   double sigma = -1.0;
   double kappa = 1;
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

   Mesh *mesh = new Mesh(20, 20, Element::QUADRILATERAL, true,
                         10, 10, true);
   ofstream sol_ofv("square_mesh.vtk");
   sol_ofv.precision(14);
   mesh->PrintVTK(sol_ofv, 0);
   std::map<int, IntegrationRule *> CutSquareIntRules;
   std::map<int, IntegrationRule *> CutSegmentIntRules;
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
   // 4. Define a finite element space on the mesh. Here we use discontinuous
   //    finite elements of the specified order >= 0.
   FiniteElementCollection *fec = new DG_FECollection(1, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "Number of unknowns: " << fespace->GetVSize() << endl;
   cout << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;
   LinearForm *b = new LinearForm(fespace);
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   //b->AddDomainIntegrator(new CutDomainLFIntegrator(one));
   b->AddBdrFaceIntegrator(
       new DGDirichletLFIntegrator(zero, one, sigma, kappa));
   b->Assemble();
   // define map for integration rule for cut elements
   GetCutElementIntRule<2>(mesh, cutelems, CutSquareIntRules);
   GetCutSegmentIntRule<2>(mesh, cutelems, CutSegmentIntRules);
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
   GridFunction x(fespace);
   x = 0.0;
   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new CutDiffusionIntegrator(one, CutSquareIntRules, EmbeddedElems));
   a->AddInteriorFaceIntegrator(new CutDGDiffusionIntegrator(one, sigma, kappa,
                                                             CutSegmentIntRules, EmbeddedElems));
   a->Assemble();
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
         // cout << pt.x[0] << " , " << pt.x[1] << endl;
         // cout << "mapped back " << endl;
         //cout << (pt.x[0] * phi.xscale) + phi.xmin << " , " << (pt.x[1] * phi.yscale) + phi.ymin  << endl;
      }
      //  cout << "element id for cut element " << elemid << endl;
      CutSquareIntRules[elemid] = ir;
   }
}

template <int N>
void GetCutSegmentIntRule(Mesh *mesh, vector<int> cutelems,
                          std::map<int, IntegrationRule *> &CutSegmentIntRules)
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
      for (const auto &pt : q.nodes)
      {
         IntegrationPoint &ip = ir->IntPoint(i);
         ip.x = pt.x[0];
         ip.y = pt.x[1];
         ip.weight = pt.w;
         i = i + 1;
         // cout << pt.x[0] << " , " << pt.x[1] << endl;
         // cout << "mapped back " << endl;
         //cout << (pt.x[0] * phi.xscale) + phi.xmin << " , " << (pt.x[1] * phi.yscale) + phi.ymin  << endl;
      }
      //  cout << "element id for cut element " << elemid << endl;
      CutSegmentIntRules[elemid] = ir;
   }
}

bool insideBoundary(Mesh *mesh, int &elemid)
{
   Element *el = mesh->GetElement(elemid);
   Array<int> v;
   el->GetVertices(v);
   int k;
   k = 0;
   for (int i = 0; i < v.Size(); ++i)
   {
      double *coord = mesh->GetVertex(v[i]);
      Vector lvsval(v.Size());
      lvsval(i) = ((coord[0] - 5) * (coord[0] - 5)) + ((coord[1] - 5) * (coord[1] - 5)) - (1.0);
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
   for (int i = 0; i < v.Size(); ++i)
   {
      double *coord = mesh->GetVertex(v[i]);
      Vector lvsval(v.Size());
      //cout << x[1] << endl;
      lvsval(i) = ((coord[0] - 5.0) * (coord[0] - 5.0)) + ((coord[1] - 5.0) * (coord[1] - 5.0)) - (1.0);
      if ((lvsval(i) < 0))
      {
         k = k + 1;
      }
      if ((lvsval(i) > 0))
      {
         l = l + 1;
      }
      if ((lvsval(i) == 0))
      {
         n = n + 1;
      }
   }
   if ((k == v.Size()) || (l == v.Size()))
   {
      return false;
   }
   if ((k == 3) || (l == 3) && (n == 1))
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
   if ((Trans.Elem2No >= 0))
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
   if (kappa_is_nonzero)
   {
      jmat.SetSize(ndofs);
      jmat = 0.;
   }
   const IntegrationRule *ir;
   if ((EmbeddedElements.at(Trans.Elem1No) == true) && (EmbeddedElements.at(Trans.Elem2No) == false))
   {
       ir = CutSegIntRules[Trans.Elem2No];
   }
   else if ((EmbeddedElements.at(Trans.Elem1No) == false) && (EmbeddedElements.at(Trans.Elem2No) == true))
   {
       ir = CutSegIntRules[Trans.Elem1No];
   }
   else
   {
      ir=NULL;
   }
   if (ir !=NULL)
   {
      cout << "element 1 and 2 " << endl;
      cout << Trans.Elem1No << " , " << Trans.Elem2No << endl;
   }
   if (ir == NULL)
   {
      if (IntRule)
      {
         ir = IntRule;
      }
      else
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
   }
   // assemble: < {(Q \nabla u).n},[v] >      --> elmat
   //           kappa < {h^{-1} Q} [u],[v] >  --> jmat
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
   //cout << "flag for  " <<  endl;
}