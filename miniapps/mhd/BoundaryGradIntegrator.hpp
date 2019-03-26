#include "mfem.hpp"

using namespace std;
using namespace mfem;

// Integrator for the boundary gradient integral from the Laplacian operator
class BoundaryGradIntegrator: public BilinearFormIntegrator
{
private:
   Vector shape1, dshape_dn, nor;
   DenseMatrix dshape, dshapedxt, invdfdx;

public:
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};

void BoundaryGradIntegrator::AssembleFaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   int i, j, ndof1;
   int dim, order;

   dim = el1.GetDim();
   ndof1 = el1.GetDof();

   // set to this for now, integration includes rational terms
   order = 2*el1.GetOrder() + 1;

   nor.SetSize(dim);
   shape1.SetSize(ndof1);
   dshape_dn.SetSize(ndof1);
   dshape.SetSize(ndof1,dim);
   dshapedxt.SetSize(ndof1,dim);
   invdfdx.SetSize(dim);

   elmat.SetSize(ndof1);
   elmat = 0.0;

   const IntegrationRule *ir = &IntRules.Get(Trans.FaceGeom, order);
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip1;
      Trans.Loc1.Transform(ip, eip1);
      el1.CalcShape(eip1, shape1);
      //d of shape function, evaluated at eip1
      el1.CalcDShape(eip1, dshape);

      Trans.Elem1->SetIntPoint(&eip1);

      CalcInverse(Trans.Elem1->Jacobian(), invdfdx); //inverse Jacobian
      //invdfdx.Transpose();
      Mult(dshape, invdfdx, dshapedxt); // dshapedxt = grad phi* J^-1

      //get normal vector
      Trans.Face->SetIntPoint(&ip);
      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Face->Jacobian(), nor);
      }

      /* this is probably the old way
      const DenseMatrix &J = Trans.Face->Jacobian(); //is this J^{-1} or J^{T}?
      else if (dim == 2)
      {
         nor(0) =  J(1,0);
         nor(1) = -J(0,0);
      }
      else if (dim == 3)
      {
         nor(0) = J(1,0)*J(2,1) - J(2,0)*J(1,1);
         nor(1) = J(2,0)*J(0,1) - J(0,0)*J(2,1);
         nor(2) = J(0,0)*J(1,1) - J(1,0)*J(0,1);
      }
      */

      // multiply weight into normal, make answer negative
      // (boundary integral is subtracted)
      nor *= -ip.weight;

      dshapedxt.Mult(nor, dshape_dn);

      for (i = 0; i < ndof1; i++)
         for (j = 0; j < ndof1; j++)
         {
            elmat(i, j) += shape1(i)*dshape_dn(j);
         }
   }
}

