#include "dg_metric.hpp"

namespace mfem
{
namespace dg
{

MetricTerms::MetricTerms()
{
   nel = 0;
   nquad = 0;
   nfaces = 0;
   dim = 0;
   nquad_face = 0;
}

void MetricTerms::Precompute(const FiniteElementSpace *fes,
                             const IntegrationRule *ir,
                             const IntegrationRule *ir_face)
{
   Mesh *mesh = fes->GetMesh();

   nquad = ir->Size();
   nquad_face = ir_face->Size();
   nel = fes->GetNE();
   dim = fes->GetFE(0)->GetDim();
   nfaces = mesh->GetNumFaces();

   w.SetSize(nquad);
   for (int iq = 0; iq < nquad; ++iq)
   {
      w(iq) = ir->IntPoint(iq).weight;
   }

   detJ.SetSize(nel*nquad);
   Jinv.SetSize(dim, dim, nel*nquad);

   for (int i = 0; i < nel; ++i)
   {
      ElementTransformation *tr = fes->GetElementTransformation(i);
      for (int iq = 0; iq < nquad; ++iq)
      {
         int idx = i*nquad + iq;
         const IntegrationPoint &ip = ir->IntPoint(iq);
         tr->SetIntPoint(&ip);
         detJ(idx) = tr->Weight();
         Jinv(idx) = tr->InverseJacobian();
      }
   }

   wface.SetSize(nquad_face);
   for (int iq = 0; iq < nquad_face; ++iq)
   {
      wface(iq) = ir_face->IntPoint(iq).weight;
   }

   nvec.SetSize(dim, nfaces*nquad_face);
   for (int i = 0; i < nfaces; ++i)
   {
      FaceElementTransformations *tr = mesh->GetFaceElementTransformations(i);
      for (int iq = 0; iq < nquad_face; ++iq)
      {
         int idx = i*nquad_face + iq;
         const IntegrationPoint &ip = ir_face->IntPoint(iq);
         tr->Face->SetIntPoint(&ip);
         if (dim == 1)
         {
            IntegrationPoint eip1;
            tr->Loc1.Transform(ip, eip1);
            nvec(0,idx) = 2*eip1.x - 1.0;
         }
         else
         {
            Vector normal;
            nvec.GetColumnReference(idx, normal);
            CalcOrtho(tr->Face->Jacobian(), normal);
         }
      }
   }
}

double MetricTerms::JacobianDeterminant(int elid, int iq) const
{
   return detJ(elid*nquad + iq);
}

double MetricTerms::Weight(int elid, int iq) const
{
   return w(iq)*JacobianDeterminant(elid, iq);
}

const DenseMatrix& MetricTerms::InverseJacobian(int elid, int iq) const
{
   return Jinv(elid*nquad + iq);
}

const Vector& MetricTerms::Normal(int fid, int iq) const
{
   nvec.GetColumnReference(fid*nquad_face + iq, n);
   return n;
}

double MetricTerms::FaceWeight(int fid, int iq) const
{
   return wface(iq);
}

} // namespace dg
} // namespace mfem