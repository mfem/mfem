#include "dg_pa.hpp"

namespace mfem
{
namespace dg
{

void PartialAssembly::FormFaceEvaluation(const FiniteElement *fe,
                                         IntegrationPointTransformation *loc,
                                         DenseMatrix &Bface)
{
   const int nquad = ir_face.Size();
   const int ndof = fe->GetDof();
   Vector shape(ndof);
   Bface.SetSize(nquad, ndof);
   for (int iq = 0; iq < nquad; ++iq)
   {
      IntegrationPoint ip;
      loc->Transform(ir_face.IntPoint(iq), ip);
      fe->CalcShape(ip, shape);
      for (int i = 0; i < ndof; ++i)
      {
         Bface(iq,i) = shape(i);
      }
   }
}

PartialAssembly::PartialAssembly(const FiniteElementSpace *fes_)
   : fes(fes_)
{
   int order = 3*fes->GetFE(0)->GetOrder();
   ir = IntRules.Get(fes->GetFE(0)->GetGeomType(), order);
   ir_face = IntRules.Get(fes->GetMesh()->GetFaceGeometryType(0), order);
   metric.Precompute(fes, &ir, &ir_face);

   // Precompute basis evaluation matrix B and basis derivative matrix G
   const int dim = fes->GetFE(0)->GetDim();
   const int nquad = ir.Size();
   const int ndof = fes->GetFE(0)->GetDof();
   B.SetSize(nquad, ndof);
   G.SetSize(nquad, ndof, dim);

   const FiniteElement *fe = fes->GetFE(0);
   Vector shape(ndof);
   DenseMatrix dshape(ndof, dim);
   for (int iq = 0; iq < nquad; ++iq)
   {
      const IntegrationPoint &ip = ir.IntPoint(iq);
      fe->CalcShape(ip, shape);
      fe->CalcDShape(ip, dshape);
      for (int i = 0; i < ndof; ++i)
      {
         B(iq,i) = shape(i);
         for (int d = 0; d < dim; ++ d)
         {
            G(iq, i, d) = dshape(i, d);
         }
      }
   }

   // Precompute face evaluation matrices
   Mesh *mesh = fes->GetMesh();
   const int nfaces = mesh->GetNumFaces();
   for (int i = 0; i < nfaces; ++i)
   {
      DenseMatrix Bface1, Bface2;
      F2E i1, i2;

      FaceElementTransformations *tr = mesh->GetFaceElementTransformations(i);
      GetF2Es(i, i1, i2);

      if (Bfaces.find(i1) == Bfaces.end())
      {
         FormFaceEvaluation(fes->GetFE(tr->Elem1No), &tr->Loc1, Bface1);
         Bfaces[i1] = Bface1;
      }

      if (tr->Elem2No >= 0)
      {
         if (Bfaces.find(i2) == Bfaces.end())
         {
            FormFaceEvaluation(fes->GetFE(tr->Elem2No), &tr->Loc2, Bface2);
            Bfaces[i2] = Bface2;
         }
      }
   }
}

int PartialAssembly::NQuad(int iel) const
{
   return ir.Size();
}

const FiniteElementSpace* PartialAssembly::GetFES() const
{
   return fes;
}

void PartialAssembly::GetF2Es(int fid, F2E &i1, F2E &i2) const
{
   Mesh *mesh = fes->GetMesh();
   int iel1, iel2;
   mesh->GetFaceElements(fid, &iel1, &iel2);
   i1.elem_type = mesh->GetElementType(iel1);
   i1.face_type = mesh->GetFaceElementType(fid);
   mesh->GetFaceInfos(fid, &i1.info, &i2.info);
   if (iel2 >= 0)
   {
      i2.elem_type = mesh->GetElementType(iel2);
      i2.face_type = i1.face_type;
   }
   else
   {
      // Invalid
      i2.info = -1;
      i2.elem_type = -1;
      i2.face_type = -1;
   }
}

const DenseMatrix& PartialAssembly::BasisEval(int iel) const
{
   return B;
}
const DenseTensor& PartialAssembly::DerivEval(int iel) const
{
   return G;
}

const DenseMatrix& PartialAssembly::FaceEval(const F2E &i) const
{
   return Bfaces.at(i);
}

} // namespace dg
} // namespace mfem