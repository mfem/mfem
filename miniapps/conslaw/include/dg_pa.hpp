#ifndef DGPA_PA
#define DGPA_PA

#include "mfem.hpp"
#include "dg_metric.hpp"
#include <unordered_map>

namespace mfem
{
namespace dg
{

class PartialAssembly
{
public:
   struct F2E
   {
      int face_type;
      int elem_type;
      int info;
      operator bool() const { return face_type != -1; }
   };
private:
   struct F2EHash
   {
      std::size_t operator()(const F2E &i) const
      {
         static std::hash<int> h;
         return h(i.face_type)^h(i.elem_type)^h(i.info);
      }
   };
   struct F2EEq
   {
      bool operator()(const F2E &i1, const F2E &i2) const
      {
         return (i1.face_type == i2.face_type) && (i1.elem_type == i2.elem_type)
                && (i1.info == i2.info);
      }
   };


   const FiniteElementSpace *fes;
   IntegrationRule ir, ir_face;
   MetricTerms metric;
   DenseMatrix B;
   DenseTensor G;
   std::unordered_map<F2E,DenseMatrix,F2EHash,F2EEq> Bfaces;

   void FormFaceEvaluation(const FiniteElement *fe,
                           IntegrationPointTransformation *loc,
                           DenseMatrix &Bface);
public:
   PartialAssembly(const FiniteElementSpace *fes_);
   const FiniteElementSpace *GetFES() const;

   int NQuad(int iel) const;

   // Partial assembly operators
   const DenseMatrix& BasisEval(int iel) const;
   const DenseTensor& DerivEval(int iel) const;
   const DenseMatrix& FaceEval(const F2E &i) const;

   // Face info
   void GetF2Es(int fid, F2E &i1, F2E &i2) const;

   // Metric terms
   const MetricTerms& GetMetricTerms() const { return metric; }
};


} // namespace dg
} // namespace mfem

#endif