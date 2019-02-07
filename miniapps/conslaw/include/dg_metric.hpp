#ifndef DGPA_METRIC
#define DGPA_METRIC

#include "mfem.hpp"

namespace mfem
{
namespace dg
{

class MetricTerms
{
   int nel, nquad, dim, nfaces, nquad_face;
   Vector detJ;
   Vector w, wface;
   DenseTensor Jinv;
   mutable DenseMatrix nvec;
   mutable Vector n;
public:
   MetricTerms();
   void Precompute(const FiniteElementSpace *fes,
                   const IntegrationRule *ir,
                   const IntegrationRule *ir_face);
   // Interface to access metric terms at quadrature points:
   double JacobianDeterminant(int elid, int iq) const;
   double Weight(int elid, int iq) const;
   const DenseMatrix& InverseJacobian(int elid, int iq) const;
   const Vector& Normal(int fid, int iq) const;
   double FaceWeight(int fid, int iq) const;
};


} // namespace dg
} // namespace mfem

#endif