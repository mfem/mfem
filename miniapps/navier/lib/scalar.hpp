#ifndef MFEM_NAVIER_SCALAR_HPP
#define MFEM_NAVIER_SCALAR_HPP

#include <mfem.hpp>
#include <vector>

namespace mfem
{
namespace navier
{

class ScalarEquation : public TimeDependentOperator
{
public:
   ScalarEquation(ParMesh &mesh, const int order,
                  const ParGridFunction &velgf);

   void Mult(const Vector &x, Vector &y) const override;

   void SetTime(double t) override;

   void Setup();

   void Reassemble() const;

   void AddReaction(Coefficient &c);

   void SetFixedValue(Coefficient &c, std::vector<int> bdr_attributes);

   void AddForcing(Coefficient &c);

   void SetViscosityCoefficient(Coefficient &c)
   {
      q_coeff = &c;
   };

   ParGridFunction &GetScalar() { return k_gf; };

protected:
   ParMesh &mesh;
   const int order;
   const DG_FECollection fec;
   ParFiniteElementSpace fes;
   const IntegrationRule *ir = nullptr, *ir_face = nullptr; /// not owned
   IntegrationRules *gll_rules = nullptr;

   ParBilinearForm mform;
   mutable ParBilinearForm kform;
   ParLinearForm bform;
   mutable ParLinearForm rform;

   mutable ParGridFunction k_gf;

   ConstantCoefficient zero_coeff;
   ConstantCoefficient one_coeff;
   VectorGridFunctionCoefficient vel_coeff;
   Coefficient *q_coeff = nullptr;
   TransformedCoefficient *neg_q_coeff = nullptr;

   mutable OperatorHandle K;

   Vector m, minv;

   Vector B;
   mutable Vector z, R;

   const double alpha = -1.0, beta = -0.5;
   double sigma = -1.0, kappa;
   const int skip_zeros = 0;
};

} // namespace navier

} // namespace mfem

#endif