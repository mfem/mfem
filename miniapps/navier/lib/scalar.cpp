#include "scalar.hpp"
#include "general/forall.hpp"

namespace mfem
{
namespace navier
{

static inline void vmul(const Vector &x, Vector &y)
{
   auto d_x = x.Read();
   auto d_y = y.ReadWrite();
   MFEM_FORALL(i, x.Size(), d_y[i] = d_x[i] * d_y[i];);
}

ScalarEquation::ScalarEquation(ParMesh &mesh, const int order,
                               const ParGridFunction &velgf) :
   mesh(mesh),
   order(order),
   fec(order, mesh.Dimension(), BasisType::GaussLobatto),
   fes(&mesh, &fec),
   mform(&fes),
   kform(&fes),
   bform(&fes),
   rform(&fes),
   k_gf(&fes),
   zero_coeff(0.0),
   one_coeff(1.0),
   vel_coeff(&velgf),
   kappa((order+1)*(order+1))
{
   const int truevszize = fes.GetTrueVSize();
   height = truevszize;
   width = truevszize;

   z.SetSize(truevszize);
   B.SetSize(truevszize);
   R.SetSize(truevszize);
   m.SetSize(truevszize);
   minv.SetSize(truevszize);

   gll_rules = new IntegrationRules(0, Quadrature1D::GaussLobatto);
   ir = &gll_rules->Get(fes.GetFE(0)->GetGeomType(),
                        2 * fes.FEColl()->GetOrder() - 1);

   ir_face = &gll_rules->Get(fes.GetMesh()->GetFaceGeometry(0),
                             2 * fes.FEColl()->GetOrder() - 1);
}

void ScalarEquation::Setup()
{
   MFEM_ASSERT(q_coeff,
               "missing ScalarEquation::SetViscosityCoefficient");

   mform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   kform.SetAssemblyLevel(AssemblyLevel::PARTIAL);

   bform.UseFastAssembly(true);
   rform.UseFastAssembly(true);

   auto mass_integ = new MassIntegrator;
   mass_integ->SetIntegrationRule(*ir);
   mform.AddDomainIntegrator(mass_integ);

   mform.Assemble(skip_zeros);
   // Build mass diagonal vector
   mform.AssembleDiagonal(m);

   // Inverse mass is the inverse of the diagonal
   {
      const auto d_m = m.Read();
      auto d_minv = minv.Write();
      MFEM_FORALL(i, m.Size(), d_minv[i] = 1.0 / d_m[i];);
   }

   // Advection
   auto convection_integ = new ConservativeConvectionIntegrator(vel_coeff,
                                                                alpha);
   convection_integ->SetIntegrationRule(*ir);
   kform.AddDomainIntegrator(convection_integ);

   auto interior_face_integ = new DGTraceIntegrator(vel_coeff, alpha, beta);
   interior_face_integ->SetIntegrationRule(*ir_face);
   kform.AddInteriorFaceIntegrator(interior_face_integ);

   auto bdr_face_integ = new DGTraceIntegrator(vel_coeff, alpha, beta);
   bdr_face_integ->SetIntegrationRule(*ir_face);
   kform.AddBdrFaceIntegrator(bdr_face_integ);

   // Diffusion
   // Sign is flipped to conform with ODE formulation
   // neg_q_coeff = new TransformedCoefficient(q_coeff, [](double q) { return -q; });
   // auto diffusion_integ = new DiffusionIntegrator(*neg_q_coeff);
   // diffusion_integ->SetIntegrationRule(*ir);
   // kform.AddDomainIntegrator(diffusion_integ);

   // auto dgdiffusion_interior_face_integ = new DGDiffusionIntegrator(*neg_q_coeff,
   //                                                                  sigma, kappa);
   // dgdiffusion_interior_face_integ->SetIntegrationRule(*ir_face);
   // kform.AddInteriorFaceIntegrator(dgdiffusion_interior_face_integ);

   // auto dgdiffusion_bdr_face_integ = new DGDiffusionIntegrator(*neg_q_coeff,
   //                                                             sigma, kappa);
   // dgdiffusion_bdr_face_integ->SetIntegrationRule(*ir_face);
   // kform.AddBdrFaceIntegrator(dgdiffusion_bdr_face_integ);

   kform.KeepNbrBlock(true);
   k_gf.ExchangeFaceNbrData();
   kform.Assemble(skip_zeros);
   kform.Finalize(skip_zeros);
   // K.Reset(kform.ParallelAssemble(), true);
   Array<int> empty;
   kform.FormSystemMatrix(empty, K);

   bform.Assemble();
   bform.ParallelAssemble(B);
}

void ScalarEquation::SetFixedValue(Coefficient &c,
                                   std::vector<int> bdr_attributes)
{
   MFEM_ASSERT(q_coeff,
               "missing ScalarEquation::SetViscosityCoefficient");

   Array<int> *attr = new Array<int>(mesh.bdr_attributes.Size());
   *attr = 0;
   for (int i = 0; i < bdr_attributes.size(); i++)
   {
      (*attr)[bdr_attributes[i]] = 1;
   }

   if (Mpi::Root())
   {
      printf("ScalarEquation::SetFixedValue attribute marker ");
      for (int i = 0; i < attr->Size(); i++)
      {
         printf("%d ", (*attr)[i]);
      }
      printf("\n");
   }

   // Advection
   auto integ1 = new BoundaryFlowIntegrator(c, vel_coeff, alpha, beta);
   integ1->SetIntRule(ir_face);
   bform.AddBdrFaceIntegrator(integ1, *attr);

   // Diffusion
   // auto integ2 = new DGDirichletLFIntegrator(c, *q_coeff, sigma, kappa);
   // integ2->SetIntRule(ir_face);
   // bform.AddBdrFaceIntegrator(integ2, *attr);
}

void ScalarEquation::AddForcing(Coefficient &c)
{
   auto integ = new DomainLFIntegrator(c);
   integ->SetIntRule(ir);
   bform.AddDomainIntegrator(integ);
}

void ScalarEquation::AddReaction(Coefficient &c)
{
   auto integ = new DomainLFIntegrator(c);
   integ->SetIntRule(ir);
   rform.AddDomainIntegrator(integ);
}

void ScalarEquation::SetTime(double t)
{
   TimeDependentOperator::SetTime(t);
}

void ScalarEquation::Reassemble() const
{
   k_gf.ExchangeFaceNbrData();
   kform.Update();
   kform.BilinearForm::operator=(0.0);
   kform.Assemble(skip_zeros);
   // K.Reset(kform.ParallelAssemble(), true);
   Array<int> empty;
   kform.FormSystemMatrix(empty, K);

   rform.Assemble();
   rform.ParallelAssemble(R);
}

void ScalarEquation::Mult(const Vector &x, Vector &y) const
{
   k_gf.SetFromTrueDofs(x);
   Reassemble();

   K->Mult(x, z);
   z += B;
   z += R;

   vmul(minv, z);
   y = z;
}

}
}