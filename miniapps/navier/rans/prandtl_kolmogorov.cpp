#include <memory>
#include "prandtl_kolmogorov.hpp"
#include "prandtl_kolmogorov_kernels.hpp"

namespace mfem
{
namespace navier
{

PrandtlKolmogorov::PrandtlKolmogorov(ParFiniteElementSpace &kfes,
                                     VectorCoefficient &vel_coeff,
                                     Coefficient &kv_coeff,
                                     Coefficient &f_coeff,
                                     Coefficient &tls_coeff,
                                     Coefficient &k_bdrcoeff,
                                     const double mu_calibration_const,
                                     Array<int> eattr) :
   kfes(kfes),
   dim(kfes.GetParMesh()->Dimension()),
   ne(kfes.GetNE()),
   vel_coeff(vel_coeff),
   kv_coeff(kv_coeff),
   f_coeff(f_coeff),
   tls_coeff(tls_coeff),
   k_bdrcoeff(k_bdrcoeff),
   mu_calibration_const(mu_calibration_const)
{
   height = kfes.GetTrueVSize();
   width = height;

   ess_attr = eattr;
   if (ess_attr.Size())
   {
      kfes.GetEssentialTrueDofs(ess_attr, ess_tdof_list);
   }

   ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;

   gll_rules = new IntegrationRules(0, Quadrature1D::GaussLobatto);
   ir = &gll_rules->Get(kfes.GetFE(0)->GetGeomType(),
                        2 * kfes.FEColl()->GetOrder() - 1);

   geom = kfes.GetParMesh()->GetGeometricFactors(
             *ir, GeometricFactors::JACOBIANS | GeometricFactors::DETERMINANTS);
   maps = &kfes.GetFE(0)->GetDofToQuad(*ir, DofToQuad::TENSOR);

   Pk = kfes.GetProlongationMatrix();
   Rk = kfes.GetElementRestriction(ordering);

   kv_q.SetSize(ir->GetNPoints() * ne);
   u_q.SetSize(ir->GetNPoints() * ne * dim);
   f_q.SetSize(ir->GetNPoints() * ne);
   tls_q.SetSize(ir->GetNPoints() * ne);
   k_l.SetSize(Pk->Height());
   k_e.SetSize(Rk->Height());
   y_l.SetSize(Pk->Height());
   y_e.SetSize(Rk->Height());
   z.SetSize(height);
   z1.SetSize(height);
   z2.SetSize(height);
   x_ess.SetSize(height);
   k_bdr_values.SetSize(height);

   k_gf.SetSpace(&kfes);
   b.SetSize(height);

   if (auto *c = dynamic_cast<VectorGridFunctionCoefficient *>(&vel_coeff))
   {
      // This class can't be used with serial objects, therefore no dynamic
      // casts.
      auto vel_pfes = static_cast<const ParGridFunction *>
                      (c->GetGridFunction())->ParFESpace();
      Pu = vel_pfes->GetProlongationMatrix();
      Ru = vel_pfes->GetElementRestriction(ordering);

      qi_vel = vel_pfes->GetQuadratureInterpolator(*ir);
      qi_vel->SetOutputLayout(QVectorLayout::byNODES);

      u_e.SetSize(Ru->Height());
   }
   else if (auto *c = dynamic_cast<VectorFunctionCoefficient *>(&vel_coeff))
   {
      // No setup needed.
   }
   else
   {
      MFEM_ABORT("Coefficient type not supported");
   }

   if (auto *c = dynamic_cast<GridFunctionCoefficient *>(&kv_coeff))
   {
      auto kv_pfes = static_cast<const ParGridFunction *>
                     (c->GetGridFunction())->ParFESpace();
      qi_kv = kv_pfes->GetQuadratureInterpolator(*ir);
      qi_kv->SetOutputLayout(QVectorLayout::byNODES);

      kv_e.SetSize(Rk->Height());
   }
   else if (auto *c = dynamic_cast<FunctionCoefficient *>(&kv_coeff))
   {
      // No setup needed.
   }
   else
   {
      MFEM_ABORT("Coefficient type not supported");
   }

   if (auto *c = dynamic_cast<GridFunctionCoefficient *>(&tls_coeff))
   {
      auto tls_pfes = static_cast<const ParGridFunction *>
                      (c->GetGridFunction())->ParFESpace();
      qi_tls = tls_pfes->GetQuadratureInterpolator(*ir);
      qi_tls->SetOutputLayout(QVectorLayout::byNODES);

      tls_e.SetSize(Rk->Height());
   }
   else if (auto *c = dynamic_cast<FunctionCoefficient *>(&tls_coeff))
   {
      // No setup needed.
   }
   else
   {
      MFEM_ABORT("Coefficient type not supported");
   }

   if (auto *c = dynamic_cast<FunctionCoefficient *>(&f_coeff))
   {
      // No setup needed.
   }
   else
   {
      MFEM_ABORT("Coefficient type not supported");
   }

   Array<int> empty;

   Mform = new ParBilinearForm(&kfes);
   auto mass_integrator = new MassIntegrator;
   mass_integrator->SetIntegrationRule(*ir);
   Mform->AddDomainIntegrator(mass_integrator);
   Mform->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   Mform->Assemble();
   Mform->FormSystemMatrix(empty, M);

   Kform = new ParBilinearForm(&kfes);
   // auto integ = new DiffusionIntegrator();
   auto integ = new ConservativeConvectionIntegrator(vel_coeff);
   integ->SetIntegrationRule(*ir);
   Kform->AddDomainIntegrator(integ);
   Kform->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   Kform->Assemble();
   Kform->FormSystemMatrix(empty, K);

   bform = new ParLinearForm(&kfes);
   auto dlf_integrator = new DomainLFIntegrator(f_coeff);
   dlf_integrator->SetIntRule(ir);
   bform->AddDomainIntegrator(dlf_integrator);
   bform->Assemble();

   M_inv = new CGSolver(MPI_COMM_WORLD);
   M_inv->iterative_mode = false;
   M_inv->SetOperator(*M);
}

void PrandtlKolmogorov::SetTime(double t)
{
   TimeDependentOperator::SetTime(t);

   // Set time of possibly time dependent coefficients.
   vel_coeff.SetTime(t);
   kv_coeff.SetTime(t);
   f_coeff.SetTime(t);
   k_bdrcoeff.SetTime(t);

   if (ess_attr.Size())
   {
      k_gf.ProjectBdrCoefficient(k_bdrcoeff, ess_attr);
      Pk->MultTranspose(k_gf, k_bdr_values);
   }

   bform->ParallelAssemble(b);

   if (auto *c = dynamic_cast<VectorGridFunctionCoefficient *>(&vel_coeff))
   {
      qi_vel->Values(u_e, u_q);
   }
   else if (auto *c = dynamic_cast<VectorFunctionCoefficient *>(&vel_coeff))
   {
      auto C = Reshape(u_q.HostWrite(), ir->GetNPoints(), dim, ne);
      DenseMatrix MQ;
      for (int e = 0; e < ne; ++e)
      {
         ElementTransformation& T = *kfes.GetElementTransformation(e);
         c->Eval(MQ, T, *ir);
         for (int qp = 0; qp < ir->GetNPoints(); ++qp)
         {
            for (int d = 0; d < dim; ++d)
            {
               C(qp,d,e) = MQ(d,qp);
            }
         }
      }
   }
   else
   {
      // Should never be reached.
   }

   if (auto *c = dynamic_cast<GridFunctionCoefficient *>(&kv_coeff))
   {
      qi_kv->Values(kv_e, kv_q);
   }
   else if (auto *c = dynamic_cast<FunctionCoefficient *>(&kv_coeff))
   {
      auto C = Reshape(kv_q.HostWrite(), ir->GetNPoints(), ne);
      for (int e = 0; e < ne; ++e)
      {
         ElementTransformation& T = *kfes.GetElementTransformation(e);
         for (int qp = 0; qp < ir->GetNPoints(); ++qp)
         {
            const IntegrationPoint &ip = ir->IntPoint(qp);
            C(qp,e) = c->Eval(T, ip);
         }
      }
   }
   else
   {
      // Should never be reached.
   }

   if (auto *c = dynamic_cast<GridFunctionCoefficient *>(&tls_coeff))
   {
      qi_tls->Values(tls_e, tls_q);
   }
   else if (auto *c = dynamic_cast<FunctionCoefficient *>(&tls_coeff))
   {
      auto C = Reshape(tls_q.HostWrite(), ir->GetNPoints(), ne);
      for (int e = 0; e < ne; ++e)
      {
         ElementTransformation& T = *kfes.GetElementTransformation(e);
         for (int qp = 0; qp < ir->GetNPoints(); ++qp)
         {
            const IntegrationPoint &ip = ir->IntPoint(qp);
            C(qp,e) = c->Eval(T, ip);
         }
      }
   }
   else
   {
      // Should never be reached.
   }

   if (auto *c = dynamic_cast<FunctionCoefficient *>(&f_coeff))
   {
      auto C = Reshape(f_q.HostWrite(), ir->GetNPoints(), ne);
      for (int e = 0; e < ne; ++e)
      {
         ElementTransformation& T = *kfes.GetElementTransformation(e);
         for (int qp = 0; qp < ir->GetNPoints(); ++qp)
         {
            const IntegrationPoint &ip = ir->IntPoint(qp);
            C(qp,e) = c->Eval(T, ip);
         }
      }
   }
   else
   {
      f_q = 0.0;
   }
}

void PrandtlKolmogorov::Mult(const Vector &x, Vector &y) const
{
   Apply(x, z1);
   M_inv->Mult(z1, y);

   for (int i = 0; i < ess_tdof_list.Size(); i++)
   {
      y(ess_tdof_list[i]) = 0.0;
   }
}

void PrandtlKolmogorov::Apply(const Vector &x, Vector &y) const
{
   const int d1d = maps->ndof, q1d = maps->nqpt;

   MFEM_ASSERT(y.Size() == Pk->Height(), "y wrong size");

   // T -> L
   Pk->Mult(x, k_l);
   // L -> E
   Rk->Mult(k_l, k_e);

   // Reset output vector.
   y_e = 0.0;

   if (dim == 2)
   {
      const int id = (d1d << 4) | q1d;
      switch (id)
      {
         case 0x33:
         {
            PrandtlKolmogorovApply2D<3, 3>(eval_mode, ne, maps->B, maps->G,
                                           ir->GetWeights(),
                                           geom->J,
                                           geom->detJ, k_e, y_e, u_q, kv_q, f_q, tls_q, mu_calibration_const);
            break;
         }
         case 0x44:
         {
            PrandtlKolmogorovApply2D<4, 4>(eval_mode, ne, maps->B, maps->G,
                                           ir->GetWeights(),
                                           geom->J,
                                           geom->detJ, k_e, y_e, u_q, kv_q, f_q, tls_q, mu_calibration_const);
            break;
         }
         case 0x55:
         {
            PrandtlKolmogorovApply2D<5, 5>(eval_mode, ne, maps->B, maps->G,
                                           ir->GetWeights(),
                                           geom->J,
                                           geom->detJ, k_e, y_e, u_q, kv_q, f_q, tls_q, mu_calibration_const);
            break;
         }
         case 0x99:
         {
            PrandtlKolmogorovApply2D<9, 9>(eval_mode, ne, maps->B, maps->G,
                                           ir->GetWeights(),
                                           geom->J,
                                           geom->detJ, k_e, y_e, u_q, kv_q, f_q, tls_q, mu_calibration_const);
            break;
         }

         default:
            printf("id=%x\n", id);
            MFEM_ABORT("unknown kernel");
      }
   }
   else if (dim == 3)
   {
      MFEM_ABORT("unknown kernel");
   }
   // E -> L
   Rk->MultTranspose(y_e, y_l);
   // L -> T
   Pk->MultTranspose(y_l, y);
}

void PrandtlKolmogorov::ApplyEssentialBC(const double t, Vector& y)
{
   k_bdrcoeff.SetTime(t);
   k_gf.ProjectBdrCoefficient(k_bdrcoeff, ess_attr);
   Pk->MultTranspose(k_gf, k_bdr_values);

   for (int i = 0; i < ess_tdof_list.Size(); i++)
   {
      y(ess_tdof_list[i]) = k_bdr_values(ess_tdof_list[i]);
   }
}

int PrandtlKolmogorov::PostProcessCallback(realtype t, N_Vector y,
                                           void *user_data)
{
   auto ark = static_cast<ARKStepSolver *>(user_data);
   SundialsNVector mfem_y(y);
   static_cast<PrandtlKolmogorov *>(ark->GetOperator())->ApplyEssentialBC(t,
                                                                          mfem_y);
   return 0;
}

}
}