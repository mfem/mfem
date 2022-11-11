#include <memory>
#include "prandtl_kolmogorov.hpp"
#include "prandtl_kolmogorov_kernels.hpp"

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

PrandtlKolmogorov::PrandtlKolmogorov(ParFiniteElementSpace &kfes,
                                     VectorCoefficient &vel_coeff,
                                     Coefficient &kv_coeff,
                                     Coefficient &f_coeff,
                                     Coefficient &wall_distance_coeff,
                                     Coefficient &k_bdrcoeff,
                                     const double mu_calibration_const,
                                     Array<int> eattr) :
   kfes(kfes),
   dim(kfes.GetParMesh()->Dimension()),
   ne(kfes.GetNE()),
   vel_coeff(vel_coeff),
   kv_coeff(kv_coeff),
   mu_calibration_const(mu_calibration_const),
   f_coeff(f_coeff),
   k_bdrcoeff(k_bdrcoeff),
   wall_distance_coeff(wall_distance_coeff)
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
   wd_q.SetSize(ir->GetNPoints() * ne);
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
   else if (auto *c = dynamic_cast<ConstantCoefficient *>(&kv_coeff))
   {
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

   if (auto *c = dynamic_cast<GridFunctionCoefficient *>(&wall_distance_coeff))
   {
      auto tls_pfes = static_cast<const ParGridFunction *>
                      (c->GetGridFunction())->ParFESpace();
      qi_tls = tls_pfes->GetQuadratureInterpolator(*ir);
      qi_tls->SetOutputLayout(QVectorLayout::byNODES);

      wd_e.SetSize(Rk->Height());
   }
   else if (auto *c = dynamic_cast<FunctionCoefficient *>(&wall_distance_coeff))
   {
      // No setup needed.
   }
   else
   {
      MFEM_ABORT("Coefficient type not supported");
   }

   if (auto *c = dynamic_cast<ConstantCoefficient *>(&f_coeff))
   {
      // No setup needed.
   }
   else if (auto *c = dynamic_cast<FunctionCoefficient *>(&f_coeff))
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

   m.SetSize(kfes.GetTrueVSize());
   minv.SetSize(kfes.GetTrueVSize());
   Mform->AssembleDiagonal(m);
   {
      const auto d_m = m.Read();
      auto d_minv = minv.Write();
      MFEM_FORALL(i, m.Size(), d_minv[i] = 1.0 / d_m[i];);
   }

   A_amg = new HypreBoomerAMG;
   A_amg->SetPrintLevel(0);

   A_inv = new GMRESSolver(MPI_COMM_WORLD);
   A_inv->iterative_mode = true;
   A_inv->SetMaxIter(100);
   A_inv->SetKDim(30);
   A_inv->SetRelTol(1e-8);
   A_inv->SetPrintLevel(0);
   A_inv->SetPreconditioner(*A_amg);
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

   if (auto *c = dynamic_cast<VectorGridFunctionCoefficient *>(&vel_coeff))
   {
      Ru->Mult(*c->GetGridFunction(), u_e);
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
      Rk->Mult(*c->GetGridFunction(), kv_e);
      qi_kv->Values(kv_e, kv_q);
   }
   if (auto *c = dynamic_cast<ConstantCoefficient *>(&kv_coeff))
   {
      auto C = Reshape(kv_q.HostWrite(), ir->GetNPoints(), ne);
      for (int e = 0; e < ne; ++e)
      {
         for (int qp = 0; qp < ir->GetNPoints(); ++qp)
         {
            C(qp, e) = c->constant;
         }
      }
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

   if (auto *c = dynamic_cast<GridFunctionCoefficient *>(&wall_distance_coeff))
   {
      Rk->Mult(*c->GetGridFunction(), wd_e);
      qi_tls->Values(wd_e, wd_q);
   }
   else if (auto *c = dynamic_cast<FunctionCoefficient *>(&wall_distance_coeff))
   {
      auto C = Reshape(wd_q.HostWrite(), ir->GetNPoints(), ne);
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

   if (auto *c = dynamic_cast<ConstantCoefficient *>(&f_coeff))
   {
      auto C = Reshape(f_q.HostWrite(), ir->GetNPoints(), ne);
      for (int e = 0; e < ne; ++e)
      {
         for (int qp = 0; qp < ir->GetNPoints(); ++qp)
         {
            C(qp, e) = c->constant;
         }
      }
   }
   else if (auto *c = dynamic_cast<FunctionCoefficient *>(&f_coeff))
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
   // y = M^-1 f(x)

   Apply(x, z1);
   y = z1;
   vmul(minv, y);

   for (int i = 0; i < ess_tdof_list.Size(); i++)
   {
      y(ess_tdof_list[i]) = 0.0;
   }
}

void PrandtlKolmogorov::Apply(const Vector &x, Vector &y) const
{
   const int d1d = maps->ndof, q1d = maps->nqpt;

   MFEM_ASSERT(x.Size() == Pk->Width(), "y wrong size");

   // T -> L
   Pk->Mult(x, k_l);
   // L -> E
   Rk->Mult(k_l, k_e);

   // Reset output vector.
   y_e = 0.0;

   // printf("apply()\n");

   if (dim == 2)
   {
      const int id = (d1d << 4) | q1d;
      switch (id)
      {
         case 0x22:
         {
            PrandtlKolmogorovApply2D<2, 2>(eval_mode, ne, maps->B, maps->G,
                                           ir->GetWeights(),
                                           geom->J,
                                           geom->detJ, k_e, y_e, u_q, kv_q, f_q, wd_q, mu_calibration_const);
            break;
         }
         case 0x33:
         {
            PrandtlKolmogorovApply2D<3, 3>(eval_mode, ne, maps->B, maps->G,
                                           ir->GetWeights(),
                                           geom->J,
                                           geom->detJ, k_e, y_e, u_q, kv_q, f_q, wd_q, mu_calibration_const);
            break;
         }
         case 0x44:
         {
            PrandtlKolmogorovApply2D<4, 4>(eval_mode, ne, maps->B, maps->G,
                                           ir->GetWeights(),
                                           geom->J,
                                           geom->detJ, k_e, y_e, u_q, kv_q, f_q, wd_q, mu_calibration_const);
            break;
         }
         case 0x55:
         {
            PrandtlKolmogorovApply2D<5, 5>(eval_mode, ne, maps->B, maps->G,
                                           ir->GetWeights(),
                                           geom->J,
                                           geom->detJ, k_e, y_e, u_q, kv_q, f_q, wd_q, mu_calibration_const);
            break;
         }
         case 0x99:
         {
            PrandtlKolmogorovApply2D<9, 9>(eval_mode, ne, maps->B, maps->G,
                                           ir->GetWeights(),
                                           geom->J,
                                           geom->detJ, k_e, y_e, u_q, kv_q, f_q, wd_q, mu_calibration_const);
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

int PrandtlKolmogorov::SUNImplicitSetup(const Vector &x, const Vector &fx,
                                        int jok, int *jcur, double gamma)
{
   const int d1d = maps->ndof, q1d = maps->nqpt;

   delete Amat;
   mat = new SparseMatrix(kfes.GetVSize());

   // T -> L
   Pk->Mult(x, k_l);
   // L -> E
   Rk->Mult(k_l, k_e);

   // Allocate
   dRdk.SetSize(d1d*d1d * d1d*d1d * ne);

   // Reset output vector.
   dRdk = 0.0;

   if (dim == 2)
   {
      const int id = (d1d << 4) | q1d;
      switch (id)
      {
         case 0x22:
         {
            PrandtlKolmogorovAssembleJacobian2D<2, 2>(ne, maps->B, maps->G,
                                                      ir->GetWeights(),
                                                      geom->J,
                                                      geom->detJ, k_e, dRdk, kv_q, wd_q, mu_calibration_const, gamma);
            break;
         }
         case 0x33:
         {
            PrandtlKolmogorovAssembleJacobian2D<3, 3>(ne, maps->B, maps->G,
                                                      ir->GetWeights(),
                                                      geom->J,
                                                      geom->detJ, k_e, dRdk, kv_q, wd_q, mu_calibration_const, gamma);
            break;
         }
         case 0x44:
         {
            PrandtlKolmogorovAssembleJacobian2D<4, 4>(ne, maps->B, maps->G,
                                                      ir->GetWeights(),
                                                      geom->J,
                                                      geom->detJ, k_e, dRdk, kv_q, wd_q, mu_calibration_const, gamma);
            break;
         }
         case 0x55:
         {
            PrandtlKolmogorovAssembleJacobian2D<5, 5>(ne, maps->B, maps->G,
                                                      ir->GetWeights(),
                                                      geom->J,
                                                      geom->detJ, k_e, dRdk, kv_q, wd_q, mu_calibration_const, gamma);
            break;
         }
         default:
            printf("id=%x\n", id);
            MFEM_ABORT("unknown kernel");
      }

      // Assemble processor local SparseMatrix
      for (int e = 0; e < ne; e++)
      {
         DenseMatrix dRdk_e(dRdk.GetData() + d1d*d1d * d1d*d1d * e, d1d*d1d, d1d*d1d);

         Array<int> vdofs;
         kfes.GetElementDofs(e, vdofs);

         const Array<int> &dmap =
            dynamic_cast<const TensorBasisElement&>(*kfes.GetFE(e)).GetDofMap();

         Array<int> vdofs_mapped(vdofs.Size());

         for (int i = 0; i < vdofs_mapped.Size(); i++)
         {
            vdofs_mapped[i] = vdofs[dmap[i]];
         }

         mat->AddSubMatrix(vdofs_mapped, vdofs_mapped, dRdk_e, 1);
      }
   }
   else
   {
      MFEM_ABORT("unknown kernel");
   }

   // dRdk.Destroy();
   mat->Finalize();

   if (Pk_mat == nullptr)
   {
      Pk_mat = kfes.Dof_TrueDof_Matrix();
   }

   auto tmp = new HypreParMatrix(kfes.GetComm(),
                                 kfes.GlobalVSize(),
                                 kfes.GetDofOffsets(),
                                 mat);
   Amat = RAP(tmp, Pk_mat);
   delete tmp;
   delete mat;

   Amat->EliminateBC(ess_tdof_list, DiagonalPolicy::DIAG_ONE);

   A_inv->SetOperator(*Amat);

   *jcur = 1;

   return 0;
}

int PrandtlKolmogorov::SUNImplicitSolve(const Vector &b, Vector &x, double tol)
{
   Vector b_mod(b.Size());
   // The RHS has to modified to have zeros at the essential boundary
   // conditions. This way the correction to those values will be zero.
   b_mod = b;
   vmul(m, b_mod);
   b_mod.SetSubVector(ess_tdof_list, 0.0);

   A_inv->SetAbsTol(tol);
   A_inv->Mult(b_mod, x);

   return 0;
}

void PrandtlKolmogorov::ApplyEssentialBC(const double t, Vector& y)
{
   k_bdrcoeff.SetTime(t);
   k_gf = 0.0;
   k_gf.ProjectBdrCoefficient(k_bdrcoeff, ess_attr);
   // Pk->MultTranspose(k_gf, k_bdr_values);
   kfes.GetRestrictionMatrix()->Mult(k_gf, k_bdr_values);

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