/// Templated AD (block) nonlinear form integrators implementations
#pragma once
#include "_dof_pg.hpp"

namespace mfem
{

/// Compute the local energy
template <ADEval... modes>
real_t ADDofPGNonlinearFormIntegrator<modes...>::GetElementEnergy(
   const Array<const FiniteElement *> &el,
   ElementTransformation &Tr,
   const Array<const Vector*> &elfun)
{
   const int numSpaces = el.Size() - entropies.size();
   MFEM_ASSERT(numSpaces == entropies.size(),
               "ADDofPGNonlinearFormIntegrator: "
               "Currently, all primal spaces must have an associated dual space");
   Array<const FiniteElement*> primal_el;
   Array<const Vector*> primal_elfun;
   Array<const FiniteElement*> dual_el;
   Array<const Vector*> dual_elfun;
   for (int i = 0; i < numSpaces; i++)
   {
      primal_el.Append(el[i]);
      primal_elfun.Append(elfun[i]);
   }
   for (int i = numSpaces; i < el.Size(); i++)
   {
      dual_el.Append(el[i]);
      dual_elfun.Append(elfun[i]);
   }
   real_t energy = ADBlockNonlinearFormIntegrator<modes...>::GetElementEnergy(
                      primal_el, Tr, primal_elfun);
   Vector x(2), primal_x(x, 0, 1), dual_x(x, 1, 1);
   Vector latent_k_val;
   real_t pg_energy = 0.0;
   for (int i=0; i<entropies.size(); i++)
   {
      GridFunction &dual_val_k = pg_functional.GetPrevLatent(i);
      auto &primal_fe = *primal_el[i];
      auto &dual_fe = *dual_el[i];
      auto &primal_val = *primal_elfun[i];
      auto &dual_val = *dual_elfun[i];
      dual_val_k.GetElementDofValues(Tr.ElementNo, latent_k_val);
      MFEM_ASSERT(primal_fe.GetDof() == dual_fe.GetDof(),
                  "ADDofPGNonlinearFormIntegrator: "
                  "primal and dual finite elements must have the same number of dofs");
      const IntegrationRule &ir = primal_fe.GetNodes();
      for (int j=0; j<ir.GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         Tr.SetIntPoint(&ip);
         primal_x[0] = (*primal_elfun[i])[j];
         dual_x[0] = (*dual_elfun[i])[j];
         pg_energy += (
                         primal_val[j]*(dual_val[j]-dual_val_k[j])
                         - (*entropies[i])(dual_x, Tr, ip)
                      )*Tr.Weight()*ip.weight;
      }
   }
   return energy + pg_energy / pg_functional.GetAlpha();
}

/// Perform the local action of the NonlinearFormIntegrator
template <ADEval... modes>
void ADDofPGNonlinearFormIntegrator<modes...>::AssembleElementVector(
   const Array<const FiniteElement *>&el,
   ElementTransformation &Tr,
   const Array<const Vector *>&elfun,
   const Array<Vector *>&elvect)
{
   const int numSpaces = el.Size() - entropies.size();
   MFEM_ASSERT(numSpaces == entropies.size(),
               "ADDofPGNonlinearFormIntegrator: "
               "Currently, all primal spaces must have an associated dual space");
   real_t alpha = pg_functional.GetAlpha();
   Array<const FiniteElement*> primal_el;
   Array<const Vector*> primal_elfun;
   Array<Vector*> primal_elvect;
   Array<const FiniteElement*> dual_el;
   Array<const Vector*> dual_elfun;
   Array<Vector*> dual_elvect;
   for (int i = 0; i < numSpaces; i++)
   {
      primal_el.Append(el[i]);
      primal_elfun.Append(elfun[i]);
      primal_elvect.Append(elvect[i]);
   }
   for (int i = numSpaces; i < el.Size(); i++)
   {
      dual_el.Append(el[i]);
      dual_elfun.Append(elfun[i]);
      dual_elvect.Append(elvect[i]);
   }
   ADBlockNonlinearFormIntegrator<modes...>::AssembleElementVector(
      primal_el, Tr, primal_elfun, primal_elvect);
   Vector x(2), primal_x(x, 0, 1), dual_x(x, 1, 1), J(1);
   Vector latent_k_val;
   real_t pg_energy = 0.0;
   real_t w;
   for (int i=0; i<entropies.size(); i++)
   {
      GridFunction &dual_val_k = pg_functional.GetPrevLatent(i);
      auto &primal_fe = *primal_el[i];
      auto &dual_fe = *dual_el[i];
      auto &primal_val = *primal_elfun[i];
      auto &dual_val = *dual_elfun[i];
      dual_val_k.GetElementDofValues(Tr.ElementNo, latent_k_val);
      MFEM_ASSERT(primal_fe.GetDof() == dual_fe.GetDof(),
                  "ADDofPGNonlinearFormIntegrator: "
                  "primal and dual finite elements must have the same number of dofs");
      const IntegrationRule &ir = primal_fe.GetNodes();
      dual_elvect[i]->SetSize(ir.GetNPoints());
      for (int j=0; j<ir.GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         Tr.SetIntPoint(&ip);
         w = Tr.Weight()*ip.weight/alpha;

         primal_x[0] = (*primal_elfun[i])[j];
         dual_x[0] = (*dual_elfun[i])[j];
         entropies[i]->Gradient(dual_x, J);
         (*primal_elvect[i])[j] += (dual_val[j]-dual_val_k[j])*w;
         (*dual_elvect[i])[j] = (primal_val[j]-J[0])*w;
      }
   }
}

/// Perform the local action of the NonlinearFormIntegrator
template <ADEval... modes>
void ADDofPGNonlinearFormIntegrator<modes...>::AssembleElementGrad(
   const Array<const FiniteElement *>&el,
   ElementTransformation &Tr,
   const Array<const Vector *>&elfun,
   const Array2D<DenseMatrix *>&elmat)
{
   const int numSpaces = el.Size() - entropies.size();
   MFEM_ASSERT(numSpaces == entropies.size(),
               "ADDofPGNonlinearFormIntegrator: "
               "Currently, all primal spaces must have an associated dual space");
   real_t alpha = pg_functional.GetAlpha();
   Array<const FiniteElement*> primal_el;
   Array<const Vector*> primal_elfun;
   Array2D<DenseMatrix*> primal_elmat(numSpaces, numSpaces);
   Array<const FiniteElement*> dual_el;
   Array<const Vector*> dual_elfun;
   Array<DenseMatrix*> dual_elmat;
   Array<DenseMatrix*> upper_mat;
   Array<DenseMatrix*> lower_mat;
   for (int j = 0; j < numSpaces; j++)
   {
      primal_el.Append(el[j]);
      primal_elfun.Append(elfun[j]);
      for (int i=0; i<numSpaces; i++)
      {
         primal_elmat(i,j) = elmat(i,j);
      }
   }
   for (int i = numSpaces; i < el.Size(); i++)
   {
      dual_el.Append(el[i]);
      dual_elfun.Append(elfun[i]);
      dual_elmat.Append(elmat(i,i));
   }
   for (int i=0; i<numSpaces; i++)
   {
      elmat(i, i+numSpaces)->SetSize(elfun[i]->Size());
      (*elmat(i, i+numSpaces)) = 0.0;
      elmat(i+numSpaces, i)->SetSize(elfun[i]->Size());
      (*elmat(i+numSpaces, i)) = 0.0;

      lower_mat.Append(elmat(i+numSpaces, i));
      upper_mat.Append(elmat(i, i+numSpaces));

      // zero out the off-diagonal of off-diagonal blocks
      for (int j=0; j<numSpaces; j++)
      {
         if (i==j) { continue; }
         elmat(i + numSpaces, j + numSpaces)->SetSize(elfun[i]->Size(), elfun[j]->Size());
         (*elmat(i + numSpaces, j + numSpaces)) = 0.0;
         elmat(j + numSpaces, i + numSpaces)->SetSize(elfun[j]->Size(), elfun[i]->Size());
         (*elmat(j + numSpaces, i + numSpaces)) = 0.0;
         elmat(i, j+numSpaces)->SetSize(elfun[i]->Size(), elfun[j]->Size());
         (*elmat(i, j+numSpaces)) = 0.0;
         elmat(i+numSpaces, j)->SetSize(elfun[i]->Size(), elfun[j]->Size());
         (*elmat(j+numSpaces, i)) = 0.0;
      }
   }

   ADBlockNonlinearFormIntegrator<modes...>::AssembleElementGrad(
      primal_el, Tr, primal_elfun, primal_elmat);
   Vector x(2), primal_x(x, 0, 1), dual_x(x, 1, 1);
   DenseMatrix H(1);
   Vector latent_k_val;
   real_t pg_energy = 0.0;
   real_t w;
   for (int i=0; i<entropies.size(); i++)
   {
      GridFunction &dual_val_k = pg_functional.GetPrevLatent(i);
      auto &primal_fe = *primal_el[i];
      auto &dual_fe = *dual_el[i];
      auto &dual_val = *dual_elfun[i];
      dual_val_k.GetElementDofValues(Tr.ElementNo, latent_k_val);
      MFEM_ASSERT(primal_fe.GetDof() == dual_fe.GetDof(),
                  "ADDofPGNonlinearFormIntegrator: "
                  "primal and dual finite elements must have the same number of dofs");
      const IntegrationRule &nodes = primal_fe.GetNodes();
      dual_elmat[i]->SetSize(nodes.GetNPoints());
      (*dual_elmat[i]) = 0.0;
      lower_mat[i]->SetSize(nodes.GetNPoints());
      (*lower_mat[i]) = 0.0;
      upper_mat[i]->SetSize(nodes.GetNPoints());
      (*upper_mat[i]) = 0.0;
      for (int j=0; j<nodes.GetNPoints(); j++)
      {
         const IntegrationPoint &ip = nodes.IntPoint(j);
         Tr.SetIntPoint(&ip);
         w = Tr.Weight()*ip.weight/alpha;

         dual_x[0] = (*dual_elfun[i])[j];
         entropies[i]->Hessian(dual_x, H);
         (*dual_elmat[i])(j,j) = -H(0,0)*w;
         (*lower_mat[i])(j,j) = w;
         (*upper_mat[i])(j,j) = w;
      }
   }
}
} // namespace mfem
