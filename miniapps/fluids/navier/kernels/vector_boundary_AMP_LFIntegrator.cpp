#include "vector_boundary_AMP_LFIntegrator.hpp"

using namespace mfem;

void VectorBoundaryTangentialAMPLFIntegrator::AssembleRHSElementVect(
    const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dim = Tr.GetSpaceDim(); 
   int vdim = VG->GetVDim();
   int dof = el.GetDof();

   Qvec.SetSize(dim);
   Uvec.SetSize(dim);
   Vector f_tangent(dim);
   Vector tangent_vec(dim);
   Vector nor(dim);
   shape.SetSize(dof);

   elvect.SetSize(dof * vdim);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = 2 * el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);
      const DenseMatrix &Jac = Tr.Jacobian();

      // Evaluate Vector Coefficients
      /*VG->Eval(Qvec, Tr, ip); 
      VU->Eval(Uvec, Tr, ip);
      // Qvec = f = (g + alpha uS)
      Qvec.Add(Q->Eval(Tr, ip), Uvec); */

      // NOTE: similar to Robin we're dropping the sigma term in the momentum eqn. The above is the original formulation
      Qvec = 0.0;
      VU->Eval(Uvec, Tr, ip);
      // Qvec = f = (g + alpha uS)
      Qvec.Add(Q->Eval(Tr, ip), Uvec);

      if (dim == 2)
      {
         // --- RETAIN OLD 2D VERSION ---
         tangent_vec(0) = Jac(0, 0);
         tangent_vec(1) = Jac(1, 0);
         real_t scale = tangent_vec.Norml2();
         tangent_vec *= 1.0 / scale; 

         // Project f onto the single tangent line
         real_t f_t = Qvec * tangent_vec;
         
         for (int k = 0; k < dim; k++)
         {
            f_tangent(k) = f_t * tangent_vec(k);
         }
      }
      else if (dim == 3)
      {
         // --- 3D PLANE PROJECTION VERSION ---
         // f_tangent = f - (f . n) n
         CalcOrtho(Jac, nor);
         nor /= nor.Norml2(); // Ensure unit normal

         real_t f_dot_n = Qvec * nor;
         
         for (int k = 0; k < dim; k++)
         {
            f_tangent(k) = Qvec(k) - f_dot_n * nor(k);
         }
      }
      else 
      {
         mfem_error("VectorBoundaryTangentialAMPLFIntegrator: dim must be 2 or 3.");
      }

      real_t w = Tr.Weight() * ip.weight;
      el.CalcShape(ip, shape);

      // Assemble into elvect (Ordering::byNodes)
      for (int k = 0; k < vdim; k++)
      {
         for (int j = 0; j < dof; j++)
         {
            elvect(dof * k + j) += w * f_tangent(k) * shape(j);
         }
      }
   }
}


void VectorBoundaryNormalAMPLFIntegrator::AssembleRHSElementVect(
    const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dim = el.GetDim() + 1;
   int vdim = VG->GetVDim();
   int dof = el.GetDof();

   normal.SetSize(dim);
   Qvec.SetSize(dim);
   Uvec.SetSize(dim);

   shape.SetSize(dof);
   elvect.SetSize(dof * vdim);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = 2 * el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint(&ip);

      // Compute normal vector
      CalcOrtho(Tr.Jacobian(), normal);
      const real_t scale = normal.Norml2();
      normal /= scale;

      // Compute shape function
      el.CalcShape(ip, shape);

      // Vector coefficient
      /* VG->Eval(Qvec, Tr, ip);          // g = sigmaS_dot_n
      VU->Eval(Uvec, Tr, ip);
      Qvec.Add(Q->Eval(Tr, ip), Uvec); // f = (g + alpha uS)
      real_t f_n = Qvec * normal;      // f_n = (g + alpha uS) \cdot n
      f_n *= Tr.Weight() * ip.weight; */


      // NOTE: similar to Robin we're dropping the sigma term in the momentum eqn. The above is the original formulation
      Qvec = 0.0;
      VU->Eval(Uvec, Tr, ip);
      // Qvec = f = (g + alpha uS)
      Qvec.Add(Q->Eval(Tr, ip), Uvec);
      real_t f_n = Qvec * normal;      // f_n = (g + alpha uS) \cdot n
      f_n *= Tr.Weight() * ip.weight;


      for (int j = 0; j < dof; j++)
      {
         for (int k = 0; k < dim; k++)
         {
            elvect(dof*k+j) += f_n * normal(k) * shape(j);
         }
      }
   }
}