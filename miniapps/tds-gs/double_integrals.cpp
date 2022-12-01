#include "double_integrals.hpp"

using namespace std;
using namespace mfem;


void DoubleBoundaryBFIntegrator::AssembleElementMatrix(
   const FiniteElement &el1,
   const FiniteElement &el2,
   ElementTransformation &Trans1,
   ElementTransformation &Trans2,
   DenseMatrix &elmat)
{
   const int sdim = Trans1.GetSpaceDim();
   const int nd1 = el1.GetDof();
   const int nd2 = el2.GetDof();

   // The result is a square matrix of size (nd1+nd2) x (nd1+nd2):
   elmat.SetSize(nd1 + nd2);
   elmat = 0.0;

   // Determine quadrature rules for both elements.
   const int ir1_order = 2*el1.GetOrder();
   const IntegrationRule &ir1 = IntRules.Get(el1.GetGeomType(), ir1_order);
   const int ir2_order = 2*el2.GetOrder();
   const IntegrationRule &ir2 = IntRules.Get(el2.GetGeomType(), ir2_order);

   const DofToQuad &dq1 = el1.GetDofToQuad(ir1, DofToQuad::FULL);
   const DofToQuad &dq2 = el2.GetDofToQuad(ir2, DofToQuad::FULL);
   const DenseMatrix Bt1(const_cast<double*>(dq1.Bt.begin()),
                         dq1.ndof, dq1.nqpt);
   const DenseMatrix Bt2(const_cast<double*>(dq2.Bt.begin()),
                         dq2.ndof, dq2.nqpt);

   Vector w_det_J1(ir1.GetNPoints());
   Vector w_det_J2(ir2.GetNPoints());
   for (int q1 = 0; q1 < ir1.GetNPoints(); q1++)
   {
      Trans1.SetIntPoint(&ir1.IntPoint(q1));
      w_det_J1(q1) = ir1.IntPoint(q1).weight * Trans1.Weight();
   }
   for (int q2 = 0; q2 < ir2.GetNPoints(); q2++)
   {
      Trans2.SetIntPoint(&ir2.IntPoint(q2));
      w_det_J2(q2) = ir2.IntPoint(q2).weight * Trans2.Weight();
   }

   Vector x1(sdim), x2(sdim);
   // Note: The complexity in the implementation below is:
   //          O(nq2 x nq1 x (nd1+nd2)^2)
   //       where nq1 = ir1.GetNPoints() and nq2 = ir2.GetNPoints().
   //       This can be improved by factoring out common terms.
   for (int q2 = 0; q2 < ir2.GetNPoints(); q2++)
   {
      Trans2.Transform(ir2.IntPoint(q2), x2);
      for (int q1 = 0; q1 < ir1.GetNPoints(); q1++)
      {
         Trans1.Transform(ir1.IntPoint(q1), x1);
         const double w_M_q1_q2 = w_det_J1(q1) * w_det_J2(q2) * M(x1, x2);

         for (int u1 = 0; u1 < nd1; u1++)
         {
            const double z = w_M_q1_q2 * Bt1(u1,q1);
            // \int_{\Gamma_2} \int_{\Gamma_1} u(x1) M(x1,x2) v(x1) dx1 dx2
            // E11(v1,u1) += wM(q1,q2) * Bt1(u1,q1) * Bt1(v1,q1)
            for (int v1 = 0; v1 < nd1; v1++)
            {
               elmat(v1,u1) += z * Bt1(v1,q1);
            }
            // \int_{\Gamma_2} \int_{\Gamma_1} [-u(x1) M(x1,x2) v(x2)] dx1 dx2
            // E21(v2,u1) -= wM(q1,q2) * Bt1(u1,q1) * Bt2(v2,q2)
            for (int v2 = 0; v2 < nd2; v2++)
            {
               elmat(nd1+v2,u1) -= z * Bt2(v2,q2);
            }
         }

         for (int u2 = 0; u2 < nd2; u2++)
         {
            const double z = w_M_q1_q2 * Bt2(u2,q2);
            // \int_{\Gamma_2} \int_{\Gamma_1} [-u(x2) M(x1,x2) v(x1)] dx1 dx2
            // E12(v1,u2) -= wM(q1,q2) * Bt2(u2,q2) * Bt1(v1,q1)
            for (int v1 = 0; v1 < nd1; v1++)
            {
               elmat(v1,nd1+u2) -= z * Bt1(v1,q1);
            }
            // \int_{\Gamma_2} \int_{\Gamma_1} u(x2) M(x1,x2) v(x2) dx1 dx2
            // E22(v2,u2) += wM(q1,q2) * Bt2(u2,q2) * Bt2(v2,q2)
            for (int v2 = 0; v2 < nd2; v2++)
            {
               elmat(nd1+v2,nd1+u2) += z * Bt2(v2,q2);
            }
         }
      }
   }
}


void AssembleDoubleBoundaryIntegrator(BilinearForm &a,
                                      DoubleIntegralBFIntegrator &di_bfi,
                                      int attribute)
{

   FiniteElementSpace &fes = *a.FESpace();
   Mesh &mesh = *fes.GetMesh();
   SparseMatrix &sp_mat = a.SpMat();
   const int skip_zeros = 0;

   IsoparametricTransformation T1, T2;
   Array<int> vdofs1, vdofs2, vdofs_all;
   DenseMatrix elmat;
   for (int i1 = 0; i1 < mesh.GetNBE(); i1++)
   {
      
      mesh.GetBdrElementTransformation(i1, &T1);

      // DAS
      if ((attribute != NULL) && (T1.Attribute != attribute)) {
        continue;
      }
      // cout << i1 << " attrib: " << T1.Attribute << endl;
        
      const FiniteElement &el1 = *fes.GetBE(i1);
      fes.GetBdrElementVDofs(i1, vdofs1);
      vdofs_all = vdofs1;
      for (int i2 = 0; i2 < mesh.GetNBE(); i2++)
      {
      
         mesh.GetBdrElementTransformation(i2, &T2);
         // DAS
         if ((attribute != NULL) && (T2.Attribute != attribute)) {
           continue;
         }
         // cout << "  " << i2 << " attrib: " << T1.Attribute << endl;
         const FiniteElement &el2 = *fes.GetBE(i2);
         fes.GetBdrElementVDofs(i2, vdofs2);

         vdofs_all.SetSize(vdofs1.Size());
         vdofs_all.Append(vdofs2);
         // Note: there may be repetitions of indices in vdofs_all due to
         // vdofs shared by elements 1 and 2.

         di_bfi.AssembleElementMatrix(el1, el2, T1, T2, elmat);

         // This is not the most efficient way to assemble the contributions
         // into the global matrix.
         sp_mat.AddSubMatrix(vdofs_all, vdofs_all, elmat, skip_zeros);
      }
   }
}

