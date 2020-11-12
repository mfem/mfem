#include "mfem.hpp"
#include "../../general/forall.hpp"
#include "genericintegrator.hpp"

#pragma once

namespace mfem
{
struct qfunc_output_type
{
   double f0;
   Vector f1;
};

template<typename qfunc_type>
class QFunctionIntegrator : public GenericIntegrator
{
protected:
#ifndef MFEM_THREAD_SAFE
   Vector shape, te_shape;
#endif
   // PA extension
   const FiniteElementSpace *fespace;
   const DofToQuad *maps;        ///< Not owned
   const GeometricFactors *geom; ///< Not owned
   int dim, ne, nq, dofs1D, quad1D;

   // Geometric factors
   Vector J;
   Vector W;

   qfunc_type qf;

public:
   QFunctionIntegrator(qfunc_type f, const IntegrationRule *ir = nullptr);

   void Setup(const FiniteElementSpace &fes);

   void Apply(const Vector &, Vector &) const;
};

// QFunc(double, mfem::Vector) -> {double, mfem::Vector}
template<int T_D1D = 0, int T_Q1D = 0, typename qfunc_type>
static void Apply2D(const int dim,
                    const int D1D,
                    const int Q1D,
                    const int NE,
                    const Array<double> &v1d_,
                    const Array<double> &dv1d_dX_,
                    const Vector &J_,
                    const Vector &W_,
                    const Vector &u_in_,
                    const qfunc_type qf,
                    Vector &y_)
{
   auto v1d = Reshape(v1d_.Read(), Q1D, D1D);
   auto dv1d_dX = Reshape(dv1d_dX_.Read(), Q1D, D1D);
   // (NQ x SDIM x DIM x NE)
   auto J = Reshape(J_.Read(), Q1D, Q1D, 2, 2, NE);
   auto W = Reshape(W_.Read(), Q1D, Q1D);
   auto u = Reshape(u_in_.Read(), D1D, D1D, NE);
   auto y = Reshape(y_.ReadWrite(), D1D, D1D, NE);

   // MFEM_FORALL(e, NE, {
   for (int e = 0; e < NE; e++)
   {
      constexpr int max_D1D = T_D1D ? T_D1D : MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : MAX_Q1D;

      // loop over quadrature points
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            double u_q = 0.0;
            double du_dX_q[2] = {0.0};
            for (int ix = 0; ix < D1D; ix++)
            {
               for (int iy = 0; iy < D1D; iy++)
               {
                  u_q += u(ix, iy, e) * v1d(qx, ix) * v1d(qy, iy);
                  du_dX_q[0] += u(ix, iy, e) * dv1d_dX(qx, ix) * v1d(qy, iy);
                  du_dX_q[1] += u(ix, iy, e) * v1d(qx, ix) * dv1d_dX(qy, iy);
               }
            }

            // du_dx_q = invJ^T * du_dX_q
            //         = (adjJ^T * du_dX_q) / detJ
            double J_q[2][2] = {{J(qx, qy, 0, 0, e),
                                 J(qx, qy, 0, 1, e)}, // J_q[0][0], J_q[0][1]
                                {J(qx, qy, 1, 0, e),
                                 J(qx, qy, 1, 1, e)}}; // J_q[1][0], J_q[1][1]

            double detJ_q = (J_q[0][0] * J_q[1][1]) - (J_q[0][1] * J_q[1][0]);

            double adjJ[2][2] = {{J_q[1][1], -J_q[0][1]},
                                 {-J_q[1][0], J_q[0][0]}};

            double du_dx_q[2]
               = {(adjJ[0][0] * du_dX_q[0] + adjJ[1][0] * du_dX_q[1]) / detJ_q,
                  (adjJ[0][1] * du_dX_q[0] + adjJ[1][1] * du_dX_q[1]) / detJ_q};

            // call Qfunction
            qfunc_output_type output = qf(u_q, du_dx_q);

            double f0_X = output.f0 * detJ_q;

            // f1_X = invJ * f1 * detJ
            //      = adjJ * f1
            double f1_X[2] = {
               adjJ[0][0] * output.f1[0] + adjJ[0][1] * output.f1[1],
               adjJ[1][0] * output.f1[0] + adjJ[1][1] * output.f1[1],
            };

            for (int ix = 0; ix < D1D; ix++)
            {
               for (int iy = 0; iy < D1D; iy++)
               {
                  // accumulate v * f0 + dot(dv_dx, f1)
                  y(ix, iy, e) += (f0_X * v1d(qx, ix) * v1d(qy, iy)
                                   + f1_X[0] * dv1d_dX(qx, ix) * v1d(qy, iy)
                                   + f1_X[1] * dv1d_dX(qy, iy) * v1d(qx, ix))
                                  * W(qx, qy);
               }
            }
         }
      }
   }
   // });
}

template<typename qfunc_type>
QFunctionIntegrator<qfunc_type>::QFunctionIntegrator(qfunc_type f,
                                                     const IntegrationRule *ir)
   : GenericIntegrator(ir), maps(nullptr), geom(nullptr), qf(f)
{}

template<typename qfunc_type>
void QFunctionIntegrator<qfunc_type>::Setup(const FiniteElementSpace &fes)
{
   // Assuming the same element type
   fespace = &fes;
   Mesh *mesh = fes.GetMesh();
   if (mesh->GetNE() == 0)
   {
      return;
   }
   const FiniteElement &el = *fes.GetFE(0);
   ElementTransformation *T = mesh->GetElementTransformation(0);
   const IntegrationRule *ir = nullptr;
   if (!IntRule)
   {
      IntRule = &IntRules.Get(el.GetGeomType(), el.GetOrder() * 2);
   }
   ir = IntRule;

   dim = mesh->Dimension();
   ne = fes.GetMesh()->GetNE();
   nq = ir->GetNPoints();
   geom = mesh->GetGeometricFactors(*ir,
                                    GeometricFactors::COORDINATES
                                       | GeometricFactors::JACOBIANS);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   //    pa_data.SetSize(ne * nq, Device::GetDeviceMemoryType());

   W.SetSize(nq, Device::GetDeviceMemoryType());
   W.GetMemory().CopyFrom(ir->GetWeights().GetMemory(), nq);

   // J.SetSize(ne * nq, Device::GetDeviceMemoryType());
   J = geom->J;
}

template<typename qfunc_type>
void QFunctionIntegrator<qfunc_type>::Apply(const Vector &x, Vector &y) const
{
   Apply2D<0, 0, qfunc_type>(
      dim, dofs1D, quad1D, ne, maps->B, maps->G, J, W, x, qf, y);
}

} // namespace mfem