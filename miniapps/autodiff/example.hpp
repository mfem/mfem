// Shared implementation ex71p/ex71 for the AD integrands and the manually
// implemented integrators

#ifndef ADEXAMPLE_HPP
#define ADEXAMPLE_HPP

#include "mfem.hpp"
#include "adnonlininteg.hpp"
#include <memory>
#include <iostream>
#include <fstream>

namespace mfem
{

///Example: Implementation of the energy and the residual
/// for p-Laplacian problem. Both, the energy and the residual
/// are evaluated at the integration points for PDE parameters
/// vparam and state fields (derivatives with respect to x,y,z
/// and primal field) stored in vector uu.
template<typename DType, typename MVType>
class MyQFunctorJ
{
public:
   ///The operator returns the energy for the  p-Laplacian problem.
   /// The input parameters vparam are: vparam[0] - the p-Laplacian
   /// power, vparam[1] small value ensuring the exsitance of an unique
   /// solution, and vparam[2] - the distributed extenal input to the
   /// equation.
   DType operator()(const Vector &vparam, MVType &uu)
   {
      double pp = vparam[0];
      double ee = vparam[1];
      double ff = vparam[2];

      DType u = uu[3];
      DType norm2 = uu[0] * uu[0] + uu[1] * uu[1] + uu[2] * uu[2];

      DType rez = pow(ee * ee + norm2, pp / 2.0) / pp - ff * u;
      return rez;
   }

   ///The operator returns the first derivative of the energy with respect
   /// to all state variables. These are set in vector uu and consist of the
   /// derivatives with respect to x,y,z and the primal field. The derivative
   /// is stored in vector rr with length equal to the length of vector uu.
   void operator()(const Vector &vparam, MVType &uu, MVType &rr)
   {
      double pp = vparam[0];
      double ee = vparam[1];
      double ff = vparam[2];

      DType norm2 = uu[0] * uu[0] + uu[1] * uu[1] + uu[2] * uu[2];
      DType tvar = pow(ee * ee + norm2, (pp - 2.0) / 2.0);

      rr[0] = tvar * uu[0];
      rr[1] = tvar * uu[1];
      rr[2] = tvar * uu[2];
      rr[3] = -ff;
   }
};

///Defines AD class  pLapIntegrandTJ utilized for the automatic
/// evaluation of the Hessian of the energy of the p-Laplacian.
/// In general the length of the residual vector rr is not know
/// in advance and therefore is provided explicitly as a second
/// template argument in the definition of the class. The
/// evaluation of the Hessian is based on the functor class MyQFunctorJ.
typedef ADQFunctionTJ<MyQFunctorJ, 4> pLapIntegrandTJ;

///Defines template class (functor) for evaluating the energy
/// of the p-Laplacian problem. The input parameters vparam are:
/// vparam[0] - the p-Laplacian power, vparam[1] small value
/// ensuring exsitance of an unique solution, and vparam[2] -
/// the distributed extenal input to the PDE.
template<typename DType, typename MVType>
class MyQFunctorH
{
public:
   ///Returns the energy of a  p-Laplacian for state field input
   /// provided in vector uu and parameters provided in vector
   /// vparam.
   DType operator()(const Vector &vparam, MVType &uu)
   {
      double pp = vparam[0];
      double ee = vparam[1];
      double ff = vparam[2];

      DType u = uu[3];
      DType norm2 = uu[0] * uu[0] + uu[1] * uu[1] + uu[2] * uu[2];

      DType rez = pow(ee * ee + norm2, pp / 2.0) / pp - ff * u;
      return rez;
   }
};

///Defines class pLapIntegrandTH for automatic evaluation
/// of the first and second derivatives of the energy for
/// a p-Laplacian problem. The energy is encoded in the
/// operator()(...) of the template MyQFunctorH class.
typedef ADQFunctionTH<MyQFunctorH> pLapIntegrandTH;

//comment the line below in order to use
//pLapIntegrandTJ for differentiation
//the user interface for both TH and TJ versions
//is exactly the same
//#define USE_ADH


///Implements integrator for a p-Laplacian problem.
/// The integrator is based on a class QFunction utilized for
/// evaluating the energy, the first derivative and the Hessian
/// of the energy. QFunction shold be replaced with pLapIntegrandTH
/// or pLapIntegrandTJ.
template<class QFunction>
class pLaplaceAD : public NonlinearFormIntegrator
{
protected:
   Coefficient *pp;
   Coefficient *coeff;
   Coefficient *load;

   QFunction qint;

public:
   pLaplaceAD()
   {
      coeff = nullptr;
      pp = nullptr;
   }

   pLaplaceAD(Coefficient &pp_) : pp(&pp_), coeff(nullptr), load(nullptr) {}

   pLaplaceAD(Coefficient &pp_, Coefficient &q, Coefficient &ld_)
      : pp(&pp_), coeff(&q), load(&ld_)
   {}

   virtual ~pLaplaceAD() {}

   virtual double GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &trans,
                                   const Vector &elfun) override
   {
      double energy = 0.0;
      int ndof = el.GetDof();
      int ndim = el.GetDim();
      int spaceDim = trans.GetSpaceDim();
      bool square = (ndim == spaceDim);
      const IntegrationRule *ir = NULL;
      int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
      ir = &IntRules.Get(el.GetGeomType(), order);

      Vector shapef(ndof);
      DenseMatrix dshape_iso(ndof, ndim);
      DenseMatrix dshape_xyz(ndof, spaceDim);
      Vector grad(spaceDim);

      Vector vparam(3); //[power, epsilon, load]
      Vector uu(4);     //[diff_x,diff_y,diff_z,u]

      uu = 0.0;
      vparam[0] = 2.0;  //default power
      vparam[1] = 1e-8; //default epsilon
      vparam[2] = 1.0;  //default load

      double w;
      double detJ;

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         w = trans.Weight();
         detJ = (square ? w : w * w);
         w = ip.weight * w;

         el.CalcDShape(ip, dshape_iso);
         el.CalcShape(ip, shapef);
         // AdjugateJacobian = / adj(J),         if J is square
         //                    \ adj(J^t.J).J^t, otherwise
         Mult(dshape_iso, trans.AdjugateJacobian(), dshape_xyz);
         // dshape_xyz should be divided by detJ for obtaining the real value
         // calculate the gradient
         dshape_xyz.MultTranspose(elfun, grad);

         // set the power
         if (pp != nullptr)
         {
            vparam[0] = pp->Eval(trans, ip);
         }

         // set the coefficient ensuring positiveness of the tangent matrix
         if (coeff != nullptr)
         {
            vparam[1] = coeff->Eval(trans, ip);
         }
         // add the contribution from the load
         if (load != nullptr)
         {
            vparam[2] = load->Eval(trans, ip);
         }
         // fill the values of vector uu
         for (int jj = 0; jj < spaceDim; jj++)
         {
            uu[jj] = grad[jj] / detJ;
         }
         uu[3] = shapef * elfun;

         energy = energy + w * (qint.QFunction(vparam, uu));
      }
      return energy;
   }

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &trans,
                                      const Vector &elfun,
                                      Vector &elvect) override
   {
      int ndof = el.GetDof();
      int ndim = el.GetDim();
      int spaceDim = trans.GetSpaceDim();
      const IntegrationRule *ir = NULL;
      int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
      ir = &IntRules.Get(el.GetGeomType(), order);

      Vector shapef(ndof);
      DenseMatrix dshape_iso(ndof, ndim);
      DenseMatrix dshape_xyz(ndof, spaceDim);
      Vector lvec(ndof);
      elvect.SetSize(ndof);
      elvect = 0.0;

      DenseMatrix B(ndof, 4); //[diff_x,diff_y,diff_z, shape]
      Vector vparam(3);       //[power, epsilon, load]
      Vector uu(4);           //[diff_x,diff_y,diff_z,u]
      Vector du(4);
      B = 0.0;
      uu = 0.0;
      //initialize the parameters - keep the same order
      //utilized in the pLapIntegrator definition
      vparam[0] = 2.0;  //default power
      vparam[1] = 1e-8; //default epsilon
      vparam[2] = 1.0;  //default load

      double w;

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         w = trans.Weight();
         //detJ = (square ? w : w*w);
         w = ip.weight * w;

         el.CalcDShape(ip, dshape_iso);
         el.CalcShape(ip, shapef);
         Mult(dshape_iso, trans.InverseJacobian(), dshape_xyz);

         // set the matrix B
         for (int jj = 0; jj < spaceDim; jj++)
         {
            B.SetCol(jj, dshape_xyz.GetColumn(jj));
         }
         B.SetCol(3, shapef);

         // set the power
         if (pp != nullptr)
         {
            vparam[0] = pp->Eval(trans, ip);
         }
         // set the coefficient ensuring positiveness of the tangent matrix
         if (coeff != nullptr)
         {
            vparam[1] = coeff->Eval(trans, ip);
         }
         // add the contribution from the load
         if (load != nullptr)
         {
            vparam[2] = load->Eval(trans, ip);
         }

         // calculate uu
         B.MultTranspose(elfun, uu);
         // calculate derivative of the energy with respect to uu
         qint.QFunctionDU(vparam, uu, du);

         B.Mult(du, lvec);
         elvect.Add(w, lvec);
      } // end integration loop
   }

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &trans,
                                    const Vector &elfun,
                                    DenseMatrix &elmat) override
   {
      int ndof = el.GetDof();
      int ndim = el.GetDim();
      int spaceDim = trans.GetSpaceDim();
      const IntegrationRule *ir = NULL;
      int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
      ir = &IntRules.Get(el.GetGeomType(), order);

      Vector shapef(ndof);
      DenseMatrix dshape_iso(ndof, ndim);
      DenseMatrix dshape_xyz(ndof, spaceDim);
      elmat.SetSize(ndof, ndof);
      elmat = 0.0;

      DenseMatrix B(ndof, 4); // [diff_x,diff_y,diff_z, shape]
      DenseMatrix A(ndof, 4);
      Vector vparam(3); // [power, epsilon, load]
      Vector uu(4);     // [diff_x,diff_y,diff_z,u]
      DenseMatrix duu(4, 4);
      B = 0.0;
      uu = 0.0;
      // initialize the parameters - keep the same order
      // utilized in the pLapIntegrator definition
      vparam[0] = 2.0;  // default power
      vparam[1] = 1e-8; // default epsilon
      vparam[2] = 1.0;  // default load

      double w;

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         w = trans.Weight();
         w = ip.weight * w;

         el.CalcDShape(ip, dshape_iso);
         el.CalcShape(ip, shapef);
         Mult(dshape_iso, trans.InverseJacobian(), dshape_xyz);

         // set the matrix B
         for (int jj = 0; jj < spaceDim; jj++)
         {
            B.SetCol(jj, dshape_xyz.GetColumn(jj));
         }
         B.SetCol(3, shapef);

         // set the power
         if (pp != nullptr)
         {
            vparam[0] = pp->Eval(trans, ip);
         }
         // set the coefficient ensuring positiveness of the tangent matrix
         if (coeff != nullptr)
         {
            vparam[1] = coeff->Eval(trans, ip);
         }
         // add the contribution from the load
         if (load != nullptr)
         {
            vparam[2] = load->Eval(trans, ip);
         }

         // calculate uu
         B.MultTranspose(elfun, uu);
         // calculate derivative of the energy with respect to uu
         qint.QFunctionDD(vparam, uu, duu);

         Mult(B, duu, A);
         AddMult_a_ABt(w, A, B, elmat);

      } // end integration loop
   }
};

///Implements hand-coded integrator for a p-Laplacian problem.
/// Utilized as alternative for the  pLaplaceAD class based on
/// automatic differentiation.
class pLaplace : public NonlinearFormIntegrator
{
protected:
   Coefficient *pp;
   Coefficient *coeff;
   Coefficient *load;

public:
   pLaplace()
   {
      coeff = nullptr;
      pp = nullptr;
   }

   pLaplace(Coefficient &pp_) : pp(&pp_), coeff(nullptr), load(nullptr) {}

   pLaplace(Coefficient &pp_, Coefficient &q, Coefficient &ld_)
      : pp(&pp_), coeff(&q), load(&ld_)
   {}

   virtual ~pLaplace() {}

   virtual double GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &trans,
                                   const Vector &elfun) override
   {
      double energy = 0.0;
      int ndof = el.GetDof();
      int ndim = el.GetDim();
      int spaceDim = trans.GetSpaceDim();
      bool square = (ndim == spaceDim);
      const IntegrationRule *ir = NULL;
      int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
      ir = &IntRules.Get(el.GetGeomType(), order);

      Vector shapef(ndof);
      DenseMatrix dshape_iso(ndof, ndim);
      DenseMatrix dshape_xyz(ndof, spaceDim);
      Vector grad(spaceDim);

      double w;
      double detJ;
      double nrgrad2;
      double ppp = 2.0;
      double eee = 0.0;

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         w = trans.Weight();
         detJ = (square ? w : w * w);
         w = ip.weight * w;

         el.CalcDShape(ip, dshape_iso);
         el.CalcShape(ip, shapef);
         // AdjugateJacobian = / adj(J),         if J is square
         //                    \ adj(J^t.J).J^t, otherwise
         Mult(dshape_iso, trans.AdjugateJacobian(), dshape_xyz);
         // dshape_xyz should be divided by detJ for obtaining the real value
         // calculate the gradient
         dshape_xyz.MultTranspose(elfun, grad);
         nrgrad2 = grad * grad / (detJ * detJ);

         // set the power
         if (pp != nullptr)
         {
            ppp = pp->Eval(trans, ip);
         }

         // set the coefficient ensuring positiveness of the tangent matrix
         if (coeff != nullptr)
         {
            eee = coeff->Eval(trans, ip);
         }

         energy = energy + w * std::pow(nrgrad2 + eee * eee, ppp / 2.0) / ppp;

         // add the contribution from the load
         if (load != nullptr)
         {
            energy = energy - w * (shapef * elfun) * load->Eval(trans, ip);
         }
      }
      return energy;
   }

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &trans,
                                      const Vector &elfun,
                                      Vector &elvect) override
   {
      int ndof = el.GetDof();
      int ndim = el.GetDim();
      int spaceDim = trans.GetSpaceDim();
      bool square = (ndim == spaceDim);
      const IntegrationRule *ir = NULL;
      int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
      ir = &IntRules.Get(el.GetGeomType(), order);

      Vector shapef(ndof);
      DenseMatrix dshape_iso(ndof, ndim);
      DenseMatrix dshape_xyz(ndof, spaceDim);
      Vector grad(spaceDim);
      Vector lvec(ndof);
      elvect.SetSize(ndof);
      elvect = 0.0;

      double w;
      double detJ;
      double nrgrad;
      double aa;
      double ppp = 2.0;
      double eee = 0.0;

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         w = trans.Weight();
         detJ = (square ? w : w * w);
         w = ip.weight * w; //w;

         el.CalcDShape(ip, dshape_iso);
         el.CalcShape(ip, shapef);
         // AdjugateJacobian = / adj(J),         if J is square
         //                    \ adj(J^t.J).J^t, otherwise
         Mult(dshape_iso, trans.AdjugateJacobian(), dshape_xyz);
         // dshape_xyz should be divided by detJ for obtaining the real value

         // calculate the gradient
         dshape_xyz.MultTranspose(elfun, grad);
         nrgrad = grad.Norml2() / detJ;
         // grad is not scaled so far, i.e., grad=grad/detJ

         // set the power
         if (pp != nullptr)
         {
            ppp = pp->Eval(trans, ip);
         }

         // set the coefficient ensuring positiveness of the tangent matrix
         if (coeff != nullptr)
         {
            eee = coeff->Eval(trans, ip);
         }

         aa = nrgrad * nrgrad + eee * eee;
         aa = std::pow(aa, (ppp - 2.0) / 2.0);
         dshape_xyz.Mult(grad, lvec);
         elvect.Add(w * aa / (detJ * detJ), lvec);

         // add loading
         if (load != nullptr)
         {
            elvect.Add(-w * load->Eval(trans, ip), shapef);
         }
      } // end integration loop
   }

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &trans,
                                    const Vector &elfun,
                                    DenseMatrix &elmat) override
   {
      int ndof = el.GetDof();
      int ndim = el.GetDim();
      int spaceDim = trans.GetSpaceDim();
      bool square = (ndim == spaceDim);
      const IntegrationRule *ir = NULL;
      int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
      ir = &IntRules.Get(el.GetGeomType(), order);

      DenseMatrix dshape_iso(ndof, ndim);
      DenseMatrix dshape_xyz(ndof, spaceDim);
      Vector grad(spaceDim);
      Vector lvec(ndof);
      elmat.SetSize(ndof, ndof);
      elmat = 0.0;

      double w;
      double detJ;
      double nrgrad;
      double aa0;
      double aa1;
      double ppp = 2.0;
      double eee = 0.0;

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         w = trans.Weight();
         detJ = (square ? w : w * w);
         w = ip.weight * w;

         el.CalcDShape(ip, dshape_iso);
         // AdjugateJacobian = / adj(J),         if J is square
         //                    \ adj(J^t.J).J^t, otherwise
         Mult(dshape_iso, trans.AdjugateJacobian(), dshape_xyz);
         // dshape_xyz should be divided by detJ for obtaining the real value
         // grad is not scaled so far,i.e., grad=grad/detJ

         //set the power
         if (pp != nullptr)
         {
            ppp = pp->Eval(trans, ip);
         }
         //set the coefficient ensuring positiveness of the tangent matrix
         if (coeff != nullptr)
         {
            eee = coeff->Eval(trans, ip);
         }

         //calculate the gradient
         dshape_xyz.MultTranspose(elfun, grad);
         nrgrad = grad.Norml2() / detJ;
         aa0 = nrgrad * nrgrad + eee * eee;
         aa1 = std::pow(aa0, (ppp - 2.0) / 2.0);
         aa0 = (ppp - 2.0) * std::pow(aa0, (ppp - 4.0) / 2.0);
         dshape_xyz.Mult(grad, lvec);
         w = w / (detJ * detJ);
         AddMult_a_VVt(w * aa0 / (detJ * detJ), lvec, elmat);
         AddMult_a_AAt(w * aa1, dshape_xyz, elmat);

      } // end integration loop
   }
};

} // namespace mfem
#endif
