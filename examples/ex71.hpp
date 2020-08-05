// shared implementation ex71p/ex71 for the AD integrands and
// the handconded integrators


#ifndef EXAMPLE71_H
#define EXAMPLE71_H

#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>


namespace mfem
{


template<typename DType, typename MVType>
class MyQFunctorJ
{
public:
   DType operator()(const mfem::Vector& vparam, MVType& uu)
   {
      double pp=vparam[0];
      double ee=vparam[1];
      double ff=vparam[2];

      DType  u=uu[3];
      DType  norm2=uu[0]*uu[0]+uu[1]*uu[1]+uu[2]*uu[2];

      DType rez= pow(ee*ee+norm2,pp/2.0)/pp-ff*u;
      return rez;
   }

   void operator()(const mfem::Vector& vparam, MVType& uu, MVType& rr)
   {
      double pp=vparam[0];
      double ee=vparam[1];
      double ff=vparam[2];

      DType norm2=uu[0]*uu[0]+uu[1]*uu[1]+uu[2]*uu[2];
      DType tvar=pow(ee*ee+norm2,(pp-2.0)/2.0);

      rr[0]=tvar*uu[0];
      rr[1]=tvar*uu[1];
      rr[2]=tvar*uu[2];
      rr[3]=-ff;

   }
};


typedef ADQFunctionTJ<MyQFunctorJ,4> pLapIntegrandTJ;


template<typename DType, typename MVType>
class MyQFunctorH
{
public:
   DType operator()(const mfem::Vector& vparam, MVType& uu)
   {
      double pp=vparam[0];
      double ee=vparam[1];
      double ff=vparam[2];

      DType  u=uu[3];
      DType  norm2=uu[0]*uu[0]+uu[1]*uu[1]+uu[2]*uu[2];

      DType rez= pow(ee*ee+norm2,pp/2.0)/pp-ff*u;
      return rez;
   }
};

typedef ADQFunctionTH<MyQFunctorH> pLapIntegrandTH;

//comment the line below in order to use
//pLapIntegrandTJ for differentiation
//the user interface for both TH and TJ versions
//is exacly the same
//#define USE_ADH

class pLaplaceAD: public mfem::NonlinearFormIntegrator
{
protected:
   mfem::Coefficient* pp;
   mfem::Coefficient* coeff;
   mfem::Coefficient* load;
#ifdef USE_ADH
   pLapIntegrandTH qint;
#else
   pLapIntegrandTJ qint;
#endif
public:
   pLaplaceAD()
   {
      coeff=nullptr;
      pp=nullptr;
   }

   pLaplaceAD(mfem::Coefficient& pp_):pp(&pp_), coeff(nullptr), load(nullptr)
   {

   }

   pLaplaceAD(mfem::Coefficient &pp_,mfem::Coefficient& q,
              mfem::Coefficient& ld_): pp(&pp_), coeff(&q), load(&ld_)
   {

   }

   virtual ~pLaplaceAD()
   {

   }

   virtual double GetElementEnergy(const mfem::FiniteElement &el,
                                   mfem::ElementTransformation &trans, const mfem::Vector &elfun) override
   {
      double energy=0.0;
      int ndof = el.GetDof();
      int ndim = el.GetDim();
      int spaceDim = trans.GetSpaceDim();
      bool square = (ndim == spaceDim);
      const mfem::IntegrationRule *ir = NULL;
      int order = 2 * trans.OrderGrad(&el) - 1; // correct order?
      ir = &mfem::IntRules.Get(el.GetGeomType(), order);

      mfem::Vector shapef(ndof);
      mfem::DenseMatrix dshape_iso(ndof,ndim);
      mfem::DenseMatrix dshape_xyz(ndof,spaceDim);
      mfem::Vector grad(spaceDim);

      mfem::Vector vparam(3);//[power, epsilon, load]
      mfem::Vector uu(4);//[diff_x,diff_y,diff_z,u]

      uu=0.0;
      vparam[0]=2.0;  //default power
      vparam[1]=1e-8; //default epsilon
      vparam[2]=1.0;  //default load

      double w;
      double detJ;

      for (int i = 0; i < ir -> GetNPoints(); i++)
      {
         const mfem::IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         w = trans.Weight();
         detJ = (square ? w : w*w);
         w = ip.weight *w;

         el.CalcDShape(ip,dshape_iso);
         el.CalcShape(ip,shapef);
         // AdjugateJacobian = / adj(J),         if J is square
         //                    \ adj(J^t.J).J^t, otherwise
         mfem::Mult(dshape_iso, trans.AdjugateJacobian(), dshape_xyz);
         // dshape_xyz should be devided by detJ for obtaining the real value
         // calculate the gradient
         dshape_xyz.MultTranspose(elfun,grad);

         //set the power
         if (pp!=nullptr)
         {
            vparam[0]=pp->Eval(trans,ip);
         }

         //set the coefficient ensuring possitiveness of the tangent matrix
         if (coeff!=nullptr)
         {
            vparam[1]=coeff->Eval(trans,ip);
         }
         //add the contribution from the load
         if (load!=nullptr)
         {
            vparam[2]=load->Eval(trans,ip);
         }
         //fill the values of vector uu
         for (int jj=0; jj<spaceDim; jj++)
         {
            uu[jj]=grad[jj]/detJ;
         }
         uu[3]=shapef*elfun;

         energy = energy + w * (qint.QFunction(vparam,uu));

      }
      return energy;
   }

   virtual void AssembleElementVector(const mfem::FiniteElement & el,
                                      mfem::ElementTransformation & trans,
                                      const mfem::Vector & elfun,
                                      mfem::Vector & elvect) override
   {
      int ndof = el.GetDof();
      int ndim = el.GetDim();
      int spaceDim = trans.GetSpaceDim();
      const mfem::IntegrationRule *ir = NULL;
      int order = 2 * trans.OrderGrad(&el) - 1; // correct order?
      ir = &mfem::IntRules.Get(el.GetGeomType(), order);

      mfem::Vector shapef(ndof);
      mfem::DenseMatrix dshape_iso(ndof,ndim);
      mfem::DenseMatrix dshape_xyz(ndof,spaceDim);
      mfem::Vector lvec(ndof);
      elvect.SetSize(ndof);
      elvect=0.0;

      mfem::DenseMatrix B(ndof,4); //[diff_x,diff_y,diff_z, shape]
      mfem::Vector vparam(3);//[power, epsilon, load]
      mfem::Vector uu(4);//[diff_x,diff_y,diff_z,u]
      mfem::Vector du(4);
      B=0.0;
      uu=0.0;
      //initialize the parameters - keep the same order
      //utilized in the pLapIntegrator definition
      vparam[0]=2.0;  //default power
      vparam[1]=1e-8; //default epsilon
      vparam[2]=1.0;  //default load

      double w;

      for (int i = 0; i < ir -> GetNPoints(); i++)
      {
         const mfem::IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         w = trans.Weight();
         //detJ = (square ? w : w*w);
         w = ip.weight * w;

         el.CalcDShape(ip,dshape_iso);
         el.CalcShape(ip,shapef);
         mfem::Mult(dshape_iso, trans.InverseJacobian(), dshape_xyz);

         //set the matrix B
         for (int jj=0; jj<spaceDim; jj++)
         {
            B.SetCol(jj,dshape_xyz.GetColumn(jj));
         }
         B.SetCol(3,shapef);


         //set the power
         if (pp!=nullptr)
         {
            vparam[0]=pp->Eval(trans,ip);
         }
         //set the coefficient ensuring possitiveness of the tangent matrix
         if (coeff!=nullptr)
         {
            vparam[1]=coeff->Eval(trans,ip);
         }
         //add the contribution from the load
         if (load!=nullptr)
         {
            vparam[2]=load->Eval(trans,ip);
         }

         //calculate uu
         B.MultTranspose(elfun,uu);
         //calculate derivative of the energy with respect to uu
         qint.QFunctionDU(vparam,uu,du);

         B.Mult(du,lvec);
         elvect.Add( w, lvec);
      }// end integration loop
   }

   virtual void AssembleElementGrad(const mfem::FiniteElement & el,
                                    mfem::ElementTransformation & trans,
                                    const mfem::Vector & elfun, mfem::DenseMatrix & elmat) override
   {
      int ndof = el.GetDof();
      int ndim = el.GetDim();
      int spaceDim = trans.GetSpaceDim();
      const mfem::IntegrationRule *ir = NULL;
      int order = 2 * trans.OrderGrad(&el) - 1; // correct order?
      ir = &mfem::IntRules.Get(el.GetGeomType(), order);

      mfem::Vector shapef(ndof);
      mfem::DenseMatrix dshape_iso(ndof,ndim);
      mfem::DenseMatrix dshape_xyz(ndof,spaceDim);
      elmat.SetSize(ndof,ndof);
      elmat=0.0;

      mfem::DenseMatrix B(ndof,4); //[diff_x,diff_y,diff_z, shape]
      mfem::DenseMatrix A(ndof,4);
      mfem::Vector vparam(3);//[power, epsilon, load]
      mfem::Vector uu(4);//[diff_x,diff_y,diff_z,u]
      mfem::DenseMatrix duu(4,4);
      B=0.0;
      uu=0.0;
      //initialize the parameters - keep the same order
      //utilized in the pLapIntegrator definition
      vparam[0]=2.0;  //default power
      vparam[1]=1e-8; //default epsilon
      vparam[2]=1.0;  //default load

      double w;

      for (int i = 0; i < ir -> GetNPoints(); i++)
      {
         const mfem::IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         w = trans.Weight();
         w = ip.weight * w;

         el.CalcDShape(ip,dshape_iso);
         el.CalcShape(ip,shapef);
         mfem::Mult(dshape_iso, trans.InverseJacobian(), dshape_xyz);

         //set the matrix B
         for (int jj=0; jj<spaceDim; jj++)
         {
            B.SetCol(jj,dshape_xyz.GetColumn(jj));
         }
         B.SetCol(3,shapef);


         //set the power
         if (pp!=nullptr)
         {
            vparam[0]=pp->Eval(trans,ip);
         }
         //set the coefficient ensuring possitiveness of the tangent matrix
         if (coeff!=nullptr)
         {
            vparam[1]=coeff->Eval(trans,ip);
         }
         //add the contribution from the load
         if (load!=nullptr)
         {
            vparam[2]=load->Eval(trans,ip);
         }

         //calculate uu
         B.MultTranspose(elfun,uu);
         //calculate derivative of the energy with respect to uu
         qint.QFunctionDD(vparam,uu,duu);

         mfem::Mult(B,duu,A);
         mfem::AddMult_a_ABt(w,A,B,elmat);

      }//end integration loop
   }


};


class pLaplace: public mfem::NonlinearFormIntegrator
{
protected:
   mfem::Coefficient* pp;
   mfem::Coefficient* coeff;
   mfem::Coefficient* load;
public:
   pLaplace()
   {
      coeff=nullptr;
      pp=nullptr;
   }

   pLaplace(mfem::Coefficient& pp_):pp(&pp_), coeff(nullptr), load(nullptr)
   {

   }

   pLaplace(mfem::Coefficient &pp_,mfem::Coefficient& q,
            mfem::Coefficient& ld_): pp(&pp_), coeff(&q), load(&ld_)
   {

   }

   virtual ~pLaplace()
   {

   }

   virtual double GetElementEnergy(const mfem::FiniteElement &el,
                                   mfem::ElementTransformation &trans, const mfem::Vector &elfun) override
   {
      double energy=0.0;
      int ndof = el.GetDof();
      int ndim = el.GetDim();
      int spaceDim = trans.GetSpaceDim();
      bool square = (ndim == spaceDim);
      const mfem::IntegrationRule *ir = NULL;
      int order = 2 * trans.OrderGrad(&el) - 1; // correct order?
      ir = &mfem::IntRules.Get(el.GetGeomType(), order);

      mfem::Vector shapef(ndof);
      mfem::DenseMatrix dshape_iso(ndof,ndim);
      mfem::DenseMatrix dshape_xyz(ndof,spaceDim);
      mfem::Vector grad(spaceDim);

      double w;
      double detJ;
      double nrgrad2;
      double ppp=2.0;
      double eee=0.0;

      for (int i = 0; i < ir -> GetNPoints(); i++)
      {
         const mfem::IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         w = trans.Weight();
         detJ = (square ? w : w*w);
         w = ip.weight *w;

         el.CalcDShape(ip,dshape_iso);
         el.CalcShape(ip,shapef);
         // AdjugateJacobian = / adj(J),         if J is square
         //                    \ adj(J^t.J).J^t, otherwise
         mfem::Mult(dshape_iso, trans.AdjugateJacobian(), dshape_xyz);
         // dshape_xyz should be devided by detJ for obtaining the real value
         // calculate the gradient
         dshape_xyz.MultTranspose(elfun,grad);
         nrgrad2=grad*grad/(detJ*detJ);

         //set the power
         if (pp!=nullptr)
         {
            ppp=pp->Eval(trans,ip);
         }

         //set the coefficient ensuring possitiveness of the tangent matrix
         if (coeff!=nullptr)
         {
            eee=coeff->Eval(trans,ip);
         }

         energy = energy + w * std::pow( nrgrad2 + eee * eee , ppp / 2.0 ) / ppp;

         //add the contribution from the load
         if (load!=nullptr)
         {
            energy = energy - w *  (shapef*elfun) * load->Eval(trans,ip);
         }
      }
      return energy;
   }

   virtual void  AssembleElementVector(const mfem::FiniteElement & el,
                                       mfem::ElementTransformation & trans,
                                       const mfem::Vector & elfun,
                                       mfem::Vector & elvect) override
   {
      int ndof = el.GetDof();
      int ndim = el.GetDim();
      int spaceDim = trans.GetSpaceDim();
      bool square = (ndim == spaceDim);
      const mfem::IntegrationRule *ir = NULL;
      int order = 2 * trans.OrderGrad(&el) - 1; // correct order?
      ir = &mfem::IntRules.Get(el.GetGeomType(), order);

      mfem::Vector shapef(ndof);
      mfem::DenseMatrix dshape_iso(ndof,ndim);
      mfem::DenseMatrix dshape_xyz(ndof,spaceDim);
      mfem::Vector grad(spaceDim);
      mfem::Vector lvec(ndof);
      elvect.SetSize(ndof);
      elvect=0.0;

      double w;
      double detJ;
      double nrgrad;
      double aa;
      double ppp=2.0;
      double eee=0.0;

      for (int i = 0; i < ir -> GetNPoints(); i++)
      {
         const mfem::IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         w = trans.Weight();
         detJ = (square ? w : w*w);
         w = ip.weight * w;//w;

         el.CalcDShape(ip,dshape_iso);
         el.CalcShape(ip,shapef);
         // AdjugateJacobian = / adj(J),         if J is square
         //                    \ adj(J^t.J).J^t, otherwise
         mfem::Mult(dshape_iso, trans.AdjugateJacobian(), dshape_xyz);
         // dshape_xyz should be devided by detJ for obtaining the real value

         //calculate the gradient
         dshape_xyz.MultTranspose(elfun,grad);
         nrgrad=grad.Norml2()/detJ;
         //grad is not scaled so far, i.e., grad=grad/detJ

         //set the power
         if (pp!=nullptr)
         {
            ppp=pp->Eval(trans,ip);
         }

         //set the coefficient ensuring possitiveness of the tangent matrix
         if (coeff!=nullptr)
         {
            eee=coeff->Eval(trans,ip);
         }

         aa = nrgrad * nrgrad + eee * eee;
         aa=std::pow( aa , ( ppp - 2.0 ) / 2.0 );
         dshape_xyz.Mult(grad,lvec);
         elvect.Add( w * aa / ( detJ * detJ ), lvec);


         //add loading
         if (load!=nullptr)
         {
            elvect.Add(-w*load->Eval(trans,ip),shapef);
         }
      }// end integration loop
   }

   virtual void AssembleElementGrad(const mfem::FiniteElement & el,
                                    mfem::ElementTransformation & trans,
                                    const mfem::Vector & elfun, mfem::DenseMatrix & elmat) override
   {
      int ndof = el.GetDof();
      int ndim = el.GetDim();
      int spaceDim = trans.GetSpaceDim();
      bool square = (ndim == spaceDim);
      const mfem::IntegrationRule *ir = NULL;
      int order = 2 * trans.OrderGrad(&el) - 1; // correct order?
      ir = &mfem::IntRules.Get(el.GetGeomType(), order);

      mfem::DenseMatrix dshape_iso(ndof,ndim);
      mfem::DenseMatrix dshape_xyz(ndof,spaceDim);
      mfem::Vector grad(spaceDim);
      mfem::Vector lvec(ndof);
      elmat.SetSize(ndof,ndof);
      elmat=0.0;

      double w;
      double detJ;
      double nrgrad;
      double aa0;
      double aa1;
      double ppp=2.0;
      double eee=0.0;

      for (int i = 0; i < ir -> GetNPoints(); i++)
      {
         const mfem::IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         w = trans.Weight();
         detJ = (square ? w : w*w);
         w = ip.weight * w;

         el.CalcDShape(ip,dshape_iso);
         // AdjugateJacobian = / adj(J),         if J is square
         //                    \ adj(J^t.J).J^t, otherwise
         mfem::Mult(dshape_iso, trans.AdjugateJacobian(), dshape_xyz);
         // dshape_xyz should be devided by detJ for obtaining the real value
         // grad is not scaled so far,i.e., grad=grad/detJ

         //set the power
         if (pp!=nullptr)
         {
            ppp=pp->Eval(trans,ip);
         }
         //set the coefficient ensuring possitiveness of the tangent matrix
         if (coeff!=nullptr)
         {
            eee=coeff->Eval(trans,ip);
         }

         //calculate the gradient
         dshape_xyz.MultTranspose(elfun,grad);
         nrgrad = grad.Norml2() / detJ;
         aa0 = nrgrad * nrgrad + eee * eee;
         aa1 = std::pow( aa0 , ( ppp - 2.0 ) / 2.0 );
         aa0 = ( ppp - 2.0 ) * std::pow(aa0, ( ppp - 4.0 ) / 2.0 );
         dshape_xyz.Mult(grad,lvec);
         w = w / ( detJ * detJ );
         mfem::AddMult_a_VVt( w * aa0 / ( detJ * detJ ), lvec, elmat);
         mfem::AddMult_a_AAt( w * aa1 , dshape_xyz, elmat);

      }//end integration loop
   }
};

}
#endif
