// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MYINTEGRATORS_HPP
#define MYINTEGRATORS_HPP

#include "mfem.hpp"

#include <map>

namespace mfem
{

class MySimpleDiffusionIntegrator: public BilinearFormIntegrator
{
protected:
  
private:
   DenseMatrix dshape, dshapedxt; //dshape = lagre derivert av shape functions, dshapedxt for lagring   
   int dim;

public:

virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);


};


class MyDiffusionIntegrator: public BilinearFormIntegrator
{
protected:
  FunctionCoefficient *Q; 
private:
   DenseMatrix dshape, dshapedxt;   
   int dim;

public:
   MyDiffusionIntegrator(const IntegrationRule *ir = nullptr)
      : BilinearFormIntegrator(ir),
        Q(NULL) { }

   /// Construct a diffusion integrator with a function coefficient q
   MyDiffusionIntegrator(FunctionCoefficient &q, const IntegrationRule *ir = nullptr)
      : BilinearFormIntegrator(ir),
        Q(&q) { }


virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);


};

//not working
class MyNitscheBilinIntegrator : public BilinearFormIntegrator
{
protected: 
   Coefficient *Q;
   real_t lambda;

   Vector shape1, dshape1dn, nor, nh, ni;
   DenseMatrix jmat, dshape1, adjJ;


private:
   int dim;


public:
   MyNitscheBilinIntegrator(Coefficient &q, const real_t k)
      : Q(&q), lambda(k) { }
   using BilinearFormIntegrator::AssembleFaceMatrix;

   void AssembleFaceMatrix(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override;
};


class MyVectorDiffusionIntegrator : public BilinearFormIntegrator
{
protected:
   Coefficient *Q = NULL;
   VectorCoefficient *VQ = NULL;
   MatrixCoefficient *MQ = NULL;

   int dim, sdim, ne, dofs1D, quad1D;

private:
   DenseMatrix dshape, dshapedxt, pelmat;
   int vdim = -1;
   DenseMatrix mcoeff;
   Vector vcoeff;


public:
   MyVectorDiffusionIntegrator() { }

   /** \brief Integrator with unit coefficient for caller-specified vector
       dimension.

       If the vector dimension does not match the true dimension of the space,
       the resulting element matrix will be mathematically invalid. */
   MyVectorDiffusionIntegrator(int vector_dimension)
      : vdim(vector_dimension) { }

   MyVectorDiffusionIntegrator(Coefficient &q)
      : Q(&q) { }

   MyVectorDiffusionIntegrator(Coefficient &q, const IntegrationRule *ir)
      : BilinearFormIntegrator(ir), Q(&q) { }

   /** \brief Integrator with scalar coefficient for caller-specified vector
       dimension.

       The element matrix is block-diagonal with \c vdim copies of the element
       matrix integrated with the \c Coefficient.

       If the vector dimension does not match the true dimension of the space,
       the resulting element matrix will be mathematically invalid. */
   MyVectorDiffusionIntegrator(Coefficient &q, int vector_dimension)
      : Q(&q), vdim(vector_dimension) { }

   MyVectorDiffusionIntegrator(VectorCoefficient &vq)
      : VQ(&vq), vdim(vq.GetVDim()) { }

   /** \brief Integrator with \c MatrixCoefficient. The vector dimension of the
       \c FiniteElementSpace is assumed to be the same as the dimension of the
       \c Matrix.

       The element matrix is populated in each block. Each block is integrated
       with coefficient $q_{ij}$.

       If the vector dimension does not match the true dimension of the space,
       the resulting element matrix will be mathematically invalid. */
   MyVectorDiffusionIntegrator(MatrixCoefficient& mq)
      : MQ(&mq), vdim(mq.GetVDim()) { }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

};


class GhostPenaltyIntegrator:public BilinearFormIntegrator
{
public:
    GhostPenaltyIntegrator(double penal_=1.0):penal(penal_)
    {

    }

    virtual
        ~GhostPenaltyIntegrator()
    {

    }

    virtual void AssembleFaceMatrix(const FiniteElement &fe1,
                                    const FiniteElement &fe2,
                                    FaceElementTransformations &Tr,
                                    DenseMatrix &elmat);
private:


    void Shape(Vector& xx, int order, Vector& sh)
    {
        if(xx.Size()==1){
            Shape1D(xx[0],order,sh);
        }else
        if(xx.Size()==2){
            Shape2D(xx[0],xx[1],order,sh);
        }else
        if(xx.Size()==3){
            Shape3D(xx[0],xx[1],xx[2],order,sh);
        }
    }

    void Shape1D(double x, int order, Vector& sh){
        sh.SetSize(order+1);
        sh[0]=1.0;
        for(int i=0;i<order;i++){sh[i+1]=sh[i]*x;}
    }

    void Shape2D(double x, double y, int order, Vector& sh)
    {
        Vector shx(order+1); Shape1D(x,order,shx);
        Vector shy(order+1); Shape1D(y,order,shy);
        sh.SetSize((order+1)*(order+2)/2);
        int k=0;
        for(int i=0;i<order+1;i++){
            for(int j=0;j<order+1;j++){
                if((i+j)<(order+1)){
                    sh[k]=shx[i]*shy[j];
                    k=k+1;
                }
            }
        }
    }

    void Shape3D(double x, double y, double z, int order, Vector& sh)
    {
        Vector shx(order+1); Shape1D(x,order,shx);
        Vector shy(order+1); Shape1D(y,order,shy);
        Vector shz(order+1); Shape1D(z,order,shz);
        sh.SetSize((order+1)*(order+2)*(order+3)/6);
        int p=0;
        for(int i=0;i<order+1;i++){
            for(int j=0;j<order+1;j++){
                for(int k=0;k<order+1;k++){
                    if((i+j+k)<(order+1)){
                        sh[p]=shx[i]*shy[j]*shz[k];
                        p=p+1;
                    }
                }
            }
        }
    }

    double penal;
};





}
#endif
