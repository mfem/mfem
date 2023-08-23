// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_CUT_INTEGRATORS_HPP
#define MFEM_CUT_INTEGRATORS_HPP

#include "mfem.hpp"
#include "marking.hpp"

namespace mfem
{

/// Cut integrator for evaluating a volume and its gradients
/// with respect to displacements of the mesh nodes
class CutVolLagrangeIntegrator: public NonlinearFormIntegrator
{

public:
    CutVolLagrangeIntegrator(int io=2)
    {
        coeff=nullptr;
        cint=nullptr;
        int_order=io;
    }

    virtual
    ~CutVolLagrangeIntegrator()
    {

    }

    void SetCutIntegration(CutIntegrationRules* cint_)
    {
        cint=cint_;
    }

    void SetCoefficient(Coefficient& cf){
        coeff=&cf;
    }

    void SetIntOrder(int io=2)
    {
        int_order=io;
    }

    /// Perform the local action of the NonlinearFormIntegrator -
    /// evaluates the gradients with respect to displacements of
    /// the mesh nodes
    virtual void AssembleElementVector(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       const Vector &elfun, Vector &elvect);


    /// Compute the local energy, i.e., the volume
    virtual double GetElementEnergy(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        const Vector &elfun);

private:
    CutIntegrationRules* cint; //cut integration rulle
    Coefficient* coeff;
    int int_order;
};

/// Ghost-penalty volume integrator
class VolGhostPenaltyIntegrator:public BilinearFormIntegrator
{
public:
    VolGhostPenaltyIntegrator(double penal_=1.0):penal(penal_)
    {

    }

    virtual
    ~VolGhostPenaltyIntegrator()
    {

    }

    virtual
    void AssembleFaceMatrix(const FiniteElement &fe1,
                            const FiniteElement &fe2,
                            FaceElementTransformations &Tr,
                            DenseMatrix &elmat);



protected:
    double penal;

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
        }}
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
        }}}
    }

    void Shape(Vector& xx, int dim, int order, Vector& sh)
    {
        if(dim==1){
            Shape1D(xx[0],order,sh);
        }else
        if(dim==2){
            Shape2D(xx[0],xx[1],order,sh);
        }else
        if(dim==3){
            Shape3D(xx[0],xx[1],xx[2],order,sh);
        }
    }

};

}
#endif
