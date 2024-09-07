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
#include "cut_marking.hpp"

#include <map>

namespace mfem
{


class ScalarErrorIntegrator: public NonlinearFormIntegrator
{
private:
    Coefficient* coeff;
public:
    ScalarErrorIntegrator(Coefficient& coef_)
    {
        coeff=&coef_;
    }
    virtual real_t GetElementEnergy(const FiniteElement &el,
                                    ElementTransformation &Tr,
                                    const Vector &elfun) override
    {
        real_t rez=0.0;
        real_t r1;
        real_t r2;
        const int dof = el.GetDof();
        Vector sh(dof);

        const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el);


        for (int i = 0; i < ir -> GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            Tr.SetIntPoint(&ip);
            el.CalcPhysShape(Tr,sh);

            r1=0.0;
            for(int j=0;j<dof;j++){
                r1=r1+elfun[j]*sh[j];
            }

            r2=coeff->Eval(Tr,ip);

            rez=rez+ip.weight*Tr.Weight()*(r1-r2)*(r1-r2);
        }

        return rez;
    }

    const IntegrationRule& GetRule(const FiniteElement &trial_fe,
                                   const FiniteElement &test_fe)
    {
        int order;
        if (trial_fe.Space() == FunctionSpace::Pk)
        {
            order = trial_fe.GetOrder() + test_fe.GetOrder() - 2;
        }
        else
        {
            // order = 2*el.GetOrder() - 2;  // <-- this seems to work fine too
            order = trial_fe.GetOrder() + test_fe.GetOrder() + trial_fe.GetDim() - 1;
        }

        if (trial_fe.Space() == FunctionSpace::rQk)
        {
            return RefinedIntRules.Get(trial_fe.GetGeomType(), order);
        }
        return IntRules.Get(trial_fe.GetGeomType(), order);
    }
};

class CutScalarErrorIntegrator: public ScalarErrorIntegrator
{
private:
    Array<int>* el_marks;
    CutIntegrationRules* irules;
public:
    CutScalarErrorIntegrator(Coefficient& coef_,
                             Array<int>* marks,
                             CutIntegrationRules* cut_int):ScalarErrorIntegrator(coef_)
    {
        el_marks=marks;
        irules=cut_int;
    }

    virtual real_t GetElementEnergy(const FiniteElement &el,
                                    ElementTransformation &Tr,
                                    const Vector &elfun) override
    {
        if((*el_marks)[Tr.ElementNo]==ElementMarker::OUTSIDE)
        {
            return real_t(0.0);
        }
        else if((*el_marks)[Tr.ElementNo]==ElementMarker::INSIDE)
        {

            //use standard integration rule
            SetIntRule(nullptr);
            return ScalarErrorIntegrator::GetElementEnergy(el,Tr,elfun);
        }
        else
        {
            //cut integration
            IntegrationRule ir;
            irules->GetVolumeIntegrationRule(Tr,ir);
            SetIntRule(&ir);
            return ScalarErrorIntegrator::GetElementEnergy(el,Tr,elfun);
        }
    }
};



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


class CutDiffusionIntegrator: public BilinearFormIntegrator
{
private:
    DiffusionIntegrator* dint;
    DiffusionIntegrator* dint2;

    Array<int>* el_marks;
    CutIntegrationRules* irules;

    bool use_weak_material;
    ConstantCoefficient epsilon;

public:
    CutDiffusionIntegrator(Coefficient& q,
                           Array<int>* marks,
                           CutIntegrationRules* cut_int,
                           bool use_weak_mat=false,
                           double eps=1e-6):
                            use_weak_material(use_weak_mat),epsilon(eps)
    {
        el_marks=marks;
        irules=cut_int;
        dint=new DiffusionIntegrator(q);
        if(use_weak_material){
            dint2 = new DiffusionIntegrator(epsilon);
        }else{
            dint2 = nullptr;
        }
    }

    ~CutDiffusionIntegrator()
    {
        delete dint;
        delete dint2;
    }

    virtual void AssembleElementMatrix(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat) override
    {

        if((*el_marks)[Trans.ElementNo]==ElementMarker::OUTSIDE)
        {
            elmat.SetSize(el.GetDof());
            elmat=0.0;
        }
        else if((*el_marks)[Trans.ElementNo]==ElementMarker::INSIDE)
        {
            //use standard integration rule
            dint->SetIntRule(nullptr);
            dint->AssembleElementMatrix(el,Trans,elmat);
        }
        else
        {
            //use cut integration
            IntegrationRule ir;
            irules->GetVolumeIntegrationRule(Trans,ir);
            dint->SetIntRule(&ir);
            dint->AssembleElementMatrix(el,Trans,elmat);

            if(use_weak_material){
                DenseMatrix elmat2;
                dint2->SetIntRule(&ir);
                dint2->AssembleElementMatrix(el,Trans,elmat2);
                elmat.Add(-1,elmat2);

                dint2->SetIntRule(nullptr);
                dint2->AssembleElementMatrix(el,Trans,elmat2);
                elmat.Add(1,elmat2);
            }
        }
    }
};




class CutGhostPenaltyIntegrator:public BilinearFormIntegrator
{
private:    
    GhostPenaltyIntegrator* dint;
    Array<int>* face_marks;

public:


    /// Constructor takes as arguments the ghost penalty and face marks.
    /// The ghost penalty is applied only on faces marked as
    /// ElementMarker::FaceType::GHOSTP.
    CutGhostPenaltyIntegrator(double penal_, Array<int>* marks)
    {
        face_marks=marks;
        dint=new GhostPenaltyIntegrator(penal_);
    }

    /// Destructor
    virtual
    ~CutGhostPenaltyIntegrator()
    {
        delete dint;
    }


    virtual void AssembleFaceMatrix(const FiniteElement &fe1,
                                    const FiniteElement &fe2,
                                    FaceElementTransformations &Trans,
                                    DenseMatrix &elmat) override
    {
        if(Trans.Elem2No<0)
        {
            elmat.SetSize(fe1.GetDof());
            elmat=0.0;
            return;
        }

        if(((*face_marks)[Trans.ElementNo])==ElementMarker::FaceType::GHOSTP)
        {
            dint->AssembleFaceMatrix(fe1,fe2,Trans,elmat);
        }
        else
        {
            const int ndofs1=fe1.GetDof();
            const int ndofs2=fe2.GetDof();
            int ndofs=ndofs1+ndofs2;
            elmat.SetSize(ndofs); elmat=0.0;
        }
    }

};


class CutMassIntegrator: public BilinearFormIntegrator
{
private:
    MassIntegrator* dint;

    Array<int>* el_marks;
    CutIntegrationRules* irules;
public:
    CutMassIntegrator(Coefficient& q,
                           Array<int>* marks,
                           CutIntegrationRules* cut_int)
    {
        el_marks=marks;
        irules=cut_int;
        dint=new MassIntegrator(q);
    }

    ~CutMassIntegrator()
    {
        delete dint;
    }

    virtual void AssembleElementMatrix(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat) override
    {

        if((*el_marks)[Trans.ElementNo]==ElementMarker::OUTSIDE)
        {
            elmat.SetSize(el.GetDof());
            elmat=0.0;
        }
        else if((*el_marks)[Trans.ElementNo]==ElementMarker::INSIDE)
        {
            //use standard integration rule
            dint->SetIntRule(nullptr);
            dint->AssembleElementMatrix(el,Trans,elmat);
        }
        else
        {
            //use cut integration
            IntegrationRule ir;
            irules->GetVolumeIntegrationRule(Trans,ir);
            dint->SetIntRule(&ir);
            dint->AssembleElementMatrix(el,Trans,elmat);

        }
    }
};

// TODO: move this to other file to not mix bilinar and linear integrator
class CutDomainLFIntegrator : public LinearFormIntegrator
{
    private:
    DomainLFIntegrator* dint;
    Array<int>* el_marks;
    CutIntegrationRules* irules;
public:
   /// Constructs a domain integrator with a given Coefficient
    CutDomainLFIntegrator(Coefficient& q,
                           Array<int>* marks,
                           CutIntegrationRules* cut_int)
    {
        el_marks=marks;
        irules=cut_int;
        dint=new DomainLFIntegrator(q);
    }
    ~CutDomainLFIntegrator()
    {
        delete dint;
    }

    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       Vector &elvect) override
    {

        if((*el_marks)[Trans.ElementNo]==ElementMarker::OUTSIDE)
        {
            elvect.SetSize(el.GetDof());
            elvect=0.0;
        }
        else if((*el_marks)[Trans.ElementNo]==ElementMarker::INSIDE)
        {
            //use standard integration rule
            dint->SetIntRule(nullptr);
            dint->AssembleRHSElementVect(el,Trans,elvect);

        }
        else
        {
            //use cut integration
            IntegrationRule ir;
            irules->GetVolumeIntegrationRule(Trans,ir);

            dint->SetIntRule(&ir);
            dint->AssembleRHSElementVect(el,Trans,elvect);
        }
    }
};

class UnfittedBoundaryLFIntegrator : public LinearFormIntegrator
{
   Vector shape;
   Coefficient &Q;
   Vector sweights;
public:
   UnfittedBoundaryLFIntegrator(Coefficient &QG)
      : Q(QG) { }


   /** Given a particular boundary Finite Element and a transformation (Tr)
       computes the element boundary vector, elvect. */
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);


   virtual void SetSurfaceWeights(Vector surface_weights) { sweights = surface_weights; }

};

class CutUnfittedBoundaryLFIntegrator: public LinearFormIntegrator
{
    private:
    UnfittedBoundaryLFIntegrator * dint;
    Array<int>* el_marks;
    CutIntegrationRules* irules;
public:
   /// Constructs a domain integrator with a given Coefficient
    CutUnfittedBoundaryLFIntegrator(Coefficient& q,
                           Array<int>* marks,
                           CutIntegrationRules* cut_int)
    {
        el_marks=marks;
        irules=cut_int;
        dint=new UnfittedBoundaryLFIntegrator(q);
    }
    ~CutUnfittedBoundaryLFIntegrator()
    {
        delete dint;
    }

    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       Vector &elvect) override
    {

        if((*el_marks)[Trans.ElementNo]==ElementMarker::OUTSIDE)
        {
            elvect.SetSize(el.GetDof());
            elvect=0.0;
        }
        else if((*el_marks)[Trans.ElementNo]==ElementMarker::INSIDE)
        {
            elvect.SetSize(el.GetDof());
            elvect=0.0;
        }
        else
        {
            Vector sweights;
            //use cut integration
            IntegrationRule ir;
            irules->GetSurfaceIntegrationRule(Trans,ir);
            irules->GetSurfaceWeights(Trans,ir,sweights);
            dint->SetIntRule(&ir);
            dint->SetSurfaceWeights(sweights);
            dint->AssembleRHSElementVect(el,Trans,elvect);
        }
    }
};

class GhostPenaltyVectorIntegrator:public BilinearFormIntegrator
{
public:
    GhostPenaltyVectorIntegrator(double penal_=1.0):penal(penal_)
    {

    }

    virtual
        ~GhostPenaltyVectorIntegrator()
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


class CutGhostPenaltyVectorIntegrator:public BilinearFormIntegrator
{
private:    
    GhostPenaltyVectorIntegrator* dint;
    Array<int>* el_marks;

public:
    CutGhostPenaltyVectorIntegrator(double penal_, Array<int>* marks)
    {
        el_marks=marks;
        dint=new GhostPenaltyVectorIntegrator(penal_);
    }
    virtual
        ~CutGhostPenaltyVectorIntegrator()
    {
        delete dint;
    }


    virtual void AssembleFaceMatrix(const FiniteElement &fe1,
                                    const FiniteElement &fe2,
                                    FaceElementTransformations &Trans,
                                    DenseMatrix &elmat) override
    {

        if(((*el_marks)[Trans.Elem1No]==ElementMarker::CUT) &&  ((*el_marks)[Trans.Elem2No]==ElementMarker::CUT))
        {
            //use standard integration rule
            dint->AssembleFaceMatrix(fe1,fe2,Trans,elmat);

        }
        else if(((*el_marks)[Trans.Elem1No]==ElementMarker::INSIDE) &&  ((*el_marks)[Trans.Elem2No]==ElementMarker::CUT))
        {
            //use standard integration rule
            dint->AssembleFaceMatrix(fe1,fe2,Trans,elmat);
        }
        else if(((*el_marks)[Trans.Elem1No]==ElementMarker::CUT) &&  ((*el_marks)[Trans.Elem2No]==ElementMarker::INSIDE))
        {
            //use standard integration rule
            dint->AssembleFaceMatrix(fe1,fe2,Trans,elmat);
        }
        else{
            const int ndim=Trans.GetSpaceDim();
            const int ndofs1=fe1.GetDof();
            const int ndofs2=fe2.GetDof();
            int ndofs=ndofs1+ndofs2;
            elmat.SetSize(ndofs*ndim); elmat=0.0;
        }
    }

};


class CutVectorDiffusionIntegrator: public BilinearFormIntegrator
{
private:
    VectorDiffusionIntegrator* dint;

    Array<int>* el_marks;
    CutIntegrationRules* irules;
public:
    CutVectorDiffusionIntegrator(Coefficient& q,
                           Array<int>* marks,
                           CutIntegrationRules* cut_int)
    {
        el_marks=marks;
        irules=cut_int;
        dint=new VectorDiffusionIntegrator(q);
    }

    ~CutVectorDiffusionIntegrator()
    {
        delete dint;
    }

    virtual void AssembleElementMatrix(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat) override
    {

        if((*el_marks)[Trans.ElementNo]==ElementMarker::OUTSIDE)
        {
            int sdim = Trans.GetSpaceDim();
            elmat.SetSize(sdim*el.GetDof());
            elmat=0.0;
        }
        else if((*el_marks)[Trans.ElementNo]==ElementMarker::INSIDE)
        {
            //use standard integration rule
            dint->SetIntRule(nullptr);
            dint->AssembleElementMatrix(el,Trans,elmat);
        }
        else
        {
            //use cut integration
            IntegrationRule ir;
            irules->GetVolumeIntegrationRule(Trans,ir);
            dint->SetIntRule(&ir);
            dint->AssembleElementMatrix(el,Trans,elmat);
        }
    }
};

class CutVectorDivergenceIntegrator: public BilinearFormIntegrator
{
private:
    VectorDivergenceIntegrator* dint;

    Array<int>* el_marks;
    CutIntegrationRules* irules;
public:
     CutVectorDivergenceIntegrator(Coefficient& q,
                           Array<int>* marks,
                           CutIntegrationRules* cut_int)
    {
        el_marks=marks;
        irules=cut_int;
        dint=new VectorDivergenceIntegrator(q);
    }

    ~CutVectorDivergenceIntegrator()
    {
        delete dint;
    }

    virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat) override
    {

        if((*el_marks)[Trans.ElementNo]==ElementMarker::OUTSIDE)
        {
            int sdim = Trans.GetSpaceDim();
            int trial_dof = trial_fe.GetDof();
            int test_dof = test_fe.GetDof();
            elmat.SetSize (test_dof, sdim*trial_dof);
            elmat=0.0;
        }
        else if((*el_marks)[Trans.ElementNo]==ElementMarker::INSIDE)
        {
            //use standard integration rule
            dint->SetIntRule(nullptr);
            dint->AssembleElementMatrix2(trial_fe,test_fe,Trans,elmat);
        }
        else
        {
            //use cut integration
            IntegrationRule ir;
            irules->GetVolumeIntegrationRule(Trans,ir);
            dint->SetIntRule(&ir);
            dint->AssembleElementMatrix2(trial_fe,test_fe,Trans,elmat);
        }
    }
};

class CutVectorDomainLFIntegrator : public LinearFormIntegrator
{
    private:
    VectorDomainLFIntegrator* dint;
    Array<int>* el_marks;
    CutIntegrationRules* irules;
public:
   /// Constructs a domain integrator with a given Coefficient
    CutVectorDomainLFIntegrator(VectorCoefficient& q,
                           Array<int>* marks,
                           CutIntegrationRules* cut_int)
    {
        el_marks=marks;
        irules=cut_int;
        dint=new VectorDomainLFIntegrator(q);
    }
    ~CutVectorDomainLFIntegrator()
    {
        delete dint;
    }

    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       Vector &elvect) override
    {

        if((*el_marks)[Trans.ElementNo]==ElementMarker::OUTSIDE)
        {
            int sdim = Trans.GetSpaceDim();
            elvect.SetSize(sdim*el.GetDof());
            elvect=0.0;
        }
        else if((*el_marks)[Trans.ElementNo]==ElementMarker::INSIDE)
        {
            //use standard integration rule
            dint->SetIntRule(nullptr);
            dint->AssembleRHSElementVect(el,Trans,elvect);

        }
        else
        {
            //use cut integration
            IntegrationRule ir;
            irules->GetVolumeIntegrationRule(Trans,ir);
            dint->SetIntRule(&ir);
            dint->AssembleRHSElementVect(el,Trans,elvect);
        }
    }
};



class UnfittedVectorBoundaryLFIntegrator : public LinearFormIntegrator
{
private:
   Vector shape, vec;
   VectorCoefficient &Q;
   Vector sweights;
public:
   /// Constructs a boundary integrator with a given VectorCoefficient QG
   UnfittedVectorBoundaryLFIntegrator(VectorCoefficient &QG) : Q(QG) { }

   /** Given a particular boundary Finite Element and a transformation (Tr)
       computes the element boundary vector, elvect. */
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

    virtual void SetSurfaceWeights(Vector surface_weights) { sweights = surface_weights; }
};



class CutUnfittedVectorBoundaryLFIntegrator : public LinearFormIntegrator
{
    private:
    UnfittedVectorBoundaryLFIntegrator* dint;
    Array<int>* el_marks;
    CutIntegrationRules* irules;
public:
   /// Constructs a domain integrator with a given Coefficient
    CutUnfittedVectorBoundaryLFIntegrator(VectorCoefficient& q,
                           Array<int>* marks,
                           CutIntegrationRules* cut_int)
    {
        el_marks=marks;
        irules=cut_int;
        dint=new UnfittedVectorBoundaryLFIntegrator(q);
    }
    ~CutUnfittedVectorBoundaryLFIntegrator()
    {
        delete dint;
    }

    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       Vector &elvect) override
    {

        if((*el_marks)[Trans.ElementNo]==ElementMarker::OUTSIDE)
        {
            int sdim = Trans.GetSpaceDim();
            elvect.SetSize(sdim*el.GetDof());
            elvect=0.0;
        }
        else if((*el_marks)[Trans.ElementNo]==ElementMarker::INSIDE)
        {
            int sdim = Trans.GetSpaceDim();
            elvect.SetSize(sdim*el.GetDof());
            elvect=0.0;

        }
        else
        {
            //use cut integration
            Vector sweights;
            IntegrationRule ir;
            irules->GetSurfaceIntegrationRule(Trans,ir);
            irules->GetSurfaceWeights(Trans,ir,sweights);
            dint->SetIntRule(&ir);
            dint->SetSurfaceWeights(sweights);
            dint->AssembleRHSElementVect(el,Trans,elvect);
        }
    }
};

}
#endif
