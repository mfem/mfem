//
//                        MFEM CFOSLS Transport equation with multigrid (div-free part)
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>

//#define BAD_TEST
//#define ONLY_DIVFREEPART
//#define K_IDENTITY

#define MYZEROTOL (1.0e-13)

using namespace std;
using namespace mfem;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

class VectorcurlDomainLFIntegrator : public LinearFormIntegrator
{
    DenseMatrix curlshape;
    DenseMatrix curlshape_dFadj;
    DenseMatrix curlshape_dFT;
    DenseMatrix dF_curlshape;
    VectorCoefficient &VQ;
    int oa, ob;
public:
    /// Constructs a domain integrator with a given Coefficient
    VectorcurlDomainLFIntegrator(VectorCoefficient &VQF, int a = 2, int b = 0)
        : VQ(VQF), oa(a), ob(b) { }

    /// Constructs a domain integrator with a given Coefficient
    VectorcurlDomainLFIntegrator(VectorCoefficient &VQF, const IntegrationRule *ir)
        : LinearFormIntegrator(ir), VQ(VQF), oa(1), ob(1) { }

    /** Given a particular Finite Element and a transformation (Tr)
       computes the element right hand side element vector, elvect. */
    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect);

    using LinearFormIntegrator::AssembleRHSElementVect;
};

void VectorcurlDomainLFIntegrator::AssembleRHSElementVect(
        const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
    int dof = el.GetDof();

    MFEM_ASSERT(dim == 3, "VectorcurlDomainLFIntegrator is working only in 3D currently \n");

    curlshape.SetSize(dof,3);           // matrix of size dof x 3, works only in 3D
    curlshape_dFadj.SetSize(dof,3);     // matrix of size dof x 3, works only in 3D
    curlshape_dFT.SetSize(dof,3);       // matrix of size dof x 3, works only in 3D
    dF_curlshape.SetSize(3,dof);        // matrix of size dof x 3, works only in 3D
    Vector vecval(3);
    Vector vecval_new(3);
    DenseMatrix invdfdx(3,3);

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        // ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob + Tr.OrderW());
        ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);
        // int order = 2 * el.GetOrder() ; // <--- OK for RTk
        // ir = &IntRules.Get(el.GetGeomType(), order);
    }

    elvect.SetSize(dof);
    elvect = 0.0;

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        el.CalcCurlShape(ip, curlshape);

        Tr.SetIntPoint (&ip);

        VQ.Eval(vecval,Tr,ip);                  // plain evaluation

        MultABt(curlshape, Tr.Jacobian(), curlshape_dFT);

        curlshape_dFT.AddMult_a(ip.weight, vecval, elvect);
    }

}

class VectordivDomainLFIntegrator : public LinearFormIntegrator
{
   Vector divshape;
   Coefficient &Q;
   int oa, ob;
public:
   /// Constructs a domain integrator with a given Coefficient
   VectordivDomainLFIntegrator(Coefficient &QF, int a = 2, int b = 0)
   // the old default was a = 1, b = 1
   // for simple elliptic problems a = 2, b = -2 is ok
      : Q(QF), oa(a), ob(b) { }

   /// Constructs a domain integrator with a given Coefficient
   VectordivDomainLFIntegrator(Coefficient &QF, const IntegrationRule *ir)
      : LinearFormIntegrator(ir), Q(QF), oa(1), ob(1) { }

   /** Given a particular Finite Element and a transformation (Tr)
       computes the element right hand side element vector, elvect. */
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};
//---------

//------------------
void VectordivDomainLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)//don't need the matrix but the vector
{
   int dof = el.GetDof();

   divshape.SetSize(dof);       // vector of size dof
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob + Tr.OrderW());
      ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);
     // int order = 2 * el.GetOrder() ; // <--- OK for RTk
     // ir = &IntRules.Get(el.GetGeomType(), order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDivShape(ip, divshape);

      Tr.SetIntPoint (&ip);
      //double val = Tr.Weight() * Q.Eval(Tr, ip);
      // Chak: Looking at how MFEM assembles in VectorFEDivergenceIntegrator, I think you dont need Tr.Weight() here
      // I think this is because the RT (or other vector FE) basis is scaled by the geometry of the mesh
      double val = Q.Eval(Tr, ip);

      add(elvect, ip.weight * val, divshape, elvect);
      //cout << "elvect = " << elvect << endl;
   }
}

class GradDomainLFIntegrator : public LinearFormIntegrator
{
    DenseMatrix dshape;
    DenseMatrix invdfdx;
    DenseMatrix dshapedxt;
    Vector bf;
    Vector bfdshapedxt;
    VectorCoefficient &Q;
    int oa, ob;
 public:
    /// Constructs a domain integrator with a given Coefficient
    GradDomainLFIntegrator(VectorCoefficient &QF, int a = 2, int b = 0)
    // the old default was a = 1, b = 1
    // for simple elliptic problems a = 2, b = -2 is ok
       : Q(QF), oa(a), ob(b) { }

    /// Constructs a domain integrator with a given Coefficient
    GradDomainLFIntegrator(VectorCoefficient &QF, const IntegrationRule *ir)
       : LinearFormIntegrator(ir), Q(QF), oa(1), ob(1) { }

    /** Given a particular Finite Element and a transformation (Tr)
        computes the element right hand side element vector, elvect. */
    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect);

    using LinearFormIntegrator::AssembleRHSElementVect;
 };

void GradDomainLFIntegrator::AssembleRHSElementVect(
    const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
    int dof = el.GetDof();
    int dim  = el.GetDim();

    dshape.SetSize(dof,dim);       // vector of size dof
    elvect.SetSize(dof);
    elvect = 0.0;

    invdfdx.SetSize(dim,dim);
    dshapedxt.SetSize(dof,dim);
    bf.SetSize(dim);
    bfdshapedxt.SetSize(dof);
    double w;

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
 //       ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob
 //                          + Tr.OrderW());
 //      ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);
      // int order = 2 * el.GetOrder() ; // <--- OK for RTk
       int order = (Tr.OrderW() + el.GetOrder() + el.GetOrder());
       ir = &IntRules.Get(el.GetGeomType(), order);
    }

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
       const IntegrationPoint &ip = ir->IntPoint(i);
       el.CalcDShape(ip, dshape);

       //double val = Tr.Weight() * Q.Eval(Tr, ip);

       Tr.SetIntPoint (&ip);
       w = ip.weight;// * Tr.Weight();
       CalcAdjugate(Tr.Jacobian(), invdfdx);
       Mult(dshape, invdfdx, dshapedxt);

       Q.Eval(bf, Tr, ip);

       dshapedxt.Mult(bf, bfdshapedxt);

       add(elvect, w, bfdshapedxt, elvect);
    }
}

/** Bilinear integrator for (curl u, v) for Nedelec and scalar finite element for v. If the trial and
    test spaces are switched, assembles the form (u, curl v). */
class VectorFECurlVQIntegrator: public BilinearFormIntegrator
{
private:
   VectorCoefficient *VQ;
#ifndef MFEM_THREAD_SAFE
   Vector shape;
   DenseMatrix curlshape;
   DenseMatrix curlshape_dFT;
   //old
   DenseMatrix curlshapeTrial;
   DenseMatrix vshapeTest;
   DenseMatrix curlshapeTrial_dFT;
#endif
   void Init(VectorCoefficient *vq)
   { VQ = vq; }
public:
   VectorFECurlVQIntegrator() { Init(NULL); }
   VectorFECurlVQIntegrator(VectorCoefficient &vq) { Init(&vq); }
   VectorFECurlVQIntegrator(VectorCoefficient *vq) { Init(vq); }
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat) { }
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

void VectorFECurlVQIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   int trial_nd = trial_fe.GetDof(), test_nd = test_fe.GetDof(), i;
   //int dim = trial_fe.GetDim();
   //int dimc = (dim == 3) ? 3 : 1;
   int dim;
   int dimc;
   int vector_dof, scalar_dof;

   MFEM_ASSERT(trial_fe.GetMapType() == mfem::FiniteElement::H_CURL ||
               test_fe.GetMapType() == mfem::FiniteElement::H_CURL,
               "At least one of the finite elements must be in H(Curl)");

   int curl_nd, vec_nd;
   if ( trial_fe.GetMapType() == mfem::FiniteElement::H_CURL )
   {
      curl_nd = trial_nd;
      vector_dof = trial_fe.GetDof();
      vec_nd  = test_nd;
      scalar_dof = test_fe.GetDof();
      dim = trial_fe.GetDim();
      dimc = dim;
   }
   else
   {
      curl_nd = test_nd;
      vector_dof = test_fe.GetDof();
      vec_nd  = trial_nd;
      scalar_dof = trial_fe.GetDof();
      dim = test_fe.GetDim();
      dimc = dim;
   }

   MFEM_ASSERT(dim == 3, "VectorFECurlVQIntegrator is working only in 3D currently \n");

#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshapeTrial(curl_nd, dimc);
   DenseMatrix curlshapeTrial_dFT(curl_nd, dimc);
   DenseMatrix vshapeTest(vec_nd, dimc);
#else
   //curlshapeTrial.SetSize(curl_nd, dimc);
   //curlshapeTrial_dFT.SetSize(curl_nd, dimc);
   //vshapeTest.SetSize(vec_nd, dimc);
#endif
   //Vector shapeTest(vshapeTest.GetData(), vec_nd);

   curlshape.SetSize(vector_dof, dim);
   curlshape_dFT.SetSize(vector_dof, dim);
   shape.SetSize(scalar_dof);
   Vector D(vec_nd);

   elmat.SetSize(test_nd, trial_nd);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = trial_fe.GetOrder() + test_fe.GetOrder() - 1; // <--
      ir = &IntRules.Get(trial_fe.GetGeomType(), order);
   }

   elmat = 0.0;
   for (i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Trans.SetIntPoint(&ip);

      double w = ip.weight;
      VQ->Eval(D, Trans, ip);
      D *= w;

      if (dim == 3)
      {
         if ( trial_fe.GetMapType() == mfem::FiniteElement::H_CURL )
         {
            trial_fe.CalcCurlShape(ip, curlshape);
            test_fe.CalcShape(ip, shape);
         }
         else
         {
            test_fe.CalcCurlShape(ip, curlshape);
            trial_fe.CalcShape(ip, shape);
         }
         MultABt(curlshape, Trans.Jacobian(), curlshape_dFT);

         ///////////////////////////
         for (int d = 0; d < dim; d++)
         {
            for (int j = 0; j < scalar_dof; j++)
            {
                for (int k = 0; k < vector_dof; k++)
                {
                    elmat(j, k) += D[d] * shape(j) * curlshape_dFT(k, d);
                }
            }
         }
         ///////////////////////////
      }
   }
}

double uFun_ex(const Vector& x); // Exact Solution
double uFun_ex_dt(const Vector& xt);
void uFun_ex_gradx(const Vector& xt, Vector& grad);

void bFun_ex (const Vector& xt, Vector& b);
double  bFundiv_ex(const Vector& xt);

void bFunRect2D_ex(const Vector& xt, Vector& b );
double  bFunRect2Ddiv_ex(const Vector& xt);

void bFunCube3D_ex(const Vector& xt, Vector& b );
double  bFunCube3Ddiv_ex(const Vector& xt);

void bFunSphere3D_ex(const Vector& xt, Vector& b );
double  bFunSphere3Ddiv_ex(const Vector& xt);

void bFunCircle2D_ex (const Vector& xt, Vector& b);
double  bFunCircle2Ddiv_ex(const Vector& xt);

double uFun3_ex(const Vector& x); // Exact Solution
double uFun3_ex_dt(const Vector& xt);
void uFun3_ex_gradx(const Vector& xt, Vector& grad);

double uFun4_ex(const Vector& x); // Exact Solution
double uFun4_ex_dt(const Vector& xt);
void uFun4_ex_gradx(const Vector& xt, Vector& grad);

//void bFun4_ex (const Vector& xt, Vector& b);

//void bFun6_ex (const Vector& xt, Vector& b);

double uFun5_ex(const Vector& x); // Exact Solution
double uFun5_ex_dt(const Vector& xt);
void uFun5_ex_gradx(const Vector& xt, Vector& grad);

double uFun6_ex(const Vector& x); // Exact Solution
double uFun6_ex_dt(const Vector& xt);
void uFun6_ex_gradx(const Vector& xt, Vector& grad);

double uFunCylinder_ex(const Vector& x); // Exact Solution
double uFunCylinder_ex_dt(const Vector& xt);
void uFunCylinder_ex_gradx(const Vector& xt, Vector& grad);

double uFun66_ex(const Vector& x); // Exact Solution
double uFun66_ex_dt(const Vector& xt);
void uFun66_ex_gradx(const Vector& xt, Vector& grad);


double uFun2_ex(const Vector& x); // Exact Solution
double uFun2_ex_dt(const Vector& xt);
void uFun2_ex_gradx(const Vector& xt, Vector& grad);

void Hdivtest_fun(const Vector& xt, Vector& out );
double  L2test_fun(const Vector& xt);

double uFun33_ex(const Vector& x); // Exact Solution
double uFun33_ex_dt(const Vector& xt);
void uFun33_ex_gradx(const Vector& xt, Vector& grad);

void videofun(const Vector& xt, Vector& vecvalue);

void hcurlFun3D_ex(const Vector& xt, Vector& vecvalue);
void curlhcurlFun3D_ex(const Vector& xt, Vector& vecvalue);
void hcurlFun3D_2_ex(const Vector& xt, Vector& vecvalue);
void curlhcurlFun3D_2_ex(const Vector& xt, Vector& vecvalue);

void DivmatFun4D_ex(const Vector& xt, Vector& vecvalue);
void DivmatDivmatFun4D_ex(const Vector& xt, Vector& vecvalue);

double zero_ex(const Vector& xt);
void zerovec_ex(const Vector& xt, Vector& vecvalue);
void zerovecx_ex(const Vector& xt, Vector& zerovecx );

void vminusone_exact(const Vector &x, Vector &vminusone);
void vone_exact(const Vector &x, Vector &vone);

double cas_weight (const Vector& xt, double * params, const int &nparams);
double deletethis (const Vector& xt);

// Exact solution, E, and r.h.s., f. See below for implementation.
void E_exact(const Vector &, Vector &);
void curlE_exact(const Vector &x, Vector &curlE);
void f_exact(const Vector &, Vector &);
double freq = 1.0, kappa;

template <double (*S)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )>
    void sigmaTemplate(const Vector& xt, Vector& sigma);
template <void (*bvecfunc)(const Vector&, Vector& )>
    void KtildaTemplate(const Vector& xt, DenseMatrix& Ktilda);
template <void (*bvecfunc)(const Vector&, Vector& )>
    void bbTTemplate(const Vector& xt, DenseMatrix& bbT);
template <void (*bvecfunc)(const Vector&, Vector& )>
    double bTbTemplate(const Vector& xt);
template <double (*S)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )>
    double minbTbSnonhomoTemplate(const Vector& xt);
template <double (*S)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )>
    void sigmaNonHomoTemplate(const Vector& xt, Vector& sigma);
template <double (*S)(const Vector&), void (*bvecfunc)(const Vector&, Vector& ), void (*opdivfreevec)(const Vector&, Vector& )>
        void minKsigmahatTemplate(const Vector& xt, Vector& minKsigmahatv);
template<double (*S)(const Vector & xt), void(*bvec)(const Vector & x, Vector & vec), void (*opdivfreevec)(const Vector&, Vector& )>
        double bsigmahatTemplate(const Vector& xt);
template<double (*S)(const Vector & xt), void (*bvecfunc)(const Vector&, Vector& ),
        void (*opdivfreevec)(const Vector&, Vector& )>
        void sigmahatTemplate(const Vector& xt, Vector& sigmahatv);
template<double (*S)(const Vector & xt), void (*bvecfunc)(const Vector&, Vector& ),
        void (*opdivfreevec)(const Vector&, Vector& )>
        void minsigmahatTemplate(const Vector& xt, Vector& sigmahatv);
template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
    double rhsideTemplate(const Vector& xt);

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
        void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
        void bfTemplate(const Vector& xt, Vector& bf);

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
        void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
        void bdivsigmaTemplate(const Vector& xt, Vector& bf);

template<double (*S)(const Vector & xt), void(*bvec)(const Vector & x, Vector & vec)>
        void bSnonhomoTemplate(const Vector& xt, Vector& bSnonhomo);

template<void(*bvec)(const Vector & x, Vector & vec)>
        void minbTemplate(const Vector& xt, Vector& minb);

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
        void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
        double divsigmaTemplate(const Vector& xt);

template<double (*S)(const Vector & xt) > double SnonhomoTemplate(const Vector& xt);

class Transport_test_divfree
{
    protected:
        int dim;
        int numsol;
        int numcurl;

    public:
        FunctionCoefficient * scalarS;
        FunctionCoefficient * S_nonhomo;              // S_nonhomo(x,t) = S(x,t=0)
        FunctionCoefficient * scalarf;                // d (S - S_nonhomo) /dt + div (b [S - S_nonhomo]), Snonhomo = S(x,0)
        FunctionCoefficient * scalardivsigma;         // = dS/dt + div (bS) = div sigma
        FunctionCoefficient * bTb;                    // b^T * b
        FunctionCoefficient * minbTbSnonhomo;         // - b^T * b * S_nonhomo
        FunctionCoefficient * bsigmahat;              // b * sigma_hat
        VectorFunctionCoefficient * sigma;
        VectorFunctionCoefficient * sigmahat;         // sigma_hat = sigma_exact - op divfreepart (curl hcurlpart in 3D)
        VectorFunctionCoefficient * b;
        VectorFunctionCoefficient * minb;
        VectorFunctionCoefficient * bf;
        VectorFunctionCoefficient * bdivsigma;        // b * div sigma = b * initial f (before modifying it due to inhomogenuity)
        MatrixFunctionCoefficient * Ktilda;
        MatrixFunctionCoefficient * bbT;
        VectorFunctionCoefficient * sigma_nonhomo;    // to incorporate inhomogeneous boundary conditions, stores (b*S0, S0) with S(t=0) = S0
        VectorFunctionCoefficient * bSnonhomo;        // b * S_nonhomo
        VectorFunctionCoefficient * divfreepart;        // additional part added for testing div-free solver
        VectorFunctionCoefficient * opdivfreepart;    // curl of the additional part which is added to sigma_exact for testing div-free solver
        VectorFunctionCoefficient * minKsigmahat;     // - K * sigma_hat
        VectorFunctionCoefficient * minsigmahat;      // -sigma_hat
    public:
        Transport_test_divfree (int Dim, int NumSol, int NumCurl);

        int GetDim() {return dim;}
        int GetNumSol() {return numsol;}
        int GetNumCurl() {return numcurl;}
        void SetDim(int Dim) { dim = Dim;}
        void SetNumSol(int NumSol) { numsol = NumSol;}
        void SetNumCurl(int NumCurl) { numcurl = NumCurl;}
        bool CheckTestConfig();

        ~Transport_test_divfree () {}
    private:
        template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
                 void(*bvec)(const Vector & x, Vector & vec), double (*divbvec)(const Vector & xt), \
                 void(*hcurlvec)(const Vector & x, Vector & vec), void(*opdivfreevec)(const Vector & x, Vector & vec)> \
        void SetTestCoeffs ( );

        void SetSFun( double (*S)(const Vector & xt))
        { scalarS = new FunctionCoefficient(S);}

        template< double (*S)(const Vector & xt)>  \
        void SetSNonhomo()
        {
            S_nonhomo = new FunctionCoefficient(SnonhomoTemplate<S>);
        }

        template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
                 void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
        void SetScalarfFun()
        { scalarf = new FunctionCoefficient(rhsideTemplate<S, dSdt, Sgradxvec, bvec, divbfunc>);}

        template<double (*S)(const Vector & xt), void(*bvec)(const Vector & x, Vector & vec)>  \
        void SetminbTbSnonhomo()
        {
            minbTbSnonhomo = new FunctionCoefficient(minbTbSnonhomoTemplate<S,bvec>);
        }

        template<void(*bvec)(const Vector & x, Vector & vec)>  \
        void SetScalarBtB()
        {
            bTb = new FunctionCoefficient(bTbTemplate<bvec>);
        }

        template<double (*S)(const Vector & xt), void(*bvec)(const Vector & x, Vector & vec)> \
        void SetHdivVec()
        {
            sigma = new VectorFunctionCoefficient(dim, sigmaTemplate<S,bvec>);
        }

        template<void(*bvec)(const Vector & x, Vector & vec)> \
        void SetminbVec()
        { minb = new VectorFunctionCoefficient(dim, minbTemplate<bvec>);}

        void SetbVec( void(*bvec)(const Vector & x, Vector & vec) )
        { b = new VectorFunctionCoefficient(dim, bvec);}

        template< void(*bvec)(const Vector & x, Vector & vec)>  \
        void SetKtildaMat()
        {
            Ktilda = new MatrixFunctionCoefficient(dim, KtildaTemplate<bvec>);
        }

        template< void(*bvec)(const Vector & x, Vector & vec)>  \
        void SetBBtMat()
        {
            bbT = new MatrixFunctionCoefficient(dim, bbTTemplate<bvec>);
        }

        template<double (*S)(const Vector & xt), void(*bvec)(const Vector & x, Vector & vec)> \
        void SetSigmaNonhomoVec()
        {
            sigma_nonhomo = new VectorFunctionCoefficient(dim, sigmaNonHomoTemplate<S,bvec>);
        }

        template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
                 void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
        void SetdivSigma()
        { scalardivsigma = new FunctionCoefficient(divsigmaTemplate<S, dSdt, Sgradxvec, bvec, divbfunc>);}

        template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
                 void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
        void SetbfVec()
        { bf = new VectorFunctionCoefficient(dim, bfTemplate<S,dSdt,Sgradxvec,bvec,divbfunc>);}

        template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
                 void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
        void SetbdivsigmaVec()
        { bdivsigma = new VectorFunctionCoefficient(dim, bdivsigmaTemplate<S,dSdt,Sgradxvec,bvec,divbfunc>);}

        void SetDivfreePart( void(*divfreevec)(const Vector & x, Vector & vec))
        { divfreepart = new VectorFunctionCoefficient(dim, divfreevec);}

        void SetOpDivfreePart( void(*opdivfreevec)(const Vector & x, Vector & vec))
        { opdivfreepart = new VectorFunctionCoefficient(dim, opdivfreevec);}

        template<double (*S)(const Vector & xt), void(*bvec)(const Vector & x, Vector & vec), void(*opdivfreevec)(const Vector & x, Vector & vec)> \
        void SetminKsigmahat()
        { minKsigmahat = new VectorFunctionCoefficient(dim, minKsigmahatTemplate<S, bvec, opdivfreevec>);}

        template<double (*S)(const Vector & xt), void(*bvec)(const Vector & x, Vector & vec), void(*opdivfreevec)(const Vector & x, Vector & vec)> \
        void Setbsigmahat()
        { bsigmahat = new FunctionCoefficient(bsigmahatTemplate<S, bvec, opdivfreevec>);}

        template<double (*S)(const Vector & xt), void (*bvec)(const Vector&, Vector& ),
                 void(*opdivfreevec)(const Vector & x, Vector & vec)> \
        void Setsigmahat()
        { sigmahat = new VectorFunctionCoefficient(dim, sigmahatTemplate<S, bvec, opdivfreevec>);}

        template<double (*S)(const Vector & xt), void (*bvec)(const Vector&, Vector& ),
                 void(*opdivfreevec)(const Vector & x, Vector & vec)> \
        void Setminsigmahat()
        { minsigmahat = new VectorFunctionCoefficient(dim, minsigmahatTemplate<S, bvec, opdivfreevec>);}

        template<double (*S)(const Vector & xt), void (*bvec)(const Vector&, Vector& )>
        void SetbSnonhomoVec()
        { bSnonhomo = new VectorFunctionCoefficient(dim, bSnonhomoTemplate<S, bvec>);}

};

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx),  \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt), \
         void(*divfreevec)(const Vector & x, Vector & vec), void(*opdivfreevec)(const Vector & x, Vector & vec)> \
void Transport_test_divfree::SetTestCoeffs ()
{
    SetSFun(S);
    SetSNonhomo<S>();
    SetScalarfFun<S, dSdt, Sgradxvec, bvec, divbfunc>();
    SetminbVec<bvec>();
    SetbVec(bvec);
    SetbSnonhomoVec<S, bvec>();
    SetbfVec<S, dSdt, Sgradxvec, bvec, divbfunc>();
    SetbdivsigmaVec<S, dSdt, Sgradxvec, bvec, divbfunc>();
    SetHdivVec<S,bvec>();
    SetKtildaMat<bvec>();
    SetScalarBtB<bvec>();
    SetminbTbSnonhomo<S, bvec>();
    SetSigmaNonhomoVec<S,bvec>();
    SetdivSigma<S, dSdt, Sgradxvec, bvec, divbfunc>();
    SetDivfreePart(divfreevec);
    SetOpDivfreePart(opdivfreevec);
    SetminKsigmahat<S, bvec, opdivfreevec>();
    Setbsigmahat<S, bvec, opdivfreevec>();
    Setsigmahat<S, bvec, opdivfreevec>();
    Setminsigmahat<S, bvec, opdivfreevec>();
    SetBBtMat<bvec>();
    return;
}


bool Transport_test_divfree::CheckTestConfig()
{
    if (dim == 4 || dim == 3)
    {
        if ( numsol == 0 && dim >= 3 )
            return true;
        if ( numsol == 2 && dim == 3 )
            return true;
        if ( numsol == 4 && dim == 3 )
            return true;
        return false;
    }
    else
        return false;

}

Transport_test_divfree::Transport_test_divfree (int Dim, int NumSol, int NumCurl)
{
    dim = Dim;
    numsol = NumSol;
    numcurl = NumCurl;

    if ( CheckTestConfig() == false )
        std::cout << "Inconsistent dim = " << dim << " and numsol = " << numsol <<  std::endl << std::flush;
    else
    {
        if (numsol == 0)
        {
            if (dim == 3)
            {
                if (numcurl == 1)
                    SetTestCoeffs<&zero_ex, &zero_ex, &zerovecx_ex, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
                else if (numcurl == 2)
                    SetTestCoeffs<&zero_ex, &zero_ex, &zerovecx_ex, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &hcurlFun3D_2_ex, &curlhcurlFun3D_2_ex>();
                else
                    SetTestCoeffs<&zero_ex, &zero_ex, &zerovecx_ex, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &zerovec_ex, &zerovec_ex>();
            }
            if (dim > 3)
            {
                if (numcurl == 1)
                    SetTestCoeffs<&zero_ex, &zero_ex, &zerovecx_ex, &bFunCube3D_ex, &bFunCube3Ddiv_ex, &DivmatFun4D_ex, &DivmatDivmatFun4D_ex>();
                else
                    SetTestCoeffs<&zero_ex, &zero_ex, &zerovecx_ex, &bFunCube3D_ex, &bFunCube3Ddiv_ex, &zerovec_ex, &zerovec_ex>();
            }
        }
        if (numsol == 2)
        {
            //std::cout << "The domain must be a cylinder over a square" << std::endl << std::flush;
            //SetTestCoeffs<&uFun4_ex, &uFun4_ex_dt, &uFun4_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &cas_weight, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
            if (numcurl == 1)
                SetTestCoeffs<&uFun2_ex, &uFun2_ex_dt, &uFun2_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
            else if (numcurl == 2)
                SetTestCoeffs<&uFun2_ex, &uFun2_ex_dt, &uFun2_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &hcurlFun3D_2_ex, &curlhcurlFun3D_2_ex>();
            else
                SetTestCoeffs<&uFun2_ex, &uFun2_ex_dt, &uFun2_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &zerovec_ex, &zerovec_ex>();
        }
        if (numsol == 4)
        {
            //std::cout << "The domain must be a cylinder over a square" << std::endl << std::flush;
            //SetTestCoeffs<&uFun4_ex, &uFun4_ex_dt, &uFun4_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &cas_weight, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
            if (numcurl == 1)
                SetTestCoeffs<&uFun4_ex, &uFun4_ex_dt, &uFun4_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
            else if (numcurl == 2)
                SetTestCoeffs<&uFun4_ex, &uFun4_ex_dt, &uFun4_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &hcurlFun3D_2_ex, &curlhcurlFun3D_2_ex>();
            else
                SetTestCoeffs<&uFun4_ex, &uFun4_ex_dt, &uFun4_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &zerovec_ex, &zerovec_ex>();
        }
    } // end of setting test coefficients in correct case
}

int main(int argc, char *argv[])
{
    int num_procs, myid;
    bool visualization = 0;

    // 1. Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    bool verbose = (myid == 0);

    int nDimensions     = 4;
    int numsol          = 0;
    int numcurl         = 1;

    int ser_ref_levels  = 0;
    int par_ref_levels  = 2;

    int generate_frombase   = 1;
    int Nsteps              = 4;
    double tau              = 0.25;
    int generate_parallel   = generate_frombase * 1;
    int whichparallel       = generate_parallel * 2;
    int bnd_method          = 1;
    int local_method        = 2;

    bool withDiv = false;
    bool withS = false;
    bool blockedversion = true;

    // solver options
    int prec_option = 0;        // defines whether to use preconditioner or not, and which one
    bool prec_is_MG = false;

    //int nlevels = 2;
    //int coarsenfactor = 8;

    //const char *mesh_file = "../build3/meshes/cube_3d_fine.mesh";
    //const char *mesh_file = "../build3/meshes/square_2d_moderate.mesh";

    //const char *mesh_file = "../build3/meshes/cube4d_low.MFEM";
    //const char *mesh_file = "../build3/meshes/cube4d.MFEM";
    const char *mesh_file = "dsadsad";
    //const char *mesh_file = "../build3/meshes/orthotope3D_moderate.mesh";
    //const char * mesh_file = "../build3/meshes/orthotope3D_fine.mesh";

    //const char * meshbase_file = "../build3/meshes/sphere3D_0.1to0.2.mesh";
    //const char * meshbase_file = "../build3/meshes/sphere3D_0.05to0.1.mesh";
    //const char * meshbase_file = "../build3/meshes/sphere3D_veryfine.mesh";
    //const char * meshbase_file = "../build3/meshes/orthotope3D_moderate.mesh";
    const char * meshbase_file = "../data/orthotope3D_fine.mesh";
    //const char * meshbase_file = "../build3/meshes/square_2d_moderate.mesh";
    //const char * meshbase_file = "../data/square_2d_fine.mesh";
    //const char * meshbase_file = "../build3/meshes/square-disc.mesh";
    //const char *meshbase_file = "dsadsad";
    //const char * meshbase_file = "../build3/meshes/circle_fine_0.1.mfem";
    //const char * meshbase_file = "../build3/meshes/circle_moderate_0.2.mfem";

    int feorder         = 0;

    kappa = freq * M_PI;

    if (verbose)
        cout << "Solving FOSLS Transport equation with MFEM & hypre" << endl;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&meshbase_file, "-mbase", "--meshbase",
                   "Mesh base file to use.");
    args.AddOption(&feorder, "-o", "--feorder",
                   "Finite element order (polynomial degree).");
    args.AddOption(&ser_ref_levels, "-sref", "--sref",
                   "Number of serial refinements 4d mesh.");
    args.AddOption(&par_ref_levels, "-pref", "--pref",
                   "Number of parallel refinements 4d mesh.");
    args.AddOption(&nDimensions, "-dim", "--whichD",
                   "Dimension of the space-time problem.");
    args.AddOption(&Nsteps, "-nstps", "--nsteps",
                   "Number of time steps.");
    args.AddOption(&tau, "-tau", "--tau",
                   "Time step.");
    args.AddOption(&generate_frombase, "-gbase", "--genfrombase",
                   "Generating mesh from the base mesh.");
    args.AddOption(&generate_parallel, "-gp", "--genpar",
                   "Generating mesh in parallel.");
    args.AddOption(&numsol, "-nsol", "--numsol",
                   "Solution number.");
    args.AddOption(&numcurl, "-ncurl", "--numcurl",
                   "Curl additive term's' number.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&prec_option, "-precopt", "--prec-option",
                   "Preconditioner choice.");

    args.Parse();
    if (!args.Good())
    {
       if (verbose)
       {
          args.PrintUsage(cout);
       }
       MPI_Finalize();
       return 1;
    }
    if (verbose)
    {
       args.PrintOptions(cout);
    }

    if (verbose)
        cout << "Number of mpi processes: " << num_procs << endl << flush;

    bool with_prec = true;

    switch (prec_option)
    {
    case 2: // MG
        with_prec = true;
        prec_is_MG = true;
        break;
    default: // no preconditioner (default)
        with_prec = false;
        prec_is_MG = false;
        break;
    }

    if (verbose)
    {
        cout << "with_prec = " << with_prec << endl;
        cout << "prec_is_MG = " << prec_is_MG << endl;
        cout << flush;
    }

    StopWatch chrono;

    //DEFAULTED LINEAR SOLVER OPTIONS
    int max_num_iter = 150000;
    double rtol = 1e-12;//1e-7;//1e-9;
    double atol = 1e-14;//1e-9;//1e-12;

    Mesh *mesh = NULL;

    shared_ptr<ParMesh> pmesh;

    if (nDimensions == 3 || nDimensions == 4)
    {
        if ( generate_frombase == 1 )
        {
            if ( verbose )
                cout << "Creating a " << nDimensions << "d mesh from a " <<
                        nDimensions - 1 << "d mesh from the file " << meshbase_file << endl;

            Mesh * meshbase;
            ifstream imesh(meshbase_file);
            if (!imesh)
            {
                 cerr << "\nCan not open mesh file for base mesh: " <<
                                                    meshbase_file << endl << flush;
                 MPI_Finalize();
                 return -2;
            }
            meshbase = new Mesh(imesh, 1, 1);
            imesh.close();

            for (int l = 0; l < ser_ref_levels; l++)
                meshbase->UniformRefinement();

            if (verbose)
                meshbase->PrintInfo();

            /*
            if ( verbose )
            {
                std::stringstream fname;
                fname << "mesh_" << nDimensions - 1 << "dbase.mesh";
                std::ofstream ofid(fname.str().c_str());
                ofid.precision(8);
                meshbase->Print(ofid);
            }
            */

            if (generate_parallel == 1) //parallel version
            {
                ParMesh * pmeshbase = new ParMesh(comm, *meshbase);

                /*
                std::stringstream fname;
                fname << "pmesh_"<< nDimensions - 1 << "dbase_" << myid << ".mesh";
                std::ofstream ofid(fname.str().c_str());
                ofid.precision(8);
                pmesh3dbase->Print(ofid);
                */

                chrono.Clear();
                chrono.Start();

                if ( whichparallel == 1 )
                {
                    if ( nDimensions == 3)
                    {
                        if  (verbose)
                            cout << "Not implemented for 2D->3D. Use parallel version2"
                                    " instead" << endl << flush;
                        MPI_Finalize();
                        return 0;
                    }
                    else // nDimensions == 4
                    {
                        mesh = new Mesh( comm, *pmeshbase, tau, Nsteps, bnd_method, local_method);
                        if ( verbose )
                            cout << "Success: ParMesh is created by deprecated method"
                                 << endl << flush;

                        std::stringstream fname;
                        fname << "mesh_par1_id" << myid << "_np_" << num_procs << ".mesh";
                        std::ofstream ofid(fname.str().c_str());
                        ofid.precision(8);
                        mesh->Print(ofid);

                        MPI_Barrier(comm);
                    }
                }
                else
                {
                    if (verbose)
                        cout << "Starting parallel \"" << nDimensions-1 << "D->"
                             << nDimensions <<"D\" mesh generator" << endl;

                    pmesh = make_shared<ParMesh>( comm, *pmeshbase, tau, Nsteps,
                                                  bnd_method, local_method);

                    if (verbose)
                        cout << "Success: ParMesh created" << endl << flush;
                    MPI_Barrier(comm);
                }

                chrono.Stop();
                if (verbose && whichparallel == 2)
                    cout << "Timing: Space-time mesh extension done in parallel in "
                              << chrono.RealTime() << " seconds.\n" << endl << flush;
                delete pmeshbase;
            }
            else // serial version
            {
                if (verbose)
                    cout << "Starting serial \"" << nDimensions-1 << "D->"
                         << nDimensions <<"D\" mesh generator" << endl;
                mesh = new Mesh( *meshbase, tau, Nsteps, bnd_method, local_method);
                if (verbose)
                    cout << "Timing: Space-time mesh extension done in serial in "
                              << chrono.RealTime() << " seconds.\n" << endl << flush;
            }

            delete meshbase;

        }
        else // not generating from a lower dimensional mesh
        {
            if (verbose)
                cout << "Reading a " << nDimensions << "d mesh from the file " << mesh_file << endl;
            ifstream imesh(mesh_file);
            if (!imesh)
            {
                 std::cerr << "\nCan not open mesh file: " << mesh_file << '\n' << std::endl;
                 MPI_Finalize();
                 return -2;
            }
            else
            {
                mesh = new Mesh(imesh, 1, 1);
                imesh.close();
            }

        }

    }
    else //if nDimensions is not 3 or 4
    {
        if (verbose)
            cerr << "Case nDimensions = " << nDimensions << " is not supported"
                 << endl << flush;
        MPI_Finalize();
        return -1;

    }

    //MPI_Finalize();
    //return 0;

    if (mesh) // if only serial mesh was generated previously, parallel mesh is initialized here
    {
        // Checking that mesh is legal
        //if (myid == 0)
            //cout << "Checking the mesh" << endl << flush;
        //mesh->MeshCheck();

        for (int l = 0; l < ser_ref_levels; l++)
            mesh->UniformRefinement();

        if ( verbose )
            cout << "Creating parmesh(" << nDimensions <<
                    "d) from the serial mesh (" << nDimensions << "d)" << endl << flush;
        pmesh = make_shared<ParMesh>(comm, *mesh);
        delete mesh;
    }

   //pmesh->ComputeSlices ( 0.1, 2, 0.3, myid);
   //MPI_Finalize();
   //return 0;

    for (int l = 0; l < par_ref_levels; l++)
    {
       pmesh->UniformRefinement();
    }

    //if(dim==3) pmesh->ReorientTetMesh();

    pmesh->PrintInfo(std::cout); if(verbose) cout << endl;

    Transport_test_divfree Mytest(nDimensions, numsol, numcurl);

    // 6. Define a parallel finite element space on the parallel mesh. Here we
    //    use the Raviart-Thomas finite elements of the specified order.

    int dim = nDimensions;

    shared_ptr<mfem::HypreParMatrix> A;
    HypreParMatrix Amat;
    Vector Xdebug;
    Vector X, B;
    ParBilinearForm *Ablock;
    ParLinearForm *ffform;

    FiniteElementCollection *hdivfree_coll;
    ParFiniteElementSpace *C_space;
    FiniteElementCollection *hdiv_coll;
    ParFiniteElementSpace *R_space;
    FiniteElementCollection *l2_coll;
    ParFiniteElementSpace *W_space;

    if (dim == 3)
    {
        hdivfree_coll = new ND_FECollection(feorder + 1, nDimensions);
        C_space = new ParFiniteElementSpace(pmesh.get(), hdivfree_coll);
    }
    else // dim == 4
    {
        hdivfree_coll = new DivSkew1_4DFECollection;
        C_space = new ParFiniteElementSpace(pmesh.get(), hdivfree_coll);

        //testing ProjectCoefficient
        VectorCoefficient * divfreepartcoeff = new VectorFunctionCoefficient(dim, DivmatFun4D_ex);
        ParGridFunction *u_exact = new ParGridFunction(C_space);
        u_exact->ProjectCoefficient(*divfreepartcoeff);//(*(Mytest.divfreepart));

        if (verbose)
            std::cout << "ProjectCoefficient is ok with vectors \n";
        //u_exact->Print();

        // checking projection error computation
        int order_quad = max(2, 2*feorder+1);
        const IntegrationRule *irs[Geometry::NumGeom];
        for (int i=0; i < Geometry::NumGeom; ++i)
        {
           irs[i] = &(IntRules.Get(i, order_quad));
        }

        double norm_u = ComputeGlobalLpNorm(2, *divfreepartcoeff, *pmesh, irs);
        double projection_error_u = u_exact->ComputeL2Error(*divfreepartcoeff, irs);

        if(verbose)
            if ( norm_u > MYZEROTOL )
                std::cout << "|| u_ex - Pi_h u_ex || / || u_ex || = " << projection_error_u / norm_u << "\n";
            else
                std::cout << "|| Pi_h u_ex || = " << projection_error_u << " (u_ex = 0) \n ";

        MPI_Finalize();
        return -1;
    }

    // how to make it working?
    //MFEM_ASSERT(dim == 3, "For now only 3D case is considered \n");
    if (nDimensions == 4)
    {
        if (verbose)
            std::cout << "4D case is not implemented - not a curl problem should be solved there! \n";
        MPI_Finalize();
        return -1;
    }


    FiniteElementCollection *h1_coll;
    ParFiniteElementSpace *H_space;
    if (withS)
    {
        h1_coll = new H1_FECollection(feorder+1, nDimensions);
        H_space = new ParFiniteElementSpace(pmesh.get(), h1_coll);
    }

    // the space for sigma in the original problem
    hdiv_coll = new RT_FECollection(feorder, nDimensions);
    R_space = new ParFiniteElementSpace(pmesh.get(), hdiv_coll);

    if (withDiv)
    {
        l2_coll = new L2_FECollection(feorder, nDimensions);
        W_space = new ParFiniteElementSpace(pmesh.get(), l2_coll);
    }

    int numblocks = 1;
    if (withS)
        numblocks++;

    Array<int> block_offsets(numblocks + 1); // number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = C_space->GetVSize();
    if (withS)
        block_offsets[2] = H_space->GetVSize();
    block_offsets.PartialSum();

    Array<int> block_trueOffsets(numblocks + 1); // number of variables + 1
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = C_space->TrueVSize();
    if (withS)
        block_trueOffsets[2] = H_space->TrueVSize();
    block_trueOffsets.PartialSum();

    HYPRE_Int dimC = C_space->GlobalTrueVSize();
    HYPRE_Int dimR = R_space->GlobalTrueVSize();
    HYPRE_Int dimH;
    if (withS)
        dimH = H_space->GlobalTrueVSize();
    if (verbose)
    {
       std::cout << "***********************************************************\n";
       std::cout << "dim(C) = " << dimC << "\n";
       if (withS)
           std::cout << "dim(H) = " << dimH << ", ";
       if (withS)
           std::cout << "dim(C+H) = " << dimC + dimH << "\n";
       if (withDiv)
           std::cout << "dim(R) = " << dimR << ", ";
       std::cout << "***********************************************************\n";
    }

    BlockVector xblks(block_offsets), rhsblks(block_offsets);
    BlockVector trueX(block_trueOffsets), trueRhs(block_trueOffsets);
    xblks = 0.0;
    rhsblks = 0.0;
    trueX = 0.0;
    trueRhs = 0.0;

    //VectorFunctionCoefficient f(dim, f_exact);
    //VectorFunctionCoefficient vone(dim, vone_exact);
    //VectorFunctionCoefficient vminusone(dim, vminusone_exact);
    //VectorFunctionCoefficient E(dim, E_exact);
    //VectorFunctionCoefficient curlE(dim, curlE_exact);

    //----------------------------------------------------------
    // Setting boundary conditions.
    //----------------------------------------------------------

    Array<int> ess_tdof_listU, ess_bdrU(pmesh->bdr_attributes.Max());
    ess_bdrU = 0;

    MFEM_ASSERT(pmesh->bdr_attributes.Max() == 3, "Remove before proceeding: are you sure about number of bdr attributes? \n");

    if (withS)
    {
        ess_bdrU[0] = 1;
        ess_bdrU[1] = 1;
        //ess_bdrU = 1;
        //ess_bdrU[pmesh->bdr_attributes.Max() - 1] = 0;
    }
    else
    {
        // correct, working
        ess_bdrU = 1;
        ess_bdrU[pmesh->bdr_attributes.Max() - 1] = 0;

        //ess_bdrU[0] = 1;
        //ess_bdrU[1] = 1;
    }
    C_space->GetEssentialTrueDofs(ess_bdrU, ess_tdof_listU);

    Array<int> ess_tdof_listS, ess_bdrS(pmesh->bdr_attributes.Max());
    ess_bdrS = 0;
    if (withS)
    {
        ess_bdrS = 0;
        ess_bdrS[0] = 1; // t = 0
        //ess_bdrS = 1;
        H_space->GetEssentialTrueDofs(ess_bdrS, ess_tdof_listS);
    }

    //VectorFunctionCoefficient * vzero = new VectorFunctionCoefficient(dim, zerovec_ex);
    //Coefficient *zero = new ConstantCoefficient(0.0);

    ParGridFunction * Sigmahat = new ParGridFunction(R_space);
    ParGridFunction * Temp = new ParGridFunction(R_space);
    if (withDiv)
    {
        if (verbose)
            std::cout << "Assembling linear system for finding sigmahat \n";

        ParGridFunction *sigma_exact;
        ParLinearForm *gform(new ParLinearForm);
        ParMixedBilinearForm *Bblock;
        HypreParMatrix *Bdiv, *BdivT;
        HypreParMatrix *BBT;
        HypreParVector *Rhs;

        sigma_exact = new ParGridFunction(R_space);
        sigma_exact->ProjectCoefficient(*(Mytest.sigma));

        gform = new ParLinearForm(W_space);
        gform->AddDomainIntegrator(new DomainLFIntegrator(*Mytest.scalardivsigma));
        gform->Assemble();

        Bblock = new ParMixedBilinearForm(R_space, W_space);
        Bblock->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
        Bblock->Assemble();
        Bblock->EliminateTrialDofs(ess_bdrU, *sigma_exact, *gform);

        Bblock->Finalize();
        Bdiv = Bblock->ParallelAssemble();
        BdivT = Bdiv->Transpose();
        BBT = ParMult(Bdiv, BdivT);
        Rhs = gform->ParallelAssemble();

        HypreBoomerAMG * invBBT = new HypreBoomerAMG(*BBT);
        invBBT->SetPrintLevel(0);

        mfem::CGSolver solver(comm);
        solver.SetPrintLevel(1);
        solver.SetMaxIter(70000);
        solver.SetRelTol(1.0e-16);
        solver.SetAbsTol(1.0e-16);
        solver.SetPreconditioner(*invBBT);
        solver.SetOperator(*BBT);

        ParGridFunction * Temphat = new ParGridFunction(W_space);
        solver.Mult(*Rhs, *Temphat);

        BdivT->Mult(*Temphat, *Temp);

        Sigmahat->Distribute(*Temp);
        //Sigmahat->SetFromTrueDofs(*Temp);
    }
    else
    {
        Sigmahat->ProjectCoefficient(*(Mytest.sigmahat));
    }

    ParGridFunction *u_exact = new ParGridFunction(C_space);
    u_exact->ProjectCoefficient(*(Mytest.divfreepart));

    ParGridFunction *S_exact;
    if (withS)
    {
        S_exact = new ParGridFunction(H_space);
        S_exact->ProjectCoefficient(*(Mytest.scalarS));
    }

    if (blockedversion || withS)
    {
        if (withDiv)
            xblks.GetBlock(0) = 0.0;
        else
            xblks.GetBlock(0) = *u_exact;
    }
    if (withS)
        xblks.GetBlock(1) = *S_exact;


    ffform = new ParLinearForm(C_space);
    if (!withDiv)
    {
        if (withS)
        {
            ffform->Update(C_space, rhsblks.GetBlock(0), 0);
#ifdef DEBUG
            ParMixedBilinearForm * Tblock = new ParMixedBilinearForm(R_space, C_space);
            Tblock->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator);
            Tblock->Assemble();
            Tblock->EliminateTestDofs(ess_bdrU);
            Tblock->Finalize();
            //HypreParMatrix * T = Tblock->ParallelAssemble();
            *Sigmahat *= -1.0;
            Tblock->Mult(*Sigmahat, *ffform);
            //T->Mult(*Sigmahat, *ffform);
            *Sigmahat *= -1.0;
#else
            ffform->AddDomainIntegrator(new VectorcurlDomainLFIntegrator(*(Mytest.minsigmahat)));
            ffform->Assemble();
#endif
        }
        else
        {
            if (blockedversion)
                ffform->Update(C_space, rhsblks.GetBlock(0), 0);
            //else
                //ffform->Update(C_space, *u_exact, 0);
#ifdef DEBUG
            ParMixedBilinearForm * Tblock = new ParMixedBilinearForm(R_space, C_space);
            Tblock->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(*Mytest.Ktilda));
            Tblock->Assemble();
            Tblock->EliminateTestDofs(ess_bdrU);
            Tblock->Finalize();
            *Sigmahat *= -1.0;

            Tblock->Mult(*Sigmahat, *ffform);

            *Sigmahat *= -1.0;
#else
            ffform->AddDomainIntegrator(new VectorcurlDomainLFIntegrator(*(Mytest.minKsigmahat)));
            ffform->Assemble();
#endif
        }
    }
    else // if withDiv = true
    {
        if (withS)
        {
            ParMixedBilinearForm * Tblock = new ParMixedBilinearForm(R_space, C_space);
            Tblock->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator);
            Tblock->Assemble();
            Tblock->EliminateTestDofs(ess_bdrU);
            Tblock->Finalize();
            //HypreParMatrix * T = Tblock->ParallelAssemble();
            *Sigmahat *= -1.0;
            Tblock->Mult(*Sigmahat, *ffform);
            //T->Mult(*Sigmahat, *ffform);
            *Sigmahat *= -1.0;
        }
        else
        {
            //if (print_progress_report)
                //std::cout << "withHdiv is not implemented for withS = false case \n";

            if (blockedversion)
                ffform->Update(C_space, rhsblks.GetBlock(0), 0);

            ParMixedBilinearForm * Tblock = new ParMixedBilinearForm(R_space, C_space);
            Tblock->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(*Mytest.Ktilda));
            Tblock->Assemble();
            Tblock->EliminateTestDofs(ess_bdrU);
            Tblock->Finalize();
            *Sigmahat *= -1.0;
            Tblock->Mult(*Sigmahat, *ffform);
            *Sigmahat *= -1.0;
        }
    }

    ParLinearForm *qform(new ParLinearForm);
    ParGridFunction * tmp_to_add;
    if (withS)
    {
        if (!withDiv)
        {
            qform->Update(H_space, rhsblks.GetBlock(1), 0);
            qform->AddDomainIntegrator(new GradDomainLFIntegrator(*Mytest.bdivsigma));
#ifdef DEBUG
            ParMixedBilinearForm * LTblock = new ParMixedBilinearForm(H_space, R_space);
            LTblock->AddDomainIntegrator(new MixedVectorProductIntegrator(*(Mytest.b)));
            LTblock->Assemble();
            LTblock->EliminateTestDofs(ess_bdrS);
            //LTblock->EliminateTrialDofs(ess_bdrU);
            LTblock->Finalize();
            //HypreParMatrix * LT = LTblock->ParallelAssemble();
            //tmp_to_add = new ParGridFunction(H_space);
            //LT->MultTranspose(*Sigmahat, *tmp_to_add);
            tmp_to_add = new ParGridFunction(H_space);
            LTblock->MultTranspose(*Sigmahat, *tmp_to_add);
#else
            qform->AddDomainIntegrator(new DomainLFIntegrator(*Mytest.bsigmahat));
#endif
            qform->Assemble();
#ifdef DEBUG
            *qform += *tmp_to_add;
#endif
        }
        else
        {
            //if (print_progress_report)
                //std::cout << "A required change of rh side qform is not implemented yet for withDiv case \n";
            qform->Update(H_space, rhsblks.GetBlock(1), 0);
            qform->AddDomainIntegrator(new GradDomainLFIntegrator(*Mytest.bdivsigma));
            qform->Assemble();

            ParMixedBilinearForm * LTblock = new ParMixedBilinearForm(H_space, R_space);
            LTblock->AddDomainIntegrator(new MixedVectorProductIntegrator(*(Mytest.b)));
            LTblock->Assemble();
            LTblock->EliminateTestDofs(ess_bdrS);
            //LTblock->EliminateTrialDofs(ess_bdrU);
            LTblock->Finalize();

            tmp_to_add = new ParGridFunction(H_space);
            LTblock->MultTranspose(*Sigmahat, *tmp_to_add);

            *qform += *tmp_to_add;
        }
    }

    Ablock = new ParBilinearForm(C_space);
    if (withS)
    {
        Coefficient *one = new ConstantCoefficient(1.0);
        Ablock->AddDomainIntegrator(new CurlCurlIntegrator(*one));
        Ablock->Assemble();
        Ablock->EliminateEssentialBC(ess_bdrU,xblks.GetBlock(0),*ffform);
        Ablock->Finalize();
        HypreParMatrix * tempA = Ablock->ParallelAssemble();
        A = make_shared<HypreParMatrix>(*tempA);
    }
    else //if (!withS)
    {
        Ablock->AddDomainIntegrator(new CurlCurlIntegrator(*(Mytest.Ktilda)));
        Ablock->Assemble();
        if (blockedversion)
        {
            Ablock->EliminateEssentialBC(ess_bdrU,xblks.GetBlock(0),*ffform);
            Ablock->Finalize();
            HypreParMatrix * tempA = Ablock->ParallelAssemble();
            A = make_shared<HypreParMatrix>(*tempA);
        }
    }

    ParBilinearForm *Cblock;
    HypreParMatrix *C;

    if (withS)
    {
        Cblock = new ParBilinearForm(H_space);
        Cblock->AddDomainIntegrator(new MassIntegrator(*Mytest.bTb));
        Cblock->AddDomainIntegrator(new DiffusionIntegrator(*Mytest.bbT));
        Cblock->Assemble();
        Cblock->EliminateEssentialBC(ess_bdrS, xblks.GetBlock(1),*qform);
        Cblock->Finalize();
        C = Cblock->ParallelAssemble();
    }

    ParMixedBilinearForm *CHblock;
    HypreParMatrix *CH, *CHT;

    if (withS)
    {
        CHblock = new ParMixedBilinearForm(C_space, H_space);
        CHblock->AddDomainIntegrator(new VectorFECurlVQIntegrator(*Mytest.minb));
        CHblock->Assemble();
        CHblock->EliminateTestDofs(ess_bdrS);

        CHblock->EliminateTrialDofs(ess_bdrU, xblks.GetBlock(0), *qform);

        CHblock->Finalize();
        CH = CHblock->ParallelAssemble();
        CHT = CH->Transpose();
    }

    if (blockedversion || withS)
    {
        ffform->ParallelAssemble(trueRhs.GetBlock(0));
    }
    if (withS)
    {
        qform->ParallelAssemble(trueRhs.GetBlock(1));
    }

    BlockOperator *MainOp = new BlockOperator(block_trueOffsets);

    if (withS)
    {
        MainOp->SetBlock(0,0, A.get());
        MainOp->SetBlock(0,1, CHT);
        MainOp->SetBlock(1,0, CH);
        MainOp->SetBlock(1,1, C);
    }
    else
    {
        if (blockedversion)
            MainOp->SetBlock(0,0, A.get());
        else
        {
            Ablock->FormLinearSystem(ess_tdof_listU, *u_exact, *ffform, Amat, Xdebug, B);
            MainOp->SetBlock(0,0, &Amat);
        }
    }

    if (verbose)
        cout << "Discretized problem is assembled" << endl << flush;

    chrono.Clear();
    chrono.Start();

    Solver *prec;
    if (with_prec)
    {
        if(dim<=3)
        {
            if (prec_is_MG)
            {
                if (verbose)
                    cout << "MG prec is not implemented" << endl;
                MPI_Finalize();
                return 0;

                //int formcurl = 1; // for H(curl)
                //prec = new MG3dPrec(&Amat, nlevels, coarsenfactor, pmesh.get(), formcurl, feorder, C_space, ess_tdof_listU, verbose);
            }
        }
        else // if(dim==4)
        {
            if (prec_is_MG)
            {
                if (verbose)
                    cout << "MG prec is not implemented in 4D" << endl;
                MPI_Finalize();
                return 0;
            }
        }

        if (verbose)
            cout << "Preconditioner is ready" << endl << flush;
    }
    else
        if (verbose)
            cout << "Using no preconditioner" << endl << flush;

    IterativeSolver * solver;
    solver = new CGSolver(comm);
    if (verbose)
        cout << "Linear solver: CG" << endl << flush;

    solver->SetAbsTol(atol);
    solver->SetRelTol(rtol);
    solver->SetMaxIter(max_num_iter);
    solver->SetOperator(*MainOp);

    if (!withS && !blockedversion)
    {
        X.SetSize(MainOp->Height());
        X = 0.0;
    }

    if (with_prec)
    {
        if (withS)
        {
            if (verbose)
                cout << "Hcurl-H1 case doesn't have a preconditioner currently" << endl;
        }
        else
            solver->SetPreconditioner(*prec);
    }
    solver->SetPrintLevel(0);
    if (withS || blockedversion )
    {
        trueX = 0.0;
        //trueRhs.GetBlock(1).Print();
        solver->Mult(trueRhs, trueX);
    }
    else
        solver->Mult(B, X);
    chrono.Stop();

    if (verbose)
    {
       if (solver->GetConverged())
          std::cout << "Linear solver converged in " << solver->GetNumIterations()
                    << " iterations with a residual norm of " << solver->GetFinalNorm() << ".\n";
       else
          std::cout << "Linear solver did not converge in " << solver->GetNumIterations()
                    << " iterations. Residual norm is " << solver->GetFinalNorm() << ".\n";
       std::cout << "Linear solver took " << chrono.RealTime() << "s. \n";
    }

    ParGridFunction * u = new ParGridFunction(C_space);
    ParGridFunction * S;
    if (blockedversion || withS)
    {
        u->Distribute(&(trueX.GetBlock(0)));
        if (withS)
        {
            S = new ParGridFunction(H_space);
            S->Distribute(&(trueX.GetBlock(1)));
        }
    }
    else
        Ablock->RecoverFEMSolution(X, *ffform, *u);

    // 13. Extract the parallel grid function corresponding to the finite element
    //     approximation X. This is the local solution on each processor. Compute
    //     L2 error norms.

    int order_quad = max(2, 2*feorder+1);
    const IntegrationRule *irs[Geometry::NumGeom];
    for (int i=0; i < Geometry::NumGeom; ++i)
    {
       irs[i] = &(IntRules.Get(i, order_quad));
    }

    double err_u = u->ComputeL2Error(*(Mytest.divfreepart), irs);
    double norm_u = ComputeGlobalLpNorm(2, *(Mytest.divfreepart), *pmesh, irs);

    if (verbose)
        if ( norm_u > MYZEROTOL )
        {
            std::cout << "norm_u = " << norm_u << "\n";
            cout << "|| u - u_ex || / || u_ex || = " << err_u / norm_u << endl;
        }
        else
            cout << "|| u || = " << err_u << " (u_ex = 0)" << endl;

    // Computing error for S
    double err_S, norm_S;
    if (withS)
    {
        err_S = S->ComputeL2Error(*(Mytest.scalarS), irs);
        norm_S = ComputeGlobalLpNorm(2, *(Mytest.scalarS), *pmesh, irs);
        if (verbose)
        {
            if ( norm_S > MYZEROTOL )
                std::cout << "|| S_h - S_ex || / || S_ex || = " <<
                         err_S / norm_S << "\n";
            else
                std::cout << "|| S_h || = " << err_S << " (S_ex = 0) \n";
        }
    }

    ParGridFunction * sigmahat = new ParGridFunction(R_space);
    sigmahat->ProjectCoefficient(*(Mytest.sigmahat));

    ParGridFunction * curlu = new ParGridFunction(R_space);
    DiscreteLinearOperator Curl_h(C_space, R_space);
    Curl_h.AddDomainInterpolator(new CurlInterpolator());
    Curl_h.Assemble();
    Curl_h.Mult(*u, *curlu);

    ParGridFunction * curlu_exact = new ParGridFunction(R_space);
    curlu_exact->ProjectCoefficient(*(Mytest.opdivfreepart));

    double err_curlu = curlu->ComputeL2Error(*(Mytest.opdivfreepart), irs);
    double norm_curlu = ComputeGlobalLpNorm(2, *(Mytest.opdivfreepart), *pmesh, irs);

    if (verbose)
        if ( norm_curlu > MYZEROTOL )
        {
            cout << "|| curlu_ex || = " << norm_curlu << endl;
            cout << "|| curl_h u_h - curlu_ex || / || curlu_ex || = " << err_curlu / norm_curlu << endl;
        }
        else
            cout << "|| curl_h u_h || = " << err_curlu << " (curlu_ex = 0)" << endl;

    // not needed
    //ParGridFunction *sigma_nonhomo = new ParGridFunction(R_space);
    //sigma_nonhomo->ProjectCoefficient(*(Mytest.sigma_nonhomo));

    ParGridFunction * sigma = new ParGridFunction(R_space);
    *sigma = *sigmahat;
    *sigma += * curlu;
    //*sigma += *sigma_nonhomo; // not needed

    ParGridFunction * sigma_exact = new ParGridFunction(R_space);
    sigma_exact->ProjectCoefficient(*(Mytest.sigmahat));

    double err_sigma = sigma->ComputeL2Error(*(Mytest.sigma), irs);
    double norm_sigma = ComputeGlobalLpNorm(2, *(Mytest.sigma), *pmesh, irs);

    if (verbose)
        cout << "sigma_h = sigma_hat + div-free part, div-free part = curl u_h \n";

    if (verbose)
        if ( norm_sigma > MYZEROTOL )
            cout << "|| sigma_h - sigma_ex || / || sigma_ex || = " << err_sigma / norm_sigma << endl;
        else
            cout << "|| sigma || = " << err_sigma << " (sigma_ex = 0)" << endl;

    double err_sigmahat = sigmahat->ComputeL2Error(*(Mytest.sigma), irs);

    if (verbose)
        if ( norm_sigma > MYZEROTOL )
            cout << "|| sigma_hat - sigma_ex || / || sigma_ex || = " << err_sigmahat / norm_sigma << endl;
        else
            cout << "|| sigma_hat || = " << err_sigmahat << " (sigma_ex = 0)" << endl;


    if (withS)
    {
        S_exact = new ParGridFunction(H_space);
        S_exact->ProjectCoefficient(*(Mytest.scalarS));

        err_S = S->ComputeL2Error(*(Mytest.scalarS), irs);
        norm_S = ComputeGlobalLpNorm(2, *(Mytest.scalarS), *pmesh, irs);
        if (verbose)
        {
            if ( norm_S > MYZEROTOL )
                std::cout << "|| S_h - S_ex || / || S_ex || = " <<
                         err_S / norm_S << "\n";
            else
                std::cout << "|| S_h || = " << err_S << " (S_ex = 0) \n";
        }
    }


    if (verbose)
        cout << "Computing projection errors" << endl;

    //double projection_error_u = u_exact->ComputeL2Error(E, irs);
    double projection_error_u = u_exact->ComputeL2Error(*(Mytest.divfreepart), irs);

    if(verbose)
        if ( norm_u > MYZEROTOL )
        {
            //std::cout << "Debug: || u_ex || = " << norm_u << "\n";
            //std::cout << "Debug: proj error = " << projection_error_u << "\n";
            cout << "|| u_ex - Pi_h u_ex || / || u_ex || = " << projection_error_u / norm_u << endl;
        }
        else
            cout << "|| Pi_h u_ex || = " << projection_error_u << " (u_ex = 0) \n ";

    if (withS)
    {
        double projection_error_S = S_exact->ComputeL2Error(*(Mytest.scalarS), irs);

        if(verbose)
            if ( norm_S > MYZEROTOL )
                cout << "|| S_ex - Pi_h S_ex || / || S_ex || = " << projection_error_S / norm_S << endl;
            else
                cout << "|| Pi_h S_ex ||  = " << projection_error_S << " (S_ex = 0) \n";
    }

    if (visualization && nDimensions < 4)
    {
       char vishost[] = "localhost";
       int  visport   = 19916;


       socketstream uex_sock(vishost, visport);
       uex_sock << "parallel " << num_procs << " " << myid << "\n";
       uex_sock.precision(8);
       uex_sock << "solution\n" << *pmesh << *u_exact << "window_title 'u_exact'"
              << endl;
       socketstream uh_sock(vishost, visport);
       uh_sock << "parallel " << num_procs << " " << myid << "\n";
       uh_sock.precision(8);
       uh_sock << "solution\n" << *pmesh << *u << "window_title 'u_h'"
              << endl;

       *u -= *u_exact;
       socketstream udiff_sock(vishost, visport);
       udiff_sock << "parallel " << num_procs << " " << myid << "\n";
       udiff_sock.precision(8);
       udiff_sock << "solution\n" << *pmesh << *u << "window_title 'u_h - u_exact'"
              << endl;


       socketstream curluex_sock(vishost, visport);
       curluex_sock << "parallel " << num_procs << " " << myid << "\n";
       curluex_sock.precision(8);
       curluex_sock << "solution\n" << *pmesh << *curlu_exact << "window_title 'curl u_exact'"
              << endl;

       socketstream curlu_sock(vishost, visport);
       curlu_sock << "parallel " << num_procs << " " << myid << "\n";
       curlu_sock.precision(8);
       curlu_sock << "solution\n" << *pmesh << *curlu << "window_title 'curl u_h'"
              << endl;

       *curlu -= *curlu_exact;
       socketstream curludiff_sock(vishost, visport);
       curludiff_sock << "parallel " << num_procs << " " << myid << "\n";
       curludiff_sock.precision(8);
       curludiff_sock << "solution\n" << *pmesh << *curlu << "window_title 'curl u_h - curl u_exact'"
              << endl;

       if (withS)
       {
           socketstream S_ex_sock(vishost, visport);
           S_ex_sock << "parallel " << num_procs << " " << myid << "\n";
           S_ex_sock.precision(8);
           S_ex_sock << "solution\n" << *pmesh << *S_exact << "window_title 'S_exact'"
                  << endl;

           socketstream S_h_sock(vishost, visport);
           S_h_sock << "parallel " << num_procs << " " << myid << "\n";
           S_h_sock.precision(8);
           S_h_sock << "solution\n" << *pmesh << *S << "window_title 'S_h'"
                  << endl;

           *S -= *S_exact;
           socketstream S_diff_sock(vishost, visport);
           S_diff_sock << "parallel " << num_procs << " " << myid << "\n";
           S_diff_sock.precision(8);
           S_diff_sock << "solution\n" << *pmesh << *S << "window_title 'S_h - S_exact'"
                  << endl;
        }

       MPI_Barrier(pmesh->GetComm());
    }

    // 17. Free the used memory.
    delete ffform;
    if (withS)
        delete qform;

    if (withS || blockedversion)
    {
        delete Ablock;
        if (withS)
        {
            delete Cblock;
            delete CHblock;
        }
    }

    delete C_space;
    delete hdivfree_coll;
    delete R_space;
    delete hdiv_coll;
    if (withS)
    {
       delete H_space;
       delete h1_coll;
    }

    MPI_Finalize();
    return 0;
}

template <void (*bvecfunc)(const Vector&, Vector& )> \
void KtildaTemplate(const Vector& xt, DenseMatrix& Ktilda)
{
    int nDimensions = xt.Size();
    Ktilda.SetSize(nDimensions);
    Vector b;
    bvecfunc(xt,b);
    double bTbInv = (-1./(b*b));
    Ktilda.Diag(1.0,nDimensions);
#ifndef K_IDENTITY
    AddMult_a_VVt(bTbInv,b,Ktilda);
#endif
}

template <void (*bvecfunc)(const Vector&, Vector& )> \
void bbTTemplate(const Vector& xt, DenseMatrix& bbT)
{
    Vector b;
    bvecfunc(xt,b);
    MultVVt(b, bbT);
}


template <double (*S)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )> \
void sigmaTemplate(const Vector& xt, Vector& sigma)
{
    Vector b;
    bvecfunc(xt, b);
    sigma.SetSize(xt.Size());
    sigma(xt.Size()-1) = S(xt);
    for (int i = 0; i < xt.Size()-1; i++)
        sigma(i) = b(i) * sigma(xt.Size()-1);

    return;
}

template <double (*S)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )> \
void sigmaNonHomoTemplate(const Vector& xt, Vector& sigma) // sigmaNonHomo = (b S0, S0)^T for S0 = S(t=0)
{
    sigma.SetSize(xt.Size());

    Vector xteq0;
    xteq0.SetSize(xt.Size()); // xt with t = 0
    xteq0 = xt;
    xteq0(xteq0.Size()-1) = 0.0;

    sigmaTemplate<S, bvecfunc>(xteq0, sigma);
/*
    Vector b;
    bvecfunc(xt, b);
    sigma.SetSize(xt.Size());
    sigma(xt.Size()-1) = ufunc(xteq0);
    for (int i = 0; i < xt.Size()-1; i++)
        sigma(i) = b(i) * sigma(xt.Size()-1);
*/
    return;
}


template <void (*bvecfunc)(const Vector&, Vector& )> \
double bTbTemplate(const Vector& xt)
{
    Vector b;
    bvecfunc(xt,b);
    return b*b;
}

template <double (*S)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )> \
double minbTbSnonhomoTemplate(const Vector& xt)
{
    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    return - bTbTemplate<bvecfunc>(xt) * S(xt0);
}



template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
double divsigmaTemplate(const Vector& xt)
{
    Vector b;
    bvec(xt,b);

    Vector gradS;
    Sgradxvec(xt,gradS);

    double res = 0.0;

    res += dSdt(xt);
    for ( int i= 0; i < xt.Size() - 1; ++i )
        res += b(i) * gradS(i);
    res += divbfunc(xt) * S(xt);

    return res;
}

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
double rhsideTemplate(const Vector& xt)
{
    Vector b;
    bvec(xt,b);

    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    Vector gradS0;
    Sgradxvec(xt0,gradS0);

    double res = divsigmaTemplate<S, dSdt, Sgradxvec, bvec, divbfunc>(xt);

    for ( int i= 0; i < xt.Size() - 1; ++i )
        res += b(i) * (- gradS0(i));
    res += divbfunc(xt) * ( - S(xt0));

    return res;
}

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
void bfTemplate(const Vector& xt, Vector& bf)
{
    bf.SetSize(xt.Size());

    Vector b;
    bvec(xt,b);

    double f = rhsideTemplate<S, dSdt, Sgradxvec, bvec, divbfunc>(xt);

    for (int i = 0; i < bf.Size(); ++i)
        bf(i) = f * b(i);
}

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
void bdivsigmaTemplate(const Vector& xt, Vector& bdivsigma)
{
    bdivsigma.SetSize(xt.Size());

    Vector b;
    bvec(xt,b);

    double divsigma = divsigmaTemplate<S, dSdt, Sgradxvec, bvec, divbfunc>(xt);

    for (int i = 0; i < bdivsigma.Size(); ++i)
        bdivsigma(i) = divsigma * b(i);
}


template<double (*S)(const Vector & xt), void(*bvec)(const Vector & x, Vector & vec)>
void bSnonhomoTemplate(const Vector& xt, Vector& bSnonhomo)
{
    bSnonhomo.SetSize(xt.Size());

    Vector b;
    bvec(xt,b);

    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    for (int i = 0; i < bSnonhomo.Size(); ++i)
        bSnonhomo(i) = S(xt0) * b(i);
}

template<void(*bvec)(const Vector & x, Vector & vec)>
void minbTemplate(const Vector& xt, Vector& minb)
{
    minb.SetSize(xt.Size());

    bvec(xt,minb);

    minb *= -1;
}

template<double (*S)(const Vector & xt) > double SnonhomoTemplate(const Vector& xt)
{
    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    return S(xt0);
}



double uFun_ex(const Vector& xt)
{
    double t = xt(xt.Size()-1);
    //return t;
    ////setback
    return sin(t)*exp(t);
}

double uFun_ex_dt(const Vector& xt)
{
    double t = xt(xt.Size()-1);
    return (cos(t) + sin(t)) * exp(t);
}

void uFun_ex_gradx(const Vector& xt, Vector& gradx )
{
    gradx.SetSize(xt.Size() - 1);
    gradx = 0.0;
}

void bFun_ex(const Vector& xt, Vector& b )
{
    b.SetSize(xt.Size());

    //for (int i = 0; i < xt.Size()-1; i++)
        //b(i) = xt(i) * (1 - xt(i));

    //if (xt.Size() == 4)
        //b(2) = 1-cos(2*xt(2)*M_PI);
        //b(2) = sin(xt(2)*M_PI);
        //b(2) = 1-cos(xt(2)*M_PI);

    b(0) = sin(xt(0)*2*M_PI)*cos(xt(1)*M_PI);
    b(1) = sin(xt(1)*M_PI)*cos(xt(0)*M_PI);
    b(2) = 1-cos(2*xt(2)*M_PI);

    b(xt.Size()-1) = 1.;

}

double bFundiv_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);
    if (xt.Size() == 4)
        return 2*M_PI * cos(x*2*M_PI)*cos(y*M_PI) + M_PI * cos(y*M_PI)*cos(x*M_PI) + 2*M_PI * sin(2*z*M_PI);
    if (xt.Size() == 3)
        return 2*M_PI * cos(x*2*M_PI)*cos(y*M_PI) + M_PI * cos(y*M_PI)*cos(x*M_PI);
    return 0.0;
}


void bFunRect2D_ex(const Vector& xt, Vector& b )
{
    double x = xt(0);
    double y = xt(1);

    b.SetSize(xt.Size());

    b(0) = sin(x*M_PI)*cos(y*M_PI);
    b(1) = - sin(y*M_PI)*cos(x*M_PI);

    b(xt.Size()-1) = 1.;

}

double bFunRect2Ddiv_ex(const Vector& xt)
{
    return 0.0;
}

void bFunCube3D_ex(const Vector& xt, Vector& b )
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);

    b.SetSize(xt.Size());

    b(0) = sin(x*M_PI)*cos(y*M_PI)*cos(z*M_PI);
    b(1) = - 0.5 * sin(y*M_PI)*cos(x*M_PI) * cos(z*M_PI);
    b(2) = - 0.5 * sin(z*M_PI)*cos(x*M_PI) * cos(y*M_PI);

    b(xt.Size()-1) = 1.;

}

double bFunCube3Ddiv_ex(const Vector& xt)
{
    return 0.0;
}

void bFunSphere3D_ex(const Vector& xt, Vector& b )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    b.SetSize(xt.Size());

    b(0) = -y;  // -x2
    b(1) = x;   // x1
    b(2) = 0.0;

    b(xt.Size()-1) = 1.;
    return;
}

double bFunSphere3Ddiv_ex(const Vector& xt)
{
    return 0.0;
}


double uFun2_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    return t * exp(t) * sin (((x - 0.5)*(x - 0.5) + y*y));
}

double uFun2_ex_dt(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    return (1.0 + t) * exp(t) * sin (((x - 0.5)*(x - 0.5) + y*y));
}

void uFun2_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = t * exp(t) * 2.0 * (x - 0.5) * cos (((x - 0.5)*(x - 0.5) + y*y));
    gradx(1) = t * exp(t) * 2.0 * y         * cos (((x - 0.5)*(x - 0.5) + y*y));
}

/*
double fFun2(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    Vector b(3);
    bFunCircle2D_ex(xt,b);
    return (t + 1) * exp(t) * sin (((x - 0.5)*(x - 0.5) + y*y)) +
             t * exp(t) * 2.0 * (x - 0.5) * cos (((x - 0.5)*(x - 0.5) + y*y)) * b(0) +
             t * exp(t) * 2.0 * y         * cos (((x - 0.5)*(x - 0.5) + y*y)) * b(1);
}
*/

void bFunCircle2D_ex(const Vector& xt, Vector& b )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    b.SetSize(xt.Size());

    b(0) = -y;  // -x2
    b(1) = x;   // x1

    b(xt.Size()-1) = 1.;
    return;
}

double bFunCircle2Ddiv_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    return 0.0;
}


double uFun3_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);
    return sin(t)*exp(t) * sin ( M_PI * (x + y + z));
}

double uFun3_ex_dt(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);
    return (sin(t) + cos(t)) * exp(t) * sin ( M_PI * (x + y + z));
}

void uFun3_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = sin(t) * exp(t) * M_PI * cos ( M_PI * (x + y + z));
    gradx(1) = sin(t) * exp(t) * M_PI * cos ( M_PI * (x + y + z));
    gradx(2) = sin(t) * exp(t) * M_PI * cos ( M_PI * (x + y + z));
}


/*
double fFun3(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);
    Vector b(4);
    bFun_ex(xt,b);

    return (cos(t)*exp(t)+sin(t)*exp(t)) * sin ( M_PI * (x + y + z)) +
            sin(t)*exp(t) * M_PI * cos ( M_PI * (x + y + z)) * b(0) +
            sin(t)*exp(t) * M_PI * cos ( M_PI * (x + y + z)) * b(1) +
            sin(t)*exp(t) * M_PI * cos ( M_PI * (x + y + z)) * b(2) +
            (2*M_PI*cos(x*2*M_PI)*cos(y*M_PI) +
             M_PI*cos(y*M_PI)*cos(x*M_PI)+
             + 2*M_PI*sin(z*2*M_PI)) * uFun3_ex(xt);
}
*/

double uFun4_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    return exp(t) * sin (((x - 0.5)*(x - 0.5) + y*y));
    //return t * exp(t) * sin (((x - 0.5)*(x - 0.5) + y*y)) + 5.0 * (x + y);
}

double uFun4_ex_dt(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    return uFun4_ex(xt);
    //return (1 + t) * exp(t) * sin (((x - 0.5)*(x - 0.5) + y*y));
}

void uFun4_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = exp(t) * 2.0 * (x - 0.5) * cos (((x - 0.5)*(x - 0.5) + y*y));
    gradx(1) = exp(t) * 2.0 * y         * cos (((x - 0.5)*(x - 0.5) + y*y));
    //gradx(0) = t * exp(t) * 2.0 * (x - 0.5) * cos (((x - 0.5)*(x - 0.5) + y*y)) + 5.0;
    //gradx(1) = t * exp(t) * 2.0 * y         * cos (((x - 0.5)*(x - 0.5) + y*y)) + 5.0;
}

double uFun33_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    return exp(t) * sin (((x - 0.5)*(x - 0.5) + y*y + (z -0.25)*(z-0.25) ));
}

double uFun33_ex_dt(const Vector& xt)
{
    return uFun33_ex(xt);
}

void uFun33_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = exp(t) * 2.0 * (x - 0.5) * cos (((x - 0.5)*(x - 0.5) + y*y + (z -0.25)*(z-0.25)));
    gradx(1) = exp(t) * 2.0 * y         * cos (((x - 0.5)*(x - 0.5) + y*y + (z -0.25)*(z-0.25)));
    gradx(2) = exp(t) * 2.0 * (z -0.25) * cos (((x - 0.5)*(x - 0.5) + y*y + (z -0.25)*(z-0.25)));
}

double uFun5_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    if ( t < MYZEROTOL)
        return exp(-100.0 * ((x - 0.5) * (x - 0.5) + y * y));
    else
        return 0.0;
}

double uFun5_ex_dt(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    return 0.0;
}

void uFun5_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = -100.0 * 2.0 * (x - 0.5) * uFun5_ex(xt);
    gradx(1) = -100.0 * 2.0 * y * uFun5_ex(xt);
}


double uFun6_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    return exp(-100.0 * ((x - 0.5) * (x - 0.5) + y * y)) * exp(-10.0*t);
}

double uFun6_ex_dt(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    return -10.0 * uFun6_ex(xt);
}

void uFun6_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = -100.0 * 2.0 * (x - 0.5) * uFun6_ex(xt);
    gradx(1) = -100.0 * 2.0 * y * uFun6_ex(xt);
}


double GaussianHill(const Vector&xvec)
{
    double x = xvec(0);
    double y = xvec(1);
    return exp(-100.0 * ((x - 0.5) * (x - 0.5) + y * y));
}

double uFunCylinder_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double r = sqrt(x*x + y*y);
    double teta = atan(y/x);
    /*
    if (fabs(x) < MYZEROTOL && y > 0)
        teta = M_PI / 2.0;
    else if (fabs(x) < MYZEROTOL && y < 0)
        teta = - M_PI / 2.0;
    else
        teta = atan(y,x);
    */
    double t = xt(xt.Size()-1);
    Vector xvec(2);
    xvec(0) = r * cos (teta - t);
    xvec(1) = r * sin (teta - t);
    return GaussianHill(xvec);
}

double uFunCylinder_ex_dt(const Vector& xt)
{
    return 0.0;
}

void uFunCylinder_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = 0.0;
    gradx(1) = 0.0;
}


double uFun66_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);
    return exp(-100.0 * ((x - 0.5) * (x - 0.5) + y * y + (z - 0.25)*(z - 0.25))) * exp(-10.0*t);
}

double uFun66_ex_dt(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);
    return -10.0 * uFun6_ex(xt);
}

void uFun66_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = -100.0 * 2.0 * (x - 0.5)  * uFun6_ex(xt);
    gradx(1) = -100.0 * 2.0 * y          * uFun6_ex(xt);
    gradx(2) = -100.0 * 2.0 * (z - 0.25) * uFun6_ex(xt);
}

void zerovecx_ex(const Vector& xt, Vector& zerovecx )
{
    zerovecx.SetSize(xt.Size() - 1);
    zerovecx = 0.0;
}

void zerovec_ex(const Vector& xt, Vector& vecvalue)
{
    vecvalue.SetSize(xt.Size());

    //vecvalue(0) = -y * (1 - t);
    //vecvalue(1) = x * (1 - t);
    //vecvalue(2) = 0;
    //vecvalue(0) = x * (1 - x);
    //vecvalue(1) = y * (1 - y);
    //vecvalue(2) = t * (1 - t);

    // Martin's function
    vecvalue = 0.0;

    return;
}

double zero_ex(const Vector& xt)
{
    return 0.0;
}

////////////////
void hcurlFun3D_ex(const Vector& xt, Vector& vecvalue)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    vecvalue.SetSize(xt.Size());

    //vecvalue(0) = -y * (1 - t);
    //vecvalue(1) = x * (1 - t);
    //vecvalue(2) = 0;
    //vecvalue(0) = x * (1 - x);
    //vecvalue(1) = y * (1 - y);
    //vecvalue(2) = t * (1 - t);

    // Martin's function
    vecvalue(0) = sin(kappa * xt(1));
    vecvalue(1) = sin(kappa * xt(2));
    vecvalue(2) = sin(kappa * xt(0));

    return;
}

void curlhcurlFun3D_ex(const Vector& xt, Vector& vecvalue)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    vecvalue.SetSize(xt.Size());

    //vecvalue(0) = 0.0;
    //vecvalue(1) = 0.0;
    //vecvalue(2) = -2.0 * (1 - t);

    // Martin's function's curl
    vecvalue(0) = - kappa * cos(kappa * xt(2));
    vecvalue(1) = - kappa * cos(kappa * xt(0));
    vecvalue(2) = - kappa * cos(kappa * xt(1));

    return;
}

////////////////
void DivmatFun4D_ex(const Vector& xt, Vector& vecvalue)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    vecvalue.SetSize(xt.Size());

    // 4D counterpart of the Martin's 3D function
    //std::cout << "Error: DivmatFun4D_ex is incorrect \n";
    vecvalue(0) = sin(kappa * xt(1));
    vecvalue(1) = sin(kappa * xt(2));
    vecvalue(2) = sin(kappa * xt(3));
    vecvalue(3) = sin(kappa * xt(0));

    return;
}

void DivmatDivmatFun4D_ex(const Vector& xt, Vector& vecvalue)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    vecvalue.SetSize(xt.Size());

    //vecvalue(0) = 0.0;
    //vecvalue(1) = 0.0;
    //vecvalue(2) = -2.0 * (1 - t);

    // Divmat of the 4D counterpart of the Martin's 3D function
    std::cout << "Error: DivmatDivmatFun4D_ex is incorrect \n";
    vecvalue(0) = - kappa * cos(kappa * xt(2));
    vecvalue(1) = - kappa * cos(kappa * xt(0));
    vecvalue(2) = - kappa * cos(kappa * xt(1));

    return;
}

////////////////
void hcurlFun3D_2_ex(const Vector& xt, Vector& vecvalue)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    vecvalue.SetSize(xt.Size());

    //
    vecvalue(0) = 100.0 * x * x * (1-x) * (1-x) * y * y * (1-y) * (1-y) * t * t * (1-t) * (1-t);
    vecvalue(1) = 0.0;
    vecvalue(2) = 0.0;

    return;
}

void curlhcurlFun3D_2_ex(const Vector& xt, Vector& vecvalue)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    vecvalue.SetSize(xt.Size());

    //
    vecvalue(0) = 0.0;
    vecvalue(1) = 100.0 * ( 2.0) * t * (1-t) * (1.-2.*t) * x * x * (1-x) * (1-x) * y * y * (1-y) * (1-y);
    vecvalue(2) = 100.0 * (-2.0) * y * (1-y) * (1.-2.*y) * x * x * (1-x) * (1-x) * t * t * (1-t) * (1-t);

    return;
}

template <double (*S)(const Vector & xt), void (*bvecfunc)(const Vector&, Vector& ),
          void (*opdivfreevec)(const Vector&, Vector& )> \
void minKsigmahatTemplate(const Vector& xt, Vector& minKsigmahatv)
{
    minKsigmahatv.SetSize(xt.Size());

    Vector b;
    bvecfunc(xt, b);

    Vector sigmahatv;
    sigmahatTemplate<S, bvecfunc, opdivfreevec>(xt, sigmahatv);

    DenseMatrix Ktilda;
    KtildaTemplate<bvecfunc>(xt, Ktilda);

    Ktilda.Mult(sigmahatv, minKsigmahatv);

    minKsigmahatv *= -1.0;
    return;
}

template <double (*S)(const Vector & xt), void (*bvecfunc)(const Vector&, Vector& ),
          void (*opdivfreevec)(const Vector&, Vector& )> \
double bsigmahatTemplate(const Vector& xt)
{
    Vector b;
    bvecfunc(xt, b);

    Vector sigmahatv;
    sigmahatTemplate<S, bvecfunc, opdivfreevec>(xt, sigmahatv);

    return b * sigmahatv;
}

template <double (*S)(const Vector & xt), void (*bvecfunc)(const Vector&, Vector& ),
          void (*opdivfreevec)(const Vector&, Vector& )> \
void sigmahatTemplate(const Vector& xt, Vector& sigmahatv)
{
    sigmahatv.SetSize(xt.Size());

    Vector b;
    bvecfunc(xt, b);

    Vector sigma(xt.Size());
    sigma(xt.Size()-1) = S(xt);
    for (int i = 0; i < xt.Size()-1; i++)
        sigma(i) = b(i) * sigma(xt.Size()-1);

    Vector opdivfree;
    opdivfreevec(xt, opdivfree);

    sigmahatv = 0.0;
    sigmahatv -= opdivfree;
#ifndef ONLY_DIVFREEPART
    sigmahatv += sigma;
#endif
    return;
}

template <double (*S)(const Vector & xt), void (*bvecfunc)(const Vector&, Vector& ),
          void (*opdivfreevec)(const Vector&, Vector& )> \
void minsigmahatTemplate(const Vector& xt, Vector& minsigmahatv)
{
    minsigmahatv.SetSize(xt.Size());
    sigmahatTemplate<S, bvecfunc, opdivfreevec>(xt, minsigmahatv);
    minsigmahatv *= -1;

    return;
}

void E_exact(const Vector &xt, Vector &E)
{
   if (xt.Size() == 3)
   {

       E(0) = sin(kappa * xt(1));
       E(1) = sin(kappa * xt(2));
       E(2) = sin(kappa * xt(0));
#ifdef BAD_TEST
       double x = xt(0);
       double y = xt(1);
       double t = xt(2);

       E(0) = x * x * (1-x) * (1-x) * y * y * (1-y) * (1-y) * t * t * (1-t) * (1-t);
       E(1) = 0.0;
       E(2) = 0.0;
#endif
   }
}


void curlE_exact(const Vector &xt, Vector &curlE)
{
   if (xt.Size() == 3)
   {
       curlE(0) = - kappa * cos(kappa * xt(2));
       curlE(1) = - kappa * cos(kappa * xt(0));
       curlE(2) = - kappa * cos(kappa * xt(1));
#ifdef BAD_TEST
       double x = xt(0);
       double y = xt(1);
       double t = xt(2);

       curlE(0) = 0.0;
       curlE(1) =  2.0 * t * (1-t) * (1.-2.*t) * x * x * (1-x) * (1-x) * y * y * (1-y) * (1-y);
       curlE(2) = -2.0 * y * (1-y) * (1.-2.*y) * x * x * (1-x) * (1-x) * t * t * (1-t) * (1-t);
#endif
   }
}


void vminusone_exact(const Vector &x, Vector &vminusone)
{
   vminusone.SetSize(x.Size());
   vminusone = -1.0;
}

void vone_exact(const Vector &x, Vector &vone)
{
   vone.SetSize(x.Size());
   vone = 1.0;
}


void f_exact(const Vector &xt, Vector &f)
{
   if (xt.Size() == 3)
   {


       //f(0) = sin(kappa * x(1));
       //f(1) = sin(kappa * x(2));
       //f(2) = sin(kappa * x(0));
       //f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
       //f(1) = (1. + kappa * kappa) * sin(kappa * x(2));
       //f(2) = (1. + kappa * kappa) * sin(kappa * x(0));

       f(0) = kappa * kappa * sin(kappa * xt(1));
       f(1) = kappa * kappa * sin(kappa * xt(2));
       f(2) = kappa * kappa * sin(kappa * xt(0));

       /*

       double x = xt(0);
       double y = xt(1);
       double t = xt(2);

       f(0) =  -1.0 * (2 * (1-y)*(1-y) + 2*y*y - 2.0 * 2 * y * 2 * (1-y)) * x * x * (1-x) * (1-x) * t * t * (1-t) * (1-t);
       f(0) += -1.0 * (2 * (1-t)*(1-t) + 2*t*t - 2.0 * 2 * t * 2 * (1-t)) * x * x * (1-x) * (1-x) * y * y * (1-y) * (1-y);
       f(1) = 2.0 * y * (1-y) * (1-2*y) * 2.0 * x * (1-x) * (1-2*x) * t * t * (1-t) * (1-t);
       f(2) = 2.0 * t * (1-t) * (1-2*t) * 2.0 * x * (1-x) * (1-2*x) * y * y * (1-y) * (1-y);
       */


   }
}
