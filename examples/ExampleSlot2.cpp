/*
 * A place holder for any temporary tests in parelag setup
 */

//                                MFEM(with 4D elements) FOSLS (no constraint!) for 3D/4D hyperbolic equation
//                                  with mesh generator and visualization
//
// Compile with: make
//
// Sample runs:  ./HybridHdivL2 -dim 3 or ./HybridHdivL2 -dim 4
//
// Description:  This example code solves a simple 3D/4D hyperbolic problem over [0,1]^3(4)
//               corresponding to the saddle point system
//                                  sigma_1 = u * b
//							 		sigma_2 - u        = 0
//                                  div_(x,t) sigma    = f
//                       with b = vector function (~velocity),
//						 NO boundary conditions (which work only in case when b * n = 0 pointwise at the domain space boundary)
//						 and initial condition:
//                                  u(x,0)            = 0
//               Here, we use a given exact solution
//                                  u(xt) = uFun_ex(xt)
//               and compute the corresponding r.h.s.
//               We discretize with Raviart-Thomas finite elements (sigma), continuous H1 elements (u) and
//					  discontinuous polynomials (mu) for the lagrange multiplier.
//
//				 If you want to run your own solution, be sure to change uFun_ex, as well as fFun_ex and check
//				 that the bFun_ex satisfies the condition b * n = 0 (see above).
// Solver: MINRES preconditioned by boomerAMG or ADS

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>

#define MYZEROTOL (1.0e-13)

//#define WITH_QHULL

#ifdef WITH_QHULL
    // headers and defines for qhull
    #include "RboxPoints.h"
    #include "QhullError.h"
    #include "Qhull.h"
    #include "QhullQh.h"
    #include "QhullFacet.h"
    #include "QhullFacetList.h"
    #include "QhullLinkedList.h"
    #include "QhullVertex.h"
    #include "QhullSet.h"
    #include "QhullVertexSet.h"

    #include "libqhull_r/qhull_ra.h"

    #define qh_QHimport
#endif

#define VTKTETRAHEDRON 10
#define VTKWEDGE 13
#define VTKTRIANGLE 5
#define VTKQUADRIL 9

#ifdef WITH_QHULL
    using namespace orgQhull;
#endif

using namespace std;
using namespace mfem;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;


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

//------------------
//********* END OF NEW BilinearForm and LinearForm integrators FOR CFOSLS 4D (used only for heat equation, so can be deleted)

namespace mfem
{

/// class for function coefficient with parameters
class FunctionCoefficientExtra : public Coefficient
{
private:
    double * parameters;
    int nparams;

protected:
   double (*Function)(const Vector &, double *, const int&);

public:
   /// Define a time-independent coefficient from a C-function
   FunctionCoefficientExtra(double (*f)(const Vector &, double *, const int&), double * Parameters, int Nparams)
   {
      Function = f;
      nparams = Nparams;
      parameters = new double[nparams];
      for ( int i = 0; i < nparams; ++i)
          parameters[i] = Parameters[i];
   }

   /// Evaluate coefficient
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
};

double FunctionCoefficientExtra::Eval(ElementTransformation & T,
                                 const IntegrationPoint & ip)
{
   double x[3];
   Vector transip(x, 3);

   T.Transform(ip, transip);

   if (Function)
   {
      return ((*Function)(transip, parameters, nparams));
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

void bFun2_ex (const Vector& xt, Vector& b);
double  bFun2div_ex(const Vector& xt);

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


double cas_weight (const Vector& xt, double * params, const int &nparams);
double deletethis (const Vector& xt);

template <double (*ufunc)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )>
    void sigmaTemplate(const Vector& xt, Vector& sigma);
template <void (*bvecfunc)(const Vector&, Vector& )>
    void KtildaTemplate(const Vector& xt, DenseMatrix& Ktilda);
template <void (*bvecfunc)(const Vector&, Vector& )>
    double bTbTemplate(const Vector& xt);
template <double (*ufunc)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )>
    void sigmaNonHomoTemplate(const Vector& xt, Vector& sigma);

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
    double rhsideTemplate(const Vector& xt);

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt), \
         double (*casweight)(const Vector & xt, double * params, const int& nparams)> \
    double rhsideWeightedTemplate(const Vector& xt, double * params, const int &nparams);


class Transport_test
    {
    protected:
        int dim;
        int numsol;
        double epsilon;

    public:
        FunctionCoefficient * scalaru;
        FunctionCoefficient * scalarf;
        FunctionCoefficient * bTb;
        VectorFunctionCoefficient * sigma;
        VectorFunctionCoefficient * conv;
        MatrixFunctionCoefficient * Ktilda;
        VectorFunctionCoefficient * sigma_nonhomo; // to incorporate inhomogeneous boundary conditions, stores (conv *S0, S0) with S(t=0) = S0
        FunctionCoefficientExtra  * casuality_weight; // weight for adding casuality in the LS functional for the conservation law
        FunctionCoefficientExtra  * weightedscalarf;
    public:
        Transport_test (int Dim, int NumSol, double Epsilon);

        int GetDim() {return dim;}
        int GetNumSol() {return numsol;}
        double GetEpsilon() {return epsilon;}
        void SetDim(int Dim) { dim = Dim;}
        void SetNumSol(int NumSol) { numsol = NumSol;}
        void SetEpsilon(double Epsilon) { epsilon = Epsilon;}
        bool CheckTestConfig();

        ~Transport_test () {}
    private:
        template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
                 void(*bvec)(const Vector & x, Vector & vec), double (*divbvec)(const Vector & xt), \
                 double (*casweight)(const Vector & xt, double * params, const int& nparams)> \
        void SetTestCoeffs ( );

        void SetScalarFun( double (*S)(const Vector & xt))
        { scalaru = new FunctionCoefficient(S);}

        template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
                 void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
        void SetScalarfFun()
        { scalarf = new FunctionCoefficient(rhsideTemplate<S, dSdt, Sgradxvec, bvec, divbfunc>);}

        template< void(*f2)(const Vector & x, Vector & vec)>  \
        void SetScalarBtB()
        {
            bTb = new FunctionCoefficient(bTbTemplate<f2>);
        }

        template<double (*f1)(const Vector & xt), void(*f2)(const Vector & x, Vector & vec)> \
        void SetHdivVec()
        {
            sigma = new VectorFunctionCoefficient(dim, sigmaTemplate<f1,f2>);
        }

        void SetConvVec( void(*f)(const Vector & x, Vector & vec))
        { conv = new VectorFunctionCoefficient(dim, f);}

        template< void(*f2)(const Vector & x, Vector & vec)>  \
        void SetKtildaMat()
        {
            Ktilda = new MatrixFunctionCoefficient(dim, KtildaTemplate<f2>);
        }

        template<double (*f1)(const Vector & xt), void(*f2)(const Vector & x, Vector & vec)> \
        void SetInitCondVec()
        {
            sigma_nonhomo = new VectorFunctionCoefficient(dim, sigmaNonHomoTemplate<f1,f2>);
        }

        void SetCasWeight( double (*casweight)(const Vector & xt, double * params, const int& nparams), double * Params, int Nparams )
        {
            casuality_weight = new FunctionCoefficientExtra(casweight, Params, Nparams);
        }

        template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
                 void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt),  \
                 double (*casweight)(const Vector & xt, double * params, const int& nparams)> \
        void SetWeightedScalarfFun( double * Params, int Nparams )
        { weightedscalarf = new FunctionCoefficientExtra(rhsideWeightedTemplate<S, dSdt, Sgradxvec, bvec, divbfunc, casweight>, Params, Nparams);}

    };

    template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx),  \
             void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt), \
             double (*casweight)(const Vector & xt, double * params, const int& nparams)> \
    void Transport_test::SetTestCoeffs ()
    {
        SetScalarFun(S);
        SetScalarfFun<S, dSdt, Sgradxvec, bvec, divbfunc>();
        SetConvVec(bvec);
        SetHdivVec<S,bvec>();
        SetKtildaMat<bvec>();
        SetScalarBtB<bvec>();
        SetInitCondVec<S,bvec>();
        SetCasWeight(casweight, &epsilon, 1);
        SetWeightedScalarfFun<S, dSdt, Sgradxvec, bvec, divbfunc, casweight>(&epsilon, 1);
        return;
    }


    bool Transport_test::CheckTestConfig()
    {
        if (dim == 4 || dim == 3)
        {
            if (numsol == 0)
                return true;
            if ( numsol == 1 && dim == 3 )
                return true;
            if ( numsol == 2 && dim == 4 )
                return true;
            if ( numsol == 3 && dim == 3 )
                return true;
            if ( numsol == 33 && dim == 4 )
                return true;
            if ( numsol == 4 && dim == 3 )
                return true;
            if ( numsol == 44 && dim == 3 )
                return true;
            if ( numsol == 100 && dim == 3 )
                return true;
            if ( numsol == 200 && dim == 3 )
                return true;
            if ( numsol == 5 && dim == 3 )
                return true;
            if ( numsol == 55 && dim == 4 )
                return true;
            if ( numsol == 444 && dim == 4 )
                return true;
            if ( numsol == 1000 && dim == 3 )
                return true;
            return false;
        }
        else
            return false;

    }

    Transport_test::Transport_test (int Dim, int NumSol, double Epsilon)
    {
        dim = Dim;
        numsol = NumSol;
        epsilon = Epsilon;

        if ( CheckTestConfig() == false )
            std::cout << "Inconsistent dim and numsol" << std::endl << std::flush;
        else
        {
            if (numsol == 0)
            {
                //std::cout << "The domain is rectangular or cubic, velocity does not"
                             //" satisfy divergence condition" << std::endl << std::flush;
                SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_gradx, &bFun_ex, &bFundiv_ex, &cas_weight>();
                //SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_gradx, &bFunCube3D_ex, &bFunCube3Ddiv_ex, &cas_weight>();
            }
            if (numsol == 1)
            {
                //std::cout << "The domain must be a cylinder over a circle" << std::endl << std::flush;
                SetTestCoeffs<&uFun2_ex, &uFun2_ex_dt, &uFun2_ex_gradx, &bFun2_ex, &bFun2div_ex, &cas_weight>();
            }
            if (numsol == 100)
            {
                //std::cout << "The domain must be a cylinder over a unit square" << std::endl << std::flush;
                SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &cas_weight>();
            }
            if (numsol == 200)
            {
                //std::cout << "The domain must be a cylinder over a unit circle" << std::endl << std::flush;
                SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_gradx, &bFun2_ex, &bFun2div_ex, &cas_weight>();
            }
            if (numsol == 2)
            {
                //std::cout << "The domain must be a cylinder over a 3D cube, velocity does not"
                             //" satisfy divergence condition" << std::endl << std::flush;
                SetTestCoeffs<&uFun3_ex, &uFun3_ex_dt, &uFun3_ex_gradx, &bFun_ex, &bFundiv_ex, &cas_weight>();
            }
            if (numsol == 3)
            {
                //std::cout << "The domain must be a cylinder over a circle" << std::endl << std::flush;
                SetTestCoeffs<&uFun4_ex, &uFun4_ex_dt, &uFun4_ex_gradx, &bFun2_ex, &bFun2div_ex, &cas_weight>();
            }
            if (numsol == 4) // no exact solution in fact, ~ unsuccessfully trying to get a picture from the report
            {
                //std::cout << "The domain must be a cylinder over a circle" << std::endl << std::flush;
                //std::cout << "Using new interface \n";
                SetTestCoeffs<&uFun5_ex, &uFun5_ex_dt, &uFun5_ex_gradx, &bFun2_ex, &bFun2div_ex, &cas_weight>();
            }
            if (numsol == 44) // no exact solution in fact, ~ unsuccessfully trying to get a picture from the report
            {
                //std::cout << "The domain must be a cylinder over a circle" << std::endl << std::flush;
                //std::cout << "Using new interface \n";
                SetTestCoeffs<&uFun6_ex, &uFun6_ex_dt, &uFun6_ex_gradx, &bFun2_ex, &bFun2div_ex, &cas_weight>();
            }
            if (numsol == 5)
            {
                //std::cout << "The domain must be a cylinder over a square" << std::endl << std::flush;
                SetTestCoeffs<&uFun4_ex, &uFun4_ex_dt, &uFun4_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &cas_weight>();
            }
            if (numsol == 1000)
            {
                //std::cout << "The domain must be a cylinder over a square" << std::endl << std::flush;
                SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &cas_weight>();
            }
            if (numsol == 33)
            {
                //std::cout << "The domain must be a cylinder over a sphere" << std::endl << std::flush;
                SetTestCoeffs<&uFun33_ex, &uFun33_ex_dt, &uFun33_ex_gradx, &bFunSphere3D_ex, &bFunSphere3Ddiv_ex, &cas_weight>();
            }
            if (numsol == 444) // no exact solution in fact, ~ unsuccessfully trying to get something beauitiful
            {
                //std::cout << "The domain must be a cylinder over a sphere" << std::endl << std::flush;
                SetTestCoeffs<&uFun66_ex, &uFun66_ex_dt, &uFun66_ex_gradx, &bFunSphere3D_ex, &bFunSphere3Ddiv_ex, &cas_weight>();
            }
            if (numsol == 55)
            {
                //std::cout << "The domain must be a cylinder over a cube" << std::endl << std::flush;
                SetTestCoeffs<&uFun33_ex, &uFun33_ex_dt, &uFun33_ex_gradx, &bFunCube3D_ex, &bFunCube3Ddiv_ex, &cas_weight>();
            }
        } // end of setting test coefficients in correct case
    }


int printArr2DInt (Array2D<int> *arrayint);
int setzero(Array2D<int>* arrayint);
int printDouble2D( double * arr, int dim1, int dim2);
int printInt2D( int * arr, int dim1, int dim2);

#ifdef WITH_QHULL
    int qhull_wrapper(int * simplices, qhT * qh, double * points, int dim, double volumetol, char *flags);
    void print_summary(qhT *qh);
    void qh_fprintf(qhT *qh, FILE *fp, int msgcode, const char *fmt, ... );
    void makePrism(qhT *qh, coordT *points, int numpoints, int dim, int seed);
#endif
__inline__ double dist( double * M, double * N , int d);
int factorial(int n);
int permutation_sign( int * permutation, int size);
double determinant4x4(DenseMatrix Mat);

__inline__ void zero_intinit ( int * arr, int size);

int main(int argc, char *argv[])
{
    int num_procs, myid;
    bool visualization = 1;

    // 1. Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    bool verbose = (myid == 0);

    int nDimensions     = 3;
    int numsol          = 1000;
    double epsilon      = -0.01;

    int ser_ref_levels  = 0;
    int par_ref_levels  = 2;
    int Nsteps          = 4;
    double tau          = 0.25;


    int generate_frombase   = 1;
    int generate_parallel   = generate_frombase * 1;
    int whichparallel       = generate_parallel * 2;
    int bnd_method          = 1;
    int local_method        = 2;

    //const char *mesh_file = "../build3/meshes/cube_3d_moderate.mesh";
    //const char *mesh_file = "../build3/meshes/square_2d_moderate.mesh";

    //const char *mesh_file = "../build3/meshes/cube4d_low.MFEM";
    //const char *mesh_file = "../build3/meshes/cube4d.MFEM";
    const char *mesh_file = "dsadsad";
    //const char *mesh_file = "../build3/pmesh_2_mwe_0.mesh";
    //const char *mesh_file = "../build3/mesh_par1_id0_np_1.mesh";
    //const char *mesh_file = "../build3/mesh_par1_id0_np_2.mesh";
    //const char *mesh_file = "../build3/meshes/tempmesh_frompmesh.mesh";
    //const char *mesh_file = "../build3/meshes/orthotope3D_moderate.mesh";
    //const char *mesh_file = "../build3/meshes/sphere3D_0.1to0.2.mesh";
    //const char * mesh_file = "../build3/meshes/orthotope3D_fine.mesh";

    //const char * meshbase_file = "../build3/meshes/sphere3D_0.1to0.2.mesh";
    //const char * meshbase_file = "../build3/meshes/sphere3D_0.05to0.1.mesh";
    //const char * meshbase_file = "../build3/meshes/sphere3D_veryfine.mesh";
    //const char * meshbase_file = "../build3/meshes/beam-tet.mesh";
    //const char * meshbase_file = "../build3/meshes/escher-p3.mesh";
    //const char * meshbase_file = "../build3/meshes/orthotope3D_moderate.mesh";
    //const char * meshbase_file = "../build3/meshes/orthotope3D_fine.mesh";
    const char * meshbase_file = "../build3/meshes/square_2d_moderate.mesh";
    //const char * meshbase_file = "../build3/meshes/square_2d_fine.mesh";
    //const char * meshbase_file = "../build3/meshes/square-disc.mesh";
    //const char *meshbase_file = "dsadsad";
    //const char * meshbase_file = "../build3/meshes/circle_fine_0.1.mfem";
    //const char * meshbase_file = "../build3/meshes/circle_moderate_0.2.mfem";

    int feorder         = 0;

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
    args.AddOption(&whichparallel, "-pv", "--parver",
                   "Version of parallel algorithm.");
    args.AddOption(&bnd_method, "-bnd", "--bndmeth",
                   "Method for generating boundary elements.");
    args.AddOption(&local_method, "-loc", "--locmeth",
                   "Method for local mesh procedure.");
    args.AddOption(&numsol, "-nsol", "--numsol",
                   "Solution number.");
    args.AddOption(&epsilon, "-timewei", "--timeweight",
                   "Casuality weight coefficient. Unit weight if negative.");
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

    StopWatch chrono;

    //DEFAULTED LINEAR SOLVER OPTIONS
    int max_num_iter = 50000;
    double rtol = 1e-9;//1e-7;//1e-9;
    double atol = 1e-12;//1e-9;//1e-12;

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
    else //if nDimensions is no 3 or 4
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

    /*
    int partitioning[50000];
    Mesh * tempmesh = new Mesh( *pmesh.get(), &partitioning);

    if (verbose)
    {
        std::stringstream fname;
        fname << "tempmesh_"<< nDimensions - 1 << ".mesh";
        std::ofstream ofid(fname.str().c_str());
        ofid.precision(8);
        tempmesh->Print(ofid);
    }
    */

    /*
    std::stringstream fname;
    fname << "pmesh_"<< nDimensions - 1 << "_mwe_" << myid << ".mesh";
    std::ofstream ofid(fname.str().c_str());
    ofid.precision(8);
    pmesh->Print(ofid);
    */


    // 6. Define a parallel finite element space on the parallel mesh. Here we
    //    use the Raviart-Thomas finite elements of the specified order.

    int dim = nDimensions;

    FiniteElementCollection *hdiv_coll;
    if ( dim == 4 )
    {
       hdiv_coll = new RT0_4DFECollection;
       if (verbose)
           cout << "RT: order 0 for 4D" << endl;
    }
    else
    {
        hdiv_coll = new RT_FECollection(feorder, dim);
        if (verbose)
            cout << "RT: order " << feorder << " for 3D" << endl;
    }

    FiniteElementCollection *l2_coll;
    if ( dim == 4 )
    {
        l2_coll = new L2_FECollection(0, dim);
        if (verbose)
            cout << "L2: order 0 for 4D" << endl;
    }
    else
    {
        l2_coll = new L2_FECollection(feorder, dim);
        if (verbose)
            cout << "L2: order " << feorder << " for 3D" << endl;
    }

    ParFiniteElementSpace *R_space = new ParFiniteElementSpace(pmesh.get(), hdiv_coll);
    ParFiniteElementSpace *W_space = new ParFiniteElementSpace(pmesh.get(), l2_coll);

   /*
   ParGridFunction *pgridfuntest = new ParGridFunction(R_space);
   VectorFunctionCoefficient Hdivtest_fun_coeff(nDimensions, Hdivtest_fun);
   pgridfuntest->ProjectCoefficient(Hdivtest_fun_coeff);
   //cout << "pgridfuntest" << endl;
   //pgridfuntest->Print();

   pgridfuntest->ComputeSlices ( 0.1, 2, 0.3, myid);
   MPI_Finalize();
   return 0;
   */



   HYPRE_Int dimR = R_space->GlobalTrueVSize();
   //HYPRE_Int dimW = W_space->GlobalTrueVSize();


   if (verbose)
   {
      std::cout << "***********************************************************\n";
      std::cout << "dim(R) = " << dimR << "\n";
      //std::cout << "dim(W) = " << dimW << "\n";
      //std::cout << "dim(R+W) = " << dimR + dimW << "\n";
      std::cout << "***********************************************************\n";
   }

   // 7. Define the two BlockStructure of the problem.  block_offsets is used
   //    for Vector based on dof (like ParGridFunction or ParLinearForm),
   //    block_trueOffstes is used for Vector based on trueDof (HypreParVector
   //    for the rhs and solution of the linear system).  The offsets computed
   //    here are local to the processor.

    ParGridFunction * x(new ParGridFunction(R_space));
    *x = 0.0;
    HypreParVector *X = x->ParallelAverage();


    // 8. Define the coefficients, analytical solution, and rhs of the PDE.
    ConstantCoefficient k(1.0);
    ConstantCoefficient zero(.0);
    ConstantCoefficient one(1.0);
    ConstantCoefficient btbcoeff(1.0);

    //MatrixFunctionCoefficient Ktilda( dim, Ktilda_ex );
    //FunctionCoefficient fcoeff(fFun);
    //FunctionCoefficient ucoeff(uFun_ex);
    //VectorFunctionCoefficient sigmacoeff(dim, sigmaFun_ex);

    Transport_test Mytest(nDimensions,numsol, epsilon);
    //FunctionCoefficient fnatcoeff(f_natural);
    FunctionCoefficient deletethiscoeff(deletethis);


    //----------------------------------------------------------
    // Setting boundary conditions.
    //----------------------------------------------------------

    Array<int> ess_tdof_list, ess_bdr(pmesh->bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[0] = 1; // t = 0
    ess_bdr[1] = 1; // lateral boundary
    R_space->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    //-----------------------


   // 9. Define the parallel grid function and parallel linear forms, solution
   //    vector and rhs.

   bool with_divdiv = true;

   if (with_divdiv == true && verbose)
       cout << "Bilinear form with div-div term" << endl;

   if (epsilon > 0 && verbose)
       cout << "Using casuality weight with exp(-t/epsilon)" << endl;


   ParLinearForm *fform(new ParLinearForm(R_space));
   if (with_divdiv == true)
       fform->AddDomainIntegrator(new VectordivDomainLFIntegrator(*(Mytest.weightedscalarf)));
       //fform->AddDomainIntegrator(new VectordivDomainLFIntegrator(*(Mytest.scalarf)));
   fform->Assemble();

   // 10. Assemble the finite element matrices for the CFOSLS operator  A
   //     where:
   //
   //     A = ( sigma, tau)_{L2} + (possible) (casuality_weight * div sigma, div tau)_{L2}

    ParBilinearForm *Ablock(new ParBilinearForm(R_space));

    if (with_divdiv == true)
        Ablock->AddDomainIntegrator(new DivDivIntegrator(*(Mytest.casuality_weight)));
        //Ablock->AddDomainIntegrator(new DivDivIntegrator(one));
    Ablock->AddDomainIntegrator(new VectorFEMassIntegrator(*(Mytest.Ktilda)));
    //Ablock->AddDomainIntegrator(new VectorFEMassIntegrator());

    bool hybridization = false;
    FiniteElementCollection *hfec = NULL;
    ParFiniteElementSpace *hfes = NULL;
    if (hybridization)
    {
       hfec = new DG_Interface_FECollection(feorder, dim);
       hfes = new ParFiniteElementSpace(pmesh.get(), hfec);
       Ablock->EnableHybridization(hfes, new NormalTraceJumpIntegrator(),
                              ess_tdof_list);
    }
    Ablock->Assemble();

    //x->ProjectCoefficient(*(Mytest.sigma));
    Ablock->EliminateEssentialBC(ess_bdr,*x,*fform);
    //Ablock->EliminateEssentialBC(ess_bdr,trueX.GetBlock(0),*fform);
    //Ablock->EliminateEssentialBC(ess_bdr,trueX.GetBlock(0),trueRhs.GetBlock(0));
    Ablock->Finalize();

    HypreParMatrix *A;
    A = Ablock->ParallelAssemble();

    HypreParVector *B = fform->ParallelAssemble();
    *X = 0.0;

   //=======================================================
   // Assembling the final Block Matrix
   //-------------------------------------------------------

   if (verbose)
       cout<< "Final saddle point matrix assembled"<<endl << flush;
   MPI_Barrier(MPI_COMM_WORLD);

   // 12. Solve the linear system with MINRES.
   //     Check the norm of the unpreconditioned residual.


   Solver * prec;

   bool with_prec = true;
   bool prec_is_ADS = true;
   bool ADS_is_for_I = true;

   ParBilinearForm *AIblock(new ParBilinearForm(R_space));
   HypreParMatrix *AI;

   if (with_prec)
   {
       if (verbose)
            cout << "Using a preconditioner:" << endl;
       if (prec_is_ADS == true)
       {
           if (verbose)
                cout << "Using ADS as a preconditioner" << endl;

           if (nDimensions == 3)
           {
               if (ADS_is_for_I == true)
               {
                   if (verbose)
                       cout << "Creating ADS for the Identity (not Ktilda!)" << endl;
                   if (with_divdiv == true)
                       AIblock->AddDomainIntegrator(new DivDivIntegrator(*(Mytest.casuality_weight)));
                       //Ablock->AddDomainIntegrator(new DivDivIntegrator(one));
                   AIblock->AddDomainIntegrator(new VectorFEMassIntegrator());

                   AIblock->Assemble();
                   AIblock->EliminateEssentialBC(ess_bdr);
                   AIblock->Finalize();

                   AI = AIblock->ParallelAssemble();

                   prec = new HypreADS (*AI, R_space);
               }
               else
                    prec = new HypreADS (*A, R_space);

           }
           else
           {
               if (verbose)
                   cout << "ADS is not working in case dim = " << nDimensions << endl;
               MPI_Finalize();
               return 0;
           }
       }
       else
       {
           if (verbose)
               cout << "Using mfem hybridization combined with boomerAMG as a preconditioner" << endl;
           prec = new HypreBoomerAMG(*A);
       }

   }

   chrono.Clear();
   chrono.Start();
   MINRESSolver solver(MPI_COMM_WORLD);
   solver.SetAbsTol(atol);
   solver.SetRelTol(rtol);
   solver.SetMaxIter(max_num_iter);
   solver.SetOperator(*A);
   if (with_prec)
        solver.SetPreconditioner(*prec);
   solver.SetPrintLevel(0);
   solver.Mult(*B, *X);
   chrono.Stop();

   if (verbose)
   {
      if (solver.GetConverged())
         std::cout << "MINRES converged in " << solver.GetNumIterations()
                   << " iterations with a residual norm of " << solver.GetFinalNorm() << ".\n";
      else
         std::cout << "MINRES did not converge in " << solver.GetNumIterations()
                   << " iterations. Residual norm is " << solver.GetFinalNorm() << ".\n";
      std::cout << "MINRES solver took " << chrono.RealTime() << "s. \n";
   }

   // 13. Extract the parallel grid function corresponding to the finite element
   //     approximation X. This is the local solution on each processor. Compute
   //     L2 error norms.

   int order_quad = max(2, 2*feorder+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   /*
   ParGridFunction *sigma(new ParGridFunction);
   sigma->MakeRef(R_space, x.GetBlock(0), 0);
   sigma->Distribute(&(trueX.GetBlock(0)));
   */

   // adding back the term from nonhomogeneous initial condition
   ParGridFunction *sigma_nonhomo = new ParGridFunction(R_space);
   sigma_nonhomo->ProjectCoefficient(*(Mytest.sigma_nonhomo));
   //HypreParVector *sigma_nonhomovec = sigma_nonhomo->ParallelAverage();

   //*sigma += *sigma_nonhomo;

   //*X += *sigma_nonhomovec;
   *x = *X;
   *x += *sigma_nonhomo;

   double err_sigma = x->ComputeL2Error(*(Mytest.sigma), irs);
   double norm_sigma = ComputeGlobalLpNorm(2, *(Mytest.sigma), *pmesh, irs);
   if (verbose)
       cout << "local: err_sigma / norm_sigma = " << err_sigma / norm_sigma << endl;

   /*
   err_sigma *= err_sigma;
   double err_sigma_global;
   MPI_Reduce(&err_sigma, &err_sigma_global, 1, MPI_DOUBLE,
              MPI_SUM, 0, comm);
   err_sigma_global = std::sqrt(err_sigma_global);

   norm_sigma *= norm_sigma;
   double norm_sigma_global;
   MPI_Reduce(&norm_sigma, &norm_sigma_global, 1, MPI_DOUBLE,
              MPI_SUM, 0, comm);
   norm_sigma_global = std::sqrt(norm_sigma_global);

   if (verbose)
       cout << "global: err_sigma / norm_sigma = " << err_sigma_global / norm_sigma_global << endl;
   */

   /*
   ParBilinearForm *DivDivblock(new ParBilinearForm(R_space));
   HypreParMatrix *DivDiv;

   DivDivblock->AddDomainIntegrator(new DivDivIntegrator());
   DivDivblock->Assemble();
   DivDivblock->EliminateEssentialBC(ess_bdr,trueX.GetBlock(0),*fform);
   DivDivblock->Finalize();
   DivDiv = DivDivblock->ParallelAssemble();
   */




   ParGridFunction *sigma_exact = new ParGridFunction(R_space);
   sigma_exact->ProjectCoefficient(*(Mytest.sigma));
   //HypreParVector *sigma_exactvec = sigma_exact->ParallelAverage();

   if (visualization && nDimensions < 4)
   //if (true)
   {
      //cout << "visualization may not work for 4D element code and not present in mfem_4d version" << endl;
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream u_sock(vishost, visport);
      u_sock << "parallel " << num_procs << " " << myid << "\n";
      u_sock.precision(8);
      u_sock << "solution\n" << *pmesh << *sigma_exact << "window_title 'sigma_exact'"
             << endl;
      // Make sure all ranks have sent their 'u' solution before initiating
      // another set of GLVis connections (one from each rank):


      socketstream uu_sock(vishost, visport);
      uu_sock << "parallel " << num_procs << " " << myid << "\n";
      uu_sock.precision(8);
      uu_sock << "solution\n" << *pmesh << *x << "window_title 'sigma'"
             << endl;

      *sigma_exact -= * sigma_nonhomo;
      socketstream uuuuu_sock(vishost, visport);
      uuuuu_sock << "parallel " << num_procs << " " << myid << "\n";
      uuuuu_sock.precision(8);
      uuuuu_sock << "solution\n" << *pmesh << *sigma_exact << "window_title 'sigma - sigmanonhomo'"
             << endl;

      *sigma_exact += *sigma_nonhomo;

      socketstream uuuu_sock(vishost, visport);
      uuuu_sock << "parallel " << num_procs << " " << myid << "\n";
      uuuu_sock.precision(8);
      uuuu_sock << "solution\n" << *pmesh << *sigma_nonhomo << "window_title 'sigma_nonhomo'"
             << endl;

      *sigma_exact -= *x;

      socketstream uuu_sock(vishost, visport);
      uuu_sock << "parallel " << num_procs << " " << myid << "\n";
      uuu_sock.precision(8);
      uuu_sock << "solution\n" << *pmesh << *sigma_exact << "window_title 'difference'"
             << endl;

      MPI_Barrier(pmesh->GetComm());
   }

   // 17. Free the used memory.
   delete fform;
   //delete CFOSLSop;
   delete A;

   delete Ablock;
   delete W_space;
   delete R_space;
   delete l2_coll;
   delete hdiv_coll;

   //delete pmesh;

   MPI_Finalize();

   return 0;
}

// nparams can be either 0, then unit wieght is used, or > 0
// if nparams > 0, then params[0] should be epsilon for the weight
// if epsilon < 0, then unit weight is used
// else exp(-t/eps) is used, where t is the last component of xt vector
double cas_weight (const Vector& xt, double * params, const int &nparams)
{
    //return 1.0;
    if (nparams < 0)
    {
        cout << "Error: nparams should be nonnegative" << endl;
        return 1.0;
    }
    if (nparams == 0)
        return 1.0;
    if (nparams > 0)
    {
        if (params[0] < 0)
            return 1.0;
        else
        {
            double t = xt[xt.Size()-1];
            return exp (-t / params[0]);
        }
    }
}

double deletethis (const Vector& xt)
{
    double t = xt[xt.Size()-1];
    return exp (-t / 0.1);
}


template <void (*bvecfunc)(const Vector&, Vector& )> \
void KtildaTemplate(const Vector& xt, DenseMatrix& Ktilda)
{
    int nDimensions = xt.Size();
    Vector b;
    bvecfunc(xt,b);
    double bTbInv = (-1./(b*b));
    Ktilda.Diag(1.0,nDimensions);
    AddMult_a_VVt(bTbInv,b,Ktilda);
}

template <double (*ufunc)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )> \
void sigmaTemplate(const Vector& xt, Vector& sigma)
{
    Vector b;
    bvecfunc(xt, b);
    sigma.SetSize(xt.Size());
    sigma(xt.Size()-1) = ufunc(xt);
    for (int i = 0; i < xt.Size()-1; i++)
        sigma(i) = b(i) * sigma(xt.Size()-1);

    return;
}

template <double (*ufunc)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )> \
void sigmaNonHomoTemplate(const Vector& xt, Vector& sigma) // sigmaNonHomo = (b u, u) for u = S(t=0)
{
    Vector xteq0(xt.Size()); // xt with t = 0
    xteq0 = xt;
    xteq0(xteq0.Size()-1) = 0.0;

    Vector b;
    bvecfunc(xt, b);
    sigma.SetSize(xt.Size());
    sigma(xt.Size()-1) = ufunc(xteq0);
    for (int i = 0; i < xt.Size()-1; i++)
        sigma(i) = b(i) * sigma(xt.Size()-1);

    return;
}


template <void (*bvecfunc)(const Vector&, Vector& )> \
double bTbTemplate(const Vector& xt)
{
    Vector b;
    bvecfunc(xt,b);
    return b*b;
}

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
double rhsideTemplate(const Vector& xt)
{
    Vector b;
    bvec(xt,b);

    Vector gradS;
    Sgradxvec(xt,gradS);

    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    Vector gradS0;
    Sgradxvec(xt0,gradS0);

    double res = 0.0;

    res += dSdt(xt);
    for ( int i= 0; i < xt.Size() - 1; ++i )
        res += b(i) * (gradS(i) - gradS0(i));
    res += divbfunc(xt) * (S(xt) - S(xt0));

    // only for debugging casuality weight usage
    //double t = xt[xt.Size()-1];
    //res *= exp (-t / 0.01);

    return res;

    /*
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    Vector b(3);
    bFun2_ex(xt,b);
    return 0.0 - (
           -100.0 * 2.0 * (x-0.5) * exp(-100.0 * ((x - 0.5) * (x - 0.5) + y * y)) * b(0) +
           -100.0 * 2.0 *    y    * exp(-100.0 * ((x - 0.5) * (x - 0.5) + y * y)) * b(1) );
    */
}

// with additional casuality weight
template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt), \
         double (*casweight)(const Vector & xt, double * params, const int& nparams)> \
double rhsideWeightedTemplate(const Vector& xt, double * params, const int &nparams)
{
    Vector b;
    bvec(xt,b);

    Vector gradS;
    Sgradxvec(xt,gradS);

    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    Vector gradS0;
    Sgradxvec(xt0,gradS0);

    double res = 0.0;

    res += dSdt(xt);
    for ( int i= 0; i < xt.Size() - 1; ++i )
        res += b(i) * (gradS(i) - gradS0(i));
    res += divbfunc(xt) * (S(xt) - S(xt0));

    res *= casweight(xt, params, nparams);

    return res;

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


/*

double fFun(const Vector& xt)
{
    double t = xt(xt.Size()-1);
    //double tmp = (xt.Size()==4) ? 1.0 - 2.0 * xt(2) : 0;
    double tmp = (xt.Size()==4) ? 2*M_PI * sin(2*xt(2)*M_PI) : 0;
    //double tmp = (xt.Size()==4) ? M_PI * cos(xt(2)*M_PI) : 0;
    //double tmp = (xt.Size()==4) ? M_PI * sin(xt(2)*M_PI) : 0;
    return cos(t)*exp(t)+sin(t)*exp(t)+(M_PI*cos(xt(1)*M_PI)*cos(xt(0)*M_PI)+
                   2*M_PI*cos(xt(0)*2*M_PI)*cos(xt(1)*M_PI)+tmp) *uFun_ex(xt);
    //return cos(t)*exp(t)+sin(t)*exp(t)+(1.0 - 2.0 * xt(0) + 1.0 - 2.0 * xt(1) +tmp) *uFun_ex(xt);
}
*/

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
    bFun2_ex(xt,b);
    return (t + 1) * exp(t) * sin (((x - 0.5)*(x - 0.5) + y*y)) +
             t * exp(t) * 2.0 * (x - 0.5) * cos (((x - 0.5)*(x - 0.5) + y*y)) * b(0) +
             t * exp(t) * 2.0 * y         * cos (((x - 0.5)*(x - 0.5) + y*y)) * b(1);
}
*/

void bFun2_ex(const Vector& xt, Vector& b )
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

double bFun2div_ex(const Vector& xt)
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
}

double uFun4_ex_dt(const Vector& xt)
{
    return uFun4_ex(xt);
}

void uFun4_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = exp(t) * 2.0 * (x - 0.5) * cos (((x - 0.5)*(x - 0.5) + y*y));
    gradx(1) = exp(t) * 2.0 * y         * cos (((x - 0.5)*(x - 0.5) + y*y));
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
/*
double fFun4(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    Vector b(3);
    bFun2_ex(xt,b);
    return exp(t) * sin (((x - 0.5)*(x - 0.5) + y*y)) +
             exp(t) * 2.0 * (x - 0.5) * cos (((x - 0.5)*(x - 0.5) + y*y)) * b(0) +
             exp(t) * 2.0 * y         * cos (((x - 0.5)*(x - 0.5) + y*y)) * b(1);
}


double f_natural(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    if ( t > MYZEROTOL)
        return 0.0;
    else
        return (-uFun5_ex(xt));
}
*/

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

void Hdivtest_fun(const Vector& xt, Vector& out )
{
    out.SetSize(xt.Size());

    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    out(0) = x;
    out(1) = 0.0;
    out(2) = 0.0;
    out(xt.Size()-1) = 0.;

}

double L2test_fun(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    return x;
}


void videofun(const Vector& xt, Vector& vecvalue )
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    vecvalue.SetSize(xt.Size());
    vecvalue(0) = 3 * x * ( 1 + 0.4 * sin (M_PI * (t + 1.0))) + 2.0 * (y * (y - 0.5) - z) * exp(-0.5*t) + exp(-100.0*(x*x + y * y + (z-0.5)*(z-0.5)));
    vecvalue(1) = 0.0;
    vecvalue(2) = 0.0;
    vecvalue(3) = 0.0;
    //return 3 * x * ( 1 + 0.2 * sin (M_PI * 0.5 * t/(t + 1))) + 2.0 * (y * (y - 0.5) - z) * exp(-0.5*t);
}

/*
double fFun5(const Vector& xt) // non zero because of initial condition = div (b S_0(x) ) = div b * S_0(x) + b * grad S_0
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    Vector b(3);
    bFun2_ex(xt,b);
    return 0.0 - (
           -100.0 * 2.0 * (x-0.5) * exp(-100.0 * ((x - 0.5) * (x - 0.5) + y * y)) * b(0) +
           -100.0 * 2.0 *    y    * exp(-100.0 * ((x - 0.5) * (x - 0.5) + y * y)) * b(1) );

    //return 0.0;
}
*/

/*

template <void (*bvecfunc)(const Vector&, Vector& )> \
void KtildaTemplate(const Vector& xt, DenseMatrix& Ktilda)
{
    int nDimensions = xt.Size();
    Vector b;
    bvecfunc(xt,b);
    double bTbInv = (-1./(b*b));
    Ktilda.Diag(1.0,nDimensions);
    AddMult_a_VVt(bTbInv,b,Ktilda);
}


template <double (*ufunc)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )> \
void sigmaTemplate(const Vector& xt, Vector& sigma)
{
    Vector b;
    bvecfunc(xt, b);
    sigma.SetSize(xt.Size());
    sigma(xt.Size()-1) = ufunc(xt);
    for (int i = 0; i < xt.Size()-1; i++)
        sigma(i) = b(i) * sigma(xt.Size()-1);

    return;
}

double uFun_ex(const Vector& xt)
{
    double t = xt(xt.Size()-1);
    //return t;
    ////setback
    return sin(t)*exp(t);
}

double fFun(const Vector& xt)
{
//    double tmp = 0.;
//    for (int i = 0; i < xt.Size()-1; i++)
//        tmp += xt(i);
//    return 1.+(2*M_PI*cos(xt(0)*2*M_PI)*cos(xt(1)*M_PI)+
//               M_PI*cos(xt(0)*M_PI)*cos(xt(1)*M_PI))*uFun_ex(xt);
//    return 1.;//+ (xt.Size()-1-2*tmp) * uFun_ex(xt);
    //return 1.0 + (3.0 - 2 * xt(0) - 2 * xt(1) - 2* xt(2)) * uFun_ex(xt);

    ////setback
    double t = xt(xt.Size()-1);
    double tmp = (xt.Size()==4) ? M_PI*sin(xt(2)*M_PI) : 0;
    return cos(t)*exp(t)+sin(t)*exp(t)+(M_PI*cos(xt(1)*M_PI)*cos(xt(0)*M_PI)+
                   2*M_PI*cos(xt(0)*2*M_PI)*cos(xt(1)*M_PI)+tmp) *uFun_ex(xt);


}

void bFun_ex(const Vector& xt, Vector& b )
{
    ////setback

    b.SetSize(xt.Size());
//    for (int i = 0; i < xt.Size()-1; i++)
//        b(i) = xt(i) * (1 - xt(i));

    b(0) = sin(xt(0)*2*M_PI)*cos(xt(1)*M_PI);
    b(1) = sin(xt(1)*M_PI)*cos(xt(0)*M_PI);
    b(2) = 1-cos(xt(2)*M_PI);

//    b(0) = -xt(0)*(1 - xt(0))*(1 - 2*xt(1));
//    b(1) = xt(1)*(1 - xt(1))*(1 - 2*xt(0));
//    if (xt.Size() == 4)
//    {
//        b(0) *= (2*(1 - 2*xt(2)));
//        b(1) *= (1 - 2*xt(2));
//        b(2) = xt(2)*(1 - xt(2))*(1 - 2*xt(0))*(1 - 2*xt(1));
//    }
    b(xt.Size()-1) = 1.;

}

*/

/*
void sigmaFun_ex(const Vector& xt, Vector& sigma)
{
    ////setback

    Vector b;
    bFun_ex(xt, b);
    sigma.SetSize(xt.Size());
    sigma(xt.Size()-1) = uFun_ex(xt);
    for (int i = 0; i < xt.Size()-1; i++)
        sigma(i) = b(i) * sigma(xt.Size()-1);

}

double bTb_ex(const Vector& xt)
{
    Vector b;
    bFun_ex(xt,b);
    return b*b;
}

void Ktilda_ex(const Vector& xt, DenseMatrix& Ktilda)
{
    int nDimensions = xt.Size();
    Vector b;
    bFun_ex(xt,b);
    double bTbInv = (-1./(b*b));
    Ktilda.Diag(1.0,nDimensions);
    AddMult_a_VVt(bTbInv,b,Ktilda);
}

double u0_function(const Vector &x )
{
    return uFun_ex(x);
}

double uFun_ex(const Vector& xt)
{
    double t = xt(xt.Size()-1);
    //return t;
    ////setback
    return sin(t)*exp(t);
}

double fFun(const Vector& xt)
{
//    double tmp = 0.;
//    for (int i = 0; i < xt.Size()-1; i++)
//        tmp += xt(i);
//    return 1.+(2*M_PI*cos(xt(0)*2*M_PI)*cos(xt(1)*M_PI)+
//               M_PI*cos(xt(0)*M_PI)*cos(xt(1)*M_PI))*uFun_ex(xt);
//    return 1.;//+ (xt.Size()-1-2*tmp) * uFun_ex(xt);
    //return 1.0 + (3.0 - 2 * xt(0) - 2 * xt(1) - 2* xt(2)) * uFun_ex(xt);

    ////setback
    double t = xt(xt.Size()-1);
    double tmp = (xt.Size()==4) ? M_PI*sin(xt(2)*M_PI) : 0;
    return cos(t)*exp(t)+sin(t)*exp(t)+(M_PI*cos(xt(1)*M_PI)*cos(xt(0)*M_PI)+
                   2*M_PI*cos(xt(0)*2*M_PI)*cos(xt(1)*M_PI)+tmp) *uFun_ex(xt);


}

void bFun_ex(const Vector& xt, Vector& b )
{
    ////setback

    b.SetSize(xt.Size());
//    for (int i = 0; i < xt.Size()-1; i++)
//        b(i) = xt(i) * (1 - xt(i));

    b(0) = sin(xt(0)*2*M_PI)*cos(xt(1)*M_PI);
    b(1) = sin(xt(1)*M_PI)*cos(xt(0)*M_PI);
    b(2) = 1-cos(xt(2)*M_PI);

//    b(0) = -xt(0)*(1 - xt(0))*(1 - 2*xt(1));
//    b(1) = xt(1)*(1 - xt(1))*(1 - 2*xt(0));
//    if (xt.Size() == 4)
//    {
//        b(0) *= (2*(1 - 2*xt(2)));
//        b(1) *= (1 - 2*xt(2));
//        b(2) = xt(2)*(1 - xt(2))*(1 - 2*xt(0))*(1 - 2*xt(1));
//    }
    b(xt.Size()-1) = 1.;

}

*/


namespace mfem
{

void Mesh::IntermeshInit( IntermediateMesh * intermesh, int dim, int nv, int ne, int nbdr, int with_gindices_flag)
{
    intermesh->dim = dim;
    intermesh->ne = ne;
    intermesh->nv = nv;
    intermesh->nbe = nbdr;

    intermesh->vertices = new double[nv * dim];
    intermesh->elements = new int[ne * (dim + 1)];
    intermesh->bdrelements = new int[nbdr * dim];
    intermesh->elattrs = new int[ne];
    intermesh->bdrattrs = new int[nbdr];

    if (with_gindices_flag != 0)
    {
        intermesh->withgindicesflag = 1;
        intermesh->vert_gindices = new int[nv];
    }
    else
        intermesh->withgindicesflag = 0;

    return;
}

void Mesh::IntermeshDelete( IntermediateMesh * intermesh_pt)
{
    delete [] intermesh_pt->vertices;
    delete [] intermesh_pt->elements;
    delete [] intermesh_pt->bdrelements;
    delete [] intermesh_pt->elattrs;
    delete [] intermesh_pt->bdrattrs;

    if ( intermesh_pt->withgindicesflag != 0)
        delete [] intermesh_pt->vert_gindices;

    delete intermesh_pt;

    return;
}

void Mesh::InterMeshPrint (IntermediateMesh * local_intermesh, int suffix, const char * filename)
{
    int dim = local_intermesh->dim;
    int ne = local_intermesh->ne;
    int nv = local_intermesh->nv;
    int nbe = local_intermesh->nbe;

    ofstream myfile;
    char csuffix[20];
    sprintf (csuffix, "_%d.intermesh", suffix);

    char fileoutput[250];
    strcpy (fileoutput, filename);
    strcat (fileoutput, csuffix);

    myfile.open (fileoutput);

    myfile << "elements: \n";
    myfile << ne << endl;
    for ( int i = 0; i < ne; ++i )
    {
        myfile << local_intermesh->elattrs[i] << " ";
        for ( int j = 0; j < dim + 1; ++j )
            myfile << local_intermesh->elements[i*(dim+1) + j] << " ";
        myfile << endl;
    }
    myfile << endl;

    myfile << "boundary: \n";
    myfile << nbe << endl;
    for ( int i = 0; i < nbe; ++i )
    {
        myfile << local_intermesh->bdrattrs[i] << " ";
        for ( int j = 0; j < dim; ++j )
            myfile << local_intermesh->bdrelements[i*dim + j] << " ";
        myfile << endl;
    }
    myfile << endl;

    myfile << "vertices: \n";
    myfile << nv << endl;
    int withgindicesflag = 0;
    if (local_intermesh->withgindicesflag != 0)
        withgindicesflag = 1;
    for ( int i = 0; i < nv; ++i )
    {
        for ( int j = 0; j < dim; ++j )
            myfile << local_intermesh->vertices[i*dim + j] << " ";
        if (withgindicesflag == 1)
            myfile << " gindex: " << local_intermesh->vert_gindices[i];
        myfile << endl;
    }
    myfile << endl;

    myfile.close();

    return;
}

// Takes the 4d mesh with elements, vertices and boundary already created
// and creates all the internal structure.
// Used inside the Mesh constructor.
// "refine" argument is added for handling 2D case, when refinement marker routines
// should be called before creating structures for shared entities which goes
// before the call to CreateInternal...()
// Probably for parallel mesh generator some tables are generated twice // FIX IT
void Mesh::CreateInternalMeshStructure (int refine)
{
    int i, j, curved = 0;
    //int refine = 1;
    bool fix_orientation = true;
    int generate_edges = 1;

    Nodes = NULL;
    own_nodes = 1;
    NURBSext = NULL;
    ncmesh = NULL;
    last_operation = Mesh::NONE;
    sequence = 0;

    InitTables();

    //for a 4d mesh sort the element and boundary element indices by the node numbers
    if(spaceDim==4)
    {
        swappedElements.SetSize(NumOfElements);
        DenseMatrix J(4,4);
        for (j = 0; j < NumOfElements; j++)
        {
            if (elements[j]->GetType() == Element::PENTATOPE)
            {
                int *v = elements[j]->GetVertices();
                Sort5(v[0], v[1], v[2], v[3], v[4]);

                GetElementJacobian(j, J);
                if(J.Det() < 0.0)
                {
                    swappedElements[j] = true;
                    Swap(v);
                }else
                {
                    swappedElements[j] = false;
                }
            }

        }
        for (j = 0; j < NumOfBdrElements; j++)
        {
            if (boundary[j]->GetType() == Element::TETRAHEDRON)
            {
                int *v = boundary[j]->GetVertices();
                Sort4(v[0], v[1], v[2], v[3]);
            }
        }
    }

    // at this point the following should be defined:
    //  1) Dim
    //  2) NumOfElements, elements
    //  3) NumOfBdrElements, boundary
    //  4) NumOfVertices, with allocated space in vertices
    //  5) curved
    //  5a) if curved == 0, vertices must be defined
    //  5b) if curved != 0 and read_gf != 0,
    //         'input' must point to a GridFunction
    //  5c) if curved != 0 and read_gf == 0,
    //         vertices and Nodes must be defined

    if (spaceDim == 0)
    {
       spaceDim = Dim;
    }

    InitBaseGeom();

    // set the mesh type ('meshgen')
    SetMeshGen();


    if (NumOfBdrElements == 0 && Dim > 2)
    {
       // in 3D, generate boundary elements before we 'MarkForRefinement'
       if(Dim==3) GetElementToFaceTable();
       else if(Dim==4)
       {
           GetElementToFaceTable4D();
       }
       GenerateFaces();
       GenerateBoundaryElements();
    }


    if (!curved)
    {
       // check and fix element orientation
       CheckElementOrientation(fix_orientation);

       if (refine)
       {
          MarkForRefinement();
       }
    }

    if (Dim == 1)
    {
       GenerateFaces();
    }

    // generate the faces
    if (Dim > 2)
    {
           if(Dim==3) GetElementToFaceTable();
           else if(Dim==4)
           {
               GetElementToFaceTable4D();
           }

           GenerateFaces();

           if(Dim==4)
           {
              ReplaceBoundaryFromFaces();

              GetElementToPlanarTable();
              GeneratePlanars();

 //			 GetElementToQuadTable4D();
 //			 GenerateQuads4D();
           }

       // check and fix boundary element orientation
       if ( !(curved && (meshgen & 1)) )
       {
          CheckBdrElementOrientation();
       }
    }
    else
    {
       NumOfFaces = 0;
    }

    // generate edges if requested
    if (Dim > 1 && generate_edges == 1)
    {
       // el_to_edge may already be allocated (P2 VTK meshes)
       if (!el_to_edge) { el_to_edge = new Table; }
       NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
       if (Dim == 2)
       {
          GenerateFaces(); // 'Faces' in 2D refers to the edges
          if (NumOfBdrElements == 0)
          {
             GenerateBoundaryElements();
          }
          // check and fix boundary element orientation
          if ( !(curved && (meshgen & 1)) )
          {
             CheckBdrElementOrientation();
          }
       }
    }
    else
    {
       NumOfEdges = 0;
    }

    //// generate the arrays 'attributes' and ' bdr_attributes'
    SetAttributes();

    return;
}


//used for comparing the d-dimensional points by their coordinates
typedef std::pair<vector<double>, int> PairPoint;
struct CmpPairPoint
{
    bool operator()(const PairPoint& a, const PairPoint& b)
    {
        int size = a.first.size();
        if ( size != b.first.size() )
        {
            cerr << "Error: Points have different dimensions" << endl << flush;
            return false;
        }
        else
        {
            for ( int i = 0; i < size; ++i)
                if ( fabs(a.first[i] - b.first[i]) > MYZEROTOL )
                    return a.first[i] < b.first[i];
            cerr << "Error, points are the same!" << endl << flush;
            cerr << "Point 1:" << endl;
            for ( int i = 0; i < size; ++i)
                cerr << a.first[i] << " ";
            cerr << endl;
            cerr << "Point 2:" << endl;
            for ( int i = 0; i < size; ++i)
                cerr << b.first[i] << " ";
            cerr << endl << flush;
            return false;
        }

    }
};

// takes coordinates of points and returns a permutation which makes the given vertices
// preserve the geometrical order (based on their coordinates comparison)
void sortingPermutationNew( const vector<vector<double>>& values, int * permutation)
{
    vector<PairPoint> pairs;
    pairs.reserve(values.size());
    for (int i = 0; i < values.size(); i++)
        pairs.push_back(PairPoint(values[i], i));

    sort(pairs.begin(), pairs.end(), CmpPairPoint());

    typedef std::vector<PairPoint>::const_iterator I;
    int count = 0;
    for (I p = pairs.begin(); p != pairs.end(); ++p)
        permutation[count++] = p->second;
}

/*
 * // old variant, without template for PairPoint
typedef std::pair<double*, int> Pair2DPoint; //double * is actually double[2]
typedef std::pair<double*, int> Pair3DPoint; //double * is actually double[3]
typedef std::pair<double*, int> Pair4DPoint; //double * is actually double[4]

struct CmpPair2DPoint
{
    bool operator()(const Pair2DPoint& a, const Pair2DPoint& b)
    {
        //cout << "Comparing tuples:" << endl;
        //cout << "tuple 1: " << a.first[0] << " " << a.first[1] << endl;
        //cout << "tuple 2: " << b.first[0] << " " << b.first[1] << endl;
        if ( fabs(a.first[0] - b.first[0]) > MYZEROTOL )
        {
            return a.first[0] < b.first[0];
        }
        else if ( fabs(a.first[1] - b.first[1]) > MYZEROTOL )
        {
            return a.first[1] < b.first[1];
        }
        else
        {
            cout << "Error, two tuples of doubles are the same!" << endl;
            cout << "tuple 1: " << a.first[0] << " " << a.first[1] << endl;
            cout << "tuple 2: " << b.first[0] << " " << b.first[1] << endl;
            return false;
        }
    }
};

struct CmpPair3DPoint
{
    bool operator()(const Pair3DPoint& a, const Pair3DPoint& b)
    {
        //cout << "Comparing triples:" << endl;
        //cout << "triple 1: " << a.first[0] << " " << a.first[1] << " " << a.first[2] << endl;
        //cout << "triple 2: " << b.first[0] << " " << b.first[1] << " " << b.first[2] << endl;
        if ( fabs(a.first[0] - b.first[0]) > MYZEROTOL )
        {
            //cout << "first case" << endl;
            //bool res = a.first[0] < b.first[0];
            //cout << "res = " << res << endl;
            return a.first[0] < b.first[0];
        }
        else if ( fabs(a.first[1] - b.first[1]) > MYZEROTOL )
        {
            //cout << "second case" << endl;
            //bool res = a.first[1] < b.first[1];
            //cout << "res = " << res << endl;
            return a.first[1] < b.first[1];
        }
        else if ( fabs(a.first[2] - b.first[2] ) > MYZEROTOL )
        {
            //cout << "third case" << endl;
            //bool res = a.first[2] < b.first[2];
            //cout << "res = " << res << endl;
            return a.first[2] < b.first[2];
        }
        else
        {
            cout << "Error, two triples of doubles are the same!" << endl;
            cout << "triple 1: " << a.first[0] << " " << a.first[1] << " " << a.first[2] << endl;
            cout << "triple 2: " << b.first[0] << " " << b.first[1] << " " << b.first[2] << endl;
            return false;
        }
    }
};

struct CmpPair4DPoint
{
    bool operator()(const Pair4DPoint& a, const Pair4DPoint& b)
    {
        //cout << "Comparing quadruples:" << endl;
        //cout << "quadruple 1: " << a.first[0] << " " << a.first[1] << " " << a.first[2] << " " << a.first[3] << endl;
        //cout << "quadruple 2: " << b.first[0] << " " << b.first[1] << " " << b.first[2] << " " << b.first[3] << endl;
        if ( fabs(a.first[0] - b.first[0]) > MYZEROTOL )
        {
            return a.first[0] < b.first[0];
        }
        else if ( fabs(a.first[1] - b.first[1]) > MYZEROTOL )
        {
            return a.first[1] < b.first[1];
        }
        else if ( fabs(a.first[2] - b.first[2] ) > MYZEROTOL )
        {
            return a.first[2] < b.first[2];
        }
        else if ( fabs(a.first[3] - b.first[3] ) > MYZEROTOL )
        {
            return a.first[3] < b.first[3];
        }
        else
        {
            cout << "Error, two quadruples of doubles are the same!" << endl;
            cout << "quadruple 1: " << a.first[0] << " " << a.first[1] << " " << a.first[2] << " " << a.first[3] << endl;
            cout << "quadruple 2: " << b.first[0] << " " << b.first[1] << " " << b.first[2] << " " << b.first[3] << endl;
            return false;
        }
    }
};


void sortingPermutation( int dim, const std::vector<double*>& values, int * permutation)
{
    if (dim == 2)
    {
        std::vector<Pair2DPoint> pairs;
        pairs.reserve(3);
        for (int i = 0; i < (int)values.size(); i++)
            pairs.push_back(Pair2DPoint(values[i], i));

        std::sort(pairs.begin(), pairs.end(), CmpPair2DPoint());

        typedef std::vector<Pair2DPoint>::const_iterator I;
        int count = 0;
        for (I p = pairs.begin(); p != pairs.end(); ++p)
            permutation[count++] = p->second;
    }
    else if (dim == 3)
    {
        std::vector<Pair3DPoint> pairs;
        pairs.reserve(3);
        for (int i = 0; i < (int)values.size(); i++)
            pairs.push_back(Pair3DPoint(values[i], i));

        std::sort(pairs.begin(), pairs.end(), CmpPair3DPoint());

        typedef std::vector<Pair3DPoint>::const_iterator I;
        int count = 0;
        for (I p = pairs.begin(); p != pairs.end(); ++p)
            permutation[count++] = p->second;
    }
    else if (dim == 4)
    {
        //cout << "I am falling here?" << endl << flush;

        //cout << "Input vector is" << endl;
        std::vector<Pair4DPoint> pairs;
        pairs.reserve(3);

        //cout << "Pushing 4D points: size = " << (int)values.size() << endl << flush;

        for (int i = 0; i < (int)values.size(); i++)
        {
            //cout << "i = " << i << endl;
            pairs.push_back(Pair4DPoint(values[i], i));
        }

        //cout << "Sorting" << endl << flush;

        std::sort(pairs.begin(), pairs.end(), CmpPair4DPoint());

        //cout << "Creating permutation" << endl << flush;

        typedef std::vector<Pair4DPoint>::const_iterator I;
        int count = 0;
        for (I p = pairs.begin(); p != pairs.end(); ++p)
            permutation[count++] = p->second;
        //cout << "Nope!" << endl;
    }
    else
        cout << "Wrong value of dim, must be 2, 3 or 4" << endl << flush;

    return;
}
*/


// Does the same as MeshSpaceTimeCylinder_onlyArrays() but outputs InterMediateMesh structure
// works only in 4d case
Mesh::IntermediateMesh * Mesh::MeshSpaceTimeCylinder_toInterMesh (double tau, int Nsteps, int bnd_method, int local_method)
{
    int Dim3D = Dimension(), NumOf3DElements = GetNE(),
            NumOf3DBdrElements = GetNBE(),
            NumOf3DVertices = GetNV();
    int NumOf4DElements, NumOf4DBdrElements, NumOf4DVertices;

    if ( Dim3D != 3 )
    {
       cerr << "Wrong dimension in MeshSpaceTimeCylinder(): " << Dim3D << endl;
       return NULL;
    }

    int Dim = Dim3D + 1;
    // for each 3D element and each time slab a 4D-prism with 3D element as a base
    // is decomposed into 4 pentatopes
    NumOf4DElements = NumOf3DElements * 4 * Nsteps;
    // no additional vertices so far
    NumOf4DVertices = NumOf3DVertices * (Nsteps + 1);
    // lateral 4d bdr faces (one for each 3d bdr face) + lower + upper bases
    // of the space-time cylinder
    NumOf4DBdrElements = NumOf3DBdrElements * 3 * Nsteps +
            NumOf3DElements + NumOf3DElements;

    // assuming that the 3D mesh contains elements of the same type
    int vert_per_base = GetElement(0)->GetNVertices();
    int vert_per_prism = 2 * vert_per_base;
    int vert_per_latface = Dim3D * 2;

    IntermediateMesh * intermesh = new IntermediateMesh;
    IntermeshInit( intermesh, Dim, NumOf4DVertices, NumOf4DElements, NumOf4DBdrElements, 1);

    Element * el;

    int * pentatops;
    if (local_method == 1 || local_method == 2)
    {
        pentatops = new int[Dim * (Dim + 1)]; // pentatop's vertices' indices
    }
    else // local_method = 0
    {
        int nsliver = 5; //why 5? how many slivers can b created by qhull? maybe 0 if we don't joggle inside qhull but perturb the coordinates before?
        pentatops = new int[(Dim + nsliver) * (Dim + 1)]; // pentatop's vertices' indices + probably sliver pentatopes
    }

    // stores indices of tetrahedron vertices produced by qhull for all lateral faces
    // (Dim) lateral faces, Dim3D tetrahedrons for each face (which is 3D prism)
    // (Dim3D + 1) vertex indices for each tetrahedron. Used in local_method = 1.
    int * tetrahedronsAll;
    if (local_method == 1 )
        tetrahedronsAll = new int[Dim3D * (Dim3D + 1) * Dim ];


    Array<int> elverts_base;
    Array<int> elverts_prism;

    int temptetra[4]; // temporary array for vertex indices of a pentatope face (used in local_method = 0 and 2)
    int temp[5]; //temp array for pentatops in local_method = 1;
    Array2D<int> vert_to_vert_prism; // for a 4D prism
    // row ~ lateral face of the 4d prism
    // first 6 columns - indices of vertices belonging to the lateral face,
    // last 2 columns - indices of the rest 2 vertices of the prism
    Array2D<int> latfacets_struct;
    // coordinates of vertices of a lateral face of 4D prism
    double * vert_latface;
    // coordinates of vertices of a 3D base (triangle) of a lateral face of 4D prism
    double * vert_3Dlatface;
    if (local_method == 1)
    {
        vert_latface =  new double[Dim * vert_per_latface];
        vert_3Dlatface = new double[Dim3D * vert_per_latface];
        latfacets_struct.SetSize(Dim, vert_per_prism);
        vert_to_vert_prism.SetSize(vert_per_prism, vert_per_prism);
    }


    // coordinates of vertices of 4D prism
    double * elvert_coordprism = new double[Dim * vert_per_prism];

    //char qhull_flags[250];
    char * qhull_flags;
    if (local_method == 0 || local_method == 1)
    {
        qhull_flags = new char[250];
        sprintf(qhull_flags, "qhull d Qbb");
    }


    if (local_method < 0 && local_method > 2)
    {
        cout << "Local method = " << local_method << " is not supported" << endl;
        return NULL;
    }
    //else
        //cout << "Using local_method = " << local_method << " for constructing pentatops" << endl;

    if ( bnd_method != 0 && bnd_method != 1)
    {
        cout << "Illegal value of bnd_method = " << bnd_method << " (must be 0 or 1)" << endl;
        return NULL;
    }
    //else
        //cout << "Using bnd_method = " << bnd_method << " for creating boundary elements" << endl;

    //cout << "Using local_method = 0 in MeshSpaceTimeCylinder_toInterMesh()" << endl;

    int almostjogglers[4];
    //int permutation[4];
    vector<double*> lcoords(Dim);
    vector<vector<double>> lcoordsNew(Dim);

    Vector vert_coord3d(Dim3D * NumOf3DVertices);
    GetVertices(vert_coord3d);
    //printDouble2D(vert_coord3d, 10, Dim3D);


    // adding all the 4d vertices to the mesh
    int vcount = 0;
    for ( int tslab = 0; tslab <= Nsteps; ++tslab)
    {
        // adding the vertices from the slab to the mesh4d
        for ( int vert = 0; vert < NumOf3DVertices; ++vert)
        {
            //tempvert[0] = vert_coord3d[vert + 0 * NumOf3DVertices];
            //tempvert[1] = vert_coord3d[vert + 1 * NumOf3DVertices];
            //tempvert[2] = vert_coord3d[vert + 2 * NumOf3DVertices];
            //tempvert[3] = tau * tslab;
            //mesh4d->AddVertex(tempvert);
            intermesh->vertices[vcount*Dim + 0] = vert_coord3d[vert + 0 * NumOf3DVertices];
            intermesh->vertices[vcount*Dim + 1] = vert_coord3d[vert + 1 * NumOf3DVertices];
            intermesh->vertices[vcount*Dim + 2] = vert_coord3d[vert + 2 * NumOf3DVertices];
            intermesh->vertices[vcount*Dim + 3] = tau * tslab;
            vcount++;
        }
    }

    //delete(tempvert);

    int facebdrmarker[Dim]; // for each (of Dim) 3d element faces stores \
    1 if it is at the boundary and 0 else
    std::set< std::vector<int> > BdrTriSet; // std::set of the 3D boundary elements \
    using set allows to perform a search with O(log N_elem) operations
    Element * bdrel;

    Array<int> face_bndflags(GetNFaces());

    // if = 0, a search algorithm is used for defining whether faces of a given 3d element
    // are at the boundary.
    // if = 1, instead an array face_bndflags is used, which stores 0 and 1 depending on
    // whether the face is at the boundary, + el_to_face table which is usually already
    // generated for the 3d mesh
    //int bnd_method = 1;

    if (bnd_method == 0)
    {
        // putting 3d boundary elements from mesh3d to the set BdrTriSet
        for ( int boundelem = 0; boundelem < NumOf3DBdrElements; ++boundelem)
        {
            //cout << "boundelem No. " << boundelem << endl;
            bdrel = GetBdrElement(boundelem);
            int * bdrverts = bdrel->GetVertices();

            std::vector<int> buff (bdrverts, bdrverts+3);
            std::sort (buff.begin(), buff.begin()+3);

            BdrTriSet.insert(buff);
        }
        /*
        for (vector<int> temp : BdrTriSet)
        {
            cout << temp[0] << " " <<  temp[1] << " " << temp[2] << endl;
        }
        cout<<endl;
        */
    }
    else // bnd_method = 1
    {
        if (el_to_face == NULL)
        {
            cout << "Have to built el_to_face" << endl;
            GetElementToFaceTable(0);
        }

        //cout << "Special print" << endl;
        //cout << mesh3d.el_to_face(elind, facelind);
        //cout << "be_to_face" << endl;
        //mesh3d.be_to_face.Print();

        //cout << "nfaces = " << mesh3d.GetNFaces();
        //cout << "nbe = " << mesh3d.GetNBE() << endl;
        //cout << "boundary.size = " << mesh3d.boundary.Size() << endl;

        face_bndflags = -1;
        for ( int i = 0; i < NumOf3DBdrElements; ++i )
            face_bndflags[be_to_face[i]] = 1;

        //cout << "face_bndflags" << endl;
        //face_bndflags.Print();
    }

    int ordering[vert_per_base];
    int antireordering[vert_per_base]; // used if bnd_method = 0 and local_method = 2
    Array<int> tempelverts(vert_per_base);

    int bdrelcount = 0;
    int elcount = 0;

    // main loop creates 4d elements for all time slabs for all 3d elements
    // loop over 3d elements
    for ( int elind = 0; elind < NumOf3DElements; elind++ )
    //for ( int elind = 0; elind < 1; ++elind )
    {
        //cout << "element " << elind << endl;

        el = GetElement(elind);

        // 1. getting indices of 3d element vertices and their coordinates in the prism
        el->GetVertices(elverts_base);

        // for local_method 2 we need to reorder the local vertices of the prism to preserve the
        // the order in some global sense  = lexicographical order of the vertex coordinates
        if (local_method == 2)
        {
            // setting vertex coordinates for 4d prism, lower base
            for ( int i = 0; i < vert_per_base; ++i)
            {
                //double * temp = vert_coord3d + Dim3D * elverts_base[i];
                //elvert_coordprism[Dim * i + 0] = temp[0];
                //elvert_coordprism[Dim * i + 1] = temp[1];
                //elvert_coordprism[Dim * i + 2] = temp[2];
                elvert_coordprism[Dim * i + 0] = vert_coord3d[elverts_base[i] + 0 * NumOf3DVertices];
                elvert_coordprism[Dim * i + 1] = vert_coord3d[elverts_base[i] + 1 * NumOf3DVertices];
                elvert_coordprism[Dim * i + 2] = vert_coord3d[elverts_base[i] + 2 * NumOf3DVertices];
            }


            /*
            // * old
            for (int vert = 0; vert < Dim; ++vert)
                lcoords[vert] = elvert_coordprism + Dim * vert;

            sortingPermutation(Dim3D, lcoords, ordering);
            */




            for (int vert = 0; vert < Dim; ++vert)
                lcoordsNew[vert].assign(elvert_coordprism + Dim * vert,
                                        elvert_coordprism + Dim * vert + Dim3D);

            sortingPermutationNew(lcoordsNew, ordering);




            // UGLY: Fix it
            for ( int i = 0; i < vert_per_base; ++i)
                tempelverts[i] = elverts_base[ordering[i]];

            for ( int i = 0; i < vert_per_base; ++i)
                elverts_base[i] = tempelverts[i];
        }

        //for ( int k = 0; k < elverts_base.Size(); ++k )
            //cout << "elverts[" << k << "] = " << elverts_base[k] << endl;

        // 2. understanding which of the 3d element faces (triangles) are at the boundary
        int local_nbdrfaces = 0;
        set<set<int>> LocalBdrs;
        if (bnd_method == 0)
        {
            vector<int> face(Dim3D);
            for (int i = 0; i < Dim; ++i )
            {
                // should be consistent with lateral faces ordering in latfacet structure
                // if used with local_method = 1
                if ( i == 0)
                {
                    face[0] = elverts_base[0];
                    face[1] = elverts_base[1];
                    face[2] = elverts_base[2];
                }
                if ( i == 1)
                {
                    face[0] = elverts_base[1];
                    face[1] = elverts_base[2];
                    face[2] = elverts_base[3];
                }
                if ( i == 2)
                {
                    face[0] = elverts_base[2];
                    face[1] = elverts_base[3];
                    face[2] = elverts_base[0];
                }
                if ( i == 3)
                {
                    face[0] = elverts_base[3];
                    face[1] = elverts_base[0];
                    face[2] = elverts_base[1];
                }

                /*
                int cnt = 0;
                for ( int j = 0; j < Dim; ++j)
                    if ( j != i )
                        face[cnt++] = elverts_base[j];
                */
                sort(face.begin(), face.begin()+Dim3D);
                //cout << face[0] << " " <<  face[1] << " " << face[2] << endl;

                if (BdrTriSet.find(face) != BdrTriSet.end() )
                {
                    //cout << "is at the boundary" << endl;
                    local_nbdrfaces++;
                    facebdrmarker[i] = 1;
                    set<int> face_as_set;
                    if ( i == 0)
                    {
                        face_as_set.insert(0);
                        face_as_set.insert(1);
                        face_as_set.insert(2);
                    }
                    if ( i == 1)
                    {
                        face_as_set.insert(1);
                        face_as_set.insert(2);
                        face_as_set.insert(3);
                    }
                    if ( i == 2)
                    {
                        face_as_set.insert(2);
                        face_as_set.insert(3);
                        face_as_set.insert(0);
                    }
                    if ( i == 3)
                    {
                        face_as_set.insert(3);
                        face_as_set.insert(0);
                        face_as_set.insert(1);
                    }
                    LocalBdrs.insert(face_as_set);
                }
                else
                    facebdrmarker[i] = 0;
            }

        } //end of if bnd_method == 0
        else
        //set<set<int>> LocalBdrs2;
        {
            int * faceinds = el_to_face->GetRow(elind);
            for ( int facelind = 0; facelind < Dim; ++facelind)
            {
                int faceind = faceinds[facelind];
                if (face_bndflags[faceind] == 1)
                {
                    Array<int> temp(3);
                    GetFaceVertices(faceind, temp);
                    //set<int> face_as_set(temp, temp+3);
                    set<int> face_as_set;
                    for ( int vert = 0; vert < Dim3D; ++vert )
                        face_as_set.insert(temp[vert]);
                    LocalBdrs.insert(face_as_set);

                    local_nbdrfaces++;
                }

            } // end of loop over element faces

        }

        //cout << "Welcome the facebdrmarker" << endl;
        //printInt2D(facebdrmarker, 1, Dim);

        /*
        cout << "Welcome the LocalBdrs" << endl;
        for ( set<int> tempset: LocalBdrs )
        {
            cout << "element of LocalBdrs for el = " << elind << endl;
            for (int ind: tempset)
                cout << ind << " ";
            cout << endl;
        }
        */


        // 3. loop over all 4D time slabs above a given 3d element
        for ( int tslab = 0; tslab < Nsteps; ++tslab)
        {
            //cout << "tslab " << tslab << endl;

            //3.1 getting vertex indices for the 4d prism
            elverts_prism.SetSize(vert_per_prism);
            for ( int i = 0; i < vert_per_base; ++i)
            {
                elverts_prism[i] = elverts_base[i] + tslab * NumOf3DVertices;
                elverts_prism[i + vert_per_base] = elverts_base[i] + (tslab + 1) * NumOf3DVertices;
            }

            // 3.2 for the first time slab we add the tetrahedrons in the lower base \
            to the bdr elements
            if ( tslab == 0 )
            {
                //cout << "zero slab: adding boundary element:" << endl;
                //NewBdrTri = new Tetrahedron(elverts_prism);
                //NewBdrTri->SetAttribute(1);
                //mesh4d->AddBdrElement(NewBdrTri);

                intermesh->bdrelements[bdrelcount*Dim + 0] = elverts_prism[0];
                intermesh->bdrelements[bdrelcount*Dim + 1] = elverts_prism[1];
                intermesh->bdrelements[bdrelcount*Dim + 2] = elverts_prism[2];
                intermesh->bdrelements[bdrelcount*Dim + 3] = elverts_prism[3];
                intermesh->bdrattrs[bdrelcount] = 1;
                bdrelcount++;

                /*
                const int nv = NewBdrTri->GetNVertices();
                const int *v = NewBdrTri->GetVertices();
                for (int j = 0; j < nv; j++)
                {
                   cout << ' ' << v[j];
                }
                cout << endl;
                */
            }
            // 3.3 for the last time slab we add the tetrahedrons in the upper base \
            to the bdr elements
            if ( tslab == Nsteps - 1 )
            {
                //cout << "last slab: adding boundary element:" << endl;
                //NewBdrTri = new Tetrahedron(elverts_prism + vert_per_base);
                //NewBdrTri->SetAttribute(3);
                //mesh4d->AddBdrElement(NewBdrTri);

                intermesh->bdrelements[bdrelcount*Dim + 0] = elverts_prism[0 + vert_per_base];
                intermesh->bdrelements[bdrelcount*Dim + 1] = elverts_prism[1 + vert_per_base];
                intermesh->bdrelements[bdrelcount*Dim + 2] = elverts_prism[2 + vert_per_base];
                intermesh->bdrelements[bdrelcount*Dim + 3] = elverts_prism[3 + vert_per_base];
                intermesh->bdrattrs[bdrelcount] = 3;
                bdrelcount++;

                /*
                const int nv = NewBdrTri->GetNVertices();
                const int *v = NewBdrTri->GetVertices();
                for (int j = 0; j < nv; j++)
                {
                   cout << ' ' << v[j];
                }
                cout << endl;
                */
            }

            //elverts_prism.Print();
            //return;

            // printInt2D(pentatops, Dim, Dim + 1);

            if (local_method == 0 || local_method == 1)
            {
                // 3.4 setting vertex coordinates for 4d prism, lower base
                for ( int i = 0; i < vert_per_base; ++i)
                {
                    //double * temp = vert_coord3d + Dim3D * elverts_base[i];
                    //elvert_coordprism[Dim * i + 0] = temp[0];
                    //elvert_coordprism[Dim * i + 1] = temp[1];
                    //elvert_coordprism[Dim * i + 2] = temp[2];
                    elvert_coordprism[Dim * i + 0] = vert_coord3d[elverts_base[i] + 0 * NumOf3DVertices];
                    elvert_coordprism[Dim * i + 1] = vert_coord3d[elverts_base[i] + 1 * NumOf3DVertices];
                    elvert_coordprism[Dim * i + 2] = vert_coord3d[elverts_base[i] + 2 * NumOf3DVertices];
                    elvert_coordprism[Dim * i + 3] = tslab * tau;


                    /*
                    std::cout << \
                                 "vert_coord3d[" << elverts_prism[i] + 0 * NumOf3DVertices << "] = " << \
                                 vert_coord3d(elverts_prism[i] + 0 * NumOf3DVertices) << " " << \
                                 "vert_coord3d[" << elverts_prism[i] + 1 * NumOf3DVertices << "] = " << \
                                 vert_coord3d[elverts_prism[i] + 1 * NumOf3DVertices] << " " << \
                                 "vert_coord3d[" << elverts_prism[i] + 2 * NumOf3DVertices << "] = "  << \
                                 vert_coord3d[elverts_prism[i] + 0 * NumOf3DVertices] << std::endl;

                    //std::cout << "indices in coordprism which were set:" << endl;
                    //std::cout << Dim * i + 0 << " " << Dim * i + 1 << " " << Dim * i + 2 << endl;
                    std::cout << "we got:" << endl;
                    std::cout << "elvert_coordprism for vertex " <<  i << ": " << \
                                 elvert_coordprism[Dim * i + 0] << " " << elvert_coordprism[Dim * i + 1] << \
                                 " " << elvert_coordprism[Dim * i + 2] << " " << \
                                 elvert_coordprism[Dim * i + 3] << endl;

                    double temp = vert_coord3d[elverts_prism[i] + 2 * NumOf3DVertices];
                    std::cout << "temp = " << temp << endl;
                    */

                }

                //cout << "Welcome the vertex coordinates for the 4d prism base " << endl;
                //printDouble2D(elvert_coordprism, vert_per_base, Dim);


                /*
                // * old
                for (int vert = 0; vert < Dim; ++vert)
                    lcoords[vert] = elvert_coordprism + Dim * vert;


                //cout << "vector double * lcoords:" << endl;
                //for ( int i = 0; i < Dim; ++i)
                    //cout << "lcoords[" << i << "]: " << lcoords[i][0] << " " << lcoords[i][1] << " " << lcoords[i][2] << endl;

                sortingPermutation(Dim3D, lcoords, permutation);
                */



                for (int vert = 0; vert < Dim; ++vert)
                    lcoordsNew[vert].assign(elvert_coordprism + Dim * vert,
                                            elvert_coordprism + Dim * vert + Dim3D);

                sortingPermutationNew(lcoordsNew, ordering);



                //cout << "Welcome the permutation:" << endl;
                //cout << permutation[0] << " " << permutation[1] << " " << permutation[2] << " " << permutation[3] << endl;

                int joggle_coeff = 0;
                for ( int i = 0; i < Dim; ++i)
                    almostjogglers[ordering[i]] = joggle_coeff++;

                //cout << "Welcome the joggle coeffs:" << endl;
                //cout << almostjogglers[0] << " " << almostjogglers[1] << " " << almostjogglers[2] << " " << almostjogglers[3] << endl;



                // 3.5 setting vertex coordinates for 4d prism, upper layer \
                with joggling of the time coordinate depending on the global vertex indices \
                Joggling is required for getting unique Delaunay tesselation and should be  \
                the same for vertices shared between different elements or at least produce \
                the same Delaunay triangulation in the shared faces.
                double joggle;
                for ( int i = 0; i < vert_per_base; ++i)
                {
                    //double * temp = vert_coord3d + Dim3D * elverts_base[i];
                    //elvert_coordprism[Dim * (vert_per_base + i) + 0] = temp[0];
                    //elvert_coordprism[Dim * (vert_per_base + i) + 1] = temp[1];
                    //elvert_coordprism[Dim * (vert_per_base + i) + 2] = temp[2];
                    elvert_coordprism[Dim * (vert_per_base + i) + 0] = elvert_coordprism[Dim * i + 0];
                    elvert_coordprism[Dim * (vert_per_base + i) + 1] = elvert_coordprism[Dim * i + 1];
                    elvert_coordprism[Dim * (vert_per_base + i) + 2] = elvert_coordprism[Dim * i + 2];

                    joggle = 1.0e-2 * (almostjogglers[i]);
                    //joggle = 1.0e-2 * elverts_prism[i + vert_per_base] * 1.0 / NumOf4DVertices;
                    //double joggle = 1.0e-2 * i;
                    elvert_coordprism[Dim * (vert_per_base + i) + 3] = (tslab + 1) * tau * ( 1.0 + joggle );

                }

                //cout << "Welcome the vertex coordinates for the 4d prism" << endl;
                //printDouble2D(elvert_coordprism, 2 * vert_per_base, Dim);

                if (local_method == 0)
                {
                    // ~ 3.6 - 3.10 (in LONGWAY): constructing pentatopes and boundary elements
#ifdef WITH_QHULL
                    qhT qh_qh;                /* Qhull's data structure.  First argument of most calls */
                    qhT *qh= &qh_qh;
                    int curlong, totlong;     /* memory remaining after qh_memfreeshort */

                    double volumetol = 1.0e-8;
                    qhull_wrapper(pentatops, qh, elvert_coordprism, Dim, volumetol, qhull_flags);

                    qh_freeqhull(qh, !qh_ALL);
                    qh_memfreeshort(qh, &curlong, &totlong);
                    if (curlong || totlong)  /* could also check previous runs */
                    {
                      fprintf(stderr, "qhull internal warning (user_eg, #3): did not free %d bytes \
                            of long memory (%d pieces)\n", totlong, curlong);
                    }
#else
                    cout << "Cannot work without WITH_QHULL" << endl;
#endif
                } // end of if local_method = 0

                if (local_method == 1)
                {
                    // 3.6 - 3.10: constructing pentatopes

                    setzero(&vert_to_vert_prism);

                    // 3.6 creating vert_to_vert for the prism before Delaunay (adding 4d prism edges)
                    for ( int i = 0; i < el->GetNEdges(); i++)
                    {
                        const int * edge = el->GetEdgeVertices(i);
                        //cout << "edge: " << edge[0] << " " << edge[1] << std::endl;
                        vert_to_vert_prism(edge[0], edge[1]) = 1;
                        vert_to_vert_prism(edge[1], edge[0]) = 1;
                        vert_to_vert_prism(edge[0] + vert_per_base, edge[1] + vert_per_base) = 1;
                        vert_to_vert_prism(edge[1] + vert_per_base, edge[0] + vert_per_base) = 1;
                    }

                    for ( int i = 0; i < vert_per_base; i++)
                    {
                        vert_to_vert_prism(i, i) = 1;
                        vert_to_vert_prism(i + vert_per_base, i + vert_per_base) = 1;
                        vert_to_vert_prism(i, i + vert_per_base) = 1;
                        vert_to_vert_prism(i + vert_per_base, i) = 1;
                    }

                    //cout << "vert_to_vert before delaunay" << endl;
                    //printArr2DInt (&vert_to_vert_prism);
                    //cout << endl;

                    // 3.7 creating latfacet structure (brute force), for 4D tetrahedron case
                    // indices are local w.r.t to the 4d prism!!!
                    latfacets_struct(0,0) = 0;
                    latfacets_struct(0,1) = 1;
                    latfacets_struct(0,2) = 2;
                    latfacets_struct(0,6) = 3;

                    latfacets_struct(1,0) = 1;
                    latfacets_struct(1,1) = 2;
                    latfacets_struct(1,2) = 3;
                    latfacets_struct(1,6) = 0;

                    latfacets_struct(2,0) = 2;
                    latfacets_struct(2,1) = 3;
                    latfacets_struct(2,2) = 0;
                    latfacets_struct(2,6) = 1;

                    latfacets_struct(3,0) = 3;
                    latfacets_struct(3,1) = 0;
                    latfacets_struct(3,2) = 1;
                    latfacets_struct(3,6) = 2;

                    for ( int i = 0; i < Dim; ++i)
                    {
                        latfacets_struct(i,3) = latfacets_struct(i,0) + vert_per_base;
                        latfacets_struct(i,4) = latfacets_struct(i,1) + vert_per_base;
                        latfacets_struct(i,5) = latfacets_struct(i,2) + vert_per_base;
                        latfacets_struct(i,7) = latfacets_struct(i,6) + vert_per_base;
                    }

                    //cout << "latfacets_struct (vertex indices)" << endl;
                    //printArr2DInt (&latfacets_struct);

                    //(*)const int * base_face = el->GetFaceVertices(i); // not implemented in MFEM for Tetrahedron ?!

                    int * tetrahedrons;
                    int shift = 0;

                    // 3.8 loop over lateral facets, creating Delaunay triangulations
                    for ( int latfacind = 0; latfacind < Dim; ++latfacind)
                    //for ( int latfacind = 0; latfacind < 1; ++latfacind)
                    {
                        //cout << "latface = " << latfacind << endl;
                        for ( int vert = 0; vert < vert_per_latface ; ++vert )
                        {
                            //cout << "vert index = " << latfacets_struct(latfacind,vert) << endl;
                            for ( int coord = 0; coord < Dim; ++coord)
                            {
                                //cout << "index righthandside " << latfacets_struct(latfacind,vert)* Dim + coord << endl;
                                vert_latface[vert*Dim + coord] =  \
                                        elvert_coordprism[latfacets_struct(latfacind,vert)
                                        * Dim + coord];
                            }

                        }

                        //cout << "Welcome the vertices of a lateral face" << endl;
                        //printDouble2D(vert_latface, vert_per_latface, Dim);

                        // creating from 3Dprism in 4D a true 3D prism in 3D by change of coordinates
                        // = computing input argument vert_3Dlatface for qhull wrapper
                        // we know that the first three coordinated of a lateral face is actually
                        // a triangle, so we set the first vertex to be the origin,
                        // the first-to-second edge to be one of the axis
                        if ( Dim == 4 )
                        {
                            double x1, x2, x3, y1, y2, y3;
                            double dist12, dist13, dist23;
                            double area, h, p;

                            dist12 = dist(vert_latface, vert_latface+Dim , Dim);
                            dist13 = dist(vert_latface, vert_latface+2*Dim , Dim);
                            dist23 = dist(vert_latface+Dim, vert_latface+2*Dim , Dim);

                            p = 0.5 * (dist12 + dist13 + dist23);
                            area = sqrt (p * (p - dist12) * (p - dist13) * (p - dist23));
                            h = 2.0 * area / dist12;

                            x1 = 0.0;
                            y1 = 0.0;
                            x2 = dist12;
                            y2 = 0.0;
                            if ( dist13 - h < 0.0 )
                                if ( fabs(dist13 - h) > 1.0e-10)
                                {
                                    std::cout << "Error: strange: dist13 = " << dist13 << " h = "
                                              << h << std::endl;
                                    return NULL;
                                }
                                else
                                    x3 = 0.0;
                            else
                                x3 = sqrt(dist13*dist13 - h*h);
                            y3 = h;


                            // the time coordinate remains the same
                            for ( int vert = 0; vert < vert_per_latface ; ++vert )
                                vert_3Dlatface[vert*Dim3D + 2] = vert_latface[vert*Dim + 3];


                            // first & fourth vertex
                            vert_3Dlatface[0*Dim3D + 0] = x1;
                            vert_3Dlatface[0*Dim3D + 1] = y1;
                            vert_3Dlatface[3*Dim3D + 0] = x1;
                            vert_3Dlatface[3*Dim3D + 1] = y1;

                            // second & fifth vertex
                            vert_3Dlatface[1*Dim3D + 0] = x2;
                            vert_3Dlatface[1*Dim3D + 1] = y2;
                            vert_3Dlatface[4*Dim3D + 0] = x2;
                            vert_3Dlatface[4*Dim3D + 1] = y2;

                            // third & sixth vertex
                            vert_3Dlatface[2*Dim3D + 0] = x3;
                            vert_3Dlatface[2*Dim3D + 1] = y3;
                            vert_3Dlatface[5*Dim3D + 0] = x3;
                            vert_3Dlatface[5*Dim3D + 1] = y3;
                        } //end of creating a true 3d prism

                        //cout << "Welcome the vertices of a lateral face in 3D" << endl;
                        //printDouble2D(vert_3Dlatface, vert_per_latface, Dim3D);

                        tetrahedrons = tetrahedronsAll + shift;
#ifdef WITH_QHULL
                        qhT qh_qh;                /* Qhull's data structure.  First argument of most calls */
                        qhT *qh= &qh_qh;
                        int curlong, totlong;     /* memory remaining after qh_memfreeshort */

                        double volumetol = 1.0e-8;
                        qhull_wrapper(tetrahedrons, qh, vert_3Dlatface, Dim3D,  \
                                        volumetol, qhull_flags);

                        qh_freeqhull(qh, !qh_ALL);
                        qh_memfreeshort(qh, &curlong, &totlong);
                        if (curlong || totlong)  /* could also check previous runs */
                          cerr<< "qhull internal warning (user_eg, #3): did not free " << totlong \
                          << "bytes of long memory (" << curlong << " pieces)" << endl;
#else
                        cout << "Cannot work without WITH_QHULL defined" << endl;
#endif
                        // convert local 3D prism (lateral face) vertex indices back to the 4D prism \
                        indices and adding boundary elements from tetrahedrins for lateral faces \
                        of the 4d prism ...
                        for ( int tetraind = 0; tetraind < Dim3D; ++tetraind)
                        {
                            //cout << "tetraind = " << tetraind << endl;

                            for ( int vert = 0; vert < Dim; ++vert)
                            {
                                int temp = tetrahedrons[tetraind*Dim + vert];
                                tetrahedrons[tetraind*Dim + vert] = latfacets_struct(latfacind, temp);
                            }

                            /*
                            cout << "tetrahedron: " << tetrahedrons[tetraind*Dim + 0] << " " << \
                                    tetrahedrons[tetraind*Dim + 1] << " " << \
                                    tetrahedrons[tetraind*Dim + 2] << " " << \
                                    tetrahedrons[tetraind*Dim + 3] << "\n";

                            cout << "elverts prism " << endl;
                            elverts_prism.Print();
                            */


                            int temptetra[4];
                            if ( bnd_method == 0 )
                            {
                                if ( facebdrmarker[latfacind] == 1 )
                                {
                                    //cout << "lateral facet " << latfacind << \
                                            " is at the boundary: adding bnd element" << endl;

                                    temptetra[0] = elverts_prism[tetrahedrons[tetraind*Dim + 0]];
                                    temptetra[1] = elverts_prism[tetrahedrons[tetraind*Dim + 1]];
                                    temptetra[2] = elverts_prism[tetrahedrons[tetraind*Dim + 2]];
                                    temptetra[3] = elverts_prism[tetrahedrons[tetraind*Dim + 3]];
                                    //elverts_prism[i]

                                    // wrong because indices in tetrahedrons are local to 4d prism
                                    //NewBdrTri = new Tetrahedron(tetrahedrons + tetraind*Dim);

                                    intermesh->bdrelements[bdrelcount*Dim + 0] = temptetra[0];
                                    intermesh->bdrelements[bdrelcount*Dim + 1] = temptetra[1];
                                    intermesh->bdrelements[bdrelcount*Dim + 2] = temptetra[2];
                                    intermesh->bdrelements[bdrelcount*Dim + 3] = temptetra[3];
                                    intermesh->bdrattrs[bdrelcount] = 2;
                                    bdrelcount++;
                                }
                            }
                            else // bnd_method = 1
                            {
                                set<int> latface3d_set;
                                for ( int i = 0; i < Dim3D; ++i)
                                    latface3d_set.insert(elverts_prism[latfacets_struct(latfacind,i)] % NumOf3DVertices);

                                // checking whether the face is at the boundary of 3d mesh
                                if ( LocalBdrs.find(latface3d_set) != LocalBdrs.end())
                                {
                                    // converting local indices to global indices and adding the new boundary element
                                    temptetra[0] = elverts_prism[tetrahedrons[tetraind*Dim + 0]];
                                    temptetra[1] = elverts_prism[tetrahedrons[tetraind*Dim + 1]];
                                    temptetra[2] = elverts_prism[tetrahedrons[tetraind*Dim + 2]];
                                    temptetra[3] = elverts_prism[tetrahedrons[tetraind*Dim + 3]];

                                    intermesh->bdrelements[bdrelcount*Dim + 0] = temptetra[0];
                                    intermesh->bdrelements[bdrelcount*Dim + 1] = temptetra[1];
                                    intermesh->bdrelements[bdrelcount*Dim + 2] = temptetra[2];
                                    intermesh->bdrelements[bdrelcount*Dim + 3] = temptetra[3];
                                    intermesh->bdrattrs[bdrelcount] = 2;
                                    bdrelcount++;
                                }
                            }



                         } //end of loop over tetrahedrons for a given lateral face

                        shift += Dim3D * (Dim3D + 1);

                        //return;
                    } // end of loop over lateral faces

                    /*
                    std::cout << "Now final tetrahedrons are:" << endl;
                    for ( int k = 0; k < Dim; ++k )
                        for (int i = 0; i < Dim3D; ++i )
                        {
                            //std::cout << "Tetrahedron " << i << ": ";
                            std::cout << "vert indices: " << endl;
                            for ( int j = 0; j < Dim3D  +1; ++j )
                            {
                                std::cout << tetrahedronsAll[k*Dim3D*(Dim3D+1) +
                                        i*(Dim3D + 1) + j] << " ";
                            }
                            std::cout << endl;
                        }
                    */

                    // 3.9 adding the new edges from created tetrahedrons into the vert_to_vert
                    for ( int k = 0; k < Dim; ++k )
                        for (int i = 0; i < Dim3D; ++i )
                        {
                            int vert0 = tetrahedronsAll[k*Dim3D*(Dim3D+1) + i*(Dim3D + 1) + 0];
                            int vert1 = tetrahedronsAll[k*Dim3D*(Dim3D+1) + i*(Dim3D + 1) + 1];
                            int vert2 = tetrahedronsAll[k*Dim3D*(Dim3D+1) + i*(Dim3D + 1) + 2];
                            int vert3 = tetrahedronsAll[k*Dim3D*(Dim3D+1) + i*(Dim3D + 1) + 3];

                            vert_to_vert_prism(vert0, vert1) = 1;
                            vert_to_vert_prism(vert1, vert0) = 1;

                            vert_to_vert_prism(vert0, vert2) = 1;
                            vert_to_vert_prism(vert2, vert0) = 1;

                            vert_to_vert_prism(vert0, vert3) = 1;
                            vert_to_vert_prism(vert3, vert0) = 1;

                            vert_to_vert_prism(vert1, vert2) = 1;
                            vert_to_vert_prism(vert2, vert1) = 1;

                            vert_to_vert_prism(vert1, vert3) = 1;
                            vert_to_vert_prism(vert3, vert1) = 1;

                            vert_to_vert_prism(vert2, vert3) = 1;
                            vert_to_vert_prism(vert3, vert2) = 1;
                        }

                    //cout << "vert_to_vert after delaunay" << endl;
                    //printArr2DInt (&vert_to_vert_prism);

                    int count_penta = 0;

                    // 3.10 creating finally 4d pentatopes:
                    // take a tetrahedron related to a lateral face, find out which of the rest \
                    2 vertices of the 4d prism (one is not) is connected to all vertices of \
                    tetrahedron, and get a pentatope from tetrahedron + this vertex \
                    If pentatope is new, add it to the final structure \
                    To make checking for new pentatopes easy, reoder the pentatope indices \
                    in the default std order

                    for ( int tetraind = 0; tetraind < Dim3D * Dim; ++tetraind)
                    {
                        // creating a pentatop temp
                        int latface_ind = tetraind / Dim3D;
                        for ( int vert = 0; vert < Dim; vert++ )
                            temp[vert] = tetrahedronsAll[tetraind * Dim + vert];

                        //cout << "tetrahedron" << endl;
                        //printInt2D(temp,1,4); // tetrahedron

                        bool isconnected = true;
                        for ( int vert = 0; vert < 4; ++vert)
                            if (vert_to_vert_prism(temp[vert], latfacets_struct(latface_ind,6)) == 0)
                                isconnected = false;

                        if ( isconnected == true)
                            temp[4] = latfacets_struct(latface_ind,6);
                        else
                        {
                            bool isconnectedCheck = true;
                            for ( int vert = 0; vert < 4; ++vert)
                                if (vert_to_vert_prism(temp[vert], latfacets_struct(latface_ind,7)) == 0)
                                    isconnectedCheck = false;
                            if (isconnectedCheck == 0)
                            {
                                cout << "Error: Both vertices are disconnected" << endl;
                                cout << "tetraind = " << tetraind << ", checking for " <<
                                             latfacets_struct(latface_ind,6) << " and " <<
                                             latfacets_struct(latface_ind,7) << endl;
                                return NULL;
                            }
                            else
                                temp[4] = latfacets_struct(latface_ind,7);
                        }

                        //printInt2D(temp,1,5);

                        // replacing local vertex indices w.r.t to 4d prism to global!
                        temp[0] = elverts_prism[temp[0]];
                        temp[1] = elverts_prism[temp[1]];
                        temp[2] = elverts_prism[temp[2]];
                        temp[3] = elverts_prism[temp[3]];
                        temp[4] = elverts_prism[temp[4]];

                        // sorting the vertex indices
                        std::vector<int> buff (temp, temp+5);
                        std::sort (buff.begin(), buff.begin()+5);

                        // looking whether the current pentatop is new
                        bool isnew = true;
                        for ( int i = 0; i < count_penta; ++i )
                        {
                            std::vector<int> pentatop (pentatops+i*(Dim+1), pentatops+(i+1)*(Dim+1));

                            if ( pentatop == buff )
                                isnew = false;
                        }

                        if ( isnew == true )
                        {
                            for ( int i = 0; i < Dim + 1; ++i )
                                pentatops[count_penta*(Dim+1) + i] = buff[i];
                            //cout << "found a new pentatop from tetraind = " << tetraind << endl;
                            //cout << "now we have " << count_penta << " pentatops" << endl;
                            //printInt2D(pentatops + count_penta*(Dim+1), 1, Dim + 1);

                            ++count_penta;
                        }
                        //cout << "element " << elind << endl;
                        //printInt2D(pentatops, count_penta, Dim + 1);
                    }

                    //cout<< count_penta << " pentatops created" << endl;
                    if ( count_penta != Dim )
                        cout << "Error: Wrong number of pentatops constructed: got " << count_penta \
                             << ", needed " << Dim << endl;
                    //printInt2D(pentatops, count_penta, Dim + 1);

                }


            } //end of if local_method = 0 or 1
            else // local_method == 2
            {
                for ( int count_penta = 0; count_penta < Dim; ++count_penta)
                {
                    for ( int i = 0; i < Dim + 1; ++i )
                    {
                        pentatops[count_penta*(Dim+1) + i] = count_penta + i;
                    }

                }
                //cout << "Welcome created pentatops" << endl;
                //printInt2D(pentatops, Dim, Dim + 1);
            }


            // adding boundary elements
            // careful, for now pentatopes give the vertex indices local to the 4D prism above a 3d element!
            if (local_method == 0 || local_method == 2)
            {
                if (local_method == 2)
                    for ( int i = 0; i < vert_per_base; ++i)
                        antireordering[ordering[i]] = i;

                if (local_nbdrfaces > 0) //if there is at least one 3d element face at the boundary for a given base element
                {
                    for ( int pentaind = 0; pentaind < Dim; ++pentaind)
                    {
                        //cout << "pentaind = " << pentaind << endl;
                        //printInt2D(pentatops + pentaind*(Dim+1), 1, 5);

                        for ( int faceind = 0; faceind < Dim + 1; ++faceind)
                        {
                            //cout << "faceind = " << faceind << endl;
                            set<int> tetraproj;

                            // creating local vertex indices for a pentatope face \
                            //and projecting the face onto the 3d base
                            if (bnd_method == 0)
                            {
                                int cnt = 0;
                                for ( int j = 0; j < Dim + 1; ++j)
                                {
                                    if ( j != faceind )
                                    {
                                        temptetra[cnt] = pentatops[pentaind*(Dim + 1) + j];
                                        if (temptetra[cnt] > vert_per_base - 1)
                                            tetraproj.insert(temptetra[cnt] - vert_per_base);
                                        else
                                            tetraproj.insert(temptetra[cnt]);
                                        cnt++;
                                    }
                                }

                                //cout << "temptetra in local indices" << endl;
                                //printInt2D(temptetra,1,4);

                                //cout << "temptetra in global indices" << endl;
                            }
                            else // for bnd_method = 1 we create temptetra and projection in global indices
                            {
                                int cnt = 0;
                                for ( int j = 0; j < Dim + 1; ++j)
                                {
                                    if ( j != faceind )
                                    {
                                        temptetra[cnt] = elverts_prism[pentatops[pentaind*(Dim + 1) + j]];
                                        tetraproj.insert(temptetra[cnt] % NumOf3DVertices );
                                        cnt++;
                                    }
                                }

                                //cout << "temptetra in global indices" << endl;
                                //printInt2D(temptetra,1,4);
                            }

                            /*
                            cout << "tetraproj:" << endl;
                            for ( int temp : tetraproj)
                                cout << temp << " ";
                            cout << endl;
                            */


                            // checking whether the projection is at the boundary of 3d mesh
                            if ( LocalBdrs.find(tetraproj) != LocalBdrs.end())
                            {
                                //cout << "Found a new boundary element" << endl;
                                //cout << "With local indices: " << endl;
                                //printInt2D(temptetra, 1, Dim);

                                // converting local indices to global indices and adding the new boundary element
                                if (bnd_method == 0)
                                {
                                    temptetra[0] = elverts_prism[temptetra[0]];
                                    temptetra[1] = elverts_prism[temptetra[1]];
                                    temptetra[2] = elverts_prism[temptetra[2]];
                                    temptetra[3] = elverts_prism[temptetra[3]];
                                }

                                //cout << "With global indices: " << endl;
                                //printInt2D(temptetra, 1, Dim);

                                intermesh->bdrelements[bdrelcount*Dim + 0] = temptetra[0];
                                intermesh->bdrelements[bdrelcount*Dim + 1] = temptetra[1];
                                intermesh->bdrelements[bdrelcount*Dim + 2] = temptetra[2];
                                intermesh->bdrelements[bdrelcount*Dim + 3] = temptetra[3];
                                intermesh->bdrattrs[bdrelcount] = 2;
                                bdrelcount++;

                            }


                        } // end of loop over pentatope faces
                    } // end of loop over pentatopes
                } // end of if local_nbdrfaces > 0

                // converting local indices in pentatopes to the global indices
                // replacing local vertex indices w.r.t to 4d prism to global!
                for ( int pentaind = 0; pentaind < Dim; ++pentaind)
                {
                    for ( int j = 0; j < Dim + 1; j++)
                    {
                        pentatops[pentaind*(Dim + 1) + j] = elverts_prism[pentatops[pentaind*(Dim + 1) + j]];
                    }
                }

            } //end of if local_method = 0 or 2

            // By this point, for the given 3d element:
            // 4d elemnts = pentatops are constructed, but stored in local array
            // boundary elements are constructed which correspond to the elements in the 4D prism


            // 3.11 adding the constructed pentatops to the 4d mesh
            for ( int penta_ind = 0; penta_ind < Dim; ++penta_ind)
            {
                intermesh->elements[elcount*(Dim + 1) + 0] = pentatops[penta_ind*(Dim+1) + 0];
                intermesh->elements[elcount*(Dim + 1) + 1] = pentatops[penta_ind*(Dim+1) + 1];
                intermesh->elements[elcount*(Dim + 1) + 2] = pentatops[penta_ind*(Dim+1) + 2];
                intermesh->elements[elcount*(Dim + 1) + 3] = pentatops[penta_ind*(Dim+1) + 3];
                intermesh->elements[elcount*(Dim + 1) + 4] = pentatops[penta_ind*(Dim+1) + 4];
                intermesh->elattrs[elcount] = 1;
                elcount++;

            }

            //printArr2DInt (&vert_to_vert_prism);


        } // end of loop over base elements
    } // end of loop over time slabs

    delete [] pentatops;
    delete [] elvert_coordprism;

    if (local_method == 1)
    {
        delete [] vert_latface;
        delete [] vert_3Dlatface;
        delete [] tetrahedronsAll;
    }
    if (local_method == 0 || local_method == 1)
        delete [] qhull_flags;

    return intermesh;
}

// serial space-time mesh constructor
Mesh::Mesh ( Mesh& meshbase, double tau, int Nsteps, int bnd_method, int local_method)
//void MeshSpaceTimeCylinder ( Mesh& mesh3d, Mesh& mesh4d, double tau, int Nsteps, int bnd_method, int local_method)
{
    MeshSpaceTimeCylinder_onlyArrays ( meshbase, tau, Nsteps, bnd_method, local_method);

    int refine = 1;
    CreateInternalMeshStructure(refine);

    return;
}

// from a given base mesh (3d tetrahedrons or 2D triangles) produces a space-time mesh
// for a space-time cylinder with the given base, Nsteps * tau height in time
// enumeration of space-time vertices: time slab after time slab
// boundary attributes: 1 for t=0, 2 for lateral boundaries, 3 for t = tau*Nsteps
void Mesh::MeshSpaceTimeCylinder_onlyArrays ( Mesh& meshbase, double tau, int Nsteps,
                                              int bnd_method, int local_method)
{
    int DimBase = meshbase.Dimension(), NumOfBaseElements = meshbase.GetNE(),
            NumOfBaseBdrElements = meshbase.GetNBE(),
            NumOfBaseVertices = meshbase.GetNV();
    int NumOfSTElements, NumOfSTBdrElements, NumOfSTVertices;

    if ( DimBase != 3 && DimBase != 2 )
    {
        cerr << "Wrong dimension in MeshSpaceTimeCylinder(): " << DimBase << endl << flush;
        return;
    }

    if ( DimBase == 2 )
    {
        if ( local_method == 1 )
        {
            cerr << "This local method = " << local_method << " is not supported by case "
                                                     "dim = " << DimBase << endl << flush;
            return;
        }
    }

    int Dim = DimBase + 1;

    // for each base element and each time slab a space-time prism with base mesh element as a base
    // is decomposed into (Dim) simplices (tetrahedrons in 3d and pentatops in 4d);
    NumOfSTElements = NumOfBaseElements * Dim * Nsteps;
    NumOfSTVertices = NumOfBaseVertices * (Nsteps + 1); // no additional vertices inbetween time slabs so far
    // lateral 4d bdr faces (one for each 3d bdr face) + lower + upper bases
    // of the space-time cylinder
    NumOfSTBdrElements = NumOfBaseBdrElements * DimBase * Nsteps + 2 * NumOfBaseElements;

    // assuming that the 3D mesh contains elements of the same type = tetrahedrons
    int vert_per_base = meshbase.GetElement(0)->GetNVertices();
    int vert_per_prism = 2 * vert_per_base;
    int vert_per_latface = DimBase * 2;

    InitMesh(Dim,Dim,NumOfSTVertices,NumOfSTElements,NumOfSTBdrElements);

    Element * el;

    int * simplexes;
    if (local_method == 1 || local_method == 2)
    {
        simplexes = new int[Dim * (Dim + 1)]; // array for storing vertex indices for constructed simplices
    }
    else // local_method = 0
    {
        int nsliver = 5; //why 5? how many slivers can b created by qhull? maybe 0 if we don't joggle inside qhull but perturb the coordinates before?
        simplexes = new int[(Dim + nsliver) * (Dim + 1)]; // array for storing vertex indices for constructed simplices + probably sliver pentatopes
    }

    // stores indices of space-time element face vertices produced by qhull for all lateral faces
    // Used in local_method = 1 only.
    int * facesimplicesAll;
    if (local_method == 1 )
        facesimplicesAll = new int[DimBase * (DimBase + 1) * Dim ];

    Array<int> elverts_base;
    Array<int> elverts_prism;

    // temporary array for vertex indices of a pentatope face (used in local_method = 0 and 2)
    int tempface[Dim];
    int temp[Dim+1]; //temp array for simplex vertices in local_method = 1;

    // three arrays below are used only in local_method = 1
    Array2D<int> vert_to_vert_prism; // for a 4D prism
    // row ~ lateral face of the 4d prism
    // first 6 columns - indices of vertices belonging to the lateral face,
    // last 2 columns - indices of the rest 2 vertices of the prism
    Array2D<int> latfacets_struct;
    // coordinates of vertices of a lateral face of 4D prism
    double * vert_latface;
    // coordinates of vertices of a 3D base (triangle) of a lateral face of 4D prism
    double * vert_3Dlatface;
    if (local_method == 1)
    {
        vert_latface =  new double[Dim * vert_per_latface];
        vert_3Dlatface = new double[DimBase * vert_per_latface];
        latfacets_struct.SetSize(Dim, vert_per_prism);
        vert_to_vert_prism.SetSize(vert_per_prism, vert_per_prism);
    }

    // coordinates of vertices of the space-time prism
    double * elvert_coordprism = new double[Dim * vert_per_prism];

    char * qhull_flags;
    if (local_method == 0 || local_method == 1)
    {
        qhull_flags = new char[250];
        sprintf(qhull_flags, "qhull d Qbb");
    }

    int simplex_count = 0;
    Element * NewEl;
    Element * NewBdrEl;

    double tempvert[Dim];

    if (local_method < 0 && local_method > 2)
    {
        cout << "Local method = " << local_method << " is not supported" << endl << flush;
        return;
    }

    if ( bnd_method != 0 && bnd_method != 1)
    {
        cout << "Illegal value of bnd_method = " << bnd_method << " (must be 0 or 1)"
             << endl << flush;
        return;
    }

    Vector vert_coord3d(DimBase * meshbase.GetNV());
    meshbase.GetVertices(vert_coord3d);
    //printDouble2D(vert_coord3d, 10, Dim3D);

    // adding all space-time vertices to the mesh
    for ( int tslab = 0; tslab <= Nsteps; ++tslab)
    {
        // adding the vertices from the slab to the output space-time mesh
        for ( int vert = 0; vert < NumOfBaseVertices; ++vert)
        {
            for ( int j = 0; j < DimBase; ++j)
            {
                tempvert[j] = vert_coord3d[vert + j * NumOfBaseVertices];
                tempvert[Dim-1] = tau * tslab;
            }
            AddVertex(tempvert);
        }
    }

    int almostjogglers[Dim];
    //int permutation[Dim];
    //vector<double*> lcoords(Dim);
    vector<vector<double>> lcoordsNew(Dim);

    // for each (of Dim) base mesh element faces stores 1 if it is at the boundary and 0 else
    int facebdrmarker[Dim];
    // std::set of the base mesh boundary elements. Using set allows one to perform a search
    // with O(log N_elem) operations
    std::set< std::vector<int> > BdrTriSet;
    Element * bdrel;

    Array<int> face_bndflags;
    if (bnd_method == 1)
    {
        if (Dim == 4)
            face_bndflags.SetSize(meshbase.GetNFaces());
        if (Dim == 3)
            face_bndflags.SetSize(meshbase.GetNEdges());
    }

    Table * localel_to_face;
    Array<int> localbe_to_face;

    // if = 0, a search algorithm is used for defining whether faces of a given base mesh element
    // are at the boundary.
    // if = 1, instead an array face_bndflags is used, which stores 0 and 1 depending on
    // whether the face is at the boundary, + el_to_face table which is usually already
    // generated for the base mesh
    //int bnd_method = 1;

    if (bnd_method == 0)
    {
        // putting base mesh boundary elements from base mesh structure to the set BdrTriSet
        for ( int boundelem = 0; boundelem < NumOfBaseBdrElements; ++boundelem)
        {
            //cout << "boundelem No. " << boundelem << endl;
            bdrel = meshbase.GetBdrElement(boundelem);
            int * bdrverts = bdrel->GetVertices();

            std::vector<int> buff (bdrverts, bdrverts+DimBase);
            std::sort (buff.begin(), buff.begin()+DimBase);

            BdrTriSet.insert(buff);
        }
        /*
        for (vector<int> temp : BdrTriSet)
        {
            cout << temp[0] << " " <<  temp[1] << " " << temp[2] << endl;
        }
        cout<<endl;
        */
    }
    else // bnd_method = 1
    {
        if (Dim == 4)
        {
            if (meshbase.el_to_face == NULL)
            {
                cout << "Have to built el_to_face" << endl;
                meshbase.GetElementToFaceTable(0);
            }
            localel_to_face = meshbase.el_to_face;
            localbe_to_face.MakeRef(meshbase.be_to_face);
        }
        if (Dim == 3)
        {
            if (meshbase.el_to_edge == NULL)
            {
                cout << "Have to built el_to_edge" << endl;
                meshbase.GetElementToEdgeTable(*(meshbase.el_to_edge), meshbase.be_to_edge);
            }
            localel_to_face = meshbase.el_to_edge;
            localbe_to_face.MakeRef(meshbase.be_to_edge);
        }

        //cout << "Special print" << endl;
        //cout << mesh3d.el_to_face(elind, facelind);
        //cout << "be_to_face" << endl;
        //mesh3d.be_to_face.Print();
        //localbe_to_face.Print();


        //cout << "nfaces = " << meshbase.GetNFaces();
        //cout << "nbe = " << meshbase.GetNBE() << endl;
        //cout << "boundary.size = " << mesh3d.boundary.Size() << endl;

        face_bndflags = -1;
        for ( int i = 0; i < meshbase.GetNBE(); ++i )
            //face_bndflags[meshbase.be_to_face[i]] = 1;
            face_bndflags[localbe_to_face[i]] = 1;

        //cout << "face_bndflags" << endl;
        //face_bndflags.Print();
    }

    int ordering[vert_per_base];
    int antireordering[vert_per_base]; // used if bnd_method = 0 and local_method = 2
    Array<int> tempelverts(vert_per_base);

    // main loop creates space-time elements over all time slabs over all base mesh elements
    // loop over base mesh elements
    for ( int elind = 0; elind < NumOfBaseElements; elind++ )
    //for ( int elind = 0; elind < 1; ++elind )
    {
        //cout << "element " << elind << endl;

        el = meshbase.GetElement(elind);

        // 1. getting indices of base mesh element vertices and their coordinates in the prism
        el->GetVertices(elverts_base);

        //for ( int k = 0; k < elverts_base.Size(); ++k )
          //  cout << "elverts[" << k << "] = " << elverts_base[k] << endl;

        // for local_method 2 we need to reorder the local vertices of the prism to preserve
        // the the order in some global sense  = lexicographical order of the vertex coordinates
        if (local_method == 2)
        {
            // using elvert_coordprism as a temporary buffer for changing elverts_base
            for ( int vert = 0; vert < vert_per_base; ++vert)
            {
                for ( int j = 0; j < DimBase; ++j)
                {
                    elvert_coordprism[Dim * vert + j] =
                            vert_coord3d[elverts_base[vert] + j * NumOfBaseVertices];
                }
            }

            /*
             * old one
            for (int vert = 0; vert < Dim; ++vert)
                lcoords[vert] = elvert_coordprism + Dim * vert;

            sortingPermutation(DimBase, lcoords, ordering);

            cout << "ordering 1:" << endl;
            for ( int i = 0; i < vert_per_base; ++i)
                cout << ordering[i] << " ";
            cout << endl;
            */

            for (int vert = 0; vert < Dim; ++vert)
                lcoordsNew[vert].assign(elvert_coordprism + Dim * vert,
                                        elvert_coordprism + Dim * vert + DimBase);

            sortingPermutationNew(lcoordsNew, ordering);

            //cout << "ordering 2:" << endl;
            //for ( int i = 0; i < vert_per_base; ++i)
                //cout << ordering[i] << " ";
            //cout << endl;

            // UGLY: Fix it
            for ( int i = 0; i < vert_per_base; ++i)
                tempelverts[i] = elverts_base[ordering[i]];

            for ( int i = 0; i < vert_per_base; ++i)
                elverts_base[i] = tempelverts[i];
        }

        // 2. understanding which of the base mesh element faces (triangles) are at the boundary
        int local_nbdrfaces = 0;
        set<set<int>> LocalBdrs;
        if (bnd_method == 0) // in this case one looks in the set of base mesh boundary elements
        {
            vector<int> face(DimBase);
            for (int i = 0; i < Dim; ++i )
            {
                // should be consistent with lateral faces ordering in latfacet structure
                // if used with local_method = 1

                for ( int j = 0; j < DimBase; ++j)
                    face[j] = elverts_base[(i+j)%Dim];

                sort(face.begin(), face.begin()+DimBase);
                //cout << face[0] << " " <<  face[1] << " " << face[2] << endl;

                if (BdrTriSet.find(face) != BdrTriSet.end() )
                {
                    local_nbdrfaces++;
                    facebdrmarker[i] = 1;
                    set<int> face_as_set;

                    for ( int j = 0; j < DimBase; ++j)
                        face_as_set.insert((i+j)%Dim);

                    LocalBdrs.insert(face_as_set);
                }
                else
                    facebdrmarker[i] = 0;
            }

        } //end of if bnd_method == 0
        else // in this case one uses el_to_face and face_bndflags to check whether mesh base
             //face is at the boundary
        {
            int * faceinds = localel_to_face->GetRow(elind);
            Array<int> temp(DimBase);
            for ( int facelind = 0; facelind < Dim; ++facelind)
            {
                int faceind = faceinds[facelind];
                if (face_bndflags[faceind] == 1)
                {
                    meshbase.GetFaceVertices(faceind, temp);

                    set<int> face_as_set;
                    for ( int vert = 0; vert < DimBase; ++vert )
                        face_as_set.insert(temp[vert]);

                    LocalBdrs.insert(face_as_set);

                    local_nbdrfaces++;
                }

            } // end of loop over element faces

        }

        //cout << "Welcome the facebdrmarker" << endl;
        //printInt2D(facebdrmarker, 1, Dim);

        /*
        cout << "Welcome the LocalBdrs" << endl;
        for ( set<int> tempset: LocalBdrs )
        {
            cout << "element of LocalBdrs for el = " << elind << endl;
            for (int ind: tempset)
                cout << ind << " ";
            cout << endl;
        }
        */

        // 3. loop over all space-time slabs above a given mesh base element
        for ( int tslab = 0; tslab < Nsteps; ++tslab)
        {
            //cout << "tslab " << tslab << endl;

            //3.1 getting vertex indices for the space-time prism
            elverts_prism.SetSize(vert_per_prism);

            for ( int i = 0; i < vert_per_base; ++i)
            {
                elverts_prism[i] = elverts_base[i] + tslab * NumOfBaseVertices;
                elverts_prism[i + vert_per_base] = elverts_base[i] +
                        (tslab + 1) * NumOfBaseVertices;
            }
            //cout << "New elverts_prism" << endl;
            //elverts_prism.Print(cout, 10);
            //return;


            // 3.2 for the first time slab we add the base mesh elements in the lower base
            // to the space-time bdr elements
            if ( tslab == 0 )
            {
                //cout << "zero slab: adding boundary element:" << endl;
                if (Dim == 3)
                    NewBdrEl = new Triangle(elverts_prism);
                if (Dim == 4)
                    NewBdrEl = new Tetrahedron(elverts_prism);
                NewBdrEl->SetAttribute(1);
                AddBdrElement(NewBdrEl);
            }
            // 3.3 for the last time slab we add the base mesh elements in the upper base
            // to the space-time bdr elements
            if ( tslab == Nsteps - 1 )
            {
                //cout << "last slab: adding boundary element:" << endl;
                if (Dim == 3)
                    NewBdrEl = new Triangle(elverts_prism + vert_per_base);
                if (Dim == 4)
                    NewBdrEl = new Tetrahedron(elverts_prism + vert_per_base);
                NewBdrEl->SetAttribute(3);
                AddBdrElement(NewBdrEl);
            }

            if (local_method == 0 || local_method == 1)
            {
                // 3.4 setting vertex coordinates for space-time prism, lower base
                for ( int vert = 0; vert < vert_per_base; ++vert)
                {
                    for ( int j = 0; j < DimBase; ++j)
                        elvert_coordprism[Dim * vert + j] =
                                vert_coord3d[elverts_base[vert] + j * NumOfBaseVertices];
                    elvert_coordprism[Dim * vert + Dim-1] = tslab * tau;
                }

                //cout << "Welcome the vertex coordinates for the 4d prism base " << endl;
                //printDouble2D(elvert_coordprism, vert_per_base, Dim);

                /*
                 * old
                for (int vert = 0; vert < Dim; ++vert)
                    lcoords[vert] = elvert_coordprism + Dim * vert;


                //cout << "vector double * lcoords:" << endl;
                //for ( int i = 0; i < Dim; ++i)
                    //cout << "lcoords[" << i << "]: " << lcoords[i][0] << " " << lcoords[i][1] << " " << lcoords[i][2] << endl;

                sortingPermutation(DimBase, lcoords, permutation);
                */

                // here we compute the permutation "ordering" which preserves the geometric order of vertices
                // which is based on their coordinates comparison and compute jogglers for qhull
                // from the "ordering"

                for (int vert = 0; vert < Dim; ++vert)
                    lcoordsNew[vert].assign(elvert_coordprism + Dim * vert,
                                            elvert_coordprism + Dim * vert + DimBase);

                sortingPermutationNew(lcoordsNew, ordering);


                //cout << "Welcome the permutation:" << endl;
                //cout << permutation[0] << " " << permutation[1] << " " << permutation[2] << " " << permutation[3] << endl;

                int joggle_coeff = 0;
                for ( int i = 0; i < Dim; ++i)
                    almostjogglers[ordering[i]] = joggle_coeff++;


                // 3.5 setting vertex coordinates for space-time prism, upper layer
                // Joggling is required for getting unique Delaunay tesselation and should be
                // the same for vertices shared between different elements or at least produce
                // the same Delaunay triangulation in the shared faces.
                // So here it is not exactly the same, but if joggle(vertex A) > joggle(vertex B)
                // on one element, then the same inequality will hold in another element which also has
                // vertices A and B.
                double joggle;
                for ( int vert = 0; vert < vert_per_base; ++vert)
                {
                    for ( int j = 0; j < DimBase; ++j)
                        elvert_coordprism[Dim * (vert_per_base + vert) + j] =
                                elvert_coordprism[Dim * vert + j];
                    joggle = 1.0e-2 * (almostjogglers[vert]);
                    //joggle = 1.0e-2 * elverts_prism[i + vert_per_base] * 1.0 / NumOf4DVertices;
                    //double joggle = 1.0e-2 * i;
                    elvert_coordprism[Dim * (vert_per_base + vert) + Dim-1] =
                            (tslab + 1) * tau * ( 1.0 + joggle );
                }

                //cout << "Welcome the vertex coordinates for the 4d prism" << endl;
                //printDouble2D(elvert_coordprism, 2 * vert_per_base, Dim);

                // 3.6 - 3.10: constructing new space-time simplices and space-time boundary elements
                if (local_method == 0)
                {
#ifdef WITH_QHULL
                    qhT qh_qh;                /* Qhull's data structure.  First argument of most calls */
                    qhT *qh= &qh_qh;
                    int curlong, totlong;     /* memory remaining after qh_memfreeshort */

                    double volumetol = 1.0e-8;
                    qhull_wrapper(simplexes, qh, elvert_coordprism, Dim, volumetol, qhull_flags);

                    qh_freeqhull(qh, !qh_ALL);
                    qh_memfreeshort(qh, &curlong, &totlong);
                    if (curlong || totlong)  /* could also check previous runs */
                    {
                      fprintf(stderr, "qhull internal warning (user_eg, #3): did not free %d bytes"
                                      " of long memory (%d pieces)\n", totlong, curlong);
                    }
#else
                    cout << "Cannot work without WITH_QHULL defined" << endl;
#endif
                } // end of if local_method = 0

                if (local_method == 1) // works only in 4D case. Just historically the first implementation
                {
                    setzero(&vert_to_vert_prism);

                    // 3.6 creating vert_to_vert for the prism before Delaunay
                    // (adding 4d prism edges)
                    for ( int i = 0; i < el->GetNEdges(); i++)
                    {
                        const int * edge = el->GetEdgeVertices(i);
                        //cout << "edge: " << edge[0] << " " << edge[1] << std::endl;
                        vert_to_vert_prism(edge[0], edge[1]) = 1;
                        vert_to_vert_prism(edge[1], edge[0]) = 1;
                        vert_to_vert_prism(edge[0] + vert_per_base, edge[1] + vert_per_base) = 1;
                        vert_to_vert_prism(edge[1] + vert_per_base, edge[0] + vert_per_base) = 1;
                    }

                    for ( int i = 0; i < vert_per_base; i++)
                    {
                        vert_to_vert_prism(i, i) = 1;
                        vert_to_vert_prism(i + vert_per_base, i + vert_per_base) = 1;
                        vert_to_vert_prism(i, i + vert_per_base) = 1;
                        vert_to_vert_prism(i + vert_per_base, i) = 1;
                    }

                    //cout << "vert_to_vert before delaunay" << endl;
                    //printArr2DInt (&vert_to_vert_prism);
                    //cout << endl;

                    // 3.7 creating latfacet structure (brute force), for 4D tetrahedron case
                    // indices are local w.r.t to the 4d prism!!!
                    latfacets_struct(0,0) = 0;
                    latfacets_struct(0,1) = 1;
                    latfacets_struct(0,2) = 2;
                    latfacets_struct(0,6) = 3;

                    latfacets_struct(1,0) = 1;
                    latfacets_struct(1,1) = 2;
                    latfacets_struct(1,2) = 3;
                    latfacets_struct(1,6) = 0;

                    latfacets_struct(2,0) = 2;
                    latfacets_struct(2,1) = 3;
                    latfacets_struct(2,2) = 0;
                    latfacets_struct(2,6) = 1;

                    latfacets_struct(3,0) = 3;
                    latfacets_struct(3,1) = 0;
                    latfacets_struct(3,2) = 1;
                    latfacets_struct(3,6) = 2;

                    for ( int i = 0; i < Dim; ++i)
                    {
                        latfacets_struct(i,3) = latfacets_struct(i,0) + vert_per_base;
                        latfacets_struct(i,4) = latfacets_struct(i,1) + vert_per_base;
                        latfacets_struct(i,5) = latfacets_struct(i,2) + vert_per_base;
                        latfacets_struct(i,7) = latfacets_struct(i,6) + vert_per_base;
                    }

                    //cout << "latfacets_struct (vertex indices)" << endl;
                    //printArr2DInt (&latfacets_struct);

                    //(*)const int * base_face = el->GetFaceVertices(i); // not implemented in MFEM for Tetrahedron ?!

                    int * tetrahedrons;
                    int shift = 0;


                    // 3.8 loop over lateral facets, creating Delaunay triangulations
                    for ( int latfacind = 0; latfacind < Dim; ++latfacind)
                    {
                        //cout << "latface = " << latfacind << endl;
                        for ( int vert = 0; vert < vert_per_latface ; ++vert )
                        {
                            //cout << "vert index = " << latfacets_struct(latfacind,vert) << endl;
                            for ( int coord = 0; coord < Dim; ++coord)
                            {
                                vert_latface[vert*Dim + coord] =
                                  elvert_coordprism[latfacets_struct(latfacind,vert) * Dim + coord];
                            }

                        }

                        //cout << "Welcome the vertices of a lateral face" << endl;
                        //printDouble2D(vert_latface, vert_per_latface, Dim);

                        // creating from 3Dprism in 4D a true 3D prism in 3D by change of
                        // coordinates = computing input argument vert_3Dlatface for qhull wrapper
                        // we know that the first three coordinated of a lateral face is actually
                        // a triangle, so we set the first vertex to be the origin,
                        // the first-to-second edge to be one of the axis
                        if ( Dim == 4 )
                        {
                            double x1, x2, x3, y1, y2, y3;
                            double dist12, dist13, dist23;
                            double area, h, p;

                            dist12 = dist(vert_latface, vert_latface+Dim , Dim);
                            dist13 = dist(vert_latface, vert_latface+2*Dim , Dim);
                            dist23 = dist(vert_latface+Dim, vert_latface+2*Dim , Dim);

                            p = 0.5 * (dist12 + dist13 + dist23);
                            area = sqrt (p * (p - dist12) * (p - dist13) * (p - dist23));
                            h = 2.0 * area / dist12;

                            x1 = 0.0;
                            y1 = 0.0;
                            x2 = dist12;
                            y2 = 0.0;
                            if ( dist13 - h < 0.0 )
                                if ( fabs(dist13 - h) > 1.0e-10)
                                {
                                    std::cout << "strange: dist13 = " << dist13 << " h = "
                                              << h << std::endl;
                                    return;
                                }
                                else
                                    x3 = 0.0;
                            else
                                x3 = sqrt(dist13*dist13 - h*h);
                            y3 = h;


                            // the time coordinate remains the same
                            for ( int vert = 0; vert < vert_per_latface ; ++vert )
                                vert_3Dlatface[vert*DimBase + 2] = vert_latface[vert*Dim + 3];

                            // first & fourth vertex
                            vert_3Dlatface[0*DimBase + 0] = x1;
                            vert_3Dlatface[0*DimBase + 1] = y1;
                            vert_3Dlatface[3*DimBase + 0] = x1;
                            vert_3Dlatface[3*DimBase + 1] = y1;

                            // second & fifth vertex
                            vert_3Dlatface[1*DimBase + 0] = x2;
                            vert_3Dlatface[1*DimBase + 1] = y2;
                            vert_3Dlatface[4*DimBase + 0] = x2;
                            vert_3Dlatface[4*DimBase + 1] = y2;

                            // third & sixth vertex
                            vert_3Dlatface[2*DimBase + 0] = x3;
                            vert_3Dlatface[2*DimBase + 1] = y3;
                            vert_3Dlatface[5*DimBase + 0] = x3;
                            vert_3Dlatface[5*DimBase + 1] = y3;
                        } //end of creating a true 3d prism

                        //cout << "Welcome the vertices of a lateral face in 3D" << endl;
                        //printDouble2D(vert_3Dlatface, vert_per_latface, Dim3D);

                        tetrahedrons = facesimplicesAll + shift;
#ifdef WITH_QHULL
                        qhT qh_qh;                /* Qhull's data structure.  First argument of most calls */
                        qhT *qh= &qh_qh;
                        int curlong, totlong;     /* memory remaining after qh_memfreeshort */

                        double volumetol = MYZEROTOL;
                        qhull_wrapper(tetrahedrons, qh, vert_3Dlatface, DimBase, volumetol, qhull_flags);

                        qh_freeqhull(qh, !qh_ALL);
                        qh_memfreeshort(qh, &curlong, &totlong);
                        if (curlong || totlong)  /* could also check previous runs */
                          cerr<< "qhull internal warning (user_eg, #3): did not free " << totlong
                          << "bytes of long memory (" << curlong << " pieces)" << endl;
#else
                        cout << "Cannot work without WITH_QHULL defined" << endl;
#endif
                        // convert local 3D prism (lateral face) vertex indices back to the
                        // 4D prism indices and adding boundary elements from tetrahedrins
                        // for lateral faces of the 4d prism ...
                        for ( int tetraind = 0; tetraind < DimBase; ++tetraind)
                        {
                            //cout << "tetraind = " << tetraind << endl;

                            for ( int vert = 0; vert < Dim; ++vert)
                            {
                                int temp = tetrahedrons[tetraind*Dim + vert];
                                tetrahedrons[tetraind*Dim + vert] = latfacets_struct(latfacind, temp);
                            }

                            if ( bnd_method == 0 )
                            {
                                if ( facebdrmarker[latfacind] == 1 )
                                {
                                    //cout << "lateral facet " << latfacind << " is at the boundary: adding bnd element" << endl;

                                    tempface[0] = elverts_prism[tetrahedrons[tetraind*Dim + 0]];
                                    tempface[1] = elverts_prism[tetrahedrons[tetraind*Dim + 1]];
                                    tempface[2] = elverts_prism[tetrahedrons[tetraind*Dim + 2]];
                                    tempface[3] = elverts_prism[tetrahedrons[tetraind*Dim + 3]];

                                    // wrong because indices in tetrahedrons are local to 4d prism
                                    //NewBdrTri = new Tetrahedron(tetrahedrons + tetraind*Dim);

                                    NewBdrEl = new Tetrahedron(tempface);
                                    NewBdrEl->SetAttribute(2);
                                    AddBdrElement(NewBdrEl);

                                }
                            }
                            else // bnd_method = 1
                            {
                                set<int> latface3d_set;
                                for ( int i = 0; i < DimBase; ++i)
                                    latface3d_set.insert(elverts_prism[latfacets_struct(latfacind,i)] % NumOfBaseVertices);

                                // checking whether a face is at the boundary of 3d mesh
                                if ( LocalBdrs.find(latface3d_set) != LocalBdrs.end())
                                {
                                    // converting local indices to global indices and
                                    // adding the new boundary element
                                    tempface[0] = elverts_prism[tetrahedrons[tetraind*Dim + 0]];
                                    tempface[1] = elverts_prism[tetrahedrons[tetraind*Dim + 1]];
                                    tempface[2] = elverts_prism[tetrahedrons[tetraind*Dim + 2]];
                                    tempface[3] = elverts_prism[tetrahedrons[tetraind*Dim + 3]];

                                    NewBdrEl = new Tetrahedron(tempface);
                                    NewBdrEl->SetAttribute(2);
                                    AddBdrElement(NewBdrEl);
                                }
                            }



                         } //end of loop over tetrahedrons for a given lateral face

                        shift += DimBase * (DimBase + 1);

                        //return;
                    } // end of loop over lateral faces

                    // 3.9 adding the new edges from created tetrahedrons into the vert_to_vert
                    for ( int k = 0; k < Dim; ++k )
                        for (int i = 0; i < DimBase; ++i )
                        {
                            int vert0 = facesimplicesAll[k*DimBase*(DimBase+1) +
                                    i*(DimBase + 1) + 0];
                            int vert1 = facesimplicesAll[k*DimBase*(DimBase+1) +
                                    i*(DimBase + 1) + 1];
                            int vert2 = facesimplicesAll[k*DimBase*(DimBase+1) +
                                    i*(DimBase + 1) + 2];
                            int vert3 = facesimplicesAll[k*DimBase*(DimBase+1) +
                                    i*(DimBase + 1) + 3];

                            vert_to_vert_prism(vert0, vert1) = 1;
                            vert_to_vert_prism(vert1, vert0) = 1;

                            vert_to_vert_prism(vert0, vert2) = 1;
                            vert_to_vert_prism(vert2, vert0) = 1;

                            vert_to_vert_prism(vert0, vert3) = 1;
                            vert_to_vert_prism(vert3, vert0) = 1;

                            vert_to_vert_prism(vert1, vert2) = 1;
                            vert_to_vert_prism(vert2, vert1) = 1;

                            vert_to_vert_prism(vert1, vert3) = 1;
                            vert_to_vert_prism(vert3, vert1) = 1;

                            vert_to_vert_prism(vert2, vert3) = 1;
                            vert_to_vert_prism(vert3, vert2) = 1;
                        }

                    //cout << "vert_to_vert after delaunay" << endl;
                    //printArr2DInt (&vert_to_vert_prism);

                    int count_penta = 0;

                    // 3.10 creating finally 4d pentatopes:
                    // take a tetrahedron related to a lateral face, find out which of the rest
                    // 2 vertices of the 4d prism (one is not) is connected to all vertices of
                    // tetrahedron, and get a pentatope from tetrahedron + this vertex
                    // If pentatope is new, add it to the final structure
                    // To make checking for new pentatopes easy, reoder the pentatope indices
                    // in the default std order

                    for ( int tetraind = 0; tetraind < DimBase * Dim; ++tetraind)
                    {
                        // creating a pentatop temp
                        int latface_ind = tetraind / DimBase;
                        for ( int vert = 0; vert < Dim; vert++ )
                            temp[vert] = facesimplicesAll[tetraind * Dim + vert];

                        //cout << "tetrahedron" << endl;
                        //printInt2D(temp,1,4); // tetrahedron

                        bool isconnected = true;
                        for ( int vert = 0; vert < 4; ++vert)
                            if (vert_to_vert_prism(temp[vert],
                                                   latfacets_struct(latface_ind,6)) == 0)
                                isconnected = false;

                        if ( isconnected == true)
                            temp[4] = latfacets_struct(latface_ind,6);
                        else
                        {
                            bool isconnectedCheck = true;
                            for ( int vert = 0; vert < 4; ++vert)
                                if (vert_to_vert_prism(temp[vert],
                                                       latfacets_struct(latface_ind,7)) == 0)
                                    isconnectedCheck = false;
                            if (isconnectedCheck == 0)
                            {
                                cout << "Error: Both vertices are disconnected" << endl;
                                cout << "tetraind = " << tetraind << ", checking for " <<
                                             latfacets_struct(latface_ind,6) << " and " <<
                                             latfacets_struct(latface_ind,7) << endl;
                                return;
                            }
                            else
                                temp[4] = latfacets_struct(latface_ind,7);
                        }

                        //printInt2D(temp,1,5);

                        // replacing local vertex indices w.r.t to 4d prism to global!
                        temp[0] = elverts_prism[temp[0]];
                        temp[1] = elverts_prism[temp[1]];
                        temp[2] = elverts_prism[temp[2]];
                        temp[3] = elverts_prism[temp[3]];
                        temp[4] = elverts_prism[temp[4]];

                        // sorting the vertex indices
                        std::vector<int> buff (temp, temp+5);
                        std::sort (buff.begin(), buff.begin()+5);

                        // looking whether the current pentatop is new
                        bool isnew = true;
                        for ( int i = 0; i < count_penta; ++i )
                        {
                            std::vector<int> pentatop (simplexes+i*(Dim+1), simplexes+(i+1)*(Dim+1));

                            if ( pentatop == buff )
                                isnew = false;
                        }

                        if ( isnew == true )
                        {
                            for ( int i = 0; i < Dim + 1; ++i )
                                simplexes[count_penta*(Dim+1) + i] = buff[i];
                            //cout << "found a new pentatop from tetraind = " << tetraind << endl;
                            //cout << "now we have " << count_penta << " pentatops" << endl;
                            //printInt2D(pentatops + count_penta*(Dim+1), 1, Dim + 1);

                            ++count_penta;
                        }
                        //cout << "element " << elind << endl;
                        //printInt2D(pentatops, count_penta, Dim + 1);
                    }

                    //cout<< count_penta << " pentatops created" << endl;
                    if ( count_penta != Dim )
                        cout << "Error: Wrong number of simplexes constructed: got " <<
                                count_penta << ", needed " << Dim << endl << flush;
                    //printInt2D(pentatops, count_penta, Dim + 1);

                }

            } //end of if local_method = 0 or 1
            else // local_method == 2
            {
                // The simplest way to generate space-time simplices.
                // But requires to reorder the vertices at first, as done before.
                for ( int count_simplices = 0; count_simplices < Dim; ++count_simplices)
                {
                    for ( int i = 0; i < Dim + 1; ++i )
                    {
                        simplexes[count_simplices*(Dim+1) + i] = count_simplices + i;
                    }

                }
                //cout << "Welcome created pentatops" << endl;
                //printInt2D(pentatops, Dim, Dim + 1);
            }


            // adding boundary elements in local method =  0 or 2
            if (local_method == 0 || local_method == 2)
            {
                if (local_method == 2)
                    for ( int i = 0; i < vert_per_base; ++i)
                        antireordering[ordering[i]] = i;

                if (local_nbdrfaces > 0) //if there is at least one base mesh element face at
                                         // the boundary for a given base element
                {
                    for ( int simplexind = 0; simplexind < Dim; ++simplexind)
                    {
                        //cout << "simplexind = " << simplexind << endl;
                        //printInt2D(pentatops + pentaind*(Dim+1), 1, 5);

                        for ( int faceind = 0; faceind < Dim + 1; ++faceind)
                        {
                            //cout << "faceind = " << faceind << endl;
                            set<int> faceproj;

                            // creating local vertex indices for a simplex face \
                            //and projecting the face onto the 3d base
                            if (bnd_method == 0)
                            {
                                int cnt = 0;
                                for ( int j = 0; j < Dim + 1; ++j)
                                {
                                    if ( j != faceind )
                                    {
                                        tempface[cnt] = simplexes[simplexind*(Dim + 1) + j];
                                        if (tempface[cnt] > vert_per_base - 1)
                                            faceproj.insert(tempface[cnt] - vert_per_base);
                                        else
                                            faceproj.insert(tempface[cnt]);
                                        cnt++;
                                    }
                                }

                                //cout << "tempface in local indices" << endl;
                                //printInt2D(tempface,1,4);
                            }
                            else // for bnd_method = 1 we create tempface and projection
                                 // in global indices
                            {
                                int cnt = 0;
                                for ( int j = 0; j < Dim + 1; ++j)
                                {
                                    if ( j != faceind )
                                    {
                                        tempface[cnt] =
                                                elverts_prism[simplexes[simplexind*(Dim + 1) + j]];
                                        faceproj.insert(tempface[cnt] % NumOfBaseVertices );
                                        cnt++;
                                    }
                                }

                                //cout << "tempface in global indices" << endl;
                                //printInt2D(tempface,1,4);
                            }

                            /*
                            cout << "faceproj:" << endl;
                            for ( int temp : faceproj)
                                cout << temp << " ";
                            cout << endl;
                            */

                            // checking whether the projection is at the boundary of base mesh
                            // using the local-to-element LocalBdrs set which has at most Dim elements
                            if ( LocalBdrs.find(faceproj) != LocalBdrs.end())
                            {
                                //cout << "Found a new boundary element" << endl;
                                //cout << "With local indices: " << endl;
                                //printInt2D(tempface, 1, Dim);

                                // converting local indices to global indices and
                                // adding the new boundary element
                                if (bnd_method == 0)
                                {
                                    for ( int facevert = 0; facevert < Dim; ++facevert )
                                        tempface[facevert] = elverts_prism[tempface[facevert]];
                                }

                                //cout << "With global indices: " << endl;
                                //printInt2D(tempface, 1, Dim);

                                if (Dim == 3)
                                    NewBdrEl = new Triangle(tempface);
                                if (Dim == 4)
                                    NewBdrEl = new Tetrahedron(tempface);
                                NewBdrEl->SetAttribute(2);
                                AddBdrElement(NewBdrEl);
                            }


                        } // end of loop over space-time simplex faces
                    } // end of loop over space-time simplices
                } // end of if local_nbdrfaces > 0

                // By this point, for the given base mesh element:
                // space-time elements are constructed, but stored in local array
                // boundary elements are constructed which correspond to the elements in the space-time prism
                // converting local-to-prism indices in simplices to the global indices
                for ( int simplexind = 0; simplexind < Dim; ++simplexind)
                {
                    for ( int j = 0; j < Dim + 1; j++)
                    {
                        simplexes[simplexind*(Dim + 1) + j] =
                                elverts_prism[simplexes[simplexind*(Dim + 1) + j]];
                    }
                }

            } //end of if local_method = 0 or 2

            // printInt2D(pentatops, Dim, Dim + 1);


            // 3.11 adding the constructed space-time simplices to the output mesh
            for ( int simplex_ind = 0; simplex_ind < Dim; ++simplex_ind)
            {
                if (Dim == 3)
                    NewEl = new Tetrahedron(simplexes + simplex_ind*(Dim+1));
                if (Dim == 4)
                    NewEl = new Pentatope(simplexes + simplex_ind*(Dim+1));
                NewEl->SetAttribute(1);
                AddElement(NewEl);
                ++simplex_count;
            }

            //printArr2DInt (&vert_to_vert_prism);

        } // end of loop over time slabs
    } // end of loop over base elements

    if ( NumOfSTElements != GetNE() )
        std::cout << "Error: Wrong number of elements generated: " << GetNE() << " instead of " <<
                        NumOfSTElements << std::endl;
    if ( NumOfSTVertices != GetNV() )
        std::cout << "Error: Wrong number of vertices generated: " << GetNV() << " instead of " <<
                        NumOfSTVertices << std::endl;
    if ( NumOfSTBdrElements!= GetNBE() )
        std::cout << "Error: Wrong number of bdr elements generated: " << GetNBE() << " instead of " <<
                        NumOfSTBdrElements << std::endl;

    delete [] simplexes;
    delete [] elvert_coordprism;

    if (local_method == 1)
    {
        delete [] vert_latface;
        delete [] vert_3Dlatface;
        delete [] facesimplicesAll;
    }
    if (local_method == 0 || local_method == 1)
        delete [] qhull_flags;

    return;
}


// parallel version 1 : creating serial space-time mesh from parallel base mesh in parallel
// from a given base mesh produces a space-time mesh for a cylinder
// with the given base and Nsteps * tau height in time
// enumeration of space-time vertices: time slab after time slab
// boundary attributes: 1 for t=0, 2 for lateral boundaries, 3 for t = tau*Nsteps
//void ParMesh3DtoMesh4D (MPI_Comm comm, ParMesh& mesh3d,
//                                       Mesh& mesh4d, double tau, int Nsteps, int bnd_method, int local_method)
Mesh::Mesh (MPI_Comm comm, ParMesh& mesh3d, double tau, int Nsteps,
            int bnd_method, int local_method)
{
    int num_procs, myid;

    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    int dim = 4;
    int dim3 = 3;
    int nvert_per_elem = dim + 1; // PENTATOPE or TETRAHEDRON cases only
    int nvert_per_bdrelem = dim; // PENTATOPE or TETRAHEDRON cases only

    // *************************************************************************
    // step 1 of 3: take the local base mesh for the proc and create a local space-time mesh
    // part as IntermediateMesh
    // *************************************************************************

    // 1.1: create gverts = array with global vertex numbers
    // can be avoided but then a lot of calls to pspace3d->GetGlobalTDofNumber will happen
    int * gvertinds = new int[mesh3d.GetNV()];

    FiniteElementCollection * h1_coll = new H1_FECollection(1, dim3);

    ParFiniteElementSpace * pspace3d = new ParFiniteElementSpace(&mesh3d, h1_coll);

    for ( int lvert = 0; lvert < mesh3d.GetNV(); ++lvert )
        gvertinds[lvert] = pspace3d->GetGlobalTDofNumber(lvert);

    // 1.2: creating local parts of space-time mesh as IntemediateMesh structure

    /*
    for ( int proc = 0; proc < num_procs; ++proc )
    {
        if ( proc == myid )
        {
            cout << "I am " << proc << ", myid = " << myid << endl << flush;
            cout << "Creating local parts of 4d mesh" << endl << flush;
            //cout << "Now it is in local indices" << endl;
        }
        MPI_Barrier(comm);
    }
    */

    IntermediateMesh * local_intermesh = mesh3d.MeshSpaceTimeCylinder_toInterMesh( tau, Nsteps, bnd_method, local_method);

    int nv3d_global = pspace3d->GlobalTrueVSize(); // global number of vertices in the 3d mesh

    // 1.3 writing the global vertex numbers inside the local IntermediateMesh(4d)
    int lvert4d;
    int tslab;
    int onslab_lindex; // = lvert3d for the projection of 4d on 3d base
    for ( int lvert4d = 0; lvert4d < local_intermesh->nv; ++lvert4d )
    {
        tslab = lvert4d / mesh3d.GetNV();
        onslab_lindex = lvert4d - tslab * mesh3d.GetNV();

        local_intermesh->vert_gindices[lvert4d] = tslab * nv3d_global + gvertinds[onslab_lindex];
        //local_intermesh->vert_gindices[lvert4d] = tslab * nv3d_global + pspace3d->GetGlobalTDofNumber(onslab_lindex);
    }

    InterMeshPrint (local_intermesh, myid, "local_intermesh");
    MPI_Barrier(comm);

    // 1.4 replacing local vertex indices by global indices from parFEspace
    // converting local to global vertex indices in elements
    for (int elind = 0; elind < local_intermesh->ne; ++elind)
    {
        //cout << "elind = " << elind << endl;
        for ( int j = 0; j < nvert_per_elem; ++j )
        {
            lvert4d = local_intermesh->elements[elind * nvert_per_elem + j];
            tslab = lvert4d / mesh3d.GetNV();
            onslab_lindex = lvert4d - tslab * mesh3d.GetNV();

            //local_intermesh->elements[elind * nvert_per_elem + j] =
                    //tslab * nv3d_global + pspace3d->GetGlobalTDofNumber(onslab_lindex);
            local_intermesh->elements[elind * nvert_per_elem + j] =
                    tslab * nv3d_global + gvertinds[onslab_lindex];
        }
    }

    // converting local to global vertex indices in boundary elements
    for (int bdrelind = 0; bdrelind < local_intermesh->nbe; ++bdrelind)
    {
        //cout << "bdrelind = " << bdrelind << endl;
        for ( int j = 0; j < nvert_per_bdrelem; ++j )
        {
            lvert4d = local_intermesh->bdrelements[bdrelind * nvert_per_bdrelem + j];
            tslab = lvert4d / mesh3d.GetNV();
            onslab_lindex = lvert4d - tslab * mesh3d.GetNV();

            //local_intermesh->bdrelements[bdrelind * nvert_per_bdrelem + j] =
                    //tslab * nv3d_global + pspace3d->GetGlobalTDofNumber(onslab_lindex);
            local_intermesh->bdrelements[bdrelind * nvert_per_bdrelem + j] =
                    tslab * nv3d_global + gvertinds[onslab_lindex];

            //cout << "lindex3d converted to gindex3d = " << pspace3d->GetGlobalTDofNumber(onslab_lindex) << endl;
        }
    }

    delete h1_coll;
    delete pspace3d;

    //InterMeshPrint (local_intermesh, myid, "local_intermesh_newer");
    //MPI_Barrier(comm);

    // *************************************************************************
    // step 2 of 3: exchange the local mesh 4d parts and exchange them
    // *************************************************************************

    // 2.1: exchanging information about local sizes between processors
    // in order to set up mpi exchange parameters and allocate the future 4d mesh;

    // nvdg_global = sum of local number of vertices (without thinking that
    // some vertices are shared between processors)
    int nvdg_global, nv_global, ne_global, nbe_global;

    int *recvcounts_el = new int[num_procs];
    MPI_Allgather( &(local_intermesh->ne), 1, MPI_INT, recvcounts_el, 1, MPI_INT, comm);

    int *rdispls_el = new int[num_procs];
    rdispls_el[0] = 0;
    for ( int i = 0; i < num_procs - 1; ++i)
        rdispls_el[i + 1] = rdispls_el[i] + recvcounts_el[i];

    ne_global = rdispls_el[num_procs - 1] + recvcounts_el[num_procs - 1];

    int *recvcounts_be = new int[num_procs];

    MPI_Allgather( &(local_intermesh->nbe), 1, MPI_INT, recvcounts_be, 1, MPI_INT, comm);

    int *rdispls_be = new int[num_procs];

    rdispls_be[0] = 0;
    for ( int i = 0; i < num_procs - 1; ++i)
        rdispls_be[i + 1] = rdispls_be[i] + recvcounts_be[i];

    nbe_global = rdispls_be[num_procs - 1] + recvcounts_be[num_procs - 1];

    int *recvcounts_v = new int[num_procs];
    MPI_Allgather( &(local_intermesh->nv), 1, MPI_INT, recvcounts_v, 1, MPI_INT, comm);

    int *rdispls_v = new int[num_procs];
    rdispls_v[0] = 0;
    for ( int i = 0; i < num_procs - 1; ++i)
        rdispls_v[i + 1] = rdispls_v[i] + recvcounts_v[i];

    nvdg_global = rdispls_v[num_procs - 1] + recvcounts_v[num_procs - 1];
    nv_global = nv3d_global * (Nsteps + 1);

    MPI_Barrier(comm);

    IntermediateMesh * intermesh_4d = new IntermediateMesh;
    IntermeshInit( intermesh_4d, dim, nvdg_global, ne_global, nbe_global, 1);

    // 2.2: exchanging attributes, elements and vertices between processes using allgatherv

    // exchanging element attributes
    MPI_Allgatherv( local_intermesh->elattrs, local_intermesh->ne, MPI_INT,
                    intermesh_4d->elattrs, recvcounts_el, rdispls_el, MPI_INT, comm);

    // exchanging bdr element attributes
    MPI_Allgatherv( local_intermesh->bdrattrs, local_intermesh->nbe, MPI_INT,
                    intermesh_4d->bdrattrs, recvcounts_be, rdispls_be, MPI_INT, comm);

    // exchanging elements, changing recvcounts_el!!!
    for ( int i = 0; i < num_procs; ++i)
    {
        recvcounts_el[i] *= nvert_per_elem;
        rdispls_el[i] *= nvert_per_elem;
    }

    MPI_Allgatherv( local_intermesh->elements, (local_intermesh->ne)*nvert_per_elem, MPI_INT,
                    intermesh_4d->elements, recvcounts_el, rdispls_el, MPI_INT, comm);

    // exchanging bdrelements, changing recvcounts_be!!!

    for ( int i = 0; i < num_procs; ++i)
    {
        recvcounts_be[i] *= nvert_per_bdrelem;
        rdispls_be[i] *= nvert_per_bdrelem;
    }

    MPI_Allgatherv( local_intermesh->bdrelements, (local_intermesh->nbe)*nvert_per_bdrelem,
              MPI_INT, intermesh_4d->bdrelements, recvcounts_be, rdispls_be, MPI_INT, comm);

    // exchanging global vertex indices
    MPI_Allgatherv( local_intermesh->vert_gindices, local_intermesh->nv, MPI_INT,
                    intermesh_4d->vert_gindices, recvcounts_v, rdispls_v, MPI_INT, comm);

    // exchanging vertices : At the moment dg-type of procedure = without considering
    // presence of shared vertices
    for ( int i = 0; i < num_procs; ++i)
    {
        recvcounts_v[i] *= dim;
        rdispls_v[i] *= dim;
    }

    MPI_Allgatherv( local_intermesh->vertices, (local_intermesh->nv)*dim, MPI_DOUBLE,
                    intermesh_4d->vertices, recvcounts_v, rdispls_v, MPI_DOUBLE, comm);

    IntermeshDelete(local_intermesh);

    // *************************************************************************
    // step 3 of 3: creating serial 4d mesh for each process
    // *************************************************************************

    InitMesh(dim,dim, nv_global, ne_global, nbe_global);

    //InterMeshPrint (intermesh_4d, myid, "intermesh4d");
    //MPI_Barrier(comm);

    // 3.1: creating the correct vertex array where each vertex is met only once
    // 3.1.1: cleaning up the vertices which are at the moment with multiple entries for
    // shared vertices

    int gindex;
    std::map<int, double*> vertices_unique; // map structure for storing only unique vertices

    // loop over all (with multiple entries) vertices, unique are added to the map object
    double * tempvert_map;
    for ( int i = 0; i < nvdg_global; ++i )
    {
        tempvert_map = new double[dim];
        for ( int j = 0; j < dim; j++ )
            tempvert_map[j] = intermesh_4d->vertices[i * dim + j];
        gindex = intermesh_4d->vert_gindices[i];
        vertices_unique[gindex] = tempvert_map;
    }

    // counting the final number of vertices. after that count_vert should be equal to nv_global
    int count_vert = 0;
    for(auto const& ent : vertices_unique)
    {
        count_vert ++;
    }

    if ( count_vert != nv_global && myid == 0 )
    {
        cout << "Wrong number of vertices! Smth is probably wrong" << endl << flush;
    }

    // 3.1.2: creating the vertices array with taking care of shared vertices
    // using the map vertices_unique

    // now actual intermesh_4d->vertices is: right unique vertices + some vertices which
    // are still alive after mpi transfer.
    // so we reuse the memory already allocated for vertices array with multiple entries.

    //delete [] intermesh_4d->vertices;
    intermesh_4d->nv = count_vert;
    //intermesh_4d->vertices = new double[count_vert * dim];

    int tmp = 0;
    for(auto const& ent : vertices_unique)
    {
        for ( int j = 0; j < dim; j++)
            intermesh_4d->vertices[tmp*dim + j] = ent.second[j];

        if ( tmp != ent.first )
            cout << "ERROR" << endl;
        tmp++;
    }

    vertices_unique.clear();

    //InterMeshPrint (intermesh_4d, myid, "intermesh4d_reduced");
    //MPI_Barrier(comm);

    // 3.2: loading created intermesh_4d into a mfem mesh object (copying the memory: FIX IT may be)
    BaseGeom = Geometry::PENTATOPE;
    LoadMeshfromArrays( intermesh_4d->nv, intermesh_4d->vertices,
                  intermesh_4d->ne, intermesh_4d->elements, intermesh_4d->elattrs,
                  intermesh_4d->nbe, intermesh_4d->bdrelements, intermesh_4d->bdrattrs, dim );

    // 3.3 create the internal structure for mesh after el-s,, bdel-s and vertices have been loaded
    int refine = 1;
    CreateInternalMeshStructure(refine);

    IntermeshDelete(intermesh_4d);

    MPI_Barrier(comm);


    return;
}

// Reads the elements, vertices and boundary from the input IntermediatMesh.
// It is like Load() in MFEM but for IntermediateMesh instead of an input stream.
// No internal mesh structures are initialized inside.
void Mesh::LoadMeshfromArrays( int nv, double * vertices, int ne, int * elements, int * elattrs,
                           int nbe, int * bdrelements, int * bdrattrs, int dim )
{
    if (BaseGeom != Geometry::PENTATOPE && BaseGeom != Geometry::TETRAHEDRON)
    {
        cout << "LoadMeshfromArrays() is implemented only for pentatops and tetrahedrons" << endl;
        return;
    }
    int nvert_per_elem = dim + 1; // PENTATOPE and TETRAHEDRON case only
    int nvert_per_bdrelem = dim; // PENTATOPE and TETRAHEDRON case only

    Element * el;

    for (int j = 0; j < ne; j++)
    {
        if (dim == 4)
            el = new Pentatope(elements + j*nvert_per_elem);
        else // dim == 3
            el = new Tetrahedron(elements + j*nvert_per_elem);
        el->SetAttribute(elattrs[j]);

        AddElement(el);
    }

    for (int j = 0; j < nbe; j++)
    {
        if (dim == 4)
            el = new Tetrahedron(bdrelements + j*nvert_per_bdrelem);
        else // dim == 3
            el = new Triangle(bdrelements + j*nvert_per_bdrelem);
        el->SetAttribute(bdrattrs[j]);
        AddBdrElement(el);
    }

    for (int j = 0; j < nv; j++)
    {
        AddVertex(vertices + j * dim );
    }

    return;
}

// parallel version 2
// from a given base mesh (tetrahedral or triangular) produces a space-time mesh for a cylinder
// with thegiven base and Nsteps * tau height in time
// enumeration of space-time vertices: time slab after time slab
// boundary attributes: 1 for t=0, 2 for lateral boundaries, 3 for t = tau*Nsteps
//void ParMesh3DtoParMesh4D (MPI_Comm comm, ParMesh& mesh3d,
//                     ParMesh& mesh4d, double tau, int Nsteps, int bnd_method, int local_method)
ParMesh::ParMesh (MPI_Comm comm, ParMesh& meshbase, double tau, int Nsteps,
                  int bnd_method, int local_method)
{
    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    int dim = meshbase.Dimension() + 1;

    if (meshbase.Dimension() != 3 && meshbase.Dimension() != 2 && myid == 0)
    {
        cout << "Case meshbase dim = " << meshbase.Dimension() << " is not supported "
                                             "in parmesh constructor" << endl << flush;
        return;
    }

    if ( bnd_method != 0 && bnd_method != 1 && myid == 0)
    {
        cout << "Illegal value of bnd_method = " << bnd_method << " (must be 0 or 1)"
             << endl << flush;
        return;
    }
    if ( (local_method < 0 || local_method > 2) && myid == 0)
    {
        cout << "Illegal value of local_method = " << local_method << " (must be 0,1 "
                                                              "or 2)" << endl << flush;
        return;
    }

    // ****************************************************************************
    // step 1 of 4: creating local space-time part of the mesh from local part of base mesh
    // ****************************************************************************

    /*
    for ( int proc = 0; proc < num_procs; ++proc )
    {
        if ( proc == myid )
        {
            cout << "I am " << proc << ", parmesh myrank = " << mesh3d.MyRank << endl;
            cout << "Creating local part of 4d mesh" << endl;
        }
        cout << flush;
        MPI_Barrier(comm);
    }
    */

    // creating local parts of space-time mesh
    MeshSpaceTimeCylinder_onlyArrays(meshbase, tau, Nsteps, bnd_method, local_method);

    MPI_Barrier(comm);

    // ****************************************************************************
    // step 2 of 4: set additional fields (except the main ones which are
    // shared entities) required for parmesh
    // In particular, set refinement flags in 2D->3D case
    // ****************************************************************************

    MyComm = comm;
    MPI_Comm_size(MyComm, &NRanks);
    MPI_Comm_rank(MyComm, &MyRank);

    gtopo.SetComm(comm);

    int i, j;

    if (dim == 4)
    {
        BaseGeom = Geometry::PENTATOPE;         // PENTATOPE case only
        BaseBdrGeom = Geometry::TETRAHEDRON;    // PENTATOPE case only
    }
    else //dim == 3
    {
        BaseGeom = Geometry::TETRAHEDRON;       // TETRAHEDRON case only
        BaseBdrGeom = Geometry::TRIANGLE;       // TETRAHEDRON case only
    }

    ncmesh = pncmesh = NULL;

    swappedElements.SetSize(GetNE());

    DenseMatrix J(4,4);
    if( dim == 4)
    {
        for ( i = 0; i < GetNE(); ++i )
        {
            if (elements[i]->GetType() == Element::PENTATOPE)
            {
                int *v = elements[i]->GetVertices();
                Sort5(v[0], v[1], v[2], v[3], v[4]);

                GetElementJacobian(i, J);

                if(J.Det() < 0.0)
                {
                    swappedElements[i] = true;
                    Swap(v);
                }else
                {
                    swappedElements[i] = false;
                }
            }
        }
    }

    meshgen = meshbase.MeshGenerator(); // FIX IT: Not sure at all what it is

    attributes.Copy(meshbase.attributes);
    bdr_attributes.Copy(meshbase.bdr_attributes);

    InitTables();

    if (dim > 1)
    {
       el_to_edge = new Table;
       NumOfEdges = GetElementToEdgeTable(*(el_to_edge), be_to_edge);
    }
    else
    {
       NumOfEdges = 0;
    }

    STable3D *faces_tbl_3d = NULL;
    if ( dim == 3 )
        faces_tbl_3d = GetElementToFaceTable(1);


    STable4D *faces_tbl_4d = NULL;
    if ( dim == 4 )
    {
        faces_tbl_4d = GetElementToFaceTable4D(1);
    }

    GenerateFaces();

    NumOfPlanars = 0;
    el_to_planar = NULL;

    STable3D *planar_tbl = NULL;
    if( dim == 4 )
    {
       planar_tbl = GetElementToPlanarTable(1);
       GeneratePlanars();
    }


    if (NumOfBdrElements == 0 && Dim > 2)
    {
       // in 3D, generate boundary elements before we 'MarkForRefinement'
       if(Dim==3) GetElementToFaceTable();
       else if(Dim==4)
       {
           GetElementToFaceTable4D();
       }
       GenerateFaces();
       GenerateBoundaryElements();
    }


    int curved = 0;
    int generate_edges = 1;

    CheckElementOrientation(true);
    if ( dim == 3)
    {
        MarkForRefinement();
    }

    // generate the faces
    if (Dim > 2)
    {
           if(Dim==3) GetElementToFaceTable();
           else if(Dim==4)
           {
               GetElementToFaceTable4D();
           }

           GenerateFaces();

           if(Dim==4)
           {
              ReplaceBoundaryFromFaces();

              GetElementToPlanarTable();
              GeneratePlanars();

 //			 GetElementToQuadTable4D();
 //			 GenerateQuads4D();
           }

       // check and fix boundary element orientation
       if ( !(curved && (meshgen & 1)) )
       {
          CheckBdrElementOrientation();
       }
    }
    else
    {
       NumOfFaces = 0;
    }

    // generate edges if requested
    if (Dim > 1 && generate_edges == 1)
    {
       // el_to_edge may already be allocated (P2 VTK meshes)
       if (!el_to_edge) { el_to_edge = new Table; }
       NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
       if (Dim == 2)
       {
          GenerateFaces(); // 'Faces' in 2D refers to the edges
          if (NumOfBdrElements == 0)
          {
             GenerateBoundaryElements();
          }
          // check and fix boundary element orientation
          if ( !(curved && (meshgen & 1)) )
          {
             CheckBdrElementOrientation();
          }
       }
    }
    else
    {
       NumOfEdges = 0;
    }

    have_face_nbr_data = false;

    // ****************************************************************************
    // step 3 of 4: set parmesh fields for shared entities for mesh4d
    // ****************************************************************************

    ParMeshSpaceTime_createShared( comm, meshbase, Nsteps );

    // some clean up for unneeded tables

    if (dim == 4)
    {
        delete faces_tbl_4d;
        delete planar_tbl;
    }
    else //dim == 3
        delete faces_tbl_3d;

    // ****************************************************************************
    // step 4 of 4: set internal mesh structure (present in both mesh and
    // parmesh classes
    // ****************************************************************************

    int refine = 0;
    CreateInternalMeshStructure(refine);

    return;
}

// Creates ParMesh internal structure (including shared entities)
// after the main arrays (elements, vertices and boundary) are already defined for the
// future space-time mesh. Used only inside the ParMesh constructor.
void ParMesh::ParMeshSpaceTime_createShared( MPI_Comm comm, ParMesh& meshbase, int Nsteps )
{
    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    int DimBase = meshbase.Dimension();
    int Dim = DimBase + 1;
    int vert_per_baseface = meshbase.faces[0]->GetNVertices();
    int nv_base = meshbase.GetNV();

    //cout << "vert_per_face =  " << vert_per_face << endl;
    //cout << "vert_per_elembase = " << vert_per_elembase << endl;

    if (DimBase != 2 && DimBase != 3 && myid == 0)
    {
        cout << "Case dimbase = " << DimBase << " is not supported in createShared()"
             << endl << flush;
        return;
    }

    if (BaseGeom != Geometry::PENTATOPE && BaseGeom != Geometry::TETRAHEDRON)
    {
        if (myid == 0)
            cout << "ParMeshSpaceTime_createShared() is implemented only for "
                    "pentatops and tetrahedrons" << endl << flush;
        return;
    }


    ListOfIntegerSets  groups; // this group list will play the same role as "groups"
    //in the ParMesh constructor from MFEM.
    IntegerSet         group;

    // ****************************************************************************
    // step 0 of 4: looking at local part of the base mesh
    // ****************************************************************************

    // ****************************************************************************
    // step 1 of 4: creating temporary arrays needed for shared entities.
    // The arrays created are related to the space-time mesh structure
    // ****************************************************************************

    // creating sets for each kind of shared entities
    // for each shared entity these array store the number of the corresponding group
    // of processors in LOCAL processors numeration
    Array<int> sface_groupbase;
    Array<int> sedge_groupbase;
    Array<int> svert_groupbase;

    // for each shared entity these array store the index of the entity in the
    // corresponding group of processors ~ position in group
    Array<int> sface_posingroupbase;
    Array<int> sedge_posingroupbase;
    Array<int> svert_posingroupbase;

    // maybe an ugly way to get sface_group from group_sface;
    // actually, just manually transposing.
    if (Dim == 4)
    {
        sface_groupbase.SetSize(meshbase.shared_faces.Size());
        sface_posingroupbase.SetSize(meshbase.shared_faces.Size());
        for ( int row = 0; row < meshbase.group_sface.Size(); ++row )
        {
            int * v = meshbase.group_sface.GetRow(row);
            for (int colno = 0; colno < meshbase.group_sface.RowSize(row); ++colno)
            {
                sface_groupbase[v[colno]] = row;
                sface_posingroupbase[v[colno]] = colno;
            }
        }
    }
    else //Dim == 3
    {
        sface_groupbase.SetSize(meshbase.shared_edges.Size());
        sface_posingroupbase.SetSize(meshbase.shared_edges.Size());
        for ( int row = 0; row < meshbase.group_sedge.Size(); ++row )
        {
            int * v = meshbase.group_sedge.GetRow(row);
            for (int colno = 0; colno < meshbase.group_sedge.RowSize(row); ++colno)
            {
                sface_groupbase[v[colno]] = row;
                sface_posingroupbase[v[colno]] = colno;
            }
        }
    }

    sedge_groupbase.SetSize(meshbase.shared_edges.Size());
    sedge_posingroupbase.SetSize(meshbase.shared_edges.Size());
    for ( int row = 0; row < meshbase.group_sedge.Size(); ++row )
    {
        int * v = meshbase.group_sedge.GetRow(row);
        for (int colno = 0; colno < meshbase.group_sedge.RowSize(row); ++colno)
        {
            sedge_groupbase[v[colno]] = row;
            sedge_posingroupbase[v[colno]] = colno;
        }
    }

    svert_groupbase.SetSize(meshbase.svert_lvert.Size());
    svert_posingroupbase.SetSize(meshbase.svert_lvert.Size());
    for ( int row = 0; row < meshbase.group_svert.Size(); ++row )
    {
        int * v = meshbase.group_svert.GetRow(row);
        for (int colno = 0; colno < meshbase.group_svert.RowSize(row); ++colno)
        {
            svert_groupbase[v[colno]] = row;
            svert_posingroupbase[v[colno]] = colno;
        }
    }

    // creating maps for each kind of base mesh shared entities

    // map structure from shared entities (faces, edges and vertices)
    // to pairs (group number, position inside the group)
    std::map<set<int>, vector<int>> ShfacesBase;
    std::map<set<int>, vector<int>> ShedgesBase;
    // could be a map<int,int>, but somehow the code gets ugly at some place,
    // around "findproj" stuff.
    std::map<set<int>, vector<int>> ShvertsBase;

    for ( int shvertind = 0; shvertind < meshbase.svert_lvert.Size(); ++shvertind)
    {
        set<int> buff (meshbase.svert_lvert + shvertind, meshbase.svert_lvert + shvertind + 1 );
        ShvertsBase[buff] = vector<int>{svert_groupbase[shvertind],
                                            svert_posingroupbase[shvertind]};
    }

    for ( int shedgeind = 0; shedgeind < meshbase.shared_edges.Size(); ++shedgeind)
    {
        Element * shedge = meshbase.shared_edges[shedgeind];

        int * verts = shedge->GetVertices();
        set<int> buff(verts, verts+2);      //edges always have two vertices

        ShedgesBase[buff] = vector<int>{sedge_groupbase[shedgeind],
                                            sedge_posingroupbase[shedgeind]};
    }

    if (Dim == 4)
    {
        for ( int shfaceind = 0; shfaceind < meshbase.shared_faces.Size(); ++shfaceind)
        {
            Element * shface = meshbase.shared_faces[shfaceind];

            int * verts = shface->GetVertices();
            set<int> buff(verts, verts+vert_per_baseface);

            ShfacesBase[buff] = vector<int>{sface_groupbase[shfaceind],
                    sface_posingroupbase[shfaceind]};
        }
    }
    else // Dim == 3
        ShfacesBase = ShedgesBase; //just a convention, that faces in 2D are the same as edges in these temporary structures


    // actually here we need group_proc, which can be obtained from gtopo.GetGroup combined
    // with converting lproc indices to proc indices using lproc_proc from gtopo.

    Array<int> lproc_proc(meshbase.gtopo.GetNumNeighbors());
    for ( int i = 0; i < lproc_proc.Size(); ++i )
        lproc_proc[i] = meshbase.gtopo.GetNeighborRank(i);

    Table group_proc;
    group_proc.MakeI(meshbase.gtopo.NGroups());
    for ( int row = 0; row < meshbase.gtopo.NGroups(); ++row )
    {
        group_proc.AddColumnsInRow(row, meshbase.gtopo.GetGroupSize(row));
    }
    group_proc.MakeJ();

    int rowsize;
    for ( int row = 0; row < meshbase.gtopo.NGroups(); ++row )
    {
        rowsize = meshbase.gtopo.GetGroupSize(row);
        const int * group = meshbase.gtopo.GetGroup(row);
        int group_with_proc[rowsize];

        for ( int col = 0; col < rowsize; ++col )
        {
            group_with_proc[col] = lproc_proc[group[col]];
        }

        group_proc.AddConnections(row, group_with_proc, rowsize);
    }
    group_proc.ShiftUpI();

    /*
    for ( int proc = 0; proc < num_procs; ++proc )
    {
        if ( proc == myid )
        //if ( proc == 1 && proc == myid )
        {
            cout << "I am " << proc << ", parmesh myrank = " << mesh3d.MyRank << endl;
            //cout << "group_lproc3d" << endl;
            //group_lproc.Print(cout,10);

            cout << "lproc_proc3d" << endl;
            lproc_proc.Print();

            //cout << "proc_lproc3d" << endl;
            //proc_lproc.Print();

            cout << " groups_proc3d " << endl;
            group_proc.Print(cout,10);

            cout << flush;
        }
        MPI_Barrier(comm);
    }
    */

    // ****************************************************************************
    // step 2 of 4: creating groups which will be for the space-time mesh exactly the same as for
    // the base mesh (because we are just extending the existing base parts in time)
    // But one should be careful with processors numeration (global vs local)
    // ****************************************************************************

    for ( int row = 0; row < group_proc.Size(); ++row )
    {
        group.Recreate(group_proc.RowSize(row), group_proc.GetRow(row));
        groups.Insert(group);
    }

    // ****************************************************************************
    // step 3 of 4: creating main parmesh structures for shared entities
    // The main idea is:
    // 1. to loop over the local entities,
    // 2. to project them onto the base (2D or 3D)
    // 3. determine whether the projection is inside the shared 3d entities list
    // 4. change correspondignly the shared 4d entities structure
    // Say, a shared 4d planar (a triangle, basically) will be projected
    // either to a shared 3d face or to a shared 3d edge.
    // ****************************************************************************

    int groupind;

    // 3.1 shared faces 3d -> shared faces 4d (or shared edges 2d -> shared faces 3d)
    // 4d case
    // Nsteps * 3 shared tetrahedrons produced by each shared triangle (shared face 3d)
    // which gives for each time slab a 3d-in-4d prism (which is decomposed into 3 tetrahedrons)
    // as lateral face of a 4d space-time prism cell.
    // 3d case
    // Nsteps * 2 shared triangles produced by each shared edge (~shared face 2D) which gives
    // a space-time rectangle (2D in 3D) which is decomposed into 2 triangles

    int face2Dto3D_coeff = DimBase; // 3 for 4d and 2 for 3d
    if ( Dim == 4 )
        shared_faces.SetSize( Nsteps * face2Dto3D_coeff * meshbase.shared_faces.Size());
    else // Dim = 3
        shared_faces.SetSize( Nsteps * face2Dto3D_coeff * meshbase.shared_edges.Size());

    sface_lface.SetSize( shared_faces.Size());

    // alternative way to construct group_sface - from I and J arrays manually
    int * group_sface_I, * group_sface_J;
    group_sface_I = new int[group_proc.Size() + 1];

    group_sface_I[0] = 0;
    if (Dim == 4)
        for ( int row = 0; row < group_proc.Size(); ++row )
        {
            group_sface_I[row + 1] = group_sface_I[row] + Nsteps * face2Dto3D_coeff *
                    meshbase.group_sface.RowSize(row);
        }
    else //Dim == 3
        for ( int row = 0; row < group_proc.Size(); ++row )
        {
            group_sface_I[row + 1] = group_sface_I[row] + Nsteps * face2Dto3D_coeff *
                    meshbase.group_sedge.RowSize(row);
        }

    group_sface_J = new int[group_sface_I[group_proc.Size() - 1]];

    cout << flush;
    MPI_Barrier(comm);


    int cnt_shfaces = 0;

    for ( int faceind = 0; faceind < GetNumFaces(); ++faceind)
    {
        Element * face = faces[faceind];
        int * v = face->GetVertices();

        set<int> faceproj;
        for ( int vert = 0; vert < face->GetNVertices() ; ++vert )
        {
            //assuming  all time slabs have the same number of nodes and \
            no additional vertices are added to the 4d prisms
            faceproj.insert( v[vert] % nv_base);
        }


        auto findproj = ShfacesBase.find(faceproj);
        if (findproj != ShfacesBase.end() )
        {

            sface_lface[cnt_shfaces] = faceind;

            groupind = findproj->second[0] + 1;

            group.Recreate(group_proc.RowSize(groupind), group_proc.GetRow(groupind));

            int temp = groups.Insert(group);


            if (Dim == 4)
            {
                if(getSwappedFaceElementInfo(faceind))
                    Swap(v); // FIX IT. 100% UNSURE about whether it is correct
                shared_faces[cnt_shfaces] = new Tetrahedron(v);
            }
            else // Dim == 3
            {
                // this is from MFEM 3.3. Have not tested with this release at all
                //Tetrahedron *tet = (Tetrahedron *)(elements[faces_info[faceind].Elem1No]);
                //tet->GetMarkedFace(faces_info[faceind].Elem1Inf/64, v);


                Tetrahedron *tet =
                   (Tetrahedron *)(elements[faces_info[faceind].Elem1No]);
                int re[2], type, flag, *tv;
                tet->ParseRefinementFlag(re, type, flag);
                tv = tet->GetVertices();

                switch (faces_info[faceind].Elem1Inf/64)
                {
                   case 0:
                      switch (re[1])
                      {
                         case 1: v[0] = tv[1]; v[1] = tv[2]; v[2] = tv[3];
                            break;
                         case 4: v[0] = tv[3]; v[1] = tv[1]; v[2] = tv[2];
                            break;
                         case 5: v[0] = tv[2]; v[1] = tv[3]; v[2] = tv[1];
                            break;
                      }
                      break;
                   case 1:
                      switch (re[0])
                      {
                         case 2: v[0] = tv[2]; v[1] = tv[0]; v[2] = tv[3];
                            break;
                         case 3: v[0] = tv[0]; v[1] = tv[3]; v[2] = tv[2];
                            break;
                         case 5: v[0] = tv[3]; v[1] = tv[2]; v[2] = tv[0];
                            break;
                      }
                      break;
                   case 2:
                      v[0] = tv[0]; v[1] = tv[1]; v[2] = tv[3];
                      break;
                   case 3:
                      v[0] = tv[1]; v[1] = tv[0]; v[2] = tv[2];
                      break;
                }


                // Here a flip is made for one of the two processes who share the face
                // To fix the things, swap is been made on the process whose rank is larger.
                //cout << "group size = " << group.Size() << endl;
                const Array<int>& groupme = group;
                //groupme.Print();
                if (myid > min(groupme[0], groupme[1]))
                {
                    //cout << "Swap is made, my id = " << myid << endl;
                    Swap(v);
                }


                /*
                 * old way of face vertices reordering which turned out to be inconsistent
                 * with refinement used for tetrahedron case

                for ( int i = 0; i < 3; ++i)
                    vcoords[i] = GetVertex(v[i]);

                sortingPermutation(3, vcoords, ordering);

                if ( permutation_sign(ordering, 3) == -1 ) //possible values are -1 or 1
                    Swap(v);

                */

                shared_faces[cnt_shfaces] = new Triangle(v);
            }

            // computing the local index of one of the tetrahedrons(4d) or triangles(3d)
            // which is projected onto the same face(edge) in 3d(2d):

            int pos;
            int tslab, tslab_localind;

            // time slab which the tetrahedron (triangle) belongs to.
            // It is initialized with the maximum possible value and then defined as a
            // minimum tslab number over all tetrahedronv vertices
            tslab = Nsteps - 1;
            for ( int vert = 0; vert < face->GetNVertices() ; ++vert )
                if (v[vert] / nv_base  < tslab)
                    tslab = v[vert]/nv_base;

            // The order within a time slab is as follows: All tetrahedra(triangles)
            // are one above the other, so 0 goes for the lowest in the timeslab,
            // 1 for the next one, etc...
            int nv_lower = 0; // number of vertices on the lower base of the time slab prism
            for ( int vert = 0; vert < face->GetNVertices() ; ++vert )
            {
                if (v[vert] / nv_base == tslab)
                    nv_lower++;
                else if (v[vert] / nv_base != tslab + 1)
                {
                    cout << "Strange: a vertex is neither on the top nor on the bottom" << endl;
                    cout << "tslab = " << tslab << " ";
                    cout << "v[vert] = " << v[vert] << " ";
                    cout << "nv_base = " << nv_base << endl;
                    cout << flush;
                }

            }

            if (nv_lower < 1 || nv_lower > DimBase)
                cout << "Strange: nv_lower = " << nv_lower << " either too many or"
                              " too few vertices on the lower base" << endl << flush;
            else
            {
                tslab_localind = DimBase - nv_lower;
            }

            pos = findproj->second[1] * Nsteps * face2Dto3D_coeff +
                    tslab * face2Dto3D_coeff + tslab_localind;


            group_sface_J[group_sface_I[temp - 1] + pos] = cnt_shfaces;

            cnt_shfaces++;

        }
    }

    if (cnt_shfaces != shared_faces.Size())
        cout << "Error: smth wrong with the number of shared faces" << endl << flush;

    group_sface.SetIJ(group_sface_I, group_sface_J, group_proc.Size() - 1);


    // 3.2 shared_edges 3d & shared_faces 3d -> shared_planars 4d
    // ...

    int cnt_inface = 0, cnt_inedge = 0;
    if (Dim == 4)
    {
        // Nsteps + 1 triangles as bases for lateral 3d-in-4d prisms for a one 4d space-time prism
        // Nsteps * 2 triangles inside decomposition of each lateral 3d-in-4d prism into
        // tetrahedrons with shared triangle3d (shared face 3d) as the base
        // Nsteps * 2 triangles on the vertical lateral sides of each 3d-in-4d lateral prism
        // for each 4d prism with shared segment3d (shared edge 3d) as the base
        shared_planars.SetSize( (Nsteps * 2 + (Nsteps + 1))*meshbase.shared_faces.Size()
                                       + Nsteps * 2 * meshbase.shared_edges.Size());
        splan_lplan.SetSize( shared_planars.Size());

        //alternative way to construct group_splan - from I and J arrays manually
        int * group_splan_I, * group_splan_J;
        group_splan_I = new int[group_proc.Size() + 1];
        group_splan_I[0] = 0;
        for ( int row = 0; row < group_proc.Size(); ++row )
        {
            group_splan_I[row + 1] = group_splan_I[row] +
                    (Nsteps * 2 + (Nsteps + 1))*meshbase.group_sface.RowSize(row) +
                    Nsteps * 2 * meshbase.group_sedge.RowSize(row);
        }
        group_splan_J = new int[group_splan_I[group_proc.Size() - 1]];

        vector<double *> vcoords(3);
        vector<vector<double>> vcoordsNew(3);
        int ordering[3];
        //int orderingNew[3];

        for ( int planind4d = 0; planind4d < GetNPlanars(); ++planind4d)
        {
            //if (myid == 4)
                //cout << "planind4d =  " << planind4d << " / " << GetNPlanars() << endl;

            Element * plan4d = planars[planind4d];
            int * v = plan4d->GetVertices();

            set<int> planproj;
            for ( int vert = 0; vert < plan4d->GetNVertices() ; ++vert )
            {
                //assuming  all time slabs have the same number of nodes and \
                no additional vertices are added to the 4d prisms
                planproj.insert( v[vert] % nv_base);
            }

            auto findproj_inface = ShfacesBase.find(planproj);
            auto findproj_inedge = ShedgesBase.find(planproj);

            // = 0 for planars projected onto the 3d face and smth for planars
            // projected onto the 3d edge
            int shift;

            if ( findproj_inface != ShfacesBase.end())
            {
                //if ( myid == 4 )
                    //cout << "appending a 4d planar because of 3d face " << endl << flush;

                groupind = findproj_inface->second[0] + 1;
                group.Recreate(group_proc.RowSize(groupind), group_proc.GetRow(groupind));

                int temp = groups.Insert(group);

                // computing the local index of one of the planars which produce
                // the same projection onto 3d which happens to be a shared face 3d:
                // For all time prisms over the shared face 3d (which is a triangle) there are
                // Nsteps + 1 bases of the prisms and 2 *  Nsteps triangles which are 2d in 4d and
                // are between the bases.
                // 0 for the lowest 3d-like base
                // 1,2 for planars above it in the same time slab with 2 and 1 points on the
                // lowest base
                // 3 ~ 0 but in the next time slab
                // etc...
                // Trying to understand, think of all 3d tetrahedrons which decompose a long prism
                // and their faces = planars(trianles) with Nsteps + 1 plane sections.
                // We consider all triangles which are projected onto the base and create
                // a numeration over them.

                shift = 0;

                // time slab which the triangle belongs to. (minimal time slab over all vertices)
                // It is initialized with the maximum possible value and then defined as a minimum
                // tslab number over all tetrahedronv vertices
                int tslab = Nsteps; //the uppermost base will formally be in this time slab
                for ( int vert = 0; vert < plan4d->GetNVertices() ; ++vert )
                    if (v[vert] / nv_base  < tslab)
                        tslab = v[vert]/nv_base;

                // index within a timeslab: 0,1, or 2 based on the following order:
                // 0 for the lower base, 1 ...
                // and 2 for the triangle which is higher than the others in the timeslab
                // because planars are actually one above of the other.
                int tslabprism_lind;
                int nv_lower = 0; // number of vertices on the lwer 3d base of the time slab prism
                for ( int vert = 0; vert < plan4d->GetNVertices() ; ++vert )
                {
                    if (v[vert] / nv_base == tslab)
                        nv_lower++;
                    else if (v[vert] / nv_base != tslab + 1)
                        cout << "Strange face-type planar: a vertex is neither on the"
                                " top nor on the bottom" << endl << flush;
                }

                if (nv_lower > 2)
                    tslabprism_lind = 0;
                else if (nv_lower < 2)
                    tslabprism_lind = 2;
                else if (nv_lower = 2)
                    tslabprism_lind = 1;
                else
                    cout << "Strange face-type planar: nv_lower is not 1,2,3" << endl;

                int pos = shift + findproj_inface->second[1] * (Nsteps * 2 + (Nsteps + 1))
                        + tslab * 3 + tslabprism_lind;

                group_splan_J[group_splan_I[temp - 1] + pos] = cnt_inface + cnt_inedge;

                cnt_inface++;
            }

            if ( findproj_inedge != ShedgesBase.end())
            {
                //if ( myid == 4 )
                    //cout << "appending a 4d planar because of 3d edge " << endl << flush;

                groupind = findproj_inedge->second[0] + 1;
                group.Recreate(group_proc.RowSize(groupind), group_proc.GetRow(groupind));

                int temp = groups.Insert(group);

                // computing the local index of one of the planars which produce
                // the same projection onto 3d which happens to be a shared edge 3d:
                // For all time prisms over the shared face 3d (which is a triangle) there are
                // Nsteps + 1 bases of the prisms and 2 *  Nsteps triangles which are 2d in 4d and
                // are between the bases.
                // 0 for the lowest 3d-like base
                // 1,2 for planars above it in the same time slab with 2 and 1 points
                // on the lowest base
                // 3 ~ 0 but in the next time slab
                // etc...
                // Trying to understand, think of all planars (triangles) which are projected
                // onto a given shared edge = long rectangle (1d + time) with Nsteps + 1 plane
                // sections.
                // We consider all triangles which are projected onto the base-edge and create
                // a numeration over them.

                // first we need to jump over places reserved for planars which are projected
                // onto the shared faces 3d for a given group of processors
                shift = (Nsteps * 2 + (Nsteps + 1))*meshbase.group_sface.RowSize(temp - 1);

                // time slab which the triangle belongs to. (minimal time slab over all vertices)
                // It is initialized with the maximum possible value and then defined as a minimum
                // tslab number over all vertices
                int tslab = Nsteps - 1;
                for ( int vert = 0; vert < plan4d->GetNVertices() ; ++vert )
                    if (v[vert] / nv_base  < tslab)
                        tslab = v[vert]/nv_base;

                // index within a timeslab: 0 or 1 based on the following order:
                // Consider a rectangle in space-time whith shared edge as the base.
                // There is one diagonal which splits it into two triangles. We set:
                // 0 for the lower one (with 2 vertices on the base) and 1 for the other.
                int tslabprism_lind;
                int nv_lower = 0; // number of vertices on the lower 3d-like base of the
                // space-time rectangle
                for ( int vert = 0; vert < plan4d->GetNVertices() ; ++vert )
                {
                    if (v[vert] / nv_base == tslab)
                        nv_lower++;
                    else if (v[vert] / nv_base != tslab + 1)
                        cout << "Strange edge-type planar: a vertex is neither on the "
                                "top nor on the bottom" << endl << flush;
                }

                if (nv_lower == 2)
                    tslabprism_lind = 0;
                else if (nv_lower == 1)
                    tslabprism_lind = 1;
                else
                    cout << "Strange edge-type planar: nv_lower is not 1 or 2" << endl << flush;

                int pos = shift + findproj_inedge->second[1] * Nsteps * 2
                        + tslab * 2 + tslabprism_lind;

                group_splan_J[group_splan_I[temp - 1] + pos] = cnt_inface + cnt_inedge;

                cnt_inedge++;
            }


            if (findproj_inface != ShfacesBase.end() || findproj_inedge != ShedgesBase.end())
            {
                // Here we swap the planars so that their orientation is consistent across
                // processes. For that we use the order based on the geometric ordering of
                // vertices:
                // Vertex A > Vertex B <-> x(A) > x(B) or ( x(A) = x(B) and y(A) > y(B) or (...))

                for ( int i = 0; i < 3; ++i)
                    vcoords[i] = GetVertex(v[i]);

                // old
                //sortingPermutation(4, vcoords, ordering);

                for (int vert = 0; vert < 3; ++vert)
                    vcoordsNew[vert].assign(vcoords[vert],
                                            vcoords[vert] + Dim);

                sortingPermutationNew(vcoordsNew, ordering);

                /*
                sortingPermutationNew(vcoordsNew, orderingNew);

                cout << " Comparing sorting permutation functions" << endl;
                for ( int i = 0; i < 3; ++i)
                    if (ordering[i] != orderingNew[i])
                        cout << "ERRRRRRRRRRRORRRRRRRRRR";
                */


                if ( permutation_sign(ordering, 3) == -1 ) //possible values are -1 or 1
                    Swap(v);

                shared_planars[cnt_inface + cnt_inedge - 1] = new Triangle(v);

                splan_lplan[cnt_inface + cnt_inedge - 1] = planind4d;

            }



        }

        group_splan.SetIJ(group_splan_I, group_splan_J, group_proc.Size() - 1);

        int cnt_shplanars = cnt_inface + cnt_inedge;

        if (cnt_shplanars != shared_planars.Size())
            cout << "Error: smth wrong with the number of shared planars" << endl;

    } // end of if Dim == 4 case for creating planars


    // 3.3 shared vertices 3d & shared edges 3d -> shared edges 4d
    // (or shared vertices 2d & shared edges 2d -> shared edges 3d)

    // 4d case (3d case):
    // Nsteps + 1 segments which are parallel to a shared edge 3d(2d)
    // Nsteps segments which are diagonals in 2D-in-4d(3d) space-time rectangles
    // with a shared edge 3d(2d) as the base
    // Nsteps segments which are vertical sides in 2D-in-4d space-time rectangles
    // with a shared edge 3d(2d) as the base, which are actually one vertical segment for
    // each shared vertex.

    shared_edges.SetSize( ((Nsteps + 1) + Nsteps)*meshbase.shared_edges.Size()
                                   + Nsteps*meshbase.svert_lvert.Size());
    sedge_ledge.SetSize( shared_edges.Size());

    // alternative way to construct group_sedge - from I and J arrays manually
    int * group_sedge_I, * group_sedge_J;
    group_sedge_I = new int[group_proc.Size() + 1];
    group_sedge_I[0] = 0;
    for ( int row = 0; row < group_proc.Size(); ++row )
    {
        group_sedge_I[row + 1] = group_sedge_I[row] +
                ((Nsteps + 1) + Nsteps)*meshbase.group_sedge.RowSize(row) +
                Nsteps*meshbase.group_svert.RowSize(row);
    }
    group_sedge_J = new int[group_sedge_I[group_proc.Size() - 1]];

    cnt_inedge = 0; // was already used for planars in 4d case
    int cnt_invert = 0;
    Array<int> verts;
    for ( int edgeind = 0; edgeind < GetNEdges(); ++edgeind)
    {
        GetEdgeVertices(edgeind, verts);

        set<int> edgeproj;
        for ( int vert = 0; vert < verts.Size() ; ++vert )
        {
            //assuming  all time slabs have the same number of nodes and
            // no additional vertices are added to the 4d prisms
            edgeproj.insert( verts[vert] % nv_base);

        }

        // = 0 for edges projected onto the 3d edge and smth for edges
        // projected onto the 3d vertex
        int shift;

        auto findproj_inedge = ShedgesBase.find(edgeproj);
        auto findproj_invert = ShvertsBase.find(edgeproj);

        if (findproj_inedge != ShedgesBase.end() || findproj_invert != ShvertsBase.end())
        {
            shared_edges[cnt_inedge + cnt_invert] = new Segment(verts,1);
            sedge_ledge[cnt_inedge + cnt_invert] = edgeind;
        }

        if ( findproj_inedge != ShedgesBase.end())
        {
            groupind = findproj_inedge->second[0] + 1;
            group.Recreate(group_proc.RowSize(groupind), group_proc.GetRow(groupind));

            int temp = groups.Insert(group);

            // computing the local index of one of the edges which produce
            // the same projection which happens to be a shared edge 3d (2d):
            // To understand, think of all 4d(3d) edges which project to the same edge.
            // They form a long space-time rectangle with Nsteps + 1 plane sections.
            // Within each time slab there is also one diagonal inside the corresponding
            // rectangle. We consider all not-vertical edges there and create a numeration
            // for them.
            // These are (omitting strictly time vertical edges):
            // Nsteps + 1 bases-edges of the rectangle and Nsteps diagonals.
            // 0 for the lowest edge parallel to the base (3d or 2d)
            // 1 for the diagonal in the same time slab
            // 2 ~ 0 but in the next time slab
            // etc...

            shift = 0;

            // time slab which the edge belongs to. (minimal time slab over all vertices)
            // It is initialized with the maximum possible value and then defined as a minimum
            // tslab number over all tetrahedronv vertices
            int tslab = Nsteps; //the uppermost base will formally be in this time slab
            for ( int vert = 0; vert < verts.Size() ; ++vert )
                if (verts[vert] / nv_base  < tslab)
                    tslab = verts[vert]/nv_base;

            // index within a timeslab: 0 or 1 based on the following order:
            // 0 for the 3d-like base-edge
            // 1 for the diagonal within the same timeslab
            int tslab_localind;
            // number of vertices on the lower 3d-like base of the space-time rectangle
            int nv_lower = 0;
            for ( int vert = 0; vert < verts.Size() ; ++vert )
            {
                if (verts[vert] / nv_base == tslab)
                    nv_lower++;
                else if (verts[vert] / nv_base != tslab + 1)
                    cout << "Strange edge-type edge: a vertex is neither on the top"
                            " nor on the bottom" << endl << flush;
            }

            if (nv_lower < 1)
                cout << "Strange edge-type edge: nv_lower is not 1 or 2" << endl << flush;
            else
            {
                tslab_localind = 2 - nv_lower;
            }

            /*
            if (nv_lower == 2)
                tslabprism_lind = 0;
            else if (nv_lower = 1)
                tslabprism_lind = 1;
            else
                cout << "Strange edge-type edge: nv_lower is not 1 or 2" << endl;
            */

            int pos = shift + findproj_inedge->second[1] * ((Nsteps + 1) + Nsteps)
                    + tslab * 2 + tslab_localind;

            group_sedge_J[group_sedge_I[temp - 1] + pos] = cnt_inedge + cnt_invert;

            cnt_inedge++;
        }

        if ( findproj_invert != ShvertsBase.end())
        {
            groupind = findproj_invert->second[0] + 1;
            group.Recreate(group_proc.RowSize(groupind), group_proc.GetRow(groupind));

            int temp = groups.Insert(group);

            // computing the local index of one of the edges which produce
            // the same projection onto 3d which happens to be a shared vertex 3d:
            // Trying to understand, think of all 4d edges which project to the same 3d vertex.
            // They form a long time vertical line with Nsteps + 1 points on it.
            // We consider all vertical edges there and create a numeration for them.
            // There are Nsteps of them, one per time slab:

            // first we need to jump over places reserved for edges which are projected
            // onto the shared edges 3d for a given group of processors
            shift = ((Nsteps + 1) + Nsteps)*meshbase.group_sedge.RowSize(temp - 1);

            // time slab which the edge belongs to. (minimal time slab over all vertices)
            // It is initialized with the maximum possible value and then defined as a minimum
            // tslab number over all tetrahedron vertices
            int tslab = Nsteps; //the uppermost base will formally be in this time slab
            for ( int vert = 0; vert < verts.Size() ; ++vert )
                if (verts[vert] / nv_base  < tslab)
                    tslab = verts[vert]/nv_base;

            int pos = shift + findproj_invert->second[1] * Nsteps + tslab;

            group_sedge_J[group_sedge_I[temp - 1] + pos] = cnt_inedge + cnt_invert;

            cnt_invert++;
        }

    }
    group_sedge.SetIJ(group_sedge_I, group_sedge_J, group_proc.Size() - 1);


    int cnt_shedges = cnt_inedge + cnt_invert;

    if (cnt_shedges != shared_edges.Size())
        cout << "Error: smth wrong with the number of shared faces" << endl << flush;

    /*

    for ( int proc = 0; proc < num_procs; ++proc )
    {
        if ( proc == myid )
        //if ( proc == 0 )
        //if ( proc == 1 && proc == myid )
        {
            cout << "I am " << proc << ", parmesh myrank = " << MyRank << endl;
            cout << "sedge_ledge 4d(3d)" << endl;
            sedge_ledge.Print(cout, 10);
            cout << "group_sedge 4d(3d)" << endl;
            group_sedge.Print(cout, 10);
            for ( int row = 0; row < group_sedge.Size(); ++row)
            {
                int rowsize = group_sedge.RowSize(row);
                int * rowcols = group_sedge.GetRow(row);

                cout << "Row = " << row << endl;
                for ( int col = 0; col < rowsize; ++col)
                {
                    cout << "Edge No." << col << endl;

                    Array<int> v;
                    GetEdgeVertices(sedge_ledge[rowcols[col]], v);

                    for ( int vertno = 0; vertno < 2; ++vertno)
                    {
                        //simple
                        //cout << v[vertno] << " ";
                        // with coords
                        double * vcoords = GetVertex(v[vertno]);
                        cout << vertno << ": (";
                        for ( int coord = 0; coord < Dim; ++coord)
                        {
                            cout << vcoords[coord] << " ";
                        }
                        cout << ")  " << endl;
                    }
                    cout << endl;
                }

            }

            cout << flush;
        }
        MPI_Barrier(comm);
    }

    cout << flush;
    MPI_Barrier(comm);
    */

    // 3.4 shared vertices in th base (3d or 2d) -> shared vertices (4d or 3d)
    // ...
    // Nsteps + 1 time slabs = Nsteps + 1 copies for each shared vertex in the base
    svert_lvert.SetSize( (Nsteps + 1)*meshbase.svert_lvert.Size());

    //alternative way to construct group_sedge - from I and J arrays manually
    int * group_svert_I, * group_svert_J;
    group_svert_I = new int[group_proc.Size() + 1];
    group_svert_I[0] = 0;
    for ( int row = 0; row < group_proc.Size(); ++row )
    {
        group_svert_I[row + 1] = group_svert_I[row] +
                (Nsteps + 1)*meshbase.group_svert.RowSize(row);
    }
    group_svert_J = new int[group_svert_I[group_proc.Size() - 1]];


    int cnt_shverts = 0;
    for ( int vertind = 0; vertind < GetNV(); ++vertind)
    {
        set<int> vertproj;
        vertproj.insert ( vertind % nv_base );

        auto findproj_inverts = ShvertsBase.find(vertproj);

        if ( findproj_inverts != ShvertsBase.end())
        {
            svert_lvert[cnt_shverts] = vertind;

            groupind = findproj_inverts->second[0] + 1;
            group.Recreate(group_proc.RowSize(groupind), group_proc.GetRow(groupind));

            int temp = groups.Insert(group);

            // computing the local index of one of the vertices which produce
            // the same projection onto 3d which is  a shared vertex 3d:
            // All such vertices are in a time-like line with shared vertex 3d at the bottom
            // There are Nsteps + 1 of them, one per each time moment (including 0):

            // time moment for the vertex
            int timemoment = vertind / nv_base;

            int pos = findproj_inverts->second[1] * (Nsteps + 1) + timemoment;

            group_svert_J[group_svert_I[temp - 1] + pos] = cnt_shverts;

            cnt_shverts++;
        }

    }

    group_svert.SetIJ(group_svert_I, group_svert_J, group_proc.Size() - 1);


    cout << flush;
    MPI_Barrier(comm);

    /*
    for ( int proc = 0; proc < num_procs; ++proc )
    {
        if ( proc == myid )
        {
            cout << "I am " << proc << ", parmesh myrank = " << MyRank << endl;
            cout << "svert_lvert 4d(3d)" << endl;
            svert_lvert.Print(cout, 10);
            cout << "group_svert 4d(3d)" << endl;
            group_svert.Print(cout, 10);
            for ( int row = 0; row < group_svert.Size(); ++row)
            {
                int rowsize = group_svert.RowSize(row);
                int * rowcols = group_svert.GetRow(row);

                cout << "Row = " << row << endl;
                for ( int col = 0; col < rowsize; ++col)
                {
                    cout << "Vert No." << col << endl;

                    double * vcoords = GetVertex(svert_lvert[rowcols[col]]);

                    cout << "(";
                    for ( int coord = 0; coord < Dim; ++coord)
                    {
                        cout << vcoords[coord] << " ";
                    }
                    cout << ")  " << endl;

                }

            }

            cout << flush;
        }
        MPI_Barrier(comm);
    }

    cout << flush;
    MPI_Barrier(comm);
    */


    // ****************************************************************************
    // step 4 of x: creating main communication structure for parmesh 4d = gtopo
    // ****************************************************************************

    gtopo.Create(groups, 822);

    MPI_Barrier(comm);
    return;
}

// Computes domain and boundary volumes, and checks,
// that faces and boundary elements lists are consistent with the actual element faces
int Mesh::MeshCheck ()
{
    int num_procs, myid;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    int dim = Dimension();

    if ( dim != 4 && dim != 3)
    {
        cout << "Case dim != 3 or 4 is not supported in MeshCheck()" << endl;
        return -1;
    }

    if (BaseGeom != Geometry::PENTATOPE && BaseGeom != Geometry::TETRAHEDRON)
    {
        cout << "MeshCheck() is implemented only for pentatops and tetrahedrons" << endl;
        return -1;
    }

    // 2.5.0: assuming that vertices are fine, nothing is done for them

    // 2.5.1: volume check (+2.5.0) means that elements don't intersect
    // and no holes inside the domain are present
    double domain_volume = 0.0;
    double el_volume;
    double * pointss[dim + 1];
    DenseMatrix VolumeEl;
    VolumeEl.SetSize(dim);

    for ( int elind = 0; elind < GetNE(); ++elind)
    {
        Element * el = GetElement(elind);
        int * v = el->GetVertices();

        for ( int i = 0; i < dim + 1; ++i)
            pointss[i] = GetVertex(v[i]);

        for ( int i = 0; i < dim; ++i)
        {
            for ( int j = 0; j < dim; ++j)
            {
                VolumeEl.Elem(i,j) = pointss[i + 1][j] - pointss[0][j];
            }
        }

        el_volume = fabs (VolumeEl.Det() / factorial(dim)); //24 = 4!

        domain_volume += el_volume;
    }

    for ( int proc = 0; proc < num_procs; ++proc )
    {
        if ( proc == myid )
        {
            cout << "I am " << proc << endl;
            cout << "Domain volume from mesh = " << domain_volume << endl;
        }
        cout << flush;
        MPI_Barrier(comm);
    }

    // 2.5.2: Checking that faces are consistent
    int nbndface = 0, nintface = 0;

    //GenerateFaces();
    int nfaces;
    if ( Dimension() == 4)
        nfaces = GetNFaces();
    else // Dim == 3
        nfaces = GetNEdges();
    for (int face = 0; face < GetNFaces(); ++face)
    {
       int el1, el2;
       GetFaceElements(face, &el1, &el2);

       //cout << "faceind = " << face << endl;
       //cout << "el indices: " << el1 << " and " << el2 << endl;

       if ( el1 == -1 || el2 == -1 )
           nbndface++;
       if ( el1 != -1 && el2 != -1 )
           nintface++;
    }

    //return 0;

    //cout << "nfaces = " << mesh.GetNFaces() << endl;
    //cout << "nbe = " << mesh.GetNBE() << endl;
    //cout << "nbndface = " << nbndface << ", nintface = " << nintface << endl;

    // 2.5.3: Checking the boundary volume
    double boundary_volume = 0.0;
    for ( int belind = 0; belind < GetNBE(); ++belind)
    {
        Element * el = GetBdrElement(belind);
        int * v = el->GetVertices();

        if (dim == 4)
        {
            double * point0 = GetVertex(v[0]);
            double * point1 = GetVertex(v[1]);
            double * point2 = GetVertex(v[2]);
            double * point3 = GetVertex(v[3]);

            double a1, a2, a3, a4, a5, a6;
            a1 = dist(point0, point1, dim);
            a2 = dist(point0, point2, dim);
            a3 = dist(point0, point3, dim);
            a4 = dist(point1, point2, dim);
            a5 = dist(point2, point3, dim);
            a6 = dist(point3, point1, dim);

            // formula from the webpage
            // http://keisan.casio.com/exec/system/1329962711
            el_volume = 0.0;
            el_volume += a1*a1*a5*a5*(a2*a2 + a3*a3 + a4*a4 + a6*a6 - a1*a1 - a5*a5);
            el_volume += a2*a2*a6*a6*(a1*a1 + a3*a3 + a4*a4 + a5*a5 - a2*a2 - a6*a6);
            el_volume += a3*a3*a4*a4*(a1*a1 + a2*a2 + a5*a5 + a6*a6 - a3*a3 - a4*a4);
            el_volume += - a1*a1*a2*a2*a4*a4 - a2*a2*a3*a3*a5*a5;
            el_volume += - a1*a1*a3*a3*a6*a6 - a4*a4*a5*a5*a6*a6;
            el_volume = el_volume/144.0;

            el_volume = sqrt(el_volume);
        }
        else // dim == 3
        {
            double * point0 = GetVertex(v[0]);
            double * point1 = GetVertex(v[1]);
            double * point2 = GetVertex(v[2]);

            double a1, a2, a3;
            a1 = dist(point0, point1, dim);
            a2 = dist(point0, point2, dim);
            a3 = dist(point1, point2, dim);

            // Heron's formula
            double halfp = 0.5 * (a1 + a2 + a3); //half of the perimeter
            el_volume = sqrt ( halfp * (halfp - a1) * (halfp - a2) * (halfp - a3) );
        }


        //cout << "bel_volume" << el_volume << endl;

        boundary_volume += el_volume;
    }

    for ( int proc = 0; proc < num_procs; ++proc )
    {
        if ( proc == myid )
        {
            cout << "I am " << proc << endl;
            cout << "Boundary volume from mesh = " << boundary_volume << endl;
        }
        cout << flush;
        MPI_Barrier(comm);
    }


    // 2.5.3: Checking faces using elements, brute-force type
    set<set<int>> BndElemSet;
    for ( int belind = 0; belind < GetNBE(); ++belind)
    {
        Element * el = GetBdrElement(belind);
        int * v = el->GetVertices();

        set<int> belset;

        for ( int i = 0; i < dim; ++i )
                belset.insert(v[i]);
        BndElemSet.insert(belset);
    }

    //cout << "BndElemSet size (how many different bdr elements in boundary) = " \
         << BndElemSet.size() << endl;

    map<set<int>,int> FaceElemMap;
    int facecount = 0;
    int bndcountcheck = 0;
    for ( int elind = 0; elind < GetNE(); ++elind)
    {
        Element * el = GetElement(elind);
        int * v = el->GetVertices();

        for ( int elface = 0; elface < dim + 1; ++elface)
        {
            set<int> faceset;

            for ( int i = 0; i < dim + 1; ++i )
                if (i != elface )
                    faceset.insert(v[i]);

            auto findset = FaceElemMap.find(faceset);
            if (findset != FaceElemMap.end() )
                FaceElemMap[faceset]++;
            else
            {
                FaceElemMap[faceset] = 1;
                facecount++;
            }

            auto findsetbel = BndElemSet.find(faceset);
            if (findsetbel != BndElemSet.end() )
                bndcountcheck++;
        }
    }

    //cout << "FaceElemMap: " << facecount << " faces" <<  endl;
    //cout << "Checking: bndcountcheck = " << bndcountcheck << endl;
    int bndmapcount = 0, intmapcount = 0;
    for(auto const& ent : FaceElemMap)
    {
        //for (int temp: ent.first)
            //cout << temp << " ";
        //cout << ": " << ent.second << endl;

        if (ent.second == 1)
            bndmapcount++;
        else if (ent.second == 2)
            intmapcount++;
        else
            cout << "ERROR: wrong intmapcount" << endl;
    }

    //cout << "Finally: bndmapcount = " << bndmapcount << ", intmapcount = " << intmapcount << endl;

    if ( bndmapcount != nbndface )
    {
        cout << "Something is wrong with bdr elements:" << endl;
        cout << "bndmapcount = " << bndmapcount << "must be equal to nbndface = " << nbndface << endl;
        return - 1;
    }
    if ( intmapcount != nintface )
    {
        cout << "Something is wrong with bdr elements:" << endl;
        cout << "intmapcount = " << intmapcount << "must be equal to nintface = " << nintface << endl;
        return - 1;
    }

    cout << "Bdr elements are consistent w.r.t elements!" << endl;

    return 0;
}

// Creates an IntermediateMesh whihc stores main arrays of the Mesh.
Mesh::IntermediateMesh * Mesh::ExtractMeshToInterMesh()
{
    int Dim = Dimension(), NumOfElements = GetNE(),
            NumOfBdrElements = GetNBE(),
            NumOfVertices = GetNV();

    if ( Dim != 4 && Dim != 3 )
    {
       cerr << "Wrong dimension in ExtractMeshToInterMesh(): " << Dim << endl;
       return NULL;
    }

    if (BaseGeom != Geometry::PENTATOPE && BaseGeom != Geometry::TETRAHEDRON)
    {
        cout << "ExtractMeshToInterMesh() is implemented only for pentatops and tetrahedrons" << endl;
        return NULL;
    }

    IntermediateMesh * intermesh = new IntermediateMesh;
    IntermeshInit( intermesh, Dim, NumOfVertices, NumOfElements, NumOfBdrElements, 1);

    for ( int elind = 0; elind < GetNE(); ++elind)
    {
        Element * el = GetElement(elind);
        int * v = el->GetVertices();

        for ( int i = 0; i < Dim + 1; ++i )
            intermesh->elements[elind*(Dim+1) + i] = v[i];
        intermesh->elattrs[elind] = el->GetAttribute();
    }

    for ( int belind = 0; belind < GetNBE(); ++belind)
    {
        Element * el = GetBdrElement(belind);
        int * v = el->GetVertices();

        for ( int i = 0; i < Dim; ++i )
            intermesh->bdrelements[belind*Dim + i] = v[i];
        intermesh->bdrattrs[belind] = el->GetAttribute();
    }

    for ( int vind = 0; vind < GetNV(); ++vind)
    {
        double * coords = GetVertex(vind);

        for ( int i = 0; i < Dim; ++i )
            intermesh->vertices[vind*Dim + i] = coords[i];
    }

    return intermesh;
}

// Converts a given ParMesh into a serial Mesh, and outputs the corresponding partioning as well.
Mesh::Mesh ( ParMesh& pmesh, int ** partioning)
{
    MPI_Comm comm = pmesh.GetComm();

    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    int dim = pmesh.Dimension();
    int nvert_per_elem = dim + 1; // PENTATOPE and TETRAHEDRON case only
    int nvert_per_bdrelem = dim;  // PENTATOPE and TETRAHEDRON case only

    // step 1: extract local parmesh parts to the intermesh and
    // replace local vertex indices by the global indices

    BaseGeom = pmesh.BaseGeom;

    if (BaseGeom != Geometry::PENTATOPE && BaseGeom != Geometry::TETRAHEDRON)
    {
        if (myid == 0)
            cout << "This Mesh constructor works only for pentatops and tetrahedrons"
                 << endl << flush;
        return;
    }

    IntermediateMesh * local_intermesh = pmesh.ExtractMeshToInterMesh();

    FiniteElementCollection * lin_coll = new LinearFECollection;
    ParFiniteElementSpace * pspace = new ParFiniteElementSpace(&pmesh, lin_coll);

    int nv_global = pspace->GlobalTrueVSize(); // global number of vertices in the 4d mesh

    // writing the global vertex numbers inside the local IntermediateMesh(4d)
    int lvert;
    for ( int lvert = 0; lvert < local_intermesh->nv; ++lvert )
    {
        local_intermesh->vert_gindices[lvert] = pspace->GetGlobalTDofNumber(lvert);
    }

    //InterMeshPrint (local_intermesh, myid, "local_intermesh_inConvert");
    //MPI_Barrier(comm);

    // replacing local vertex indices by global indices from parFEspace
    // converting local to global vertex indices in elements
    for (int elind = 0; elind < local_intermesh->ne; ++elind)
    {
        for ( int j = 0; j < nvert_per_elem; ++j )
        {
            lvert = local_intermesh->elements[elind * nvert_per_elem + j];
            local_intermesh->elements[elind * nvert_per_elem + j] =
                    local_intermesh->vert_gindices[lvert];
        }
    }

    // converting local to global vertex indices in boundary elements
    for (int bdrelind = 0; bdrelind < local_intermesh->nbe; ++bdrelind)
    {
        for ( int j = 0; j < nvert_per_bdrelem; ++j )
        {
            lvert = local_intermesh->bdrelements[bdrelind * nvert_per_bdrelem + j];
            local_intermesh->bdrelements[bdrelind * nvert_per_bdrelem + j] =
                    local_intermesh->vert_gindices[lvert];
        }
    }

    delete lin_coll;
    delete pspace;


    // step 2: exchange local intermeshes between processors

    // 2.1: exchanging information about local sizes between processors
    // in order to set up mpi exchange parameters and allocate the future 4d mesh;

    // nvdg_global = sum of local number of vertices (without thinking that \
    some vertices are shared between processors)
    int nvdg_global, ne_global, nbe_global;

    int *recvcounts_el = new int[num_procs];
    MPI_Allgather( &(local_intermesh->ne), 1, MPI_INT, recvcounts_el, 1, MPI_INT, comm);

    int *rdispls_el = new int[num_procs];
    rdispls_el[0] = 0;
    for ( int i = 0; i < num_procs - 1; ++i)
        rdispls_el[i + 1] = rdispls_el[i] + recvcounts_el[i];

    ne_global = rdispls_el[num_procs - 1] + recvcounts_el[num_procs - 1];

    //cout << "ne_global = " << ne_global << endl;

    int * partioning_ = new int[ne_global];
    int elcount = 0;
    for ( int proc = 0; proc < num_procs; ++proc )
    {
        for ( int el = 0; el < recvcounts_el[proc]; ++el )
        {
            partioning_[elcount++] = proc;
        }
    }

    *partioning = partioning_;

    int *recvcounts_be = new int[num_procs];

    MPI_Allgather( &(local_intermesh->nbe), 1, MPI_INT, recvcounts_be, 1, MPI_INT, comm);

    int *rdispls_be = new int[num_procs];

    rdispls_be[0] = 0;
    for ( int i = 0; i < num_procs - 1; ++i)
        rdispls_be[i + 1] = rdispls_be[i] + recvcounts_be[i];

    nbe_global = rdispls_be[num_procs - 1] + recvcounts_be[num_procs - 1];

    int *recvcounts_v = new int[num_procs];
    MPI_Allgather( &(local_intermesh->nv), 1, MPI_INT, recvcounts_v, 1, MPI_INT, comm);

    int *rdispls_v = new int[num_procs];
    rdispls_v[0] = 0;
    for ( int i = 0; i < num_procs - 1; ++i)
        rdispls_v[i + 1] = rdispls_v[i] + recvcounts_v[i];

    nvdg_global = rdispls_v[num_procs - 1] + recvcounts_v[num_procs - 1];

    MPI_Barrier(comm);

    Mesh::IntermediateMesh * intermesh_global = new IntermediateMesh;
    Mesh::IntermeshInit( intermesh_global, dim, nvdg_global, ne_global, nbe_global, 1);

    // 2.2: exchanging attributes, elements and vertices between processes using allgatherv

    // exchanging element attributes
    MPI_Allgatherv( local_intermesh->elattrs, local_intermesh->ne, MPI_INT,
                    intermesh_global->elattrs, recvcounts_el, rdispls_el, MPI_INT, comm);

    // exchanging bdr element attributes
    MPI_Allgatherv( local_intermesh->bdrattrs, local_intermesh->nbe, MPI_INT,
                    intermesh_global->bdrattrs, recvcounts_be, rdispls_be, MPI_INT, comm);

    // exchanging elements, changing recvcounts_el!!!
    for ( int i = 0; i < num_procs; ++i)
    {
        recvcounts_el[i] *= nvert_per_elem;
        rdispls_el[i] *= nvert_per_elem;
    }


    MPI_Allgatherv( local_intermesh->elements, (local_intermesh->ne)*nvert_per_elem, MPI_INT,
                    intermesh_global->elements, recvcounts_el, rdispls_el, MPI_INT, comm);

    // exchanging bdrelements, changing recvcounts_be!!!
    for ( int i = 0; i < num_procs; ++i)
    {
        recvcounts_be[i] *= nvert_per_bdrelem;
        rdispls_be[i] *= nvert_per_bdrelem;
    }


    MPI_Allgatherv( local_intermesh->bdrelements, (local_intermesh->nbe)*nvert_per_bdrelem, MPI_INT,
                    intermesh_global->bdrelements, recvcounts_be, rdispls_be, MPI_INT, comm);

    // exchanging global vertex indices
    MPI_Allgatherv( local_intermesh->vert_gindices, local_intermesh->nv, MPI_INT,
                    intermesh_global->vert_gindices, recvcounts_v, rdispls_v, MPI_INT, comm);

    // exchanging vertices : At the moment dg-type procedure = without considering presence
    // of shared vertices
    for ( int i = 0; i < num_procs; ++i)
    {
        recvcounts_v[i] *= dim;
        rdispls_v[i] *= dim;
    }

    MPI_Allgatherv( local_intermesh->vertices, (local_intermesh->nv)*dim, MPI_DOUBLE,
                    intermesh_global->vertices, recvcounts_v, rdispls_v, MPI_DOUBLE, comm);

    IntermeshDelete(local_intermesh);

    // step 3: load serial mesh4d from the created global intermesh4d

    InitMesh(dim,dim, nv_global, ne_global, nbe_global);

    // 3.1: creating the correct vertex array where each vertex is met only once
    // 3.1.1: cleaning up the vertices which are at the moment with multiple entries for shared
    // vertices

    int gindex;
    std::map<int, double*> vertices_unique; // map structure for storing only unique vertices

    // loop over all (with multiple entries) vertices, unique are added to the map object
    double * tempvert_map;
    for ( int i = 0; i < nvdg_global; ++i )
    {
        tempvert_map = new double[dim];
        for ( int j = 0; j < dim; j++ )
            tempvert_map[j] = intermesh_global->vertices[i * dim + j];
        gindex = intermesh_global->vert_gindices[i];
        vertices_unique[gindex] = tempvert_map;
    }

    // counting the final number of vertices. after that count_vert should be equal to nv_global
    int count_vert = 0;
    for(auto const& ent : vertices_unique)
    {
        count_vert ++;
    }

    if ( count_vert != nv_global && myid == 0 )
    {
        cout << "Wrong number of vertices! Smth is probably wrong" << endl << flush;
    }

    // 3.1.2: creating the vertices array with taking care of shared vertices using
    // the std::map vertices_unique
    //delete [] intermesh_global->vertices;
    intermesh_global->nv = count_vert;
    //intermesh_global->vertices = new double[count_vert * dim];

    // now actual intermesh_global->vertices is:
    // right unique vertices + some vertices which are still alive after mpi transfer.
    // so we reuse the memory already allocated for vertices array with multiple entries.

    int tmp = 0;
    for(auto const& ent : vertices_unique)
    {
        for ( int j = 0; j < dim; j++)
        {
            intermesh_global->vertices[tmp*dim + j] = ent.second[j];
        }

        if ( tmp != ent.first )
            cout << "ERROR" << endl;
        tmp++;
    }

    vertices_unique.clear();

    //InterMeshPrint (intermesh_global, myid, "intermesh_reduced_inConvert");
    //MPI_Barrier(comm);

    // 3.2: loading created intermesh_global into a mfem mesh object
    // (temporarily copying the memory: FIX IT may be)
    if (dim == 4)
    {
        BaseGeom = Geometry::PENTATOPE;
        BaseBdrGeom = Geometry::TETRAHEDRON;
    }
    else // dim == 3
    {
        BaseGeom = Geometry::TETRAHEDRON;
        BaseBdrGeom = Geometry::TRIANGLE;
    }
    LoadMeshfromArrays( intermesh_global->nv, intermesh_global->vertices,
             intermesh_global->ne, intermesh_global->elements, intermesh_global->elattrs,
             intermesh_global->nbe, intermesh_global->bdrelements, intermesh_global->bdrattrs, dim );

    // 3.3 create the internal structure for mesh after el-s,, bdel-s and vertices have been loaded
    int refine = 1;
    CreateInternalMeshStructure(refine);

    IntermeshDelete (intermesh_global);

    MPI_Barrier (comm);
    return;
}

void ParMesh::PrintSharedStructParMesh ( int* permutation )
{
    int num_procs, myid;
    MPI_Comm comm = GetComm();
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    if (BaseGeom != Geometry::PENTATOPE && BaseGeom != Geometry::TETRAHEDRON && BaseGeom != Geometry::TRIANGLE)
    {
        if (myid == 0)
            cout << "PrintSharedStructParMesh() is implemented only for pentatops, "
                    "tetrahedrons and triangles" << endl << flush;
        return;
    }

    cout << flush;
    MPI_Barrier(comm);
    if (myid == 0)
        cout << "PrintSharedStructParMesh:" << endl;
    cout << flush;
    MPI_Barrier(comm);


    for ( int proc = 0; proc < num_procs; ++proc )
    {
        if ( proc == myid )
        {
            cout << "I am " << proc << ", parmesh myrank = " << MyRank << endl;
            cout << "myid = " << myid << ", num_procs = " << num_procs << endl;


            if ( Dimension() >= 3 )
            {
                cout << "group_sface" << endl;
                group_sface.Print(cout,10);
            }
            if (Dimension() == 4)
            {
                cout << "group_splan" << endl;
                group_splan.Print(cout,20);
            }
            cout << "group_svert" << endl;
            group_svert.Print();

            for ( int row = 0; row < group_svert.Size(); ++row)
            {
                int rowsize = group_svert.RowSize(row);
                int * rowcols = group_svert.GetRow(row);

                cout << "Row = " << row << endl;
                for ( int col = 0; col < rowsize; ++col)
                {
                    cout << "Vert No." << col << endl;

                    cout << "(";
                    double * vcoords = GetVertex(svert_lvert[rowcols[col]]);
                    for ( int coord = 0; coord < Dimension(); ++coord)
                    {
                        cout << vcoords[coord] << " ";
                    }
                    cout << ")  " << endl;
                    //cout << "rowcols[col] = " << rowcols[col];
                }

            }



            if (Dimension() >= 3)
            {
                cout << "shared_faces" << endl;
                for ( int i = 0; i < shared_faces.Size(); ++i)
                {
                    Element * el = shared_faces[i];
                    int *v = el->GetVertices();
                    if ( !permutation)
                    {
                        for ( int vert = 0; vert < Dimension(); ++vert)
                            cout << v[vert] << " ";
                        cout << endl;
                        //cout << v[0] << " " << v[1] << " " << v[2] << " " << v[3] << endl;
                    }
                    else
                    {
                        for ( int vert = 0; vert < Dimension(); ++vert)
                            cout << permutation[v[vert]] << " ";
                        cout << endl;
                    }
                }



                for ( int row = 0; row < group_sface.Size(); ++row)
                {
                    int rowsize = group_sface.RowSize(row);
                    int * rowcols = group_sface.GetRow(row);

                    cout << "Row = " << row << endl;
                    for ( int col = 0; col < rowsize; ++col)
                    {
                        cout << "Face No." << col << endl;

                        cout << "rowcols[col] = " << rowcols[col] << endl;

                        Element * el = shared_faces[rowcols[col]];
                        int *v = el->GetVertices();

                        for ( int vertno = 0; vertno < el->GetNVertices(); ++vertno)
                        {
                            //simple
                            //cout << v[vertno] << " ";
                            // with coords
                            double * vcoords = GetVertex(v[vertno]);
                            cout << vertno << ": (";
                            for ( int coord = 0; coord < Dimension(); ++coord)
                            {
                                cout << vcoords[coord] << " ";
                            }
                            cout << ")  " << endl;
                        }
                        cout << endl;
                    }

                }

            } // end of priting shared faces

            if (Dimension() == 4)
            {
                cout << "shared_planars" << endl;
                for ( int i = 0; i < shared_planars.Size(); ++i)
                {
                    Element * el = shared_planars[i];
                    int *v = el->GetVertices();
                    if ( !permutation)
                        cout << v[0] << " " << v[1] << " " << v[2] << endl;
                    else
                        cout << permutation[v[0]] << " " <<
                                permutation[v[1]] << " " << permutation[v[2]] << endl;
                }

                for ( int row = 0; row < group_splan.Size(); ++row)
                {
                    int rowsize = group_splan.RowSize(row);
                    int * rowcols = group_splan.GetRow(row);

                    cout << "Row = " << row << endl;
                    for ( int col = 0; col < rowsize; ++col)
                    {
                        cout << "Planar No." << col << endl;

                        cout << "rowcols[col] = " << rowcols[col] << endl;

                        Element * el = shared_planars[rowcols[col]];
                        int *v = el->GetVertices();

                        for ( int vertno = 0; vertno < el->GetNVertices(); ++vertno)
                        {
                            //simple
                            //cout << v[vertno] << " ";
                            // with coords
                            double * vcoords = GetVertex(v[vertno]);
                            cout << vertno << ": (";
                            for ( int coord = 0; coord < Dimension(); ++coord)
                            {
                                cout << vcoords[coord] << " ";
                            }
                            cout << ")  " << endl;
                        }
                        cout << endl;
                    }

                }
            }


            cout << "shared_edges" << endl;
            for ( int i = 0; i < shared_edges.Size(); ++i)
            {
                Element * el = shared_edges[i];
                int *v = el->GetVertices();
                if ( !permutation)
                    cout << v[0] << " " << v[1] << endl;
                else
                    cout << permutation[v[0]] << " " << permutation[v[1]] << endl;
            }
            cout << "sedge_ledge" << endl;
            sedge_ledge.Print();
            cout << "group_sedge" << endl;
            group_sedge.Print(cout, 10);


            //GetEdgeVertexTable(); //this call crashes everything because it changes the edges
            // if you don't delete edge_vertex and set it to NULL afterwards
            //delete edge_vertex;
            //edge_vertex = NULL;

            /*
            if (edge_vertex)
            {
                cout << "I already have edge_vertex" << endl;
                edge_vertex->Print();
            }
            else
                cout << "I don't have here edge_vertex" << endl;


            DSTable v_to_v(NumOfVertices);
            GetVertexToVertexTable(v_to_v);

            int nedges = v_to_v.NumberOfEntries();

            if (!edge_vertex)
            {
                cout << "Creating edge_vertex" << endl;

                edge_vertex = new Table(nedges, 2);


                for (int i = 0; i < NumOfVertices; i++)
                {
                   for (DSTable::RowIterator it(v_to_v, i); !it; ++it)
                   {
                      int j = it.Index();
                      edge_vertex->Push(j, i);
                      edge_vertex->Push(j, it.Column());
                   }
                }

                edge_vertex->Finalize();
                delete edge_vertex;
                edge_vertex = NULL;
            }

            */


            for ( int row = 0; row < group_sedge.Size(); ++row)
            {
                int rowsize = group_sedge.RowSize(row);
                int * rowcols = group_sedge.GetRow(row);

                cout << "Row = " << row << endl;
                for ( int col = 0; col < rowsize; ++col)
                {
                    cout << "Edge No." << col << endl;

                    Array<int> v;
                    GetEdgeVertices(sedge_ledge[rowcols[col]], v);

                    for ( int vertno = 0; vertno < 2; ++vertno)
                    {
                        //simple
                        //cout << v[vertno] << " ";
                        // with coords
                        double * vcoords = GetVertex(v[vertno]);
                        cout << vertno << ": (";
                        for ( int coord = 0; coord < Dim; ++coord)
                        {
                            cout << vcoords[coord] << " ";
                        }
                        cout << ")  " << endl;
                    }
                    cout << endl;
                }

            }
            //if not delete here, get segfault for more than two parallel refinements
            delete edge_vertex;
            edge_vertex = NULL;




            cout << "sface_lface" << endl;
            sface_lface.Print();
            if (Dimension() == 4)
            {
                cout << "splan_lplan" << endl;
                splan_lplan.Print();
            }
            cout << "sedge_ledge" << endl;
            sedge_ledge.Print();
            cout << "svert_lvert" << endl;
            svert_lvert.Print();


            cout << flush;
        }
        MPI_Barrier(comm);
    }

    return;
}

// scalar product of two vectors (outputs 0 if vectors have different length)
double sprod(vector<double> vec1, vector<double> vec2)
{
    if (vec1.size() != vec2.size())
        return 0.0;
    double res = 0.0;
    for ( int c = 0; c < vec1.size(); ++c)
        res += vec1[c] * vec2[c];
    return res;
}
double l2Norm(vector<double> vec)
{
    return sqrt(sprod(vec,vec));
}

// compares pairs<int,double> with respect to the second (double) elements
bool intdComparison(const pair<int,double> &a,const pair<int,double> &b)
{
       return a.second>b.second;
}

// only first 2 coordinates of each element of Points is used (although now the
// input is 4 3-dimensional points but the last coordinate is time so it is not used
// because the slice is with t = const planes
// sorts in a counter-clock fashion required by VTK format for quadrilateral
// the main output is the permutation of the input points array
bool sortQuadril2d(vector<vector<double>> & Points, int * permutation)
{
    bool verbose = false;

    if (Points.size() != 4)
    {
        cout << "Error: sortQuadril2d should be called only for a vector storing 4 points" << endl;
        return false;
    }
    /*
    for ( int p = 0; p < Points.size(); ++p)
        if (Points[p].size() != 2)
        {
            cout << "Error: sortQuadril2d should be called only for a vector storing 4 2d-points" << endl;
            return false;
        }
    */

    /*
    cout << "Points inside sortQuadril2d() \n";
    for (int i = 0; i < 4; ++i)
    {
        cout << "vert " << i << ":";
        for ( int j = 0; j < 2; ++j)
            cout << Points[i][j] << " ";
        cout << endl;
    }
    */


    int argbottom = 0; // index of the the vertex with the lowest y-coordinate
    for (int p = 1; p < 4; ++p)
        if (Points[p][1] < Points[argbottom][1])
            argbottom = p;

    if (verbose)
        cout << "argbottom = " << argbottom << endl;

    // cosinuses of angles between radius vectors from vertex argbottom to the others and positive x-direction
    vector<pair<int, double>> cos(3);
    vector<vector<double>> radiuses(3);
    vector<double> xort(2);
    xort[0] = 1.0;
    xort[1] = 0.0;
    int cnt = 0;
    for (int p = 0; p < 4; ++p)
    {
        if (p != argbottom)
        {
            cos[cnt].first = p;
            for ( int c = 0; c < 2; ++c)
                radiuses[cnt].push_back(Points[p][c] - Points[argbottom][c]);
            cos[cnt].second = sprod(radiuses[cnt], xort) / l2Norm(radiuses[cnt]);
            cnt ++;
        }
    }

    //int permutation[4];
    permutation[0] = argbottom;

    std::sort(cos.begin(), cos.end(), intdComparison);

    for ( int i = 0; i < 3; ++i)
        permutation[1 + i] = cos[i].first;

    if (verbose)
    {
        cout << "permutation:" << endl;
        for (int i = 0; i < 4; ++i)
            cout << permutation[i] << " ";
        cout << endl;
    }

    // not needed actually. onlt for debugging. actually the output is the correct permutation
    /*
    vector<vector<double>> temp(4);
    for ( int p = 0; p < 4; ++p)
        for ( int i = 0; i < 3; ++i)
            temp[p].push_back(Points[permutation[p]][i]);

    for ( int p = 0; p < 4; ++p)
        for ( int i = 0; i < 3; ++i)
            Points[p][i] = temp[p][i];
    */
    return true;
}

// sorts the vertices in order for the points to form a proper vtk wedge
// first three vertices should be the base, with normal to (0,1,2)
// looking opposite to the direction of where the second base is.
// This ordering is required by VTK format for wedges, look
// in vtk wedge class definitio for explanations
// the main output is the permutation of the input vertexes array
bool sortWedge3d(vector<vector<double>> & Points, int * permutation)
{
    /*
    cout << "wedge points:" << endl;
    for ( int i = 0; i < Points.size(); ++i)
    {
        for ( int j = 0; j < Points[i].size(); ++j)
            cout << Points[i][j] << " ";
        cout << endl;
    }
    */

    vector<double> p1 = Points[0];
    int pn2 = -1;
    vector<int> pnum2;

    //bestimme die 2 quadrate
    for(unsigned int i=1; i<Points.size(); i++)
    {
        vector<double> dets;
        for(unsigned int k=1; k<Points.size()-1; k++)
        {
            for(unsigned int l=k+1; l<Points.size(); l++)
            {
                if(k!=i && l!=i)
                {
                    vector<double> Q1(3);
                    vector<double> Q2(3);
                    vector<double> Q3(3);

                    for ( int c = 0; c < 3; c++)
                        Q1[c] = p1[c] - Points[i][c];
                    for ( int c = 0; c < 3; c++)
                        Q2[c] = p1[c] - Points[k][c];
                    for ( int c = 0; c < 3; c++)
                        Q3[c] = p1[c] - Points[l][c];

                    //vector<double> Q1 = p1 - Points[i];
                    //vector<double> Q2 = p1 - Points[k];
                    //vector<double> Q3 = p1 - Points[l];

                    DenseMatrix MM(3,3);
                    MM(0,0) = Q1[0]; MM(0,1) = Q2[0]; MM(0,2) = Q3[0];
                    MM(1,0) = Q1[1]; MM(1,1) = Q2[1]; MM(1,2) = Q3[1];
                    MM(2,0) = Q1[2]; MM(2,1) = Q2[2]; MM(2,2) = Q3[2];
                    double determ = MM.Det();

                    dets.push_back(determ);
                }
            }
        }

        double max_ = 0; double min_ = fabs(dets[0]);
        for(unsigned int m=0; m<dets.size(); m++)
        {
            if(max_<fabs(dets[m])) max_ = fabs(dets[m]);
            if(min_>fabs(dets[m])) min_ = fabs(dets[m]);
        }

        //for ( int in = 0; in < dets.size(); ++in)
            //cout << "det = " << dets[in] << endl;

        if(max_!=0) for(unsigned int m=0; m<dets.size(); m++) dets[m] /= max_;

        //cout << "max_ = " << max_ << endl;

        int count = 0;
        vector<bool> el;
        for(unsigned int m=0; m<dets.size(); m++) { if(fabs(dets[m]) < 1e-8) { count++; el.push_back(true); } else el.push_back(false); }

        if(count==2)
        {
            for(unsigned int k=1, m=0; k<Points.size()-1; k++)
                for(unsigned int l=k+1; l<Points.size(); l++)
                {
                    if(k!=i && l!=i)
                    {
                        if(el[m]) { pnum2.push_back(k); pnum2.push_back(l); }
                        m++;
                    }

                }

            pn2 = i;
            break;
        }

        if(count == 0 || count > 2)
        {
            //cout << "count == 0 || count > 2" << endl;
            //cout << "count = " << count << endl;
            return false;
        }
    }

    if(pn2<0)
    {
        //cout << "pn2 < 0" << endl;
        return false;
    }


    vector<int> oben(3); oben[0] = pn2;
    vector<int> unten(3); unten[0] = 0;

    //winkel berechnen
    vector<double> pp1(3);
    vector<double> pp2(3);
    for ( int c = 0; c < 3; c++)
        pp1[c] = Points[0][c] - Points[pn2][c];
    for ( int c = 0; c < 3; c++)
        pp2[c] = Points[pnum2[0]][c] - Points[pn2][c];
    //vector<double> pp1 = Points[0] - Points[pn2];
    //vector<double> pp2 = Points[pnum2[0]] - Points[pn2];
    double w1 = sprod(pp1, pp2)/(l2Norm(pp1)*l2Norm(pp2));
    for ( int c = 0; c < 3; c++)
        pp2[c] = Points[pnum2[1]][c] - Points[pn2][c];
    //pp2 = Points[pnum2[1]]- Points[pn2];
    double w2 = sprod(pp1, pp2)/(l2Norm(pp1)*l2Norm(pp2));

    if(w1 < w2)  { oben[1] = pnum2[0]; unten[1] = pnum2[1]; }
    else{ oben[1] = pnum2[1]; unten[1] = pnum2[0]; }

    for ( int c = 0; c < 3; c++)
        pp2[c] = Points[pnum2[2]][c] - Points[pn2][c];
    //pp2 = Points[pnum2[2]] - Points[pn2];
    w1 = sprod(pp1, pp2)/(l2Norm(pp1)*l2Norm(pp2));
    for ( int c = 0; c < 3; c++)
        pp2[c] = Points[pnum2[3]][c] - Points[pn2][c];
    //pp2 = Points[pnum2[3]]- Points[pn2];
    w2 = sprod(pp1, pp2)/(l2Norm(pp1)*l2Norm(pp2));

    if(w1 < w2)  { oben[2] = pnum2[2]; unten[2] = pnum2[3]; }
    else{ oben[2] = pnum2[3]; unten[2] = pnum2[2]; }

    for(int i=0; i<unten.size(); i++) permutation[i] = unten[i];
    for(int i=0; i<oben.size(); i++)  permutation[i + unten.size()] = oben[i];

    //not needed since we actually need the permutation only
    /*
    vector<vector<double>> pointssort;
    for(unsigned int i=0; i<unten.size(); i++) pointssort.push_back(Points[unten[i]]);
    for(unsigned int i=0; i<oben.size(); i++) pointssort.push_back(Points[oben[i]]);

    for(unsigned int i=0; i<pointssort.size(); i++) Points[i] = pointssort[i];
    */

    return true;
}

// reorders the cell vertices so as to have the cell vertex ordering compatible with VTK format
// the output is the sorted elvertexes (which is also the input)
void reorder_cellvertices ( int dim, int nip, vector<vector<double>> & cellpnts, vector<int> & elvertexes)
{
    bool verbose = false;
    // used only for checking the orientation of tetrahedrons
    DenseMatrix Mtemp(3, 3);

    // special reordering of vertices is required for the vtk wedge, so that
    // vertices are added one base after another and not as a mix

    if (nip == 6)
    {

        /*
        cout << "Sorting the future wedge" << endl;
        cout << "Before sorting: " << endl;
        for (int i = 0; i < 6; ++i)
        {
            cout << "vert " << i << ":";
            for ( int j = 0; j < dim - 1; ++j)
                cout << cellpnts[i][j] << " ";
            cout << endl;
        }
        */


        // FIX IT: NOT TESTED AT ALL
        int permutation[6];
        if ( sortWedge3d (cellpnts, permutation) == false )
        {
            cout << "sortWedge returns false, possible bad behavior" << endl;
            return;
        }

        /*
        cout << "After sorting: " << endl;
        for (int i = 0; i < 6; ++i)
        {
            cout << "vert " << i << ":";
            for ( int j = 0; j < dim - 1; ++j)
                cout << cellpnts[permutation[i]][j] << " ";
            cout << endl;
        }
        */

        int temp[6];
        for ( int i = 0; i < 6; ++i)
            temp[i] = elvertexes[permutation[i]];
        for ( int i = 0; i < 6; ++i)
            elvertexes[i] = temp[i];


        double det = 0.0;

        for ( int i = 0; i < dim - 1; ++i)
            Mtemp(i,0) = (1.0/3.0)*(cellpnts[permutation[3]][i] + cellpnts[permutation[4]][i] + cellpnts[permutation[5]][i])
                    - cellpnts[permutation[0]][i];

        for ( int i = 0; i < dim - 1; ++i)
            Mtemp(i,1) = cellpnts[permutation[2]][i] - cellpnts[permutation[0]][i];

        for ( int i = 0; i < dim - 1; ++i)
            Mtemp(i,2) = cellpnts[permutation[1]][i] - cellpnts[permutation[0]][i];

        det = Mtemp.Det();

        if (verbose)
        {
            if (det < 0)
                cout << "orientation for wedge = negative" << endl;
            else if (det = 0.0)
                cout << "error for wedge: bad volume" << endl;
            else
                cout << "orientation for wedge = positive" << endl;
        }

        if (det < 0)
        {
            if (verbose)
                cout << "Have to swap the vertices to change the orientation of wedge" << endl;
            int tmp;
            tmp = elvertexes[1];
            elvertexes[1] = elvertexes[0];
            elvertexes[1] = tmp;
            //Swap(*(elvrtindices[momentind].end()));
            tmp = elvertexes[4];
            elvertexes[4] = elvertexes[3];
            elvertexes[4] = tmp;
        }

    }


    // positive orientation is required for vtk tetrahedron
    // normal to the plane with first three vertexes should poit towards the 4th vertex

    if (nip == 4 && dim == 4)
    {
        /*
        cout << "tetrahedra points" << endl;
        for (int i = 0; i < 4; ++i)
        {
            cout << "vert " << i << ":";
            for ( int j = 0; j < dim - 1; ++j)
                cout << cellpnts[i][j] << " ";
            cout << endl;
        }
        */

        double det = 0.0;

        for ( int i = 0; i < dim - 1; ++i)
            Mtemp(i,0) = cellpnts[3][i] - cellpnts[0][i];

        for ( int i = 0; i < dim - 1; ++i)
            Mtemp(i,1) = cellpnts[2][i] - cellpnts[0][i];

        for ( int i = 0; i < dim - 1; ++i)
            Mtemp(i,2) = cellpnts[1][i] - cellpnts[0][i];

        //Mtemp.Print();

        det = Mtemp.Det();

        if (verbose)
        {
            if (det < 0)
                cout << "orientation for tetra = negative" << endl;
            else if (det = 0.0)
                cout << "error for tetra: bad volume" << endl;
            else
                cout << "orientation for tetra = positive" << endl;
        }

        //return;

        if (det < 0)
        {
            if (verbose)
                cout << "Have to swap the vertices to change the orientation of tetrahedron" << endl;
            int tmp = elvertexes[1];
            elvertexes[1] = elvertexes[0];
            elvertexes[1] = tmp;
            //Swap(*(elvrtindices[momentind].end()));
        }

    }


    // in 2D case the vertices of a quadrilateral should be umbered in a counter-clock wise fashion
    if (nip == 4 && dim == 3)
    {
        /*
        cout << "Sorting the future quadrilateral" << endl;
        cout << "Before sorting: " << endl;
        for (int i = 0; i < nip; ++i)
        {
            cout << "vert " << elvertexes[i] << ":";
            for ( int j = 0; j < dim - 1; ++j)
                cout << cellpnts[i][j] << " ";
            cout << endl;
        }
        */

        int permutation[4];
        sortQuadril2d(cellpnts, permutation);

        int temp[4];
        for ( int i = 0; i < 4; ++i)
            temp[i] = elvertexes[permutation[i]];
        for ( int i = 0; i < 4; ++i)
            elvertexes[i] = temp[i];

        /*
        cout << "After sorting: " << endl;
        for (int i = 0; i < nip; ++i)
        {
            cout << "vert " << elvertexes[i] << ":";
            for ( int j = 0; j < dim - 1; ++j)
                cout << cellpnts[permutation[i]][j] << " ";
            cout << endl;
        }
        */

    }

    return;
}

// computes elpartition array which is used for computing slice meshes over different time moments
// elpartition is the output
// elpartition stores for each time moment a vector of integer indices of the mesh elements which intersect
// with the corresponding time plane
void Mesh::Compute_elpartition (double t0, int Nmoments, double deltat, vector<vector<int> > & elpartition)
{
    bool verbose = false;
    int dim = Dimension();

    Element * el;
    int * vind;
    double * vcoords;
    double eltmin, eltmax;

    for ( int elind = 0; elind < GetNE(); ++elind)
    {
        if (verbose)
            cout << "elind = " << elind << endl;
        el = GetElement(elind);
        vind = el->GetVertices();

        // computing eltmin and eltmax for an element = minimal and maximal time moments for each element
        eltmin = t0 + Nmoments * deltat;
        eltmax = 0.0;
        for (int vno = 0; vno < el->GetNVertices(); ++vno )
        {
            vcoords = GetVertex(vind[vno]);
            if ( vcoords[dim - 1] > eltmax )
                eltmax = vcoords[dim - 1];
            if ( vcoords[dim - 1] < eltmin )
                eltmin = vcoords[dim - 1];
        }


        if (verbose)
        {
            cout << "Special print: elind = " << elind << endl;
            for (int vno = 0; vno < el->GetNVertices(); ++vno )
            {
                cout << "vertex: ";
                vcoords = GetVertex(vind[vno]);
                for ( int coo = 0; coo < dim; ++coo )
                    cout << vcoords[coo] << " ";
                cout << endl;
            }

            cout << "eltmin = " << eltmin << " eltmax = " << eltmax << endl;
        }




        // deciding which time moments intersect the element if any
        //if ( (eltmin > t0 && eltmin < t0 + (Nmoments-1) * deltat) ||  (eltmax > t0 && eltmax < t0 + (Nmoments-1) * deltat))
        if ( (eltmax > t0 && eltmin < t0 + (Nmoments-1) * deltat))
        {
            if (verbose)
            {
                cout << "the element is intersected by some time moments" << endl;
                cout << "t0 = " << t0 << " deltat = " << deltat << endl;
                cout << fixed << setprecision(6);
                cout << "low bound = " << ceil( (max(eltmin,t0) - t0) / deltat  ) << endl;
                cout << "top bound = " << floor ((min(eltmax,t0+(Nmoments-1)*deltat) - t0) / deltat) << endl;
                cout << "4isl for low = " << max(eltmin,t0) - t0 << endl;
                cout << "magic number for low = " << (max(eltmin,t0) - t0) / deltat << endl;
                cout << "magic number for top = " << (min(eltmax,t0+(Nmoments-1)*deltat) - t0) / deltat << endl;
            }
            for ( int k = ceil( (max(eltmin,t0) - t0) / deltat  ); k <= floor ((min(eltmax,t0+(Nmoments-1)*deltat) - t0) / deltat) ; ++k)
            {
                //if (myid == 0 )
                if (verbose)
                {
                    cout << "k = " << k << endl;
                }
                elpartition[k].push_back(elind);
            }
        }
        else
        {
            if (verbose)
                cout << "the element is not intersected by any time moments" << endl;
        }
    }

    // intermediate output
    /*
    for ( int i = 0; i < Nmoments; ++i)
    {
        cout << "moment " << i << ": time = " << t0 + i * deltat << endl;
        cout << "size for this partition = " << elpartition[i].size() << endl;
        for ( int j = 0; j < elpartition[i].size(); ++j)
            cout << "el: " << elpartition[i][j] << endl;
    }
    */
    return;
}

// computes number of slice cell vertexes, slice cell vertex indices and coordinates
// for a given element with index = elind.
// updates the edgemarkers and vertex_count correspondingly
// pvec defines the slice plane
void Mesh::computeSliceCell (int elind, vector<vector<double> > & pvec, vector<vector<double>> & ipoints, vector<int>& edgemarkers,
                             vector<vector<double>>& cellpnts, vector<int>& elvertslocal, int & nip, int & vertex_count )
{
    bool verbose = false; // probably should be a function argument
    int dim = Dimension();

    int * edgeindices;
    int edgenolen, edgeind;
    Array<int> edgev(2);
    double * v1, * v2;

    vector<vector<double>> edgeends(dim);
    edgeends[0].reserve(dim);
    edgeends[1].reserve(dim);

    DenseMatrix M(dim, dim);
    Vector sol(4), rh(4);

    vector<double> ip(dim);

    edgeindices = el_to_edge->GetRow(elind);
    edgenolen = el_to_edge->RowSize(elind);

    nip = 0;

    for ( int edgeno = 0; edgeno < edgenolen; ++edgeno)
    {
        // true mesh edge index
        edgeind = edgeindices[edgeno];

        if (verbose)
            cout << "edgeind " << edgeind << endl;
        if (edgemarkers[edgeind] == -2) // if this edge was not considered
        {
            GetEdgeVertices(edgeind, edgev);

            // vertex coordinates
            v1 = GetVertex(edgev[0]);
            v2 = GetVertex(edgev[1]);

            // vertex coordinates as vectors of doubles, edgeends 0 is lower in time coordinate than edgeends[1]
            if (v1[dim-1] < v2[dim-1])
            {
                for ( int coo = 0; coo < dim; ++coo)
                {
                    edgeends[0][coo] = v1[coo];
                    edgeends[1][coo] = v2[coo];
                }
            }
            else
            {
                for ( int coo = 0; coo < dim; ++coo)
                {
                    edgeends[0][coo] = v2[coo];
                    edgeends[1][coo] = v1[coo];
                }
            }


            if (verbose)
            {
                cout << "edge vertices:" << endl;
                for (int i = 0; i < 2; ++i)
                {
                    cout << "vert ";
                    for ( int coo = 0; coo < dim; ++coo)
                        cout << edgeends[i][coo] << " ";
                    cout << "   ";
                }
                cout << endl;
            }


            // creating the matrix for computing the intersection point
            for ( int i = 0; i < dim; ++i)
                for ( int j = 0; j < dim - 1; ++j)
                    M(i,j) = pvec[j + 1][i];
            for ( int i = 0; i < dim; ++i)
                M(i,dim - 1) = edgeends[0][i] - edgeends[1][i];

            /*
            cout << "M" << endl;
            M.Print();
            cout << "M.Det = " << M.Det() << endl;
            */

            if ( fabs(M.Det()) > MYZEROTOL )
            {
                M.Invert();

                // setting righthand side
                for ( int i = 0; i < dim; ++i)
                    rh[i] = edgeends[0][i] - pvec[0][i];

                // solving the system
                M.Mult(rh, sol);

                if ( sol[dim-1] > 0.0 - MYZEROTOL && sol[dim-1] <= 1.0 + MYZEROTOL)
                {
                    for ( int i = 0; i < dim; ++i)
                        ip[i] = edgeends[0][i] + sol[dim-1] * (edgeends[1][i] - edgeends[0][i]);

                    if (verbose)
                    {
                        cout << "intersection point for this edge: " << endl;
                        for ( int i = 0; i < dim; ++i)
                            cout << ip[i] << " ";
                        cout << endl;
                    }

                    ipoints.push_back(ip);
                    //vrtindices[momentind].push_back(vertex_count);
                    elvertslocal.push_back(vertex_count);
                    edgemarkers[edgeind] = vertex_count;
                    cellpnts.push_back(ip);
                    nip++;
                    vertex_count++;
                }
                else
                {
                    if (verbose)
                        cout << "Line but not edge intersects" << endl;
                    edgemarkers[edgeind] = -1;
                }

            }
            else
                if (verbose)
                    cout << "Edge is parallel" << endl;
        }
        else // the edge was already considered -> edgemarkers store the vertex index
        {
            if (edgemarkers[edgeind] >= 0)
            {
                elvertslocal.push_back(edgemarkers[edgeind]);
                cellpnts.push_back(ipoints[edgemarkers[edgeind]]);
                nip++;
            }
        }

        //cout << "tempvec.size = " << tempvec.size() << endl;

    } // end of loop over element edges

    return;
}

// outputs the slice mesh information in VTK format
void Mesh::outputSliceMeshVTK ( std::stringstream& fname, std::vector<std::vector<double> > & ipoints,
                                std::list<int> &celltypes, int cellstructsize, std::list<std::vector<int> > &elvrtindices)
{
    int dim = Dimension();
    // output in the vtk format for paraview
    std::ofstream ofid(fname.str().c_str());
    ofid.precision(8);

    ofid << "# vtk DataFile Version 3.0" << endl;
    ofid << "Generated by MFEM" << endl;
    ofid << "ASCII" << endl;
    ofid << "DATASET UNSTRUCTURED_GRID" << endl;

    ofid << "POINTS " << ipoints.size() << " double" << endl;
    for (int vno = 0; vno < ipoints.size(); ++vno)
    {
        for ( int c = 0; c < dim - 1; ++c )
        {
            ofid << ipoints[vno][c] << " ";
        }
        if (dim == 3)
            ofid << ipoints[vno][dim - 1] << " ";
        ofid << endl;
    }

    ofid << "CELLS " << celltypes.size() << " " << cellstructsize << endl;
    std::list<int>::const_iterator iter;
    std::list<vector<int> >::const_iterator iter2;
    for (iter = celltypes.begin(), iter2 = elvrtindices.begin();
         iter != celltypes.end() && iter2 != elvrtindices.end()
         ; ++iter, ++iter2)
    {
        //cout << *it;
        int npoints;
        if (*iter == VTKTETRAHEDRON)
            npoints = 4;
        else if (*iter == VTKWEDGE)
            npoints = 6;
        else if (*iter == VTKQUADRIL)
            npoints = 4;
        else //(*iter == VTKTRIANGLE)
            npoints = 3;
        ofid << npoints << " ";

        for ( int i = 0; i < npoints; ++i)
            ofid << (*iter2)[i] << " ";
        ofid << endl;
    }

    ofid << "CELL_TYPES " << celltypes.size() << endl;
    for (iter = celltypes.begin(); iter != celltypes.end(); ++iter)
    {
        ofid << *iter << endl;
    }

    // test lines for cell data
    ofid << "CELL_DATA " << celltypes.size() << endl;
    ofid << "SCALARS cekk_scalars double 1" << endl;
    ofid << "LOOKUP_TABLE default" << endl;
    int cnt = 0;
    for (iter = celltypes.begin(); iter != celltypes.end(); ++iter)
    {
        ofid << cnt * 1.0 << endl;
        cnt++;
    }
    return;
}


// Computes and outputs in VTK format slice meshes of a given 3D or 4D mesh
// by time-like planes t = t0 + k * deltat, k = 0, ..., Nmoments - 1
// myid is used for creating different output files by different processes
// if the mesh is parallel
// usually it is reasonable to refer myid to the process id in the communicator
// so as to produce a correct output for parallel ParaView visualization
void Mesh::ComputeSlices(double t0, int Nmoments, double deltat, int myid)
{
    bool verbose = false;

    if (BaseGeom != Geometry::PENTATOPE && BaseGeom != Geometry::TETRAHEDRON )
    {
        //if (myid == 0)
            cout << "Mesh::ComputeSlices() is implemented only for pentatops "
                    "and tetrahedrons" << endl << flush;
        return;
    }

    int dim = Dimension();

    if (!el_to_edge)
    {
        el_to_edge = new Table;
        NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
    }

    // = -2 if not considered, -1 if considered, but does not intersected, index of this vertex in the new 3d mesh otherwise
    // refilled for each time moment
    vector<int> edgemarkers(GetNEdges());

    // stores indices of elements which are intersected by planes related to the time moments
    vector<vector<int>> elpartition(Nmoments);
    // can make it faster, if any estimates are known for how many elements are intersected by a single time plane
    //for ( int i = 0; i < Nmoments; ++i)
        //elpartition[i].reserve(100);

    // *************************************************************************
    // step 1 of x: loop over all elememnts and compute elpartition for all time
    // moments.
    // *************************************************************************

    Compute_elpartition (t0, Nmoments, deltat, elpartition);


    // *************************************************************************
    // step 2 of x: looping over time momemnts and slicing elements for each
    // given time moment, and outputs the resulting slice mesh in VTK format
    // *************************************************************************

    // slicing the elements, time moment over time moment
    Element * el;
    int elind;

    vector<vector<double>> pvec(dim);
    for ( int i = 0; i < dim; ++i)
        pvec[i].reserve(dim);

    // used only for checking the orientation of tetrahedrons and quadrilateral vertexes reordering
    //DenseMatrix Mtemp(3, 3);

    // output data structures for vtk format
    // for each time moment holds a list with cell type for each cell
    vector<std::list<int>> celltypes(Nmoments);
    // for each time moment holds a list with vertex indices
    //vector<std::list<int>> vrtindices(Nmoments);
    // for each time moment holds a list with cell type for each cell
    vector<std::list<vector<int>>> elvrtindices(Nmoments);

    // number of integers in cell structure - for each cell 1 integer (number of vertices) +
    // + x integers (vertex indices)
    int cellstructsize;
    int vertex_count; // number of vertices in the slice mesh for a single time moment

    // loop over time moments
    for ( int momentind = 0; momentind < Nmoments; ++momentind )
    {
        if (verbose)
            cout << "Time moment " << momentind << ": time = " << t0 + momentind * deltat << endl;

        // refilling edgemarkers, resetting vertex_count and cellstructsize
        for ( int i = 0; i < GetNEdges(); ++i)
            edgemarkers[i] = -2;

        vertex_count = 0;
        cellstructsize = 0;

        vector<vector<double> > ipoints; // one of main arrays: all intersection points for a given time moment

        // vectors, defining the plane of the slice p0, p1, p2 (and p3 in 4D)
        // p0 is the time aligned vector for the given time moment
        // p1, p2 (and p3) - basis orts for the plane
        // pvec is {p0,p1,p2,p3} vector
        for ( int i = 0; i < dim; ++i)
            for ( int j = 0; j < dim; ++j)
                pvec[i][dim - 1 - j] = ( i == j ? 1.0 : 0.0);
        pvec[0][dim - 1] = t0 + momentind * deltat;

        // loop over elements intersected by the plane realted to a given time moment
        // here, elno = index in elpartition[momentind]
        for ( int elno = 0; elno < elpartition[momentind].size(); ++elno)
        //for ( int elno = 0; elno < 3; ++elno)
        {
            vector<int> tempvec;             // vertex indices for the cell of the slice mesh
            tempvec.reserve(6);
            vector<vector<double>> cellpnts; //points of the cell of the slice mesh
            cellpnts.reserve(6);

            // true mesh element index
            elind = elpartition[momentind][elno];
            el = GetElement(elind);

            if (verbose)
                cout << "Element: " << elind << endl;

            // computing number of intersection points, indices and coordinates for
            // local slice cell vertexes (cellpnts and tempvec)  and adding new intersection
            // points and changing edges markers for a given element elind
            // and plane defined by pvec
            int nip;
            computeSliceCell (elind, pvec, ipoints, edgemarkers, cellpnts, tempvec, nip, vertex_count);

            if ( (dim == 4 && (nip != 4 && nip != 6)) || (dim == 3 && (nip != 3 && nip != 4)) )
                cout << "Strange nip =  " << nip << " for elind = " << elind << ", time = " << t0 + momentind * deltat << endl;
            else
            {
                if (nip == 4) // tetrahedron in 3d or quadrilateral in 2d
                    if (dim == 4)
                        celltypes[momentind].push_back(VTKTETRAHEDRON);
                    else // dim == 3
                        celltypes[momentind].push_back(VTKQUADRIL);
                else if (nip == 6) // prism
                    celltypes[momentind].push_back(VTKWEDGE);
                else // nip == 3 = triangle
                    celltypes[momentind].push_back(VTKTRIANGLE);

                cellstructsize += nip + 1;

                elvrtindices[momentind].push_back(tempvec);

                // special reordering of cell vertices, required for the wedge,
                // tetrahedron and quadrilateral cells
                reorder_cellvertices (dim, nip, cellpnts, elvrtindices[momentind].back());

                if (verbose)
                    cout << "nip for the element = " << nip << endl;
            }

        } // end of loop over elements for a given time moment

        // intermediate output
        std::stringstream fname;
        fname << "slicemesh_"<< dim - 1 << "d_myid_" << myid << "_moment_" << momentind << ".vtk";
        outputSliceMeshVTK (fname, ipoints, celltypes[momentind], cellstructsize, elvrtindices[momentind]);


    } //end of loop over time moments

    // if not deleted here, gets segfault for more than two parallel refinements afterwards
    delete edge_vertex;
    edge_vertex = NULL;

    //

    return;
}

// This function is similar to the Mesh::computeSliceCell() but additionally computes the
// values of the grid function in the slice cell vertexes.
// (It is the absolute value for vector finite elements)
// computes number of slice cell vertexes, slice cell vertex indices and coordinates and
// for a given element with index = elind.
// updates the edgemarkers and vertex_count correspondingly
// pvec defines the slice plane
void GridFunction::computeSliceCellValues (int elind, vector<vector<double> > & pvec, vector<vector<double>> & ipoints, vector<int>& edgemarkers,
                             vector<vector<double>>& cellpnts, vector<int>& elvertslocal, int & nip, int & vertex_count, vector<double>& vertvalues)
{
    Mesh * mesh = FESpace()->GetMesh();

    bool verbose = false; // probably should be a function argument
    int dim = mesh->Dimension();

    Array<int> edgev(2);
    double * v1, * v2;

    vector<vector<double>> edgeends(dim);
    edgeends[0].reserve(dim);
    edgeends[1].reserve(dim);

    DenseMatrix M(dim, dim);
    Vector sol(4), rh(4);

    vector<double> ip(dim);

    int edgenolen, edgeind;
    //int * edgeindices;
    //edgeindices = mesh->el_to_edge->GetRow(elind);
    //edgenolen = mesh->el_to_edge->RowSize(elind);
    Array<int> cor; // dummy
    Array<int> edgeindices;
    mesh->GetElementEdges(elind, edgeindices, cor);
    edgenolen = mesh->GetElement(elind)->GetNEdges();

    nip = 0;

    Array<int> vertices;
    mesh->GetElementVertices(elind, vertices);
    double val1, val2;

    /*
    cout << "vertices:" << endl;
    double * temp;
    for ( int i = 0; i < vertices.Size(); ++i)
    {
        temp = mesh->GetVertex(vertices[i]);
        for ( int coo = 0; coo < dim; ++coo)
            cout << temp[coo] << " ";
        cout << endl;
    }
    */

    //int vdim = VectorDim();
    //Array < double > nodal_values;
    //GetNodalValues (elind, nodal_values, vdim);
    //cout << "vdim = " << vdim << endl;
    //cout << "nodal_values.size = " << nodal_values.Size() << endl;

    //cout << "nodal values:" << endl;
    //for ( int i = 0; i < nodal_values.Size(); ++i)
        //cout << nodal_values[i] << " " << endl;
    //cout << endl;

    /*
    Vector pointval;
    IntegrationPoint integp;
    integp.Init();
    integp.Set3(0.5, 0.0, 0.0);
    GetVectorValue(elind, integp, pointval);
    cout << "pointval" << endl;
    for ( int i = 0; i < pointval.Size(); ++i)
        cout << pointval[i] << " ";
    cout << endl;

    Vector pointcoovec(3);
    double pointcoo[3];
    pointcoo[0] = mesh->GetVertex(vertices[0])[0];
    pointcoo[1] = mesh->GetVertex(vertices[0])[1];
    pointcoo[2] = mesh->GetVertex(vertices[0])[2];
    pointcoovec.SetData(pointcoo);
    cout << "pointcoovec" << endl;
    for ( int i = 0; i < pointcoovec.Size(); ++i)
        cout << pointcoovec[i] << " ";
    cout << endl;

    ElementTransformation *Tr = FESpace()->GetElementTransformation(elind);
    Tr->TransformBack(pointcoovec, integp);
    GetVectorValue(elind, integp, pointval);

    cout << "pointval after transform" << endl;
    for ( int i = 0; i < pointval.Size(); ++i)
        cout << pointval[i] << " ";
    cout << endl;
    */

    double pvalue; // value of the grid function at the middle of the edge
    int permut[2]; // defines which of the edge vertexes is the lowest w.r.t time

    Vector pointval1, pointval2;
    IntegrationPoint integp;
    integp.Init();
    /*
    integp.Set3(0.0, 0.0, 0.0);
    pgridfuntest->GetVectorValue(0, integp, pointval);
    cout << "pointval" << endl;
    for ( int i = 0; i < pointval.Size(); ++i)
        cout << pointval[i] << " ";
    cout << endl;
    */


    for ( int edgeno = 0; edgeno < edgenolen; ++edgeno)
    {
        // true mesh edge index
        edgeind = edgeindices[edgeno];

        mesh->GetEdgeVertices(edgeind, edgev);

        // vertex coordinates
        v1 = mesh->GetVertex(edgev[0]);
        v2 = mesh->GetVertex(edgev[1]);

        // vertex coordinates as vectors of doubles, edgeends 0 is lower in time coordinate than edgeends[1]
        if (v1[dim-1] < v2[dim-1])
        {
            for ( int coo = 0; coo < dim; ++coo)
            {
                edgeends[0][coo] = v1[coo];
                edgeends[1][coo] = v2[coo];
            }
            permut[0] = 0;
            permut[1] = 1;
        }
        else
        {
            for ( int coo = 0; coo < dim; ++coo)
            {
                edgeends[0][coo] = v2[coo];
                edgeends[1][coo] = v1[coo];
            }
            permut[0] = 1;
            permut[1] = 0;
        }

        for ( int vno = 0; vno < mesh->GetElement(elind)->GetNVertices(); ++vno)
        {
            int vind = vertices[vno];
            if (vno == 0)
            {
                if (dim == 3)
                    integp.Set3(0.0,0.0,0.0);
                else // dim == 4
                    integp.Set4(0.0,0.0,0.0,0.0);
            }
            if (vno == 1)
            {
                if (dim == 3)
                    integp.Set3(1.0,0.0,0.0);
                else // dim == 4
                    integp.Set4(1.0,0.0,0.0,0.0);
            }
            if (vno == 2)
            {
                if (dim == 3)
                    integp.Set3(0.0,1.0,0.0);
                else // dim == 4
                    integp.Set4(0.0,1.0,0.0,0.0);
            }
            if (vno == 3)
            {
                if (dim == 3)
                    integp.Set3(0.0,0.0,1.0);
                else // dim == 4
                    integp.Set4(0.0,0.0,1.0,0.0);
            }
            if (vno == 4)
            {
                integp.Set4(0.0,0.0,0.0,1.0);
            }

            if (edgev[permut[0]] == vind)
                GetVectorValue(elind, integp, pointval1);
            if (edgev[permut[1]] == vind)
                GetVectorValue(elind, integp, pointval2);
        }

        val1 = 0.0; val2 = 0.0;
        for ( int coo = 0; coo < dim; ++coo)
        {
            val1 += pointval1[coo] * pointval1[coo];
            val2 += pointval2[coo] * pointval2[coo];
        }
        //cout << "val1 = " << val1 << " val2 = " << val2 << endl;

        val1 = sqrt (val1); val2 = sqrt (val2);

        if (verbose)
        {
            cout << "vertex 1: val1 = " << val1 << endl;
            /*
            for ( int vno = 0; vno < mesh->Dimension(); ++vno)
                cout << v1[vno] << " ";
            cout << endl;
            */
            cout << "vertex 2: val2 = " << val2 <<  endl;
            /*
            for ( int vno = 0; vno < mesh->Dimension(); ++vno)
                cout << v2[vno] << " ";
            cout << endl;
            */
        }

        if (verbose)
        {
            cout << "edgeind " << edgeind << endl;

            cout << "edge vertices:" << endl;
            for (int i = 0; i < 2; ++i)
            {
                cout << "vert ";
                for ( int coo = 0; coo < dim; ++coo)
                    cout << edgeends[i][coo] << " ";
                cout << "   ";
            }
            cout << endl;
        }

        // creating the matrix for computing the intersection point
        for ( int i = 0; i < dim; ++i)
            for ( int j = 0; j < dim - 1; ++j)
                M(i,j) = pvec[j + 1][i];
        for ( int i = 0; i < dim; ++i)
            M(i,dim - 1) = edgeends[0][i] - edgeends[1][i];

        /*
        cout << "M" << endl;
        M.Print();
        cout << "M.Det = " << M.Det() << endl;
        */

        if ( fabs(M.Det()) > MYZEROTOL )
        {
            M.Invert();

            // setting righthand side
            for ( int i = 0; i < dim; ++i)
                rh[i] = edgeends[0][i] - pvec[0][i];

            // solving the system
            M.Mult(rh, sol);

        }
        else
            if (verbose)
                cout << "Edge is parallel" << endl;

        //val1 = edgeends[0][dim-1]; val2 = edgeends[1][dim-1]; only for debugging: delete this
        pvalue = sol[dim-1] * val1 + (1.0 - sol[dim-1]) * val2;

        if (verbose)
        {
            cout << fixed << setprecision(6);
            cout << "val1 = " << val1 << " val2 = " << val2 << endl;
            cout << "sol = " << sol[dim-1];
            cout << "pvalue = " << pvalue << endl << endl;
            //cout << fixed << setprecision(4);
        }


        if (edgemarkers[edgeind] == -2) // if this edge was not considered
        {
            if ( fabs(M.Det()) > MYZEROTOL )
            {
                if ( sol[dim-1] > 0.0 - MYZEROTOL && sol[dim-1] <= 1.0 + MYZEROTOL)
                {
                    for ( int i = 0; i < dim; ++i)
                        ip[i] = edgeends[0][i] + sol[dim-1] * (edgeends[1][i] - edgeends[0][i]);

                    if (verbose)
                    {
                        cout << "intersection point for this edge: " << endl;
                        for ( int i = 0; i < dim; ++i)
                            cout << ip[i] << " ";
                        cout << endl;
                    }

                    ipoints.push_back(ip);
                    //vrtindices[momentind].push_back(vertex_count);
                    elvertslocal.push_back(vertex_count);
                    vertvalues.push_back(pvalue);
                    edgemarkers[edgeind] = vertex_count;
                    cellpnts.push_back(ip);
                    nip++;
                    vertex_count++;
                }
                else
                {
                    if (verbose)
                        cout << "Line but not edge intersects" << endl;
                    edgemarkers[edgeind] = -1;
                }

            }
            else
                if (verbose)
                    cout << "Edge is parallel" << endl;
        }
        else // the edge was already considered -> edgemarkers store the vertex index
        {
            if (verbose)
                cout << "Edge was already considered" << endl;
            if (edgemarkers[edgeind] >= 0)
            {
                elvertslocal.push_back(edgemarkers[edgeind]);
                vertvalues.push_back(pvalue);
                cellpnts.push_back(ipoints[edgemarkers[edgeind]]);
                nip++;
            }
        }

        if (verbose)
            cout << endl;

        //cout << "tempvec.size = " << tempvec.size() << endl;

    } // end of loop over element edges

    /*
    cout << "vertvalues in the end of slicecompute" << endl;
    for ( int i = 0; i < nip; ++i)
    {
        cout << "vertval = " << vertvalues[i] << endl;
    }
    */

    return;
}

void GridFunction::outputSliceGridFuncVTK ( std::stringstream& fname, std::vector<std::vector<double> > & ipoints,
                                std::list<int> &celltypes, int cellstructsize, std::list<std::vector<int> > &elvrtindices, std::list<double > & cellvalues, bool forvideo)
                                //std::list<int> &celltypes, int cellstructsize, std::list<std::vector<int> > &elvrtindices, std::list<std::vector<double> > & cellvertvalues)
{
    Mesh * mesh = FESpace()->GetMesh();

    int dim = mesh->Dimension();
    // output in the vtk format for paraview
    std::ofstream ofid(fname.str().c_str());
    ofid.precision(8);

    ofid << "# vtk DataFile Version 3.0" << endl;
    ofid << "Generated by MFEM" << endl;
    ofid << "ASCII" << endl;
    ofid << "DATASET UNSTRUCTURED_GRID" << endl;

    ofid << "POINTS " << ipoints.size() << " double" << endl;
    for (int vno = 0; vno < ipoints.size(); ++vno)
    {
        for ( int c = 0; c < dim - 1; ++c )
        {
            ofid << ipoints[vno][c] << " ";
        }
        if (dim == 3)
            if (forvideo == true)
                ofid << 0.0 << " ";
            else
                ofid << ipoints[vno][dim - 1] << " ";
        ofid << endl;
    }

    ofid << "CELLS " << celltypes.size() << " " << cellstructsize << endl;
    std::list<int>::const_iterator iter;
    std::list<vector<int> >::const_iterator iter2;
    for (iter = celltypes.begin(), iter2 = elvrtindices.begin();
         iter != celltypes.end() && iter2 != elvrtindices.end()
         ; ++iter, ++iter2)
    {
        //cout << *it;
        int npoints;
        if (*iter == VTKTETRAHEDRON)
            npoints = 4;
        else if (*iter == VTKWEDGE)
            npoints = 6;
        else if (*iter == VTKQUADRIL)
            npoints = 4;
        else //(*iter == VTKTRIANGLE)
            npoints = 3;
        ofid << npoints << " ";

        for ( int i = 0; i < npoints; ++i)
            ofid << (*iter2)[i] << " ";
        ofid << endl;
    }

    ofid << "CELL_TYPES " << celltypes.size() << endl;
    for (iter = celltypes.begin(); iter != celltypes.end(); ++iter)
    {
        ofid << *iter << endl;
    }


    // cell data
    ofid << "CELL_DATA " << celltypes.size() << endl;
    ofid << "SCALARS cell_scalars double 1" << endl;
    ofid << "LOOKUP_TABLE default" << endl;
    //int cnt = 0;
    std::list<double>::const_iterator iterd;
    for (iterd = cellvalues.begin(); iterd != cellvalues.end(); ++iterd)
    {
        //cout << "cell data: " << *iterd << endl;
        ofid << *iterd << endl;
        //cnt++;
    }
    return;
}

// Computes and outputs in VTK format slice meshes of a given 3D or 4D mesh
// by time-like planes t = t0 + k * deltat, k = 0, ..., Nmoments - 1
// myid is used for creating different output files by different processes
// if the mesh is parallel
// usually it is reasonable to refeer myid to the process id in the communicator
// For each cell, an average of the values of the grid function is computed over
// slice cell vertexes.
void GridFunction::ComputeSlices(double t0, int Nmoments, double deltat, int myid, bool forvideo)
{
    bool verbose = false;

    Mesh * mesh = FESpace()->GetMesh();
    int dim = mesh->Dimension();

    // = -2 if not considered, -1 if considered, but does not intersected, index of this vertex in the new 3d mesh otherwise
    // refilled for each time moment
    vector<int> edgemarkers(mesh->GetNEdges());

    vector<vector<int>> elpartition(mesh->GetNEdges());
    mesh->Compute_elpartition (t0, Nmoments, deltat, elpartition);

    // *************************************************************************
    // step 2 of x: looping over time momemnts and slicing elements for each
    // given time moment, and outputs the resulting slice mesh in VTK format
    // *************************************************************************

    // slicing the elements, time moment over time moment
    Element * el;
    int elind;

    vector<vector<double>> pvec(dim);
    for ( int i = 0; i < dim; ++i)
        pvec[i].reserve(dim);

    // output data structures for vtk format
    // for each time moment holds a list with cell type for each cell
    vector<std::list<int> > celltypes(Nmoments);
    // for each time moment holds a list with vertex indices
    //vector<std::list<int>> vrtindices(Nmoments);
    // for each time moment holds a list with cell type for each cell
    vector<std::list<vector<int> > > elvrtindices(Nmoments);
    //vector<std::list<vector<double> > > cellvertvalues(Nmoments); // decided not to use this - don't understand how to output correctly in vtk format afterwards
    vector<std::list<double > > cellvalues(Nmoments);

    // number of integers in cell structure - for each cell 1 integer (number of vertices) +
    // + x integers (vertex indices)
    int cellstructsize;
    int vertex_count; // number of vertices in the slice mesh for a single time moment

    // loop over time moments
    for ( int momentind = 0; momentind < Nmoments; ++momentind )
    {
        if (verbose)
            cout << "Time moment " << momentind << ": time = " << t0 + momentind * deltat << endl;

        // refilling edgemarkers, resetting vertex_count and cellstructsize
        for ( int i = 0; i < mesh->GetNEdges(); ++i)
            edgemarkers[i] = -2;

        vertex_count = 0;
        cellstructsize = 0;

        vector<vector<double> > ipoints;    // one of main arrays: all intersection points for a given time moment
        double cellvalue;                   // averaged cell value computed from vertvalues

        // vectors, defining the plane of the slice p0, p1, p2 (and p3 in 4D)
        // p0 is the time aligned vector for the given time moment
        // p1, p2 (and p3) - basis orts for the plane
        // pvec is {p0,p1,p2,p3} vector
        for ( int i = 0; i < dim; ++i)
            for ( int j = 0; j < dim; ++j)
                pvec[i][dim - 1 - j] = ( i == j ? 1.0 : 0.0);
        pvec[0][dim - 1] = t0 + momentind * deltat;

        // loop over elements intersected by the plane realted to a given time moment
        // here, elno = index in elpartition[momentind]
        for ( int elno = 0; elno < elpartition[momentind].size(); ++elno)
        //for ( int elno = 0; elno < 2; ++elno)
        {
            vector<int> tempvec;             // vertex indices for the cell of the slice mesh
            tempvec.reserve(6);
            vector<vector<double>> cellpnts; //points of the cell of the slice mesh
            cellpnts.reserve(6);

            vector<double> vertvalues;          // values of the grid function at the nodes of the slice cell

            // true mesh element index
            elind = elpartition[momentind][elno];
            el = mesh->GetElement(elind);

            if (verbose)
                cout << "Element: " << elind << endl;

            // computing number of intersection points, indices and coordinates for
            // local slice cell vertexes (cellpnts and tempvec)  and adding new intersection
            // points and changing edges markers for a given element elind
            // and plane defined by pvec
            int nip;
            //mesh->computeSliceCell (elind, pvec, ipoints, edgemarkers, cellpnts, tempvec, nip, vertex_count);

            computeSliceCellValues (elind, pvec, ipoints, edgemarkers, cellpnts, tempvec, nip, vertex_count, vertvalues);

            if ( (dim == 4 && (nip != 4 && nip != 6)) || (dim == 3 && (nip != 3 && nip != 4)) )
                cout << "Strange nip =  " << nip << " for elind = " << elind << ", time = " << t0 + momentind * deltat << endl;
            else
            {
                if (nip == 4) // tetrahedron in 3d or quadrilateral in 2d
                    if (dim == 4)
                        celltypes[momentind].push_back(VTKTETRAHEDRON);
                    else // dim == 3
                        celltypes[momentind].push_back(VTKQUADRIL);
                else if (nip == 6) // prism
                    celltypes[momentind].push_back(VTKWEDGE);
                else // nip == 3 = triangle
                    celltypes[momentind].push_back(VTKTRIANGLE);

                cellstructsize += nip + 1;

                elvrtindices[momentind].push_back(tempvec);

                cellvalue = 0.0;
                for ( int i = 0; i < nip; ++i)
                {
                    //cout << "vertval = " << vertvalues[i] << endl;
                    cellvalue += vertvalues[i];
                }
                cellvalue /= nip * 1.0;

                if (verbose)
                    cout << "cellvalue = " << cellvalue << endl;

                //cellvertvalues[momentind].push_back(vertvalues);
                cellvalues[momentind].push_back(cellvalue);

                // special reordering of cell vertices, required for the wedge,
                // tetrahedron and quadrilateral cells
                reorder_cellvertices (dim, nip, cellpnts, elvrtindices[momentind].back());

                if (verbose)
                    cout << "nip for the element = " << nip << endl;
            }

        } // end of loop over elements for a given time moment

        // intermediate output
        std::stringstream fname;
        fname << "slicegridfunc_"<< dim - 1 << "d_myid_" << myid << "_moment_" << momentind << ".vtk";
        //outputSliceGridFuncVTK (fname, ipoints, celltypes[momentind], cellstructsize, elvrtindices[momentind], cellvertvalues[momentind]);
        outputSliceGridFuncVTK (fname, ipoints, celltypes[momentind], cellstructsize, elvrtindices[momentind], cellvalues[momentind], forvideo);


    } //end of loop over time moments

    // if not deleted here, gets segfault for more than two parallel refinements afterwards, but this is for GridFunction
    //delete mesh->edge_vertex;
    //mesh->edge_vertex = NULL;

    //

    return;
}


} // end of adding functions definitions to mfem namespace


int printArr2DInt (Array2D<int>* arrayint)
{
    for ( int i = 0; i < arrayint->NumRows(); ++i )
    {
        for ( int j = 0; j < arrayint->NumCols(); ++j)
            std::cout << (*arrayint)(i,j) << " ";
         std::cout << std::endl;
    }


    return 0;
}

int setzero(Array2D<int>* arrayint)
{
    for ( int i = 0; i < arrayint->NumRows(); ++i )
        for ( int j = 0; j < arrayint->NumCols(); ++j)
            (*arrayint)(i,j) = 0;
    return 0;
}

int printDouble2D( double * arr, int dim1, int dim2)
{
    for ( int i = 0; i < dim1; i++)
    {
        for ( int j = 0; j < dim2; j++)
            std::cout << arr[i*dim2 + j] << " ";
         std::cout << std::endl;
    }
    return 0;
}

int printInt2D( int * arr, int dim1, int dim2)
{
    for ( int i = 0; i < dim1; i++)
    {
        for ( int j = 0; j < dim2; j++)
            std::cout << arr[i*dim2 + j] << " ";
         std::cout << std::endl;
    }
    return 0;
}

#ifdef WITH_QHULL

/*-------------------------------------------------
-print_summary(qh)
*/
void print_summary(qhT *qh) {
  facetT *facet;
  int k;

  printf("\n%d vertices and %d facets with normals:\n",
                 qh->num_vertices, qh->num_facets);
  FORALLfacets {
    for (k=0; k < qh->hull_dim; k++)
      printf("%6.2g ", facet->normal[k]);
    printf("\n");
  }
}

void qh_fprintf(qhT *qh, FILE *fp, int msgcode, const char *fmt, ... ) {
    va_list args;

    if (!fp) {
        if(!qh){
            qh_fprintf_stderr(6241, "userprintf_r.c: fp and qh not defined for qh_fprintf '%s'", fmt);
            qh_exit(qhmem_ERRqhull);  /* can not use qh_errexit() */
        }
        /* could use qh->qhmem.ferr, but probably better to be cautious */
        qh_fprintf_stderr(6232, "Qhull internal error (userprintf_r.c): fp is 0.  Wrong qh_fprintf called.\n");
        qh_errexit(qh, 6232, NULL, NULL);
    }
    va_start(args, fmt);
    if (qh && qh->ANNOTATEoutput) {
      fprintf(fp, "[QH%.4d]", msgcode);
    }else if (msgcode >= MSG_ERROR && msgcode < MSG_STDERR ) {
      fprintf(fp, "QH%.4d ", msgcode);
    }
    vfprintf(fp, fmt, args);
    va_end(args);

    /* Place debugging traps here. Use with option 'Tn' */

} /* qh_fprintf */

/*--------------------------------------------------
-makePrism- set points for dim Delaunay triangulation of 3D prism
  with 2 x dim points.
notes:
only 3D here!
*/
void makePrism(qhT *qh, coordT *points, int numpoints, int dim) {
  if ( dim != 3 )
  {
      std::cerr << " makePrism() does not work for dim = " << dim << " (only for dim = 3)" << std::endl;
      return;
  }
  if ( numpoints != 6 )
  {
      std::cerr << "Wrong numpoints in makePrism" << endl;
  }
  int j,k;
  coordT *point, realr;

  for (j=0; j<numpoints; j++) {
    point= points + j*dim;
    if (j == 0)
    {
        point[0] = 0.0;
        point[1] = 0.0;
        point[2] = 0.0;
    }
    if (j == 1)
    {
        point[0] = 1.0;
        point[1] = 0.0;
        point[2] = 0.0;
    }
    if (j == 2)
    {
        point[0] = 0.0;
        point[1] = 1.0;
        point[2] = 0.0;
    }
    if (j == 3)
    {
        point[0] = 0.0;
        point[1] = 0.0;
        point[2] = 3.0;
    }
    if (j == 4)
    {
        point[0] = 1.0;
        point[1] = 0.0;
        point[2] = 2.9;
    }
    if (j == 5)
    {
        point[0] = 0.0;
        point[1] = 1.0;
        point[2] = 3.1;
    }
  } // loop over points
} /*.makePrism.*/

/*--------------------------------------------------
-makeOrthotope - set points for dim Delaunay triangulation of dim-dimensional orthotope
  with 2 x (dim + 1) points.
notes:
With joggling the base coordinates
*/
void makeOrthotope(qhT *qh, coordT *points, int numpoints, int dim) {
  if ( numpoints != 1 << dim )
  {
      std::cerr << "Wrong numpoints in makeOrthotope" << endl;
  }

  //cout << "numpoints = " << numpoints << endl;
  int j,k;
  coordT *point, realr;

  if ( dim == 3)
  {
      for (j=0; j<numpoints; j++) {
        point= points + j*dim;
        if (j == 0)
        {
            point[0] = 0.0;
            point[1] = 0.0;
            point[2] = 0.0;
        }
        if (j == 1)
        {
            point[0] = 1.0;
            point[1] = 0.0;
            point[2] = 0.0;
        }
        if (j == 2)
        {
            point[0] = 0.0;
            point[1] = 1.0;
            point[2] = 0.0;
        }
        if (j == 3)
        {
            point[0] = 1.0;
            point[1] = 1.0;
            point[2] = 0.0;
        }
        if (j == 4)
        {
            point[0] = 0.0;
            point[1] = 0.0;
            point[2] = 3.0;
        }
        if (j == 5)
        {
            point[0] = 1.0;
            point[1] = 0.0;
            point[2] = 3.0;
        }
        if (j == 6)
        {
            point[0] = 0.0;
            point[1] = 1.0;
            point[2] = 3.0;
        }
        if (j == 7)
        {
            point[0] = 1.0;
            point[1] = 1.0;
            point[2] = 3.0;
        }

        for ( int coord = 0; coord < dim; ++coord)
            point[coord] += 1.0e-2 * coord * j;
      } // loop over points
  }

  if ( dim == 4)
  {
      for (j=0; j<numpoints; j++) {
        point= points + j*dim;
        if (j == 0)
        {
            point[0] = 0.0;
            point[1] = 0.0;
            point[2] = 0.0;
            point[3] = 0.0;
        }
        if (j == 1)
        {
            point[0] = 1.0;
            point[1] = 0.0;
            point[2] = 0.0;
            point[3] = 0.0;
        }
        if (j == 2)
        {
            point[0] = 0.0;
            point[1] = 1.0;
            point[2] = 0.0;
            point[3] = 0.0;
        }
        if (j == 3)
        {
            point[0] = 1.0;
            point[1] = 1.0;
            point[2] = 0.0;
            point[3] = 0.0;
        }
        if (j == 4)
        {
            point[0] = 0.0;
            point[1] = 0.0;
            point[2] = 1.0;
            point[3] = 0.0;
        }
        if (j == 5)
        {
            point[0] = 1.0;
            point[1] = 0.0;
            point[2] = 1.0;
            point[3] = 0.0;
        }
        if (j == 6)
        {
            point[0] = 0.0;
            point[1] = 1.0;
            point[2] = 1.0;
            point[3] = 0.0;
        }
        if (j == 7)
        {
            point[0] = 1.0;
            point[1] = 1.0;
            point[2] = 1.0;
            point[3] = 0.0;
        }
        if (j == 8)
        {
            point[0] = 0.0;
            point[1] = 0.0;
            point[2] = 0.0;
            point[3] = 3.0;
        }
        if (j == 9)
        {
            point[0] = 1.0;
            point[1] = 0.0;
            point[2] = 0.0;
            point[3] = 3.0;
        }
        if (j == 10)
        {
            point[0] = 0.0;
            point[1] = 1.0;
            point[2] = 0.0;
            point[3] = 3.0;
        }
        if (j == 11)
        {
            point[0] = 1.0;
            point[1] = 1.0;
            point[2] = 0.0;
            point[3] = 3.0;
        }
        if (j == 12)
        {
            point[0] = 0.0;
            point[1] = 0.0;
            point[2] = 1.0;
            point[3] = 3.0;
        }
        if (j == 13)
        {
            point[0] = 1.0;
            point[1] = 0.0;
            point[2] = 1.0;
            point[3] = 3.0;
        }
        if (j == 14)
        {
            point[0] = 0.0;
            point[1] = 1.0;
            point[2] = 1.0;
            point[3] = 3.0;
        }
        if (j == 15)
        {
            point[0] = 1.0;
            point[1] = 1.0;
            point[2] = 1.0;
            point[3] = 3.0;
        }

        for ( int coord = 0; coord < dim; ++coord)
            point[coord] += 1.0e-2 * coord * j;

      } // loop over points
  }


} /*.makeOrthotope.*/

// works only for a 3D or 4D prism (see def. of numpoints, etc.)
// volumetol should be large enough to eliminate zero-volume tetrahedrons
// and not too small to keep the proper ones.
// basically it should be about tau * h^n if used for the space-time mesh
// set outfile = NULL to have no output
int qhull_wrapper(int * simplices, qhT * qh, double * points, int dim, double volumetol, char * flags)
{
    if (dim != 4 && dim != 3)
    {
        cout << "Case dim = " << dim << " is not supported by qhull_wrapper" << endl;
        return -1;
    }
    int numpoints = dim * 2;  /* number of points */
    boolT ismalloc= False;    /* True if qhull should free points in qh_freeqhull() or reallocation */
    FILE *outfile= NULL;      /* output from qh_produce_output() \
                                 use NULL to skip qh_produce_output() */
    FILE *errfile= stderr;    /* error messages from qhull code */
    int exitcode;             /* 0 if no error from qhull */
    facetT *facet;            /* set by FORALLfacets */
    int curlong, totlong;     /* memory remaining after qh_memfreeshort */
    int i;

    //QHULL_LIB_CHECK

    qh_zero(qh, errfile);

    //printf( "\ncompute %d-d Delaunay triangulation for my prism \n", dim);
    //sprintf(flags, "qhull QJ s i d Qbb");
    //numpoints = SIZEprism;
    //makePrism(qh, points, numpoints, dim, (int)time(NULL));
    //for (i=numpoints; i--; )
      //rows[i]= points+dim*i;
    //qh_printmatrix(qh, outfile, "input", rows, numpoints, dim);
    exitcode= qh_new_qhull(qh, dim, numpoints, points, ismalloc,
                        flags, outfile, errfile);

    zero_intinit (simplices, dim*(dim+1));

    if (!exitcode) {                  /* if no error */
      /* 'qh->facet_list' contains the convex hull */
      /* If you want a Voronoi diagram ('v') and do not request output (i.e., outfile=NULL),
         call qh_setvoronoi_all() after qh_new_qhull(). */
      //print_summary(qh);
      //qh_printfacet3vertex(qh, stdout, facet, qh_PRINToff);
      //qh_printfacetNvertex_simplicial(qh, qh->fout, qh->facet_list, qh_PRINToff);
      //qh_printfacets(qh, qh->fout, qh->PRINTout[i], qh->facet_list, NULL, !qh_ALL);
      //qh_printsummary(qh, qh->ferr);

      facetT *facet, **facetp;
      setT *vertices;
      vertexT *vertex, **vertexp;

      int temp[dim+1];

      DenseMatrix Volume;
      Volume.SetSize(dim);

      int count = 0;
      FORALLfacet_(qh->facet_list)
      {
          if (facet->good)
          {
              int count2 = 0;

              FOREACHvertexreverse12_(facet->vertices)

              {
                  //qh_fprintf(qh, fp, 9131, "%d ", qh_pointid(qh, vertex->point));

                  //fprintf(qh->fout, "%d ", qh_pointid(qh, vertex->point));
                  //fprintf(qh->fout, "\n ");
                  temp[count2] = qh_pointid(qh, vertex->point);
                  //int pointid = qh_pointid(qh, vertex->point);

                  ++count2;
              }

              double volumesq = 0.0;

              double * pointss[dim + 1];
              for ( int i = 0; i < dim + 1; ++i)
                  pointss[i] = points + temp[i] * dim;

              for ( int i = 0; i < dim; ++i)
              {
                  for ( int j = 0; j < dim; ++j)
                  {
                      Volume.Elem(i,j) = pointss[i + 1][j] - pointss[0][j];
                  }
              }

              double volume = Volume.Det() / factorial(dim);
              //double volume = determinant4x4 ( Volume );

              volumesq = volume * volume;

              if ( fabs(sqrt(volumesq)) > volumetol )
              {
                  for ( int i = 0; i < count2; i++ )
                      simplices[count*(dim + 1) + i] = temp[i];
                  ++count;
              }
              else
              {
                  std::cout << "sliver pentatop rejected" << endl;
                  std::cout << "volume^2 = " << volumesq << endl;
              }
              //std::cout << "volume^2 = " << volumesq << endl;

          }// if facet->good
      } // loop over all facets


      /*
      std::cout << "Now final " << count << " simplices (in qhull wrapper) are:" << endl;
      for (int i = 0; i < dim; i++ ) // or count instead of dim if debugging
      {
          std::cout << "Tetrahedron " << i << ": ";
          std::cout << "vert indices: " << endl;
          for ( int j = 0; j < dim  +1; j++ )
          {
              std::cout << simplices[i*(dim + 1) + j] << " ";
          }
          std::cout << endl;
      }
      */

      qh->NOerrexit= True;
    }

    return 0;
}

#endif

// M and N are two d-dimensional points 9double * arrays with their coordinates
__inline__ double dist( double * M, double * N , int d)
{
    double res = 0.0;
    for ( int i = 0; i < d; ++i )
        res += (M[i] - N[i])*(M[i] - N[i]);
    return sqrt(res);
}

__inline__ void zero_intinit (int *arr, int size)
{
    for ( int i = 0; i < size; ++i )
        arr[i] = 0;
}

__inline__ void zero_doubleinit (double *arr, int size)
{
    for ( int i = 0; i < size; ++i )
        arr[i] = 0.0;
}

int factorial(int n)
{
  return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

// simple algorithm which computes sign of a given permutatation
// for now, this function is applied to permutations of size 3
// so there is no sense in implementing anything more complicated
// the sign is defined so that it is 1 for the loop of length = size
int permutation_sign( int * permutation, int size)
{
    int res = 0;
    int temp[size]; //visited or not
    for ( int i = 0; i < size; ++i)
        temp[i] = -1;

    int pos = 0;
    while ( pos < size )
    {
        if (temp[pos] == -1) // if element is unvisited
        {
            int cycle_len = 1;

            //computing cycle length which starts with unvisited element
            int k = pos;
            while (permutation[k] != pos )
            {
                temp[permutation[k]] = 1;
                k = permutation[k];
                cycle_len++;
            }
            //cout << "pos = " << pos << endl;
            //cout << "cycle of len " << cycle_len << " was found there" << endl;

            res += (cycle_len-1)%2;

            temp[pos] = 1;
        }

        pos++;
    }

    if (res % 2 == 0)
        return 1;
    else
        return -1;
}

// Actually, in mfem 4d case of 4x4 matrix determinant is implemented, but
// not in mfem. Now this function is not used.
double determinant4x4 ( DenseMatrix Mat)
{
    double det = 0;
    double subdet3;

    subdet3 = ( \
                Mat(1,1) * Mat(2,2) * Mat(3,3) + \
                Mat(2,1) * Mat(3,2) * Mat(1,3) + \
                Mat(1,2) * Mat(2,3) * Mat(3,1) - \
                Mat(3,1) * Mat(2,2) * Mat(1,3) - \
                Mat(1,1) * Mat(2,3) * Mat(3,2) - \
                Mat(2,1) * Mat(1,2) * Mat(3,3));
    det += Mat(0,0) * subdet3;
    subdet3 = ( \
                Mat(1,0) * Mat(2,2) * Mat(3,3) + \
                Mat(2,0) * Mat(3,2) * Mat(1,3) + \
                Mat(1,2) * Mat(2,3) * Mat(3,0) - \
                Mat(3,0) * Mat(2,2) * Mat(1,3) - \
                Mat(2,0) * Mat(1,2) * Mat(3,3) - \
                Mat(3,2) * Mat(2,3) * Mat(1,0));
    det -= Mat(0,1) * subdet3;
    subdet3 = ( \
                Mat(1,0) * Mat(2,1) * Mat(3,3) + \
                Mat(2,0) * Mat(3,1) * Mat(1,3) + \
                Mat(1,1) * Mat(2,3) * Mat(3,0) - \
                Mat(3,0) * Mat(2,1) * Mat(1,3) - \
                Mat(2,0) * Mat(1,1) * Mat(3,3) - \
                Mat(3,1) * Mat(2,3) * Mat(1,0));
    det += Mat(0,2) * subdet3;
    subdet3 = (\
                Mat(1,0) * Mat(2,1) * Mat(3,2) + \
                Mat(2,0) * Mat(3,1) * Mat(1,2) + \
                Mat(1,1) * Mat(2,2) * Mat(3,0) - \
                Mat(3,0) * Mat(2,1) * Mat(1,2) - \
                Mat(2,0) * Mat(1,1) * Mat(3,2) - \
                Mat(3,1) * Mat(2,2) * Mat(1,0));
    det -= Mat(0,3) * subdet3;

    return det;
}



