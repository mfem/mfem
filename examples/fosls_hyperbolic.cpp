//                                MFEM(with 4D elements) FOSLS (no constraint!) for 3D/4D hyperbolic equation
//                                  with mesh generator and visualization
//
// Description:  This example code solves a simple 3D/4D hyperbolic problem over [0,1]^3(4)
//               corresponding to the saddle point system
//                                  sigma_1 = S * b
//							 		sigma_2 - S        = 0
//                                  div_(x,t) sigma    = f
//                       with b = vector function (~velocity),
//						 NO boundary conditions (which work only in case when b * n = 0 pointwise at the domain space boundary)
//						 and initial condition:
//                                  S(x,0)             = 0
//               Here, we use a given exact solution and compute the corresponding r.h.s.
//               Discretization: Raviart-Thomas finite elements (sigma) and discontinuous L2 elements (S).
//               Scalar unknown S is eliminated using weak form of
//                               (b, 1)^T (b,1) S(x,t) = (b,1)^T sigma
//               tested against discontinuous L2 elements.
//               Final form of the problem is:
//                                 (Ktilda sigma, phi) = (f, phi) \forall phi from RT,
//               where Ktilda is symmetric semi-definite and hides hyperbolicity inside.
//               Optional use of time weight exp(-t/eps).
//
//               Solver: MINRES preconditioned by boomerAMG or ADS (either for standard mass matrix or weighted)
//               For weighted mass matrix ADS sometimes produces nans maybe due to the singular nature of diff. operator.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>

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
      //std::cerr << "elvect = " << elvect << endl;
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

// A set of test solutions for different domains with different dimension (3 or 4)
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
            if ( numsol == 1003 && dim == 3 )
                return true;
            if ( numsol == 1004 && dim == 4 )
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
            std::cerr << "Inconsistent dim and numsol" << std::endl << std::flush;
        else
        {
            if (numsol == 0)
            {
                //std::std::cerr << "The domain is rectangular or cubic, velocity does not"
                             //" satisfy divergence condition" << std::endl << std::flush;
                SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_gradx, &bFun_ex, &bFundiv_ex, &cas_weight>();
                //SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_gradx, &bFunCube3D_ex, &bFunCube3Ddiv_ex, &cas_weight>();
            }
            if (numsol == 1)
            {
                //std::std::cerr << "The domain must be a cylinder over a circle" << std::endl << std::flush;
                SetTestCoeffs<&uFun2_ex, &uFun2_ex_dt, &uFun2_ex_gradx, &bFun2_ex, &bFun2div_ex, &cas_weight>();
            }
            if (numsol == 100)
            {
                //std::std::cerr << "The domain must be a cylinder over a unit square" << std::endl << std::flush;
                SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &cas_weight>();
            }
            if (numsol == 200)
            {
                //std::std::cerr << "The domain must be a cylinder over a unit circle" << std::endl << std::flush;
                SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_gradx, &bFun2_ex, &bFun2div_ex, &cas_weight>();
            }
            if (numsol == 2)
            {
                //std::std::cerr << "The domain must be a cylinder over a 3D cube, velocity does not"
                             //" satisfy divergence condition" << std::endl << std::flush;
                SetTestCoeffs<&uFun3_ex, &uFun3_ex_dt, &uFun3_ex_gradx, &bFun_ex, &bFundiv_ex, &cas_weight>();
            }
            if (numsol == 3)
            {
                //std::std::cerr << "The domain must be a cylinder over a circle" << std::endl << std::flush;
                SetTestCoeffs<&uFun4_ex, &uFun4_ex_dt, &uFun4_ex_gradx, &bFun2_ex, &bFun2div_ex, &cas_weight>();
            }
            if (numsol == 4) // no exact solution in fact, ~ unsuccessfully trying to get a picture from the report
            {
                //std::std::cerr << "The domain must be a cylinder over a circle" << std::endl << std::flush;
                //std::std::cerr << "Using new interface \n";
                SetTestCoeffs<&uFun5_ex, &uFun5_ex_dt, &uFun5_ex_gradx, &bFun2_ex, &bFun2div_ex, &cas_weight>();
            }
            if (numsol == 44) // no exact solution in fact, ~ unsuccessfully trying to get a picture from the report
            {
                //std::std::cerr << "The domain must be a cylinder over a circle" << std::endl << std::flush;
                //std::std::cerr << "Using new interface \n";
                SetTestCoeffs<&uFun6_ex, &uFun6_ex_dt, &uFun6_ex_gradx, &bFun2_ex, &bFun2div_ex, &cas_weight>();
            }
            if (numsol == 5)
            {
                //std::std::cerr << "The domain must be a cylinder over a square" << std::endl << std::flush;
                SetTestCoeffs<&uFun4_ex, &uFun4_ex_dt, &uFun4_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &cas_weight>();
            }
            if (numsol == 1003)
            {
                //std::std::cerr << "The domain must be a cylinder over a square" << std::endl << std::flush;
                SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex, &cas_weight>();
            }
            if (numsol == 1004)
            {
                //std::std::cerr << "The domain must be a cylinder over a cube" << std::endl << std::flush;
                SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_gradx, &bFunCube3D_ex, &bFunCube3Ddiv_ex, &cas_weight>();
            }
            if (numsol == 33)
            {
                //std::std::cerr << "The domain must be a cylinder over a sphere" << std::endl << std::flush;
                SetTestCoeffs<&uFun33_ex, &uFun33_ex_dt, &uFun33_ex_gradx, &bFunSphere3D_ex, &bFunSphere3Ddiv_ex, &cas_weight>();
            }
            if (numsol == 444) // no exact solution in fact, ~ unsuccessfully trying to get something beauitiful
            {
                //std::std::cerr << "The domain must be a cylinder over a sphere" << std::endl << std::flush;
                SetTestCoeffs<&uFun66_ex, &uFun66_ex_dt, &uFun66_ex_gradx, &bFunSphere3D_ex, &bFunSphere3Ddiv_ex, &cas_weight>();
            }
            if (numsol == 55)
            {
                //std::std::cerr << "The domain must be a cylinder over a cube" << std::endl << std::flush;
                SetTestCoeffs<&uFun33_ex, &uFun33_ex_dt, &uFun33_ex_gradx, &bFunCube3D_ex, &bFunCube3Ddiv_ex, &cas_weight>();
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

    int nDimensions     = 3;
    int numsol          = 1003;
    double epsilon      = -0.01; // no time weight if negative

    int ser_ref_levels  = 0;
    int par_ref_levels  = 1;

    int generate_frombase   = 1;
    int Nsteps              = 4;
    double tau              = 0.25;
    int generate_parallel   = generate_frombase * 1;
    int whichparallel       = generate_parallel * 2;
    int bnd_method          = 1;
    int local_method        = 2;

    bool with_divdiv = true;    //unlike CFOSLS, here it should be always true.
    bool hybridization = true; //with hybridization the code is not working currently

    // solver options
    int prec_option = 3; //defines whether to use preconditioner or not, and which one

    //const char *mesh_file = "../build3/meshes/cube_3d_moderate.mesh";
    //const char *mesh_file = "../build3/meshes/square_2d_moderate.mesh";

    //const char *mesh_file = "../build3/meshes/cube4d.MFEM";
    const char *mesh_file = "dsadsad";
    //const char *mesh_file = "../build3/pmesh_2_mwe_0.mesh";
    //const char *mesh_file = "../build3/meshes/orthotope3D_moderate.mesh";
    //const char *mesh_file = "../build3/meshes/sphere3D_0.1to0.2.mesh";
    //const char * mesh_file = "../build3/meshes/orthotope3D_fine.mesh";

    //const char * meshbase_file = "../build3/meshes/sphere3D_0.1to0.2.mesh";
    //const char * meshbase_file = "../build3/meshes/sphere3D_0.05to0.1.mesh";
    //const char * meshbase_file = "../build3/meshes/sphere3D_veryfine.mesh";
    //const char * meshbase_file = "../build3/meshes/orthotope3D_moderate.mesh";
    //const char * meshbase_file = "./data/orthotope3D_fine.mesh";
    const char * meshbase_file = "../data/square_2d_moderate.mesh";
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
    args.AddOption(&hybridization, "-hybr", "--hybr", "-no-hybr",
                   "--no-hybr",
                   "Enable or disable MFEM's hybridization.");
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
          args.PrintUsage(std::cerr);
       }
       MPI_Finalize();
       return 1;
    }
    if (verbose)
    {
       args.PrintOptions(std::cerr);
    }

    if (verbose)
        std::cerr << "Number of mpi processes: " << num_procs << endl << flush;

    bool with_prec = true;
    bool prec_is_ADS = true;
    bool ADS_is_for_I = false;

    switch (prec_option)
    {
    case 1: // ADS for weighted mass matrix
        with_prec = true;
        prec_is_ADS = true;
        ADS_is_for_I = false;
        break;
    case 2: // ADS for identity
        with_prec = true;
        prec_is_ADS = true;
        ADS_is_for_I = true;
        break;
    case 3: // boomerAMG
        with_prec = true;
        prec_is_ADS = false;
        break;
    default: // no preconditioner (default)
        with_prec = false;
        break;
    }

    if (verbose)
    {
        std::cerr << "with_prec = " << with_prec << endl;
        std::cerr << "prec_is_ADS = " << prec_is_ADS << endl;
        std::cerr << "ADS_is_for_I = " << ADS_is_for_I << endl;
        std::cerr << flush;
    }

    if ((prec_option == 1 || prec_option == 2) && nDimensions == 4)
    {
        if (verbose)
            std::cerr << "ADS cannot be used in 4D" << endl;
        MPI_Finalize();
        return -1;
    }


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
                std::cerr << "Creating a " << nDimensions << "d mesh from a " <<
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

            if (meshbase->Dimension() != nDimensions - 1)
            {
                if (verbose)
                    std::cerr << "Dimension of the meshbase != dimension of problem - 1: Mismatch!" << endl << flush;
                MPI_Finalize();
                return 0;
            }

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
                            std::cerr << "Not implemented for 2D->3D. Use parallel version2"
                                    " instead" << endl << flush;
                        MPI_Finalize();
                        return 0;
                    }
                    else // nDimensions == 4
                    {
                        mesh = new Mesh( comm, *pmeshbase, tau, Nsteps, bnd_method, local_method);
                        if ( verbose )
                            std::cerr << "Success: ParMesh is created by deprecated method"
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
                        std::cerr << "Starting parallel \"" << nDimensions-1 << "D->"
                             << nDimensions <<"D\" mesh generator" << endl;

                    pmesh = make_shared<ParMesh>( comm, *pmeshbase, tau, Nsteps,
                                                  bnd_method, local_method);

                    if (verbose)
                        std::cerr << "Success: ParMesh created" << endl << flush;
                    MPI_Barrier(comm);
                }

                chrono.Stop();
                if (verbose && whichparallel == 2)
                    std::cerr << "Timing: Space-time mesh extension done in parallel in "
                              << chrono.RealTime() << " seconds.\n" << endl << flush;
                delete pmeshbase;
            }
            else // serial version
            {
                if (verbose)
                    std::cerr << "Starting serial \"" << nDimensions-1 << "D->"
                         << nDimensions <<"D\" mesh generator" << endl;
                mesh = new Mesh( *meshbase, tau, Nsteps, bnd_method, local_method);
                if (verbose)
                    std::cerr << "Timing: Space-time mesh extension done in serial in "
                              << chrono.RealTime() << " seconds.\n" << endl << flush;
            }

            delete meshbase;

        }
        else // not generating from a lower dimensional mesh
        {
            if (verbose)
                std::cerr << "Reading a " << nDimensions << "d mesh from the file " << mesh_file << endl;
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
                if (mesh->Dimension() != nDimensions)
                {
                    if (verbose)
                        std::cerr << "Dimension of the mesh and dimension of the problem mismatch!" << endl << flush;
                    MPI_Finalize();
                    return 0;
                }
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

    if (mesh) // if only serial mesh was generated previously, parallel mesh is initialized here
    {
        // Checking that mesh is legal
        //if (myid == 0)
            //std::cerr << "Checking the mesh" << endl << flush;
        //mesh->MeshCheck();

        for (int l = 0; l < ser_ref_levels; l++)
            mesh->UniformRefinement();

        if ( verbose )
            std::cerr << "Creating parmesh(" << nDimensions <<
                    "d) from the serial mesh (" << nDimensions << "d)" << endl << flush;
        pmesh = make_shared<ParMesh>(comm, *mesh);
        delete mesh;
    }

    for (int l = 0; l < par_ref_levels; l++)
    {
       pmesh->UniformRefinement();
    }

    //if(dim==3) pmesh->ReorientTetMesh();

    pmesh->PrintInfo(std::cerr); if(verbose) std::cerr << endl;

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
           std::cerr << "RT: order 0 for 4D" << endl;
    }
    else
    {
        hdiv_coll = new RT_FECollection(feorder, dim);
        if (verbose)
            std::cerr << "RT: order " << feorder << " for 3D" << endl;
    }

    ParFiniteElementSpace *R_space = new ParFiniteElementSpace(pmesh.get(), hdiv_coll);

    HYPRE_Int dimR = R_space->GlobalTrueVSize();
    //HYPRE_Int dimW = W_space->GlobalTrueVSize();


    if (verbose)
    {
       std::cerr << "***********************************************************\n";
       std::cerr << "dim(R) = " << dimR << "\n";
       std::cerr << "***********************************************************\n";
    }

    // 7. Define the two BlockStructure of the problem.  block_offsets is used
    //    for Vector based on dof (like ParGridFunction or ParLinearForm),
    //    block_trueOffstes is used for Vector based on trueDof (HypreParVector
    //    for the rhs and solution of the linear system).  The offsets computed
    //    here are local to the processor.

    ParGridFunction * x(new ParGridFunction(R_space));
    *x = 0.0;

    // 8. Define the coefficients, analytical solution, and rhs of the PDE.
    ConstantCoefficient one(1.0);

    //MatrixFunctionCoefficient Ktilda( dim, Ktilda_ex );
    //FunctionCoefficient fcoeff(fFun);
    //FunctionCoefficient ucoeff(uFun_ex);
    //VectorFunctionCoefficient sigmacoeff(dim, sigmaFun_ex);

    Transport_test Mytest(nDimensions,numsol, epsilon);

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

    if (with_divdiv == true && verbose)
        std::cerr << "Bilinear form with div-div term" << endl;

    if (epsilon > 0 && verbose)
        std::cerr << "Using casuality weight with exp(-t/epsilon)" << endl;


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

    FiniteElementCollection *hfec = NULL;
    ParFiniteElementSpace *hfes = NULL;
    if (hybridization)
    {
        if (verbose)
            std::cerr << "Using mfem hybridization" << endl;
        hfec = new DG_Interface_FECollection(feorder, dim);
        hfes = new ParFiniteElementSpace(pmesh.get(), hfec);
        Ablock->EnableHybridization(hfes, new NormalTraceJumpIntegrator(),
                               ess_tdof_list);
    }
    Ablock->Assemble();

    HypreParMatrix Amat;
    Vector B, X;
    Ablock->FormLinearSystem(ess_tdof_list, *x, *fform, Amat, X, B);

    if (verbose)
        std::cerr<< "Final saddle point matrix assembled"<<endl << flush;
    MPI_Barrier(MPI_COMM_WORLD);

    // 12. Solve the linear system with MINRES.
    //     Check the norm of the unpreconditioned residual.

    Solver * prec;

    ParBilinearForm *AIblock(new ParBilinearForm(R_space));
    HypreParMatrix *AI;

    if (with_prec)
    {
        if (verbose)
             std::cerr << "Using a preconditioner:" << endl;
        if (!hybridization)
        {
            if (prec_is_ADS == true)
            {
                if (verbose)
                     std::cerr << "Using ADS as a preconditioner" << endl;

                if (nDimensions == 3)
                {
                    if (ADS_is_for_I == true)
                    {
                        if (verbose)
                            std::cerr << "Creating ADS for the Identity (not Ktilda!)" << endl;
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
                        prec = new HypreADS (Amat, R_space);

                }
                else
                {
                    if (verbose)
                        std::cerr << "ADS is not working in case dim = " << nDimensions << endl;
                    MPI_Finalize();
                    return 0;
                }
            }
            else
            {
                if (verbose)
                    std::cerr << "Using boomerAMG as a preconditioner" << endl;
                prec = new HypreBoomerAMG(Amat);
            }
        }
        else // for hybridization
        {
            if (verbose)
                std::cerr << "Using mfem hybridization combined with boomerAMG as a preconditioner" << endl;
            prec = new HypreBoomerAMG(Amat);
        }

    } // end of if with_prec == true

    chrono.Clear();
    chrono.Start();
    MINRESSolver solver(MPI_COMM_WORLD);
    solver.SetAbsTol(atol);
    solver.SetRelTol(rtol);
    solver.SetMaxIter(max_num_iter);
    solver.SetOperator(Amat);
    if (with_prec)
         solver.SetPreconditioner(*prec);
    solver.SetPrintLevel(0);
    solver.Mult(B, X);
    chrono.Stop();

    if (verbose)
    {
       if (solver.GetConverged())
          std::cerr << "MINRES converged in " << solver.GetNumIterations()
                    << " iterations with a residual norm of " << solver.GetFinalNorm() << ".\n";
       else
          std::cerr << "MINRES did not converge in " << solver.GetNumIterations()
                    << " iterations. Residual norm is " << solver.GetFinalNorm() << ".\n";
       std::cerr << "MINRES solver took " << chrono.RealTime() << "s. \n";
    }

    // 13. Extract the parallel grid function corresponding to the finite element
    //     approximation X. This is the local solution on each processor. Compute
    //     L2 error norms.

    Ablock->RecoverFEMSolution(X, *fform, *x);

    int order_quad = max(2, 2*feorder+1);
    const IntegrationRule *irs[Geometry::NumGeom];
    for (int i=0; i < Geometry::NumGeom; ++i)
    {
       irs[i] = &(IntRules.Get(i, order_quad));
    }

    // adding back the term from nonhomogeneous initial condition
    ParGridFunction *sigma_nonhomo = new ParGridFunction(R_space);
    sigma_nonhomo->ProjectCoefficient(*(Mytest.sigma_nonhomo));
    *x += *sigma_nonhomo;

    double err_sigma = x->ComputeL2Error(*(Mytest.sigma), irs);
    double norm_sigma = ComputeGlobalLpNorm(2, *(Mytest.sigma), *pmesh, irs);
    if (verbose)
        std::cerr << "local: err_sigma / norm_sigma = " << err_sigma / norm_sigma << endl;

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
        std::cerr << "global: err_sigma / norm_sigma = " << err_sigma_global / norm_sigma_global << endl;


    FiniteElementCollection *l2_coll;
    if ( dim == 4 )
    {
        l2_coll = new L2_FECollection(0, dim);
        if (verbose)
            std::cerr << "L2: order 0 for 4D" << endl;
    }
    else
    {
       l2_coll = new L2_FECollection(feorder, dim);
       if (verbose)
            std::cerr << "L2: order " << feorder << " for 3D" << endl;
    }
    ParFiniteElementSpace *W_space = new ParFiniteElementSpace(pmesh.get(), l2_coll);

    // Computing error for S (thus introducing finite element space for L2)

    BilinearForm *Cform(new BilinearForm(W_space));
    Cform->AddDomainIntegrator(new MassIntegrator(*(Mytest.bTb)));
    Cform->Assemble();
    Cform->Finalize();
    SparseMatrix C = Cform->SpMat();

    MixedBilinearForm *Bform(
                new MixedBilinearForm(R_space, W_space));
    Bform->AddDomainIntegrator(
                new VectorFEMassIntegrator(*(Mytest.conv)));

    Bform->Assemble();
    Bform->Finalize();
    SparseMatrix Bagain = Bform->SpMat();
    Vector bTsigma(C.Size());
    Bagain.Mult(*x,bTsigma);

    GridFunction S(W_space); S = 0.;
    GSSmoother Smooth(C);
    PCG(C, Smooth, bTsigma, S, 0, 5000, 1e-9, 1e-12);

    FunctionCoefficient Scoeff(*(Mytest.scalaru));
    double err_S  = S.ComputeL2Error(Scoeff, irs);
    double norm_S = ComputeGlobalLpNorm(2, Scoeff, *pmesh, irs);
    err_S *= err_S;
    double err_S_global;
    MPI_Reduce(&err_S, &err_S_global, 1, MPI_DOUBLE,
               MPI_SUM, 0, comm);
    err_S_global = std::sqrt(err_S_global);
    if (verbose)
    {
        std::cerr << "|| S_h - S_ex || / || S_ex || = " <<
                     err_S_global / norm_S << "\n";
    }

    delete W_space;
    delete l2_coll;

    delete Bform;
    delete Cform;

    // Visualization.

    ParGridFunction *sigma_exact = new ParGridFunction(R_space);
    sigma_exact->ProjectCoefficient(*(Mytest.sigma));
    //HypreParVector *sigma_exactvec = sigma_exact->ParallelAverage();

    if (visualization && nDimensions < 4)
    {
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

    // Free the used memory.
    delete fform;
    delete Ablock;
    delete R_space;
    delete hdiv_coll;

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
        std::cerr << "Error: nparams should be nonnegative" << endl;
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
