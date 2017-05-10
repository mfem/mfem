/*
 * Currently used for tests of solving heat equation in 4d without parelag combined
 * with parallel mesh generator
 *
*/


//                                MFEM CFOSLS Heat equation (+ mesh generator) solved by hypre
//
// Compile with: make
//
// Sample runs:  ./exHeatp4d -dim 3 or ./exHeatp4d -dim 4
//
// Description:  This example code solves a simple 4D  Heat problem over [0,1]^4
//               corresponding to the saddle point system
//                                  sigma_1 + grad u   = 0
//                                  sigma_2 - u        = 0
//                                  div_(x,t) sigma    = f
//                       with boundary conditions:
//                                   u(0,t)  = u(1,t)  = 0
//                                   u(x,0)            = 0
//               Here, we use a given exact solution
//                                  u(xt) = uFun_ex(xt)
//               and compute the corresponding r.h.s.
//               We discretize with Raviart-Thomas finite elements (sigma), continuous H1 elements (u) and
//		 discontinuous polynomials (mu) for the lagrange multiplier.
//               Solver: ~ hypre with a block-diagonal preconditioner with BoomerAMG
//

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


//********* NEW STUFF FOR 4D CFOSLS
//-----------------------
/// Integrator for (Q u, v) for VectorFiniteElements

class PAUVectorFEMassIntegrator: public BilinearFormIntegrator
{
private:
    Coefficient *Q;
    VectorCoefficient *VQ;
    MatrixCoefficient *MQ;
    void Init(Coefficient *q, VectorCoefficient *vq, MatrixCoefficient *mq)
    { Q = q; VQ = vq; MQ = mq; }

#ifndef MFEM_THREAD_SAFE
    Vector shape;
    Vector D;
    Vector trial_shape;
    Vector test_shape;//<<<<<<<
    DenseMatrix K;
    DenseMatrix test_vshape;
    DenseMatrix trial_vshape;
    DenseMatrix trial_dshape;//<<<<<<<<<<<<<<
    DenseMatrix test_dshape;//<<<<<<<<<<<<<<

#endif

public:
    PAUVectorFEMassIntegrator() { Init(NULL, NULL, NULL); }
    PAUVectorFEMassIntegrator(Coefficient *_q) { Init(_q, NULL, NULL); }
    PAUVectorFEMassIntegrator(Coefficient &q) { Init(&q, NULL, NULL); }
    PAUVectorFEMassIntegrator(VectorCoefficient *_vq) { Init(NULL, _vq, NULL); }
    PAUVectorFEMassIntegrator(VectorCoefficient &vq) { Init(NULL, &vq, NULL); }
    PAUVectorFEMassIntegrator(MatrixCoefficient *_mq) { Init(NULL, NULL, _mq); }
    PAUVectorFEMassIntegrator(MatrixCoefficient &mq) { Init(NULL, NULL, &mq); }

    virtual void AssembleElementMatrix(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
    virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                        const FiniteElement &test_fe,
                                        ElementTransformation &Trans,
                                        DenseMatrix &elmat);
};

//=-=-=-=--=-=-=-=-=-=-=-=-=
/// Integrator for (Q u, v) for VectorFiniteElements
class PAUVectorFEMassIntegrator2: public BilinearFormIntegrator
{
private:
    Coefficient *Q;
    VectorCoefficient *VQ;
    MatrixCoefficient *MQ;
    void Init(Coefficient *q, VectorCoefficient *vq, MatrixCoefficient *mq)
    { Q = q; VQ = vq; MQ = mq; }

#ifndef MFEM_THREAD_SAFE
    Vector shape;
    Vector D;
    Vector trial_shape;
    Vector test_shape;//<<<<<<<
    DenseMatrix K;
    DenseMatrix test_vshape;
    DenseMatrix trial_vshape;
    DenseMatrix trial_dshape;//<<<<<<<<<<<<<<
    DenseMatrix test_dshape;//<<<<<<<<<<<<<<
    DenseMatrix dshape;
    DenseMatrix dshapedxt;
    DenseMatrix invdfdx;

#endif

public:
    PAUVectorFEMassIntegrator2() { Init(NULL, NULL, NULL); }
    PAUVectorFEMassIntegrator2(Coefficient *_q) { Init(_q, NULL, NULL); }
    PAUVectorFEMassIntegrator2(Coefficient &q) { Init(&q, NULL, NULL); }
    PAUVectorFEMassIntegrator2(VectorCoefficient *_vq) { Init(NULL, _vq, NULL); }
    PAUVectorFEMassIntegrator2(VectorCoefficient &vq) { Init(NULL, &vq, NULL); }
    PAUVectorFEMassIntegrator2(MatrixCoefficient *_mq) { Init(NULL, NULL, _mq); }
    PAUVectorFEMassIntegrator2(MatrixCoefficient &mq) { Init(NULL, NULL, &mq); }

    virtual void AssembleElementMatrix(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
    virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                        const FiniteElement &test_fe,
                                        ElementTransformation &Trans,
                                        DenseMatrix &elmat);
};

//=-=-=-=-=-=-=-=-=-=-=-=-=-
void PAUVectorFEMassIntegrator::AssembleElementMatrix(
        const FiniteElement &el,
        ElementTransformation &Trans,
        DenseMatrix &elmat)
{}

void PAUVectorFEMassIntegrator::AssembleElementMatrix2(
        const FiniteElement &trial_fe, const FiniteElement &test_fe,
        ElementTransformation &Trans, DenseMatrix &elmat)
{
    // assume both test_fe and trial_fe are vector FE
    int dim  = test_fe.GetDim();
    int trial_dof = trial_fe.GetDof();
    int test_dof = test_fe.GetDof();
    double w;

    if (VQ || MQ) // || = or
        mfem_error("VectorFEMassIntegrator::AssembleElementMatrix2(...)\n"
                   "   is not implemented for vector/tensor permeability");

    DenseMatrix trial_dshapedxt(trial_dof,dim);
    DenseMatrix invdfdx(dim,dim);

#ifdef MFEM_THREAD_SAFE
    // DenseMatrix trial_vshape(trial_dof, dim);
    Vector trial_shape(trial_dof); //PAULI
    DenseMatrix trial_dshape(trial_dof,dim);
    DenseMatrix test_vshape(test_dof,dim);
#else
    //trial_vshape.SetSize(trial_dof, dim);
    trial_shape.SetSize(trial_dof); //PAULI
    trial_dshape.SetSize(trial_dof,dim); //Pauli
    test_vshape.SetSize(test_dof,dim);
#endif
    //elmat.SetSize (test_dof, trial_dof);
    elmat.SetSize (test_dof, trial_dof);

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        int order = (Trans.OrderW() + test_fe.GetOrder() + trial_fe.GetOrder());
        ir = &IntRules.Get(test_fe.GetGeomType(), order);
    }

    elmat = 0.0;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);

        trial_fe.CalcShape(ip, trial_shape);
        trial_fe.CalcDShape(ip, trial_dshape);

        Trans.SetIntPoint (&ip);
        test_fe.CalcVShape(Trans, test_vshape);

        w = ip.weight * Trans.Weight();
        CalcInverse(Trans.Jacobian(), invdfdx);
        Mult(trial_dshape, invdfdx, trial_dshapedxt);
        if (Q)
        {
            w *= Q -> Eval (Trans, ip);
        }

        for (int j = 0; j < test_dof; j++)
        {
            for (int k = 0; k < trial_dof; k++)
            {
                for (int d = 0; d < dim - 1; d++ )
                    elmat(j, k) += 1.0 * w * test_vshape(j, d) * trial_dshapedxt(k, d);
                elmat(j, k) -= w * test_vshape(j, dim - 1) * trial_shape(k);
            }
        }
    }
}

void PAUVectorFEMassIntegrator2::AssembleElementMatrix(
        const FiniteElement &el,
        ElementTransformation &Trans,
        DenseMatrix &elmat)
{
    int dof = el.GetDof();
    int dim  = el.GetDim();
    double w;

    if (VQ || MQ) // || = or
        mfem_error("VectorFEMassIntegrator::AssembleElementMatrix2(...)\n"
                   "   is not implemented for vector/tensor permeability");

#ifdef MFEM_THREAD_SAFE
    Vector shape(dof);
    DenseMatrix dshape(dof,dim);
    DenseMatrix dshapedxt(dof,dim);
    DenseMatrix invdfdx(dim,dim);
#else
    shape.SetSize(dof);
    dshape.SetSize(dof,dim);
    dshapedxt.SetSize(dof,dim);
    invdfdx.SetSize(dim,dim);
#endif
    //elmat.SetSize (test_dof, trial_dof);
    elmat.SetSize (dof, dof);
    elmat = 0.0;

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        int order = (Trans.OrderW() + el.GetOrder() + el.GetOrder());
        ir = &IntRules.Get(el.GetGeomType(), order);
    }

    elmat = 0.0;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);

        //chak Trans.SetIntPoint (&ip);

        el.CalcShape(ip, shape);
        el.CalcDShape(ip, dshape);

        Trans.SetIntPoint (&ip);
        CalcInverse(Trans.Jacobian(), invdfdx);
        w = ip.weight * Trans.Weight();
        Mult(dshape, invdfdx, dshapedxt);

        if (Q)
        {
            w *= Q -> Eval (Trans, ip);
        }

        for (int j = 0; j < dof; j++)
            for (int k = 0; k < dof; k++)
            {
                for (int d = 0; d < dim - 1; d++ )
                    elmat(j, k) +=  w * dshapedxt(j, d) * dshapedxt(k, d);
                elmat(j, k) +=  w * shape(j) * shape(k);
            }

    }
}

void PAUVectorFEMassIntegrator2::AssembleElementMatrix2(
        const FiniteElement &trial_fe, const FiniteElement &test_fe,
        ElementTransformation &Trans, DenseMatrix &elmat)
{}

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

void VectordivDomainLFIntegrator::AssembleRHSElementVect(
        const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
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
        // Chak: Looking at how MFEM assembles in VectorFEDivergenceIntegrator,
        // I think you dont need Tr.Weight() here I think this is because the RT
        // (or other vector FE) basis is scaled by the geometry of the mesh
        double val = Q.Eval(Tr, ip);

        add(elvect, ip.weight * val, divshape, elvect);
    }

}



// Define the analytical solution and forcing terms / boundary conditions
//double u0_function(const Vector &x);
double uFun_ex(const Vector & x); // Exact Solution
double uFun_ex_dt(const Vector & xt);
double uFun_ex_laplace(const Vector & xt);
void uFun_ex_gradx(const Vector& xt, Vector& gradx );

//double fFun(const Vector & x); // Source f
//void sigmaFun_ex (const Vector &x, Vector &u);

double uFun1_ex(const Vector & x); // Exact Solution
double uFun1_ex_dt(const Vector & xt);
double uFun1_ex_laplace(const Vector & xt);
void uFun1_ex_gradx(const Vector& xt, Vector& gradx );

//double fFun1(const Vector & x); // Source f
//void sigmaFun1_ex (const Vector &x, Vector &u);

int printArr2DInt (Array2D<int> *arrayint);
int setzero(Array2D<int>* arrayint);
int printDouble2D( double * arr, int dim1, int dim2);
int printInt2D( int * arr, int dim1, int dim2);

#ifdef WITH_QHULL
int qhull_wrapper(int * tetrahedrons, qhT * qh, double * points, int dim, double volumetol, char *flags);
void print_summary(qhT *qh);
void qh_fprintf(qhT *qh, FILE *fp, int msgcode, const char *fmt, ... );
void makePrism(qhT *qh, coordT *points, int numpoints, int dim, int seed);
#endif
__inline__ double dist( double * M, double * N , int d);
int factorial(int n);
int permutation_sign( int * permutation, int size);
double determinant4x4(DenseMatrix Mat);

__inline__ void zero_intinit ( int * arr, int size);


template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), double (*Slaplace)(const Vector & xt)> \
    double rhsideTemplate(const Vector& xt);

template<double (*S)(const Vector & xt)> \
    double SnonhomoTemplate(const Vector& xt);

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), double (*Slaplace)(const Vector & xt)> \
    double divsigmaTemplate(const Vector& xt);


template <double (*ufunc)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )>
    void sigmaTemplate(const Vector& xt, Vector& sigma);

template <double (*ufunc)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )>
    void sigmaNonHomoTemplate(const Vector& xt, Vector& sigma);


class Heat_test
{
protected:
    int dim;
    int numsol;

public:
    FunctionCoefficient * scalaru;             // S
    FunctionCoefficient * scalarSnonhomo;             // S(t=0)
    FunctionCoefficient * scalarf;             // = dS/dt - laplace S + laplace S(t=0) - what is used for solving
    FunctionCoefficient * scalardivsigma;      // = dS/dt - laplace S                  - what is used for computing error
    VectorFunctionCoefficient * sigma;
    VectorFunctionCoefficient * sigma_nonhomo; // to incorporate inhomogeneous boundary conditions, stores (conv *S0, S0) with S(t=0) = S0
public:
    Heat_test (int Dim, int NumSol);

    int GetDim() {return dim;}
    int GetNumSol() {return numsol;}
    int SetDim(int Dim) { dim = Dim;}
    int SetNumSol(int NumSol) { numsol = NumSol;}
    bool CheckTestConfig();

    ~Heat_test () {}
private:
    void SetScalarFun( double (*f)(const Vector & xt))
    { scalaru = new FunctionCoefficient(f);}

    template<double (*S)(const Vector & xt)> \
    void SetScalarSnonhomo()
    { scalarSnonhomo = new FunctionCoefficient(SnonhomoTemplate<S>);}

    template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), double (*Slaplace)(const Vector & xt)> \
    void SetRhandFun()
    { scalarf = new FunctionCoefficient(rhsideTemplate<S, dSdt, Slaplace>);}

    template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), double (*Slaplace)(const Vector & xt)> \
    void SetDivSigma()
    { scalardivsigma = new FunctionCoefficient(divsigmaTemplate<S, dSdt, Slaplace>);}

    //void SetRhandFun( double (*f)(const Vector & xt))
    //{ scalarf = new FunctionCoefficient(f);}


    template<double (*f1)(const Vector & xt), void(*f2)(const Vector & x, Vector & vec)> \
    void SetHdivFun()
    {
        sigma = new VectorFunctionCoefficient(dim, sigmaTemplate<f1,f2>);
    }

    //void SetHdivFun( void(*f)(const Vector & x, Vector & vec))
    //{ sigma = new VectorFunctionCoefficient(dim, f);}

    template<double (*f1)(const Vector & xt), void(*f2)(const Vector & x, Vector & vec)> \
    void SetInitCondVec()
    {
        sigma_nonhomo = new VectorFunctionCoefficient(dim, sigmaNonHomoTemplate<f1,f2>);
    }


    template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), double (*Slaplace)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx) > \
    void SetTestCoeffs ( );
};


template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), double (*Slaplace)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx) > \
void Heat_test::SetTestCoeffs ()
{
    SetScalarFun(S);
    SetScalarSnonhomo<S>();
    SetRhandFun<S, dSdt, Slaplace>();
    SetHdivFun<S,Sgradxvec>();
    SetInitCondVec<S,Sgradxvec>();
    SetDivSigma<S, dSdt, Slaplace>();
    return;
}


bool Heat_test::CheckTestConfig()
{
    if (dim == 4 || dim == 3)
    {
        if (numsol == 0 || numsol == 1)
            return true;
        return false;
    }
    else
        return false;

}

Heat_test::Heat_test (int Dim, int NumSol)
{
    dim = Dim;
    numsol = NumSol;

    if ( CheckTestConfig() == false )
        std::cout << "Inconsistent dim and numsol" << std::endl << std::flush;
    else
    {
        if (numsol == 0)
        {
            std::cout << "The domain should be either a unit rectangle or cube" << std::endl << std::flush;
            SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_laplace, &uFun_ex_gradx>();
            //SetScalarFun(&uFun_ex);
            //SetRhandFun(&fFun);
            //SetHdivFun(&sigmaFun_ex);
        }
        if (numsol == 1)
        {
            std::cout << "The domain should be either a unit rectangle or cube" << std::endl << std::flush;
            SetTestCoeffs<&uFun1_ex, &uFun1_ex_dt, &uFun1_ex_laplace, &uFun1_ex_gradx>();
            //SetScalarFun(&uFun1_ex);
            //SetRhandFun(&fFun1);
            //SetHdivFun(&sigmaFun1_ex);
        }
    }
}

int main(int argc, char *argv[])
{
    StopWatch chrono;

    // 1. Initialize MPI.
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    bool verbose = (myid == 0);
    bool visualization = 1;

    int nDimensions     = 3;
    int numsol          = 0;

    int ser_ref_levels  = 2;
    int par_ref_levels  = 1;
    int Nsteps          = 2;
    double tau          = 0.5;


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
    //const char * mesh_file = "./data/orthotope3D_fine.mesh";
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
    //const char * meshbase_file = "./data/orthotope3D_fine.mesh";
    //const char * meshbase_file = "../build3/meshes/square_2d_moderate.mesh";
    const char * meshbase_file = "../data/square_2d_fine.mesh";
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

    //DEFAULTED LINEAR SOLVER OPTIONS
    int print_iter = 0;
    int max_num_iter = 50000;
    double rtol = 1e-9;
    double atol = 1e-12;

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
                        if  (myid == 0)
                            cout << "Not implemented for 2D->3D. Use parallel version2"
                                    " instead" << endl << flush;
                        MPI_Finalize();
                        return 0;
                    }
                    else // nDimensions == 4
                    {
                        mesh = new Mesh( comm, *pmeshbase, tau, Nsteps, bnd_method, local_method);
                        if ( myid == 0)
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
                    if (myid == 0)
                        cout << "Starting parallel \"" << nDimensions-1 << "D->"
                             << nDimensions <<"D\" mesh generator" << endl;

                    pmesh = make_shared<ParMesh>( comm, *pmeshbase, tau, Nsteps,
                                                  bnd_method, local_method);

                    if ( myid == 0)
                        cout << "Success: ParMesh created" << endl << flush;
                    MPI_Barrier(comm);
                }

                chrono.Stop();
                if (myid == 0 && whichparallel == 2)
                    cout << "Timing: Space-time mesh extension done in parallel in "
                              << chrono.RealTime() << " seconds.\n" << endl << flush;
                delete pmeshbase;
            }
            else // serial version
            {
                if (myid == 0)
                    cout << "Starting serial \"" << nDimensions-1 << "D->"
                         << nDimensions <<"D\" mesh generator" << endl;
                mesh = new Mesh( *meshbase, tau, Nsteps, bnd_method, local_method);
                if ( myid == 0)
                    cout << "Timing: Space-time mesh extension done in serial in "
                              << chrono.RealTime() << " seconds.\n" << endl << flush;
            }

            delete meshbase;

        }
        else // not generating from a lower dimensional mesh
        {
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
        if (myid == 0)
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
        if (myid == 0)
            cout << "Checking the mesh" << endl << flush;
        mesh->MeshCheck(verbose);

        for (int l = 0; l < ser_ref_levels; l++)
            mesh->UniformRefinement();

        if ( verbose )
            cout << "Creating parmesh(" << nDimensions <<
                    "d) from the serial mesh (" << nDimensions << "d)" << endl << flush;
        pmesh = make_shared<ParMesh>(comm, *mesh);
        delete mesh;
    }

    /*
    bool verbose = (myid == 0);

    int ser_ref_levels  = 0;
    int par_ref_levels  = 2;
    int Nsteps          = 8;
    double tau          = 0.125;

    int generate_frombase   = 1;
    int generate_parallel   = generate_frombase * 1;
    int whichparallel       = generate_parallel * 2;
    int bnd_method          = 1;
    int local_method        = 2;


    int numsol = 0;

    //const char *mesh_file = "../build3/meshes/cube4d_low.MFEM";
    const char *mesh_file = "../build3/meshes/cube4d.MFEM";
    //const char *mesh_file = "../build3/pmesh_par2_ortho_n1.mesh";
    //const char * mesh_file = "../build3/meshes/orthotope3D_moderate.mesh";
    //const char * mesh_file = "../build3/meshes/orthotope3D_moderate.mesh";
    //const char *mesh_file = "dsadsad";
    //const char *mesh_file = "../build3/pmesh_tempmesh_par2_np2.mesh";
    //const char *mesh_file = "../build3/pmesh_ser.mesh";
    //const char *mesh_file = "../build3/mesh4d_from_parmesh3d_afterLoad_np2.mesh";
    //const char *mesh_file = "../build3/meshes/orthotope3D_moderate.mesh";
    //const char *mesh_file = "../build3/meshes/sphere3D_0.1to0.2.mesh";
    //const char * mesh_file = "../build3/meshes/orthotope3D_fine.mesh";

    //const char * meshbase_file = "../build3/meshes/sphere3D_0.1to0.2.mesh";
    //const char * meshbase_file = "../build3/meshes/sphere3D_0.05to0.1.mesh";
    //const char * meshbase_file = "../build3/meshes/sphere3D_veryfine.mesh";
    //const char * meshbase_file = "../build3/meshes/beam-tet.mesh";
    //const char * meshbase_file = "../build3/meshes/escher-p3.mesh";
    //const char * meshbase_file = "../build3/meshes/orthotope3D_moderate.mesh";
    const char * meshbase_file = "../build3/meshes/orthotope3D_fine.mesh";

    //const char * meshbase_file = "../build3/meshes/square_2d_moderate.mesh";
    //const char * meshbase_file = "../build3/meshes/square_2d_fine.mesh";


    int order = 0;
    bool visualization = 0;

    int nDimensions     = 4;                            //data("nDimensions", 4);
    int feorder         = 0;                            //data("feorder", 0);
    int ser_ref_levels  = 0;                             //cmm_line("--nref_serial", 0);
    int par_ref_levels  = 3;                          //cmm_line("--nref_parallel", nref);

    OptionsParser args(argc, argv);


    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&meshbase_file, "-m3", "--mesh3D",
                   "Mesh file to use.");
    args.AddOption(&feorder, "-o", "--feorder",
                   "Finite element order (polynomial degree).");
    args.AddOption(&ser_ref_levels, "-sref", "--sref",
                   "Number of serial refinements.");
    args.AddOption(&par_ref_levels, "-pref", "--pref",
                   "Number of parallel refinements.");
    args.AddOption(&nDimensions, "-dim", "--whichD",
                   "Dimension of the space-time problem.");
    args.AddOption(&Nsteps, "-nstps", "--nsteps",
                   "Number of time steps.");
    args.AddOption(&tau, "-tau", "--tau",
                   "Time step.");
    args.AddOption(&generate_from3d, "-g3d", "--genfrom3d",
                   "Generating mesh from 3d.");
    args.AddOption(&generate_parallel, "-gp", "--genpar",
                   "Generating mesh in parallel from 3d.");
    args.AddOption(&whichparallel, "-pv", "--parver",
                   "Version of parallel algorithm.");
    args.AddOption(&bnd_method, "-bnd", "--bndmeth",
                   "Method for generating boundary elements.");
    args.AddOption(&local_method, "-loc", "--locmeth",
                   "Method for local mesh procedure.");
    args.AddOption(&numsol, "-nsol", "--numsol",
                   "Solution number.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");

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

    // 3. Read the (serial) mesh from the given mesh file on all processors.  We
    //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
    //    and volume meshes with the same code.

    Mesh *mesh;

    shared_ptr<ParMesh> pmesh;

    // 2. Create a serial mesh and distribute it.
    if (nDimensions == 3)
    {
        mesh = new Mesh(1,1,1,mfem::Element::TETRAHEDRON,1);
        //mesh = new Mesh(1,1,1,mfem::Element::HEXAHEDRON,1); // = CUBE

    }
    else
    {
        if (nDimensions == 4)
        {
            if ( generate_from3d == 1 )
            {
                if ( verbose )
                    cout << "Creating a 4d mesh from a 3d mesh from the file " << meshbase_file << endl;

                Mesh * mesh3dbase;
                ifstream imesh(meshbase_file);
                if (!imesh)
                {
                     std::cerr << "\nCan not open mesh file for base mesh: " << meshbase_file << '\n' << std::endl;
                     MPI_Finalize();
                     return -2;
                }
                mesh3dbase = new Mesh(imesh, 1, 1);
                imesh.close();

                // doing one refinement because otherwise the particular mesh example is \
                too small for using more than 2 processes
                for (int l = 0; l < ser_ref_levels; l++)
                    mesh3dbase->UniformRefinement();

                if ( verbose )
                {
                    std::stringstream fname;
                    fname << "mesh_3dbase.mesh";
                    std::ofstream ofid(fname.str().c_str());
                    ofid.precision(8);
                    mesh3dbase->Print(ofid);
                }

                if (generate_parallel == 1) //parallel version
                {
                    ParMesh * pmesh3dbase = new ParMesh(comm, *mesh3dbase);
                    delete mesh3dbase;

                    pmesh3dbase->PrintInfo();

                    chrono.Clear();
                    chrono.Start();

                    if ( whichparallel == 1 )
                    {
                        mesh = new Mesh( comm, *pmesh3dbase, tau, Nsteps, bnd_method, local_method);
                        if ( myid == 0)
                            cout << "Success: 4D Mesh (Serial) is created from ParMesh 3D" << endl << flush;
                    }
                    else
                    {
                        //ParMesh * pmeshtemp = new ParMesh ( comm, *pmesh3dbase, tau, Nsteps, bnd_method, local_method);
                        //pmesh = shared_ptr<ParMesh>(pmeshtemp);
                        pmesh = make_shared<ParMesh>( comm, *pmesh3dbase, tau, Nsteps, bnd_method, local_method);

                        if ( myid == 0)
                            cout << "Success: 4D Mesh (Parallel) is created from ParMesh 3D" << endl << flush;

                        //MPI_Finalize();
                        //return 0;
                    }

                    MPI_Barrier(comm);
                    chrono.Stop();
                    if (myid == 0 && whichparallel == 1)
                        std::cout << "Timing: ParMesh3d->Mesh4d: done in "
                                  << chrono.RealTime() << " seconds.\n";
                    if (myid == 0 && whichparallel == 2)
                        std::cout << "Timing: ParMesh3d->ParMesh4d: done in "
                                  << chrono.RealTime() << " seconds.\n";
                }
                else // serial version
                {
                    mesh = new Mesh( *mesh3dbase, tau, Nsteps, bnd_method, local_method);
                    if ( myid == 0)
                        cout << "Success: ParMesh3DtoParMesh4D has been called" << endl << flush;
                }

            } //end of the case of generating the 4d mesh
            else
            {
                cout << "Reading a 4d mesh from the file " << mesh_file << endl;
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
        } //end of nDimensions = 4 case
        else
        {
            std::cerr << "Case nDimensions = " << nDimensions << " is not supported" << std::endl;
            MPI_Finalize();
            return -1;
        }
    }

    if ( generate_parallel == 0 || (generate_parallel == 1 && whichparallel == 1) )
    {
         if (nDimensions == 4)
             // Checking that mesh is legal
             mesh->MeshCheck();

         for (int l = 0; l < ser_ref_levels; l++)
             mesh->UniformRefinement();
    }

    if ( generate_parallel == 0 || (generate_parallel == 1 && whichparallel == 1) )
    {
        if ( verbose )
            cout << "Creating parmesh(4d) from the serial mesh (4d)" << endl << flush;
        pmesh = make_shared<ParMesh>(comm, *mesh);
        delete mesh;
    }

    /*
    if (verbose)
    {
        cout << "Check shared structure..." << endl;
        cout << flush;
    }
    cout << flush;
    MPI_Barrier(comm);

    pmesh->PrintSharedStructParMesh();
    */

    /*
     * old

    Mesh *mesh;
    ifstream imesh(mesh_file);
    if (!imesh)
    {
       if (verbose)
       {
          cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
       }
       MPI_Finalize();
       return 2;
    }

     if ( nDimensions == 4 )
     {
        mesh = new Mesh(imesh, 1, 1);
        imesh.close();
     }

     //cout << "Trying Make4D" << endl;
     //mesh = new Mesh(4,4,4,4,mfem::Element::TESSERACT,0);
     //mesh = new Mesh(4,4,4,4,mfem::Element::PENTATOPE,0);


     if ( nDimensions == 3 )
        mesh = new Mesh(1,1,1,mfem::Element::HEXAHEDRON,1);

    int dim = mesh->Dimension();

     //cout << "Success 3!" << endl;
    //MPI_Finalize();
    //return 2;


    // 4. Refine the serial mesh on all processors to increase the resolution. In
    //    this example we do 'ref_levels' of uniform refinement. We choose
    //    'ref_levels' to be the largest number that gives a final mesh with no
    //    more than 10,000 elements.
    {
    //   int ref_levels =
     //     (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
       for (int l = 0; l < ser_ref_levels; l++)
       {
          mesh->UniformRefinement();
       }
    }

    // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
    //    this mesh further in parallel to increase the resolution. Once the
    //    parallel mesh is defined, the serial mesh can be deleted.
    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;

    */

    {
       //int par_ref_levels = 0;
       for (int l = 0; l < par_ref_levels; l++)
       {
          pmesh->UniformRefinement();
       }
    }

    int dim = pmesh->Dimension();

    //if(dim==3) pmesh->ReorientTetMesh();

    pmesh->PrintInfo(std::cout); if(verbose) cout << endl;

    // 6. Define a parallel finite element space on the parallel mesh. Here we
    //    use the Raviart-Thomas finite elements of the specified order.

    FiniteElementCollection *hdiv_coll, *h1_coll, *l2_coll;
    if (dim == 4)
    {
        hdiv_coll = new RT0_4DFECollection;
        if (verbose)cout << "RT: order 0 for 4D" << endl;
        if(feorder <= 1)
        {
            h1_coll = new LinearFECollection;
            if (verbose)cout << "H1: order 1 for 4D" << endl;
        }
        else
        {
            h1_coll = new QuadraticFECollection;
            if (verbose)cout << "H1: order 2 for 4D" << endl;
        }
        l2_coll = new L2_FECollection(0, dim);
        if (verbose)cout << "L2: order 0 for 4D" << endl;
    }
    else
    {
        hdiv_coll = new RT_FECollection(feorder, dim);
        if (verbose)cout << "RT: order " << feorder << " for 3D" << endl;
        h1_coll = new H1_FECollection(feorder+1, dim);
        if (verbose)cout << "H1: order " << feorder + 1 << " for 3D" << endl;
        l2_coll = new L2_FECollection(feorder, dim);
        if (verbose)cout << "L2: order " << feorder << " for 3D" << endl;
    }

    ParFiniteElementSpace *R_space = new ParFiniteElementSpace(pmesh.get(), hdiv_coll);
    ParFiniteElementSpace *H_space = new ParFiniteElementSpace(pmesh.get(), h1_coll);
    ParFiniteElementSpace *W_space = new ParFiniteElementSpace(pmesh.get(), l2_coll);

    HYPRE_Int dimR = R_space->GlobalTrueVSize();
    HYPRE_Int dimH = H_space->GlobalTrueVSize();
    HYPRE_Int dimW = W_space->GlobalTrueVSize();

    if (verbose)
    {
        std::cout << "***********************************************************\n";
        std::cout << "dim(R) = " << dimR << "\n";
        std::cout << "dim(H) = " << dimH << "\n";
        std::cout << "dim(W) = " << dimW << "\n";
        std::cout << "dim(R+H+W) = " << dimR + dimH + dimW << "\n";
        std::cout << "***********************************************************\n";
    }

    // 7. Define the two BlockStructure of the problem.  block_offsets is used
    //    for Vector based on dof (like ParGridFunction or ParLinearForm),
    //    block_trueOffstes is used for Vector based on trueDof (HypreParVector
    //    for the rhs and solution of the linear system).  The offsets computed
    //    here are local to the processor.

    Array<int> block_offsets(4); // number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = R_space->GetVSize();
    block_offsets[2] = H_space->GetVSize();
    block_offsets[3] = W_space->GetVSize();
    block_offsets.PartialSum();

    Array<int> block_trueOffsets(4); // number of variables + 1
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = R_space->TrueVSize();
    block_trueOffsets[2] = H_space->TrueVSize();
    block_trueOffsets[3] = W_space->TrueVSize();
    block_trueOffsets.PartialSum();


    // 8. Define the coefficients, analytical solution, and rhs of the PDE.

    Heat_test Mytest(nDimensions,numsol);

    ConstantCoefficient k(1.0);
    ConstantCoefficient zero(.0);
    Vector vzero(dim); vzero = 0.;
    VectorConstantCoefficient vzero_coeff(vzero);

    /*
    FunctionCoefficient fcoeff(fFun);//<<<<<<
    FunctionCoefficient ucoeff(uFun_ex);//<<<<<<
    //FunctionCoefficient u0(u0_function); //initial condition
    VectorFunctionCoefficient sigmacoeff(dim, sigmaFun_ex);
    */

    //----------------------------------------------------------
    // Setting boundary conditions.
    //----------------------------------------------------------

    Array<int> ess_bdr(pmesh->bdr_attributes.Max()); // applied to H^1 variable
    ess_bdr = 1;
    ess_bdr[pmesh->bdr_attributes.Max()-1] = 0;

     //-----------------------


    // 9. Define the parallel grid function and parallel linear forms, solution
    //    vector and rhs.
    BlockVector x(block_offsets), rhs(block_offsets);
    BlockVector trueX(block_trueOffsets), trueRhs(block_trueOffsets);
    trueX =0.0;
    ParGridFunction *u(new ParGridFunction);
    u->MakeRef(H_space, x.GetBlock(1), 0);
    *u = 0.0;
    //u->ProjectCoefficient(*(Mytest.scalaru));
    trueRhs=.0;

    ParLinearForm *fform(new ParLinearForm);
    fform->Update(R_space, rhs.GetBlock(0), 0);
    fform->AddDomainIntegrator(new VectordivDomainLFIntegrator(zero));
    fform->Assemble();
    fform->ParallelAssemble(trueRhs.GetBlock(0));

    ParLinearForm *qform(new ParLinearForm);
    qform->Update(H_space, rhs.GetBlock(1), 0);
    qform->AddDomainIntegrator(new VectorDomainLFIntegrator(vzero_coeff));
    qform->Assemble();

    ParLinearForm *gform(new ParLinearForm);
    gform->Update(W_space, rhs.GetBlock(2), 0);
    gform->AddDomainIntegrator(new DomainLFIntegrator(*(Mytest.scalarf)));
    gform->Assemble();
    gform->ParallelAssemble(trueRhs.GetBlock(2));

    // 10. Assemble the finite element matrices for the Darcy operator
    //
    //                       CFOSLS = [  A   B  D^T ]
    //                                [ B^T  C   0  ]
    //                                [  D   0   0  ]
    //     where:
    //
    //     A = ( sigma, tau)_{H(div)}
    //     B = (sigma, [ dx(S), -S] )
    //     C = ( [dx(S), S], [dx(V),V] )
    //     D = ( div(sigma), mu )

    chrono.Clear();
    chrono.Start();

    //---------------
    //  A Block:
    //---------------

    ParBilinearForm *Ablock(new ParBilinearForm(R_space));
    HypreParMatrix *A;

    Ablock->AddDomainIntegrator(new VectorFEMassIntegrator);
    Ablock->Assemble();
    Ablock->Finalize();
    A = Ablock->ParallelAssemble();

    //---------------
    //  C Block:
    //---------------

    ParBilinearForm *Cblock(new ParBilinearForm(H_space));
    HypreParMatrix *C;
    Cblock->AddDomainIntegrator(new PAUVectorFEMassIntegrator2);
    Cblock->Assemble();
    Cblock->EliminateEssentialBC(ess_bdr, x.GetBlock(1), rhs.GetBlock(1));
    Cblock->Finalize();
    C = Cblock->ParallelAssemble();

    //---------------
    //  B Block:
    //---------------

    ParMixedBilinearForm *Bblock(new ParMixedBilinearForm(H_space, R_space));
    HypreParMatrix *B;
    Bblock->AddDomainIntegrator(new PAUVectorFEMassIntegrator);
    Bblock->Assemble();
    Bblock->EliminateTrialDofs(ess_bdr, x.GetBlock(1), rhs.GetBlock(0));
    Bblock->Finalize();
    B = Bblock->ParallelAssemble();
    HypreParMatrix *BT = B->Transpose();

    //----------------
    //  D Block:
    //-----------------

    ParMixedBilinearForm *Dblock(new ParMixedBilinearForm(R_space, W_space));
    HypreParMatrix *D;

    Dblock->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
    Dblock->Assemble();
    Dblock->Finalize();
    D = Dblock->ParallelAssemble();
    HypreParMatrix *DT = D->Transpose();

    //=======================================================
    // Assembling the Matrix
    //-------------------------------------------------------

    fform->ParallelAssemble(trueRhs.GetBlock(0));
    qform->ParallelAssemble(trueRhs.GetBlock(1));
    BlockOperator *CFOSLSop = new BlockOperator(block_trueOffsets);
    CFOSLSop->SetBlock(0,0, A);
    CFOSLSop->SetBlock(0,1, B);
    CFOSLSop->SetBlock(1,0, BT);
    CFOSLSop->SetBlock(1,1, C);
    CFOSLSop->SetBlock(0,2, DT);
    CFOSLSop->SetBlock(2,0, D);

    if (verbose)
        std::cout << "System built in " << chrono.RealTime() << "s. \n";

    // 11. Construct the operators for preconditioner
    //
    //                 P = [ diag(M)         0         ]
    //                     [  0       B diag(M)^-1 B^T ]
    //
    //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
    //     pressure Schur Complement.
    chrono.Clear();
    chrono.Start();
    HypreParMatrix *AinvDt = D->Transpose();
    HypreParVector *Ad = new HypreParVector(MPI_COMM_WORLD, A->GetGlobalNumRows(),
                                            A->GetRowStarts());
    A->GetDiag(*Ad);

    AinvDt->InvScaleRows(*Ad);
    HypreParMatrix *S = ParMult(D, AinvDt);

    Solver * invA;
    if (false) //false
        invA = new HypreADS(*A, R_space);
    else
        invA = new HypreDiagScale(*A);
    HypreBoomerAMG * invC = new HypreBoomerAMG(*C);
    invC->SetPrintLevel(0);
 //    HypreDiagScale * invS = new HypreDiagScale(*S);
    HypreBoomerAMG * invS = new HypreBoomerAMG(*S);
    invS->SetPrintLevel(0);

    invA->iterative_mode = false;
    invC->iterative_mode = false;
    invS->iterative_mode = false;

    BlockDiagonalPreconditioner prec(block_trueOffsets);
    prec.SetDiagonalBlock(0, invA);
    prec.SetDiagonalBlock(1, invC);
    prec.SetDiagonalBlock(2, invS);

    if (verbose)
        std::cout << "Preconditioner built in " << chrono.RealTime() << "s. \n";

    // 12. Solve the linear system with MINRES.
    //     Check the norm of the unpreconditioned residual.

    int maxIter(30000);

    chrono.Clear();
    chrono.Start();
    MINRESSolver solver(MPI_COMM_WORLD);
    solver.SetAbsTol(atol);
    solver.SetRelTol(rtol);
    solver.SetMaxIter(maxIter);
    solver.SetOperator(*CFOSLSop);
    solver.SetPreconditioner(prec);
    solver.SetPrintLevel(0);
    trueX = 0.0;
    solver.Mult(trueRhs, trueX);
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
        irs[i] = &(IntRules.Get(i, order_quad));

    ParGridFunction *sigma(new ParGridFunction);
    sigma->MakeRef(R_space, x.GetBlock(0), 0);
    sigma->Distribute(&(trueX.GetBlock(0)));

    // adding back the term from nonhomogeneous initial condition
    ParGridFunction *sigma_nonhomo = new ParGridFunction(R_space);
    sigma_nonhomo->ProjectCoefficient(*(Mytest.sigma_nonhomo));

    *sigma += *sigma_nonhomo;



    double err_sigma_loc  = sigma->ComputeL2Error(*(Mytest.sigma), irs);
    err_sigma_loc *= err_sigma_loc;
    double err_sigma;
    MPI_Reduce(&err_sigma_loc, &err_sigma, 1,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double norm_sigma_loc = ComputeGlobalLpNorm(2, *(Mytest.sigma), *pmesh, irs);
    norm_sigma_loc *= norm_sigma_loc;
    double norm_sigma;
    MPI_Reduce(&norm_sigma_loc, &norm_sigma, 1,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (verbose)
    {
        std::cout << "|| sigma_h - sigma_ex || / || sigma_ex || = "
                  << sqrt(err_sigma)/sqrt(norm_sigma)  << "\n";
    }


    err_sigma *= err_sigma;
    norm_sigma *= norm_sigma;

    DiscreteLinearOperator Div(R_space, W_space);
    Div.AddDomainInterpolator(new DivergenceInterpolator());
    ParGridFunction DivSigma(W_space);
    Div.Assemble();
    Div.Mult(*sigma, DivSigma);

    double err_div_loc = DivSigma.ComputeL2Error(*(Mytest.scalardivsigma),irs);
    err_div_loc *= err_div_loc;
    double err_div = 0.0;
    MPI_Reduce(&err_div_loc, &err_div, 1,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double norm_div_loc = ComputeGlobalLpNorm(2, *(Mytest.scalardivsigma), *pmesh, irs);
    norm_div_loc *= norm_div_loc;
    double norm_div = 0.0;
    MPI_Reduce(&norm_div_loc, &norm_div, 1,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (verbose)
    {
        std::cout << "|| div (sigma_h - sigma_ex) || / ||div (sigma_ex)|| = "
                  << sqrt(err_div)/sqrt(norm_div)  << "\n";
    }


    norm_sigma += norm_div;
    err_sigma += err_div;


    ParGridFunction *S_exact = new ParGridFunction(H_space);
    S_exact->ProjectCoefficient(*(Mytest.scalaru));

    ParGridFunction *Svar(new ParGridFunction);
    Svar->MakeRef(H_space, x.GetBlock(1), 0);
    Svar->Distribute(&(trueX.GetBlock(1)));

    ParGridFunction *S_nonhomo = new ParGridFunction(H_space);
    S_nonhomo->ProjectCoefficient(*(Mytest.scalarSnonhomo));

    *Svar += *S_nonhomo;

    double err_S  = Svar->ComputeL2Error(*(Mytest.scalaru), irs);
    double norm_S = ComputeGlobalLpNorm(2, *(Mytest.scalaru), *pmesh, irs);

    if (verbose)
    {
        //std::cout << "|| sigma_h - sigma_ex ||_E = " << sqrt(err_sigma) << endl;
        std::cout << "|| sigma_h - sigma_ex ||_E / || sigma_ex ||_E = "
                  << sqrt(err_sigma)/sqrt(norm_sigma)  << "\n";
        //std::cout << "norm_u = " << norm_u << std::endl;
        std::cout << "|| S_h - S_ex || / || S_ex || = "
                  << err_S/norm_S  << "\n";
    }

    ParGridFunction *sigma_exact = new ParGridFunction(R_space);
    sigma_exact->ProjectCoefficient(*(Mytest.sigma));

    ParGridFunction Proju(H_space);
    Proju = 0.0; Proju.ProjectCoefficient(*(Mytest.scalaru));
    //Proju = 0.0; Proju.ProjectCoefficient(ucoeff);


    HypreParVector * Projupv = Proju.ParallelAssemble();
    HypreParVector * upv = u->ParallelAssemble();

    //*upv -= *Projupv;

    Vector * Projuv = Projupv->GlobalVector();
    Vector * uv = upv->GlobalVector();
    *uv -= *Projuv;


    //cout << "uv.size = " << uv->Size() << endl;

    double projection_norm = (*Projuv)*(*Projuv);
    double projection_error = (*uv) * (*uv);
    if(!myid) std::cout << "|| u_H - u_ex ||_h / || u_ex ||_h = "
                        << std::sqrt(projection_error) / std::sqrt(projection_norm)
                        << std::endl;

    ParGridFunction Projsigma(R_space);
    //Projsigma = 0.0; Projsigma.ProjectCoefficient(sigmacoeff);
    Projsigma = 0.0; Projsigma.ProjectCoefficient(*(Mytest.sigma));

    HypreParVector * Projsigmapv = Projsigma.ParallelAssemble();
    HypreParVector * sigmapv = sigma->ParallelAssemble();


    Vector * Projsigmav = Projsigmapv->GlobalVector();
    Vector * sigmav = sigmapv->GlobalVector();
    *sigmav -= *Projsigmav;

    if (myid == 0)
        cout << "Projsigmav.size = " << Projsigmav->Size() << endl;


    double projectionsigma_norm = (*Projsigmav)*(*Projsigmav);
    double projectionsigma_error = (*sigmav) * (*sigmav);
    if(!myid) std::cout << "|| sigma_H - sigma_ex ||_h / || sigma_ex ||_h = "
                        << std::sqrt(projectionsigma_error) / std::sqrt(projectionsigma_norm)
                        << std::endl;


    BilinearForm *m = new BilinearForm(R_space);
    m->AddDomainIntegrator(new DivDivIntegrator);
    m->AddDomainIntegrator(new VectorFEMassIntegrator);
    m->Assemble(); m->Finalize();
    SparseMatrix E = m->SpMat();
    Vector Asigma(sigmav->Size());
    E.Mult(*Projsigmav,Asigma);
    double weighted_norm = (*Projsigmav)*Asigma;

    Vector Ae(sigmav->Size());
    E.Mult(*sigmav,Ae);
    double weighted_error = (*sigmav)*Ae;

    if(!myid) std::cout << "|| sigma_H - sigma_ex ||_h,E / || sigma_ex ||_h,E = " <<
                        sqrt(weighted_error)/sqrt(weighted_norm)
                        << std::endl;

    if (visualization)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;
        socketstream u_sock(vishost, visport);
        u_sock << "parallel " << num_procs << " " << myid << "\n";
        u_sock.precision(8);
        MPI_Barrier(pmesh->GetComm());
        u_sock << "solution\n" << *pmesh << *sigma_exact
               << "window_title 'sigma_exact'" << endl;
        // Make sure all ranks have sent their 'u' solution before initiating
        // another set of GLVis connections (one from each rank):

        socketstream uu_sock(vishost, visport);
        uu_sock << "parallel " << num_procs << " " << myid << "\n";
        uu_sock.precision(8);
        MPI_Barrier(pmesh->GetComm());
        uu_sock << "solution\n" << *pmesh << *sigma << "window_title 'sigma'"
                << endl;

        *sigma_exact -= *sigma;
        socketstream uuu_sock(vishost, visport);
        uuu_sock << "parallel " << num_procs << " " << myid << "\n";
        uuu_sock.precision(8);
        MPI_Barrier(pmesh->GetComm());
        uuu_sock << "solution\n" << *pmesh << *sigma_exact
                 << "window_title 'difference'" << endl;


        socketstream s_sock(vishost, visport);
        s_sock << "parallel " << num_procs << " " << myid << "\n";
        s_sock.precision(8);
        MPI_Barrier(pmesh->GetComm());
        s_sock << "solution\n" << *pmesh << *S_exact << "window_title 'S_exact'"
                << endl;

        socketstream ss_sock(vishost, visport);
        ss_sock << "parallel " << num_procs << " " << myid << "\n";
        ss_sock.precision(8);
        MPI_Barrier(pmesh->GetComm());
        ss_sock << "solution\n" << *pmesh << *Svar << "window_title 'S'"
                << endl;

        *S_exact -= *Svar;
        socketstream sss_sock(vishost, visport);
        sss_sock << "parallel " << num_procs << " " << myid << "\n";
        sss_sock.precision(8);
        MPI_Barrier(pmesh->GetComm());
        sss_sock << "solution\n" << *pmesh << *S_exact
                 << "window_title 'difference for S'" << endl;
    }

    // 17. Free the used memory.
    delete fform;
    delete gform;
    delete u;
    delete CFOSLSop;
    delete DT;
    delete D;
    delete C;
    delete BT;
    delete B;
    delete A;

    delete Ablock;
    delete Bblock;
    delete Cblock;
    delete Dblock;
    delete W_space;
    delete H_space;
    delete R_space;
    delete l2_coll;
    delete hdiv_coll;
    delete h1_coll;

    MPI_Finalize();

    return 0;
}

template <double (*S)(const Vector&), void (*Sgradxvec)(const Vector&, Vector& )> \
void sigmaTemplate(const Vector& xt, Vector& sigma)
{
    sigma.SetSize(xt.Size());

    Vector gradS;
    Sgradxvec(xt,gradS);

    sigma(xt.Size()-1) = S(xt);
    for (int i = 0; i < xt.Size()-1; i++)
        sigma(i) = - gradS(i);

    return;
}

template <double (*S)(const Vector&), void (*Sgradxvec)(const Vector&, Vector& )> \
void sigmaNonHomoTemplate(const Vector& xt, Vector& sigma) // sigmaNonHomo = ( - grad u, u) for u = S(t=0)
{
    sigma.SetSize(xt.Size());

    Vector xteq0(xt.Size()); // xt with t = 0
    xteq0 = xt;
    xteq0(xteq0.Size()-1) = 0.0;

    Vector gradS;
    Sgradxvec(xteq0,gradS);

    sigma(xt.Size()-1) = S(xteq0);
    for (int i = 0; i < xt.Size()-1; i++)
        sigma(i) = - gradS(i);

    return;
}


template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), double (*Slaplace)(const Vector & xt) > \
double rhsideTemplate(const Vector& xt)
{
    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    return dSdt(xt) - Slaplace(xt) + Slaplace(xt0);
}

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), double (*Slaplace)(const Vector & xt) > \
double divsigmaTemplate(const Vector& xt)
{
    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    return dSdt(xt) - Slaplace(xt);
}

template<double (*S)(const Vector & xt)> \
    double SnonhomoTemplate(const Vector& xt)
{
    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    return S(xt0);
}


double uFun_ex(const Vector & xt)
{
    const double PI = 3.141592653589793;
    double xi(xt(0));
    double yi(xt(1));
    double zi(0.0);
    double vi(0.0);

    if (xt.Size() == 3)
    {
        zi = xt(2);
        return sin(PI*xi)*sin(PI*yi)*zi;
    }
    if (xt.Size() == 4)
    {
        zi = xt(2);
        vi = xt(3);
        //cout << "sol for 4D" << endl;
        return sin(PI*xi)*sin(PI*yi)*sin(PI*zi)*vi;
    }

    return 0.0;
}


double uFun_ex_dt(const Vector & xt)
{
    const double PI = 3.141592653589793;
    double xi(xt(0));
    double yi(xt(1));
    double zi(0.0);

    if (xt.Size() == 3)
        return sin(PI*xi)*sin(PI*yi);
    if (xt.Size() == 4)
    {
        zi = xt(2);
        return sin(PI*xi)*sin(PI*yi)*sin(PI*zi);
    }

    return 0.0;
}

double uFun_ex_laplace(const Vector & xt)
{
    const double PI = 3.141592653589793;
    return (-(xt.Size()-1) * PI * PI) *uFun_ex(xt);
}

void uFun_ex_gradx(const Vector& xt, Vector& gradx )
{
    const double PI = 3.141592653589793;

    double x = xt(0);
    double y = xt(1);
    double z(0.0);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    if (xt.Size() == 3)
    {
        gradx(0) = t * PI * cos (PI * x) * sin (PI * y);
        gradx(1) = t * PI * sin (PI * x) * cos (PI * y);
    }
    if (xt.Size() == 4)
    {
        z = xt(2);
        gradx(0) = t * PI * cos (PI * x) * sin (PI * y) * sin (PI * z);
        gradx(1) = t * PI * sin (PI * x) * cos (PI * y) * sin (PI * z);
        gradx(2) = t * PI * sin (PI * x) * sin (PI * y) * cos (PI * z);
    }

}


double fFun(const Vector & x)
{
    const double PI = 3.141592653589793;
    double xi(x(0));
    double yi(x(1));
    double zi(0.0);
    double vi(0.0);
    if (x.Size() == 3)
    {
     zi = x(2);
       return 2*PI*PI*sin(PI*xi)*sin(PI*yi)*zi+sin(PI*xi)*sin(PI*yi);
    }

    if (x.Size() == 4)
    {
     zi = x(2);
         vi = x(3);
         //cout << "rhand for 4D" << endl;
       return 3*PI*PI*sin(PI*xi)*sin(PI*yi)*sin(PI*zi)*vi + sin(PI*xi)*sin(PI*yi)*sin(PI*zi);
    }

    return 0.0;
}

void sigmaFun_ex(const Vector & x, Vector & u)
{
    const double PI = 3.141592653589793;
    double xi(x(0));
    double yi(x(1));
    double zi(0.0);
    double vi(0.0);
    if (x.Size() == 3)
    {
        zi = x(2);
        u(0) = - PI * cos (PI * xi) * sin (PI * yi) * zi;
        u(1) = - PI * cos (PI * yi) * sin (PI * xi) * zi;
        u(2) = uFun_ex(x);
        return;
    }

    if (x.Size() == 4)
    {
        zi = x(2);
        vi = x(3);
        u(0) = - PI * cos (PI * xi) * sin (PI * yi) * sin(PI * zi) * vi;
        u(1) = - sin (PI * xi) * PI * cos (PI * yi) * sin(PI * zi) * vi;
        u(2) = - sin (PI * xi) * sin(PI * yi) * PI * cos (PI * zi) * vi;
        u(3) = uFun_ex(x);
        return;
    }

    if (x.Size() == 2)
    {
        u(0) =  exp(-PI*PI*yi)*PI*cos(PI*xi);
        u(1) = -sin(PI*xi)*exp(-1*PI*PI*yi);
        return;
    }

    return;
}



double uFun1_ex(const Vector & xt)
{
    double tmp = (xt.Size() == 4) ? sin(M_PI*xt(2)) : 1.0;
    return exp(-xt(xt.Size()-1))*sin(M_PI*xt(0))*sin(M_PI*xt(1))*tmp;
}

double uFun1_ex_dt(const Vector & xt)
{
    return - uFun1_ex(xt);
}

double uFun1_ex_laplace(const Vector & xt)
{
    return (- (xt.Size() - 1) * M_PI * M_PI ) * uFun1_ex(xt);
}

void uFun1_ex_gradx(const Vector& xt, Vector& gradx )
{
    const double PI = 3.141592653589793;

    double x = xt(0);
    double y = xt(1);
    double z(0.0);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    if (xt.Size() == 3)
    {
        gradx(0) = exp(-t) * PI * cos (PI * x) * sin (PI * y);
        gradx(1) = exp(-t) * PI * sin (PI * x) * cos (PI * y);
    }
    if (xt.Size() == 4)
    {
        z = xt(2);
        gradx(0) = exp(-t) * PI * cos (PI * x) * sin (PI * y) * sin (PI * z);
        gradx(1) = exp(-t) * PI * sin (PI * x) * cos (PI * y) * sin (PI * z);
        gradx(2) = exp(-t) * PI * sin (PI * x) * sin (PI * y) * cos (PI * z);
    }

}

double fFun1(const Vector & x)
{
    return ( (x.Size()-1)*M_PI*M_PI - 1. ) * uFun1_ex(x);
}

void sigmaFun1_ex(const Vector & x, Vector & sigma)
{
    sigma.SetSize(x.Size());
    sigma(0) = -M_PI*exp(-x(x.Size()-1))*cos(M_PI*x(0))*sin(M_PI*x(1));
    sigma(1) = -M_PI*exp(-x(x.Size()-1))*sin(M_PI*x(0))*cos(M_PI*x(1));
    if (x.Size() == 4)
    {
        sigma(0) *= sin(M_PI*x(2));
        sigma(1) *= sin(M_PI*x(2));
        sigma(2) = -M_PI*exp(-x(x.Size()-1))*sin(M_PI*x(0))
                *sin(M_PI*x(1))*cos(M_PI*x(2));
    }
    sigma(x.Size()-1) = uFun1_ex(x);

    return;
}
