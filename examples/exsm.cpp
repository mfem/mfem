//                               ETHOS Example 1
//
// Compile with: make exsm
//
// Sample runs:  exsm -m tipton.mesh
//
// Description: This example code performs a simple mesh smoothing based on a
//              topologically defined "mesh Laplacian" matrix.
//
//              The example highlights meshes with curved elements, the
//              assembling of a custom finite element matrix, the use of vector
//              finite element spaces, the definition of different spaces and
//              grid functions on the same mesh, and the setting of values by
//              iterating over the interior and the boundary elements.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <limits>
#include <unistd.h>

using namespace mfem;

using namespace std;


// 1. Define the bilinear form corresponding to a mesh Laplacian operator. This
//    will be used to assemble the global mesh Laplacian matrix based on the
//    local matrix provided in the AssembleElementMatrix method. More examples
//    of bilinear integrators can be found in ../fem/bilininteg.hpp.
class VectorMeshLaplacianIntegrator : public BilinearFormIntegrator
{
private:
    int geom, type;
    LinearFECollection lfec;
    IsoparametricTransformation T;
    VectorDiffusionIntegrator vdiff;
    
public:
    VectorMeshLaplacianIntegrator(int type_) { type = type_; geom = -1; }
    virtual void AssembleElementMatrix(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
    virtual ~VectorMeshLaplacianIntegrator() { }
};

// 2. Implement the local stiffness matrix of the mesh Laplacian. This is a
//    block-diagonal matrix with each block having a unit diagonal and constant
//    negative off-diagonal entries, such that the row sums are zero.
void VectorMeshLaplacianIntegrator::AssembleElementMatrix(
                                                          const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
    if (type == 0)
    {
        int dim = el.GetDim(); // space dimension
        int dof = el.GetDof(); // number of element degrees of freedom
        
        elmat.SetSize(dim*dof); // block-diagonal element matrix
        
        for (int d = 0; d < dim; d++)
            for (int k = 0; k < dof; k++)
                for (int l = 0; l < dof; l++)
                    if (k==l)
                    {
                        elmat (dof*d+k, dof*d+l) = 1.0;
                    }
                    else
                    {
                        elmat (dof*d+k, dof*d+l) = -1.0/(dof-1);
                    }
    }
    else
    {
        if (el.GetGeomType() != geom)
        {
            geom = el.GetGeomType();
            T.SetFE(lfec.FiniteElementForGeometry(geom));
            Geometries.GetPerfPointMat(geom, T.GetPointMat());
        }
        T.Attribute = Trans.Attribute;
        T.ElementNo = Trans.ElementNo;
        vdiff.AssembleElementMatrix(el, T, elmat);
    }
}

class HarmonicModel : public HyperelasticModel
{
public:
    virtual double EvalW(const DenseMatrix &J) const
    {
        return 0.5*(J*J);
    }
    
    virtual void EvalP(const DenseMatrix &J, DenseMatrix &P) const
    {
        P = J;
    }
    
    virtual void AssembleH(const DenseMatrix &J, const DenseMatrix &DS,
                           const double weight, DenseMatrix &A) const
    {
        int dof = DS.Height(), dim = DS.Width();
        
        for (int i = 0; i < dof; i++)
            for (int j = 0; j <= i; j++)
            {
                double a = 0.0;
                for (int d = 0; d < dim; d++)
                {
                    a += DS(i,d)*DS(j,d);
                }
                a *= weight;
                for (int d = 0; d < dim; d++)
                {
                    A(i+d*dof,j+d*dof) += a;
                    if (i != j)
                    {
                        A(j+d*dof,i+d*dof) += a;
                    }
                }
            }
    }
};

class HyperelasticMeshIntegrator : public NonlinearFormIntegrator
{
protected:
    int geom, type;
    LinearFECollection lfec;
    IsoparametricTransformation T;
    HyperelasticNLFIntegrator hi;
    
    void SetT(const FiniteElement &el, ElementTransformation &Tr)
    {
        if (el.GetGeomType() != geom)
        {
            geom = el.GetGeomType();
            T.SetFE(lfec.FiniteElementForGeometry(geom));
            Geometries.GetPerfPointMat(geom, T.GetPointMat());
        }
        T.Attribute = Tr.Attribute;
        T.ElementNo = Tr.ElementNo;
    }
    
public:
    // type controls what the optimal (target) element shapes are:
    // 1 - the current mesh elements
    // 2 - the perfect reference element
    HyperelasticMeshIntegrator(HyperelasticModel *m, int _type)
    : hi(m) { geom = -1; type = _type; }
    
    virtual double GetElementEnergy(const FiniteElement &el,
                                    ElementTransformation &Tr,
                                    const Vector &elfun)
    {
        if (type == 1)
        {
            return hi.GetElementEnergy(el, Tr, elfun);
        }
        else
        {
            SetT(el, Tr);
            return hi.GetElementEnergy(el, T, elfun);
        }
    }
    
    virtual void AssembleElementVector(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       const Vector &elfun, Vector &elvect)
    {
        if (type == 1)
        {
            hi.AssembleElementVector(el, Tr, elfun, elvect);
        }
        else
        {
            SetT(el, Tr);
            hi.AssembleElementVector(el, T, elfun, elvect);
        }
    }
    
    virtual void AssembleElementGrad(const FiniteElement &el,
                                     ElementTransformation &Tr,
                                     const Vector &elfun, DenseMatrix &elmat)
    {
        if (type == 1)
        {
            hi.AssembleElementGrad(el, Tr, elfun, elmat);
        }
        else
        {
            SetT(el, Tr);
            hi.AssembleElementGrad(el, T, elfun, elmat);
        }
    }
};

// This is where K10 methods start
// Implement Relaxed Newton solver to avoid mesh becoming invalid during the
// optimization process
class RelaxedNewtonSolver : public IterativeSolver
{
protected:
    mutable Vector r, c;
public:
    RelaxedNewtonSolver() { }
    
#ifdef MFEM_USE_MPI
    RelaxedNewtonSolver(MPI_Comm _comm) : IterativeSolver(_comm) { }
#endif
        virtual void SetOperator(const Operator &op);
    
        virtual void SetSolver(Solver &solver) { prec = &solver; }
    
        virtual void Mult(const Vector &b, Vector &x) const;
    
        virtual void Mult2(const Vector &b, Vector &x,
                      const Mesh &mesh, const IntegrationRule &ir,
                           int *itnums, const NonlinearForm &nlf) const;
    
    
};

void RelaxedNewtonSolver::SetOperator(const Operator &op)
{
    oper = &op;
    height = op.Height();
    width = op.Width();
    MFEM_ASSERT(height == width, "square Operator is required.");
    
    r.SetSize(width);
    c.SetSize(width);
}

void RelaxedNewtonSolver::Mult(const Vector &b, Vector &x) const
{
    return;
}

void RelaxedNewtonSolver::Mult2(const Vector &b, Vector &x,
                                const Mesh &mesh, const IntegrationRule &ir,
                                int *itnums, const NonlinearForm &nlf) const
{
    MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
    MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");
    
    int it;
    double norm, norm_goal;
    bool have_b = (b.Size() == Height());
    
    if (!iterative_mode)
    {
        x = 0.0;
    }
    
    oper->Mult(x, r);
    if (have_b)
    {
        r -= b;
    }
    
    norm = Norm(r);
    norm_goal = std::max(rel_tol*norm, abs_tol);
    
    prec->iterative_mode = false;
    
    // x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i)-b]
    for (it = 0; true; it++)
    {
        MFEM_ASSERT(IsFinite(norm), "norm = " << norm);
        if (print_level >= 0)
            cout << "Newton iteration " << setw(2) << it
            << " : ||r|| = " << norm << '\n';
        
        if (norm <= norm_goal)
        {
            converged = 1;
            break;
        }
        
        
        if (it >= max_iter)
        {
            converged = 0;
            break;
        }
        
        prec->SetOperator(oper->GetGradient(x));
        
        prec->Mult(r, c);  // c = [DF(x_i)]^{-1} [F(x_i)-b]
        
        
        //k10 some changes
        const int NE = mesh.GetNE();
        const GridFunction &nodes = *mesh.GetNodes();
        
    
        Array<int> dofs;
        Vector xsav = x; //create a copy of x
        Vector csav = c;
        int tchk = 0;
        int iters = 0;
        double alpha = 1.;
        int jachk = 1;
        double initenergy = 0.0;
        double finenergy = 0.0;
        double nanchk;
        
        
        initenergy =nlf.GetEnergy(x);
        
        while (tchk !=1 && iters < 20)
        {
            iters += 1;
            //cout << "number of iters is " << iters << "\n";
            jachk = 1;
            c = csav;
            x = xsav;
            add (xsav,-alpha,csav,x);
            finenergy = nlf.GetEnergy(x);
            nanchk = isnan(finenergy);
            for (int i = 0; i < NE; i++)
            {
                const FiniteElement &fe = *nodes.FESpace()->GetFE(i);
                const int dim = fe.GetDim(), nsp = ir.GetNPoints(),
                dof = fe.GetDof();
                

                
                //cout << dof << " " << dim << " " << nsp << " dof,dim,nsp-k10\n";
                
                DenseTensor Jtr(dim, dim, nsp);
                
                const GridFunction *nds;
                nds = &nodes;
                
                DenseMatrix dshape(dof, dim), pos(dof, dim);
                Array<int> xdofs(dof * dim);
                Vector posV(pos.Data(), dof * dim);
                
                //cout << "element number " << i << "\n";
                
                nds->FESpace()->GetElementVDofs(i, xdofs);
                nds->GetSubVector(xdofs, posV);
                for (int j = 0; j < nsp; j++)
                {
                    //cout << "point number " << j << "\n";
                    fe.CalcDShape(ir.IntPoint(j), dshape);
                    MultAtB(pos, dshape, Jtr(j));
                    double det = Jtr(j).Det();
                    if (det<=0.)
                    {
                        jachk *= 0;
                    }
                    //cout << i << " " << j << " "<< det << " \n";
                }
            }
            
            //finenergy = initenergy - 1; //WARNING: THIS IS JUST FOR TIPTONUNI.MESH
            if (finenergy>1.0*initenergy || nanchk!=0 || jachk==0)
            {
                tchk = 0;
                alpha *= 0.5;
                //cout << "some element became inverted..reducing alpha\n";
            }
            else
            {
                tchk = 1;
            }
            
            
            
        }
        
        cout << initenergy << " " << finenergy <<  " energy value before and after newton iteration\n";
        cout << "alpha value is " << alpha << " \n";
        
        if (tchk==0)
        {
            alpha =0;
        }
        
        add (xsav,-alpha,csav,x);
        
        //k10
        oper->Mult(x, r);
        if (have_b)
        {
            r -= b;
        }
        norm = Norm(r);
        
        if (alpha == 0)
        {
            converged = 0;
            cout << "Line search was not successfull.. stopping Newton\n";
            break;
        }
        
    }
    
    final_iter = it;
    final_norm = norm;
    *itnums = final_iter;
}

// Relaxed newton iterations which is kind of like descent.. it can handle inverted
// elements but the metric value must decrease :)
class DescentNewtonSolver : public IterativeSolver
{
protected:
    mutable Vector r, c;
public:
    DescentNewtonSolver() { }
    
#ifdef MFEM_USE_MPI
    DescentNewtonSolver(MPI_Comm _comm) : IterativeSolver(_comm) { }
#endif
    virtual void SetOperator(const Operator &op);
    
    virtual void SetSolver(Solver &solver) { prec = &solver; }
    
    virtual void Mult(const Vector &b, Vector &x) const;
    
    virtual void Mult2(const Vector &b, Vector &x,
                       const Mesh &mesh, const IntegrationRule &ir,
                       int *itnums, const NonlinearForm &nlf,
                       double &tauval) const;
};
void DescentNewtonSolver::SetOperator(const Operator &op)
{
    oper = &op;
    height = op.Height();
    width = op.Width();
    MFEM_ASSERT(height == width, "square Operator is required.");
    
    r.SetSize(width);
    c.SetSize(width);
}

void DescentNewtonSolver::Mult(const Vector &b, Vector &x) const
{
    return;
}

void DescentNewtonSolver::Mult2(const Vector &b, Vector &x,
                                const Mesh &mesh, const IntegrationRule &ir,
                                int *itnums, const NonlinearForm &nlf,
                                double &tauval) const
{
    MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
    MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");
    
    int it;
    double norm, norm_goal;
    bool have_b = (b.Size() == Height());
    
    if (!iterative_mode)
    {
        x = 0.0;
    }
    
    oper->Mult(x, r);
    if (have_b)
    {
        r -= b;
    }
    
    norm = Norm(r);
    norm_goal = std::max(rel_tol*norm, abs_tol);
    
    prec->iterative_mode = false;
    // x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i)-b]
    for (it = 0; true; it++)
    {
        MFEM_ASSERT(IsFinite(norm), "norm = " << norm);
        if (print_level >= 0)
            cout << "Newton iteration " << setw(2) << it
            << " : ||r|| = " << norm << '\n';
        
        if (norm <= norm_goal)
        {
            converged = 1;
            break;
        }
        
        
        if (it >= max_iter)
        {
            converged = 0;
            break;
        }
        
        const int NE = mesh.GetNE();
        const GridFunction &nodes = *mesh.GetNodes();
        
        // This is for model22 only.. figure out how to call it for that only
        Array<int> dofs;
        tauval = 1e+6;
        int nelinvorig= 0;
        for (int i = 0; i < NE; i++)
        {
            const FiniteElement &fe = *nodes.FESpace()->GetFE(i);
            const int dim = fe.GetDim(), nsp = ir.GetNPoints(),
            dof = fe.GetDof();
            DenseTensor Jtr(dim, dim, nsp);
            const GridFunction *nds;
            nds = &nodes;
            DenseMatrix dshape(dof, dim), pos(dof, dim);
            Array<int> xdofs(dof * dim);
            Vector posV(pos.Data(), dof * dim);
            nds->FESpace()->GetElementVDofs(i, xdofs);
            nds->GetSubVector(xdofs, posV);
            for (int j = 0; j < nsp; j++)
            {
                //cout << "point number " << j << "\n";
                fe.CalcDShape(ir.IntPoint(j), dshape);
                MultAtB(pos, dshape, Jtr(j));
                double det = Jtr(j).Det();
                tauval = min(tauval,det);
                if (det<=0.)
                {
                    nelinvorig += 1;
                }
            }
        }
        if (tauval>0)
        {
            tauval = 1e-4;
        }
        else
        {
            tauval -= 1e-2;
        }
        cout << "the determine tauval is " << tauval << "\n";
        ////

        prec->SetOperator(oper->GetGradient(x));
        
        prec->Mult(r, c);  // c = [DF(x_i)]^{-1} [F(x_i)-b]
        
        //k10 some changes
        //const GridFunction &nodes = *mesh.GetNodes();
        
        Vector xsav = x; //create a copy of x
        int tchk = 0;
        int iters = 0;
        double alpha = 1;
        double initenergy = 0.0;
        double finenergy = 0.0;
        double nanchk;
        int nelinvnew;
        nelinvnew = 0;
        
        initenergy =nlf.GetEnergy(x);
        cout << "energy level is " << initenergy << " \n";
        const int nsp = ir.GetNPoints();
        
        while (tchk !=1 && iters <  15)
        {
            iters += 1;
            add (xsav,-alpha,c,x);
            finenergy = nlf.GetEnergy(x);
            nanchk = isnan(finenergy);
            //cout << "energy level is " << finenergy << " at iteration " << iters << " \n";
            
            // check if more than initial inverted elements are created
            nelinvnew = 0;
            for (int i = 0; i < NE; i++)
            {
                const FiniteElement &fe = *nodes.FESpace()->GetFE(i);
                const int dim = fe.GetDim(),
                dof = fe.GetDof();
                DenseTensor Jtr(dim, dim, nsp);
                const GridFunction *nds;
                nds = &nodes;
                DenseMatrix dshape(dof, dim), pos(dof, dim);
                Array<int> xdofs(dof * dim);
                Vector posV(pos.Data(), dof * dim);
                nds->FESpace()->GetElementVDofs(i, xdofs);
                nds->GetSubVector(xdofs, posV);
                for (int j = 0; j < nsp; j++)
                {
                    fe.CalcDShape(ir.IntPoint(j), dshape);
                    MultAtB(pos, dshape, Jtr(j));
                    double det = Jtr(j).Det();
                    if (det<=0.)
                    {
                        nelinvnew += 1;
                    }
                }
            }
            //
            
            if (finenergy>initenergy || nanchk!=0) //|| nelinvnew > nelinvorig)
            {
                alpha *= 0.1;
            }
                else
            {
                tchk = 1;
            }
        }
        cout << "number of sub newton iters is " << iters << "\n";
        cout << nelinvorig << " " << nelinvnew << " num inverted old & new\n";
        if (tchk==0)
        {
            alpha =0;
        }
        cout << "alpha determined is " << alpha  << "\n";
        add (xsav,-alpha,c,x);
        
        //k10
        oper->Mult(x, r);
        if (have_b)
        {
            r -= b;
        }
        norm = Norm(r);
        
        if (alpha == 0)
        {
            converged = 0;
            cout << "Line search was not successfull.. stopping Newton\n";
            break;
        }
        
    }
    
    final_iter = it;
    final_norm = norm;
    *itnums = final_iter;
        
}


// Metric 22
class TMOPHyperelasticModel022 : public HyperelasticModel
{
private: double& tauptr;
    
public:
    TMOPHyperelasticModel022(double& tauval): tauptr(tauval) {}
    
    virtual double EvalW(const DenseMatrix &Jpt) const;
    
    virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;
    
    virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                           const double weight, DenseMatrix &A) const;
    
    ~TMOPHyperelasticModel022() {}
    
};

//M022 = (I1I2 - 2*I2) / (2I2-2*Beta)
double TMOPHyperelasticModel022::EvalW(const DenseMatrix &Jpt) const
{
    const double I1 = Dim2Invariant1(Jpt), I2 = Dim2Invariant2(Jpt);
    double beta = tauptr;
    return  (I1*I2 - 2.*I2)/(2.*I2-2.*beta);
    
}

void TMOPHyperelasticModel022::EvalP(const DenseMatrix &Jpt,
                                     DenseMatrix &P) const
{
    const double I1 = Dim2Invariant1(Jpt), I2 = Dim2Invariant2(Jpt);
    double beta = tauptr;
    double alpha = (2.*I2 - 2.*beta);
    
    Dim2Invariant1_dM(Jpt, P);
    P *= I2*alpha;
    
    
    DenseMatrix PP(P.Size());
    Dim2Invariant2_dM(Jpt, PP); //DI2/DM
    PP *= (4.*beta - 2.*I1*beta);
    
    P += PP;
    P *= 1./(alpha*alpha);
}

void TMOPHyperelasticModel022::AssembleH(const DenseMatrix &Jpt,
                                         const DenseMatrix &DS,
                                         const double weight,
                                         DenseMatrix &A) const
{
    const int dof = DS.Height(), dim = DS.Width();
    const double I1 = Dim2Invariant1(Jpt), I2 = Dim2Invariant2(Jpt);
    DenseMatrix dI1_dM(dim), dI1_dMdM(dim), dI2_dM(dim), dI2_dMdM(dim);
    Dim2Invariant1_dM(Jpt, dI1_dM);
    Dim2Invariant2_dM(Jpt, dI2_dM);
    double beta = tauptr;
    double alpha = (2.*I2 - 2.*beta);
    double gamma = (I1*I2 - 2.*I2);
    
    for (int r = 0; r < dim; r++)
        for (int c = 0; c < dim; c++)
        {
            Dim2Invariant1_dMdM(Jpt, r, c, dI1_dMdM);
            Dim2Invariant2_dMdM(Jpt, r, c, dI2_dMdM);
            // Compute each entry of d(Grc)_dJ.
            for (int rr = 0; rr < dim; rr++)
            {
                for (int cc = 0; cc < dim; cc++)
                {
                    /*const double entry_rr_cc2 =
                    (alpha*alpha*(2.*dI2_dM(rr,cc)*(I1*dI2_dM(r,c) + dI1_dM(r,c)*I2 -2.*dI2_dM(r,c))
        + alpha*(dI1_dM(rr,cc)*dI2_dM(r,c)+I1*dI2_dMdM(rr,cc)+I2*dI1_dMdM(rr,cc)+dI1_dM(r,c)*dI2_dM(rr,cc)
                   -2.*dI2_dMdM(rr,cc))
        - gamma*2.*dI2_dMdM(rr,cc) - 2.*dI2_dM(r,c)*(dI1_dM(rr,cc)*I2 + I1*dI2_dM(rr,cc) - 2.*dI2_dM(rr,cc)))
                    - (alpha*(I1*dI2_dM(r,c)+dI1_dM(r,c)*I2-2*dI2_dM(r,c)) - (gamma*2.*dI2_dM(r,c)))*(2.*alpha*2.*dI2_dM(rr,cc)))/(alpha*alpha*alpha*alpha);*/
                    
                    const double entry_rr_cc =
                    (alpha*alpha*(4*dI2_dM(rr,cc)*I2*dI1_dM(r,c) + 2*I2*I2*dI1_dMdM(rr,cc)
                    -2*beta*dI2_dM(rr,cc)*dI1_dM(r,c) - 2*beta*I2*dI1_dMdM(rr,cc)
                    +4*beta*dI2_dMdM(rr,cc)
                    -2*beta*dI1_dM(rr,cc)*dI2_dM(r,c)-2*beta*I1*dI2_dMdM(rr,cc))-
                (2*I2*I2*dI1_dM(r,c)-2*beta*I2*dI1_dM(r,c)+4*beta*dI2_dM(r,c)-2*I1*beta*dI2_dM(r,c))*(4*alpha*dI2_dM(rr,cc)))/(alpha*alpha*alpha*alpha);
                    
                    
                    for (int i = 0; i < dof; i++)
                        for (int j = 0; j < dof; j++)
                        {
                            A(i+r*dof, j+rr*dof) +=
                            weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                        }
                }
            }
        }
}


// Metric 211 (I2-1)^2 + (-I2 + sqrt(I2^2 + beta))
class TMOPHyperelasticModel211 : public HyperelasticModel
{
public:
    virtual double EvalW(const DenseMatrix &Jpt) const;
    
    virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;
    
    virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                           const double weight, DenseMatrix &A) const;
};

//Metric 211 (I2-1)^2 + (-I2 + sqrt(I2^2 + beta))
double TMOPHyperelasticModel211::EvalW(const DenseMatrix &Jpt) const
{
    double det = Dim2Invariant2(Jpt);
    double beta = 1e-4;
    return  (det*det) -3.*det + sqrt(det*det + beta) + 1.;
}

void TMOPHyperelasticModel211::EvalP(const DenseMatrix &Jpt,
                                     DenseMatrix &P) const
{
    double det = Dim2Invariant2(Jpt);
    double beta = 1e-4;
    double alpha = det/sqrt(det*det+beta);
    
    Dim2Invariant2_dM(Jpt, P);     //DI2/DM
    P *= (2.*det - 3. + alpha);
}

void TMOPHyperelasticModel211::AssembleH(const DenseMatrix &Jpt,
                                         const DenseMatrix &DS,
                                         const double weight,
                                         DenseMatrix &A) const
{
    const int dof = DS.Height(), dim = DS.Width();
    const double I2 = Dim2Invariant2(Jpt);
    DenseMatrix dI2_dM(dim), dI2_dMdM(dim);
    Dim2Invariant2_dM(Jpt, dI2_dM);
    double det = I2;
    double beta = 1e-4;
    double alpha = det/sqrt(det*det+beta);
    double alpha2 = pow(det*det+beta,1.5);
    
    for (int r = 0; r < dim; r++)
        for (int c = 0; c < dim; c++)
        {
            Dim2Invariant2_dMdM(Jpt, r, c, dI2_dMdM);
            // Compute each entry of d(Grc)_dJ.
            for (int rr = 0; rr < dim; rr++)
            {
                for (int cc = 0; cc < dim; cc++)
                {
                    const double entry_rr_cc =
                    2.*I2*dI2_dMdM(rr,cc) - 3.*dI2_dMdM(rr,cc) + alpha*dI2_dMdM(rr,cc)
                    + 2.*dI2_dM(rr,cc)*dI2_dM(r,c)
                    + dI2_dM(rr,cc)*dI2_dM(r,c)*1./(sqrt(det*det+beta))
                    -(I2*I2/alpha2)*dI2_dM(r,c)*dI2_dM(rr,cc);
                    
                    
                    
                    for (int i = 0; i < dof; i++)
                        for (int j = 0; j < dof; j++)
                        {
                            A(i+r*dof, j+rr*dof) +=
                            weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                        }
                }
            }
        }
}

// Metric 56
class TMOPHyperelasticModel056 : public HyperelasticModel
{
public:
    virtual double EvalW(const DenseMatrix &Jpt) const;
    
    virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;
    
    virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                           const double weight, DenseMatrix &A) const;
    
};

//M056 = 1/2 ( sqrt(tau) - 1/sqrt(tau))^2  = 1/2 (I2 + 1/I2) - 1
double TMOPHyperelasticModel056::EvalW(const DenseMatrix &Jpt) const
{
    const double I2 = Dim2Invariant2(Jpt);
    return  0.5*(I2 + (1./I2)) - 1.;
}

void TMOPHyperelasticModel056::EvalP(const DenseMatrix &Jpt,
                                     DenseMatrix &P) const
{
    const double I2 = Dim2Invariant2(Jpt);
    double alpha = 1./(I2*I2);
    
    Dim2Invariant2_dM(Jpt, P);
    P *= (0.5 - 0.5*alpha);
}

void TMOPHyperelasticModel056::AssembleH(const DenseMatrix &Jpt,
                                         const DenseMatrix &DS,
                                         const double weight,
                                         DenseMatrix &A) const
{
    const int dof = DS.Height(), dim = DS.Width();
    const double I2 = Dim2Invariant2(Jpt);
    DenseMatrix dI2_dM(dim), dI2_dMdM(dim);
    Dim2Invariant2_dM(Jpt, dI2_dM);
    double alpha = 1./(I2*I2*I2);
    
    for (int r = 0; r < dim; r++)
        for (int c = 0; c < dim; c++)
        {
            Dim2Invariant2_dMdM(Jpt, r, c, dI2_dMdM);
            // Compute each entry of d(Grc)_dJ.
            for (int rr = 0; rr < dim; rr++)
            {
                for (int cc = 0; cc < dim; cc++)
                {
                    const double entry_rr_cc =
                    alpha*dI2_dM(rr,cc)*dI2_dM(r,c) + dI2_dMdM(rr,cc)*(0.5 - 0.5/(I2*I2));
                    
                    for (int i = 0; i < dof; i++)
                        for (int j = 0; j < dof; j++)
                        {
                            A(i+r*dof, j+rr*dof) +=
                            weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                        }
                }
            }
        }
}

// Metric 77
class TMOPHyperelasticModel077 : public HyperelasticModel
{
public:
    virtual double EvalW(const DenseMatrix &Jpt) const;
    
    virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;
    
    virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                           const double weight, DenseMatrix &A) const;
    
};

//M077 = 1/2 ( (tau) - 1/(tau))^2
double TMOPHyperelasticModel077::EvalW(const DenseMatrix &Jpt) const
{
    const double I2 = Dim2Invariant2(Jpt);
    return  0.5*(I2*I2 + 1./(I2*I2) - 2.);
}

void TMOPHyperelasticModel077::EvalP(const DenseMatrix &Jpt,
                                     DenseMatrix &P) const
{
    const double I2 = Dim2Invariant2(Jpt);
    double alpha = 1./(I2*I2*I2);
    
    Dim2Invariant2_dM(Jpt, P);
    P *= (I2 - alpha);
}

void TMOPHyperelasticModel077::AssembleH(const DenseMatrix &Jpt,
                                         const DenseMatrix &DS,
                                         const double weight,
                                         DenseMatrix &A) const
{
    const int dof = DS.Height(), dim = DS.Width();
    const double I2 = Dim2Invariant2(Jpt);
    DenseMatrix dI2_dM(dim), dI2_dMdM(dim);
    Dim2Invariant2_dM(Jpt, dI2_dM);
    double alpha = 1./(I2*I2*I2);
    double alpha2 = 1./(I2*I2*I2*I2);
    
    for (int r = 0; r < dim; r++)
        for (int c = 0; c < dim; c++)
        {
            Dim2Invariant2_dMdM(Jpt, r, c, dI2_dMdM);
            // Compute each entry of d(Grc)_dJ.
            for (int rr = 0; rr < dim; rr++)
            {
                for (int cc = 0; cc < dim; cc++)
                {
                    const double entry_rr_cc =
                    dI2_dMdM(rr,cc)*(I2 - alpha) + dI2_dM(rr,cc)*dI2_dM(r,c) +
                    3.*alpha2*dI2_dM(rr,cc)*dI2_dM(r,c);
                    
                    for (int i = 0; i < dof; i++)
                        for (int j = 0; j < dof; j++)
                        {
                            A(i+r*dof, j+rr*dof) +=
                            weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                        }
                }
            }
        }
}

//3D METRICS
// Metric 301 - (I1I2)^(1/2)/3 - 1
class TMOPHyperelasticModel301 : public HyperelasticModel
{
public:
    virtual double EvalW(const DenseMatrix &Jpt) const;
    
    virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;
    
    virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                           const double weight, DenseMatrix &A) const;
};

//M301 =  (I1I2)^(1/2)/3 - 1
double TMOPHyperelasticModel301::EvalW(const DenseMatrix &Jpt) const
{
    const double I1 = Dim3Invariant1(Jpt), I2 = Dim3Invariant2(Jpt);
    return  pow(I1*I2,0.5)/3. - 1.;
    
}

void TMOPHyperelasticModel301::EvalP(const DenseMatrix &Jpt,
                                     DenseMatrix &P) const
{
    const double I1 = Dim3Invariant1(Jpt), I2 = Dim3Invariant2(Jpt);
    double I1I2 = I1*I2;
    
    Dim3Invariant1_dM(Jpt, P);
    DenseMatrix PP(P.Size());
    Dim3Invariant2_dM(Jpt, PP); //DI2/DM
    
    P *= I2; //I2*DI1/DM
    PP *= (I1);//I1*DI2/DM
    P += PP;
    
    P *= (1./6.)*pow(I1I2,-0.5);
}

void TMOPHyperelasticModel301::AssembleH(const DenseMatrix &Jpt,
                                         const DenseMatrix &DS,
                                         const double weight,
                                         DenseMatrix &A) const
{
    const int dof = DS.Height(), dim = DS.Width();
    const double I1 = Dim3Invariant1(Jpt), I2 = Dim3Invariant2(Jpt);
    DenseMatrix dI1_dM(dim), dI1_dMdM(dim), dI2_dM(dim), dI2_dMdM(dim);
    Dim3Invariant1_dM(Jpt, dI1_dM);
    Dim3Invariant2_dM(Jpt, dI2_dM);
    double I1I2 = I1*I2;
    
    for (int r = 0; r < dim; r++)
        for (int c = 0; c < dim; c++)
        {
            Dim3Invariant1_dMdM(Jpt, r, c, dI1_dMdM);
            Dim3Invariant2_dMdM(Jpt, r, c, dI2_dMdM);
            // Compute each entry of d(Grc)_dJ.
            for (int rr = 0; rr < dim; rr++)
            {
                for (int cc = 0; cc < dim; cc++)
                {
                    const double entry_rr_cc =
                    (-1./12.)*pow(I1I2,-1.5)*(dI1_dM(rr,cc)*I2 + I1*dI2_dM(rr,cc))*(dI1_dM(r,c)*I2 + I1*dI2_dM(r,c))
                    +   (1./6)*pow(I1I2,-0.5)*(dI1_dMdM(rr,cc)*I2 +dI1_dM(r,c)*dI2_dM(rr,cc) + dI1_dM(rr,cc)*dI2_dM(r,c) + I1*dI2_dMdM(rr,cc));
                    for (int i = 0; i < dof; i++)
                        for (int j = 0; j < dof; j++)
                        {
                            A(i+r*dof, j+rr*dof) +=
                            weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                        }
                }
            }
        }
}


// Metric 2 - (I1I2)/9 - 1
class TMOPHyperelasticModel302 : public HyperelasticModel
{
public:
    virtual double EvalW(const DenseMatrix &Jpt) const;
    
    virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;
    
    virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                           const double weight, DenseMatrix &A) const;
};

//M302 = (I1I2) / 9 - 1
double TMOPHyperelasticModel302::EvalW(const DenseMatrix &Jpt) const
{
    const double I1 = Dim3Invariant1(Jpt), I2 = Dim3Invariant2(Jpt);
    return  I1*I2/9. - 1;
    
}

void TMOPHyperelasticModel302::EvalP(const DenseMatrix &Jpt,
                                     DenseMatrix &P) const
{
    const double I1 = Dim3Invariant1(Jpt), I2 = Dim3Invariant2(Jpt);
    Dim3Invariant1_dM(Jpt, P);
    P *= I2; //I2*DI1/DM
    //cout << I1 << " " << I2 << " " << " k10\n";
    DenseMatrix PP(P.Size());
    Dim3Invariant2_dM(Jpt, PP); //DI2/DM
    PP *= (I1);
    P += PP;
    
    P *= 1./(9.);
}

void TMOPHyperelasticModel302::AssembleH(const DenseMatrix &Jpt,
                                         const DenseMatrix &DS,
                                         const double weight,
                                         DenseMatrix &A) const
{
    const int dof = DS.Height(), dim = DS.Width();
    const double I1 = Dim3Invariant1(Jpt), I2 = Dim3Invariant2(Jpt);
    DenseMatrix dI1_dM(dim), dI1_dMdM(dim), dI2_dM(dim), dI2_dMdM(dim);
    Dim3Invariant1_dM(Jpt, dI1_dM);
    Dim3Invariant2_dM(Jpt, dI2_dM);
    
    for (int r = 0; r < dim; r++)
        for (int c = 0; c < dim; c++)
        {
            Dim3Invariant1_dMdM(Jpt, r, c, dI1_dMdM);
            Dim3Invariant2_dMdM(Jpt, r, c, dI2_dMdM);
            // Compute each entry of d(Grc)_dJ.
            for (int rr = 0; rr < dim; rr++)
            {
                for (int cc = 0; cc < dim; cc++)
                {
                    const double entry_rr_cc =
                    (1./9.)*(dI1_dMdM(rr,cc)*I2 +dI1_dM(r,c)*dI2_dM(rr,cc) + dI1_dM(rr,cc)*dI2_dM(r,c)
                           +dI2_dMdM(rr,cc)*I1);
                    
                    for (int i = 0; i < dof; i++)
                        for (int j = 0; j < dof; j++)
                        {
                            A(i+r*dof, j+rr*dof) +=
                            weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                        }
                }
            }
        }
}

// Metric 3 - I1/3 - 1
class TMOPHyperelasticModel303 : public HyperelasticModel
{
public:
    virtual double EvalW(const DenseMatrix &Jpt) const;
    
    virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;
    
    virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                           const double weight, DenseMatrix &A) const;
};

//M303 - I1/3 - 1
double TMOPHyperelasticModel303::EvalW(const DenseMatrix &Jpt) const
{
    const double I1 = Dim3Invariant1(Jpt);
    return  I1/3. - 1;
    
}

void TMOPHyperelasticModel303::EvalP(const DenseMatrix &Jpt,
                                     DenseMatrix &P) const
{
    const double I1 = Dim3Invariant1(Jpt);
    Dim3Invariant1_dM(Jpt, P);
    P *= (1./3.); //(1/3)*DI1/DM
}

void TMOPHyperelasticModel303::AssembleH(const DenseMatrix &Jpt,
                                         const DenseMatrix &DS,
                                         const double weight,
                                         DenseMatrix &A) const
{
    const int dof = DS.Height(), dim = DS.Width();
    const double I1 = Dim3Invariant1(Jpt);
    DenseMatrix dI1_dMdM(dim);
    
    for (int r = 0; r < dim; r++)
        for (int c = 0; c < dim; c++)
        {
            Dim3Invariant1_dMdM(Jpt, r, c, dI1_dMdM);
            // Compute each entry of d(Grc)_dJ.
            for (int rr = 0; rr < dim; rr++)
            {
                for (int cc = 0; cc < dim; cc++)
                {
                    const double entry_rr_cc = (1./3.)*dI1_dMdM(rr,cc);
                    
                    for (int i = 0; i < dof; i++)
                        for (int j = 0; j < dof; j++)
                        {
                            A(i+r*dof, j+rr*dof) +=
                            weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                        }
                }
            }
        }
}


// Metric 21 - (I3^2/3)*I1 + I3^(-2/3)*I2 - 6
class TMOPHyperelasticModel321 : public HyperelasticModel
{
public:
    virtual double EvalW(const DenseMatrix &Jpt) const;
    
    virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;
    
    virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                           const double weight, DenseMatrix &A) const;
};

//M321 - (I3^2/3)*I1 + I3^(-2/3)*I2 - 6
double TMOPHyperelasticModel321::EvalW(const DenseMatrix &Jpt) const
{
    const double I1 = Dim3Invariant1(Jpt), I2 = Dim3Invariant2(Jpt), I3 = Dim3Invariant3(Jpt);
    return  pow(I3,2./3.)*I1 + pow(I3,-2./3.)*I2 - 6.;
    
}

void TMOPHyperelasticModel321::EvalP(const DenseMatrix &Jpt,
                                     DenseMatrix &P) const
{
    const double I1 = Dim3Invariant1(Jpt), I2 = Dim3Invariant2(Jpt), I3 = Dim3Invariant3(Jpt);
    DenseMatrix PP(P.Size());
    DenseMatrix PPP(P.Size());
    
    Dim3Invariant1_dM(Jpt, P); //DI1/DM
    Dim3Invariant2_dM(Jpt, PP); //DI2/DM
    Dim3Invariant3_dM(Jpt, PPP); //DI3/DM
    
    P *= pow(I3,2./3.);
    PP *= pow(I3,-2./3.);
    PPP *= I1*(2./3.)*pow(I3,-1./3.) + I2*(-2./3.)*pow(I3,-5./3.);
    
    P += PP;
    P += PPP;
    }

void TMOPHyperelasticModel321::AssembleH(const DenseMatrix &Jpt,
                                         const DenseMatrix &DS,
                                         const double weight,
                                         DenseMatrix &A) const
{
    const int dof = DS.Height(), dim = DS.Width();
    const double I1 = Dim3Invariant1(Jpt), I2 = Dim3Invariant2(Jpt), I3 = Dim3Invariant3(Jpt);
    DenseMatrix dI1_dM(dim), dI1_dMdM(dim), dI2_dM(dim), dI2_dMdM(dim), dI3_dM(dim), dI3_dMdM(dim);
    Dim3Invariant1_dM(Jpt, dI1_dM);
    Dim3Invariant2_dM(Jpt, dI2_dM);
    Dim3Invariant3_dM(Jpt, dI3_dM);
    double pwn43 =pow(I3,-4./3.);
    double pwn13 =pow(I3,-1./3.);
    double pwn53 =pow(I3,-5./3.);
    double pwn23 =pow(I3,-2./3.);
    double pwn83 =pow(I3,-8./3.);
    
    
    for (int r = 0; r < dim; r++)
        for (int c = 0; c < dim; c++)
        {
            Dim3Invariant1_dMdM(Jpt, r, c, dI1_dMdM);
            Dim3Invariant2_dMdM(Jpt, r, c, dI2_dMdM);
            Dim3Invariant3_dMdM(Jpt, r, c, dI3_dMdM);
            // Compute each entry of d(Grc)_dJ.
            for (int rr = 0; rr < dim; rr++)
            {
                for (int cc = 0; cc < dim; cc++)
                {
                    const double entry_rr_cc =
                    (-2./9.)*pwn43*dI3_dM(rr,cc)*dI3_dM(r,c)*I1 + (2./3.)*pwn13*dI3_dMdM(rr,cc)*I1 +
                    + (2./3.)*pwn13*dI3_dM(r,c)*dI1_dM(rr,cc) +  //T1
                    
                    (2./3.)*pwn13*dI3_dM(rr,cc)*dI1_dM(r,c)+pow(I3,2./3.)*dI1_dMdM(rr,cc) + //T2
                    
                    (10./9.)*pwn83*dI3_dM(rr,cc)*dI3_dM(r,c)*I2 + (-2./3.)*pwn53*dI3_dMdM(rr,cc)*I2 +
                    + (-2./3.)*pwn53*dI3_dM(r,c)*dI2_dM(rr,cc) +  //T3
                    
                    (-2./3.)*pwn53*dI3_dM(rr,cc)*dI2_dM(r,c)+pwn23*dI2_dMdM(rr,cc); //T4
                    
                    for (int i = 0; i < dof; i++)
                        for (int j = 0; j < dof; j++)
                        {
                            A(i+r*dof, j+rr*dof) +=
                            weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                        }
                }
            }
        }
}

//Size metrics
//Metric 315 = (tua-1)^2 = (I3^2-2*I3+1)
class TMOPHyperelasticModel315 : public HyperelasticModel
{
public:
    virtual double EvalW(const DenseMatrix &Jpt) const;
    
    virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;
    
    virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                           const double weight, DenseMatrix &A) const;
};

//M315 = (I3^2-2*I3+1)
double TMOPHyperelasticModel315::EvalW(const DenseMatrix &Jpt) const
{
    const double I3 = Dim3Invariant3(Jpt);
    return  I3*I3 - 2.*I3 + 1;
}

void TMOPHyperelasticModel315::EvalP(const DenseMatrix &Jpt,
                                     DenseMatrix &P) const
{
    const double I3 = Dim3Invariant3(Jpt);
    
    Dim3Invariant3_dM(Jpt, P);
    P *= (2.*I3 - 1.);
}

void TMOPHyperelasticModel315::AssembleH(const DenseMatrix &Jpt,
                                         const DenseMatrix &DS,
                                         const double weight,
                                         DenseMatrix &A) const
{
    const int dof = DS.Height(), dim = DS.Width();
    const double I3 = Dim3Invariant3(Jpt);
    DenseMatrix dI3_dM(dim), dI3_dMdM(dim);
    Dim3Invariant3_dM(Jpt, dI3_dM);
    
    for (int r = 0; r < dim; r++)
        for (int c = 0; c < dim; c++)
        {
            Dim3Invariant3_dMdM(Jpt, r, c, dI3_dMdM);
            // Compute each entry of d(Grc)_dJ.
            for (int rr = 0; rr < dim; rr++)
            {
                for (int cc = 0; cc < dim; cc++)
                {
                    const double entry_rr_cc =
                    2.*dI3_dMdM(rr,cc)*(I3 - 1.) + 2.*dI3_dM(rr,cc)*dI3_dM(r,c);
                    
                    for (int i = 0; i < dof; i++)
                        for (int j = 0; j < dof; j++)
                        {
                            A(i+r*dof, j+rr*dof) +=
                            weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                        }
                }
            }
        }
}
//Metric 316 = 1/2 ( sqrt(tau) - 1/sqrt(tau))^2  = 1/2 (I3 + 1/I3) - 1
class TMOPHyperelasticModel316 : public HyperelasticModel
{
public:
    virtual double EvalW(const DenseMatrix &Jpt) const;
    
    virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;
    
    virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                           const double weight, DenseMatrix &A) const;
};

//M316 = 1/2 ( sqrt(tau) - 1/sqrt(tau))^2  = 1/2 (I3 + 1/I3) - 1
double TMOPHyperelasticModel316::EvalW(const DenseMatrix &Jpt) const
{
    const double I3 = Dim3Invariant3(Jpt);
    return  0.5*(I3 + (1./I3)) - 1.;
}

void TMOPHyperelasticModel316::EvalP(const DenseMatrix &Jpt,
                                     DenseMatrix &P) const
{
    const double I3 = Dim3Invariant3(Jpt);
    double alpha = 1./(I3*I3);
    
    Dim3Invariant3_dM(Jpt, P);
    P *= (0.5 - 0.5*alpha);
}

void TMOPHyperelasticModel316::AssembleH(const DenseMatrix &Jpt,
                                         const DenseMatrix &DS,
                                         const double weight,
                                         DenseMatrix &A) const
{
    const int dof = DS.Height(), dim = DS.Width();
    const double I3 = Dim3Invariant3(Jpt);
    DenseMatrix dI3_dM(dim), dI3_dMdM(dim);
    Dim3Invariant3_dM(Jpt, dI3_dM);
    double alpha = 1./(I3*I3*I3);
    
    for (int r = 0; r < dim; r++)
        for (int c = 0; c < dim; c++)
        {
            Dim3Invariant3_dMdM(Jpt, r, c, dI3_dMdM);
            // Compute each entry of d(Grc)_dJ.
            for (int rr = 0; rr < dim; rr++)
            {
                for (int cc = 0; cc < dim; cc++)
                {
                    const double entry_rr_cc =
                    alpha*dI3_dM(rr,cc)*dI3_dM(r,c) + dI3_dMdM(rr,cc)*(0.5 - 0.5/(I3*I3));
                    
                    for (int i = 0; i < dof; i++)
                        for (int j = 0; j < dof; j++)
                        {
                            A(i+r*dof, j+rr*dof) +=
                            weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                        }
                }
            }
        }
}


// Metric 52 - Untangling
class TMOPHyperelasticModel352 : public HyperelasticModel
{
private: double& tauptr;
    
public:
    TMOPHyperelasticModel352(double& tauval): tauptr(tauval) {}
    
    virtual double EvalW(const DenseMatrix &Jpt) const;
    
    virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;
    
    virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                           const double weight, DenseMatrix &A) const;
    
    ~TMOPHyperelasticModel352() {}
    
};

//M351 = (I3^2 -2I3+1) / (2I3-2*Beta)
double TMOPHyperelasticModel352::EvalW(const DenseMatrix &Jpt) const
{
    const double I3 = Dim3Invariant3(Jpt);
    double beta = tauptr;
    double val = (I3*I3 - 2.*I3 + 1.)/(2.*I3-2.*beta);
    //cout << beta << " " << val << " " << I3 << " "<< 2.*I3-2.*beta << " k10 beta\n";
    return  (I3*I3 - 2.*I3 + 1.)/(2.*I3-2.*beta);
    
}

void TMOPHyperelasticModel352::EvalP(const DenseMatrix &Jpt,
                                     DenseMatrix &P) const
{
    const double I3 = Dim3Invariant3(Jpt);
    double beta = tauptr;
    double alpha = (2.*I3 - 2.*beta);
    double alphainv = 1./alpha;
    
    Dim3Invariant3_dM(Jpt, P);
    P *= (2*I3*I3 - 4.*beta*I3 + 4.*beta - 2.)*alphainv*alphainv;  //T1
}

void TMOPHyperelasticModel352::AssembleH(const DenseMatrix &Jpt,
                                         const DenseMatrix &DS,
                                         const double weight,
                                         DenseMatrix &A) const
{
    const int dof = DS.Height(), dim = DS.Width();
    const double I3 = Dim3Invariant3(Jpt);
    DenseMatrix dI3_dM(dim), dI3_dMdM(dim);
    Dim3Invariant3_dM(Jpt, dI3_dM);
    double beta = tauptr;
    double alpha = (2.*I3 - 2.*beta);
    double alphainv = 1./alpha;
    
    
    for (int r = 0; r < dim; r++)
        for (int c = 0; c < dim; c++)
        {
            Dim3Invariant3_dMdM(Jpt, r, c, dI3_dMdM);
            // Compute each entry of d(Grc)_dJ.
            for (int rr = 0; rr < dim; rr++)
            {
                for (int cc = 0; cc < dim; cc++)
                {
        const double entry_rr_cc = (alpha*alpha*(4.*I3*dI3_dM(rr,cc)*dI3_dM(r,c) + 2*I3*I3*dI3_dMdM(rr,cc) - 4.*beta*dI3_dM(r,c)*dI3_dM(rr,cc) - 4.*beta*I3*dI3_dMdM(rr,cc)) - 4.*alpha*dI3_dM(rr,cc)*(2*I3*I3-4.*beta*I3 + 4.*beta-2.)*dI3_dM(r,c))/(alpha*alpha*alpha*alpha);
                    
                    for (int i = 0; i < dof; i++)
                        for (int j = 0; j < dof; j++)
                        {
                            A(i+r*dof, j+rr*dof) +=
                            weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                        }
                }
            }
        }
}




#define BIG_NUMBER 1e+100 // Used when a matrix is outside the metric domain.
#define NBINS 25          // Number of intervals in the metric histogram.
#define GAMMA 0.9         // Used for composite metrics 73, 79, 80.
#define BETA0 0.01        // Used for adaptive pseudo-barrier metrics.
#define TAU0_EPS 0.001    // Used for adaptive shifted-barrier metrics.

int main (int argc, char *argv[])
{
    Mesh *mesh;
    char vishost[] = "localhost";
    int  visport   = 19916;
    int  ans;
    vector<double> logvec (100);
    double weight_fun(const Vector &x);
    double tstart_s=clock();
    
    
    bool dump_iterations = false;
    
    // 3. Read the mesh from the given mesh file. We can handle triangular,
    //    quadrilateral, tetrahedral or hexahedral elements with the same code.
    const char *mesh_file = "../data/tipton.mesh";
    int order = 1;
    bool static_cond = false;
    bool visualization = 1;
    
    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree) or -1 for"
                   " isoparametric space.");
    args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                   "--no-static-condensation", "Enable static condensation.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(cout);
        return 1;
    }
    args.PrintOptions(cout);
    cout << mesh_file << " about to read a mesh file\n";
    mesh = new Mesh(mesh_file, 1, 1,false);
    cout << "read a mesh file\n";
    
    int dim = mesh->Dimension();
    
    // 4. Refine the mesh to increase the resolution. In this example we do
    //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
    //    largest number that gives a final mesh with no more than 1000
    //    elements.
    {
        int ref_levels =
        (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
        cout << "enter refinement levels [" << ref_levels << "] --> " << flush;
        cin >> ref_levels;
        for (int l = 0; l < ref_levels; l++)
        {
            mesh->UniformRefinement();
        }
        
        logvec[0]=ref_levels;
    }
        cout << "refinements specified\n";
    
    // 5. Define a finite element space on the mesh. Here we use vector finite
    //    elements which are tensor products of quadratic finite elements. The
    //    dimensionality of the vector finite element space is specified by the
    //    last parameter of the FiniteElementSpace constructor.
    cout << "Mesh curvature: ";
    if (mesh->GetNodes())
    {
        cout << mesh->GetNodes()->OwnFEC()->Name();
    }
    else
    {
        cout << "(NONE)";
    }
    cout << endl;
    
    int mesh_poly_deg = 1;
    cout <<
    "Enter polynomial degree of mesh finite element space:\n"
    "0) QuadraticPos (quads only)\n"
    "p) Degree p >= 1\n"
    " --> " << flush;
    cin >> mesh_poly_deg;
    FiniteElementCollection *fec;
    if (mesh_poly_deg <= 0)
    {
        fec = new QuadraticPosFECollection;
        mesh_poly_deg = 2;
    }
    else
    {
        fec = new H1_FECollection(mesh_poly_deg, dim);
    }
    FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec, dim);
    logvec[1]=mesh_poly_deg;
    
    // 6. Make the mesh curved based on the above finite element space. This
    //    means that we define the mesh elements through a fespace-based
    //    transformation of the reference element.
    mesh->SetNodalFESpace(fespace);
    
    // 7. Set up the right-hand side vector b. In this case we do not need to use
    //    a LinearForm object because b=0.
    Vector b(fespace->GetVSize());
    b = 0.0;
    
    // 8. Get the mesh nodes (vertices and other quadratic degrees of freedom in
    //    the finite element space) as a finite element grid function in fespace.
    GridFunction *x;
    x = mesh->GetNodes();
    
    
    // 9. Define a vector representing the minimal local mesh size in the mesh
    //    nodes. We index the nodes using the scalar version of the degrees of
    //    freedom in fespace.
    Vector h0(fespace->GetNDofs());
    h0 = numeric_limits<double>::infinity();

    {
        Array<int> dofs;
        // loop over the mesh elements
        for (int i = 0; i < fespace->GetNE(); i++)
        {
            // get the local scalar element degrees of freedom in dofs
            fespace->GetElementDofs(i, dofs);
            // adjust the value of h0 in dofs based on the local mesh size
            for (int j = 0; j < dofs.Size(); j++)
            {
                h0(dofs[j]) = min(h0(dofs[j]), mesh->GetElementSize(i));
            }
        }
    }
    
    // 10. Add a random perturbation of the nodes in the interior of the domain.
    //     We define a random grid function of fespace and make sure that it is
    //     zero on the boundary and its values are locally of the order of h0.
    //     The latter is based on the DofToVDof() method which maps the scalar to
    //     the vector degrees of freedom in fespace.
    GridFunction rdm(fespace);
    double jitter = 0.0; // perturbation scaling factor
    /* k10commenting this for input
    cout << "Enter jitter --> " << flush;
    cin >> jitter;
    */
    rdm.Randomize();
    rdm -= 0.2; // shift to random values in [-0.5,0.5]
    rdm *= jitter;
    {
        // scale the random values to be of order of the local mesh size
        for (int i = 0; i < fespace->GetNDofs(); i++)
            for (int d = 0; d < dim; d++)
            {
                rdm(fespace->DofToVDof(i,d)) *= h0(i);
            }
        
        Array<int> vdofs;
        // loop over the boundary elements
        for (int i = 0; i < fespace->GetNBE(); i++)
        {
            // get the vector degrees of freedom in the boundary element
            fespace->GetBdrElementVDofs(i, vdofs);
            // set the boundary values to zero
            for (int j = 0; j < vdofs.Size(); j++)
            {
                rdm(vdofs[j]) = 0.0;
            }
        }
    }
    *x -= rdm;
    
    // 11. Save the perturbed mesh to a file. This output can be viewed later
    //     using GLVis: "glvis -m perturbed.mesh".
    {
        ofstream mesh_ofs("perturbed.mesh");
        mesh->Print(mesh_ofs);
    }
    
    // 12. (Optional) Send the initially perturbed mesh with the vector field
    //     representing the displacements to the original mesh to GLVis.
    /* k10 commenting this for input
    cout << "Visualize the initial random perturbation? [0/1] --> ";
    cin >> ans;
    if (ans)
    {
        osockstream sol_sock(visport, vishost);
        sol_sock << "solution\n";
        mesh->Print(sol_sock);
        rdm.Save(sol_sock);
        sol_sock.send();
    }
    k10*/
    
    int smoother = 1;
    
    // 14. Simple mesh smoothing can be performed by relaxing the node coordinate
    //     grid function x with the matrix A and right-hand side b. This process
    //     converges to the solution of Ax=b, which we solve below with PCG. Note
    //     that the computed x is the A-harmonic extension of its boundary values
    //     (the coordinates of the boundary vertices). Furthermore, note that
    //     changing x automatically changes the shapes of the elements in the
    //     mesh. The vector field that gives the displacements to the perturbed
    //     positions is saved in the grid function x0.
    GridFunction x0(fespace);
    x0 = *x;
    
    L2_FECollection mfec(mesh_poly_deg, mesh->Dimension(), BasisType::GaussLobatto); //this for vis
    FiniteElementSpace mfes(mesh, &mfec, 1);
    GridFunction metric(&mfes);
    
    if (smoother == 1)
    {
        HyperelasticModel *model;
        NonlinearFormIntegrator *nf_integ;
        
        Coefficient *c = NULL;
        TargetJacobian *tj = NULL;
        
        HyperelasticModel *model2; //this is for combo
        Coefficient *c2 = NULL;     //this is for combo
        TargetJacobian *tj2 = NULL; //this is for combo
        
        //{IDEAL, IDEAL_EQ_SIZE, IDEAL_INIT_SIZE,
        // IDEAL_CUSTOM_SIZE, TARGET_MESH, ALIGNED}
        int tjtype;
        int modeltype;
        cout <<
        "Specify Target Jacobian type:\n"
        "1) IDEAL\n"
        "2) IDEAL_EQ_SIZE\n"
        "3) IDEAL_INIT_SIZE\n"
        "4) IDEAL_EQ_SCALE_SIZE\n"
        " --> " << flush;
        cin >> tjtype;
        
        tj    = new TargetJacobian(TargetJacobian::IDEAL);
        if (tjtype == 1)
        {
            tj    = new TargetJacobian(TargetJacobian::IDEAL);
            cout << " you chose Target Jacobian - Ideal \n";
        }
        else if (tjtype == 2)
        {
            tj    = new TargetJacobian(TargetJacobian::IDEAL_EQ_SIZE);
            cout << " you chose Target Jacobian - Ideal_eq_size \n";
        }
        else if (tjtype == 3)
        {
            tj    = new TargetJacobian(TargetJacobian::IDEAL_INIT_SIZE);
            cout << " you chose Target Jacobian - Ideal_init_size \n";
        }
        else if (tjtype == 4)
        {
            tj    = new TargetJacobian(TargetJacobian::IDEAL_EQ_SCALE_SIZE);
            cout << " you chose Target Jacobian - Ideal_eq_scale_size \n";
        }
        else
        {
            cout << tjtype;
            cout << "You did not choose a valid option\n";
            cout << "Target Jacobian will default to IDEAL\n";
        }
        
        //Metrics for 2D
        double tauval = -0.1;
        if (dim==2) {
            cout << "Choose optimization metric:\n"
            << "1  : |T|^2 \n"
            << "     shape.\n"
            << "2  : 0.5 |T|^2 / tau  - 1 \n"
            << "     shape, condition number metric.\n"
            << "7  : |T - T^-t|^2 \n"
            << "     shape+size.\n"
            << "22  : |T|^2 - 2*tau / (2*tau - 2*tau_0)\n"
            << "     untangling.\n"
            << "56  : 0.5*(sqrt(tau) - 1/sqrt(tau))^2\n"
            << "     size metric\n"
            << "77  : 0.5*(tau - 1/tau)^2\n"
            << "     size metric\n"
            << "211  : (tau-1)^2 - tau + sqrt(tau^2+beta)\n"
            << "      untangling.\n"
            " --> " << flush;
            double tauval = -0.1;
            cin >> modeltype;
            model    = new TMOPHyperelasticModel001;
            if (modeltype == 1)
            {
                model = new TMOPHyperelasticModel001;
                cout << " you chose metric 1 \n";
            }
            else if (modeltype == 2)
            {
                model = new TMOPHyperelasticModel002;
                cout << " you chose metric 2 \n";
            }
            else if (modeltype == 7)
            {
                model = new TMOPHyperelasticModel007;
                cout << " you chose metric 7 \n";
            }
            else if (modeltype == 22)
            {
                model = new TMOPHyperelasticModel022(tauval);
                cout << " you chose metric 22\n";
            }
            else if (modeltype == 56)
            {
                model = new TMOPHyperelasticModel056;
                cout << " you chose metric 56\n";
            }
            else if (modeltype == 77)
            {
                model = new TMOPHyperelasticModel077;
                cout << " you chose metric 77\n";
            }
            else if (modeltype == 211)
            {
                model = new TMOPHyperelasticModel211;
                cout << " you chose metric 211 \n";
            }
            else
            {
                cout << "You did not choose a valid option\n";
                cout << "Model type will default to 1\n";
                cout << modeltype;
            }
        }
        else {  //Metrics for 3D
            cout << "Choose optimization metric:\n"
            << "1  : (|T||T^-1|)/3 - 1 \n"
            << "     shape.\n"
            << "2  : (|T|^2|T^-1|^2)/9 - 1 \n"
            << "     shape.\n"
            << "3  : (|T|^2)/3*tau^(2/3) - 1 \n"
            << "     shape.\n"
            << "15  : (tau-1)^2\n"
            << "     size.\n"
            << "16  : 1/2 ( sqrt(tau) - 1/sqrt(tau))^2\n"
            << "     size.\n"
            << "21  : (|T-T^-t|^2)\n"
            << "     shape+size.\n"
            << "52  : (tau-1)^2/ (2*tau - 2*tau_0)\n"
            << "     untangling.\n"
            " --> " << flush;
            double tauval = -0.1;
            cin >> modeltype;
            model    = new TMOPHyperelasticModel302;
            if (modeltype == 1)
            {
                model = new TMOPHyperelasticModel301;
                cout << " you chose metric 1 \n";
            }
            if (modeltype == 2)
            {
                model = new TMOPHyperelasticModel302;
                cout << " you chose metric 2 \n";
            }
            else if (modeltype == 3)
            {
                model = new TMOPHyperelasticModel303;
                cout << " you chose metric 3\n";
            }
            else if (modeltype == 15)
            {
                model = new TMOPHyperelasticModel315;
                cout << " you chose metric 15\n";
            }
            else if (modeltype == 16)
            {
                model = new TMOPHyperelasticModel316;
                cout << " you chose metric 16\n";
            }
            else if (modeltype == 21)
            {
                model = new TMOPHyperelasticModel321;
                cout << " you chose metric 21\n";
            }
            else if (modeltype == 52)
            {
                model = new TMOPHyperelasticModel352(tauval);
                cout << " you chose metric 52\n";
            }
            else
            {
                cout << "You did not choose a valid option\n";
                cout << "Model type will default to 2\n";
                cout << modeltype;
            }
        }
        logvec[2]=tjtype;
        logvec[3]=modeltype;
        
        tj->SetNodes(*x);
        tj->SetInitialNodes(x0);
        HyperelasticNLFIntegrator *he_nlf_integ;
        he_nlf_integ = new HyperelasticNLFIntegrator(model, tj);
        
        int ptflag = 1; //if 1 - GLL, else uniform
        int nptdir = 8; //number of sample points in each direction
        const IntegrationRule *ir =
        &IntRulesLo.Get(fespace->GetFE(0)->GetGeomType(),nptdir); //this for GLL points "LO"
        he_nlf_integ->SetIntegrationRule(*ir);
        if (ptflag==1) {
            cout << "Sample point distribution is GLL based\n";
        }
        else {
            const IntegrationRule *ir =
            &IntRulesCU.Get(fespace->GetFE(0)->GetGeomType(),nptdir); //this for uniform points "CU"
            he_nlf_integ->SetIntegrationRule(*ir);
            cout << "Sample point distribution is uniformly spaced\n";
        }
        
        //
        nf_integ = he_nlf_integ;
        NonlinearForm a(fespace);
        
        // This is for trying a combo of two integrators
        const int combomet = 0;
        if (combomet==1) {
            c = new ConstantCoefficient(0.5);  //weight of original metric
            he_nlf_integ->SetCoefficient(*c);
            nf_integ = he_nlf_integ;
            a.AddDomainIntegrator(nf_integ);
            
            
            tj2    = new TargetJacobian(TargetJacobian::IDEAL_EQ_SCALE_SIZE);
            model2 = new TMOPHyperelasticModel077;
            tj2->SetNodes(*x);
            tj2->SetInitialNodes(x0);
            HyperelasticNLFIntegrator *he_nlf_integ2;
            he_nlf_integ2 = new HyperelasticNLFIntegrator(model2, tj2);
            const IntegrationRule *ir =
            &IntRulesLo.Get(fespace->GetFE(0)->GetGeomType(),nptdir); //this for metric
            he_nlf_integ2->SetIntegrationRule(*ir);
            
            
            //c2 = new ConstantCoefficient(2.5);
            //he_nlf_integ2->SetCoefficient(*c2);     //weight of new metric
            
            FunctionCoefficient rhs_coef (weight_fun);
            he_nlf_integ2->SetCoefficient(rhs_coef);     //weight of new metric as function
            //he_nlf_integ2->SetLimited(1e-4, x0);
            cout << "You have added a combo metric \n";
            a.AddDomainIntegrator(he_nlf_integ2);
        }
        else{
            a.AddDomainIntegrator(nf_integ);
        }
        //
        
        
        InterpolateHyperElasticModel(*model, *tj, *mesh, metric);
        osockstream sol_sock2(visport, vishost);
        sol_sock2 << "solution\n";
        mesh->Print(sol_sock2);
        metric.Save(sol_sock2);
        sol_sock2.send();
        sol_sock2 << "keys " << "JRem" << endl;
        
        // Set essential vdofs by hand for x = 0 and y = 0 (2D). These are
        // attributes 1 and 2.
        const int nd  = x->FESpace()->GetBE(0)->GetDof();
        int n = 0;
        for (int i = 0; i < mesh->GetNBE(); i++)
        {
            const int attr = mesh->GetBdrElement(i)->GetAttribute();
            if (attr == 1 || attr == 2) { n += nd; }
            if (attr == 3) { n += nd * dim; }
        }
        Array<int> ess_vdofs(n), vdofs;
        n = 0;
        for (int i = 0; i < mesh->GetNBE(); i++)
        {
            const int attr = mesh->GetBdrElement(i)->GetAttribute();
            
            
            x->FESpace()->GetBdrElementVDofs(i, vdofs);
            if (attr == 1) // y = 0; fix y components.
            {
                for (int j = 0; j < nd; j++)
                { ess_vdofs[n++] = vdofs[j+nd]; }
            }
            else if (attr == 2) // x = 0; fix x components.
            {
                for (int j = 0; j < nd; j++)
                { ess_vdofs[n++] = vdofs[j]; }
            }
            else if (attr == 3) // vdofs on the other boundary.
            {
                for (int j = 0; j < vdofs.Size(); j++)
                { ess_vdofs[n++] = vdofs[j]; }
            }
        }
        a.SetEssentialVDofs(ess_vdofs);
        
        
        // Fix all boundary nodes.
        int bndrflag;
        cout <<
        "Allow boundary node movement:\n"
        "1 - yes\n"
        "2 - no\n"
        " --> " << flush;
        cin >> bndrflag;
        tj    = new TargetJacobian(TargetJacobian::IDEAL);
        if (bndrflag != 1)
        {
            //k10partend - the three lines below used to be uncommented
            Array<int> ess_bdr(mesh->bdr_attributes.Max());
            ess_bdr = 1;
            a.SetEssentialBC(ess_bdr);
        }
        logvec[4]=bndrflag;
        
        cout << "Choose linear smoother:\n"
        "0) l1-Jacobi\n"
        "1) CG\n"
        "2) MINRES\n" << " --> \n" << flush;
        cin >> ans;
        Solver *S;
        logvec[5]=ans;
        const double rtol = 1e-12;
        if (ans == 0)
        {
            cout << "Enter number of linear smoothing iterations -->\n " << flush;
            cin >> ans;
            S = new DSmoother(1, 1., ans);
        }
        else if (ans == 1)
        {
            cout << "Enter number of CG smoothing iterations -->\n " << flush;
            cin >> ans;
            CGSolver *cg = new CGSolver;
            cg->SetMaxIter(ans);
            cg->SetRelTol(rtol);
            cg->SetAbsTol(0.0);
            cg->SetPrintLevel(3);
            S = cg;
        }
        else
        {
            cout << "Enter number of MINRES smoothing iterations -->\n " << flush;
            cin >> ans;
            MINRESSolver *minres = new MINRESSolver;
            minres->SetMaxIter(ans);
            minres->SetRelTol(rtol);
            minres->SetAbsTol(0.0);
            minres->SetPrintLevel(3);
            S = minres;
            
        }
        logvec[6]=ans;
        cout << "Enter number of Newton iterations -->\n " << flush;
        cin >> ans;
        logvec[7]=ans;
        cout << "Initial strain energy : " << a.GetEnergy(*x) << endl;
        logvec[8]=a.GetEnergy(*x);
        
        // save original
        Vector xsav = *x;
        //set value of tau_0 for metric 22 and get min jacobian for mesh statistic
        //
        Array<int> dofs;
        tauval = 1e+6;
        double minjaco;
        const int NE = mesh->GetNE();
        const GridFunction &nodes = *mesh->GetNodes();
        for (int i = 0; i < NE; i++)
        {
            const FiniteElement &fe = *nodes.FESpace()->GetFE(i);
            const int dim = fe.GetDim();
            int nsp = ir->GetNPoints();
            int dof = fe.GetDof();
            DenseTensor Jtr(dim, dim, nsp);
            const GridFunction *nds;
            nds = &nodes;
            DenseMatrix dshape(dof, dim), pos(dof, dim);
            Array<int> xdofs(dof * dim);
            Vector posV(pos.Data(), dof * dim);
            nds->FESpace()->GetElementVDofs(i, xdofs);
            nds->GetSubVector(xdofs, posV);
            for (int j = 0; j < nsp; j++)
            {
                //cout << "point number " << j << "\n";
                fe.CalcDShape(ir->IntPoint(j), dshape);
                MultAtB(pos, dshape, Jtr(j));
                double det = Jtr(j).Det();
                tauval = min(tauval,det);
            }
        }
        minjaco = tauval;
        cout << "minimum jacobian in the original mesh is " << minjaco << " \n";
        int newtonits = 0;
        if (tauval>0)
        {
            tauval = 1e-1;
            RelaxedNewtonSolver *newt= new RelaxedNewtonSolver;
            newt->SetPreconditioner(*S);
            newt->SetMaxIter(ans);
            newt->SetRelTol(rtol);
            newt->SetAbsTol(0.0);
            newt->SetPrintLevel(1);
            newt->SetOperator(a);
            Vector b;
            cout << " Relaxed newton solver will be used \n";
            newt->Mult2(b, *x, *mesh, *ir, &newtonits, a );
            if (!newt->GetConverged())
                cout << "NewtonIteration : rtol = " << rtol << " not achieved."
                << endl;
        }
        else
        {
            if (dim==2) {
                model = new TMOPHyperelasticModel022(tauval);
                cout << "model 22 will be used since mesh has negative jacobians\n";
            }
            else{
                model = new TMOPHyperelasticModel352(tauval);
                cout << "model 52 will be used since mesh has negative jacobians\n";
            }
            tauval -= 100.;
            DescentNewtonSolver *newt= new DescentNewtonSolver;
            newt->SetPreconditioner(*S);
            newt->SetMaxIter(ans);
            newt->SetRelTol(rtol);
            newt->SetAbsTol(0.0);
            newt->SetPrintLevel(1);
            newt->SetOperator(a);
            Vector b;
            cout << " There are inverted elements in the mesh \n";
            cout << " Descent newton solver will be used \n";
            newt->Mult2(b, *x, *mesh, *ir, &newtonits, a, tauval );
            if (!newt->GetConverged())
                cout << "NewtonIteration : rtol = " << rtol << " not achieved."
                << endl;
        }
        
        logvec[9]=a.GetEnergy(*x);
        cout << "Final strain energy   : " << a.GetEnergy(*x) << endl;
        cout << "Initial strain energy was  : " << logvec[8] << endl;
        cout << "% change is  : " << (logvec[8]-logvec[9])*100/logvec[8] << endl;
        logvec[10] = (logvec[8]-logvec[9])*100/logvec[8];
        logvec[11] = newtonits;
        
        if (tj)
        {
            InterpolateHyperElasticModel(*model, *tj, *mesh, metric);
            osockstream sol_sock(visport, vishost);
            sol_sock << "solution\n";
            mesh->Print(sol_sock);
            metric.Save(sol_sock);
            sol_sock.send();
            sol_sock << "keys " << "JRem" << endl;
        }
        
        // 17. Get some mesh statistics
        double minjacs = 1.e+100;
        for (int i = 0; i < NE; i++)
        {
            const FiniteElement &fe = *nodes.FESpace()->GetFE(i);
            const int dim = fe.GetDim();
            int nsp = ir->GetNPoints();
            int dof = fe.GetDof();
            DenseTensor Jtr(dim, dim, nsp);
            const GridFunction *nds;
            nds = &nodes;
            DenseMatrix dshape(dof, dim), pos(dof, dim);
            Array<int> xdofs(dof * dim);
            Vector posV(pos.Data(), dof * dim);
            nds->FESpace()->GetElementVDofs(i, xdofs);
            nds->GetSubVector(xdofs, posV);
            for (int j = 0; j < nsp; j++)
            {
                //cout << "point number " << j << "\n";
                fe.CalcDShape(ir->IntPoint(j), dshape);
                MultAtB(pos, dshape, Jtr(j));
                double det = Jtr(j).Det();
                minjacs = min(minjacs,det);
            }
        }
        
        cout << "minimum jacobian before and after smoothing " << minjaco << " " << minjacs << " \n";
        
        
        delete S;
        delete model;
        delete c;
    }
    else
    {
        printf("unknown smoothing option, smoother = %d\n",smoother);
        exit(1);
    }
    
    // Define mesh displacement
    x0 -= *x;
    
    // 15. Save the smoothed mesh to a file. This output can be viewed later
    //     using GLVis: "glvis -m smoothed.mesh".
    {
        ofstream mesh_ofs("smoothed.mesh");
        mesh_ofs.precision(14);
        mesh->Print(mesh_ofs);
    }
    
    
    
    // 17. Free the used memory.
    delete fespace;
    delete fec;
    delete mesh;
    
    // Execution time
    double tstop_s=clock();
    cout << "The total time it took for this example's execution is: " << (tstop_s-tstart_s)/1000000. << " seconds\n";
    
    // write log to text file-k10
    cout << "How do you want to write log to a new file:\n"
    "0) New file\n"
    "1) Append\n" << " --> " << flush;
    cin >> ans;
    ofstream outputFile;
    if (ans==0)
    {
        outputFile.open("logfile.txt");
        outputFile << mesh_file << " ";
    }
    else
    {
        outputFile.open("logfile.txt",fstream::app);
        outputFile << "\n" << mesh_file << " ";
    }
    
    for (int i=0;i<12;i++)
    {
        outputFile << logvec[i] << " ";
    }
    outputFile.close();
    // pause 1 second.. this is because X11 cannot restart if you push stuff too soon
    usleep(1000000);
    //k10 end
    
}
double weight_fun(const Vector &x)
{
    double r2 = x(0)*x(0) + x(1)*x(1);
    double l2;
    if (r2>0)
    {
        r2 = sqrt(r2);
    }
    l2 = 0;
    //This is for tipton
    if (r2 >= 0.10 && r2 <= 0.15 )
    {
        l2 = 1;
    }
    l2 = 0.01+0.5*std::tanh((r2-0.13)/0.01)-(0.5*std::tanh((r2-0.14)/0.01))
            +0.5*std::tanh((r2-0.21)/0.01)-(0.5*std::tanh((r2-0.22)/0.01));
    l2 = 0.01+0.5*std::tanh((r2-0.12)/0.005)-(0.5*std::tanh((r2-0.13)/0.005))
    +0.5*std::tanh((r2-0.18)/0.005)-(0.5*std::tanh((r2-0.19)/0.005));
    //l2 = 10*r2;
    /*
    //This is for blade
    int l4 = 0, l3 = 0;
    double xmin, xmax, ymin, ymax,dx ,dy;
    xmin = 0.9; xmax = 1.;
    ymin = -0.2; ymax = 0.;
    dx = (xmax-xmin)/2;dy = (ymax-ymin)/2;
    
    if (abs(x(0)-xmin)<dx && abs(x(0)-xmax)<dx) {
        l4 = 1;
    }
    if (abs(x(1)-ymin)<dy && abs(x(1)-ymax)<dy) {
        l3 = 1;
    }
    l2 = l4*l3;
     */
    
    // This is for square perturbed
    /*
     if (r2 < 0.5) {
        l2 = 1;
    }
     */
    
    return l2;
}

