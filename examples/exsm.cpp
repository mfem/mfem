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
            
            
            if (finenergy>initenergy || nanchk!=0 || jachk==0)
            //if (jachk==0)
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
        
        cout << initenergy << " " << finenergy <<  " initial and final value\n";
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
            tauval -= 0.01;
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
            
            if (finenergy>initenergy || nanchk!=0 || nelinvnew > nelinvorig)
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

//M201 = (I1I2 - 2*I2) / (2I2-2*Beta)
double TMOPHyperelasticModel022::EvalW(const DenseMatrix &Jpt) const
{
    const double I1 = Dim2Invariant1(Jpt), I2 = Dim2Invariant2(Jpt);
    double beta = tauptr;
    return  (I1*I2 - 2*I2)/(2*I2-2*beta);
    
}

void TMOPHyperelasticModel022::EvalP(const DenseMatrix &Jpt,
                                     DenseMatrix &P) const
{
    const double I1 = Dim2Invariant1(Jpt), I2 = Dim2Invariant2(Jpt);
    double beta = tauptr;
    double alpha = (2*I2 - 2*beta);
    double gamma = (I1*I2 - 2*I2);
    
    Dim2Invariant1_dM(Jpt, P);
    P *= I2*alpha;
    
    
    DenseMatrix PP(P.Size());
    Dim2Invariant2_dM(Jpt, PP); //DI2/DM
    PP *= (I1*alpha - 2*alpha - 2*gamma);     //(mu/det^2)*DI2/DM
    
    P += PP;
    P *= 1/(alpha*alpha);
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
    double alpha = (2*I2 - 2*beta);
    double gamma = (I1*I2 - 2*I2);
    
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
                    const double entry_rr_cc =
                    (alpha*alpha*(2*dI2_dM(rr,cc)*(I1*dI2_dM(r,c) + dI1_dM(r,c)*I2 -2*dI2_dM(r,c))
        + alpha*(dI1_dM(rr,cc)*dI2_dM(r,c)+I1*dI2_dMdM(rr,cc)+I2*dI1_dMdM(rr,cc)+dI1_dM(r,c)*dI2_dM(rr,cc)
                   -2*dI2_dMdM(rr,cc))
        - gamma*2*dI2_dMdM(rr,cc) - 2*dI2_dM(r,c)*(dI1_dM(rr,cc)*I2 + I1*dI2_dM(rr,cc) - 2*dI2_dM(rr,cc)))
                    - (alpha*(I1*dI2_dM(r,c)+dI1_dM(r,c)*I2-2*dI2_dM(r,c)) - (gamma*2*dI2_dM(r,c)))*(2*alpha*2*dI2_dM(rr,cc)))/(alpha*alpha*alpha*alpha);
                    
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



// Metric 200
class TMOPHyperelasticModel200 : public HyperelasticModel
{
public:
    virtual double EvalW(const DenseMatrix &Jpt) const;
    
    virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;
    
    virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                           const double weight, DenseMatrix &A) const;
};

//M201 = - mu/(I2) and mu = 0 if I2 > 0 i.e. if element is valid
double TMOPHyperelasticModel200::EvalW(const DenseMatrix &Jpt) const
{
    double det = Dim2Invariant2(Jpt);
    double mu = 1e+20;
    //cout << det << " k10 det\n";
    if (det > 0.0) {mu=0.;}
    //cout << det << " " << mu << " k10det and mu\n";
    //cout << sumres << " k10 metric value\n";
    return  0.- mu/(det);
}

void TMOPHyperelasticModel200::EvalP(const DenseMatrix &Jpt,
                                     DenseMatrix &P) const
{
    double det = Dim2Invariant2(Jpt);
    double mu = 1e+20;
    if (det > 0.0) {mu=0.; }
    
    Dim2Invariant2_dM(Jpt, P); //DI2/DM
    P *= mu/(det*det);       //(mu/det^2)*DI2/DM
    
}

void TMOPHyperelasticModel200::AssembleH(const DenseMatrix &Jpt,
                                         const DenseMatrix &DS,
                                         const double weight,
                                         DenseMatrix &A) const
{
    const int dof = DS.Height(), dim = DS.Width();
    const double I1 = Dim2Invariant1(Jpt), I2 = Dim2Invariant2(Jpt);
    DenseMatrix dI1_dM(dim), dI1_dMdM(dim), dI2_dM(dim), dI2_dMdM(dim);
    Dim2Invariant1_dM(Jpt, dI1_dM);
    Dim2Invariant2_dM(Jpt, dI2_dM);
    double mu = 1e+20;
    if (I2 > 0.0) {mu=0.; }
    
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
                    const double entry_rr_cc = // This stuff from 001
                    -2*(mu/(I2*I2*I2))*dI2_dM(rr,cc)*dI2_dM(r,c) + //-2*(mu/I3^3)*(DI2/DM)*(DI2/DM)
                    (mu/(I2*I2))*dI2_dMdM(rr,cc); // + (mu/I2^2)*D2I2/DM^2
                    
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



//Metric 201
class TMOPHyperelasticModel201 : public HyperelasticModel
{
public:
    virtual double EvalW(const DenseMatrix &Jpt) const;
    
    virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;
    
    virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                           const double weight, DenseMatrix &A) const;
};

//M201 = I1*I2 - mu/(I2) and mu = 0 if I2 > 0 i.e. if element is valid
double TMOPHyperelasticModel201::EvalW(const DenseMatrix &Jpt) const
{
    double det = Dim2Invariant2(Jpt);
    double mu = 1e+20;
    //cout << det << " k10 det\n";
    if (det > 0.0) {mu=0.;}
    //cout << det << " " << mu << " k10det and mu\n";
    double sumres = Dim2Invariant1(Jpt) * Dim2Invariant2(Jpt) - mu/(det);
    //cout << sumres << " k10 metric value\n";
    return Dim2Invariant1(Jpt) * Dim2Invariant2(Jpt) - mu/(det);
}

void TMOPHyperelasticModel201::EvalP(const DenseMatrix &Jpt,
                                     DenseMatrix &P) const
{
    double det = Dim2Invariant2(Jpt);
    double mu = 1e+20;
    if (det > 0.0) {mu=0.; }
    
    Dim2Invariant1_dM(Jpt, P); //DI1/DM
    P *= Dim2Invariant2(Jpt);  // I2*DI1/DM
    
    DenseMatrix PP(P.Size());
    Dim2Invariant2_dM(Jpt, PP); //DI2/DM
    PP *= Dim2Invariant1(Jpt);  //I1*DI2/DM
    
    DenseMatrix PPP(P.Size());
    Dim2Invariant2_dM(Jpt, PPP); //DI2/DM
    PPP *= mu/(det*det);         //(mu/det^2)*DI2/DM
    
    P += PP;
    P += PPP;
}

void TMOPHyperelasticModel201::AssembleH(const DenseMatrix &Jpt,
                                         const DenseMatrix &DS,
                                         const double weight,
                                         DenseMatrix &A) const
{
    const int dof = DS.Height(), dim = DS.Width();
    const double I1 = Dim2Invariant1(Jpt), I2 = Dim2Invariant2(Jpt);
    DenseMatrix dI1_dM(dim), dI1_dMdM(dim), dI2_dM(dim), dI2_dMdM(dim);
    Dim2Invariant1_dM(Jpt, dI1_dM);
    Dim2Invariant2_dM(Jpt, dI2_dM);
    double mu = 1e+20;
    if (I2 > 0.0) {mu=0.; }
    
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
                    const double entry_rr_cc =
                    dI1_dMdM(rr,cc) * I2 +
                    dI1_dM(r, c)    * dI2_dM(rr,cc) +
                    dI2_dMdM(rr,cc) * I1 +
                    dI2_dM(r, c)    * dI1_dM(rr,cc) - // This stuff from 001
                    2*(mu/(I2*I2*I2))*dI2_dM(rr,cc)*dI2_dM(r,c) + //-2*(mu/I3^3)*(DI2/DM)*(DI2/DM)
                    (mu/(I2*I2))*dI2_dMdM(rr,cc); // + (mu/I2^2)*D2I2/DM^2
                    
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

// Metric 204
class TMOPHyperelasticModel204 : public HyperelasticModel
{
public:
    virtual double EvalW(const DenseMatrix &Jpt) const;
    
    virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;
    
    virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                           const double weight, DenseMatrix &A) const;
};

//M204 = -I2 + sqrt(I2^2+beta) and beta = 0.0001
double TMOPHyperelasticModel204::EvalW(const DenseMatrix &Jpt) const
{
    double det = Dim2Invariant2(Jpt);
    double beta = 1e-4;
    //cout << det << " " << -det + sqrt(det*det + beta) << " k10 det\n";
    //cout << -det  << " k10 det\n";
    return  -det + sqrt(det*det + beta);
}

void TMOPHyperelasticModel204::EvalP(const DenseMatrix &Jpt,
                                     DenseMatrix &P) const
{
    double det = Dim2Invariant2(Jpt);
    double beta = 1e-4;
    double alpha = det/sqrt(det*det+beta);

    Dim2Invariant2_dM(Jpt, P);     //DI2/DM
    P *= (alpha-1.0);                //(alpha)*DI2/DM
    //cout << det << " "<< alpha << " " << det*det+beta <<  " k10alpha\n";
}

void TMOPHyperelasticModel204::AssembleH(const DenseMatrix &Jpt,
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
                    -dI2_dMdM(rr,cc) + alpha*dI2_dMdM(rr,cc) +
                    dI2_dM(rr,cc)*dI2_dM(r,c)*1/(sqrt(det*det+beta))+
                    (I2*dI2_dM(r,c))*(-I2/alpha2)*dI2_dM(rr,cc);
                    
                    
                    
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

// Metric 211 (I2-1)^2 - (I2 + sqrt(I2^2 + beta))
class TMOPHyperelasticModel211 : public HyperelasticModel
{
public:
    virtual double EvalW(const DenseMatrix &Jpt) const;
    
    virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;
    
    virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                           const double weight, DenseMatrix &A) const;
};

//Metric 211 (I2-1)^2 - (I2 + sqrt(I2^2 + beta))
double TMOPHyperelasticModel211::EvalW(const DenseMatrix &Jpt) const
{
    double det = Dim2Invariant2(Jpt);
    double beta = 1e-4;
    return  (det*det) -3*det + sqrt(det*det + beta) + 1;
}

void TMOPHyperelasticModel211::EvalP(const DenseMatrix &Jpt,
                                     DenseMatrix &P) const
{
    double det = Dim2Invariant2(Jpt);
    double beta = 1e-4;
    double alpha = det/sqrt(det*det+beta);
    
    Dim2Invariant2_dM(Jpt, P);     //DI2/DM
    P *= (2*det - 3 + alpha);
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
                    2*I2*dI2_dMdM(rr,cc) + 2*dI2_dM(rr,cc)*dI2_dM(r,c)
                    - 3*dI2_dMdM(rr,cc) + alpha*dI2_dMdM(rr,cc) +
                    dI2_dM(rr,cc)*dI2_dM(r,c)*1/(sqrt(det*det+beta))+
                    (I2*dI2_dM(r,c))*(-I2/alpha2)*dI2_dM(rr,cc);
                    
                    
                    
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
    rdm -= 0.5; // shift to random values in [-0.5,0.5]
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
    
    L2_FECollection mfec(3, mesh->Dimension(), BasisType::GaussLobatto); //this for vis
    FiniteElementSpace mfes(mesh, &mfec, 1);
    GridFunction metric(&mfes);
    
    if (smoother == 1)
    {
        HyperelasticModel *model;
        NonlinearFormIntegrator *nf_integ;
        
        Coefficient *c = NULL;
        TargetJacobian *tj = NULL;
        
        //{IDEAL, IDEAL_EQ_SIZE, IDEAL_INIT_SIZE,
        // IDEAL_CUSTOM_SIZE, TARGET_MESH, ALIGNED}
        int tjtype;
        int modeltype;
        cout <<
        "Specify Target Jacobian type:\n"
        "1) IDEAL\n"
        "2) IDEAL_EQ_SIZE\n"
        "3) IDEAL_INIT_SIZE\n"
        " --> " << flush;
        cin >> tjtype;
        
        tj    = new TargetJacobian(TargetJacobian::IDEAL);
        if (tjtype == 1)
        {
            tj    = new TargetJacobian(TargetJacobian::IDEAL);
        }
        else if (tjtype == 2)
        {
            tj    = new TargetJacobian(TargetJacobian::IDEAL_EQ_SIZE);
        }
        else if (tjtype == 3)
        {
            tj    = new TargetJacobian(TargetJacobian::IDEAL_INIT_SIZE);
        }
        else
        {
            cout << tjtype;
            cout << "You did not choose a valid option\n";
            cout << "Target Jacobian will default to IDEAL\n";
        }
        
        cout << "Choose optimization metric:\n"
        << "1  : |T|^2 \n"
        << "     shape.\n"
        << "2  : 0.5 |T|^2 / tau  - 1 \n"
        << "     shape, condition number metric.\n"
        << "7  : |T - T^-t|^2 \n"
        << "     shape+size.\n"
        << "22  : |T|^2 - 2*tau / (2*tau - 2*tau_0)\n"
        << "     untangling.\n"
        << "200  : - mu/tau\n"
        << "     untangling.\n"
        << "201  : |T|^2 - mu/tau\n"
        << "     shape + untangling.\n"
        << "204  : -tau + sqrt(tau^2+beta)\n"
        << "      untangling.\n"
        << "211  : (tau-1)^2 - tau + sqrt(tau^2+beta)\n"
        << "      untangling.\n"
        " --> " << flush;
        double tauval = -0.1;
        cin >> modeltype;
        model    = new TMOPHyperelasticModel001;
        if (modeltype == 1)
        {
            model = new TMOPHyperelasticModel001;
        }
        else if (modeltype == 2)
        {
            model = new TMOPHyperelasticModel002;
        }
        else if (modeltype == 7)
        {
            model = new TMOPHyperelasticModel007;
        }
        else if (modeltype == 22)
        {
            model = new TMOPHyperelasticModel022(tauval);
            cout << " you chose 22 metric \n";
        }
        else if (modeltype == 200)
        {
            model = new TMOPHyperelasticModel201;
        }
        else if (modeltype == 201)
        {
            model = new TMOPHyperelasticModel201;
        }
        else if (modeltype == 204)
        {
            model = new TMOPHyperelasticModel204;
        }
        else if (modeltype == 211)
        {
            model = new TMOPHyperelasticModel211;
        }
        else
        {
            cout << "You did not choose a valid option\n";
            cout << "Model type will default to 1\n";
            cout << modeltype;
        }
        
        logvec[2]=tjtype;
        logvec[3]=modeltype;
        
        
        tj->SetNodes(*x);
        tj->SetInitialNodes(x0);
        HyperelasticNLFIntegrator *he_nlf_integ;
        he_nlf_integ = new HyperelasticNLFIntegrator(model, tj);
        
        const IntegrationRule *ir =
        &IntRulesLo.Get(fespace->GetFE(0)->GetGeomType(), 8); //this for metric
        he_nlf_integ->SetIntegrationRule(*ir);
        
        //c = new ConstantCoefficient(0.5);
        //he_nlf_integ->SetCoefficient(*c);
        //he_nlf_integ->SetLimited(1e-4, x0);
        
        nf_integ = he_nlf_integ;
        
        InterpolateHyperElasticModel(*model, *tj, *mesh, metric);
        osockstream sol_sock2(visport, vishost);
        sol_sock2 << "solution\n";
        mesh->Print(sol_sock2);
        metric.Save(sol_sock2);
        sol_sock2.send();
        sol_sock2 << "keys " << "JREM" << endl;
        
        
        NonlinearForm a(fespace);
        a.AddDomainIntegrator(nf_integ);
        
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
        //k10partbegin
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
        "2) MINRES\n" << " --> " << flush;
        cin >> ans;
        Solver *S;
        logvec[5]=ans;
        const double rtol = 1e-12;
        if (ans == 0)
        {
            cout << "Enter number of linear smoothing iterations --> " << flush;
            cin >> ans;
            S = new DSmoother(1, 1., ans);
        }
        else if (ans == 1)
        {
            cout << "Enter number of CG smoothing iterations --> " << flush;
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
            cout << "Enter number of MINRES smoothing iterations --> " << flush;
            cin >> ans;
            MINRESSolver *minres = new MINRESSolver;
            minres->SetMaxIter(ans);
            minres->SetRelTol(rtol);
            minres->SetAbsTol(0.0);
            minres->SetPrintLevel(3);
            S = minres;
            
        }
        logvec[6]=ans;
        cout << "Enter number of Newton iterations --> " << flush;
        cin >> ans;
        logvec[7]=ans;
        cout << "Initial strain energy : " << a.GetEnergy(*x) << endl;
        logvec[8]=a.GetEnergy(*x);
        
        // save original
        Vector xsav = *x;
        //set value of tau_0 for metric 22
        //
        Array<int> dofs;
        tauval = 1e+6;
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
            cout << " Relaxed newton solver will be user \n";
            newt->Mult2(b, *x, *mesh, *ir, &newtonits, a );
            if (!newt->GetConverged())
                cout << "NewtonIteration : rtol = " << rtol << " not achieved."
                << endl;
        }
        else
        {
            tauval -= 0.015;
            DescentNewtonSolver *newt= new DescentNewtonSolver;
            newt->SetPreconditioner(*S);
            newt->SetMaxIter(ans);
            newt->SetRelTol(rtol);
            newt->SetAbsTol(0.0);
            newt->SetPrintLevel(1);
            newt->SetOperator(a);
            Vector b;
            cout << " Descent newton solver will be user \n";
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
            sol_sock << "keys " << "JREM" << endl;
        }
        
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
    
    // puase 1 second.. this is because X11 can restart if you push stuff too soon
    usleep(1000000);
    //k10 end
    
}
