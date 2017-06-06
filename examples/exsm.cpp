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
    
    bool dump_iterations = false;
    
    if (argc == 1)
    {
        cout << "Usage: exsm <mesh_file>" << endl;
        return 1;
    }
    
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
    
    mesh = new Mesh(mesh_file, 1, 1);
    
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
    }
    
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
    double jitter = 0.25; // perturbation scaling factor
    cout << "Enter jitter --> " << flush;
    cin >> jitter;
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
    
    int smoother;
    cout <<
    "Select smoother:\n"
    "1) Hyperelastic model\n"
    "2) TMOP\n"
    " --> " << flush;
    cin >> smoother;
    
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
    
    L2_FECollection mfec(3, mesh->Dimension(), BasisType::GaussLobatto);
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
        " --> " << flush;
        cin >> modeltype;
        model    = new TMOPHyperelasticModel001;
        if (tjtype == 1)
        {
            model = new TMOPHyperelasticModel001;
        }
        else if (tjtype == 2)
        {
            model = new TMOPHyperelasticModel002;
        }
        else if (tjtype == 7)
        {
            model = new TMOPHyperelasticModel007;
        }
        else
        {
            cout << "You did not choose a valid option\n";
            cout << "Model type will default to 1\n";
        }
        
        
        tj->SetNodes(*x);
        tj->SetInitialNodes(x0);
        HyperelasticNLFIntegrator *he_nlf_integ;
        he_nlf_integ = new HyperelasticNLFIntegrator(model, tj);
        
        const IntegrationRule *ir =
        &IntRulesLo.Get(fespace->GetFE(0)->GetGeomType(), 8);
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
        cout << mesh->GetNBE() << "k10 number of boundary elements\n";
        
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
        
        // Check attribute numbers, arc lengths, coordinates.
        /*
         GridFunction &X = *x;
         for (int i = 0; i < mesh->GetNBE(); i++)
         {
         x->FESpace()->GetBdrElementVDofs(i, vdofs);
         int a = mesh->GetBdrElement(i)->GetAttribute();
         
         cout << "BE " << i << " with attr " << a << ": " << endl;
         int nd = vdofs.Size()/2;
         for (int j = 0; j < nd; j++)
         {
         if (a == 8)
         {
         cout << "-- Node " << j
         << ": x = " << X(vdofs[j])
         << ", y = " << X(vdofs[j+nd])
         << ", r-0.3 = " << 0.3 - sqrt(X(vdofs[j]) * X(vdofs[j]) +
         X(vdofs[j+nd]) * X(vdofs[j+nd]))
         << endl;
         }
         }
         if (a == 8)
         {
         const double dx02 = X(vdofs[0]) - X(vdofs[2]);
         const double dy02 = X(vdofs[0 + nd]) - X(vdofs[2 + nd]);
         cout << "-- Distance 0 2: "
         << sqrt( dx02 * dx02 + dy02 * dy02 ) << endl;
         
         const double dx12 = X(vdofs[1]) - X(vdofs[2]);
         const double dy12 = X(vdofs[1 + nd]) - X(vdofs[2 + nd]);
         cout << "-- Distance 1 2: "
         << sqrt( dx12 * dx12 + dy12 * dy12 ) << endl;
         
         
         const double dx01 = X(vdofs[0]) - X(vdofs[1]);
         const double dy01 = X(vdofs[0 + nd]) - X(vdofs[1 + nd]);
         cout << "-- Distance 0 1: "
         << sqrt( dx01 * dx01 + dy01 * dy01 ) << endl;
         }
         cout << endl;
         }
         */
        
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
        
        
        cout << "Choose linear smoother:\n"
        "0) l1-Jacobi\n"
        "1) CG\n"
        "2) MINRES\n" << " --> " << flush;
        cin >> ans;
        Solver *S;
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
        
        cout << "Enter number of Newton iterations --> " << flush;
        cin >> ans;
        
        cout << "Initial strain energy : " << a.GetEnergy(*x) << endl;
        
        // note: (*x) are the mesh nodes
        NewtonSolver *newt= new NewtonSolver;
        newt->SetPreconditioner(*S);
        newt->SetMaxIter(ans);
        newt->SetRelTol(rtol);
        newt->SetAbsTol(0.0);
        newt->SetPrintLevel(1);
        newt->SetOperator(a);
        Vector b;
        newt->Mult(b, *x);
        
        if (!newt->GetConverged())
            cout << "NewtonIteration : rtol = " << rtol << " not achieved."
            << endl;
        
        cout << "Final strain energy   : " << a.GetEnergy(*x) << endl;
        
        if (tj)
        {
            InterpolateHyperElasticModel(*model, *tj, *mesh, metric);
            osockstream sol_sock(visport, vishost);
            sol_sock << "solution\n";
            mesh->Print(sol_sock);
            metric.Save(sol_sock);
            sol_sock.send();
        }
        
        delete newt;
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
    // save subdivided VTK mesh?
    if (1)
    {
        cout << "Enter VTK mesh subdivision factor or 0 to skip --> " << flush;
        cin >> ans;
        if (ans > 0)
        {
            ofstream vtk_mesh("smoothed.vtk");
            vtk_mesh.precision(8);
            mesh->PrintVTK(vtk_mesh, ans);
        }
    }
    
    // 16. (Optional) Send the relaxed mesh with the vector field representing
    //     the displacements to the perturbed mesh by socket to a GLVis server.
    cout << "Visualize the smoothed mesh? [0/1] --> ";
    cin >> ans;
    if (ans)
    {
        osockstream sol_sock(visport, vishost);
        sol_sock << "solution\n";
        mesh->Print(sol_sock);
        x0.Save(sol_sock);
        sol_sock.send();
    }
    
    // 17. Free the used memory.
    delete fespace;
    delete fec;
    delete mesh;
}
