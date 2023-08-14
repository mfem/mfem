//                                MFEM Example jp
//
// Compile with: make exjp
//
// Sample runs:  exjp
//               exjp -i surface
//               exjp -i surface -o 0
//               exjp -i surface -r 1
//               exjp -i surface -o 4
//               exjp -i surface -o 4 -r 5
//               exjp -i volumetric
//               exjp -i volumetric -o 0
//               exjp -i volumetric -r 1
//               exjp -i volumetric -o 4
//               exjp -i volumetric -o 4 -r 5
//
// Description: TODO

#include "mfem.hpp"
#include <iostream>

using namespace std;
using namespace mfem;

enum class IntegrationType { Surface, Volumetric };
IntegrationType itype;

double lvlset(const Vector& X)
{
    switch(itype)
    {
        case IntegrationType::Surface:
            return 1. - (pow(X(0), 2.) + pow(X(1), 2.));
        case IntegrationType::Volumetric:
            return 1. - (pow(X(0) / 1.5, 2.) + pow(X(1) / .75, 2.));
        default:
            return 1.;
    }
}

double integrand(const Vector& X)
{
     switch(itype)
    {
        case IntegrationType::Surface:
            return 3. * pow(X(0), 2.) - pow(X(1), 2.);
        case IntegrationType::Volumetric:
            return 1.;
        default:
            return 0.;
    }
}

double Surface()
{
    switch(itype)
    {
        case IntegrationType::Surface:
            return 2. * M_PI;
        case IntegrationType::Volumetric:
            return 7.26633616541076;
        default:
            return 0.;
    }
}

double Volume()
{
    switch(itype)
    {
        case IntegrationType::Surface:
            return NAN;
        case IntegrationType::Volumetric:
            return 9. / 8. * M_PI;
        default:
            return 0.;
    }
}

class SurfaceLFIntegrator : public LinearFormIntegrator
{
protected:
    Vector shape;
    SIntegrationRule* SIntRule;
    Coefficient &LevelSet;
    Coefficient &Q;

public:
    SurfaceLFIntegrator(Coefficient &q, Coefficient &levelset,
                        SIntegrationRule* ir)
        : LinearFormIntegrator(), Q(q), LevelSet(levelset), SIntRule(ir) {}

    SurfaceLFIntegrator(Coefficient &q, Coefficient &levelset)
        : LinearFormIntegrator(), Q(q), LevelSet(levelset), SIntRule(NULL) {}

    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect) override
    {
        int dof = el.GetDof();
        shape.SetSize(dof);
        elvect.SetSize(dof);
        elvect = 0.;
        IsoparametricTransformation Trafo;
        Tr.mesh->GetElementTransformation(Tr.ElementNo, &Trafo);

        SIntRule->Update(Trafo);

        for(int ip = 0; ip < SIntRule->GetNPoints(); ip++)
        {
            Trafo.SetIntPoint((&(SIntRule->IntPoint(ip))));
            double val = Trafo.Weight() * Q.Eval(Trafo, SIntRule->IntPoint(ip));
            el.CalcShape(SIntRule->IntPoint(ip), shape);
            add(elvect, SIntRule->IntPoint(ip).weight * val, shape, elvect);
        }
    }

    void SetSurface(Coefficient &levelset) { LevelSet = levelset; }

    void SetSIntRule(SIntegrationRule *ir) { SIntRule = ir; }
    const SIntegrationRule* GetSIntRule() { return SIntRule; }
};

class SubdomainLFIntegrator : public LinearFormIntegrator
{
protected:
    Vector shape;
    CutIntegrationRule* CutIntRule;
    Coefficient &LevelSet;
    Coefficient &Q;

public:
    SubdomainLFIntegrator(Coefficient &q, Coefficient &levelset,
                          CutIntegrationRule* ir)
        : LinearFormIntegrator(), Q(q), LevelSet(levelset), CutIntRule(ir) {}

    SubdomainLFIntegrator(Coefficient &q, Coefficient &levelset)
        : LinearFormIntegrator(), Q(q), LevelSet(levelset), CutIntRule(NULL) {}

    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect) override
    {
        int dof = el.GetDof();
        shape.SetSize(dof);
        elvect.SetSize(dof);
        elvect = 0.;
        IsoparametricTransformation Trafo;
        Tr.mesh->GetElementTransformation(Tr.ElementNo, &Trafo);

        CutIntRule->Update(Trafo);

        for(int ip = 0; ip < CutIntRule->GetNPoints(); ip++)
        {
            Trafo.SetIntPoint((&(CutIntRule->IntPoint(ip))));
            double val = Trafo.Weight()
                        * Q.Eval(Trafo, CutIntRule->IntPoint(ip));
            el.CalcPhysShape(Trafo, shape);
            add(elvect, CutIntRule->IntPoint(ip).weight * val, shape, elvect);
        }
    }

    void SetSurface(Coefficient &levelset) { LevelSet = levelset; }

    void SetCutIntRule(CutIntegrationRule *ir) { CutIntRule = ir; }
    const CutIntegrationRule* GetCutIntRule() { return CutIntRule; }
};

int main(int argc, char *argv[])
{
    int ref_levels = 3;
    int order = 2;
    const char *inttype = "surface";

#ifndef MFEM_USE_LAPACK
    cout << "MFEM must be build with LAPACK for this example." << endl;
    return EXIT_FAILURE;
#else
    OptionsParser args(argc, argv);
    args.AddOption(&order, "-o", "--order", "Order of quadrature rule");
    args.AddOption(&ref_levels, "-r", "--refine", "Number of meh refinements");
    args.AddOption(&inttype, "-i", "--integrationtype",
                           "IntegrationType to demonstrate");
    args.ParseCheck();

    if(strcmp(inttype, "surface") == 0 || strcmp(inttype, "Surface") == 0)
        itype = IntegrationType::Surface;
    else if(strcmp(inttype, "volume") == 0 || strcmp(inttype, "Volume") == 0)
    itype = IntegrationType::Volumetric;

    Mesh *mesh = new Mesh(2, 4, 1, 0, 2);
    mesh->AddVertex(-1.6,-1.6);
    mesh->AddVertex(1.6,-1.6);
    mesh->AddVertex(1.6,1.6);
    mesh->AddVertex(-1.6,1.6);
    mesh->AddQuad(0,1,2,3);
    mesh->FinalizeQuadMesh(1, 0, 1);

    for (int lev = 0; lev < ref_levels; lev++)
    {
        mesh->UniformRefinement();
    }

    H1_FECollection fe_coll(1, mesh->Dimension());
    FiniteElementSpace *fespace = new FiniteElementSpace(mesh, &fe_coll);

    FunctionCoefficient levelset(lvlset);
    FunctionCoefficient u(integrand);

    IsoparametricTransformation Tr;
    mesh->GetElementTransformation(0, &Tr);
    SIntegrationRule* sir = new SIntegrationRule(order,
                                Tr, levelset);
    CutIntegrationRule* cir = new CutIntegrationRule(order,
                                Tr, levelset);
    
    LinearForm surface(fespace);
    LinearForm volume(fespace);

    surface.AddDomainIntegrator(new SurfaceLFIntegrator(u, levelset, sir));
    surface.Assemble();

    if(itype == IntegrationType::Volumetric)
    {
        volume.AddDomainIntegrator(new SubdomainLFIntegrator(u, levelset, cir));
        volume.Assemble();
    }

    int qorder = 0;
    int nbasis = 2 * (order + 1) + (int)(order * (order + 1) / 2);
    IntegrationRules irs(0, Quadrature1D::GaussLegendre);
	IntegrationRule ir = irs.Get(Geometry::SQUARE, qorder);
    for (; ir.GetNPoints() <= nbasis; qorder++)
        ir = irs.Get(Geometry::SQUARE, qorder);
    cout << "============================================" << endl;
    cout << "Mesh size dx:                       ";
    cout << 3.2 / pow(2., (double)ref_levels) << endl;
    cout << "Number of div free basis functions: " << nbasis << endl;
    cout << "Number of quadrature points:        " << ir.GetNPoints() << endl;
    cout << scientific << setprecision(2);
    cout << "============================================" << endl;
    cout << "Computed value of surface integral: " << surface.Sum() << endl;
    cout << "True value of surface integral:     " << Surface() << endl;
    cout << "Absolut Error (Surface):            ";
    cout << abs(surface.Sum() - Surface()) << endl;
    cout << "Relative Error (Surface):           ";
    cout << abs(surface.Sum() - Surface()) / Surface() << endl;
    if(itype == IntegrationType::Volumetric)
    {
        cout << "--------------------------------------------" << endl;
        cout << "Computed value of volume integral:  " << volume.Sum() << endl;
        cout << "True value of volume integral:      " << Volume() << endl;
        cout << "Absolut Error (Volume):             ";
        cout << abs(volume.Sum() - Volume()) << endl;
        cout << "Relative Error (Volume):            ";
        cout << abs(volume.Sum() - Volume()) / Volume() << endl;
    }
    cout << "============================================" << endl;

    H1_FECollection fe_coll2(5, 2);
    FiniteElementSpace fespace2(mesh, &fe_coll2);
    FunctionCoefficient levelset_coeff(levelset);
    GridFunction lgf(&fespace2);
    lgf.ProjectCoefficient(levelset_coeff);
    char vishost[] = "localhost";
    int  visport   = 19916;
    socketstream sol_sock(vishost, visport);
    sol_sock.precision(8);
    sol_sock << "solution\n" << *mesh << lgf << "pause\n" << flush;

    delete sir;
    delete fespace;
    delete mesh;
    return EXIT_SUCCESS;
#endif //MFEM_USE_LAPACK
}