// //                                MFEM Example 1
// //
// // Compile with: make ex1
// //

// #include "mfem.hpp"
// #include <fstream>
// #include <iostream>

// using namespace std;
// using namespace mfem;

// void SetElemAttr(Mesh * mesh);
// double SolExact(const Vector & x);
// double ChiExact(const Vector & x);
// double BumpFncn(const Vector & x);

// int main(int argc, char *argv[])
// {
//    // 1. Parse command-line options.
//    const char *mesh_file = "../data/star.mesh";
//    int order = 1;
//    bool visualization = true;

//    OptionsParser args(argc, argv);
//    args.AddOption(&mesh_file, "-m", "--mesh",
//                   "Mesh file to use.");
//    args.AddOption(&order, "-o", "--order",
//                   "Finite element order (polynomial degree) or -1 for"
//                   " isoparametric space.");
//    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
//                   "--no-visualization",
//                   "Enable or disable GLVis visualization.");
//    args.Parse();
//    if (!args.Good())
//    {
//       args.PrintUsage(cout);
//       return 1;
//    }
//    args.PrintOptions(cout);

//    Mesh *mesh = new Mesh(mesh_file, 1, 1);
//    int dim = mesh->Dimension();

//    int ref_levels = 4;
//    for (int l = 0; l < ref_levels; l++)
//    {
//       mesh->UniformRefinement();
//    }

//    // SetElemAttr(mesh);

//    // Array<int> attr;
//    // if (mesh->attributes.Size())
//    // {
//    //    attr.SetSize(mesh->attributes.Max());
//    //    attr = 0;   attr[1] = 1;
//    // }


//    FiniteElementCollection *fec = new H1_FECollection(order, dim);
//    FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
//    cout << "Number of finite element unknowns: "
//         << fespace->GetTrueVSize() << endl;

//    Array<int> ess_tdof_list;

//    // mesh->bdr_attributes.Print();
//    // if (mesh->bdr_attributes.Size())
//    // {
//    //    Array<int> ess_bdr(mesh->bdr_attributes.Max());
//    //    ess_bdr = 0;
//    //    // ess_bdr[3] = 1;
//    //    // ess_bdr[1] = 1;
//    //    fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
//    // }


//    // LinearForm *b = new LinearForm(fespace);
//    // ConstantCoefficient one(1.0);
//    // RestrictedCoefficient restr(one,attr);


//    // b->AddDomainIntegrator(new DomainLFIntegrator(restr));
//    // b->AddDomainIntegrator(new DomainLFIntegrator(one));
//    // b->Assemble();

//    // GridFunction x(fespace);
//    // FunctionCoefficient chi(ChiExact);
//    // x.ProjectCoefficient(chi);
//    // x = 0.0;

//    // BilinearForm *a = new BilinearForm(fespace);
//    // // a->AddDomainIntegrator(new DiffusionIntegrator(one));
//    // ConstantCoefficient epsilon(0.000001);
//    // a->AddDomainIntegrator(new DiffusionIntegrator(epsilon));
//    // a->AddDomainIntegrator(new MassIntegrator(one));
//    // a->Assemble();

//    // OperatorPtr A;
//    // Vector B, X;
//    // a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
//    // cout << "Size of linear system: " << A->Height() << endl;


//    // UMFPackSolver umf_solver;
//    // umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
//    // umf_solver.SetOperator(*A);
//    // umf_solver.Mult(B, X);


//    // a->RecoverFEMSolution(X, *b, x);


//    GridFunction bump(fespace);
//    FunctionCoefficient c1(BumpFncn);
//    bump.ProjectCoefficient(c1);


//    // GridFunction uex(fespace);
//    // FunctionCoefficient u_ex(SolExact);
//    // uex.ProjectCoefficient(u_ex);

//    // int order_quad = max(2, 2 * order + 1);
//    // const IntegrationRule *irs[Geometry::NumGeom];
//    // for (int i = 0; i < Geometry::NumGeom; ++i)
//    // {
//    //    irs[i] = &(IntRules.Get(i, order_quad));
//    // }
//    // double l2error = x.ComputeL2Error(u_ex, irs);
//    // cout << "l2error = "<< l2error << endl;
//    // 14. Send the solution by socket to a GLVis server.
//    if (visualization)
//    {
//       char vishost[] = "localhost";
//       int  visport   = 19916;
//       socketstream sol_sock(vishost, visport);
//       sol_sock.precision(8);
//       // sol_sock << "solution\n" << *mesh << x << flush;
//       sol_sock << "solution\n" << *mesh << bump << flush;

//       // socketstream ex_sock(vishost, visport);
//       // ex_sock.precision(8);
//       // ex_sock << "solution\n" << *mesh << uex << flush;

//       // GridFunction err(uex);
//       // err-= x;
//       // socketstream diff_sock(vishost, visport);
//       // diff_sock.precision(8);
//       // diff_sock << "solution\n" << *mesh << err << flush;

//    }

//    // 15. Free the used memory.
//    // delete a;
//    // delete b;
//    delete fespace;
//    delete fec;
//    delete mesh;

//    return 0;
// }


// void SetElemAttr(Mesh * mesh)
// {
//    int dim=mesh->Dimension();
//    double h = 1.0/sqrt(mesh->GetNE());
//    for (int iel=0; iel<mesh->GetNE(); iel++)
//    {
//       Vector center(dim);
//       int geom = mesh->GetElementBaseGeometry(iel);
//       ElementTransformation * tr = mesh->GetElementTransformation(iel);
//       tr->Transform(Geometries.GetCenter(geom), center);
//       int attr = (center[0] < 15*h) ? 1 : 2;
//       mesh->SetAttribute(iel,attr);
//    }
//    mesh->SetAttributes();
// }

// double SolExact(const Vector & x)
// {
//    double u;
//    if (x(0) < 0.5) 
//    {
//       u = x(0)/8.0;
//    }
//    else
//    {
//       u = - x(0)*x(0)/2.0 + 5.0 * x(0) / 8.0 - 1.0/8.0;
//    }

//    return u;
// }

// double ChiExact(const Vector & x)
// {
//    double u;
//    if (x(0) == 0.0) 
//    {
//       u = 0.0;
//    }
//    else
//    {
//       u = 0.0;
//    }

//    return u;
// }

// double BumpFncn(const Vector & x)
// {
//    double u;
//    if (x(0) == 0.0) 
//    {
//       u = 0.0;
//    }
//    else
//    {
//       u = exp(- 0.01/(1.0-pow(x(0)-1.0,2)));
//    }

//    return u;
// }