#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "mtop_integrators.hpp"

int main(int argc, char *argv[])
{
    //const char *mesh_file = "../../data/beam-tet.mesh";
    const char *mesh_file = "../../data/star.vtk";
    int ser_ref_levels = 1;
    int order = 2;
    bool visualization = true;
    double newton_rel_tol = 1e-4;
    double newton_abs_tol = 1e-6;
    int newton_iter = 10;
    int print_level = 0;

    mfem::OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
    args.AddOption(&ser_ref_levels,
                     "-rs",
                     "--refine-serial",
                     "Number of times to refine the mesh uniformly in serial.");
    args.AddOption(&order,
                      "-o",
                      "--order",
                      "Order (degree) of the finite elements.");
    args.AddOption(&visualization,
                      "-vis",
                      "--visualization",
                      "-no-vis",
                      "--no-visualization",
                      "Enable or disable GLVis visualization.");
    args.AddOption(&newton_rel_tol,
                      "-rel",
                      "--relative-tolerance",
                      "Relative tolerance for the Newton solve.");
    args.AddOption(&newton_abs_tol,
                      "-abs",
                      "--absolute-tolerance",
                      "Absolute tolerance for the Newton solve.");
    args.AddOption(&newton_iter,
                      "-it",
                      "--newton-iterations",
                      "Maximum iterations for the Newton solve.");

    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(std::cout);
        return 1;
    }
    args.PrintOptions(std::cout);

    // 3. Read the (serial) mesh from the given mesh file on all processors.  We
    //    can handle triangular, quadrilateral, tetrahedral and hexahedral meshes
    //    with the same code.
    mfem::Mesh *mesh = new mfem::Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    // 4. Refine the mesh in serial to increase the resolution. In this example
    //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
    //    a command-line parameter.
    for (int lev = 0; lev < ser_ref_levels; lev++)
    {
        mesh->UniformRefinement();
    }

    // Define the q-function
    // Diffusion coefficient
    mfem::ConstantCoefficient* diffco=new mfem::ConstantCoefficient(1.0);
    // Heat source
    mfem::ConstantCoefficient* loadco=new mfem::ConstantCoefficient(1.0);
    mfem::QLinearDiffusion* qfun=new mfem::QLinearDiffusion(*diffco,*loadco,1.0,1e-7,4.0,0.5);

    // Define FE collection and space for the state solution
    mfem::H1_FECollection sfec(order, dim);
    mfem::FiniteElementSpace* sfes=new mfem::FiniteElementSpace(mesh,&sfec,1);
    // Define FE collection and space for the density field
    mfem::L2_FECollection pfec(order, dim);
    mfem::FiniteElementSpace* pfes=new mfem::FiniteElementSpace(mesh,&pfec,1);

    // Define the arrays fpr the nonlinear form
    mfem::Array<mfem::FiniteElementSpace*> asfes;
    mfem::Array<mfem::FiniteElementSpace*> apfes;

    asfes.Append(sfes);
    apfes.Append(pfes);
    // Define parametric block nonlinear form using single scalar H1 field
    // and L2 scalar density field
    mfem::ParametricBNLForm* nf=new mfem::ParametricBNLForm(asfes,apfes);
    // add the parametric integrator
    nf->AddDomainIntegrator(new mfem::ParametricLinearDiffusion(*qfun));

    // Define density grid function
    mfem::GridFunction* gfdens=new mfem::GridFunction(pfes);
    // Define state grid function
    mfem::GridFunction* gfstat=new mfem::GridFunction(sfes);

    // Define true block vectors for state, adjoint, resudual
    mfem::BlockVector solbv; solbv.Update(nf->GetBlockTrueOffsets());    solbv=0.0;
    mfem::BlockVector adjbv; adjbv.Update(nf->GetBlockTrueOffsets());    adjbv=0.0;
    mfem::BlockVector resbv; resbv.Update(nf->GetBlockTrueOffsets());    resbv=0.0;
    // Define true block vectors for parametric field and gradients
    mfem::BlockVector prmbv; prmbv.Update(nf->PrmGetBlockTrueOffsets()); prmbv=0.0;
    mfem::BlockVector grdbv; grdbv.Update(nf->PrmGetBlockTrueOffsets()); grdbv=0.0;

    //set the BC for the physics
    mfem::Array<mfem::Array<int> *> ess_bdr;
    mfem::Array<mfem::Vector*>      ess_rhs;
    ess_bdr.Append(new mfem::Array<int>(mesh->bdr_attributes.Max()));
    ess_rhs.Append(nullptr);
    (*ess_bdr[0]) = 1;
    nf->SetEssentialBC(ess_bdr,ess_rhs);
    delete ess_bdr[0];

    // Define the linear solvers
    mfem::GMRESSolver *gmres;
    gmres = new mfem::GMRESSolver();
    gmres->SetAbsTol(newton_abs_tol/10);
    gmres->SetRelTol(newton_rel_tol/10);
    gmres->SetMaxIter(300);
    gmres->SetPrintLevel(print_level);

    // Define the Newton solver
    mfem::NewtonSolver *ns;
    ns = new mfem::NewtonSolver();
    ns->iterative_mode = true;
    ns->SetSolver(*gmres);
    ns->SetOperator(*nf);
    ns->SetPrintLevel(print_level);
    ns->SetRelTol(newton_rel_tol);
    ns->SetAbsTol(newton_abs_tol);
    ns->SetMaxIter(newton_iter);

    // Solve the problem
    prmbv=0.5;
    nf->SetPrmFields(prmbv); //set the density
    mfem::Vector b; //RHS is zero
    solbv=0.0;
    // Newton solve
    ns->Mult(b, solbv);

    //Compute the residual
    nf->Mult(solbv,resbv);
    std::cout<<"Norm residual="<<resbv.Norml2()<<std::endl;

    // Compute the energy of the state system
    double energy = nf->GetEnergy(solbv);
    std::cout<<"energy ="<< energy<<std::endl;


    // Define the block nonlinear form utilized for
    // representing the objective and use the state array
    mfem::BlockNonlinearForm* ob=new mfem::BlockNonlinearForm(asfes);
    // Add the integrator for the objective
    ob->AddDomainIntegrator(new mfem::DiffusionObjIntegrator());

    // Compute the objective
    double obj=ob->GetEnergy(solbv);
    std::cout<<"Objective ="<<obj<<std::endl;

    //Solve the adjoint
    {
        mfem::BlockVector adjrhs; adjrhs.Update(nf->GetBlockTrueOffsets());  adjrhs=0.0;
        // Compute the RHS for the adjoint
        ob->Mult(solbv, adjrhs);
        // Get the tangent matrix from the state problem
        mfem::BlockOperator& A=nf->GetGradient(solbv);
        // we do not need to transpose the operator for diffusion
        gmres->SetOperator(A.GetBlock(0,0));
        // Compute the adjoint solution
        gmres->Mult(adjrhs.GetBlock(0), adjbv.GetBlock(0));
    }

    // Compute gradients
    nf->SetAdjointFields(adjbv);
    nf->SetStateFields(solbv);
    nf->PrmMult(prmbv, grdbv);

    //Dump out the data
    {
        mfem::ParaViewDataCollection *dacol=new mfem::ParaViewDataCollection("SeqHeat",mesh);
        mfem::GridFunction gfgrd(pfes); gfgrd.SetFromTrueDofs(grdbv.GetBlock(0));
        mfem::GridFunction gfdns(pfes); gfdns.SetFromTrueDofs(prmbv.GetBlock(0));
        // Define state grid function
        mfem::GridFunction gfsol(sfes); gfsol.SetFromTrueDofs(solbv.GetBlock(0));
        mfem::GridFunction gfadj(sfes); gfadj.SetFromTrueDofs(adjbv.GetBlock(0));

        dacol->SetLevelsOfDetail(order);
        dacol->RegisterField("sol", &gfsol);
        dacol->RegisterField("adj", &gfadj);
        dacol->RegisterField("dns", &gfdns);
        dacol->RegisterField("grd", &gfgrd);

        dacol->SetTime(1.0);
        dacol->SetCycle(1);
        dacol->Save();

        delete dacol;
    }



    // FD check
    {
        mfem::BlockVector prtbv;
        mfem::BlockVector tmpbv;
        prtbv.Update(nf->PrmGetBlockTrueOffsets());
        tmpbv.Update(nf->PrmGetBlockTrueOffsets());
        prtbv.GetBlock(0).Randomize();
        prtbv*=1.0;
        double lsc=1.0;


        double gQoI=ob->GetEnergy(solbv);
        double lQoI;

        double nd=mfem::InnerProduct(prtbv,prtbv);
        double td=mfem::InnerProduct(prtbv,grdbv);
        td=td/nd;

        for(int l=0;l<10;l++)
        {
            lsc/=10.0;
            prtbv/=10.0;
            add(prmbv,prtbv,tmpbv);
            nf->SetPrmFields(tmpbv);
            //solve the physics
            ns->Mult(b,solbv);
            //compute the objective
            lQoI=ob->GetEnergy(solbv);
            double ld=(lQoI-gQoI)/lsc;
            std::cout<<"dx="<<lsc<<" DLQoI="<<ld/nd<<" DGQoI="<<td<<" err="<<std::fabs(ld/nd-td);
            std::cout<<std::endl;
        }
    }



    delete ob;

    delete ns;
    delete gmres;

    delete gfstat;
    delete gfdens;
    delete nf;
    delete pfes;
    delete sfes;

    delete qfun;
    delete loadco;
    delete diffco;

    delete mesh;

    return 0;
}

