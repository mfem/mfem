#include "snavier_cg.hpp"

namespace mfem{

SNavierPicardCGSolver::SNavierPicardCGSolver(ParMesh* mesh_,
                                             int vorder,
                                             int porder)
{
    pmesh=mesh_;
    int dim=pmesh->Dimension();

    vfec=new H1_FECollection(vorder,dim);
    vfes=new mfem::ParFiniteElementSpace(pmesh,vfec,dim, Ordering::byVDIM);

    pfec=new H1_FECollection(porder,dim);
    pfes=new mfem::ParFiniteElementSpace(pmesh,pfec,1);

    vgf.SetSpace(vfes); vgf=0.0;
    pgf.SetSpace(pfes); pgf=0.0;

    vsol.SetSize(vfes->GetTrueVSize()); vsol=0.0;
    psol.SetSize(pfes->GetTrueVSize()); psol=0.0;

    ptmp.SetSize(pfes->GetTrueVSize()); ptmp=0.0;
    vtmp.SetSize(vfes->GetTrueVSize()); vtmp=0.0;

    prec=nullptr;
    ls=nullptr;
}

SNavierPicardCGSolver::~SNavierPicardCGSolver()
{
    delete vfes;
    delete vfec;
    delete pfes;
    delete pfec;
}


void SNavierPicardCGSolver::SetFixedPointSolver(double rtol,
                                           double atol,
                                           int miter,
                                           int prt_level)
{
    rel_tol=rtol;
    abs_tol=atol;
    max_iter=miter;
    print_level=prt_level;
}

void SNavierPicardCGSolver::SetLinearSolver(double rtol,
                                            double atol,
                                            int miter)
{
    linear_rtol=rtol;
    linear_atol=atol;
    linear_iter=miter;
}

void SNavierPicardCGSolver::FSolve()
{

}




}
