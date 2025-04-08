#include "navier_solver_gcn.hpp"
#include "../../general/forall.hpp"
#include <fstream>
#include <iomanip>

namespace mfem {


NavierSolverGCN::NavierSolverGCN(ParMesh* mesh, int order, std::shared_ptr<Coefficient> visc_):
    pmesh(mesh), order(order), visc(visc),
    thet1(real_t(0.5)), thet2(real_t(0.5)),thet3(real_t(0.5)),thet4(real_t(0.5))
{

    vfec.reset(new H1_FECollection(order, pmesh->Dimension()));
    pfec.reset(new H1_FECollection(order));
    vfes.reset(new ParFiniteElementSpace(pmesh, vfec, pmesh->Dimension()));
    pfes.reset(new ParFiniteElementSpace(pmesh, pfec));

    int vfes_truevsize = vfes->GetTrueVSize();
    int pfes_truevsize = pfes->GetTrueVSize();

    cvel.SetSpace(vfes);
    pvel.SetSpace(vfes);
    pres.SetSpace(pfes);


}

NavierSolverGCN::~NavierSolverGCN()
{

}

void NavierSolverGCN::Setup(real_t dt)
{

}

void NavierSolverGCN::Step(real_t &time, real_t dt, int cur_step, bool provisional = false)
{

}


}//end namespace mfem
