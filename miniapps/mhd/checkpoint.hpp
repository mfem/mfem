//save check points

#include "mfem.hpp"
using namespace std;
using namespace mfem;

void checkpoint(int myid, const double time,
                const ParMesh & pmesh, 
                const ParGridFunction & phi, 
                const ParGridFunction & psi, 
                const ParGridFunction & w)
{
    if (myid==0) cout << " Save checkpoints at t = "<<time<< endl;

    ofstream ofs_mesh(MakeParFilename("checkpt-mesh.", myid));
    ofstream ofs_phi(MakeParFilename("checkpt-phi.", myid));
    ofstream ofs_psi(MakeParFilename("checkpt-psi.", myid));
    ofstream   ofs_w(MakeParFilename("checkpt-w.", myid));

    ofs_mesh.precision(16);
    ofs_phi.precision(16);
    ofs_psi.precision(16);
      ofs_w.precision(16);

    pmesh.ParPrint(ofs_mesh);

    phi.Save(ofs_phi);
    psi.Save(ofs_psi);
      w.Save(ofs_w);
}
