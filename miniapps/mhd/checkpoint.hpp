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

void checkpoint_rs(int myid, const double time,
                const ParMesh & pmesh, 
                const ParGridFunction & phi, 
                const ParGridFunction & psi, 
                const ParGridFunction & w,
                const int restart_count = 0)
{
    if (myid==0) cout <<" Save checkpoints at t = "<<time
                      <<" restart count = "<<restart_count<<endl;

    string mesh_name, phi_name, psi_name, w_name;
    string rs = to_string(restart_count);

    mesh_name = "restart-mesh" + rs + ".";
     phi_name = "restart-phi"  + rs + ".";
     psi_name = "restart-psi"  + rs + ".";
       w_name = "restart-w"    + rs + ".";

    ofstream ofs_mesh(MakeParFilename(mesh_name, myid));
    ofstream  ofs_phi(MakeParFilename( phi_name, myid));
    ofstream  ofs_psi(MakeParFilename( psi_name, myid));
    ofstream    ofs_w(MakeParFilename(   w_name, myid));

    ofs_mesh.precision(16);
    ofs_phi.precision(16);
    ofs_psi.precision(16);
      ofs_w.precision(16);

    pmesh.ParPrint(ofs_mesh);

    phi.Save(ofs_phi);
    psi.Save(ofs_psi);
      w.Save(ofs_w);
}

