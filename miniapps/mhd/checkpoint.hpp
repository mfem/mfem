//save check points

#include "mfem.hpp"

#include <cerrno>      // errno
#include <sstream>
#include <regex>
#ifndef _WIN32
#include <sys/stat.h>  // mkdir
#else
#include <direct.h>    // _mkdir
#define mkdir(dir, mode) _mkdir(dir)
#endif

using namespace std;
using namespace mfem;

int create_directory(const string &dir_name,
                     const Mesh *mesh, int myid)
{
   // create directories recursively
   const char path_delim = '/';
   string::size_type pos = 0;
   int err_flag;
#ifdef MFEM_USE_MPI
   const ParMesh *pmesh = dynamic_cast<const ParMesh*>(mesh);
#endif

   do
   {
      pos = dir_name.find(path_delim, pos+1);
      string subdir = dir_name.substr(0, pos);

#ifndef MFEM_USE_MPI
      err_flag = mkdir(subdir.c_str(), 0777);
      err_flag = (err_flag && (errno != EEXIST)) ? 1 : 0;
#else
      if (myid == 0 || pmesh == NULL)
      {
         err_flag = mkdir(subdir.c_str(), 0777);
         err_flag = (err_flag && (errno != EEXIST)) ? 1 : 0;
      }
#endif
   }
   while ( pos != string::npos );

#ifdef MFEM_USE_MPI
   if (pmesh)
   {
      MPI_Bcast(&err_flag, 1, MPI_INT, 0, pmesh->GetComm());
   }
#endif

   return err_flag;
}

void checkpoint(int myid, const double time,
                const ParMesh & pmesh, 
                const ParGridFunction & phi, 
                const ParGridFunction & psi, 
                const ParGridFunction & w)
{
    if (myid==0) cout << " Save checkpoints at t = "<<time<< endl;

    //create a directory
    string dir_name="imMHDp-checkpoint";
    int error_code = create_directory(dir_name, &pmesh, myid);
    if (error_code){
       MFEM_WARNING("Error creating directory: " << dir_name);
       return; // do not even try to write the checkpoint
    }
    dir_name+="/";

    ofstream ofs_mesh(MakeParFilename(dir_name+"checkpt-mesh.", myid));
    ofstream  ofs_phi(MakeParFilename(dir_name+"checkpt-phi.", myid));
    ofstream  ofs_psi(MakeParFilename(dir_name+"checkpt-psi.", myid));
    ofstream    ofs_w(MakeParFilename(dir_name+"checkpt-w.", myid));

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

    //create a directory
    string dir_name="imMHDp-checkpoint";
    int error_code = create_directory(dir_name, &pmesh, myid);
    if (error_code){
       MFEM_WARNING("Error creating directory: " << dir_name);
       return; // do not even try to write the checkpoint
    }
    dir_name+="/";

    string mesh_name, phi_name, psi_name, w_name;
    string rs = to_string(restart_count);

    mesh_name = dir_name + "restart-mesh" + rs + ".";
     phi_name = dir_name + "restart-phi"  + rs + ".";
     psi_name = dir_name + "restart-psi"  + rs + ".";
       w_name = dir_name + "restart-w"    + rs + ".";

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

