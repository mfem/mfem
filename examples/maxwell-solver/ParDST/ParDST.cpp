//Diagonal Source Transfer Preconditioner

#include "ParDST.hpp"


ParDST::ParDST(ParSesquilinearForm * bf_, Array2D<double> & Pmllength_, 
         double omega_, Coefficient * ws_,  int nrlayers_ , int nx_, int ny_, int nz_)
   : Solver(2*bf_->ParFESpace()->GetTrueVSize(), 2*bf_->ParFESpace()->GetTrueVSize()), 
     bf(bf_), Pmllength(Pmllength_), omega(omega_), ws(ws_), nrlayers(nrlayers_)
{
   pfes = bf->ParFESpace();
   fec = pfes->FEColl();
 
   // Indentify problem ... Helmholtz or Maxwell
   int prob_kind = fec->GetContType();

   pmesh = pfes->GetParMesh();
   dim = pmesh->Dimension();

   ParMeshPartition part(pmesh,nx_,ny_,nz_,nrlayers);

   nx = part.nxyz[0]; ny = part.nxyz[1];  nz = part.nxyz[2];
   nrsubdomains = part.nrsubdomains;


}

void ParDST::Mult(const Vector &r, Vector &z) const
{
}


// void ParDST::Getijk(int ip, int & i, int & j, int & k) const
// {
//    k = ip/(nx*ny);
//    j = (ip-k*nx*ny)/nx;
//    i = (ip-k*nx*ny)%nx;
// }

// int ParDST::GetPatchId(const Array<int> & ijk) const
// {
//    int d=ijk.Size();
//    int z = (d==2)? 0 : ijk[2];
//    return part->subdomains(ijk[0],ijk[1],z);
// }

ParDST::~ParDST() {}
