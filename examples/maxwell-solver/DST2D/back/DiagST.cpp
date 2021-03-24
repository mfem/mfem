
// //Diagonal Source Transfer Preconditioner

// #include "DiagST.hpp"

// DiagST::DiagST(SesquilinearForm * bf_, Array2D<double> & Pmllength_, 
//          double omega_, Coefficient * ws_,  int nrlayers_)
//    : Solver(2*bf_->FESpace()->GetTrueVSize(), 2*bf_->FESpace()->GetTrueVSize()), 
//      bf(bf_), Pmllength(Pmllength_), omega(omega_), ws(ws_), nrlayers(nrlayers_)
// {
//    Mesh * mesh = bf->FESpace()->GetMesh();
//    dim = mesh->Dimension();

//    // ----------------- Step 1 --------------------
//    // Introduce 2 layered partitions of the domain 
//    // 
//    int partition_kind;

//    // 1. Ovelapping partition with overlap = 2h 
//    partition_kind = 2; // Overlapping partition 
//    int nx=2;
//    int ny=1; 
//    int nz=1;
//    ovlpnrlayers = nrlayers;
//    povlp = new MeshPartition(mesh, partition_kind,nx,ny,nz,ovlpnrlayers);

//    partition_kind = 1; // Non Overlapping partition 
//    novlp = new MeshPartition(mesh, partition_kind,nx,ny,nz);
//    nxyz[0] = povlp->nxyz[0];
//    nxyz[1] = povlp->nxyz[1];
//    nxyz[2] = povlp->nxyz[2];
//    nrpatch = povlp->nrpatch;
//    subdomains = povlp->subdomains;

//    //
//    // ----------------- Step 1a -------------------
//    // Save the partition for visualization
//    SaveMeshPartition(povlp->patch_mesh, "output/mesh_ovlp.", "output/sol_ovlp.");
//    SaveMeshPartition(novlp->patch_mesh, "output/mesh_nvlp.", "output/sol_nvlp.");

//    // // // ------------------Step 2 --------------------
//    // // // Construct the dof maps from subdomains to global (for the extended and not)
   
//    ovlp_prob  = new DofMap(bf,povlp,nrlayers); 
//    nvlp_prob  = new DofMap(bf,novlp); 

//    // ------------------Step 3 --------------------
//    // Assemble the PML Problem matrices and factor them
//    PmlMat.SetSize(nrpatch);
//    PmlMatInv.SetSize(nrpatch);
//    for (int ip=0; ip<nrpatch; ip++)
//    {
//       PmlMat[ip] = GetPmlSystemMatrix(ip);
//       PmlMatInv[ip] = new KLUSolver;
//       PmlMatInv[ip]->SetOperator(*PmlMat[ip]);
//    }


//    nsweeps = pow(2,dim);
//    sweeps.SetSize(nsweeps,dim);
//    // 2D
//    sweeps(0,0) =  1; sweeps(0,1) = 1;
//    sweeps(1,0) = -1; sweeps(1,1) = 1;
//    sweeps(2,0) =  1; sweeps(2,1) =-1;
//    sweeps(3,0) = -1; sweeps(3,1) =-1;

//    // Set up src arrays size
//    f_orig.SetSize(nrpatch);
//    f_transf.SetSize(nrpatch);

//    usol.SetSize(nrpatch);

//    // Construct a simple map used for directions of transfer
//    ConstructDirectionsMap();
//    for (int ip=0; ip<nrpatch; ip++)
//    {
//       int m = 2*nvlp_prob->fespaces[ip]->GetTrueVSize();
//       usol[ip] = new Vector(m); *usol[ip] = 0.0;
//       int n = 2*ovlp_prob->fespaces[ip]->GetTrueVSize(); // (x 2 for complex ) 
//       int npml = 2*ovlp_prob->PmlFespaces[ip]->GetTrueVSize(); // (x 2 for complex ) 
//       f_orig[ip] = new Vector(n); *f_orig[ip] = 0.0;
//       f_transf[ip].SetSize(nsweeps);
//       for (int i=0;i<nsweeps; i++)
//       {
//          f_transf[ip][i] = new Vector(n);
//       }
//    }
// }

// SparseMatrix * DiagST::GetPmlSystemMatrix(int ip)
// {
//    double h = GetUniformMeshElementSize(ovlp_prob->PmlMeshes[ip]);
//    Array2D<double> length(dim,2);
//    length = h*(nrlayers);

//    CartesianPML pml(ovlp_prob->PmlMeshes[ip], length);
//    pml.SetOmega(omega);

//    Array <int> ess_tdof_list;
//    if (ovlp_prob->PmlMeshes[ip]->bdr_attributes.Size())
//    {
//       Array<int> ess_bdr(ovlp_prob->PmlMeshes[ip]->bdr_attributes.Max());
//       ess_bdr = 1;
//       ovlp_prob->PmlFespaces[ip]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
//    }

//    ConstantCoefficient one(1.0);
//    ConstantCoefficient sigma(-pow(omega, 2));

//    PmlMatrixCoefficient c1_re(dim,pml_detJ_JT_J_inv_Re,&pml);
//    PmlMatrixCoefficient c1_im(dim,pml_detJ_JT_J_inv_Im,&pml);

//    PmlCoefficient detJ_re(pml_detJ_Re,&pml);
//    PmlCoefficient detJ_im(pml_detJ_Im,&pml);

//    ProductCoefficient c2_re0(sigma, detJ_re);
//    ProductCoefficient c2_im0(sigma, detJ_im);

//    ProductCoefficient c2_re(c2_re0, *ws);
//    ProductCoefficient c2_im(c2_im0, *ws);

//    SesquilinearForm a(ovlp_prob->PmlFespaces[ip],ComplexOperator::HERMITIAN);

//    a.AddDomainIntegrator(new DiffusionIntegrator(c1_re),
//                          new DiffusionIntegrator(c1_im));
//    a.AddDomainIntegrator(new MassIntegrator(c2_re),
//                          new MassIntegrator(c2_im));
//    a.Assemble();

//    OperatorPtr Alocal;
//    a.FormSystemMatrix(ess_tdof_list,Alocal);
//    ComplexSparseMatrix * AZ_ext = Alocal.As<ComplexSparseMatrix>();
//    SparseMatrix * Mat = AZ_ext->GetSystemMatrix();
//    Mat->Threshold(0.0);
//    return Mat;
// }


// void DiagST::Mult(const Vector &r, Vector &z) const
// {
//    // Step 0
//    // Restrict original sources to the patches
//    for (int ip=0; ip<nrpatch; ip++)
//    {
//       *f_orig[ip] = 0.0;
//       for (int i=0;i<nsweeps; i++)
//       {
//          *f_transf[ip][i] = 0.0;
//       }
//    }
//    for (int ip=0; ip<nrpatch; ip++)
//    {
//       Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
//       r.GetSubVector(*Dof2GlobalDof,*f_orig[ip]);
//    }

//    char vishost[] = "localhost";
//    int  visport   = 19916;
//    z = 0.0; 
//    Vector znew(z);
//    Vector rnew(r);
//    Vector raux(r); raux = 0.0;
//    Vector z1(z);
//    Vector z2(z);
//    Vector z3(z);
//    Vector z4(z);
//    znew = 0.0;
   
//    // in 2D there are a total of 4 sweeps
//    // with nx + ny - 1 serial steps each
//    // --------------------------------------------
//    //       Sweep in the direction (1,1)
//    // --------------------------------------------
//    int nx = nxyz[0];
//    int ny = nxyz[1];

//    int nsteps = nx + ny - 1;
//    // Sweep number
//    for (int l=0; l<1; l++)
//    {
//       for (int s = 0; s<nsteps; s++)
//       {
//          // the patches involved are the ones such that
//          // i+j = s
//          for (int i=0;i<nx; i++)
//          {
//             int j;
//             switch (l)
//             {
//                case 0:  j = s-i;         break;
//                case 1:  j = s-nx+i+1;    break;
//                case 2:  j = nx+i-s-1;    break;
//                default: j = nx+ny-i-s-2; break;
//             }
//             if (j<0 || j>=ny) continue;

//             // find patch id
//             Array<int> ij(2); ij[0] = i; ij[1]=j;
//             int ip = GetPatchId(ij);
            
//             // cout << "Mult:ip = " << ip << endl;

//             // Solve the PML problem in patch ip with all sources
//             // Original and all transfered (maybe some of them)
//             Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
//             Array<int> * Dof2PmlDof = &ovlp_prob->Dof2PmlDof[ip];
//             int ndofs = Dof2GlobalDof->Size();
//             Vector sol_local(ndofs); sol_local = 0.0;
//             Vector res_local(ndofs); res_local = 0.0;
//             if (l==0) res_local += *f_orig[ip];
//             // res_local += *f_orig[ip];

//             res_local += *f_transf[ip][l];
//             // if (res_local.Norml2() < 1e-12) continue;
//             // Extend by zero to the PML mesh
//             int nrdof_ext = PmlMat[ip]->Height();
//             Vector res_ext(nrdof_ext); res_ext = 0.0;
//             Vector sol_ext(nrdof_ext); sol_ext = 0.0;

//             res_ext.SetSubVector(*Dof2PmlDof,res_local);
//             PmlMatInv[ip]->Mult(res_ext, sol_ext);

//             // socketstream resipsock(vishost, visport);
//             // PlotSolution(res_ext,resipsock,ip,true, true); cin.get();

//             // socketstream solipsock(vishost, visport);
//             // PlotSolution(sol_ext,solipsock,ip,true, true); cin.get();
            
//             // Multiply with the cutoff functions, find the new sources and 
//             // and propagate to all neighboring subdomains 
//             // (possible 8 in 2D, 26 in 3D)
//             // socketstream lsock(vishost, visport);
//             // PlotSolution(sol_ext,lsock,ip,true, true); cin.get();
//             TransferSources(l,ip, sol_ext);
//             Vector cfsol_ext(sol_ext.Size());

//             // cut off the ip solution to all possible directions
//             Array<int>directions(2); directions = 0; 

//             // switch (l)
//             // {
//             //    case 0:  
//             //       if (i+1<nx) directions[0] = 1;
//             //       if (j+1<ny) directions[1] = 1;
//             //         break;
//             //    case 1:  
//             //       if (i>0) directions[0] = -1;
//             //       if (j+1<ny) directions[1] = 1;
//             //       break;
//             //    case 2:  
//             //       if (i+1<nx) directions[0] = 1;
//             //       if (j>0) directions[1] = -1;
//             //       break;
//             //    default: 
//             //       if (i>0) directions[0] = -1;
//             //       if (j>0) directions[1] = -1;
//             //    break;
//             // }

//             if (i+1<nx) directions[0] = 1;
//             if (j+1<ny) directions[1] = 1;
//             // cout << "ip = " << ip << endl;
//             // cout << "i, j = " << i <<", " << j << endl;
//             GetCutOffSolution(sol_ext,cfsol_ext,ip,directions,ovlpnrlayers,true);
//             sol_ext = cfsol_ext;
//             directions = 0;
//             if (i>0) directions[0] = -1;
//             if (j>0) directions[1] = -1;
//             GetCutOffSolution(sol_ext,cfsol_ext,ip,directions,ovlpnrlayers,true);
//             // std::cout << std::fixed;
//             // std::cout << std::setprecision(10);
//             // cout << "cf_local_ext norm " << cfsol_ext.Norml2()<< endl;
//             cfsol_ext.GetSubVector(*Dof2PmlDof, sol_local);
//             // cout << "cf_local norm " << cfsol_ext.Norml2()<< endl;
//             // socketstream zsock(vishost, visport);
//             // PlotSolution(cfsol_ext,zsock,ip,true,true); cin.get();


//             znew = 0.0;
//             znew.SetSubVector(*Dof2GlobalDof, sol_local);
//             z+=znew;
//             // std::cout << std::fixed;
//             // std::cout << std::setprecision(10);
//             // cout << "znew norm " << znew.Norml2()<< endl;
//             // cout << "z norm " << z.Norml2()<< endl;

//             Array<int> * nDof2GlobalDof = &nvlp_prob->Dof2GlobalDof[ip];
//             // Vector zsol(nDof2GlobalDof->Size()); zsol=0.0;
//             // z.GetSubVector(*nDof2GlobalDof, zsol);
//             // *usol[ip] += zsol;
//             // cout << "ip = " << ip << endl;
//             // socketstream znewsock(vishost, visport);
//             // PlotSolution(znew,znewsock,0); cin.get();
//             // socketstream zsock(vishost, visport);
//             // PlotSolution(z,zsock,0); cin.get();
//          }
//          // socketstream zsock(vishost, visport);
//          // PlotSolution(z,zsock,0); cin.get();
//       }
//    }

//    // // Put the solution together
//    // z = 0.0;
//    // for (int ip = nrpatch-1; ip>=0; ip--)
//    // {
//    //    Array<int> * nDof2GlobalDof = &nvlp_prob->Dof2GlobalDof[ip];
//    //    z.SetSubVector(*nDof2GlobalDof, *usol[ip]);
//    // }

//    // socketstream z1sock(vishost, visport);
//    // PlotSolution(z,z1sock,0); cin.get();

// }

// void DiagST::PlotSolution(Vector & sol, socketstream & sol_sock, int ip, 
//                   bool localdomain, bool pmldomain) const
// {
//    FiniteElementSpace * fes;
//    if (!localdomain)
//    {
//       fes = bf->FESpace();
//    }
//    else if (!pmldomain)
//    {
//       fes = ovlp_prob->fespaces[ip];
//    }
//    else
//    {
//       fes = ovlp_prob->PmlFespaces[ip];
//    }
//    Mesh * mesh = fes->GetMesh();
//    GridFunction gf(fes);
//    double * data = sol.GetData();
//    gf.SetData(data);
   
//    string keys;
//    // if (ip == 0) 
//    keys = "keys mrRljc\n";
//    // sol_sock << "solution\n" << *mesh << gf << keys << "valuerange -0.1 0.1 \n"  << flush;
//    sol_sock << "solution\n" << *mesh << gf << keys << flush;
// }

// void DiagST::GetCutOffSolution(const Vector & sol, Vector & cfsol, 
//                                int ip, Array<int> directions, int ovlpnlayers, bool local) const
// {
//    // int l,k;
//    int d = directions.Size();
//    int directx = directions[0]; // 1,0,-1
//    int directy = directions[1]; // 1,0,-1
//    int directz;
//    if (d ==3) directz = directions[2];

//    Mesh * mesh = ovlp_prob->fespaces[ip]->GetMesh();
   
//    Vector pmin, pmax;
//    mesh->GetBoundingBox(pmin, pmax);
//    double h = GetUniformMeshElementSize(mesh);

//    Array2D<double> pmlh(dim,2); pmlh = 0.0;
   
//    if (directions[0]==1)
//    {
//       pmlh[0][1] = h*ovlpnlayers;
//    }
//    if (directions[0]==-1)
//    {
//       pmlh[0][0] = h*ovlpnlayers;
//    }
//    if (directions[1]==1)
//    {
//       pmlh[1][1] = h*ovlpnlayers;
//    }
//    if (directions[1]==-1)
//    {
//       pmlh[1][0] = h*ovlpnlayers;
//    }


//    CutOffFnCoefficient cf(CutOffFncn, pmin, pmax, pmlh);


//    double * data = sol.GetData();

//    FiniteElementSpace * fespace;
//    if (!local)
//    {
//       fespace = bf->FESpace();
//    }
//    else
//    {
//       fespace = ovlp_prob->PmlFespaces[ip];
//    }
   
//    int n = fespace->GetTrueVSize();

//    GridFunction solgf_re(fespace, data);
//    GridFunction solgf_im(fespace, &data[n]);

//    GridFunctionCoefficient coeff1_re(&solgf_re);
//    GridFunctionCoefficient coeff1_im(&solgf_im);

//    ProductCoefficient prod_re(coeff1_re, cf);
//    ProductCoefficient prod_im(coeff1_im, cf);

//    ComplexGridFunction gf(fespace);
//    gf.ProjectCoefficient(prod_re,prod_im);

//    cfsol.SetSize(sol.Size());
//    cfsol = gf;
// }


// void DiagST::GetChiRes(const Vector & res, Vector & cfres, 
//                        int ip, Array<int> directions, int nlayers) const
// {
//    // int l,k;
//    int d = directions.Size();
//    int directx = directions[0]; // 1,0,-1
//    int directy = directions[1]; // 1,0,-1
//    int directz;
//    if (d ==3) directz = directions[2];

//    Mesh * mesh = ovlp_prob->fespaces[ip]->GetMesh();
//    double h = GetUniformMeshElementSize(mesh);
   
//    Vector pmin, pmax;
//    mesh->GetBoundingBox(pmin, pmax);

//    Array2D<double> pmlh(dim,2); pmlh = 0.0;
   
//    if (directions[0]==1)
//    {
//       // pmlh[0][1] = h*nlayers;
//       pmlh[0][1] = h;
//    }
//    if (directions[0]==-1)
//    {
//       // pmlh[0][0] = h*nlayers;
//       pmlh[0][0] = h;
//    }
//    if (directions[1]==1)
//    {
//       // pmlh[1][1] = h*nlayers;
//       pmlh[1][1] = h;
//    }
//    if (directions[1]==-1)
//    {
//       // pmlh[1][0] = h*nlayers;
//       pmlh[1][0] = h;
//    }

//    // CutOffFnCoefficient cf(ChiFncn, pmin, pmax, pmlh);
//    // instead of the discontinous ChiFncn call the cutoff with 
//    // very fast decay!!
//    CutOffFnCoefficient cf(CutOffFncn, pmin, pmax, pmlh);

//    double * data = res.GetData();

//    FiniteElementSpace * fespace;
//    fespace = ovlp_prob->fespaces[ip];
   
//    int n = fespace->GetTrueVSize();

//    GridFunction solgf_re(fespace, data);
//    GridFunction solgf_im(fespace, &data[n]);

//    GridFunctionCoefficient coeff1_re(&solgf_re);
//    GridFunctionCoefficient coeff1_im(&solgf_im);

//    ProductCoefficient prod_re(coeff1_re, cf);
//    ProductCoefficient prod_im(coeff1_im, cf);

//    ComplexGridFunction gf(fespace);
//    gf.ProjectCoefficient(prod_re,prod_im);

//    cfres.SetSize(res.Size());
//    cfres = gf;

//    GridFunction cgf(fespace);
//    cgf.ProjectCoefficient(cf);

//    // char vishost[] = "localhost";
//    // int  visport   = 19916;
//    // socketstream cfsock(vishost, visport);
//    // PlotSolution(cgf,cfsock,ip,true,false); cin.get();
// }

// DiagST::~DiagST()
// {
//    for (int ip = 0; ip<nrpatch; ++ip)
//    {
//       delete PmlMatInv[ip];
//       delete PmlMat[ip];
//    }
//    PmlMat.DeleteAll();
//    PmlMatInv.DeleteAll();
//    for (int ip=0; ip<nrpatch; ip++)
//    {
//       delete f_orig[ip];
//       for (int i=0;i<nsweeps; i++)
//       {
//          delete f_transf[ip][i];
//       }
//    }
// }

// void DiagST::Getijk(int ip, int & i, int & j, int & k) const
// {
//    k = ip/(nxyz[0]*nxyz[1]);
//    j = (ip-k*nxyz[0]*nxyz[1])/nxyz[0];
//    i = (ip-k*nxyz[0]*nxyz[1])%nxyz[0];
// }

// int DiagST::GetPatchId(const Array<int> & ijk) const
// {
//    int d=ijk.Size();
//    if (d==2)
//    {
//       return subdomains(ijk[0],ijk[1],0);
//    }
//    else
//    {
//       return subdomains(ijk[0],ijk[1],ijk[2]);
//    }
// }

// int DiagST::SourceTransfer(const Vector & Psi0, Array<int> direction, int ip0, Vector & Psi1) const
// {
//    // For now 2D problems only
//    // Directions
//    // direction (1,1)
//    int i0,j0,k0;
//    Getijk(ip0,i0,j0,k0);

//    int i1 = i0+direction[0];   
//    int j1 = j0+direction[1];   
//    Array<int> ij(2); ij[0]=i1; ij[1]=j1;
//    int ip1 = GetPatchId(ij);

//    MFEM_VERIFY(i1 < nxyz[0] && i1>=0, "SourceTransfer: i1 out of bounds");
//    MFEM_VERIFY(j1 < nxyz[1] && j1>=0, "SourceTransfer: j1 out of bounds");

//    Array<int> * Dof2GlobalDof0 = &ovlp_prob->Dof2GlobalDof[ip0];
//    Array<int> * Dof2GlobalDof1 = &ovlp_prob->Dof2GlobalDof[ip1];
//    Psi1.SetSize(Dof2GlobalDof1->Size()); Psi1=0.0;
//    Vector r(2*bf->FESpace()->GetTrueVSize());
//    r = 0.0;
//    r.SetSubVector(*Dof2GlobalDof0,Psi0);

//    Vector zloc(Psi1.Size()); zloc = 0.0;
//    r.GetSubVector(*Dof2GlobalDof1,zloc);

//    char vishost[] = "localhost";
//    int  visport   = 19916;

//    // cout << "ip1 = " << ip1 << endl;
//    // socketstream sock(vishost, visport);
//    // PlotSolution(zloc,sock,ip1,true,false); cin.get();

//       // extend
//    Vector psi_ext(PmlMat[ip1]->Height()); psi_ext = 0.0;
//    Vector zloc_ext(PmlMat[ip1]->Height()); zloc_ext = 0.0;
//    Array<int> * Dof2PmlDof1 = &ovlp_prob->Dof2PmlDof[ip1];
//    zloc_ext.SetSubVector(*Dof2PmlDof1,zloc);

//    // socketstream extsock(vishost, visport);
//    // PlotSolution(zloc_ext,extsock,ip1,true,true); cin.get();

//    PmlMat[ip1]->Mult(zloc_ext,psi_ext);
//    psi_ext *=-1.0;

//    // socketstream ressock(vishost, visport);
//    // PlotSolution(psi_ext,ressock,ip1,true,true); cin.get();

//    Vector Psi(Dof2GlobalDof1->Size()); Psi=0.0;
//    psi_ext.GetSubVector(*Dof2PmlDof1,Psi);

      

//    // restrict to the novlp
//    //-------------------------------------
//    // Array<int> * nDof2GlobalDof1 = &nvlp_prob->Dof2GlobalDof[ip1];
//    // Vector raux(nDof2GlobalDof1->Size()); raux = 0.0;
//    // r = 0.0;
//    // r.SetSubVector(*Dof2GlobalDof1,Psi);
//    // r.GetSubVector(*nDof2GlobalDof1,raux);
//    // r = 0.0;
//    // r.SetSubVector(*nDof2GlobalDof1,raux);
//    // r.GetSubVector(*Dof2GlobalDof1,Psi1);
//    // Vector psicopy(Psi1);
   
//    //-------------------------------------


//    // Psi1 = Psi;
//    // int nx = nxyz[0];
//    // int ny = nxyz[1];
//    Array<int> direct(2); direct = 0;

//    // if (i1>0) direct[0] = -1;
//    // if (j1>0) direct[1] = -1;
//    // GetChiRes(Psi1, Psi,ip1,direct, ovlpnrlayers);

//    direct = 0;
//    direct[0] = -direction[0];
//    direct[1] = -direction[1];
//    // if (i1+1<nx) direct[0] = 1;
//    // if (j1+1<ny) direct[1] = 1;
//    GetChiRes(Psi, Psi1,ip1,direct, ovlpnrlayers);

//    // socketstream zsock(vishost, visport);
//    // PlotSolution(Psi1,zsock,ip1,true); cin.get();

//    // socketstream cressock(vishost, visport);
//    // psicopy-=Psi1;

//    // socketstream cressock(vishost, visport);
//    // PlotSolution(psicopy,cressock,ip1,true,false); cin.get();
//    // cout << "norm = " << psicopy.Norml2();
//    return ip1;
// }

// void DiagST::ConstructDirectionsMap()
// {
//    // total of 8 possible directions of transfer (2D)
//    // form left            ( 1 ,  0)
//    // form left-above      ( 1 , -1)
//    // form left-below      ( 1 ,  1)
//    // form right           (-1 ,  0)
//    // form right-below     (-1 ,  1)
//    // form right-above     (-1 , -1)
//    // form above           ( 0 , -1)
//    // form below           ( 0 ,  1)
//    ntransf_directions = pow(3,dim);

//    dirx.SetSize(ntransf_directions);
//    diry.SetSize(ntransf_directions);
//    int n=3;
//    Array<int> ijk(dim);
//    if (dim==2)
//    {
//       for (int i=-1; i<=1; i++) // directions x
//       {
//          for (int j=-1; j<=1; j++) // directions y
//          {
//             ijk[0]=i;
//             ijk[1]=j;
//             int k=GetDirectionId(ijk);
//             dirx[k]=i;
//             diry[k]=j;
//          }
//       }
//    }
//    else if (dim==3)
//    {
//       dirz.SetSize(ntransf_directions);
//       for (int i=-1; i<=1; i++) // directions x
//       {
//          for (int j=-1; j<=1; j++) // directions y
//          {
//             for (int k=-1; k<=1; k++) // directions zß
//             {
//                ijk[0]=i;
//                ijk[1]=j;
//                ijk[2]=k;
//                int l=GetDirectionId(ijk);
//                dirx[l]=i;
//                diry[l]=j;
//                dirz[l]=k;
//             }
//          }
//       }
//    }

//    // cout << "dirx = " << endl;
//    // dirx.Print(cout,ntransf_directions);
//    // cout << "diry = " << endl;
//    // diry.Print(cout,ntransf_directions);

//    // if (dim==2)
//    // {
//    //    for (int id=0; id<9; id++)
//    //    {
//    //       GetDirectionijk(id,ijk);
//    //       // cout << "for id = " << id << ": (" <<ijk[0] << ", " << ijk[1] << ")" << endl;
//    //    }
//    // }
//    // else
//    // {
//    //    cout << "dirz = " << endl;
//    //    dirz.Print(cout,ntransf_directions);
//    //    for (int id=0; id<27; id++)
//    //    {
//    //       GetDirectionijk(id,ijk);
//    //       // cout << "for id = " << id << ": (" <<ijk[0] << ", " <<ijk[1] << ", " << ijk[2] << ")" << endl;
//    //    }
//    // }
// }

// int DiagST::GetDirectionId(const Array<int> & ijk) const
// {
//    int d = ijk.Size();
//    int n=3;
//    if (d==2)
//    {
//       return (ijk[0]+1)*n+(ijk[1]+1);
//    }
//    else
//    {
//       return (ijk[0]+1)*n*n+(ijk[1]+1)*n+ijk[2]+1;
//    }
// }

// void DiagST::GetDirectionijk(int id, Array<int> & ijk) const
// {
//    int d = ijk.Size();
//    int n=3;
//    if (d==2)
//    {
//       ijk[0]=id/n - 1;
//       ijk[1]=id%n - 1;
//    }
//    else
//    {
//       ijk[0]=id/(n*n)-1;
//       ijk[1]=(id-(ijk[0]+1)*n*n)/n - 1;
//       ijk[2]=(id-(ijk[0]+1)*n*n)%n - 1;
//    }
//    // cout << "ijk = " ; ijk.Print();
// }




// void DiagST::TransferSources(int sweep, int ip0, Vector & sol_ext) const
// {
//    // Find all neighbors of patch ip
//    int nx = nxyz[0];
//    int ny = nxyz[1];
//    int i0, j0, k0;
//    Getijk(ip0, i0,j0,k0);
//    // cout << "Transfer to : " << endl;
//    // loop through possible directions
//    for (int i=-1; i<2; i++)
//    {
//       int i1 = i0 + i;
//       if (i1 <0 || i1>=nx) continue;
//       for (int j=-1; j<2; j++)
//       {
//          if (i==0 && j==0) continue;
//          int j1 = j0 + j;
//          if (j1 <0 || j1>=ny) continue;
//          // cout << "(" << i1 << "," << j1 <<"), ";
//          // Find ip 1
//          Array<int> ij1(2); ij1[0] = i1; ij1[1]=j1;
//          int ip1 = GetPatchId(ij1);
//          // cout << "ip1 = " << ip1;
//          // cout << " in the direction of (" << i <<", " <<j <<")" << endl;
//          Array<int> directions(2);
//          directions[0] = i;
//          directions[1] = j;
//          Vector cfsol_ext;
//          Vector res_ext(sol_ext.Size());
//          GetCutOffSolution(sol_ext,cfsol_ext,ip0,directions,ovlpnrlayers,true);
        
//          // std::cout << std::fixed;
//          // std::cout << std::setprecision(10);
//          // cout << "sol  norm = " << sol_ext.Norml2() << endl;
//          // cout << "cfsol  norm = " << cfsol_ext.Norml2() << endl;

//          // //---------------------------------------
//          // char vishost[] = "localhost";
//          // int  visport   = 19916;
//          // socketstream cfsock(vishost, visport);
//          // PlotSolution(cfsol_ext,cfsock,ip0,true,true); cin.get();
//          //---------------------------------------
         
//          // res_ext*= -1.0;
//          Array<int> *Dof2PmlDof = &ovlp_prob->Dof2PmlDof[ip0];
//          // Vector res_local(Dof2PmlDof->Size()); res_local = 0.0;
//          // res_ext.GetSubVector(*Dof2PmlDof,res_local);

//          //-----------------------------
//          // pass to ip1 and calculate residual there
//          Vector sol_local(Dof2PmlDof->Size()); sol_local = 0.0;
//          cfsol_ext.GetSubVector(*Dof2PmlDof,sol_local);
//          //-----------------------------

//          // cout << "ip0 = " << ip0 << endl;
//          // socketstream sock(vishost, visport);
//          // PlotSolution(sol_local,sock,ip0,true,false); cin.get();




//          // Find the minumum sweep number to transfer the source that 
//          // satisfies the two rules
//          for (int l=sweep; l<nsweeps; l++)
//          {
//             // Conditions on sweeps
//             // Rule 1: the transfer source direction has to be similar with 
//             // the sweep direction
//             int is = sweeps(l,0); 
//             int js = sweeps(l,1);
//             int ddot = is*i + js * j;
//             // cout << "(i,j) = (" << i <<"," <<j <<")" << endl;
//             // cout << "(is,js) = (" << is <<"," <<js <<")" << endl;
//             // cout << "ip0 , ip1 = " << ip0 << ", " << ip1 << endl;
//             if (ddot <= 0) continue;

//             // Rule 2: The horizontal or vertical transfer source cannot be used
//             // in a later sweep that with opposite directions

//             if (i==0 || j == 0) // Case of horizontal or vertical transfer source
//             {
//                int il = sweeps(l,0);
//                int jl = sweeps(l,1);
//                // skip if the two sweeps have opposite direction
//                if (is == -il && js == -jl) continue;
//             }
//             // cout << "Passing ip0 = " << ip0 << " to ip1 = " << ip1 
//                //   << " to sweep no l = " << l << endl;  
//             Vector raux;
//             // if (directions[0] ==  1 && directions[1] == 1)
//             // {
//                // res_local = sol_local;
//             // }
//             // cout << res_local.Norml2() << endl;

//             int jp1 = SourceTransfer(sol_local,directions,ip0,raux);
//             // int jp1 = SourceTransfer(res_local,directions,ip0,raux);
//             MFEM_VERIFY(ip1 == jp1, "Check SourceTransfer patch id");
//             MFEM_VERIFY(f_transf[ip1][l]->Size()==raux.Size(), 
//                         "Transfer Sources: inconsistent size");
//             // cout << "l = " << l << endl;
//             *f_transf[ip1][l]+=raux;
//             // *f_transf[ip1][l]-=raux;
//             // *f_transf[ip1][l] *= -1.0;

//             break;
//          }
            
//       }  
//    }
// }










// // //Diagonal Source Transfer Preconditioner

// // #include "DiagST.hpp"

// // DiagST::DiagST(SesquilinearForm * bf_, Array2D<double> & Pmllength_, 
// //          double omega_, Coefficient * ws_,  int nrlayers_)
// //    : Solver(2*bf_->FESpace()->GetTrueVSize(), 2*bf_->FESpace()->GetTrueVSize()), 
// //      bf(bf_), Pmllength(Pmllength_), omega(omega_), ws(ws_), nrlayers(nrlayers_)
// // {
// //    Mesh * mesh = bf->FESpace()->GetMesh();
// //    dim = mesh->Dimension();

// //    // ----------------- Step 1 --------------------
// //    // Introduce 2 layered partitios of the domain 
// //    // 
// //    int partition_kind;

// //    // 1. Ovelapping partition with overlap = 2h 
// //    partition_kind = 2; // Non Overlapping partition 
// //    int nx=2;
// //    int ny=2; 
// //    int nz=1;
// //    povlp = new MeshPartition(mesh, partition_kind,nx,ny,nz,2);
// //    nxyz[0] = povlp->nxyz[0];
// //    nxyz[1] = povlp->nxyz[1];
// //    nxyz[2] = povlp->nxyz[2];
// //    nrpatch = povlp->nrpatch;
// //    subdomains = povlp->subdomains;

// //    //
// //    // ----------------- Step 1a -------------------
// //    // Save the partition for visualization
// //    // SaveMeshPartition(povlp->patch_mesh, "output/mesh_ovlp.", "output/sol_ovlp.");

// //    // // // ------------------Step 2 --------------------
// //    // // // Construct the dof maps from subdomains to global (for the extended and not)
   
// //    ovlp_prob  = new DofMap(bf,povlp,nrlayers); 

// //    // ------------------Step 3 --------------------
// //    // Assemble the PML Problem matrices and factor them
// //    PmlMat.SetSize(nrpatch);
// //    PmlMatInv.SetSize(nrpatch);
// //    for (int ip=0; ip<nrpatch; ip++)
// //    {
// //       PmlMat[ip] = GetPmlSystemMatrix(ip);
// //       PmlMatInv[ip] = new KLUSolver;
// //       PmlMatInv[ip]->SetOperator(*PmlMat[ip]);
// //    }


// //    nsweeps = pow(2,dim);
// //    sweeps.SetSize(nsweeps,dim);
// //    // 2D
// //    sweeps(0,0) =  1; sweeps(0,1) = 1;
// //    sweeps(1,0) = -1; sweeps(1,1) = 1;
// //    sweeps(2,0) =  1; sweeps(2,1) =-1;
// //    sweeps(3,0) = -1; sweeps(3,1) =-1;

// //    // Set up src arrays size
// //    f_orig.SetSize(nrpatch);
// //    f_transf.SetSize(nrpatch);
// //    usol.SetSize(nrpatch);


// //    // Construct a simple map used for directions of transfer
// //    ConstructDirectionsMap();
// //    for (int ip=0; ip<nrpatch; ip++)
// //    {
// //       int n = 2*ovlp_prob->fespaces[ip]->GetTrueVSize(); // (x 2 for complex ) 
// //       int npml = 2*ovlp_prob->PmlFespaces[ip]->GetTrueVSize(); // (x 2 for complex ) 
// //       f_orig[ip] = new Vector(n); *f_orig[ip] = 0.0;
// //       f_transf[ip].SetSize(nsweeps);
// //       usol[ip].SetSize(nsweeps);
// //       for (int i=0;i<nsweeps; i++)
// //       {
// //          f_transf[ip][i] = new Vector(n);
// //          usol[ip][i] = new Vector(npml);
// //       }
// //    }

// // }

// // SparseMatrix * DiagST::GetPmlSystemMatrix(int ip)
// // {
// //    double h = GetUniformMeshElementSize(ovlp_prob->PmlMeshes[ip]);
// //    Array2D<double> length(dim,2);
// //    length = h*(nrlayers);

// //    CartesianPML pml(ovlp_prob->PmlMeshes[ip], length);
// //    pml.SetOmega(omega);

// //    Array <int> ess_tdof_list;
// //    if (ovlp_prob->PmlMeshes[ip]->bdr_attributes.Size())
// //    {
// //       Array<int> ess_bdr(ovlp_prob->PmlMeshes[ip]->bdr_attributes.Max());
// //       ess_bdr = 1;
// //       ovlp_prob->PmlFespaces[ip]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
// //    }

// //    ConstantCoefficient one(1.0);
// //    ConstantCoefficient sigma(-pow(omega, 2));

// //    PmlMatrixCoefficient c1_re(dim,pml_detJ_JT_J_inv_Re,&pml);
// //    PmlMatrixCoefficient c1_im(dim,pml_detJ_JT_J_inv_Im,&pml);

// //    PmlCoefficient detJ_re(pml_detJ_Re,&pml);
// //    PmlCoefficient detJ_im(pml_detJ_Im,&pml);

// //    ProductCoefficient c2_re0(sigma, detJ_re);
// //    ProductCoefficient c2_im0(sigma, detJ_im);

// //    ProductCoefficient c2_re(c2_re0, *ws);
// //    ProductCoefficient c2_im(c2_im0, *ws);

// //    SesquilinearForm a(ovlp_prob->PmlFespaces[ip],ComplexOperator::HERMITIAN);

// //    a.AddDomainIntegrator(new DiffusionIntegrator(c1_re),
// //                          new DiffusionIntegrator(c1_im));
// //    a.AddDomainIntegrator(new MassIntegrator(c2_re),
// //                          new MassIntegrator(c2_im));
// //    a.Assemble();

// //    OperatorPtr Alocal;
// //    a.FormSystemMatrix(ess_tdof_list,Alocal);
// //    ComplexSparseMatrix * AZ_ext = Alocal.As<ComplexSparseMatrix>();
// //    SparseMatrix * Mat = AZ_ext->GetSystemMatrix();
// //    Mat->Threshold(0.0);
// //    return Mat;
// // }


// // void DiagST::Mult(const Vector &r, Vector &z) const
// // {
// //    // Step 0
// //    // Restrict original sources to the patches
// //    for (int ip=0; ip<nrpatch; ip++)
// //    {
// //       *f_orig[ip] = 0.0;
// //       for (int i=0;i<nsweeps; i++)
// //       {
// //          *f_transf[ip][i] = 0.0;
// //          *usol[ip][i] = 0.0;
// //       }
// //    }
// //    for (int ip=0; ip<nrpatch; ip++)
// //    {
// //       Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
// //       r.GetSubVector(*Dof2GlobalDof,*f_orig[ip]);
// //    }

// //    char vishost[] = "localhost";
// //    int  visport   = 19916;
// //    z = 0.0; 
// //    Vector znew(z);
// //    Vector rnew(r);
// //    Vector raux(r); raux = 0.0;
// //    Vector z1(z);
// //    Vector z2(z);
// //    Vector z3(z);
// //    Vector z4(z);
// //    znew = 0.0;
   
// //    // in 2D there are a total of 4 sweeps
// //    // with nx + ny - 1 serial steps each
// //    // --------------------------------------------
// //    //       Sweep in the direction (1,1)
// //    // --------------------------------------------
// //    int nx = nxyz[0];
// //    int ny = nxyz[1];

// //    int nsteps = nx + ny - 1;
// //    // Sweep number
// //    for (int l=0; l<1; l++)
// //    {
// //       for (int s = 0; s<nsteps; s++)
// //       {
// //          // the patches involved are the ones such that
// //          // i+j = s
// //          for (int i=0;i<nx; i++)
// //          {
// //             int j = s-i;
// //             if (j<0 || j>=ny) continue;

// //             // find patch id
// //             Array<int> ij(2); ij[0] = i; ij[1]=j;
// //             int ip = GetPatchId(ij);

// //             // Solve the PML problem in patch ip with all sources
// //             // Original and all transfered (maybe some of them)
// //             Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
// //             Array<int> * Dof2PmlDof = &ovlp_prob->Dof2PmlDof[ip];
// //             int ndofs = Dof2GlobalDof->Size();
// //             Vector sol_local(ndofs);
// //             Vector res_local(ndofs);
// //             res_local = *f_orig[ip];

// //             res_local += *f_transf[ip][l];
// //             // Extend by zero to the PML mesh
// //             int nrdof_ext = PmlMat[ip]->Height();
         
// //             Vector res_ext(nrdof_ext); res_ext = 0.0;
// //             Vector sol_ext(nrdof_ext); sol_ext = 0.0;

// //             res_ext.SetSubVector(*Dof2PmlDof,res_local);
// //             PmlMatInv[ip]->Mult(res_ext, sol_ext);
// //             // *usol[l][ip] = sol_ext;
// //             // Multiply with the cutoff functions, find the new sources and 
// //             // and propagate to all neighboring subdomains 
// //             // (possible 8 in 2D, 26 in 3D)
// //             TransferSources(l,ip, sol_ext);
// //             Vector cfsol_ext(sol_ext.Size());

// //             // cut off the ip solution to all possible directions
// //             Array<int>directions(2); directions = 0; 
// //             if (i+1<nx) directions[0] = 1;
// //             if (j+1<ny) directions[1] = 1;
// //             GetCutOffSolution(sol_ext,cfsol_ext,ip,directions,true);
// //             // sol_ext = cfsol_ext;
// //             // directions = 0.0;
// //             // if (i>0) directions[0] = -1;
// //             // if (j>0) directions[1] = -1;
// //             // GetCutOffSolution(sol_ext,cfsol_ext,ip,directions,true);
// //             cfsol_ext.GetSubVector(*Dof2PmlDof, sol_local);
// //             znew = 0.0;
// //             znew.SetSubVector(*Dof2GlobalDof, sol_local);
// //             // z1.AddElementVector(*Dof2GlobalDof, sol_local);
// //             z1+=znew;
// //             socketstream sub1_sock1(vishost, visport);
// //             PlotSolution(z1,sub1_sock1,0); cin.get();
// //          }
// //       }
// //    }

// //    // PlotSolution(z1,sub_sock1, 0);
// //    // cin.get();

// //    z +=z1;
// //    // A->Mult(z,raux); rnew = r; rnew -=raux; 

// //    // for (int ip=0; ip<nrpatch; ip++)
// //    // {
// //    //    *f_orig[ip] = 0.0;
// //    //    for (int i=0;i<nsweeps; i++)
// //    //    {
// //    //       *f_transf[ip][i] = 0.0;
// //    //    }
// //    // }
// //    // for (int ip=0; ip<nrpatch; ip++)
// //    // {
// //    //    Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
// //    //    rnew.GetSubVector(*Dof2GlobalDof,*f_orig[ip]);
// //    // }


// //    // for (int l=1; l<2; l++)
// //    // {
// //    //    for (int s = 0; s<nsteps; s++)
// //    //    {
// //    //       // the patches involved are the ones such that
// //    //       // i+j = s
// //    //       // cout << "Step no: " << s << endl;
// //    //       for (int i=0;i<nx; i++)
// //    //       {
// //    //          int j = s-nx+i+1;
// //    //          // cout << "1:Patch no: (" << i <<"," << j << ")" << endl; 
// //    //          if (j<0 || j>=ny) continue;
// //    //          // cout << "2:Patch no: (" << i <<"," << j << ")" << endl; 
// //    //          // cin.get();
// //    //          // find patch id
// //    //          Array<int> ij(2); ij[0] = i; ij[1]=j;
// //    //          int ip = GetPatchId(ij);
// //    //          // cout << "ip = " << ip << endl;

// //    //          // Solve the PML problem in patch ip with all sources
// //    //          // Original and all transfered (maybe some of them)
// //    //          Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
// //    //          Array<int> * Dof2PmlDof = &ovlp_prob->Dof2PmlDof[ip];
// //    //          int ndofs = Dof2GlobalDof->Size();
// //    //          Vector sol_local(ndofs);
// //    //          Vector res_local(ndofs);
// //    //          res_local = *f_orig[ip];
// //    //          // res_local = 0.0;
// //    //          res_local += *f_transf[ip][l];
// //    //          // Extend by zero to the PML mesh
// //    //          int nrdof_ext = PmlMat[ip]->Height();
         
// //    //          Vector res_ext(nrdof_ext); res_ext = 0.0;
// //    //          Vector sol_ext(nrdof_ext); sol_ext = 0.0;

// //    //          res_ext.SetSubVector(*Dof2PmlDof,res_local);
// //    //          PmlMatInv[ip]->Mult(res_ext, sol_ext);

// //    //          // Multiply with the cutoff functions, find the new sources and 
// //    //          // and propagate to all neighboring subdomains 
// //    //          // (possible 8 in 2D, 26 in 3D)
// //    //          TransferSources(l,ip, sol_ext);
// //    //          Vector cfsol_ext(sol_ext.Size());

// //    //          // cut off the ip solution to all possible directions
// //    //          Array<int>directions(2); directions = 0; 
// //    //          if (i>0) directions[0] = -1;
// //    //          if (j+1<ny) directions[1] = 1;
// //    //          GetCutOffSolution(sol_ext,cfsol_ext,ip,directions,true);
// //    //          // sol_ext = cfsol_ext;
// //    //          // directions = 0;
// //    //          // if (i+1<nx) directions[0] = 1;
// //    //          // if (j>0) directions[1] = -1;
// //    //          // GetCutOffSolution(sol_ext,cfsol_ext,ip,directions,true);

// //    //          cfsol_ext.GetSubVector(*Dof2PmlDof, sol_local);
// //    //          znew = 0.0;
// //    //          znew.SetSubVector(*Dof2GlobalDof, sol_local);
// //    //          // z2.AddElementVector(*Dof2GlobalDof, sol_local);
// //    //          z2+=znew;
// //    //       }
// //    //    }
// //    // }

// //    // // // // socketstream sub_sock2(vishost, visport);
// //    // // // // PlotSolution(z2,sub_sock2, 0);
// //    // // // // cin.get();


// //    // z +=z2;
// //    // A->Mult(z,raux); rnew = r;  rnew -=raux; 
// //    // for (int ip=0; ip<nrpatch; ip++)
// //    // {
// //    //    *f_orig[ip] = 0.0;
// //    //    for (int i=0;i<nsweeps; i++)
// //    //    {
// //    //       *f_transf[ip][i] = 0.0;
// //    //    }
// //    // }
// //    // for (int ip=0; ip<nrpatch; ip++)
// //    // {
// //    //    Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
// //    //    rnew.GetSubVector(*Dof2GlobalDof,*f_orig[ip]);
// //    // }

// //    // for (int l=2; l<3; l++)
// //    // {
// //    //    for (int s = 0; s<nsteps; s++)
// //    //    {
// //    //       // the patches involved are the ones such that
// //    //       // i+j = s
// //    //       // cout << "Step no: " << s << endl;
// //    //       for (int i=0;i<nx; i++)
// //    //       {
// //    //          int j = nx+i-s-1;
// //    //          // cout << "1:Patch no: (" << i <<"," << j << ")" << endl; 
// //    //          if (j<0 || j>=ny) continue;
// //    //          // cout << "2:Patch no: (" << i <<"," << j << ")" << endl; 
// //    //          // cin.get();
// //    //          // find patch id
// //    //          Array<int> ij(2); ij[0] = i; ij[1]=j;
// //    //          int ip = GetPatchId(ij);
// //    //          // cout << "ip = " << ip << endl;

// //    //          // Solve the PML problem in patch ip with all sources
// //    //          // Original and all transfered (maybe some of them)
// //    //          Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
// //    //          Array<int> * Dof2PmlDof = &ovlp_prob->Dof2PmlDof[ip];
// //    //          int ndofs = Dof2GlobalDof->Size();
// //    //          Vector sol_local(ndofs);
// //    //          Vector res_local(ndofs);
// //    //          res_local = *f_orig[ip];
// //    //          // res_local = 0.0;
// //    //          res_local += *f_transf[ip][l];
// //    //          // Extend by zero to the PML mesh
// //    //          int nrdof_ext = PmlMat[ip]->Height();
         
// //    //          Vector res_ext(nrdof_ext); res_ext = 0.0;
// //    //          Vector sol_ext(nrdof_ext); sol_ext = 0.0;

// //    //          res_ext.SetSubVector(*Dof2PmlDof,res_local);
// //    //          PmlMatInv[ip]->Mult(res_ext, sol_ext);

// //    //          // Multiply with the cutoff functions, find the new sources and 
// //    //          // and propagate to all neighboring subdomains 
// //    //          // (possible 8 in 2D, 26 in 3D)
// //    //          TransferSources(l,ip, sol_ext);
// //    //          Vector cfsol_ext(sol_ext.Size());

// //    //          // cut off the ip solution to all possible directions
// //    //          Array<int>directions(2); directions = 0; 
// //    //          if (i+1<nx) directions[0] = 1;
// //    //          if (j>0) directions[1] = -1;
// //    //          GetCutOffSolution(sol_ext,cfsol_ext,ip,directions,true);
// //    //          // sol_ext = cfsol_ext;
// //    //          // directions = 0;
// //    //          // if (i>0) directions[0] = -1;
// //    //          // if (j+1<ny) directions[1] = 1;
// //    //          // GetCutOffSolution(sol_ext,cfsol_ext,ip,directions,true);

// //    //          cfsol_ext.GetSubVector(*Dof2PmlDof, sol_local);
// //    //          znew = 0.0;
// //    //          znew.SetSubVector(*Dof2GlobalDof, sol_local);
// //    //          // z3.AddElementVector(*Dof2GlobalDof, sol_local);
// //    //          z3+=znew;
// //    //       }
// //    //    }
// //    // }

// //    // z +=z3;
// //    // A->Mult(z,raux); rnew=r; rnew -=raux; 

// //    // for (int ip=0; ip<nrpatch; ip++)
// //    // {
// //    //    *f_orig[ip] = 0.0;
// //    //    for (int i=0;i<nsweeps; i++)
// //    //    {
// //    //       *f_transf[ip][i] = 0.0;
// //    //    }
// //    // }
// //    // for (int ip=0; ip<nrpatch; ip++)
// //    // {
// //    //    Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
// //    //    rnew.GetSubVector(*Dof2GlobalDof,*f_orig[ip]);
// //    // }


// //    // for (int l=3; l<4; l++)
// //    // {
// //    //    for (int s = 0; s<nsteps; s++)
// //    //    {
// //    //       // the patches involved are the ones such that
// //    //       // i+j = s
// //    //       // cout << "Step no: " << s << endl;
// //    //       for (int i=0;i<nx; i++)
// //    //       {
// //    //          int j = nx+ny-i-s-2;
// //    //          // cout << "1:Patch no: (" << i <<"," << j << ")" << endl; 
// //    //          if (j<0 || j>=ny) continue;
// //    //          // cout << "2:Patch no: (" << i <<"," << j << ")" << endl; 
// //    //          // cin.get();
// //    //          // find patch id
// //    //          Array<int> ij(2); ij[0] = i; ij[1]=j;
// //    //          int ip = GetPatchId(ij);
// //    //          // cout << "ip = " << ip << endl;

// //    //          // Solve the PML problem in patch ip with all sources
// //    //          // Original and all transfered (maybe some of them)
// //    //          Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
// //    //          Array<int> * Dof2PmlDof = &ovlp_prob->Dof2PmlDof[ip];
// //    //          int ndofs = Dof2GlobalDof->Size();
// //    //          Vector sol_local(ndofs);
// //    //          Vector res_local(ndofs);
// //    //          res_local = *f_orig[ip];
// //    //          // res_local = 0.0;
// //    //          res_local += *f_transf[ip][l];
// //    //          // Extend by zero to the PML mesh
// //    //          int nrdof_ext = PmlMat[ip]->Height();
         
// //    //          Vector res_ext(nrdof_ext); res_ext = 0.0;
// //    //          Vector sol_ext(nrdof_ext); sol_ext = 0.0;

// //    //          res_ext.SetSubVector(*Dof2PmlDof,res_local);
// //    //          PmlMatInv[ip]->Mult(res_ext, sol_ext);

// //    //          // Multiply with the cutoff functions, find the new sources and 
// //    //          // and propagate to all neighboring subdomains 
// //    //          // (possible 8 in 2D, 26 in 3D)
// //    //          TransferSources(l,ip, sol_ext);
// //    //          Vector cfsol_ext(sol_ext.Size());

// //    //          // cut off the ip solution to all possible directions
// //    //          Array<int>directions(2); directions = 0; 
// //    //          if (i>0) directions[0] = -1;
// //    //          if (j>0) directions[1] = -1;
// //    //          GetCutOffSolution(sol_ext,cfsol_ext,ip,directions,true);
// //    //          // sol_ext = cfsol_ext;
// //    //          // directions = 0;
// //    //          // if (i+1<nx) directions[0] = 1;
// //    //          // if (j+1<ny) directions[1] = 1;
// //    //          // GetCutOffSolution(sol_ext,cfsol_ext,ip,directions,true);

// //    //          cfsol_ext.GetSubVector(*Dof2PmlDof, sol_local);
// //    //          znew = 0.0;
// //    //          znew.SetSubVector(*Dof2GlobalDof, sol_local);
// //    //          // z4.AddElementVector(*Dof2GlobalDof, sol_local);
// //    //          z4+=znew;
// //    //       }
// //    //    }
// //    // }

// //    // z+=z4;
// //    // socketstream sock(vishost, visport);
// //    // PlotSolution(z,sock, 0);
// // }

// // void DiagST::PlotSolution(Vector & sol, socketstream & sol_sock, int ip) const
// // {
// //    FiniteElementSpace * fespace = bf->FESpace();
// //    Mesh * mesh = fespace->GetMesh();
// //    GridFunction gf(fespace);
// //    double * data = sol.GetData();
// //    // gf.SetData(&data[fespace->GetTrueVSize()]);
// //    gf.SetData(data);
   
// //    string keys;
// //    if (ip == 0) keys = "keys mrRljc\n";
// //    sol_sock << "solution\n" << *mesh << gf << keys << "valuerange -0.1 0.1 \n"  << flush;
// //    // sol_sock << "solution\n" << *mesh << gf << keys << flush;
// // }

// // void DiagST::GetCutOffSolution(const Vector & sol, Vector & cfsol, 
// //                                int ip0, Array<int> directions, bool local) const
// // {
// //    // int l,k;
// //    int d = directions.Size();
// //    int directx = directions[0]; // 1,0,-1
// //    int directy = directions[1]; // 1,0,-1
// //    int directz;
// //    if (d ==3) directz = directions[2];

// //    // cout << "ip0 = " << ip0 << endl; 

// //    int i0, j0, k0;
// //    Getijk(ip0,i0, j0, k0);
// //    // cout << "(i0,j0) = " << "(" <<i0 <<","<<j0<<")" << endl;

// //    // 2D for now...
// //    // Find the id of the neighboring patch
// //    int i1 = i0 + directx;
// //    int j1 = j0 + directy;
// //    MFEM_VERIFY(i1 < nxyz[0] && i1>=0, "GetCutOffSolution: i1 out of bounds");
// //    MFEM_VERIFY(j1 < nxyz[1] && j1>=0, "GetCutOffSolution: j1 out of bounds");
   
// //    Array<int> ijk(d);
// //    ijk[0] = i1;
// //    ijk[1] = j1;
// //    int ip1 = GetPatchId(ijk);

// //    // cout << "ip1 = " << ip1 << endl;
// //    // cout << "(i1,j1) = " << "(" << i1 <<","<<j1<<")" << endl;

// //    Mesh * mesh0 = ovlp_prob->fespaces[ip0]->GetMesh();
// //    Mesh * mesh1 = ovlp_prob->fespaces[ip1]->GetMesh();
   
// //    Vector pmin0, pmax0;
// //    Vector pmin1, pmax1;
// //    mesh0->GetBoundingBox(pmin0, pmax0);
// //    mesh1->GetBoundingBox(pmin1, pmax1);

// //    Array2D<double> h(dim,2); h = 0.0;
   
// //    if (directions[0]==1)
// //    {
// //       h[0][1] = pmax0[0] - pmin1[0];
// //    }
// //    if (directions[0]==-1)
// //    {
// //       h[0][0] = pmax1[0] - pmin0[0];
// //    }
// //    if (directions[1]==1)
// //    {
// //       h[1][1] = pmax0[1] - pmin1[1];
// //    }
// //    if (directions[1]==-1)
// //    {
// //       h[1][0] = pmax1[1] - pmin0[1];
// //    }

// //    pmin0.Print();
// //    pmax0.Print();
// //    h.Print();

// //    CutOffFnCoefficient cf(CutOffFncn, pmin0, pmax0, h);

// //    double * data = sol.GetData();

// //    FiniteElementSpace * fespace;
// //    if (!local)
// //    {
// //       fespace = bf->FESpace();
// //    }
// //    else
// //    {
// //       fespace = ovlp_prob->PmlFespaces[ip0];
// //    }
   
// //    int n = fespace->GetTrueVSize();
// //    // GridFunction cutF(fespace);
// //    // cutF.ProjectCoefficient(cf);
// //    // char vishost[] = "localhost";
// //    // int  visport   = 19916;

// //    // socketstream sub_sock1(vishost, visport);
// //    // sub_sock1 << "solution\n" << *fespace->GetMesh() << cutF << flush;
// //    // cin.get();


// //    GridFunction solgf_re(fespace, data);
// //    GridFunction solgf_im(fespace, &data[n]);

// //    // socketstream sub_sock(vishost, visport);
// //    // sub_sock << "solution\n" << *fespace->GetMesh() << solgf_re << flush;
// //    // cin.get();


// //    GridFunctionCoefficient coeff1_re(&solgf_re);
// //    GridFunctionCoefficient coeff1_im(&solgf_im);

// //    ProductCoefficient prod_re(coeff1_re, cf);
// //    ProductCoefficient prod_im(coeff1_im, cf);

// //    ComplexGridFunction gf(fespace);
// //    gf.ProjectCoefficient(prod_re,prod_im);

// //    cfsol.SetSize(sol.Size());
// //    cfsol = gf;
// //    // socketstream sub_sock2(vishost, visport);
// //    // sub_sock2 << "solution\n" << *fespace->GetMesh() << gf.real() << flush;
// //    // cin.get();
// // }


// // void DiagST::GetChiRes(const Vector & res, Vector & cfres, 
// //                        int ip, Array<int> directions, int nlayers) const
// // {
// //    // int l,k;
// //    int d = directions.Size();
// //    int directx = directions[0]; // 1,0,-1
// //    int directy = directions[1]; // 1,0,-1
// //    int directz;
// //    if (d ==3) directz = directions[2];

// //    Mesh * mesh = ovlp_prob->fespaces[ip]->GetMesh();
// //    double h = GetUniformMeshElementSize(mesh);
   
// //    Vector pmin, pmax;
// //    mesh->GetBoundingBox(pmin, pmax);

// //    Array2D<double> pmlh(dim,2); pmlh = 0.0;
   
// //    if (directions[0]==1)
// //    {
// //       pmlh[0][1] = h*nlayers;
// //    }
// //    if (directions[0]==-1)
// //    {
// //       pmlh[0][0] = h*nlayers;
// //    }
// //    if (directions[1]==1)
// //    {
// //       pmlh[1][1] = h*nlayers;
// //    }
// //    if (directions[1]==-1)
// //    {
// //       pmlh[1][0] = h*nlayers;
// //    }

// //    CutOffFnCoefficient cf(CutOffFncn, pmin, pmax, pmlh);

// //    double * data = res.GetData();

// //    FiniteElementSpace * fespace;
// //    fespace = ovlp_prob->PmlFespaces[ip];
   
// //    int n = fespace->GetTrueVSize();

// //    GridFunction solgf_re(fespace, data);
// //    GridFunction solgf_im(fespace, &data[n]);

// //    GridFunctionCoefficient coeff1_re(&solgf_re);
// //    GridFunctionCoefficient coeff1_im(&solgf_im);

// //    ProductCoefficient prod_re(coeff1_re, cf);
// //    ProductCoefficient prod_im(coeff1_im, cf);

// //    ComplexGridFunction gf(fespace);
// //    gf.ProjectCoefficient(prod_re,prod_im);

// //    cfres.SetSize(res.Size());
// //    cfres = gf;
// // }

// // DiagST::~DiagST()
// // {
// //    for (int ip = 0; ip<nrpatch; ++ip)
// //    {
// //       delete PmlMatInv[ip];
// //       delete PmlMat[ip];
// //    }
// //    PmlMat.DeleteAll();
// //    PmlMatInv.DeleteAll();
// //    for (int ip=0; ip<nrpatch; ip++)
// //    {
// //       delete f_orig[ip];
// //       for (int i=0;i<nsweeps; i++)
// //       {
// //          delete f_transf[ip][i];
// //       }
// //    }
// // }

// // void DiagST::Getijk(int ip, int & i, int & j, int & k) const
// // {
// //    k = ip/(nxyz[0]*nxyz[1]);
// //    j = (ip-k*nxyz[0]*nxyz[1])/nxyz[0];
// //    i = (ip-k*nxyz[0]*nxyz[1])%nxyz[0];
// // }

// // int DiagST::GetPatchId(const Array<int> & ijk) const
// // {
// //    int d=ijk.Size();
// //    if (d==2)
// //    {
// //       return subdomains(ijk[0],ijk[1],0);
// //    }
// //    else
// //    {
// //       return subdomains(ijk[0],ijk[1],ijk[2]);
// //    }
// // }

// // int DiagST::SourceTransfer(const Vector & Psi0, Array<int> direction, int ip0, Vector & Psi1) const
// // {
// //    // For now 2D problems only
// //    // Directions
// //    // direction (1,1)
// //    int i0,j0,k0;
// //    Getijk(ip0,i0,j0,k0);

// //    int i1 = i0+direction[0];   
// //    int j1 = j0+direction[1];   
// //    Array<int> ij(2); ij[0]=i1; ij[1]=j1;
// //    int ip1 = GetPatchId(ij);

// //    MFEM_VERIFY(i1 < nxyz[0] && i1>=0, "SourceTransfer: i1 out of bounds");
// //    MFEM_VERIFY(j1 < nxyz[1] && j1>=0, "SourceTransfer: j1 out of bounds");

// //    Array<int> * Dof2GlobalDof0 = &ovlp_prob->Dof2GlobalDof[ip0];
// //    Array<int> * Dof2GlobalDof1 = &ovlp_prob->Dof2GlobalDof[ip1];
// //    Psi1.SetSize(Dof2GlobalDof1->Size()); Psi1=0.0;
// //    Vector r(2*bf->FESpace()->GetTrueVSize());
// //    r = 0.0;
// //    r.SetSubVector(*Dof2GlobalDof0,Psi0);

// //    // if (direction[0] ==  1 && direction[1] == 1)
// //    // {
// //    //    Vector zloc(Psi1.Size());
// //    //    r.GetSubVector(*Dof2GlobalDof1,zloc);
// //    //    // extend
// //    //    Vector psi_ext(PmlMat[ip1]->Height());
// //    //    Vector zloc_ext(PmlMat[ip1]->Height());
// //    //    Array<int> * Dof2PmlDof1 = &ovlp_prob->Dof2PmlDof[ip1];
// //    //    zloc_ext.SetSubVector(*Dof2PmlDof1,zloc);
// //    //    PmlMat[ip1]->Mult(zloc_ext,psi_ext);
// //    //    psi_ext *=-1.0;
// //    //    psi_ext.GetSubVector(*Dof2PmlDof1,Psi1);
// //    // // }
// //    // else
// //    // {
// //       r.GetSubVector(*Dof2GlobalDof1,Psi1);
// //    // }
// //    return ip1;
// // }

// // void DiagST::ConstructDirectionsMap()
// // {
// //    // total of 8 possible directions of transfer (2D)
// //    // form left            ( 1 ,  0)
// //    // form left-above      ( 1 , -1)
// //    // form left-below      ( 1 ,  1)
// //    // form right           (-1 ,  0)
// //    // form right-below     (-1 ,  1)
// //    // form right-above     (-1 , -1)
// //    // form above           ( 0 , -1)
// //    // form below           ( 0 ,  1)
// //    ntransf_directions = pow(3,dim);

// //    dirx.SetSize(ntransf_directions);
// //    diry.SetSize(ntransf_directions);
// //    int n=3;
// //    Array<int> ijk(dim);
// //    if (dim==2)
// //    {
// //       for (int i=-1; i<=1; i++) // directions x
// //       {
// //          for (int j=-1; j<=1; j++) // directions y
// //          {
// //             ijk[0]=i;
// //             ijk[1]=j;
// //             int k=GetDirectionId(ijk);
// //             dirx[k]=i;
// //             diry[k]=j;
// //          }
// //       }
// //    }
// //    else if (dim==3)
// //    {
// //       dirz.SetSize(ntransf_directions);
// //       for (int i=-1; i<=1; i++) // directions x
// //       {
// //          for (int j=-1; j<=1; j++) // directions y
// //          {
// //             for (int k=-1; k<=1; k++) // directions zß
// //             {
// //                ijk[0]=i;
// //                ijk[1]=j;
// //                ijk[2]=k;
// //                int l=GetDirectionId(ijk);
// //                dirx[l]=i;
// //                diry[l]=j;
// //                dirz[l]=k;
// //             }
// //          }
// //       }
// //    }

// //    // cout << "dirx = " << endl;
// //    // dirx.Print(cout,ntransf_directions);
// //    // cout << "diry = " << endl;
// //    // diry.Print(cout,ntransf_directions);

// //    // if (dim==2)
// //    // {
// //    //    for (int id=0; id<9; id++)
// //    //    {
// //    //       GetDirectionijk(id,ijk);
// //    //       // cout << "for id = " << id << ": (" <<ijk[0] << ", " << ijk[1] << ")" << endl;
// //    //    }
// //    // }
// //    // else
// //    // {
// //    //    cout << "dirz = " << endl;
// //    //    dirz.Print(cout,ntransf_directions);
// //    //    for (int id=0; id<27; id++)
// //    //    {
// //    //       GetDirectionijk(id,ijk);
// //    //       // cout << "for id = " << id << ": (" <<ijk[0] << ", " <<ijk[1] << ", " << ijk[2] << ")" << endl;
// //    //    }
// //    // }
// // }

// // int DiagST::GetDirectionId(const Array<int> & ijk) const
// // {
// //    int d = ijk.Size();
// //    int n=3;
// //    if (d==2)
// //    {
// //       return (ijk[0]+1)*n+(ijk[1]+1);
// //    }
// //    else
// //    {
// //       return (ijk[0]+1)*n*n+(ijk[1]+1)*n+ijk[2]+1;
// //    }
// // }

// // void DiagST::GetDirectionijk(int id, Array<int> & ijk) const
// // {
// //    int d = ijk.Size();
// //    int n=3;
// //    if (d==2)
// //    {
// //       ijk[0]=id/n - 1;
// //       ijk[1]=id%n - 1;
// //    }
// //    else
// //    {
// //       ijk[0]=id/(n*n)-1;
// //       ijk[1]=(id-(ijk[0]+1)*n*n)/n - 1;
// //       ijk[2]=(id-(ijk[0]+1)*n*n)%n - 1;
// //    }
// //    // cout << "ijk = " ; ijk.Print();
// // }




// // void DiagST::TransferSources(int sweep, int ip0, Vector & sol_ext) const
// // {
// //    // Find all neighbors of patch ip
// //    int nx = nxyz[0];
// //    int ny = nxyz[1];
// //    int i0, j0, k0;
// //    Getijk(ip0, i0,j0,k0);
// //    // cout << "Transfer to : " << endl;
// //    // loop through possible directions
// //    for (int i=-1; i<2; i++)
// //    {
// //       int i1 = i0 + i;
// //       if (i1 <0 || i1>=nx) continue;
// //       for (int j=-1; j<2; j++)
// //       {
// //          if (i==0 && j==0) continue;
// //          int j1 = j0 + j;
// //          if (j1 <0 || j1>=ny) continue;
// //          // cout << "(" << i1 << "," << j1 <<"), ";
// //          // Find ip 1
// //          Array<int> ij1(2); ij1[0] = i1; ij1[1]=j1;
// //          int ip1 = GetPatchId(ij1);
// //          // cout << "ip1 = " << ip1;
// //          // cout << " in the direction of (" << i <<", " <<j <<")" << endl;
// //          Array<int> directions(2);
// //          directions[0] = i;
// //          directions[1] = j;
// //          Vector cfsol_ext;
// //          Vector res_ext(sol_ext.Size());
// //          GetCutOffSolution(sol_ext,cfsol_ext,ip0,directions,true);
// //          // sol_ext = cfsol_ext;
// //          // Calculate source to be transfered


         


// //          PmlMat[ip0]->Mult(cfsol_ext, res_ext); 
         
// //          //---------------------------------------
// //          // FiniteElementSpace * fes = ovlp_prob->PmlFespaces[ip0];
// //          // Mesh * mesh = fes->GetMesh();
// //          // GridFunction gf(fes);
// //          // double * data = res_ext.GetData();
// //          // // gf.SetData(&data[fespace->GetTrueVSize()]);
// //          // gf.SetData(data);
// //          // char vishost[] = "localhost";
// //          // int  visport   = 19916;
// //          // socketstream pmlsock(vishost, visport);

// //          // string keys;
// //          // keys = "keys mrRljc\n";
// //          // pmlsock << "solution\n" << *mesh << gf << keys << "valuerange -0.1 0.1 \n"  << flush;
// //          // // sol_sock << "solution\n" << *mesh << gf << keys << flush;
// //          // cin.get();
// //          //---------------------------------------
         
         
         
// //          res_ext*= -1.0;
// //          Array<int> *Dof2PmlDof = &ovlp_prob->Dof2PmlDof[ip0];
// //          Vector res_local(Dof2PmlDof->Size()); res_local = 0.0;
// //          res_ext.GetSubVector(*Dof2PmlDof,res_local);


// //          // Vector sol_local(Dof2PmlDof->Size()); sol_local = 0.0;
// //          // cfsol_ext.GetSubVector(*Dof2PmlDof,sol_local);
// //          // Vector znew(A->Height()); znew = 0.0;
// //          // Vector rnew(A->Height()); rnew = 0.0;
// //          // Array<int> *Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip0];
// //          // znew.SetSubVector(*Dof2GlobalDof,sol_local);
// //          // A->Mult(znew,rnew); rnew *=-1.0;
// //          // rnew.GetSubVector(*Dof2GlobalDof, res_local);

// //          //-----------------------------
// //          // pass to ip1 and calculate residual there
// //          // Vector sol_local(Dof2PmlDof->Size()); sol_local = 0.0;
// //          // cfsol_ext.GetSubVector(*Dof2PmlDof,sol_local);
// //          //-----------------------------





// //          // Find the minumum sweep number that to transfer the source that 
// //          // satisfies the two rules
// //          for (int l=sweep; l<nsweeps; l++)
// //          {
// //             // Conditions on sweeps
// //             // Rule 1: the transfer source direction has to be similar with 
// //             // the sweep direction
// //             int is = sweeps(l,0); 
// //             int js = sweeps(l,1);
// //             int ddot = is*i + js * j;
// //             // cout << "(i,j) = (" << i <<"," <<j <<")" << endl;
// //             // cout << "(is,js) = (" << is <<"," <<js <<")" << endl;
// //             // cout << "ip0 , ip1 = " << ip0 << ", " << ip1 << endl;
// //             if (ddot <= 0) continue;

// //             // Rule 2: The horizontal or vertical transfer source cannot be used
// //             // in a later sweep that with opposite directions

// //             if (i==0 || j == 0) // Case of horizontal or vertical transfer source
// //             {
// //                int il = sweeps(l,0);
// //                int jl = sweeps(l,1);
// //                // skip if the two sweeps have opposite direction
// //                if (is == -il && js == -jl) continue;
// //             }
// //             // cout << "Passing ip0 = " << ip0 << " to ip1 = " << ip1 
// //                //   << " to sweep no l = " << l << endl;  
// //             Vector raux;
// //             // if (directions[0] ==  1 && directions[1] == 1)
// //             // {
// //                // res_local = sol_local;
// //             // }
// //             // cout << res_local.Norml2() << endl;

// //             int jp1 = SourceTransfer(res_local,directions,ip0,raux);
// //             MFEM_VERIFY(ip1 == jp1, "Check SourceTransfer patch id");
// //             MFEM_VERIFY(f_transf[ip1][l]->Size()==raux.Size(), 
// //                         "Transfer Sources: inconsistent size");
// //             *f_transf[ip1][l]+=raux;


// //          //    FiniteElementSpace * fes1 = ovlp_prob->fespaces[ip1];
// //          //    Mesh * mesh1 = fes1->GetMesh();
// //          // GridFunction gf1(fes1);
// //          // double * data1 = raux.GetData();
// //          // gf1.SetData(data1);
// //          // socketstream sock(vishost, visport);
// //          // sock << "solution\n" << *mesh1 << gf1 << keys << "valuerange -0.1 0.1 \n"  << flush;
// //          // // sol_sock << "solution\n" << *mesh << gf << keys << flush;
// //          // cin.get();
// //          break;
// //          }
            
// //       }  
// //    }
// // }


