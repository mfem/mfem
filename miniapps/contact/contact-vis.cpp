//                               contact-visualization test

// srun -n 8 ./contact-vis -testno 6 -paraview -sr 0 -pr 0 -tr 1.0 -sn 1.0
// srun -n 8 ./contact-vis -testno 6 -paraview -sr 1 -pr 0 -tr 16.0 -sn 1.0
// srun -n 8 ./contact-vis -testno 6 -paraview -sr 2 -pr 0 -tr 4.0 -sn 1.0
// srun -n 8 ./contact-vis -testno 6 -paraview -sr 3 -pr 0 -tr 4.0 -sn 1.0



#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "axom/slic.hpp"

#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"

using namespace std;
using namespace mfem;

void GetEssentialTdofs(const ParFiniteElementSpace & fes, const Array<int> & ess_bdr_attr, 
                       const Array<int> & ess_bdr_attr_comp, Array<int> & ess_tdof_list);
HypreParMatrix * SetupTribol(ParMesh * pmesh, ParGridFunction * coords, 
                             const Array<int> & ess_tdofs, const std::set<int> & mortar_attrs, 
                             const std::set<int> & non_mortar_attrs,
                             Vector &gap,  double ratio, int tribol_nprocs);

HypreParMatrix * GetContactProlongation(ParFiniteElementSpace * fes, 
                                        HypreParMatrix *J, 
                                        Array<int> ess_tdof_list);

int main(int argc, char *argv[])
{
    Mpi::Init();
    int myid = Mpi::WorldRank();
    int num_procs = Mpi::WorldSize();
    Hypre::Init();

    int sref = 1;
    int pref = 0;
    bool visualization = true;
    bool paraview = false;
    bool visit = false;
    int testNo = -1; // 0-6
    double tribol_ratio = 8.0;
    double separation = 0.1;
    bool disable_essbdr = false;
    double scale_nodes = 1.0;
   int tribol_nprocs = num_procs;

    // 1. Parse command-line options.
    OptionsParser args(argc, argv);
    args.AddOption(&testNo, "-testno", "--test-number",
                   "Choice of test problem:"
                   "-1: default (original 2 block problem)"
                   "0: not implemented yet"
                   "1: not implemented yet"
                   "2: not implemented yet"
                   "3: not implemented yet"
                   "4: two block problem - diablo"
                   "41: two block problem - twisted"
                   "5: ironing problem"
                   "51: ironing problem extended"
                   "6: nested spheres problem");
    args.AddOption(&sref, "-sr", "--serial-refinements",
                   "Number of uniform refinements.");                  
    args.AddOption(&pref, "-pr", "--parallel-refinements",
                   "Number of uniform refinements.");
    args.AddOption(&separation, "-s", "--mesh-separation",
                    "Mesh separation distance.");                   
    args.AddOption(&tribol_ratio, "-tr", "--tribol-proximity-parameter",
                   "Tribol-proximity-parameter.");
    args.AddOption(&scale_nodes, "-sn", "--scale-nodes",
                   "Scale the nodes of the mesh.");                     
    args.AddOption(&disable_essbdr, "-noessbdr", "--no-ess-bdr", "-ess-bdr",
                   "--ess-bdr",
                   "Enable or disable essential boundary.");                  
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                   "--no-paraview",
                   "Enable or disable ParaView visualization.");
    args.AddOption(&visit, "-visit", "--visit", "-no-visit",
                    "--no-visit",
                    "Enable or disable VisIT visualization.");     
    args.AddOption(&tribol_nprocs, "-tn", "--tribol-nprocs",
                        "Number of ranks used in tribol redecomposition" );                                  
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0)
        {
            args.PrintUsage(cout);
        }
        return 1;
    }
    if (myid == 0)
    {
        args.PrintOptions(cout);
    }

    if (Mpi::Root())
    {
        mfem::out << "Visualizing test problem number: " << testNo << endl;
    }

    const char *mesh_file = nullptr;

    switch (testNo)
    {
        case 4:
            mesh_file = "meshes/Test4.mesh";
            break;
        case 40:
            mesh_file = "meshes/Test40.mesh";
            break;   
        case 5:
            mesh_file = "meshes/Test5.mesh";
            break;
        case 51:
            mesh_file = "meshes/Test51.mesh";
            break;
        case 6:
            mesh_file = "meshes/Test6mod2.mesh";
            break;
        case -1:
            mesh_file = "meshes/two-block2.mesh";
            break;    
        default:
            MFEM_ABORT("Should be unreachable");
            break;
    }

    Mesh mesh(mesh_file,1);

    for (int i = 0; i<sref; i++)
    {
        mesh.UniformRefinement();
    }

    // move nodes of mesh elements with attr = 2
    // mesh.EnsureNodes();
    // GridFunction * nodes = mesh.GetNodes();
    // Array<int> nodes_marker(nodes->Size());
    // nodes_marker = 0;
    // const FiniteElementSpace * meshfes = mesh.GetNodalFESpace();
    // Array<int> vdofs;
    // for (int i=0; i<mesh.GetNE(); i++)
    // {
    //     int attr = mesh.GetAttribute(i);
    //     if (attr == 2)
    //     {
    //         // get element size
    //         double h = mesh.GetElementSize(i,0);
    //         mfem::out << "attr 2 h = " << h << endl;
    //         meshfes->GetElementVDofs(i,vdofs);
    //         for (int j = 0; j<vdofs.Size()/3; j++)
    //         {

    //             int xdof = vdofs[j];
    //             if (!nodes_marker[xdof])
    //             {
    //                 (*nodes)[xdof] += separation;
    //                 nodes_marker[xdof] = 1;    
    //             }
    //         }
    //     }
    //     else
    //     {
    //         double h = mesh.GetElementSize(i,0);
    //         mfem::out << "attr 1 h = " << h << endl;
    //     }
    // }


    ParMesh pmesh(MPI_COMM_WORLD,mesh);
    mesh.Clear();
    for (int i = 0; i<pref; i++)
    {
        pmesh.UniformRefinement();
    }

    int dim = pmesh.Dimension();

    std::set<int> mortar_attr;
    std::set<int> nonmortar_attr;
    Array<int> ess_bdr_attr;
    Array<int> ess_bdr_attr_comp;

    // count faces of 9 and 8

    if (testNo == 6)
    {
        mortar_attr.insert(6);
        mortar_attr.insert(9);
        nonmortar_attr.insert(7);
        nonmortar_attr.insert(8);
        
        // mortar_attr.insert(8);
        // nonmortar_attr.insert(9);


        ess_bdr_attr.Append(1); ess_bdr_attr_comp.Append(1);
        ess_bdr_attr.Append(2); ess_bdr_attr_comp.Append(2);
        ess_bdr_attr.Append(4); ess_bdr_attr_comp.Append(0);
        ess_bdr_attr.Append(5); ess_bdr_attr_comp.Append(-1);
    }
    else
    {
        mortar_attr.insert(3);
        nonmortar_attr.insert(4);
        ess_bdr_attr.Append(2); ess_bdr_attr_comp.Append(-1);
        ess_bdr_attr.Append(6); ess_bdr_attr_comp.Append(-1);
    }

    H1_FECollection fec(1,dim);
    ParFiniteElementSpace fes(&pmesh,&fec,dim,Ordering::byVDIM);   
    pmesh.SetNodalFESpace(&fes);

    GridFunction * pnodes = pmesh.GetNodes();
    for (int i = 0; i < pnodes->Size(); i++)
    {
        (*pnodes)(i) *= scale_nodes;
    }

    int gndofs = fes.GlobalTrueVSize();
    if (myid == 0)
    {
        mfem::out << "--------------------------------------" << endl;
        mfem::out << "Global number of dofs = " << gndofs << endl;
        mfem::out << "--------------------------------------" << endl;
    }
    ParGridFunction contact_gf(&fes); contact_gf = 0.0;
    Vector contact_tdofs(fes.GetTrueVSize()); contact_tdofs = 0.0;
    ParaViewDataCollection * paraview_dc = nullptr;
    VisItDataCollection * visit_dc = nullptr;

    // Get the essential true dofs
    Array<int> ess_tdof_list;
    if (!disable_essbdr)
    {
        GetEssentialTdofs(fes, ess_bdr_attr, ess_bdr_attr_comp, ess_tdof_list);
    }
    // Set up the Tribol contact problem
    ParGridFunction ref_coords(&fes);
    pmesh.GetNodes(ref_coords);
    Vector gap;
    HypreParMatrix * J = SetupTribol(&pmesh, &ref_coords, ess_tdof_list, mortar_attr, nonmortar_attr, gap, tribol_ratio, tribol_nprocs);

    // Get the contact prolongation operator
    HypreParMatrix * Pc = GetContactProlongation(&fes, J, ess_tdof_list);

    if (paraview)
    {
        std::ostringstream paraview_file_name;
        paraview_file_name << "ContactTribolVis_TestNo_" << testNo
                           << "_par_ref_" << pref
                           << "_ser_ref_" << sref
                           << "_tribol-scale_"  << tribol_ratio
                           << "_mesh_scale_" << scale_nodes;
        paraview_dc = new ParaViewDataCollection(paraview_file_name.str(), &pmesh);
        paraview_dc->SetPrefixPath("ParaView");
        paraview_dc->SetLevelsOfDetail(1);
        paraview_dc->SetDataFormat(VTKFormat::BINARY32);
        paraview_dc->SetHighOrderOutput(true);
        paraview_dc->RegisterField("u_c", &contact_gf);
        paraview_dc->SetCycle(0);
        paraview_dc->SetTime(0.0);
        paraview_dc->Save();
    }
    if (visit)
    {
        std::ostringstream visit_file_name;
        visit_file_name << "ContactTribolVis_TestNo_" << testNo
                           << "_par_ref_" << pref
                           << "_ser_ref_" << sref
                           << "_tribol-scale_"  << tribol_ratio
                           << "_mesh_scale_" << scale_nodes;
        visit_dc = new VisItDataCollection(visit_file_name.str(), &pmesh);
        visit_dc->SetPrefixPath("VisIT");
        visit_dc->RegisterField("u_c", &contact_gf);
        visit_dc->SetCycle(0);
        visit_dc->SetTime(0.0);
        visit_dc->Save();
    }



    socketstream sol_sock;
    if (visualization)
    {
        char vishost[] = "localhost";
        int visport = 19916;
        sol_sock.open(vishost, visport);
        sol_sock.precision(8);
    }
   
    int gncols = Pc->GetGlobalNumCols();
    int gnrows = Pc->GetGlobalNumRows();

    if (Mpi::Root())
    {
        mfem::out << "--------------------------------------" << endl;
        mfem::out << "Global number of contact dofs   = " << gncols << endl;
        mfem::out << "Global number of contact rows   = " << gnrows << endl;
        mfem::out << "--------------------------------------" << endl;
    }

    contact_tdofs = 0.0;
    Vector Ptc(Pc->Width()); Ptc = 1.0;
    Pc->Mult(Ptc, contact_tdofs);
    contact_gf.SetFromTrueDofs(contact_tdofs);

    if (visualization)
    {
        sol_sock << "parallel " << num_procs << " " << myid << "\n"
                 << "solution\n" << pmesh << contact_gf << flush;
    }

    if (paraview)
    {
        paraview_dc->SetCycle(0);
        paraview_dc->SetTime(0.0);
        paraview_dc->Save();
    }
    if (visit)
    {
        visit_dc->SetCycle(0);
        visit_dc->SetTime(0.0);
        visit_dc->Save();
    }

    if (paraview_dc) 
    {
        delete paraview_dc;
    }
    if (visit_dc) 
    {
        delete visit_dc;
    }
    return 0;
}

void GetEssentialTdofs(const ParFiniteElementSpace & fes, const Array<int> & ess_bdr_attr, 
                       const Array<int> & ess_bdr_attr_comp, Array<int> & ess_tdof_list)
{
    ess_tdof_list.SetSize(0);
    ParMesh * pmesh = fes.GetParMesh();
    Array<int> ess_bdr;
    if (pmesh->bdr_attributes.Size())
    {
        ess_bdr.SetSize(pmesh->bdr_attributes.Max());
    }
    ess_bdr = 0; 
    Array<int> ess_tdof_list_temp;
    for (int i = 0; i < ess_bdr_attr.Size(); i++ )
    {
        ess_bdr[ess_bdr_attr[i]-1] = 1;
        fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list_temp, ess_bdr_attr_comp[i]);
        ess_tdof_list.Append(ess_tdof_list_temp);
        ess_bdr[ess_bdr_attr[i]-1] = 0;
    }
}

HypreParMatrix * SetupTribol(ParMesh * pmesh, ParGridFunction * coords, 
                             const Array<int> & ess_tdofs, const std::set<int> & mortar_attrs, 
                             const std::set<int> & non_mortar_attrs,
                             Vector &gap,  double ratio, int tribol_nprocs)
{
    axom::slic::SimpleLogger logger;
    axom::slic::setIsRoot(mfem::Mpi::Root());

    int coupling_scheme_id = 0;
    int mesh1_id = 0; int mesh2_id = 1;

    tribol::registerMfemCouplingScheme(
        coupling_scheme_id, mesh1_id, mesh2_id,
        *pmesh, *coords, mortar_attrs, non_mortar_attrs,
        tribol::SURFACE_TO_SURFACE,
        tribol::NO_SLIDING,
        tribol::SINGLE_MORTAR,
        tribol::FRICTIONLESS,
        tribol::LAGRANGE_MULTIPLIER,
        tribol::BINNING_GRID
    );

    tribol::setBinningProximityScale(coupling_scheme_id, ratio);
    tribol::CouplingSchemeManager::getInstance().findData( coupling_scheme_id )->getParameters().gap_separation_ratio = ratio;
    // tribol::setMfemLORFactor(coupling_scheme_id, 8);
    // tribol::setContactAreaFrac( coupling_scheme_id, 1e-8);

    // Access Tribol's pressure grid function (on the contact surface)
    auto& pressure = tribol::getMfemPressure(coupling_scheme_id);
    int vsize = pressure.ParFESpace()->GlobalTrueVSize();
    if (mfem::Mpi::Root())
    {
        std::cout << "Number of pressure unknowns: " <<
        vsize << std::endl;
    }

    // Set Tribol options for Lagrange multiplier enforcement
    tribol::setLagrangeMultiplierOptions(
        coupling_scheme_id,
        tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN
    );

    // Update contact mesh decomposition
    tribol::updateMfemParallelDecomposition(tribol_nprocs);

    // Update contact gaps, forces, and tangent stiffness
    int cycle = 1;   // pseudo cycle
    double t = 1.0;  // pseudo time
    double dt = 1.0; // pseudo dt
    tribol::update(cycle, t, dt);

    // tribol::saveRedecompMesh(0);

    // Return contact contribution to the tangent stiffness matrix
    auto A_blk = tribol::getMfemBlockJacobian(coupling_scheme_id);



    HypreParMatrix * Mfull = (HypreParMatrix *)(&A_blk->GetBlock(1,0));
    
    // mfem::out << "Mfull size  = " << Mfull->GetGlobalNumRows() << " x " << Mfull->GetGlobalNumCols() << std::endl;
    
    HypreParMatrix * Me = Mfull->EliminateCols(ess_tdofs);
    delete Me;

    int h = Mfull->Height();
    SparseMatrix merged;
    Mfull->MergeDiagAndOffd(merged);
    Array<int> nonzero_rows;
    for (int i = 0; i<h; i++)
    {
        if (!merged.RowIsEmpty(i))
        {
            nonzero_rows.Append(i);
        }
    }

    int hnew = nonzero_rows.Size();


    SparseMatrix P(hnew,h);

    for (int i = 0; i<hnew; i++)
    {
        int col = nonzero_rows[i];
        P.Set(i,col,1.0);
    }
    P.Finalize();

    SparseMatrix * reduced_merged = Mult(P,merged);

    int rows[2];
    int cols[2];
    cols[0] = Mfull->ColPart()[0];
    cols[1] = Mfull->ColPart()[1];
    int nrows = reduced_merged->Height();

    int row_offset;
    MPI_Scan(&nrows,&row_offset,1,MPI_INT,MPI_SUM,Mfull->GetComm());

    row_offset-=nrows;
    rows[0] = row_offset;
    rows[1] = row_offset+nrows;
    int glob_nrows;
    MPI_Allreduce(&nrows, &glob_nrows,1,MPI_INT,MPI_SUM,Mfull->GetComm());

    int glob_ncols = reduced_merged->Width();
    HypreParMatrix * M = new HypreParMatrix(Mfull->GetComm(), nrows, glob_nrows,
                                            glob_ncols, reduced_merged->GetI(), reduced_merged->GetJ(),
                                            reduced_merged->GetData(), rows,cols); 
    delete reduced_merged;                          

    Vector gap_full;
    tribol::getMfemGap(coupling_scheme_id, gap_full);

    // mfem::out << "gapsize = " << gap_full.Size() << endl;
    // mfem::out << "gap norm  = " << gap_full.Norml1() << endl;

    //  count zeros in gap
    int gap_nonzeros=0;
    for (int i = 0; i<gap_full.Size(); i++)
    {
        if (gap_full[i] > 1e-15)
        {
            gap_nonzeros++;
        }
    }
    // mfem::out << "gap_nonzeros = " << gap_nonzeros << endl;

    auto& P_submesh = *pressure.ParFESpace()->GetProlongationMatrix();
    Vector gap_true(P_submesh.Width());
    P_submesh.MultTranspose(gap_full,gap_true);
    gap.SetSize(nrows);

    for (int i = 0; i<nrows; i++) 
    {
        gap[i] = gap_true[nonzero_rows[i]];
    }
    tribol::finalize();
    return M;
}

HypreParMatrix * GetContactProlongation(ParFiniteElementSpace * fes, 
                                        HypreParMatrix *J, 
                                        Array<int> ess_tdof_list)
{
    HypreParMatrix * Jt = J->Transpose();
    Jt->EliminateRows(ess_tdof_list);
    int hJt = Jt->Height();
    SparseMatrix mergedJt;
    Jt->MergeDiagAndOffd(mergedJt);
    Array<int> nonzerorows;
    for (int i = 0; i<hJt; i++)
    {
        if (!mergedJt.RowIsEmpty(i))
        {
            nonzerorows.Append(i);
        }
    }
    int hc = nonzerorows.Size();
    SparseMatrix Pct(hc,fes->GlobalTrueVSize());

    // mfem::out << "number of nonzero cols (contact dofs) = " << hc << std::endl;

    for (int i = 0; i<hc; i++)
    {
        int col = nonzerorows[i]+fes->GetMyTDofOffset();
        Pct.Set(i,col,1.0);
    }
    Pct.Finalize();

    int rows_c[2];
    int cols_c[2];
    int nrows_c = Pct.Height();

    int row_offset_c;
    MPI_Scan(&nrows_c,&row_offset_c,1,MPI_INT,MPI_SUM,J->GetComm());

    row_offset_c-=nrows_c;
    rows_c[0] = row_offset_c;
    rows_c[1] = row_offset_c+nrows_c;
    for (int i = 0; i < 2; i++)
    {
        cols_c[i] = fes->GetTrueDofOffsets()[i];
    }
    int glob_nrows_c;
    int glob_ncols_c = fes->GlobalTrueVSize();
    MPI_Allreduce(&nrows_c, &glob_nrows_c,1,MPI_INT,MPI_SUM,J->GetComm());

    HypreParMatrix * P_ct = new HypreParMatrix(J->GetComm(), nrows_c, glob_nrows_c,
                                               glob_ncols_c, Pct.GetI(), Pct.GetJ(),
                                               Pct.GetData(), rows_c,cols_c); 
    HypreParMatrix * Pc = P_ct->Transpose();
    delete P_ct;                         
    return Pc;
}