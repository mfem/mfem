// Buffers to store all the necessary
// information to advance an element in time
struct EvalCache {
  // Buffers for the element action
  mfem::Vector phimelvals;
  mfem::Vector phimelvals2;
  mfem::Vector phimelvalsfacerhs;
  mfem::Vector phimelvals_rhs;
  mfem::Vector phimelvals_rhs2;
  mfem::Vector selvals;
  mfem::Vector selvals2;

  mfem::DenseMatrix a_el;
  mfem::DenseMatrix a2_el;

  mfem::DenseMatrix minv_el;
  mfem::DenseMatrix f_el;
  mfem::Array<int> e_faces;
  mfem::Array<int> f_ori;

  // Buffers for the face action
  mfem::Array<int> facedofblocks;
  mfem::BlockVector phimrhsfacevals; // Block vector for face action (element pairs)
  mfem::Vector phimrhsfacevals_e1view;
  mfem::Vector phimrhsfacevals_e2view;
  mfem::Vector phimrhsfacevals_view_prev; // Last time step value
  mfem::Vector phimrhsfacevals_view_pred; // Predictor value
  mfem::BlockVector phimrhsfacevals_buf;
  mfem::BlockVector phimrhsfacevals_buf2;

  // Dof buffers
  mfem::Array<int> phimdofs;
  mfem::Array<int> sdofs;
  mfem::Array<int> vdofs1, vdofs2;

  // ndofs = number of field dofs per element
  // idim  = dimension of the internal problem
  // nfaces = number of faces per elements
  EvalCache(int ndofs, int idim, int nfaces) {
    phimelvals        = mfem::Vector(ndofs);
    phimelvals2       = mfem::Vector(ndofs);
    phimelvalsfacerhs = mfem::Vector(ndofs);
    phimelvals_rhs    = mfem::Vector(ndofs);
    phimelvals_rhs2   = mfem::Vector(ndofs);

    e_faces = mfem::Array<int>(nfaces);
    f_ori   = mfem::Array<int>(nfaces);
    
    a_el  = mfem::DenseMatrix(ndofs);
    a2_el = mfem::DenseMatrix(ndofs);

    facedofblocks.Append(0);
    facedofblocks.Append(ndofs);
    facedofblocks.Append(2 * ndofs);

    phimrhsfacevals.Update(facedofblocks);
    phimrhsfacevals_buf.Update(facedofblocks);
    phimrhsfacevals_buf2.Update(facedofblocks);

    phimrhsfacevals_e1view    = mfem::Vector(ndofs);
    phimrhsfacevals_e2view    = mfem::Vector(ndofs);
    phimrhsfacevals_view_prev = mfem::Vector(ndofs);
    phimrhsfacevals_view_pred = mfem::Vector(ndofs);

    phimdofs   = mfem::Array<int>(ndofs);
    sdofs    = mfem::Array<int>(idim * ndofs);
    selvals  = mfem::Vector(idim * ndofs);
    selvals2 = mfem::Vector(idim * ndofs);
  }
};

// AMR update for fe spaces, gridfunctions and the forms
inline void Update(std::vector<mfem::FiniteElementSpace *> &fespaces, std::vector<mfem::GridFunction *> &xs,
            mfem::BilinearForm &a, mfem::BilinearForm &m) {
  MFEM_PERF_FUNCTION;

  // Update the space: recalculate the number of DOFs and construct a matrix
  // that will adjust any GridFunctions to the new mesh state.
  MFEM_PERF_BEGIN("FESpaces");
  for (auto fespace : fespaces) {
    fespace->Update();
  }
  MFEM_PERF_END("FESpaces");

  // Interpolate the solution on the new mesh by applying the transformation
  // matrix computed in the finite element space. Multiple GridFunctions could
  // be updated here.
  MFEM_PERF_BEGIN("GridFunctions");
  for (auto x : xs) {
    x->Update();
  }
  MFEM_PERF_END("GridFunctions");

  // Inform the linear and bilinear forms that the space has changed.
  MFEM_PERF_BEGIN("Forms");
  a.Update();
  m.Update();
  MFEM_PERF_END("Forms");

  // Free any transformation matrices to save memory.
  MFEM_PERF_BEGIN("FESpaces");
  for (auto fespace : fespaces) {
    fespace->UpdatesFinished();
  }
  MFEM_PERF_END("FESpaces");
}

// Could not find this function in MFEM
inline void GetFaceIndices(mfem::Array<int>&faces, mfem::Array<int>&orientations, mfem::Mesh* mesh, int ei) {
  if(mesh->Dimension() == 1) {
    mesh->GetElementVertices(ei, faces);
  } else if(mesh->Dimension() == 2) {
    mesh->GetElementEdges(ei, faces, orientations);
  } else {
    mesh->GetElementFaces(ei, faces, orientations);
  }
}


// Linearize vector dofs to simplify the iteration order
inline void GetElementVDofsLinear(const mfem::FiniteElementSpace &fespace, const int element, mfem::Array<int> &dofs)
{
  const int vdim = fespace.GetVDim();
  fespace.GetElementDofs(element, dofs);
  const int size = dofs.Size();
  dofs.SetSize(size*vdim);
  // Stride dofs first
  for(int node = size-1; node >= 0; node--)
  {
    dofs[node*vdim] = dofs[node];
  }
  for(int node = 0; node < size; node++)
  {
    for (int vd = 1; vd < vdim; vd++)
    {
      dofs[node*vdim+vd] = -1;
    }
  }
  // Fill gaps
  for (int node = 0; node < size; node++)
  {
    for (int vd = 1; vd < vdim; vd++)
    {
      dofs[node*vdim+vd] = fespace.DofToVDof(dofs[node*vdim], vd);
    }
    dofs[node*vdim] = fespace.DofToVDof(dofs[node*vdim], 0);
  }
}

// Helper for the initial condition
template<class GF>
void FibrillationInit(GF& phimgf, GF& sgf, int h_internal_offset) {
  auto phimfespace = phimgf.FESpace();
  auto sfespace  = sgf.FESpace();
  auto mesh = phimfespace->GetMesh();
  const int sdim = mesh->SpaceDimension();

  mfem::Vector mesh_min(sdim), mesh_max(sdim);
//   auto pmesh = dynamic_cast<mfem::ParMesh*>(mesh);
//   if(pmesh)
//   {
//     pmesh->GetBoundingBox(mesh_min, mesh_max, 0);
//   }
//   else 
  {
    mesh->GetBoundingBox(mesh_min, mesh_max, 0);
  }

  int el = -1;
  mfem::ElementTransformation *T = NULL;
  const mfem::FiniteElement *fe = NULL;

  mfem::Vector pos(sdim);
  phimfespace->BuildDofToArrays();
  for (int dof = 0; dof < phimfespace->GetNDofs(); dof++)
  {
      int j = phimfespace->GetElementForDof(dof);
      if (el != j)
      {
          el = j;
          T = phimfespace->GetElementTransformation(el);
          fe = phimfespace->GetFE(el);
      }
      int ld = phimfespace->GetLocalDofForDof(dof);
      const mfem::IntegrationPoint &ip = fe->GetNodes().IntPoint(ld);
      T->SetIntPoint(&ip);
      T->Transform(ip, pos);

      const double width = mesh_max(0)-mesh_min(0);
      const double offset = (pos(0)-mesh_min(0))/width;
      phimgf(dof) = (1.0-offset);
  }

  if(sdim == 1) return;

  sfespace->BuildDofToArrays();
  for (int dof = 0; dof < sfespace->GetNDofs(); dof++)
  {
      int j = sfespace->GetElementForDof(dof);
      if (el != j)
      {
          el = j;
          T = sfespace->GetElementTransformation(el);
          fe = sfespace->GetFE(el);
      }
      int vdof = sfespace->DofToVDof(dof, h_internal_offset);
      int ld = sfespace->GetLocalDofForDof(dof);
      const mfem::IntegrationPoint &ip = fe->GetNodes().IntPoint(ld);
      T->SetIntPoint(&ip);
      T->Transform(ip, pos);

      const double width = mesh_max(1)-mesh_min(1);
      const double offset = (pos(1)-mesh_min(1))/width;
      sgf(vdof) = 0.1*offset + (1.0-offset)*0.6;
  }
}

// Compute the element Jacobian for an SIPG diffusion element in ODE form
inline void ComputeElementDiffusion(mfem::DenseMatrix& element_jac, mfem::DenseMatrix& element_jac_massinv, mfem::Mesh* mesh, int ei, const mfem::NCMesh::NCList& ncfacelist, mfem::BilinearForm& minv, mfem::BilinearForm& a, const std::vector<mfem::DenseMatrix>& face_matrices, EvalCache& ec)
{
  GetFaceIndices(ec.e_faces, ec.f_ori, mesh, ei);

  element_jac = 0.0;
  a.ComputeElementMatrix(ei, element_jac);

  for (int fi : ec.e_faces) {
    int fe1, fe2;
    mesh->GetFaceElements(fi,&fe1,&fe2);
    DEBUG_PRINT("    Visiting face " << fi << " with elements " << fe1 << " " << fe2);
    if (fe2 >= 0) { // Not on boundary
      MFEM_PERF_FINE_BEGIN("EvaluateFaceCall");
      //AddFace
      const auto& face_mat = face_matrices[fi];
      const auto ndofs = ec.facedofblocks[1];
      if(fe1 == ei) {
        for(int i=0;i<ndofs;i++) {
            for(int j=0;j<ndofs;j++) {
                element_jac(i,j) += face_mat(i,j);
            }
        }
      } else {
        for(int i=0;i<ndofs;i++) {
            for(int j=0;j<ndofs;j++) {
                element_jac(i,j) += face_mat(i+ndofs,j+ndofs);
            }
        }
      }
      MFEM_PERF_FINE_END("EvaluateFaceCall");
    } else {
      int Inf1, Inf2, NCFace;
      mesh->GetFaceInfos(fi, &Inf1, &Inf2, &NCFace);
      if(NCFace < 0) {
        // No-flux
        // const auto& face_mat = face_matrices[fi];
        // const auto ndofs = ec.facedofblocks[1];
        // for(int i=0;i<ndofs;i++) {
        //     for(int j=0;j<ndofs;j++) {
        //         element_jac(i,j) += face_mat(i,j);
        //     }
        // }
        continue; // Not a master face (i.e. a boundary face)
      }
      // MFEM_PERF_FINE_BEGIN("NCInfo");
      DEBUG_PRINT("      NCFace=" << NCFace);
      auto& masterinfo = ncfacelist.masters[NCFace];
      DEBUG_PRINT("      master index=" << masterinfo.index << " local=" << int(masterinfo.local));
      DEBUG_PRINT("      slave range=" << masterinfo.slaves_begin << ":" << masterinfo.slaves_end);
      // MFEM_PERF_FINE_END("NCInfo");
      for(int slave = masterinfo.slaves_begin; slave < masterinfo.slaves_end; slave++) {
        auto& slaveinfo = ncfacelist.slaves[slave];
        DEBUG_PRINT("        slave index=" << slaveinfo.index << " local=" << int(slaveinfo.local));
        if(slaveinfo.index < 0) { // Degenerate face-edge constraint
          continue;
        }
        mesh->GetFaceElements(slaveinfo.index, &fe1, &fe2);
        const auto& face_mat = face_matrices[slaveinfo.index];
        const auto ndofs1 = ec.facedofblocks[1];
        const auto ndofs2 = ec.facedofblocks[2]-ec.facedofblocks[1];
        MFEM_PERF_FINE_BEGIN("EvaluateFaceCall2");
        if(ei == fe1) {
          for(int i=0;i<ndofs1;i++) {
            for(int j=0;j<ndofs1; j++) {
              element_jac(i,j) += face_mat(i,j);
            }
          }
        } else if(ei == fe2) {
          for(int i=0;i<ndofs2;i++) {
            for(int j=0;j<ndofs2; j++) {
              element_jac(i,j) += face_mat(i+ndofs1,j+ndofs1);
            }
          }
        } else {
          std::cout << "Face-element table corrupted? (Case 1) ei=" << ei << " fi=" << fi << std::endl;
          std::exit(-1);
        }
        MFEM_PERF_FINE_END("EvaluateFaceCall2");
      }
    }
  }

  minv.ComputeElementMatrix(ei, ec.minv_el); // M_e^-1
  mfem::Mult(ec.minv_el, element_jac, element_jac_massinv); // J_e = M_e^-1 * K_e
}

// Wrapper to make clear what we intend do to
inline void ComputeElementJacobian(mfem::DenseMatrix& element_jac, mfem::DenseMatrix& element_jac_massinv, mfem::Mesh* mesh, int ei, const mfem::NCMesh::NCList& ncfacelist, mfem::BilinearForm& minv, mfem::BilinearForm& a, std::vector<mfem::DenseMatrix>& face_matrices, std::shared_ptr<const AbstractCellSolver> stepper, EvalCache& ec)
{
  ComputeElementDiffusion(element_jac, element_jac_massinv, mesh, ei, ncfacelist, minv, a, face_matrices, ec);
}

// Dimension independent getter for faces
inline const mfem::NCMesh::NCList& GetNCFaceList(const mfem::Mesh* mesh) {
  static mfem::NCMesh::NCList emptylist;
  if(mesh->ncmesh == nullptr || mesh->Dimension() == 1) {
    return emptylist;
  }
  return mesh->Dimension() == 2 ? mesh->ncmesh->GetEdgeList() : mesh->ncmesh->GetFaceList();
}

// TODO try to integrate this with some kind of EABilinearForm
inline std::vector<mfem::DenseMatrix> ComputeFaceMatrices(mfem::Mesh* mesh, mfem::FiniteElementSpace* fespace, mfem::BilinearFormIntegrator* dginteg, mfem::BilinearFormIntegrator* dgbdrinteg)
{
    std::vector<mfem::DenseMatrix> face_matrices(mesh->GetNumFaces());
    mfem::DenseMatrix f_el;
    for (int fi = 0; fi < mesh->GetNumFaces(); fi++) {
        face_matrices[fi] = 0.0;
        int fe1, fe2;
        mesh->GetFaceElements(fi,&fe1,&fe2);
        if (fe2 < 0) { // Maybe on boundary
            // Logic taken from https://github.com/mfem/mfem/blob/541f10f9b44fd52bacde793a08dc19e53a8972ec/fem/bilinearform.cpp#L615-L628
            // const auto fbi = face_to_be[fi];
            // if(fbi < 0) { // Filter ncface
            //     continue;
            // }
            // auto tr = mesh->GetBdrFaceTransformations(fbi);
            // if (tr == nullptr) { 
            //     continue;
            // }
            // dgbdrinteg->AssembleFaceMatrix(*fespace->GetFE(tr->Elem1No),
            //                             *fespace->GetFE(tr->Elem1No), *tr, f_el);
            // face_matrices[fi] = f_el;
            continue;
        }
        auto tr = mesh->GetInteriorFaceTransformations(fi);
        if (tr == nullptr) {
            continue;
        }
        dginteg->AssembleFaceMatrix(*fespace->GetFE(tr->Elem1No),
                                    *fespace->GetFE(tr->Elem2No), *tr, f_el);
        face_matrices[fi] = f_el;
    }
    return face_matrices;
}

inline void EigenvaluesGershgorin(const mfem::DenseMatrix& A, mfem::Vector& evs)
{
  for(int row=0;row<A.Height();row++) {
    evs(row) = A(row,row);
    for(int col=0;col<row;col++) {
      evs(row) += std::abs(A(row,col));
    }
    for(int col=row+1;col<A.Width();col++) {
      evs(row) += std::abs(A(row,col));
    }
  }
}

struct Stimulus {
  double dist_max = 0.0;
  double t_max = 0.0;
  double stim_max = 0.0;
  double capacitance = 1.0;
};

void PrintBanner(std::ostream &out) {
  out << "+-----------------------------------------------------------------+"<< std::endl;
  out << "                                                                  "<< std::endl;
  out << "        __     __       _    _      ____       @@@@       @@@@    "<< std::endl;
  out << "       /..|   /..|     /./  /./   /. ___/    @@@@@@@@   @@@@@@@@  "<< std::endl;
  out << "      /. .|  /. .|    /./__/./   /. /__     @@@@@@@@@@@@@@@@@@@@@ "<< std::endl;
  out << "     /./|.| /./|.|   /.____./   /_... /     @@@@@@@@@@@@@@@@@@@@@ "<< std::endl;
  out << "    /./ |.|/./ |.|  /./  /./   ___/. /       @@@@@@@@@@@@@@@@@@@  "<< std::endl;
  out << "   /_/  |___/  |_| /_/  /_/   /_____/          @@@@@@@@@@@@@@@    "<< std::endl;
  out << "                                                 @@@@@@@@@@@      "<< std::endl;
  out << "       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                @@@@@         "<< std::endl;
  out << "       ~~~~ Spinner LTS Miniapp ~~~~                  @           "<< std::endl;
  out << "       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                              "<< std::endl;
  out << "                                                                  "<< std::endl;
  out << "+-----------------------------------------------------------------+"<< std::endl;
}
