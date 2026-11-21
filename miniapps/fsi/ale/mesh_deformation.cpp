#include "mesh_laplacian.hpp"

using namespace mfem;

MeshDeformation::MeshDeformation(ParMesh *fluid_mesh_)
    : fluid_mesh(fluid_mesh_)
{
}

void MeshDeformation::SetEssentialBoundary(Array<int> &ess_bdr_)
{
   // Set the essential boundary attributes
   ess_bdr = ess_bdr_;
}


MeshDeformation::~MeshDeformation()
{
   delete ref_nodes;
   //delete nodes;

   delete linear_solver;
   delete prec;

   if (periodic)
   {
      delete fes_mesh;
      delete fec;
   }

   delete local_stiffening_coeff;
   delete fes_stiffening;
}

void MeshDeformation::Initialize(ParMesh *mesh)
{
      // Initialize MPI variables
   MPI_Comm_size(mesh->GetComm(), &num_procs);
   MPI_Comm_rank(mesh->GetComm(), &my_id);

   // Check that mesh has nodes GridFunction
   MFEM_VERIFY(mesh->GetNodes() != nullptr, "MeshDeformation requires a mesh with nodes GridFunction."
                                                  "To ensure that the nodal GridFunction exists, first call Mesh::EnsureNodes().");

   // Extract the original finite element space for the mesh nodes
   original_fes_mesh = (ParFiniteElementSpace *)mesh->GetNodes()->FESpace();
   periodic = original_fes_mesh->IsDGSpace(); // Check if the mesh is periodic (DG space for nodes)

   // If mesh is periodic the  fespace for the nodes is DG, so GetBE() will be null.
   if (periodic)
   {
      const int sdim = mesh->SpaceDimension();
      int gf_order_h1 = std::max(original_fes_mesh->GetMaxElementOrder(), 1); // H1 should be at least order 1
      fec = new H1_FECollection(gf_order_h1, mesh->Dimension());
      const int ncomp = sdim;
      fes_mesh = new ParFiniteElementSpace(mesh, fec, ncomp, original_fes_mesh->GetOrdering()); // Ordering should be by VDIM, but check
   }
   else
   {
      fec = original_fes_mesh->FEColl();
      fes_mesh = original_fes_mesh;
   }

   //nodes = new ParGridFunction(original_fes_mesh); --> We just retrieve the current nodes from the mesh directly 
   ref_nodes = new ParGridFunction(original_fes_mesh);
   *ref_nodes = *mesh->GetNodes(); // same space: exact copy (no projection)

   X.SetSize(fes_mesh->GetTrueVSize());
   X = 0.0;
   B.SetSize(fes_mesh->GetTrueVSize());
   B = 0.0;
   b.SetSize(fes_mesh->GetVSize());
   b = 0.0;
}

void MeshDeformation::DeformMeshWithVelocity(ParGridFunction &mesh_velocity_gf, real_t dt, bool update_gf)
{
   MFEM_ASSERT(dt > 0.0, "Time step size must be positive for mesh deformation with velocity.");

   // Deform the mesh nodes based on the computed mesh velocity
   // This is done by adding the mesh velocity to the current mesh nodes
   ParGridFunction *mesh_velocity = nullptr;
   mesh_velocity_gf.SetTrueVector();
   mesh_velocity_gf.SetFromTrueVector();
   if (periodic)
   {
      mesh_velocity = new ParGridFunction(original_fes_mesh);
      mesh_velocity->ProjectGridFunction(mesh_velocity_gf);
   }
   else
   {
      mesh_velocity = &mesh_velocity_gf;
   }

   auto *nodes = fluid_mesh->GetNodes();
   nodes->Add(dt, *mesh_velocity);
   NodesUpdated();

   if (periodic)
      delete mesh_velocity;

   // Update the provided and internal gfs
   if (update_gf)
   {
      mesh_velocity_gf.Update();
   }
}


void MeshDeformation::DeformMeshWithDisplacement(ParGridFunction &displacement_gf, bool update_gf)
{
   // Deform the mesh nodes based on the computed mesh velocity
   // This is done by adding the mesh velocity to the current mesh nodes
   ParGridFunction *mesh_displacement = nullptr;
   displacement_gf.SetTrueVector();
   displacement_gf.SetFromTrueVector();
   if (periodic)
   {
      mesh_displacement = new ParGridFunction(original_fes_mesh);
      mesh_displacement->ProjectGridFunction(displacement_gf);
   }
   else
   {
      mesh_displacement = &displacement_gf;
   }

   // Deform the fluid mesh
   auto *nodes = fluid_mesh->GetNodes();
   *nodes = *ref_nodes;
   nodes->Add(1.0, *mesh_displacement);
   NodesUpdated();

   if (periodic)
      delete mesh_displacement;

   // Update the provided and internal gfs
   if (update_gf)
   {
      displacement_gf.Update();
   }
}

void MeshDeformation::EnableLocalStiffening(Array<int> &wall_attrs_, int stiffening_type_, int distance_solver_type, real_t stiffening_degree_, bool filter_ls)
{
   enable_local_stiffening = true;
   wall_attrs = wall_attrs_;
   stiffening_type = stiffening_type_;
   stiffening_degree = stiffening_degree_;

   // Release a previously created coefficient/space (derived classes re-assemble afterwards)
   delete local_stiffening_coeff;
   local_stiffening_coeff = nullptr;
   delete fes_stiffening;
   fes_stiffening = nullptr;

   // Create scalar finite element space for local stiffening on the mesh the operator is assembled on:
   // the reference mesh (non-incremental) or the fluid mesh (incremental)
   fes_stiffening = new ParFiniteElementSpace(fes_mesh->GetParMesh(), fec);

   switch (stiffening_type)
   {
   case 0: // Jacobian-Based
   {
      local_stiffening_coeff = new JacobianBasedStiffeningCoefficient(fes_stiffening, wall_attrs, stiffening_degree);
      break;
   }
   case 1: // Distance-Based
   {
      local_stiffening_coeff = new DistanceBasedStiffeningCoefficient(fes_stiffening, wall_attrs, distance_solver_type, filter_ls);
      // Transition zone in units of the average element size
      const real_t h = AvgElementSize(*fes_stiffening->GetParMesh());
      real_t Rmin = 2.0 * h; // Minimum distance to wall boundaries (start of transition zone)
      real_t Rmax = 6.0 * h; // Maximum distance to wall boundaries (end of transition zone)
      real_t alpha = 10.0;   // Non-dimensional sharpness of the transition
      real_t Amax = 1e2;     // Maximum stiffening coefficient
      real_t Amin = 1e0;     // Minimum stiffening coefficient
      static_cast<DistanceBasedStiffeningCoefficient *>(local_stiffening_coeff)->SetParameters(Rmin, Rmax, alpha, Amax, Amin, stiffening_degree);
      break;
   }
   default:
      mfem_error("Unknown stiffening type");
      break;
   }
}


void MeshDeformation::EnableLocalStiffening(int wall_attr_, int stiffening_type_, int distance_solver_type, real_t stiffening_degree_, bool filter_ls)
{
   Array<int> tmp_wall_attrs(fluid_mesh->bdr_attributes.Max());
   tmp_wall_attrs = 0;
   MFEM_VERIFY(wall_attr_ >= 1 && wall_attr_ <= tmp_wall_attrs.Size(), "Invalid wall attribute " << wall_attr_);
   tmp_wall_attrs[wall_attr_ - 1] = 1; // boundary attributes are 1-based
   EnableLocalStiffening(tmp_wall_attrs, stiffening_type_, distance_solver_type, stiffening_degree_, filter_ls);
}

void MeshDeformation::NodesUpdated()
{
   // Invalidate the geometric factors (the nodes were modified in place, no need for SetNodes self-assignment)
   fluid_mesh->NodesUpdated();
   // Refresh the face-neighbor node data if it was ever exchanged (have_face_nbr_data is uniform across ranks)
   if (fluid_mesh->have_face_nbr_data)
   {
      fluid_mesh->ExchangeFaceNbrNodes();
   }
}

void MeshDeformation::Update()
{
   // Update the local stiffening with new distance gf
   if (enable_local_stiffening)
   {
      local_stiffening_coeff->Update();
   }
}
