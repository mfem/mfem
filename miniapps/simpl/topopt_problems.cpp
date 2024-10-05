#include "topopt_problems.hpp"

namespace mfem
{

void MarkBoundaries(Mesh &mesh, int attr,
                    std::function<bool(const Vector &x)> marker)
{
   const int dim = mesh.SpaceDimension();
   Vector center(mesh.SpaceDimension());
   Array<int> vertices;
   for (int i=0; i<mesh.GetNBE(); i++)
   {
      center = 0.0;
      mesh.GetBdrElement(i)->GetVertices(vertices);
      for (auto v:vertices)
      {
         real_t * coord = mesh.GetVertex(v);
         for (int d=0; d<dim; d++)
         {
            center[d] += coord[d];
         }
      }
      center *= 1.0 / vertices.Size();
      if (marker(center))
      {
         mesh.SetBdrAttribute(i, attr);
      }
   }
   mesh.SetAttributes();
}

void MarkElements(Mesh &mesh, int attr,
                  std::function<bool(const Vector &x)> marker)
{
   Vector center(mesh.SpaceDimension());
   for (int i=0; i<mesh.GetNE(); i++)
   {
      mesh.GetElementCenter(i, center);
      if (marker(center))
      {
         mesh.SetAttribute(i, attr);
      }
   }
   mesh.SetAttributes();
}


Mesh * GetTopoptMesh(TopoptProblem prob, std::stringstream &filename,
                     real_t &r_min, real_t &tot_vol, real_t &min_vol, real_t &max_vol,
                     real_t &E, real_t &nu,
                     Array2D<int> &ess_bdr_displacement, Array<int> &ess_bdr_filter,
                     int ser_ref_levels, int par_ref_levels)
{
   Mesh * mesh;
   tot_vol = 0.0;
   switch (prob)
   {

      case Cantilever2:
      {
         filename << "Cantilever2";
         if (r_min < 0) { r_min = 0.05; }
         if (E < 0) { E = 1.0; }
         if (nu < 0) { nu = 0.3; }
         mesh = new Mesh(Mesh::MakeCartesian2D(3, 1, Element::Type::QUADRILATERAL, false,
                                               3.0, 1.0));
         tot_vol = 0.0;
         for (int i=0; i<mesh->GetNE(); i++) { tot_vol += mesh->GetElementVolume(i); }
         if (min_vol < 0) { min_vol = 0.0; }
         if (max_vol < 0) { max_vol = tot_vol*0.5; }
         for (int i=0; i<ser_ref_levels; i++)
         {
            mesh->UniformRefinement();
         }
         if (par_ref_levels > -1)
         {
#ifdef MFEM_USE_MPI
            Mesh * ser_mesh = mesh;
            mesh = new ParMesh(MPI_COMM_WORLD, *ser_mesh);
            ser_mesh->Clear();
            delete ser_mesh;
            for (int i=0; i<par_ref_levels; i++)
            {
               mesh->UniformRefinement();
            }
#else
            MFEM_ABORT("MFEM is built without MPI but tried to use parallel refinement");
#endif
         }
         int num_bdr_attr = 4;
         ess_bdr_displacement.SetSize(3, 4);
         ess_bdr_displacement = 0;
         ess_bdr_displacement(0,3) = 1;

         ess_bdr_filter.SetSize(4);
         ess_bdr_filter = 0;
         break;
      }

      case MBB2:
      {
         filename << "MBB2";
         if (r_min < 0) { r_min = 0.05; }
         if (E < 0) { E = 1.0; }
         if (nu < 0) { nu = 0.3; }
         mesh = new Mesh(Mesh::MakeCartesian2D(3, 1, Element::Type::QUADRILATERAL, false,
                                               3.0, 1.0));
         tot_vol = 0.0;
         for (int i=0; i<mesh->GetNE(); i++) { tot_vol += mesh->GetElementVolume(i); }
         if (min_vol < 0) { min_vol = 0.0; }
         if (max_vol < 0) { max_vol = tot_vol*0.5; }
         for (int i=0; i<ser_ref_levels; i++)
         {
            mesh->UniformRefinement();
         }
         if (par_ref_levels > -1)
         {
#ifdef MFEM_USE_MPI
            Mesh * ser_mesh = mesh;
            mesh = new ParMesh(MPI_COMM_WORLD, *ser_mesh);
            ser_mesh->Clear();
            delete ser_mesh;
            for (int i=0; i<par_ref_levels; i++)
            {
               mesh->UniformRefinement();
            }
#else
            MFEM_ABORT("MFEM is built without MPI but tried to use parallel refinement");
#endif
         }
         int num_bdr_attr = 4;
         MarkBoundaries(*mesh, 5,
                        [](const Vector &x)
         {
            return x[0] > 3.0 - std::pow(2.0, -5.0) && x[1] < 1e-09;
         });
         num_bdr_attr++;
         ess_bdr_displacement.SetSize(3, num_bdr_attr);
         ess_bdr_displacement = 0;
         ess_bdr_displacement(1, 3) = 1; // left: x-fixed
         ess_bdr_displacement(2, 4) = 1; // right: y-fixed

         ess_bdr_filter.SetSize(num_bdr_attr);
         ess_bdr_filter = 0;
         break;
      }

      case Arch2: {MFEM_ABORT("Undefined yet"); break; }

      case Bridge2: {MFEM_ABORT("Undefined yet"); break; }

      case Cantilever3:
      {
         filename << "Cantilever3";
         if (r_min < 0) { r_min = 0.02; }
         if (E < 0) { E = 1.0; }
         if (nu < 0) { nu = 0.3; }
         mesh = new Mesh(Mesh::MakeCartesian3D(2, 1, 1, Element::Type::HEXAHEDRON,
                                               2.0, 1.0, 1.0));
         tot_vol = 0.0;
         for (int i=0; i<mesh->GetNE(); i++) { tot_vol += mesh->GetElementVolume(i); }
         if (min_vol < 0) { min_vol = 0.0; }
         if (max_vol < 0) { max_vol = tot_vol*0.12; }
         for (int i=0; i<ser_ref_levels; i++)
         {
            mesh->UniformRefinement();
         }
         if (par_ref_levels > -1)
         {
#ifdef MFEM_USE_MPI
            Mesh * ser_mesh = mesh;
            mesh = new ParMesh(MPI_COMM_WORLD, *ser_mesh);
            ser_mesh->Clear();
            delete ser_mesh;
            for (int i=0; i<par_ref_levels; i++)
            {
               mesh->UniformRefinement();
            }
#else
            MFEM_ABORT("MFEM is built without MPI but tried to use parallel refinement");
#endif
         }
         int num_bdr_attr = 6;
         ess_bdr_displacement.SetSize(4, num_bdr_attr);
         ess_bdr_displacement = 0;
         ess_bdr_displacement(0, 4) = 1; // left: x-fixed

         ess_bdr_filter.SetSize(num_bdr_attr);
         ess_bdr_filter = 0;
         ess_bdr_filter[0] = -1; // bottom: no material
         ess_bdr_filter[5] = -1; // top: no material
         break;
      }

      case MBB3: {MFEM_ABORT("Undefined yet"); break; }

      case Arch3: {MFEM_ABORT("Undefined yet"); break; }

      case Bridge3: {MFEM_ABORT("Undefined yet"); break; }

      case ForceInverter2: {MFEM_ABORT("Undefined yet"); break; }
   }
   return mesh;
}

void SetupTopoptProblem(TopoptProblem prob, ElasticityProblem &elasticity,
                        GridFunction &gf_filter, GridFunction &gf_state)
{

   switch (prob)
   {
      case Cantilever2:
      {
         auto load = new VectorFunctionCoefficient(
            2, [](const Vector &x, Vector &f)
         {
            f = 0.0;
            if (std::pow(x[0]-2.9, 2.0) + std::pow(x[1] - 0.5, 2.0) < 0.05*0.05)
            {
               f[1] = -1.0;
            }
         });
         elasticity.MakeCoefficientOwner(load);
         elasticity.GetLinearForm()->AddDomainIntegrator(
            new VectorDomainLFIntegrator(*load));
         break;
      }

      case MBB2:
      {
         auto load = new VectorFunctionCoefficient(
            2, [](const Vector &x, Vector &f)
         {

            f = 0.0;
            if (std::pow(x[0], 2.0) + std::pow(x[1] - 1.0, 2.0) < 0.05*0.05)
            {
               f[1]=-1.0;
            }
         });
         elasticity.MakeCoefficientOwner(load);
         elasticity.GetLinearForm()->AddDomainIntegrator(
            new VectorDomainLFIntegrator(*load));
         break;
      }

      case Arch2: {MFEM_ABORT("Undefined yet"); break; }

      case Bridge2: {MFEM_ABORT("Undefined yet"); break; }

      case Cantilever3:
      {
         auto load = new VectorFunctionCoefficient(
            3, [](const Vector &x, Vector &f)
         {
            f = 0.0;
            if (std::pow(x[0]-1.9, 2.0) + std::pow(x[2] - 0.1, 2.0) < 0.1*0.1)
            {
               f[2] = -1.0;
            }
         });
         elasticity.MakeCoefficientOwner(load);
         elasticity.GetLinearForm()->AddDomainIntegrator(
            new VectorDomainLFIntegrator(*load));
         break;
      }

      case MBB3: {MFEM_ABORT("Undefined yet"); break; }

      case Arch3: {MFEM_ABORT("Undefined yet"); break; }

      case Bridge3: {MFEM_ABORT("Undefined yet"); break; }

      case ForceInverter2: {MFEM_ABORT("Undefined yet"); break; }
   }
}
} // end of namespace
