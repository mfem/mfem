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

real_t DistanceToSegment(const Vector &p, const Vector &v, const Vector &w)
{
   const double l2 = v.DistanceSquaredTo(w);
   const double t = std::max(0.0, std::min(1.0,
                                           ((p*w) - (p*v) - (v*w) + (v*v))/l2));
   Vector projection(v);
   projection.Add(t, w);
   projection.Add(-t, v);
   return p.DistanceTo(projection);
}

void ForceInverterInitialDesign(GridFunction &x, LegendreEntropy *entropy)
{
   Array<Vector*> ports(0);
   ports.Append(new Vector({0.0,1.0}));
   ports.Append(new Vector({0.0,0.0}));
   ports.Append(new Vector({2.0,1.0}));
   Vector domain_center({1.0,0.5});


   FunctionCoefficient dist([&domain_center, &ports](const Vector &x)
   {
      double d = infinity();
      for (auto &port : ports) {d = std::min(d, DistanceToSegment(x, domain_center, *port));}
      return d;
   });
   x.ProjectCoefficient(dist);
   ConstantCoefficient zero_cf(0.0);
   real_t max_d = x.ComputeMaxError(zero_cf);
   for (auto &val : x) { val = (max_d - val) / max_d; }
   if (entropy)
   {
      for (real_t &val : x)
      {
         val = entropy->forward(val);
      }
   }
   for (auto port:ports) { delete port; }
}

Mesh * GetTopoptMesh(TopoptProblem prob, std::stringstream &filename,
                     real_t &r_min, real_t &tot_vol, real_t &min_vol, real_t &max_vol,
                     real_t &E, real_t &nu,
                     Array2D<int> &ess_bdr_displacement, Array<int> &ess_bdr_filter,
                     int &solid_attr, int &void_attr,
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
         ess_bdr_displacement.SetSize(3, num_bdr_attr);
         ess_bdr_displacement = 0;
         ess_bdr_displacement(0,3) = 1;

         ess_bdr_filter.SetSize(num_bdr_attr);
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
         if (max_vol < 0) { max_vol = tot_vol*0.3; }
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

      case Arch2:
      {
         filename << "Arch2";
         if (r_min < 0) { r_min = 0.05; }
         if (E < 0) { E = 1.0; }
         if (nu < 0) { nu = 0.3; }
         mesh = new Mesh(Mesh::MakeCartesian2D(3, 2, Element::Type::QUADRILATERAL, false,
                                               1.5, 1.0));
         tot_vol = 0.0;
         for (int i=0; i<mesh->GetNE(); i++) { tot_vol += mesh->GetElementVolume(i); }
         if (min_vol < 0) { min_vol = tot_vol*0.3;}
         if (max_vol < 0) { max_vol = tot_vol*1.0; }
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
            return x[0] > 1.5 - std::pow(2.0, -5.0) && x[1] < 1e-09;
         });
         num_bdr_attr++;
         ess_bdr_displacement.SetSize(3, num_bdr_attr);
         ess_bdr_displacement = 0;
         ess_bdr_displacement(1, 3) = 1; // left: x-fixed
         ess_bdr_displacement(0, 4) = 1; // right-bottom: fixed

         ess_bdr_filter.SetSize(num_bdr_attr);
         ess_bdr_filter = 0;
         break;
      }

      case Bridge2:
      {
         filename << "Bridge2";
         if (r_min < 0) { r_min = 0.02; }
         if (E < 0) { E = 1.0; }
         if (nu < 0) { nu = 0.3; }
         mesh = new Mesh(Mesh::MakeCartesian2D(2, 1, Element::Type::QUADRILATERAL, false,
                                               2.0, 1.0));
         tot_vol = 0.0;
         for (int i=0; i<mesh->GetNE(); i++) { tot_vol += mesh->GetElementVolume(i); }
         if (min_vol < 0) { min_vol = tot_vol*0.2; }
         if (max_vol < 0) { max_vol = tot_vol*0.7; }
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
         MarkBoundaries(*mesh,++num_bdr_attr, // 5
                        [](const Vector &x)
         {
            return x[0] > 2.0 - std::pow(2.0, -5.0) && x[1] < 1e-09;
         });
         ess_bdr_displacement.SetSize(3, num_bdr_attr);
         ess_bdr_displacement = 0;
         ess_bdr_displacement(1, 3) = 1; // left: x-fixed
         ess_bdr_displacement(0, 4) = 1; // right: fixed

         ess_bdr_filter.SetSize(num_bdr_attr);
         ess_bdr_filter = 0;
         MarkElements(*mesh, 2, [](const Vector &x) {return x[1]>1.0 - std::pow(2,-5);});
         solid_attr = 2;

         break;
      }

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
         ess_bdr_filter = -1; // default no material
         ess_bdr_filter[2] = 0; // right: free
         ess_bdr_filter[4] = 0; // left: free

         MarkElements(*mesh, 2,
         [](const Vector &x) {return std::pow(x[0]-1.9, 2.0) + std::pow(x[2] - 0.1, 2.0) < 0.05*0.05;});
         solid_attr = 2;
         break;
      }

      case MBB3: {MFEM_ABORT("Undefined yet"); break; }

      case Arch3: {MFEM_ABORT("Undefined yet"); break; }

      case Bridge3: {MFEM_ABORT("Undefined yet"); break; }

      case Torsion3:
      {
         filename << "Torsion3";
         if (r_min < 0) { r_min = 0.02; }
         if (E < 0) { E = 1.0; }
         if (nu < 0) { nu = 0.3; }
         mesh = new Mesh(Mesh::MakeCartesian3D(5, 6, 6, Element::Type::HEXAHEDRON,
                                               0.5, 0.6, 0.6));
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
         ess_bdr_displacement(0, 2) = 1; // Right is fixed.


         ess_bdr_filter.SetSize(num_bdr_attr);
         ess_bdr_filter = -1; // default: void boundary
         ess_bdr_filter[2] = 0; // Right: free
         ess_bdr_filter[4] = 0; // Left: fee
         break;
      }

      case ForceInverter2:
      {
         filename << "ForceInverter2";
         if (r_min < 0) { r_min = 0.02; }
         if (E < 0) { E = 1.0; }
         if (nu < 0) { nu = 0.3; }
         mesh = new Mesh(Mesh::MakeCartesian2D(2, 1, Element::Type::QUADRILATERAL, false,
                                               2.0, 1.0));
         tot_vol = 0.0;
         for (int i=0; i<mesh->GetNE(); i++) { tot_vol += mesh->GetElementVolume(i); }
         if (min_vol < 0) { min_vol = tot_vol*0.0; }
         if (max_vol < 0) { max_vol = tot_vol*0.3; }
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
         MarkBoundaries(*mesh, ++num_bdr_attr,
                        [](const Vector &x)
         {
            // Top right 5: Output port
            return x[0] > 2.0 - 1e-09 && x[1] > 1.0 - std::pow(2,-5);
         });
         MarkBoundaries(*mesh, ++num_bdr_attr,
                        [](const Vector &x)
         {
            // Top left 6: Input Port
            return x[0] < 0.0 + 1e-09 && x[1] > 1.0 - std::pow(2,-5);
         });
         MarkBoundaries(*mesh, ++num_bdr_attr,
                        [](const Vector &x)
         {
            // Bottom left 7: Fixed
            return x[0] < 0.0 + 1e-09 && x[1] < 0.0 + std::pow(2,-5);
         });
         ess_bdr_displacement.SetSize(3, num_bdr_attr);
         ess_bdr_displacement = 0;
         ess_bdr_displacement(2, 2) = 1; // top: x-roller
         ess_bdr_displacement(0, 6) = 1;

         ess_bdr_filter.SetSize(num_bdr_attr);
         ess_bdr_filter = 0;
         ess_bdr_filter[4] = 1;
         ess_bdr_filter[5] = 1;
         ess_bdr_filter[6] = 1;
         break;
      }
      default: MFEM_ABORT("Problem Undefined");
   }
   return mesh;
}

void SetupTopoptProblem(TopoptProblem prob,
                        HelmholtzFilter &filter, ElasticityProblem &elasticity,
                        GridFunction &filter_gf, GridFunction &state_gf)
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

      case Arch2:
      {
         auto g = new Vector({0.0, -9.8});
         auto gravity_cf = new VectorConstantCoefficient(*g);
         auto filter_cf = new GridFunctionCoefficient(&filter_gf);
         auto state_cf = new VectorGridFunctionCoefficient(&state_gf);

         auto weight_cf = new ScalarVectorProductCoefficient(*filter_cf, *gravity_cf);
         auto gu = new InnerProductCoefficient(*gravity_cf, *state_cf);

         elasticity.GetLinearForm()->AddDomainIntegrator(
            new VectorDomainLFIntegrator(*weight_cf)
         );
         elasticity.MakeCoefficientOwner(gravity_cf);
         elasticity.MakeCoefficientOwner(filter_cf);
         elasticity.MakeCoefficientOwner(weight_cf);
         elasticity.MakeVectorOwner(g);

         filter.GetAdjLinearForm()->AddDomainIntegrator(
            new DomainLFIntegrator(*gu)
         );
         filter.MakeCoefficientOwner(state_cf);
         filter.MakeCoefficientOwner(gu);
         break;
      }

      case Bridge2:
      {
         auto g = new Vector({0.0, -9.8});
         auto gravity_cf = new VectorConstantCoefficient(*g);
         auto filter_cf = new GridFunctionCoefficient(&filter_gf);
         auto state_cf = new VectorGridFunctionCoefficient(&state_gf);

         auto load = new VectorFunctionCoefficient(
            2, [](const Vector &x, Vector &f)
         {
            f = 0.0;
            if (x[1] > 1 - std::pow(2, -5))
            {
               f[1] = -40;
            }
         }
         );
         auto weight_cf = new ScalarVectorProductCoefficient(*filter_cf, *gravity_cf);
         auto total_load_cf = new VectorSumCoefficient(*weight_cf, *load);
         auto gu = new InnerProductCoefficient(*gravity_cf, *state_cf);

         elasticity.GetLinearForm()->AddDomainIntegrator(
            new VectorDomainLFIntegrator(*total_load_cf)
         );
         elasticity.MakeCoefficientOwner(gravity_cf);
         elasticity.MakeCoefficientOwner(filter_cf);
         elasticity.MakeCoefficientOwner(weight_cf);
         elasticity.MakeCoefficientOwner(total_load_cf);
         elasticity.MakeVectorOwner(g);

         filter.GetAdjLinearForm()->AddDomainIntegrator(
            new DomainLFIntegrator(*gu)
         );
         filter.MakeCoefficientOwner(state_cf);
         filter.MakeCoefficientOwner(gu);
         break;
      }

      case Cantilever3:
      {
         auto load = new VectorFunctionCoefficient(
            3, [](const Vector &x, Vector &f)
         {
            f = 0.0;
            if (std::pow(x[0]-1.9, 2.0) + std::pow(x[2] - 0.1, 2.0) < 0.05*0.05)
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

      case Torsion3:
      {
         auto load = new VectorFunctionCoefficient(
            3, [](const Vector &x, Vector &f)
         {
            // Apply rotating force on the left center
            f=0.0;
            if (std::pow(x[1]-0.6, 2.0) + std::pow(x[2]-0.6, 2.0) < 0.2*0.2 && x[0] < 0.1)
            {
               f[1] = -(x[2]-0.6);
               f[2] = x[1]-0.6;
            }
         });
         elasticity.MakeCoefficientOwner(load);
         elasticity.GetLinearForm()->AddDomainIntegrator(
            new VectorDomainLFIntegrator(*load)
         );
         break;
      }

      case ForceInverter2:
      {
         real_t k_in(1), k_out(0.0001);

         auto d_in = new Vector({1.0, 0.0});
         auto d_out = new Vector({-1.0, 0.0});
         auto d_in_cf = new VectorConstantCoefficient(*d_in);
         auto d_out_cf = new VectorConstantCoefficient(*d_out);

         auto load = new ScalarVectorProductCoefficient(1.0*std::pow(2.0, 8.0),
                                                        *d_in_cf);
         auto obj = new ScalarVectorProductCoefficient(-1.0*std::pow(2.0, 8.0),
                                                       *d_out_cf);

         auto input_bdr = new Array<int>(7); *input_bdr = 0; (*input_bdr)[5] = 1;
         auto output_bdr = new Array<int>(7); *output_bdr = 0; (*output_bdr)[4] = 1;

         elasticity.GetBilinearForm()->AddBdrFaceIntegrator(
            new DirectionalHookesLawBdrIntegrator(k_in, d_in_cf), *input_bdr);
         elasticity.GetBilinearForm()->AddBdrFaceIntegrator(
            new DirectionalHookesLawBdrIntegrator(k_out, d_out_cf), *output_bdr);
         elasticity.GetLinearForm()->AddBdrFaceIntegrator(
            new VectorBoundaryLFIntegrator(*load), *input_bdr);
         elasticity.GetAdjLinearForm()->AddBdrFaceIntegrator(
            new VectorBoundaryLFIntegrator(*obj), *output_bdr);

         elasticity.MakeVectorOwner(d_in);
         elasticity.MakeVectorOwner(d_out);
         elasticity.MakeVectorOwner(input_bdr);
         elasticity.MakeVectorOwner(output_bdr);
         elasticity.MakeCoefficientOwner(d_in_cf);
         elasticity.MakeCoefficientOwner(d_out_cf);
         elasticity.MakeCoefficientOwner(load);
         elasticity.MakeCoefficientOwner(obj);

         break;
      }
      default: MFEM_ABORT("Problem Undefined");
   }
}
} // end of namespace
