// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
// Created on: Jan 22, 2019
// Author: William F Godoy godoywf@ornl.gov
// adios2: Adaptable Input/Output System https://github.com/ornladios/ADIOS2

#include "adios2stream.hpp"

#ifdef MFEM_USE_ADIOS2

#include "../fem/geom.hpp"
#include "../general/array.hpp"
#include "../mesh/element.hpp"
#include "../mesh/mesh.hpp"
#include "../fem/gridfunc.hpp"

#include <algorithm>

namespace mfem
{

namespace
{
// these functions might be included in adios2 upstream next release
template <class T>
adios2::Variable<T> SafeDefineVariable(adios2::IO io,
                                       const std::string& variable_name,
                                       const adios2::Dims& shape = adios2::Dims(),
                                       const adios2::Dims& start = adios2::Dims(),
                                       const adios2::Dims& count = adios2::Dims())
{
   adios2::Variable<T> variable = io.InquireVariable<T>(variable_name);
   if (variable)
   {
      if (variable.Count() != count &&
          variable.ShapeID() == adios2::ShapeID::LocalArray)
      {
         variable.SetSelection({start, count});
      }
   }
   else
   {
      variable = io.DefineVariable<T>(variable_name, shape, start, count);
   }

   return variable;
}

template <class T>
adios2::Attribute<T> SafeDefineAttribute(adios2::IO io,
                                         const std::string& attribute_name,
                                         const T& value,
                                         const std::string& variable_name = "",
                                         const std::string separator = "/")
{
   adios2::Attribute<T> attribute = io.InquireAttribute<T>(attribute_name);
   if (attribute)
   {
      return attribute;
   }
   return io.DefineAttribute<T>(attribute_name, value, variable_name, separator );
}

template <class T>
adios2::Attribute<T> SafeDefineAttribute(adios2::IO io,
                                         const std::string& attribute_name,
                                         const T* values, const size_t size,
                                         const std::string& variable_name = "",
                                         const std::string separator = "/")
{
   adios2::Attribute<T> attribute = io.InquireAttribute<T>(attribute_name);
   if (attribute)
   {
      return attribute;
   }
   return io.DefineAttribute<T>(attribute_name, values, size, variable_name,
                                separator );
}

bool SetBoolParameter(const std::string key,
                      const std::map<std::string, std::string>& parameters,
                      const bool default_value) noexcept
{
   auto it = parameters.find(key);
   if (it != parameters.end())
   {
      std::string value = it->second;
      std::transform(value.begin(), value.end(), value.begin(), ::tolower);
      if (value == "on" || value == "true")
      {
         return true;
      }
      else if ( value == "off" || value == "false")
      {
         return false;
      }
   }
   return default_value;
}

} //end empty namespace

// PUBLIC
#ifdef MFEM_USE_MPI
adios2stream::adios2stream(const std::string& name, const openmode mode,
                           MPI_Comm comm, const std::string engineType)
   : name(name),
     adios2_openmode(mode),
     adios(new adios2::ADIOS(comm)),
     io(adios->DeclareIO(name))
{
   io.SetEngine(engineType);
}
#else
adios2stream::adios2stream(const std::string& name, const openmode mode,
                           const std::string engineType)
   : name(name),
     adios2_openmode(mode),
     adios(new adios2::ADIOS()),
     io(adios->DeclareIO(name))
{
   io.SetEngine(engineType);
}
#endif

adios2stream::~adios2stream()
{
   if (engine)
   {
      SafeDefineAttribute<std::string>(io, "vtk.xml", VTKSchema() );
      engine.Close();
   }
}

void adios2stream::SetParameters(
   const std::map<std::string, std::string>& parameters)
{
   io.SetParameters(parameters);
   refine = SetBoolParameter("RefinedData", parameters, true);
}

void adios2stream::SetParameter(const std::string key,
                                const std::string value) noexcept
{
   io.SetParameter(key, value);
   if (key == "RefinedData")
   {
      refine = SetBoolParameter("RefinedData", io.Parameters(), true);
   }
}

void adios2stream::BeginStep()
{
   if (!engine)
   {
      engine = io.Open(name, adios2::Mode::Write);
   }
   engine.BeginStep();
   active_step = true;
}

void adios2stream::EndStep()
{
   if (!engine || !active_step)
   {
      const std::string message = "MFEM adios2stream error: calling EndStep "
                                  "on uninitialized step (need BeginStep)";
      mfem_error(message.c_str());
   }

   SafeDefineAttribute<std::string>(io, "vtk.xml", VTKSchema() );
   engine.EndStep();
   active_step = false;
}

void adios2stream::SetTime(const double time)
{
   adios2::Variable<double> var_time = SafeDefineVariable<double>(io, "TIME");
   engine.Put(var_time, time);
   transient = true;
}

void adios2stream::SetCycle(const int cycle)
{
   adios2::Variable<int> var_cycle = SafeDefineVariable<int>(io,"CYCLE");
   engine.Put(var_cycle, cycle);
}

void adios2stream::SetRefinementLevel(const int level) noexcept
{
   refinement_level = level;
}

size_t adios2stream::CurrentStep() const
{
   return engine.CurrentStep();
}

void adios2stream::Close()
{
   if (engine)
   {
      if (!active_step)
      {
         SafeDefineAttribute<std::string>(io, "vtk.xml", VTKSchema() );
      }
      engine.Close();
   }
   if (adios)
   {
      adios.reset();
   }
}


// PROTECTED (accessible by friend class Mesh)
void adios2stream::Print(const Mesh& mesh, const mode print_mode)
{
   auto lf_DefineMeshMetadata = [this](Mesh& mesh)
   {
      // check types are constant
      if (!IsConstantElementType(mesh.elements))
      {
         throw std::invalid_argument("MFEM::adios2stream ERROR: non-constant "
                                     " element types not yet implemented\n");
      }

      // format info
      SafeDefineAttribute<std::string>(io, "format", "MFEM ADIOS2 BP v0.2" );
      SafeDefineAttribute<std::string>(io, "format/version", "0.2" );
      std::string mesh_type = "Unknown";
      std::vector<std::string> viz_tools;
      viz_tools.reserve(2); //for now
      if (mesh.NURBSext)
      {
         mesh_type = "MFEM NURBS";
         viz_tools.push_back("NONE");
      }
      else if (mesh.ncmesh)
      {
         mesh_type = "MFEM mesh v1.1";
         viz_tools.push_back("NONE");
      }
      else
      {
         mesh_type = "MFEM mesh v1.0";
         viz_tools.push_back("Paraview: ADIOS2VTXReader");
         viz_tools.push_back("VTK: vtkADIOS2VTXReader.h");
      }

      SafeDefineAttribute<std::string>(io, "format/mfem_mesh", mesh_type );
      SafeDefineAttribute<std::string>(io, "format/viz_tools", viz_tools.data(),
                                       viz_tools.size() );

      // elements
      const uint32_t dimension = static_cast<int32_t>(mesh.Dimension());
      SafeDefineAttribute<uint32_t>(io, "dimension", dimension);
      SafeDefineVariable<uint32_t>(io,"NumOfElements", {adios2::LocalValueDim});
      SafeDefineVariable<uint32_t>(io, "types");
      size_t nelements = 0;
      size_t element_nvertices = 0;
      size_t nvertices = 0;

      if (refine)
      {
         for (int i = 0; i < mesh.GetNE(); ++i)
         {
            const Geometry::Type type = mesh.GetElementBaseGeometry(i);
            RefinedGeometry* refined_geometry = GlobGeometryRefiner.Refine(type,
                                                                           refinement_level, 1);
            if (refined_geometry == nullptr)
            {
               mfem_error("ERROR: could not refine geometry in call to Save with adios2stream \n");
            }


            element_nvertices = static_cast<size_t>(Geometries.GetVertices(
                                                       type)->GetNPoints());
            nelements += refined_geometry->RefGeoms.Size() / element_nvertices;
            nvertices  += refined_geometry->RefPts.GetNPoints();
         }

         refined_mesh_nelements = nelements;
         refined_mesh_nvertices = nvertices;
      }
      else
      {
         nelements = static_cast<size_t>(mesh.GetNE());
         element_nvertices = static_cast<size_t>(mesh.elements[0]->GetNVertices());
      }
      SafeDefineVariable<uint64_t>(io, "connectivity", {}, {}, {nelements, element_nvertices+1});
      SafeDefineVariable<int32_t>(io, "attribute", {}, {}, {nelements});

      // vertices
      SafeDefineVariable<uint32_t>(io,"NumOfVertices", {adios2::LocalValueDim});
      if (refine)
      {
         SafeDefineVariable<double>( io, "vertices", {}, {}, {nvertices, static_cast<size_t>(dimension)});
      }
      else
      {
         const GridFunction* grid_function = mesh.GetNodes();
         if (grid_function == nullptr)
         {
            const size_t nVertices = static_cast<size_t>(mesh.GetNV());
            const size_t spaceDim = static_cast<size_t>(mesh.SpaceDimension());
            // similar to Ordering::byVDIM
            SafeDefineVariable<double>( io, "vertices", {}, {}, {nVertices, spaceDim});
         }
         else
         {
            const size_t size = static_cast<size_t>(grid_function->Size());
            const FiniteElementSpace* fes = grid_function->FESpace();
            const size_t components = static_cast<size_t>(fes->GetVDim());
            const size_t tuples = size /components;
            SafeDefineVariable<double>(io, "vertices", {}, {}, {tuples, components} );

            if (fes->GetOrdering() == Ordering::byNODES)
            {
               ordering_by_node = true;
            }
         }
      }
   };

   auto lf_PrintRefinedMeshData = [this](Mesh& mesh)
   {
      // elements and vertices
      engine.Put("NumOfElements", static_cast<uint32_t>(refined_mesh_nelements));
      engine.Put("NumOfVertices", static_cast<uint32_t>(refined_mesh_nvertices));

      const uint32_t vtkType =
         GLVISToVTKType(static_cast<int>(mesh.elements[0]->GetGeometryType()));
      engine.Put("types", vtkType);

      adios2::Variable<double> var_vertices = io.InquireVariable<double>("vertices");
      adios2::Variable<double>::Span span_vertices = engine.Put<double>(var_vertices);

      adios2::Variable<uint64_t> var_connectivity =
         io.InquireVariable<uint64_t>("connectivity");
      adios2::Variable<uint64_t>::Span span_connectivity = engine.Put<uint64_t>
                                                           (var_connectivity);

      adios2::Variable<int32_t> var_element_attribute =
         io.InquireVariable<int32_t>("attribute");
      adios2::Variable<int32_t>::Span span_element_attribute = engine.Put<int32_t>
                                                               (var_element_attribute);

      size_t span_vertices_offset = 0;
      size_t span_connectivity_offset = 0;
      size_t span_element_attribute_offset = 0;
      // use for setting absolute node id for each element
      size_t point_id = 0;
      DenseMatrix pmatrix;

      for (int e = 0; e < mesh.GetNE(); ++e)
      {
         const Geometry::Type type = mesh.GetElementBaseGeometry(e);
         RefinedGeometry* refined_geometry = GlobGeometryRefiner.Refine(type,
                                                                        refinement_level, 1);
         // vertices
         mesh.GetElementTransformation(e)->Transform(refined_geometry->RefPts, pmatrix);
         for (int i = 0; i < pmatrix.Width(); ++i)
         {
            for (int j = 0; j < pmatrix.Height(); ++j)
            {
               span_vertices[span_vertices_offset +  i*pmatrix.Height() + j] = pmatrix(j, i);
            }
         }
         span_vertices_offset += static_cast<size_t>(pmatrix.Width()*pmatrix.Height());

         // element attribute
         const int element_attribute = mesh.GetAttribute(e);

         // connectivity
         const int nv = Geometries.GetVertices(type)->GetNPoints();
         const Array<int> &element_vertices = refined_geometry->RefGeoms;

         for (int v = 0; v < element_vertices.Size();)
         {
            span_connectivity[span_connectivity_offset] = static_cast<uint64_t>(nv);
            ++span_connectivity_offset;

            span_element_attribute[span_element_attribute_offset] = static_cast<int32_t>
                                                                    (element_attribute);
            ++span_element_attribute_offset;

            for (int k =0; k < nv; k++, v++ )
            {
               span_connectivity[span_connectivity_offset] = static_cast<uint64_t>
                                                             (point_id + element_vertices[v]);
               ++span_connectivity_offset;
            }
         }

         point_id += static_cast<size_t>(refined_geometry->RefPts.GetNPoints());
      }

      for (int e = 0; e < mesh.GetNE(); ++e)
      {
         const Geometry::Type type = mesh.GetElementBaseGeometry(e);
         RefinedGeometry* refined_geometry = GlobGeometryRefiner.Refine(type,
                                                                        refinement_level, 1);
      }
   };

   auto lf_PrintMeshData = [&](Mesh& mesh)
   {
      if (refine)
      {
         lf_PrintRefinedMeshData(mesh);
         return;
      }

      // elements
      engine.Put("NumOfElements", static_cast<uint32_t>(mesh.GetNE()));

      const uint32_t vtkType =
         GLVISToVTKType(static_cast<int>(mesh.elements[0]->GetGeometryType()));
      engine.Put("types", vtkType);

      adios2::Variable<uint64_t> varConnectivity =
         io.InquireVariable<uint64_t>("connectivity");
      // zero-copy access to adios2 buffer to put non-contiguous to contiguous memory
      adios2::Variable<uint64_t>::Span spanConnectivity =
         engine.Put<uint64_t>(varConnectivity);

      adios2::Variable<int32_t> varElementAttribute =
         io.InquireVariable<int32_t>("attribute");
      // zero-copy access to adios2 buffer to put non-contiguous to contiguous memory
      adios2::Variable<int32_t>::Span spanElementAttribute =
         engine.Put<int32_t>(varElementAttribute);

      size_t elementPosition = 0;
      for (int e = 0; e < mesh.GetNE(); ++e)
      {
         spanElementAttribute[e] = static_cast<int32_t>(mesh.GetAttribute(e));

         const int nVertices = mesh.elements[e]->GetNVertices();
         spanConnectivity[elementPosition] = nVertices;
         for (int v = 0; v < nVertices; ++v)
         {
            spanConnectivity[elementPosition + v + 1] =
               mesh.elements[e]->GetVertices()[v];
         }
         elementPosition += nVertices + 1;
      }

      // vertices
      engine.Put("NumOfVertices", static_cast<uint32_t>(mesh.GetNV()));

      if (mesh.GetNodes() == nullptr)
      {
         adios2::Variable<double> varVertices = io.InquireVariable<double>("vertices");
         // zero-copy access to adios2 buffer to put non-contiguous to contiguous memory
         adios2::Variable<double>::Span spanVertices = engine.Put(varVertices);

         for (int v = 0; v < mesh.GetNV(); ++v)
         {
            const int space_dim = mesh.SpaceDimension();
            for (int coord = 0; coord < space_dim; ++coord)
            {
               spanVertices[v * space_dim + coord] = mesh.vertices[v](coord);
            }
         }
      }
      else
      {
         const GridFunction* grid_function = mesh.GetNodes();
         if (ordering_by_node)
         {
            adios2::Variable<double> varVertices = io.InquireVariable<double>("vertices");
            // zero-copy access to adios2 buffer to put non-contiguous to contiguous memory
            adios2::Variable<double>::Span spanVertices = engine.Put(varVertices);

            const size_t size = static_cast<size_t>(grid_function->Size());
            const FiniteElementSpace* fes = grid_function->FESpace();
            const size_t components = static_cast<size_t>(fes->GetVDim());
            const size_t tuples = size /components;

            const double* data = grid_function->GetData();

            for (size_t i = 0; i < tuples; ++i)
            {
               for (size_t j = 0; j < components; ++j)
               {
                  spanVertices[i*components + j] = data[j*tuples + i];
               }
            }
         }
         else
         {
            grid_function->Print(*this, "vertices");
         }
      }
   };

   // BODY OF FUNCTION STARTS HERE
   try
   {
      Mesh ref_mesh(mesh);
      lf_DefineMeshMetadata(ref_mesh);

      if (!engine) // if Engine is closed
      {
         engine = io.Open(name, adios2::Mode::Write);
      }

      lf_PrintMeshData(ref_mesh);

      if (print_mode == mode::sync)
      {
         engine.PerformPuts();
      }
   }
   catch (std::exception& e)
   {
      const std::string warning =
         "MFEM: adios2stream exception caught, invalid bp dataset: " + name +
         "," + e.what();
      mfem_warning( warning.c_str());
   }
}

void adios2stream::Save(const GridFunction& grid_function,
                        const std::string& variable_name, const data_type type)
{
   auto lf_SafeDefine = [&](const std::string& variable_name,
                            const size_t tuples, const size_t components,
                            const Ordering::Type ordering,
                            const std::string& fespace_name)
   {
      adios2::Variable<double> var = io.InquireVariable<double>(variable_name);
      if (!var)
      {
         if (components == 1 && type == adios2stream::data_type::point_data)
         {
            io.DefineVariable<double>(variable_name, {}, {}, {tuples*components});
         }
         else
         {
            const adios2::Dims count = (ordering == Ordering::byNODES) ?
                                       adios2::Dims{components, tuples} :
                                       adios2::Dims{tuples, components};
            io.DefineVariable<double>(variable_name, {}, {}, count);
         }
         SafeDefineAttribute<std::string>(io, "FiniteElementSpace",
                                          fespace_name, variable_name);

      }
   };

   // BODY OF FUNCTION STARTS HERE
   const std::map<std::string, std::string> parameters = io.Parameters();
   const bool full_data = SetBoolParameter("FullData", parameters, false);

   if (!full_data && !refine)
   {
      return;
   }

   const FiniteElementSpace* fes = grid_function.FESpace();

   if (refine)
   {
      const Mesh *mesh = fes->GetMesh();
      const size_t components = static_cast<size_t>(grid_function.VectorDim());
      // const size_t tuples = static_cast<size_t>(mesh->GetNV());
      const size_t tuples = refined_mesh_nvertices;

      lf_SafeDefine(variable_name, tuples, components,
                    Ordering::byVDIM, std::string(fes->FEColl()->Name()));
      if (type == adios2stream::data_type::point_data)
      {
         point_data_variables.insert(variable_name);
      }

      RefinedGeometry* refined_geometry;
      DenseMatrix transform;

      // zero-copy access to adios2 buffer to put non-contiguous to contiguous memory
      adios2::Variable<double> variable = io.InquireVariable<double>(variable_name);
      adios2::Variable<double>::Span span = engine.Put<double>(variable);

      size_t offset = 0;
      if (components == 1)
      {
         Vector scalar;

         const int nelements = mesh->GetNE();
         for (int e = 0; e < nelements; ++e)
         {
            refined_geometry = GlobGeometryRefiner.Refine(
                                  mesh->GetElementBaseGeometry(e), refinement_level, 1);

            grid_function.GetValues(e, refined_geometry->RefPts, scalar, transform);

            const int size = scalar.Size();

            for (int i = 0; i < size; ++i)
            {
               const double value = scalar(i);
               span.at(offset+i) = value;
            }
            offset += static_cast<size_t>(size);
         }
      }
      else
      {
         DenseMatrix vector;
         for (int e = 0; e < mesh->GetNE(); ++e)
         {
            refined_geometry = GlobGeometryRefiner.Refine(
                                  mesh->GetElementBaseGeometry(e), refinement_level, 1);
            grid_function.GetVectorValues(e, refined_geometry->RefPts, vector, transform);

            for (int i = 0; i < vector.Width(); ++i)
            {
               for (int j = 0; j < vector.Height(); ++j)
               {
                  span[offset +  i*vector.Height() + j] = vector(j, i);
               }
            }
            offset += static_cast<size_t>(vector.Width()*vector.Height());
         }
      }
   }

   if (full_data)
   {
      const size_t size = static_cast<size_t>(grid_function.Size());
      const size_t components = static_cast<size_t>(fes->GetVDim());
      const size_t tuples = size /components;
      lf_SafeDefine(variable_name +"/full", tuples, components,
                    fes->GetOrdering(),
                    std::string(fes->FEColl()->Name()) );
      // calls Vector::Print
      grid_function.Print(*this, variable_name+"/full");

      if (!refine && type == adios2stream::data_type::point_data)
      {
         point_data_variables.insert(variable_name+"/full");
      }
   }
}

// PRIVATE
int32_t adios2stream::GLVISToVTKType(
   const int glvisType) const noexcept
{
   uint32_t vtkType = 0;
   switch (glvisType)
   {
      case Geometry::Type::POINT:
         vtkType = 1;
         break;
      case Geometry::Type::SEGMENT:
         vtkType = 3;
         break;
      case Geometry::Type::TRIANGLE:
         vtkType = 5;
         break;
      case Geometry::Type::SQUARE:
         // vtkType = 8;
         vtkType = 9;
         break;
      case Geometry::Type::TETRAHEDRON:
         vtkType = 10;
         break;
      case Geometry::Type::CUBE:
         // vtkType = 11;
         vtkType = 12;
         break;
      case Geometry::Type::PRISM:
         vtkType = 13;
         break;
      default:
         vtkType = 0;
         break;
   }
   return vtkType;
}

bool adios2stream::IsConstantElementType(const Array<Element*>& elements ) const
noexcept
{
   bool isConstType = true;
   const Geometry::Type type = elements[0]->GetGeometryType();

   for (int e = 1; e < elements.Size(); ++e)
   {
      if (type != elements[e]->GetGeometryType())
      {
         isConstType = false;
         break;
      }
   }
   return isConstType;
}

std::string adios2stream::VTKSchema() const noexcept
{
   std::string vtkSchema = R"(
<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.2" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="NumOfVertices" NumberOfCells="NumOfElements">
      <Points>
        <DataArray Name="vertices" />)";

   vtkSchema += R"(
      </Points>
	  <CellData>
        <DataArray Name="attribute" />
	  </CellData>
      <Cells>
        <DataArray Name="connectivity" />
        <DataArray Name="types" />
      </Cells>
      <PointData>)";

   if (point_data_variables.empty())
   {
      vtkSchema += "\n";
   }
   else
   {
      for (const std::string& point_datum : point_data_variables )
      {
         vtkSchema += "        <DataArray Name=\"" + point_datum +"\"/>\n";
      }
   }

   if (transient)
   {
      vtkSchema += "        <DataArray Name=\"TIME\">\n";
      vtkSchema += "          TIME\n";
      vtkSchema += "        </DataArray>\n";
   }

   vtkSchema += R"(
       </PointData>
       </Piece>
     </UnstructuredGrid>
   </VTKFile>)";

   return vtkSchema;
}

adios2::Mode adios2stream::ToADIOS2Mode(const adios2stream::openmode mode) const
noexcept
{
   adios2::Mode adios2Mode = adios2::Mode::Undefined;
   switch (mode)
   {
      case adios2stream::openmode::out:
         adios2Mode = adios2::Mode::Write;
         break;
      case adios2stream::openmode::in:
         adios2Mode = adios2::Mode::Read;
         break;
      default:
         const std::string message = "MFEM adios2stream ERROR: only "
                                     "openmode::out and openmode::in "
                                     " are valid, in call to adios2stream constructor";
         mfem_error(message.c_str());
   }
   return adios2Mode;
}

}  // end namespace mfem

#endif // MFEM_USE_ADIOS2
