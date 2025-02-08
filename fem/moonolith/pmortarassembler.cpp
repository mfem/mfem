// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "pmortarassembler.hpp"
#include "transferutils.hpp"

#include "cut.hpp"

#include "moonolith_bounding_volume_with_span.hpp"
#include "moonolith_n_tree_mutator_factory.hpp"
#include "moonolith_n_tree_with_span_mutator_factory.hpp"
#include "moonolith_n_tree_with_tags_mutator_factory.hpp"
#include "moonolith_profiler.hpp"
#include "moonolith_redistribute.hpp"
#include "moonolith_sparse_matrix.hpp"
#include "moonolith_tree.hpp"
#include "par_moonolith.hpp"

#include <memory>

using namespace mfem::internal;

namespace mfem
{

struct ParMortarAssembler::Impl
{
public:
   MPI_Comm comm;
   std::shared_ptr<ParFiniteElementSpace> source;
   std::shared_ptr<ParFiniteElementSpace> destination;
   std::vector<std::shared_ptr<MortarIntegrator>> integrators;

   std::shared_ptr<HypreParMatrix> coupling_matrix;
   std::shared_ptr<HypreParMatrix> mass_matrix;

   std::shared_ptr<IterativeSolver> solver;

   bool verbose{false};

   bool assemble_mass_and_coupling_together{true};

   void ensure_solver()
   {
      if(!solver) {
         solver = std::make_shared<BiCGSTABSolver>(comm);
         solver->SetMaxIter(20000);
         solver->SetRelTol(1e-6);
      }
   }

   bool is_vector_fe() const
   {
      bool is_vector_fe = false;
      for (auto i_ptr : integrators)
      {
         if (i_ptr->is_vector_fe())
         {
            is_vector_fe = true;
            break;
         }
      }

      return is_vector_fe;
   }

   BilinearFormIntegrator * new_mass_integrator() const
   {
      if (is_vector_fe())
      {
         return new VectorFEMassIntegrator();
      }
      else
      {
         return new MassIntegrator();
      }
   }
};

ParMortarAssembler::~ParMortarAssembler() = default;

void ParMortarAssembler::SetSolver(const std::shared_ptr<IterativeSolver> &solver)
{
   impl_->solver = solver;
}

void ParMortarAssembler::SetAssembleMassAndCouplingTogether(const bool value)
{
   impl_->assemble_mass_and_coupling_together = value;
}

void ParMortarAssembler::SetMaxSolverIterations(const int max_solver_iterations)
{
   impl_->ensure_solver();

   impl_->solver->SetMaxIter(max_solver_iterations);
}

void ParMortarAssembler::AddMortarIntegrator(
   const std::shared_ptr<MortarIntegrator> &integrator)
{
   impl_->integrators.push_back(integrator);
}

void ParMortarAssembler::SetVerbose(const bool verbose)
{
   impl_->verbose = verbose;
}

template <int Dimension> class ElementAdapter : public moonolith::Serializable
{
public:
   using Bound = moonolith::AABBWithKDOPSpan<Dimension, double>;
   using Point = moonolith::Vector<double, Dimension>;

   inline int tag() const { return tag_; }

   const Bound &bound() const { return bound_; }

   Bound &bound() { return bound_; }

   void applyRW(moonolith::Stream &stream)
   {
      stream &bound_;
      stream &element_;
      stream &element_handle_;
   }

   ElementAdapter(FiniteElementSpace &fe, const long element,
                  const long element_handle, const int tag)
      : fe_(&fe), element_(element), element_handle_(element_handle), tag_(tag),
        dof_map_(nullptr)
   {
      assert(element < fe.GetNE());

      DenseMatrix pts;
      fe_->GetMesh()->GetPointMatrix(element, pts);

      Point p;
      for (int j = 0; j < pts.Width(); ++j)
      {
         for (int i = 0; i < pts.Height(); ++i)
         {
            p[i] = pts.Elem(i, j);
         }

         bound_.static_bound() += p;
         bound_.dynamic_bound() += p;
      }
   }

   ElementAdapter()
      : fe_(nullptr), element_(-1), element_handle_(-1), tag_(-1),
        dof_map_(nullptr) {}

   inline long handle() const { return element_handle_; }

   inline long element() const { return element_; }

   inline const FiniteElement &get() const
   {
      assert(fe_);
      assert(element_ < fe_->GetNE());
      return *fe_->GetFE(element_);
   }

   inline const FiniteElementSpace &space() const
   {
      assert(fe_);
      return *fe_;
   }

   void set_dof_map(std::vector<long> *ptr) { dof_map_ = ptr; }

   void get_elements_vdofs(Array<int> &vdofs) const
   {
      fe_->GetElementVDofs(element_, vdofs);

      if (dof_map_)
      {
         assert(dof_map_->size() == vdofs.Size());

         for (int i = 0; i < vdofs.Size(); ++i)
         {
            vdofs[i] = dof_map_->at(i);
         }

      }
      else
      {
         assert(false);
      }
   }

private:
   FiniteElementSpace *fe_;
   long element_;
   long element_handle_;
   int tag_;
   Bound bound_;
   std::vector<long> *dof_map_;
};

template <int _Dimension> class TreeTraits
{
public:
   enum { Dimension = _Dimension };

   using Bound = moonolith::AABBWithKDOPSpan<Dimension, double>;
   using DataType = mfem::ElementAdapter<Dimension>;
};

template <int Dimension>
class MFEMTree : public moonolith::Tree<TreeTraits<Dimension>>
{
public:
   using Traits = mfem::TreeTraits<Dimension>;

   MFEMTree() {};

   static std::shared_ptr<MFEMTree>
   New(const int maxElementsXNode = moonolith::DEFAULT_REFINE_MAX_ELEMENTS,
       const int maxDepth = moonolith::DEFAULT_REFINE_DEPTH)
   {
      using namespace moonolith;

      std::shared_ptr<MFEMTree> tree = std::make_shared<MFEMTree>();
      std::shared_ptr<NTreeWithSpanMutatorFactory<MFEMTree>> factory =
                                                             std::make_shared<NTreeWithSpanMutatorFactory<MFEMTree>>();
      factory->set_refine_params(maxElementsXNode, maxDepth);
      tree->set_mutator_factory(factory);
      return tree;
   }

   static std::shared_ptr<MFEMTree>
   New(const std::shared_ptr<moonolith::Predicate> &predicate,
       const int maxElementsXNode = moonolith::DEFAULT_REFINE_MAX_ELEMENTS,
       const int maxDepth = moonolith::DEFAULT_REFINE_DEPTH)
   {
      using namespace moonolith;

      if (!predicate)
      {
         return New(maxElementsXNode, maxDepth);
      }

      std::shared_ptr<MFEMTree> tree = std::make_shared<MFEMTree>();
      std::shared_ptr<NTreeWithTagsMutatorFactory<MFEMTree>> factory =
                                                             std::make_shared<NTreeWithTagsMutatorFactory<MFEMTree>>(predicate);
      factory->set_refine_params(maxElementsXNode, maxDepth);
      tree->set_mutator_factory(factory);
      return tree;
   }
};

class ElementDofMap : public moonolith::Serializable
{
public:
   void read(moonolith::InputStream &is) override
   {
      int n;
      is >> n;
      global.resize(n);
      is.read(&global[0], n);
   }

   void write(moonolith::OutputStream &os) const override
   {
      int n = global.size();
      os << n;
      os.write(&global[0], n);
   }

   std::vector<long> global;
};

class Spaces
{
public:
   explicit Spaces(const moonolith::Communicator &comm) : comm(comm)
   {
      must_destroy_attached[0] = false;
      must_destroy_attached[1] = false;
   }

   Spaces(const std::shared_ptr<ParFiniteElementSpace> &source,
          const std::shared_ptr<ParFiniteElementSpace> &destination)
   {
      spaces_.reserve(2);
      spaces_.push_back(source);
      spaces_.push_back(destination);

      must_destroy_attached[0] = false;
      must_destroy_attached[1] = false;

      copy_global_dofs(*source, dof_maps_[0]);
      copy_global_dofs(*destination, dof_maps_[1]);
   }

   ~Spaces()
   {
      Mesh *m = nullptr;
      FiniteElementCollection *fec = nullptr;

      for (int i = 0; i < spaces_.size(); ++i)
      {
         if (spaces_[i] && must_destroy_attached[0])
         {
            m = spaces_[i]->GetMesh();
            fec = const_cast<FiniteElementCollection *>(spaces_[i]->FEColl());

            // make it null
            spaces_[i] = std::shared_ptr<FiniteElementSpace>();

            delete m;
            delete fec;
         }
      }
   }

   inline long n_elements() const
   {
      long ret = 0;
      for (auto s : spaces_)
      {
         if (s)
         {
            ret += s->GetNE();
         }
      }

      return ret;
   }

   inline std::vector<std::shared_ptr<FiniteElementSpace>> &spaces()
   {
      return spaces_;
   }

   inline const std::vector<std::shared_ptr<FiniteElementSpace>> &
                                                              spaces() const
   {
      return spaces_;
   }

   inline std::vector<ElementDofMap> &dof_map(const int i)
   {
      assert(i < 2);
      assert(i >= 0);
      return dof_maps_[i];
   }

   inline const std::vector<ElementDofMap> &dof_map(const int i) const
   {
      assert(i < 2);
      assert(i >= 0);
      return dof_maps_[i];
   }

   inline void set_must_destroy_attached(const int index, const bool value)
   {
      assert(index < 2);
      assert(index >= 0);
      must_destroy_attached[index] = value;
   }

private:
   std::vector<std::shared_ptr<FiniteElementSpace>> spaces_;
   moonolith::Communicator comm;
   std::vector<ElementDofMap> dof_maps_[2];
   bool must_destroy_attached[2];

   inline static void copy_global_dofs(ParFiniteElementSpace &fe,
                                       std::vector<ElementDofMap> &dof_map)
   {
      dof_map.resize(fe.GetNE());
      Array<int> vdofs;
      for (int i = 0; i < fe.GetNE(); ++i)
      {
         fe.GetElementVDofs(i, vdofs);
         for (int k = 0; k < vdofs.Size(); ++k)
         {
            long g_dof = 0;
            if (vdofs[k] >= 0)
            {
               g_dof = fe.GetGlobalTDofNumber(vdofs[k]);
            }
            else
            {
               g_dof = -1 - fe.GetGlobalTDofNumber(-1 - vdofs[k]);
            }

            dof_map[i].global.push_back(g_dof);
         }
      }
   }
};

template <class Iterator>
static void write_space(const Iterator &begin, const Iterator &end,
                        FiniteElementSpace &space,
                        const std::vector<ElementDofMap> &dof_map,
                        const int role, moonolith::OutputStream &os)
{
   const int dim = space.GetMesh()->Dimension();
   const long n_elements = std::distance(begin, end);

   std::set<long> nodeIds;
   std::map<long, long> mapping;

   Array<int> verts;
   for (Iterator it = begin; it != end; ++it)
   {
      const int i = *it;
      space.GetElementVertices(i, verts);

      for (int j = 0; j < verts.Size(); ++j)
      {
         nodeIds.insert(verts[j]);
      }
   }

   long n_nodes = nodeIds.size();

   // Estimate for allocation
   os.request_space((n_elements * 8 + n_nodes * dim) *
                    (sizeof(double) + sizeof(long)));

   auto fe_coll = space.FEColl();
   const char *name = fe_coll->Name();
   const int name_lenght = strlen(name);
   // WRITE 1
   os << dim << role;
   os << name_lenght;
   os.write(name, name_lenght);

   long index = 0;
   for (auto nodeId : nodeIds)
   {
      mapping[nodeId] = index++;
   }

   // WRITE 2
   os << n_nodes;
   // WRITE 6
   os << n_elements;

   Array<int> vdofs;
   for (auto node_id : nodeIds)
   {
      double *v = space.GetMesh()->GetVertex(node_id);
      for (int i = 0; i < dim; ++i)
      {
         // WRITE 3
         os << v[i];
      }
   }

   for (Iterator it = begin; it != end; ++it)
   {
      const int k = *it;
      space.GetElementVertices(k, verts);

      const int attribute = space.GetAttribute(k);
      const int e_n_nodes = verts.Size();
      const int type = space.GetElementType(k);
      const int order = space.GetOrder(k);

      // WRITE 7
      os << type << attribute << order << e_n_nodes;

      for (int i = 0; i < e_n_nodes; ++i)
      {
         auto it = mapping.find(verts[i]);
         assert(it != mapping.end());

         int index = it->second;

         // WRITE 8
         os << index;
      }

      // WRITE 9
      os << dof_map.at(k);
   }
}

template <class Iterator>
static void write_element_selection(const Iterator &begin, const Iterator &end,
                                    const Spaces &spaces,
                                    moonolith::OutputStream &os)
{
   if (spaces.spaces().empty())
   {
      assert(false);
      return;
   }

   auto m = spaces.spaces()[0];
   std::shared_ptr<FiniteElementSpace> s = nullptr;

   if (spaces.spaces().size() > 1)
   {
      s = spaces.spaces()[1];
   }

   std::vector<long> source_selection;
   std::vector<long> destination_selection;

   bool met_destination_selection = false;

   for (Iterator it = begin; it != end; ++it)
   {
      long index = *it;

      if (m && index >= m->GetNE())
      {
         index -= m->GetNE();
         destination_selection.push_back(index);
      }
      else if (!m)
      {
         met_destination_selection = true;
         destination_selection.push_back(index);
      }
      else
      {
         assert(!met_destination_selection);
         assert(index < m->GetNE());
         source_selection.push_back(index);
      }
   }

   const bool has_source = !source_selection.empty();
   const bool has_destination = !destination_selection.empty();

   os << has_source << has_destination;

   if (has_source)
   {
      write_space(source_selection.begin(), source_selection.end(), *m,
                  spaces.dof_map(0), 0, os);
   }

   if (has_destination)
   {
      write_space(destination_selection.begin(), destination_selection.end(), *s,
                  spaces.dof_map(1), 1, os);
   }
}

static FiniteElementCollection *FECollFromName(const std::string &comp_name)
{
   return FiniteElementCollection::New(comp_name.c_str());
}

static void read_space(moonolith::InputStream &is,
                       std::shared_ptr<FiniteElementSpace> &space,
                       std::vector<ElementDofMap> &dof_map)
{
   using namespace std;

   // READ 1
   int dim, role, name_lenght;
   is >> dim >> role;
   is >> name_lenght;

   std::string name(name_lenght, 0);
   is.read(&name[0], name_lenght);

   // READ 2
   long n_nodes;
   is >> n_nodes;

   // READ 6
   long n_elements;
   is >> n_elements;

   auto fe_coll = FECollFromName(name);
   auto mesh_ptr = new Mesh(dim, n_nodes, n_elements);

   for (long i = 0; i < n_nodes; ++i)
   {
      double v[3];
      for (int i = 0; i < dim; ++i)
      {
         // READ 3
         is >> v[i];
      }

      mesh_ptr->AddVertex(v);
   }

   dof_map.resize(n_elements);
   std::vector<int> e2v;
   for (long i = 0; i < n_elements; ++i)
   {
      // READ 7
      int type, attribute, order, e_n_nodes;
      is >> type >> attribute >> order >> e_n_nodes;
      e2v.resize(e_n_nodes);
      int index;
      for (int i = 0; i < e_n_nodes; ++i)
      {
         // READ 8
         is >> index;
         e2v[i] = index;
      }

      mesh_ptr->AddElement(NewElem(type, &e2v[0], attribute));
      // READ 9
      is >> dof_map.at(i);
   }

   // if(mesh_ptr->Dimension() == 3) {
   Finalize(*mesh_ptr, true);
   // }

   space = make_shared<FiniteElementSpace>(mesh_ptr, fe_coll);
}

static void read_spaces(moonolith::InputStream &is, Spaces &spaces)
{
   bool has_source, has_destination;
   is >> has_source >> has_destination;

   spaces.spaces().resize(2);

   if (has_source)
   {
      read_space(is, spaces.spaces()[0], spaces.dof_map(0));
      spaces.set_must_destroy_attached(0, true);
   }
   else
   {
      spaces.spaces()[0] = nullptr;
   }

   if (has_destination)
   {
      read_space(is, spaces.spaces()[1], spaces.dof_map(1));
      spaces.set_must_destroy_attached(1, true);
   }
   else
   {
      spaces.spaces()[1] = nullptr;
   }
}

template <int Dimensions, class Fun>
static bool Assemble(moonolith::Communicator &comm,
                     std::shared_ptr<ParFiniteElementSpace> &source,
                     std::shared_ptr<ParFiniteElementSpace> &destination,
                     Fun process_fun, const moonolith::SearchSettings &settings,
                     const bool verbose)
{
   using namespace moonolith;

   typedef mfem::MFEMTree<Dimensions> NTreeT;
   typedef typename NTreeT::DataContainer DataContainer;
   typedef typename NTreeT::DataType Adapter;

   long maxNElements = settings.max_elements;
   long maxDepth = settings.max_depth;

   const int n_elements_source = source->GetNE();
   const int n_elements_destination = destination->GetNE();
   const int n_elements = n_elements_source + n_elements_destination;

   auto predicate = std::make_shared<MasterAndSlave>();
   predicate->add(0, 1);

   MOONOLITH_EVENT_BEGIN("create_adapters");

   std::shared_ptr<NTreeT> tree = NTreeT::New(predicate, maxNElements, maxDepth);
   tree->reserve(n_elements);

   std::shared_ptr<Spaces> local_spaces =
      std::make_shared<Spaces>(source, destination);

   int offset = 0;
   int space_num = 0;

   for (auto s : local_spaces->spaces())
   {
      if (s)
      {
         for (int i = 0; i < s->GetNE(); ++i)
         {
            Adapter a(*s, i, offset + i, space_num);
            a.set_dof_map(&local_spaces->dof_map(space_num)[i].global);
            tree->insert(a);
         }

         offset += s->GetNE();
      }

      ++space_num;
   }

   tree->root()->bound().static_bound().enlarge(1e-6);

   MOONOLITH_EVENT_END("create_adapters");

   // Just to have an indexed-storage
   std::map<long, std::shared_ptr<Spaces>> spaces;
   std::map<long, std::vector<std::shared_ptr<Spaces>>> migrated_spaces;

   auto read = [&spaces, &migrated_spaces,
                         comm](const long ownerrank, const long /*senderrank*/,
                      bool is_forwarding, DataContainer &data, InputStream &in)
   {
      CHECK_STREAM_READ_BEGIN("vol_proj", in);

      std::shared_ptr<Spaces> proc_space = std::make_shared<Spaces>(comm);

      read_spaces(in, *proc_space);

      if (!is_forwarding)
      {
         assert(!spaces[ownerrank]);
         spaces[ownerrank] = proc_space;
      }
      else
      {
         migrated_spaces[ownerrank].push_back(proc_space);
      }

      data.reserve(data.size() + proc_space->n_elements());

      int space_num = 0;
      long offset = 0;
      for (auto s : proc_space->spaces())
      {
         if (s)
         {
            for (int i = 0; i < s->GetNE(); ++i)
            {
               data.push_back(Adapter(*s, i, offset + i, space_num));
               data.back().set_dof_map(&proc_space->dof_map(space_num)[i].global);
            }

            offset += s->GetNE();
         }

         ++space_num;
      }

      CHECK_STREAM_READ_END("vol_proj", in);
   };

   auto write = [&local_spaces, &spaces,
                                &comm](const long ownerrank, const long /*recvrank*/,
                        const std::vector<long>::const_iterator &begin,
                        const std::vector<long>::const_iterator &end,
                        const DataContainer &/*data*/, OutputStream &out)
   {
      CHECK_STREAM_WRITE_BEGIN("vol_proj", out);

      if (ownerrank == comm.rank())
      {
         write_element_selection(begin, end, *local_spaces, out);
      }
      else
      {
         auto it = spaces.find(ownerrank);
         assert(it != spaces.end());
         std::shared_ptr<Spaces> spaceptr = it->second;
         assert(std::distance(begin, end) > 0);
         write_element_selection(begin, end, *spaceptr, out);
      }

      CHECK_STREAM_WRITE_END("vol_proj", out);
   };

   long n_false_positives = 0, n_intersections = 0;
   auto fun = [&n_false_positives, &n_intersections,
                                   &process_fun](Adapter &source, Adapter &destination) -> bool
   {
      bool ok = process_fun(source, destination);

      if (ok)
      {
         n_intersections++;
         return true;
      }
      else
      {
         n_false_positives++;
         return false;
      }

      return true;
   };

   moonolith::search_and_compute(comm, tree, predicate, read, write, fun,
                                 settings);

   if (verbose)
   {
      long n_total_candidates = n_intersections + n_false_positives;

      long n_collection[3] = {n_intersections, n_total_candidates,
                              n_false_positives
                             };
      comm.all_reduce(n_collection, 3, moonolith::MPISum());

      if (comm.is_root())
      {
         mfem::out << "n_intersections: " << n_collection[0]
                   << ", n_total_candidates: " << n_collection[1]
                   << ", n_false_positives: " << n_collection[2] << std::endl;
      }
   }

   return true;
}

static void add_matrix(const Array<int> &destination_vdofs, const Array<int> &source_vdofs,
                       const DenseMatrix &elem_mat, moonolith::SparseMatrix<double> &mat_buffer)
{
   for (int i = 0; i < destination_vdofs.Size(); ++i)
   {
      long dof_I = destination_vdofs[i];

      double sign_I = 1.0;

      if (dof_I < 0)
      {
         sign_I = -1.0;
         dof_I = -dof_I - 1;
      }

      for (int j = 0; j < source_vdofs.Size(); ++j)
      {
         long dof_J = source_vdofs[j];

         double sign_J = sign_I;

         if (dof_J < 0)
         {
            sign_J = -sign_I;
            dof_J = -dof_J - 1;
         }

         mat_buffer.add(dof_I, dof_J, sign_J * elem_mat.Elem(i, j));
      }
   }
}

std::shared_ptr<HypreParMatrix> convert_to_hypre_matrix(
   const std::vector<moonolith::Integer> &destination_ranges,
    HYPRE_BigInt *s_offsets,
    HYPRE_BigInt *m_offsets,
   moonolith::SparseMatrix<double> &mat_buffer)  {

   auto && comm = mat_buffer.comm();

   moonolith::Redistribute<moonolith::SparseMatrix<double>> redist(comm.get());
   redist.apply(destination_ranges, mat_buffer, moonolith::AddAssign<double>());

   std::vector<int> I(s_offsets[1] - s_offsets[0] + 1);
   I[0] = 0;

   std::vector<HYPRE_Int> J;
   std::vector<double> data;
   J.reserve(mat_buffer.n_local_entries());
   data.reserve(J.size());

   for (auto it = mat_buffer.iter(); it; ++it)
   {
      I[it.row() - s_offsets[0] + 1]++;
      J.push_back(it.col());
      data.push_back(*it);
   }

   for (int i = 1; i < I.size(); ++i)
   {
      I[i] += I[i - 1];
   }

   return std::make_shared<HypreParMatrix>(
             comm.get(), s_offsets[1] - s_offsets[0], mat_buffer.rows(),
             mat_buffer.cols(), &I[0], &J[0], &data[0], s_offsets, m_offsets);
}


int order_multiplier(const Geometry::Type type, const int dim)
{
   return
   (type == Geometry::TRIANGLE || type == Geometry::TETRAHEDRON ||
      type == Geometry::SEGMENT)? 1 : dim;
}

template <int Dimensions>
static bool
Assemble(moonolith::Communicator &comm,
         ParMortarAssembler::Impl &impl,
         const moonolith::SearchSettings &settings, const bool verbose)
{
   auto &source = impl.source;
   auto &destination = impl.destination;

   auto &integrators = impl.integrators;
   auto &pmat  = impl.coupling_matrix;


   bool lump_mass = false;
   int max_q_order = 0;

   for (auto i_ptr : integrators)
   {
      max_q_order = std::max(i_ptr->GetQuadratureOrder(), max_q_order);
   }

   const int dim = source->GetMesh()->Dimension();
   std::shared_ptr<Cut> cut = NewCut(dim);
   if (!cut)
   {
      assert(false && "NOT Supported!");
      return false;
   }

   Array<int> source_vdofs, destination_vdofs;
   DenseMatrix elemmat;
   DenseMatrix cumulative_elemmat;
   IntegrationRule src_ir;
   IntegrationRule dest_ir;

   double local_element_matrices_sum = 0.0;

   const auto m_global_n_dofs = source->GlobalVSize();
   auto *m_offsets = source->GetDofOffsets();

   const auto s_global_n_dofs = destination->GlobalVSize();
   auto *s_offsets = destination->GetDofOffsets();

   moonolith::SparseMatrix<double> mat_buffer(comm);
   mat_buffer.set_size(s_global_n_dofs, m_global_n_dofs);

   moonolith::SparseMatrix<double> mass_mat_buffer(comm);
   mass_mat_buffer.set_size(s_global_n_dofs, s_global_n_dofs);

   std::unique_ptr<BilinearFormIntegrator> mass_integr(impl.new_mass_integrator());

   auto fun = [&](const ElementAdapter<Dimensions> &source,
                  const ElementAdapter<Dimensions> &destination) -> bool
   {
      const auto &src = source.space();
      const auto &dest = destination.space();

      const int src_index = source.element();
      const int dest_index = destination.element();

      auto &src_fe = *src.GetFE(src_index);
      auto &dest_fe = *dest.GetFE(dest_index);

      ElementTransformation &dest_Trans =
      *dest.GetElementTransformation(dest_index);

      // Quadrature order mangling

      int src_order_mult = order_multiplier(src_fe.GetGeomType(), Dimensions);
      int dest_order_mult = order_multiplier(dest_fe.GetGeomType(), Dimensions);

      const int src_order = src_order_mult * src_fe.GetOrder();
      const int dest_order = dest_order_mult * dest_fe.GetOrder();

      int contraction_order = src_order + dest_order;
      if(impl.assemble_mass_and_coupling_together) {
         contraction_order = std::max(contraction_order, 2 * dest_order);
      }

      const int order = contraction_order + dest_order_mult * dest_Trans.OrderW() + max_q_order;

      cut->SetIntegrationOrder(order);

      if (cut->BuildQuadrature(src, src_index, dest, dest_index, src_ir,
                               dest_ir))
      {
         // make reference quadratures
         ElementTransformation &src_Trans =
         *src.GetElementTransformation(src_index);

         source.get_elements_vdofs(source_vdofs);
         destination.get_elements_vdofs(destination_vdofs);

         bool first = true;
         for (auto i_ptr : integrators)
         {
            if (first)
            {
               i_ptr->AssembleElementMatrix(src_fe, src_ir, src_Trans, dest_fe,
                                            dest_ir, dest_Trans, cumulative_elemmat);
               first = false;
            }
            else
            {
               i_ptr->AssembleElementMatrix(src_fe, src_ir, src_Trans, dest_fe,
                                            dest_ir, dest_Trans, elemmat);
               cumulative_elemmat += elemmat;
            }
         }

         local_element_matrices_sum += Sum(cumulative_elemmat);
         add_matrix(destination_vdofs, source_vdofs, cumulative_elemmat, mat_buffer);

         if(impl.assemble_mass_and_coupling_together) {
            mass_integr->SetIntRule(&dest_ir);
            mass_integr->AssembleElementMatrix(dest_fe, dest_Trans, elemmat);

            if(lump_mass) {
               int n = destination_vdofs.Size();

               for(int i = 0; i < n; ++i) {
                  double row_sum = 0.;
                  for(int j = 0; j < n; ++j) {
                     row_sum += elemmat(i, j);
                     elemmat(i, j) = 0.;
                  }

                  elemmat(i, i) = row_sum;
               }
            }

            add_matrix(destination_vdofs, destination_vdofs, elemmat, mass_mat_buffer);
         }

         return true;
      }
      else
      {
         return false;
      }
   };

   if (!Assemble<Dimensions>(comm, source, destination, fun, settings,
                             verbose))
   {
      return false;
   }

   if (verbose)
   {
      double volumes[2] = {local_element_matrices_sum};
      comm.all_reduce(volumes, 2, moonolith::MPISum());

      cut->Describe();

      if (comm.is_root())
      {
         mfem::out << "sum(B): " << volumes[0] << std::endl;
      }
   }

   std::vector<moonolith::Integer> destination_ranges(comm.size() + 1, 0);

   std::copy(s_offsets, s_offsets + 2, destination_ranges.begin() + comm.rank());
   comm.all_reduce(&destination_ranges[0], destination_ranges.size(),
                   moonolith::MPIMax());

   pmat = convert_to_hypre_matrix(
      destination_ranges,
      s_offsets,
       m_offsets,
      mat_buffer);

   if(impl.assemble_mass_and_coupling_together) {
      impl.mass_matrix = convert_to_hypre_matrix(destination_ranges, s_offsets, s_offsets, mass_mat_buffer);
   }

   return true;
}

ParMortarAssembler::ParMortarAssembler(
   const std::shared_ptr<ParFiniteElementSpace> &source,
   const std::shared_ptr<ParFiniteElementSpace> &destination)
   : impl_(new Impl())
{
   impl_->comm = source->GetComm();
   impl_->source = source;
   impl_->destination = destination;
}

bool ParMortarAssembler::Assemble(std::shared_ptr<HypreParMatrix> &pmat)
{
   assert(!impl_->integrators.empty() &&
          "it must have at least on integrator see class MortarIntegrator");

   moonolith::SearchSettings settings;
   // settings.set("disable_redistribution", moonolith::Boolean(true));
   // settings.set("disable_asynch", moonolith::Boolean(true));

   bool ok = false;

   moonolith::Communicator comm(impl_->comm);
   if (impl_->source->GetMesh()->Dimension() == 2)
   {
      ok = mfem::Assemble<2>(comm, *impl_, settings,
                               impl_->verbose);
   }

   if (impl_->source->GetMesh()->Dimension() == 3)
   {
      ok = mfem::Assemble<3>(comm, *impl_, settings,
                               impl_->verbose);
   }

   pmat = impl_->coupling_matrix;
   return ok;
}

bool ParMortarAssembler::Transfer(const ParGridFunction &src_fun,
                                  ParGridFunction &dest_fun)
{

   return Update() && Apply(src_fun, dest_fun);
}

bool ParMortarAssembler::Apply(const ParGridFunction &src_fun,
                               ParGridFunction &dest_fun)
{
   const bool verbose = impl_->verbose;

   if (!impl_->coupling_matrix)
   {
      if (!Update())
      {
         return false;
      }
   }

   auto &B = *impl_->coupling_matrix;
   auto &D = *impl_->mass_matrix;

   auto &P_source = *impl_->source->Dof_TrueDof_Matrix();
   auto &P_destination = *impl_->destination->Dof_TrueDof_Matrix();

   impl_->ensure_solver();

   // BlockILU prec(D);
   // impl_->solver->SetPreconditioner(prec);

   impl_->solver->SetOperator(D);

   if(verbose) {
      impl_->solver->SetPrintLevel(3);
   }

   Vector P_x_src_fun(B.Width());
   P_source.MultTranspose(src_fun, P_x_src_fun);

   Vector B_x_src_fun(B.Height());
   B_x_src_fun = 0.0;

   B.Mult(P_x_src_fun, B_x_src_fun);

   Vector R_x_dest_fun(D.Height());
   R_x_dest_fun = 0.0;

   impl_->solver->Mult(B_x_src_fun, R_x_dest_fun);

   P_destination.Mult(R_x_dest_fun, dest_fun);

   dest_fun.Update();

   // Sanity check!
   int converged = impl_->solver->GetFinalNorm() < 1e-5;
   MPI_Allreduce(MPI_IN_PLACE, &converged, 1, MPI_INT, MPI_MIN, impl_->comm);
   return converged;
}

bool ParMortarAssembler::Update()
{
   using namespace std;
   const bool verbose = impl_->verbose;
   static const bool dof_transformation = true;

   moonolith::Communicator comm(impl_->comm);

   if (verbose)
   {

      moonolith::root_describe(
         "--------------------------------------------------------"
         "\nAssembly begin: ",
         comm, mfem::out);
   }

   if (!Assemble(impl_->coupling_matrix))
   {
      return false;
   }

   if (verbose)
   {
      moonolith::root_describe(
         "\nAssembly end: "
         "--------------------------------------------------------",
         comm, mfem::out);
   }

   if (dof_transformation)
   {
      impl_->coupling_matrix.reset(RAP(impl_->destination->Dof_TrueDof_Matrix(),
                                       impl_->coupling_matrix.get(),
                                       impl_->source->Dof_TrueDof_Matrix()));
   }

   auto &B = *impl_->coupling_matrix;

   if(!impl_->assemble_mass_and_coupling_together) {
      ParBilinearForm b_form(impl_->destination.get());
      b_form.AddDomainIntegrator(impl_->new_mass_integrator());
      b_form.Assemble();
      b_form.Finalize();
      impl_->mass_matrix = shared_ptr<HypreParMatrix>(b_form.ParallelAssemble());
   }


   comm.barrier();

   if (verbose && comm.is_root())
   {
      mfem::out << "P in R^(" << impl_->source->Dof_TrueDof_Matrix()->Height();
      mfem::out << " x " << impl_->source->Dof_TrueDof_Matrix()->Width() << ")\n";
      mfem::out << "Q in R^("
                << impl_->destination->Dof_TrueDof_Matrix()->Height();
      mfem::out << " x " << impl_->destination->Dof_TrueDof_Matrix()->Width()
                << ")\n";
   }

   if (dof_transformation && impl_->assemble_mass_and_coupling_together)
   {
      impl_->mass_matrix.reset(RAP(impl_->destination->Dof_TrueDof_Matrix(),
                                   impl_->mass_matrix.get(),
                                   impl_->destination->Dof_TrueDof_Matrix()));
   }

   auto &D = *impl_->mass_matrix;

   comm.barrier();

   if (verbose && comm.is_root())
   {
      mfem::out << "--------------------------------------------------------"
                << std::endl;
      mfem::out << "B in R^(" << B.GetGlobalNumRows() << " x "
                << B.GetGlobalNumCols() << ")" << std::endl;
      mfem::out << "D in R^(" << D.GetGlobalNumRows() << " x "
                << D.GetGlobalNumCols() << ")" << std::endl;
      mfem::out << "--------------------------------------------------------"
                << std::endl;
   }

   comm.barrier();

   if (verbose)
   {
      Vector v(D.Width());
      v = 1.0;

      Vector Dv(D.Height());
      D.Mult(v, Dv);

      double sum_Dv = Dv.Sum();
      comm.all_reduce(&sum_Dv, 1, MPI_SUM);

      if (comm.is_root())
      {
         mfem::out << "sum(D): " << sum_Dv << std::endl;
      }
   }

   return true;
}

} // namespace mfem

#endif // MFEM_USE_MPI
