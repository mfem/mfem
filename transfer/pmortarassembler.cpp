#include "pmortarassembler.hpp"
#include "transferutils.hpp"

#include "cut.hpp"

#include "moonolith_profiler.hpp"
#include "moonolith_redistribute.hpp"
#include "moonolith_tree.hpp"
#include "moonolith_n_tree_mutator_factory.hpp"
#include "moonolith_n_tree_with_span_mutator_factory.hpp"
#include "moonolith_n_tree_with_tags_mutator_factory.hpp"
#include "moonolith_sparse_matrix.hpp"
#include "moonolith_bounding_volume_with_span.hpp"
#include "par_moonolith.hpp"

#include <memory>

namespace mfem
{

template<int Dimension>
class ElementAdapter : public moonolith::Serializable
{
public:
   using Bound = moonolith::AABBWithKDOPSpan<Dimension, double>;
   using Point = moonolith::Vector<double, Dimension>;

   inline int tag() const
   {
      return tag_;
   }

   const Bound &bound() const
   {
      return bound_;
   }

   Bound &bound()
   {
      return bound_;
   }

   void applyRW(moonolith::Stream &stream)
   {
      stream & bound_;
      stream & element_;
      stream & element_handle_;
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

         bound_.static_bound()  += p;
         bound_.dynamic_bound() += p;
      }
   }

   ElementAdapter()
      : fe_(nullptr), element_(-1), element_handle_(-1), tag_(-1),
        dof_map_(nullptr) {}

   inline long handle() const
   {
      return element_handle_;
   }

   inline long element() const
   {
      return element_;
   }

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

   void set_dof_map(std::vector<long> * ptr)
   {
      dof_map_ = ptr;
   }

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
   std::vector<long> * dof_map_;
};

template<int _Dimension>
class TreeTraits
{
public:
   enum
   {
      Dimension = _Dimension
   };

   using Bound = moonolith::AABBWithKDOPSpan<Dimension, double>;
   using DataType = mfem::ElementAdapter<Dimension>;
};

template<int Dimension>
class MFEMTree : public moonolith::Tree< TreeTraits<Dimension> >
{
public:
   using Traits = mfem::TreeTraits<Dimension>;

   MFEMTree() {};

   static std::shared_ptr<MFEMTree> New(const int maxElementsXNode =
                                           moonolith::DEFAULT_REFINE_MAX_ELEMENTS,
                                        const int maxDepth = moonolith::DEFAULT_REFINE_DEPTH)
   {
      using namespace moonolith;

      std::shared_ptr<MFEMTree> tree = std::make_shared<MFEMTree>();
      std::shared_ptr< NTreeWithSpanMutatorFactory<MFEMTree> > factory =
         std::make_shared< NTreeWithSpanMutatorFactory<MFEMTree> >();
      factory->set_refine_params(maxElementsXNode, maxDepth);
      tree->set_mutator_factory(factory);
      return tree;
   }

   static std::shared_ptr<MFEMTree> New(const std::shared_ptr<moonolith::Predicate>
                                        &predicate,
                                        const int maxElementsXNode = moonolith::DEFAULT_REFINE_MAX_ELEMENTS,
                                        const int maxDepth = moonolith::DEFAULT_REFINE_DEPTH)
   {
      using namespace moonolith;

      if (!predicate)
      {
         return New(maxElementsXNode, maxDepth);
      }

      std::shared_ptr<MFEMTree> tree = std::make_shared<MFEMTree>();
      std::shared_ptr< NTreeWithTagsMutatorFactory<MFEMTree> > factory =
         std::make_shared< NTreeWithTagsMutatorFactory<MFEMTree> > (predicate);
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

   Spaces(const std::shared_ptr<ParFiniteElementSpace> &master,
          const std::shared_ptr<ParFiniteElementSpace> &slave)
   {
      spaces_.reserve(2);
      spaces_.push_back(master);
      spaces_.push_back(slave);

      must_destroy_attached[0] = false;
      must_destroy_attached[1] = false;

      copy_global_dofs(*master, dof_maps_[0]);
      copy_global_dofs(*slave,  dof_maps_[1]);
   }

   ~Spaces()
   {
      Mesh * m = nullptr;
      FiniteElementCollection * fec = nullptr;

      for (int i = 0; i < spaces_.size(); ++i)
      {
         if (spaces_[i] && must_destroy_attached[0])
         {
            m         = spaces_[i]->GetMesh();
            fec       = const_cast<FiniteElementCollection *>(spaces_[i]->FEColl());

            //make it null
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

   inline std::vector< std::shared_ptr<FiniteElementSpace> > &spaces()
   {
      return spaces_;
   }

   inline const std::vector< std::shared_ptr<FiniteElementSpace> > &spaces() const
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
   std::vector< std::shared_ptr<FiniteElementSpace> > spaces_;
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

template<class Iterator>
static void write_space(const Iterator &begin, const Iterator &end,
                        FiniteElementSpace &space, const std::vector<ElementDofMap> &dof_map,
                        const int role, moonolith::OutputStream &os)
{
   const int dim       = space.GetMesh()->Dimension();
   const long n_elements = std::distance(begin, end);

   std::set<long> nodeIds;
   std::map<long, long> mapping;

   Array< int > verts;
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
   os.request_space( (n_elements * 8 + n_nodes * dim) * (sizeof(double) + sizeof(
                                                            long)) );

   auto fe_coll         = space.FEColl();
   const char * name       = fe_coll->Name();
   const int name_lenght  = strlen(name);
   //WRITE 1
   os << dim << role;
   os << name_lenght;
   os.write(name, name_lenght);

   long index = 0;
   for (auto nodeId : nodeIds)
   {
      mapping[nodeId] = index++;
   }

   //WRITE 2
   os << n_nodes;
   //WRITE 6
   os << n_elements;

   Array<int> vdofs;
   for (auto node_id : nodeIds)
   {
      double * v = space.GetMesh()->GetVertex(node_id);
      for (int i = 0; i < dim; ++i)
      {
         //WRITE 3
         os << v[i];
      }
   }

   for (Iterator it = begin; it != end; ++it)
   {
      const int k = *it;
      auto &elem = *space.GetFE(k);
      space.GetElementVertices(k, verts);

      const int attribute = space.GetAttribute(k);
      const int e_n_nodes = verts.Size();
      const int type       = space.GetElementType(k);
      const int order      = space.GetOrder(k);

      //WRITE 7
      os << type << attribute << order << e_n_nodes;

      for (int i = 0; i < e_n_nodes; ++i)
      {
         auto it = mapping.find(verts[i]);
         assert(it != mapping.end());

         int index = it->second;

         //WRITE 8
         os << index;
      }

      //WRITE 9
      os << dof_map.at(k);
   }
}

template<class Iterator>
static void write_element_selection(const Iterator &begin, const Iterator &end,
                                    const Spaces &spaces, moonolith::OutputStream &os)
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
      s  = spaces.spaces()[1];
   }

   std::vector<long> master_selection;
   std::vector<long> slave_selection;

   bool met_slave_selection = false;

   for (Iterator it = begin; it != end; ++it)
   {
      long index = *it;

      if (m && index >= m->GetNE())
      {
         index -= m->GetNE();
         slave_selection.push_back(index);
      }
      else if (!m)
      {
         met_slave_selection = true;
         slave_selection.push_back(index);
      }
      else
      {
         assert(!met_slave_selection);
         assert(index < m->GetNE());
         master_selection.push_back(index);
      }
   }

   const bool has_master = !master_selection.empty();
   const bool has_slave  = !slave_selection.empty();

   os << has_master << has_slave;

   if (has_master)
   {
      write_space(master_selection.begin(), master_selection.end(), *m,
                  spaces.dof_map(0), 0, os);
   }

   if (has_slave)
   {
      write_space(slave_selection.begin(), slave_selection.end(), *s,
                  spaces.dof_map(1), 1, os);
   }
}



static FiniteElementCollection * FECollFromName(const std::string &comp_name)
{
   return FiniteElementCollection::New(comp_name.c_str());
}

static void read_space(moonolith::InputStream &is,
                       std::shared_ptr<FiniteElementSpace> &space, std::vector<ElementDofMap> &dof_map)
{
   using namespace std;

   //READ 1
   int dim, role, name_lenght;
   is >> dim >> role;
   is >> name_lenght;

   std::string name(name_lenght, 0);
   is.read(&name[0], name_lenght);

   //READ 2
   long n_nodes;
   is >> n_nodes;

   //READ 6
   long n_elements;
   is >> n_elements;

   auto fe_coll = FECollFromName(name);
   auto mesh_ptr = new Mesh(dim, n_nodes, n_elements);

   for (long i = 0; i < n_nodes; ++i)
   {
      double v[3];
      for (int i = 0; i < dim; ++i)
      {
         //READ 3
         is >> v[i];
      }

      mesh_ptr->AddVertex(v);
   }

   dof_map.resize(n_elements);
   std::vector<int> e2v;
   for (long i = 0; i < n_elements; ++i)
   {
      //READ 7
      int type, attribute, order, e_n_nodes;
      is >> type >> attribute >> order >> e_n_nodes;
      e2v.resize(e_n_nodes);
      int index, global_id;
      for (int i = 0; i < e_n_nodes; ++i)
      {
         //READ 8
         is >> index;
         e2v[i] = index;
      }

      mesh_ptr->AddElement(NewElem(type, &e2v[0], attribute));
      //READ 9
      is >> dof_map.at(i);
   }

   // if(mesh_ptr->Dimension() == 3) {
   Finalize(*mesh_ptr, true);
   // }

   space = make_shared<FiniteElementSpace>(mesh_ptr, fe_coll);
}

static void read_spaces(moonolith::InputStream &is, Spaces &spaces)
{
   bool has_master, has_slave;
   is >> has_master >> has_slave;

   spaces.spaces().resize(2);

   if (has_master)
   {
      read_space(is, spaces.spaces()[0], spaces.dof_map(0));
      spaces.set_must_destroy_attached(0, true);
   }
   else
   {
      spaces.spaces()[0] = nullptr;
   }

   if (has_slave)
   {
      read_space(is, spaces.spaces()[1], spaces.dof_map(1));
      spaces.set_must_destroy_attached(1, true);
   }
   else
   {
      spaces.spaces()[1] = nullptr;
   }
}

template<int Dimensions, class Fun>
static bool Assemble(moonolith::Communicator &comm,
                     std::shared_ptr<ParFiniteElementSpace> &master,
                     std::shared_ptr<ParFiniteElementSpace> &slave,
                     Fun process_fun,
                     const moonolith::SearchSettings &settings)
{
   using namespace moonolith;

   typedef mfem::MFEMTree<Dimensions> NTreeT;
   typedef typename NTreeT::DataContainer DataContainer;
   typedef typename NTreeT::DataType Adapter;

   long maxNElements = settings.max_elements;
   long maxDepth = settings.max_depth;
   static const bool verbose = false;


   const int n_elements_master = master->GetNE();
   const int n_elements_slave  = slave->GetNE();
   const int n_elements       = n_elements_master + n_elements_slave;

   auto predicate = std::make_shared<MasterAndSlave>();
   predicate->add(0, 1);

   MOONOLITH_EVENT_BEGIN("create_adapters");
   ////////////////////////////////////////////////////////////////////////////////////////////////////
   std::shared_ptr<NTreeT> tree = NTreeT::New(predicate, maxNElements, maxDepth);
   tree->reserve(n_elements);

   std::shared_ptr<Spaces> local_spaces = std::make_shared<Spaces>(master, slave);

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
   ////////////////////////////////////////////////////////////////////////////////////////////////////
   MOONOLITH_EVENT_END("create_adapters");

   //Just to have an indexed-storage
   std::map<long, std::shared_ptr<Spaces> > spaces;
   std::map<long, std::vector<std::shared_ptr<Spaces> > > migrated_spaces;

   auto read = [&spaces, &migrated_spaces, comm]
               (
                  const long ownerrank,
                  const long senderrank,
                  bool is_forwarding, DataContainer &data,
                  InputStream &in
               )
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
               data.push_back( Adapter(*s, i, offset + i, space_num) );
               data.back().set_dof_map(&proc_space->dof_map(space_num)[i].global);
            }

            offset += s->GetNE();
         }

         ++space_num;
      }

      CHECK_STREAM_READ_END("vol_proj", in);
   };

   auto write = [&local_spaces, &spaces, &comm](const long ownerrank,
                                                const long recvrank,
                                                const std::vector<long>::const_iterator &begin,
                                                const std::vector<long>::const_iterator &end,
                                                const DataContainer &data,
                                                OutputStream &out)
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
   auto fun = [&n_false_positives, &n_intersections, &process_fun](
                 Adapter &master, Adapter &slave) -> bool
   {

      bool ok = process_fun(master, slave);

      if (ok)
      {
         n_intersections++;
         return true;
      }
      else {
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

      long n_collection[3] = {n_intersections, n_total_candidates, n_false_positives};
      comm.all_reduce(n_collection, 3, moonolith::MPISum());

      if (comm.is_root())
      {
         mfem::out <<  "n_intersections: "         << n_collection[0]
                   << ", n_total_candidates: "   << n_collection[1]
                   << ", n_false_positives: "    << n_collection[2] << std::endl;
      }
   }

   return true;
}


template<int Dimensions>
static bool Assemble(
   moonolith::Communicator &comm,
   std::shared_ptr<ParFiniteElementSpace> &master,
   std::shared_ptr<ParFiniteElementSpace> &slave,
   std::vector< std::shared_ptr<MortarIntegrator> > &integrators,
   std::shared_ptr<HypreParMatrix> &pmat,
   const moonolith::SearchSettings &settings)
{
   static const bool verbose = false;
   int max_q_order = 0;

   for (auto i_ptr : integrators)
   {
      max_q_order = std::max(i_ptr->GetQuadratureOrder(), max_q_order);
   }

   std::shared_ptr<CutBase> cut;

   const int dim = master->GetMesh()->Dimension();

   if(dim == 2) {
      cut = std::make_shared<Cut2D>();
   } else if(dim == 3) {
      cut = std::make_shared<Cut3D>();
   } else {
      assert(false && "NOT Supported!");
      return false;
   }

   //////////////////////////////////////////////////
   int skip_zeros = 1;
   Array<int> master_vdofs, slave_vdofs;
   DenseMatrix elemmat;
   DenseMatrix cumulative_elemmat;
   IntegrationRule src_ir;
   IntegrationRule dest_ir;
   //////////////////////////////////////////////////

   double local_element_matrices_sum = 0.0;

   /////////////////////////////////////////////////

   const auto m_global_n_dofs = master->GlobalVSize();
   auto * m_offsets         = master->GetDofOffsets();

   const auto s_global_n_dofs = slave->GlobalVSize();
   auto * s_offsets           = slave->GetDofOffsets();

   moonolith::SparseMatrix<double> mat_buffer(comm);
   mat_buffer.set_size(s_global_n_dofs, m_global_n_dofs);

   auto fun = [&](const ElementAdapter<Dimensions> &master,
                  const ElementAdapter<Dimensions> &slave) -> bool
   {
      const auto &src  = master.space();
      const auto &dest = slave.space();

      const auto &src_mesh  = *src.GetMesh();
      const auto &dest_mesh = *dest.GetMesh();

      const int src_index  = master.element();
      const int dest_index = slave.element();

      auto &src_fe  = *src.GetFE(src_index);
      auto &dest_fe = *dest.GetFE(dest_index);

      ElementTransformation &dest_Trans = *dest.GetElementTransformation(dest_index);
      const int order = src_fe.GetOrder() + dest_fe.GetOrder() + dest_Trans.OrderW() + max_q_order;

      cut->SetIntegrationOrder(order);
      if (cut->BuildQuadrature(src, src_index, dest, dest_index, src_ir, dest_ir))
      {
         //make reference quadratures
         ElementTransformation &src_Trans  = *src.GetElementTransformation(src_index);

         master.get_elements_vdofs(master_vdofs);
         slave.get_elements_vdofs(slave_vdofs);

         bool first = true;
         for (auto i_ptr : integrators)
         {
            if (first)
            {
               i_ptr->AssembleElementMatrix(src_fe, src_ir, src_Trans, dest_fe, dest_ir,
               dest_Trans, cumulative_elemmat);
               first = false;
            }
            else
            {
               i_ptr->AssembleElementMatrix(src_fe, src_ir, src_Trans, dest_fe, dest_ir,
                                            dest_Trans, elemmat);
               cumulative_elemmat += elemmat;
            }
         }

         local_element_matrices_sum += Sum(cumulative_elemmat);


         for (int i = 0; i < slave_vdofs.Size(); ++i)
         {
            long dof_I = slave_vdofs[i];

            double sign_I = 1.0;

            if (dof_I < 0)
            {
               sign_I = -1.0;
               dof_I  = -dof_I - 1;
            }

            for (int j = 0; j < master_vdofs.Size(); ++j)
            {
               long dof_J = master_vdofs[j];

               double sign_J = sign_I;

               if (dof_J < 0)
               {
                  sign_J = -sign_I;
                  dof_J  = -dof_J - 1;
               }

               mat_buffer.add(dof_I, dof_J, sign_J * cumulative_elemmat.Elem(i, j));
            }
         }

         return true;
      }
      else {
         return false;
      }
   };

   if (!Assemble<Dimensions>(comm, master, slave, fun, settings))
   {
      return false;
   }

   if (verbose)
   {
      double volumes[2] = { local_element_matrices_sum  };
      comm.all_reduce(volumes, 2, moonolith::MPISum());

      cut->describe();

      if (comm.is_root())
      {
         mfem::out <<  "sum(B): " << volumes[0] << std::endl;
      }
   }

   std::vector<moonolith::Integer> slave_ranges(comm.size() + 1, 0);

   std::copy(s_offsets, s_offsets + 2, slave_ranges.begin()  + comm.rank());
   comm.all_reduce(&slave_ranges[0],  slave_ranges.size(), moonolith::MPIMax());


   moonolith::Redistribute< moonolith::SparseMatrix<double> > redist(comm.get());
   redist.apply(slave_ranges, mat_buffer, moonolith::AddAssign<double>());

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
      I[i] += I[i-1];
   }

   pmat = std::make_shared<HypreParMatrix>(comm.get(),
                                           s_offsets[1] - s_offsets[0],
                                           mat_buffer.rows(),
                                           mat_buffer.cols(),
                                           &I[0],
                                           &J[0],
                                           &data[0],
                                           s_offsets,
                                           m_offsets
                                          );

   return true;
}

ParMortarAssembler::ParMortarAssembler(
   const MPI_Comm comm,
   const std::shared_ptr<ParFiniteElementSpace> &master,
   const std::shared_ptr<ParFiniteElementSpace> &slave)
   : comm_(comm), master_(master), slave_(slave)
{ }


bool ParMortarAssembler::Assemble(std::shared_ptr<HypreParMatrix> &pmat)
{
   assert(!integrators_.empty() &&
          "it must have at least on integrator see class MortarIntegrator");

   moonolith::SearchSettings settings;
   // settings.set("disable_redistribution", moonolith::Boolean(true));
   // settings.set("disable_asynch", moonolith::Boolean(true));

   moonolith::Communicator comm(comm_);
   if (master_->GetMesh()->Dimension() == 2)
   {
      return mfem::Assemble<2>(comm, master_, slave_, integrators_, pmat, settings);
   }

   if (master_->GetMesh()->Dimension() == 3)
   {
      return mfem::Assemble<3>(comm, master_, slave_, integrators_, pmat, settings);
   }

   assert(false && "Dimension not supported!");
   return false;
}

bool ParMortarAssembler::Transfer(ParGridFunction &src_fun,
                                  ParGridFunction &dest_fun, bool is_vector_fe)
{
   using namespace std;
   static const bool verbose = false;
   static const bool dof_transformation = true;

   shared_ptr<HypreParMatrix> B = nullptr;

   moonolith::Communicator comm(comm_);

   if (verbose)
   {

      moonolith::root_describe(
         "--------------------------------------------------------"
         "Assembly begin: ", comm, mfem::out);

   }
   // moonolith::Clock c;

   if (!Assemble(B))
   {
      return false;
   }

   // c.tock();

   if (verbose)
   {
      moonolith::root_describe(
         "Assembly end: "
         "--------------------------------------------------------"
         , comm, mfem::out);
      // moonolith::root_describe(c, comm, mfem::out);
   }

   if (dof_transformation)
   {
      B.reset(RAP( slave_->Dof_TrueDof_Matrix(),
                   B.get(),
                   master_->Dof_TrueDof_Matrix() ));
   }

   ParBilinearForm b_form(slave_.get());
   if (is_vector_fe)
   {
      b_form.AddDomainIntegrator(new VectorFEMassIntegrator());
   }
   else
   {
      b_form.AddDomainIntegrator(new MassIntegrator());
   }

   b_form.Assemble();
   b_form.Finalize();

   shared_ptr<HypreParMatrix> Dptr(b_form.ParallelAssemble());


   comm.barrier();


   if (verbose && comm.is_root())
   {
      mfem::out << "P in R^(" << master_->Dof_TrueDof_Matrix()->Height();
      mfem::out << " x "      << master_->Dof_TrueDof_Matrix()->Width() << ")\n";
      mfem::out <<  "Q in R^(" << slave_->Dof_TrueDof_Matrix()->Height();
      mfem::out <<  " x "     << slave_->Dof_TrueDof_Matrix()->Width() << ")\n";
   }

   // if(dof_transformation) {
   //    Dptr.reset( RAP(Dptr.get(), slave_->Dof_TrueDof_Matrix()) );
   // }


   auto &D = *Dptr;

   comm.barrier();

   if (verbose && comm.is_root())
   {
      mfem::out <<  "--------------------------------------------------------"    <<
                std::endl;
      mfem::out <<  "B in R^(" << B->GetGlobalNumRows()    << " x " <<
                B->GetGlobalNumCols()    << ")" << std::endl;
      mfem::out <<  "D in R^(" << Dptr->GetGlobalNumRows() << " x " <<
                Dptr->GetGlobalNumCols() << ")" << std::endl;
      mfem::out <<  "--------------------------------------------------------"    <<
                std::endl;
   }

   comm.barrier();

   // B->Print("B.txt");
   // D.Print("D.txt");

   if (verbose)
   {
      Vector v(D.Width());
      v = 1.0;

      Vector Dv(D.Height());
      D.Mult(v, Dv);

      double sum_Dv = Dv.Sum();

      if (comm.is_root())
      {
         mfem::out <<  "sum(D): " << sum_Dv << std::endl;
      }
   }

   auto &P_master = *master_->Dof_TrueDof_Matrix();
   auto &P_slave  = *slave_->Dof_TrueDof_Matrix();

   CGSolver Dinv(comm.get());
   Dinv.SetOperator(D);
   Dinv.SetRelTol(1e-6);
   Dinv.SetMaxIter(20);
   // Dinv.SetPrintLevel(1);

   Vector P_x_src_fun(B->Width());
   P_master.MultTranspose(src_fun, P_x_src_fun);

   Vector B_x_src_fun(B->Height());
   B_x_src_fun = 0.0;

   B->Mult(P_x_src_fun, B_x_src_fun);

   Vector R_x_dest_fun(D.Height());
   R_x_dest_fun = 0.0;

   Dinv.Mult(B_x_src_fun, R_x_dest_fun);

   P_slave.Mult(R_x_dest_fun, dest_fun);

   dest_fun.Update();
   return true;
}
}
