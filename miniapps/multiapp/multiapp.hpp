// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.


#ifndef MFEM_MULTIAPP_HPP
#define MFEM_MULTIAPP_HPP

#include "mfem.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

/// Forward declarations needed below
class FieldTransfer;
class LinkedFields;
class LinkedFieldsCollection;
class Application;
class CoupledOperator;
class OperatorCoupler;



/**
   @brief Base class for transfering fields on meshes. The extension should
   implement the Transfer(const ParGridFunction&, ParGridFunction&).
 */
class FieldTransfer
{
protected:
    ParFiniteElementSpace *src_fes = nullptr, *tar_fes = nullptr;

public:
    /// Flag to determine the type of transfer
    enum Type {
        NATIVE,     ///< Native MFEM transfer using SubMesh ParTransferMap
        GSLIB       ///< GSLIB-based transfer using FindPointsGSLIB
    };

    /**
       @brief Construct a abstract FieldTransfer object

       @param src Source @a ParFiniteElementSpace
       @param tar Target @a ParFiniteElementSpace
     */
    FieldTransfer(ParFiniteElementSpace *src = nullptr, 
                  ParFiniteElementSpace *tar = nullptr) : 
                  src_fes(src), tar_fes(tar) {}

    virtual void Transfer(const ParGridFunction &src, ParGridFunction &tar) = 0;
    virtual ~FieldTransfer() {}

    /// Function for selecting the desired FieldTransfer scheme
    /// Returns a pointer to the selected FieldTransfer object
    /// Caller gets ownership is responsible for its deletion
    static MFEM_EXPORT FieldTransfer* Select(ParFiniteElementSpace *src,
                                             ParFiniteElementSpace *tar,
                                             Type type);    
};

/**
   @brief Native MFEM transfer between two FiniteElementSpaces 
   using SubMesh ParTransferMap.
 */
class NativeTransfer : public FieldTransfer
{
protected:
    ParTransferMap transfer_map;

public:

    /**
       @brief Construct a new NativeTransfer between 
       two @a ParFiniteElementSpace.
       
       @param src Source @a ParFiniteElementSpace
       @param tar Target @a ParFiniteElementSpace
     */
    NativeTransfer(ParFiniteElementSpace *src,
                   ParFiniteElementSpace *tar) :
                   FieldTransfer(src, tar),
                   transfer_map(*src_fes, *tar_fes){}
    NativeTransfer(ParFiniteElementSpace &src,
                   ParFiniteElementSpace &tar) :
                   NativeTransfer(&src, &tar){}

    /**
       @brief Construct a new NativeTransfer between 
       @a ParFiniteElementSpace of two @a ParGridFunction.
       @param src Source @a ParGridFunction
       @param tar Target @a ParGridFunction
     */
    NativeTransfer(ParGridFunction *src, ParGridFunction *tar) :
                   NativeTransfer(src->ParFESpace(), tar->ParFESpace()){}
    NativeTransfer(ParGridFunction &src, ParGridFunction &tar) :
                   NativeTransfer(src.ParFESpace(), tar.ParFESpace()){}

    /**
       @brief Transfer from source to target ParGridFunction.
       @param src Source @a ParGridFunction
       @param tar Target @a ParGridFunction
     */
    virtual void Transfer(const ParGridFunction &src, ParGridFunction &tar) override
    {
        transfer_map.Transfer(src, tar);
    }
};

/**
   @brief GSLib-based FindPoints transfer.
 */
class GSLibTransfer : public FieldTransfer
{
#ifdef MFEM_USE_GSLIB    
protected:
    ParMesh *src_mesh = nullptr; /// Source mesh (does not own)
    FindPointsGSLIB finder;      /// FindPointsGSLIB object
    Array<int> attr;             /// Attributes (optional) mesh markers for transfer
    Array<int> dofs;
    Array<int> tar_dofs;
    Vector coords;              /// Coordinates where the values are to be interpolated
    Vector interp_vals;         /// Interpolated values from the source mesh

public:
    /**
       @brief Constructor given a source mesh and coordinates to be transferred.
       @param src Source mesh (does not own)
       @param coordinates Coordinates where the values are to be interpolated
     */
    GSLibTransfer(ParMesh *src, Vector &coordinates) :
                 src_mesh(src), finder(src_mesh->GetComm())
                 {
                    SetCoordinates(coordinates);
                    finder.Setup(*src_mesh);
                    Update();
                 }
    
    /**
       @brief Constructor given a source and target ParFiniteElementSpace.

       @param src Source ParFiniteElementSpace
       @param tar Target ParFiniteElementSpace
       @param attr Attributes (optional) mesh markers for transfer
     */
    GSLibTransfer(ParFiniteElementSpace *src, ParFiniteElementSpace *tar,
                  Array<int> attr = Array<int>() ) : FieldTransfer(src, tar),
                  src_mesh(src_fes->GetParMesh()), finder(src_fes->GetComm()),
                  attr(attr)
                  {
                    ComputeCoordinates();
                    finder.Setup(*src_mesh);
                    Update();
                  }
    GSLibTransfer(ParFiniteElementSpace &src, ParFiniteElementSpace &tar,
                  Array<int> attr = Array<int>() ) :
                  GSLibTransfer(&src, &tar,attr){}

    /**
       @brief Constructor given a source and target ParGridFunction.
       
       @param src Source ParGridFunction
       @param tar Target ParGridFunction
       @param attr Attributes (optional) mesh markers for transfer
     */
    GSLibTransfer(ParGridFunction *src_gf, ParGridFunction *tar_gf,
                  Array<int> attr = Array<int>()) :
                  GSLibTransfer(src_gf->ParFESpace(), tar_gf->ParFESpace(), attr)
                  {}
    GSLibTransfer(ParGridFunction &src_gf, ParGridFunction &tar_gf,
                  Array<int> attr = Array<int>()):
                  GSLibTransfer(&src_gf, &tar_gf, attr){}

    /**
       @brief Set coordinates at which to interpolate.

       @param coordinates Coordinates where the values are to be interpolated
       @param update (optional) Whether to update the finder after setting new coordinates
     */
    void SetCoordinates(Vector &coordinates, bool update = true)
    {
        coords.SetDataAndSize(coordinates.GetData(), coordinates.Size());
        dofs.SetSize(coords.Size()/src_mesh->Dimension());
        std::iota(std::begin(dofs), std::end(dofs), 0);
        if(update) Update();
    }

    /**
       @brief Compute coordinates from the target ParFiniteElementSpaces.
     */
    void ComputeCoordinates()
    {
        ParMesh *mesh = tar_fes->GetParMesh();
        Vector nodes = mesh->GetNodes()->GetTrueVector();

        if(attr.Size()>0)
        {
            tar_fes->GetEssentialTrueDofs(attr,dofs,0);
        }
        else
        {
            dofs.SetSize(tar_fes->GetTrueVSize());
            std::iota(std::begin(dofs), std::end(dofs), 0);
        }

        int dim = mesh->Dimension();
        int nnodes = nodes.Size()/dim;
        int nstep = 1, vstep = 0;

        if(tar_fes->GetOrdering() == Ordering::byVDIM){
            nstep = 0;
            vstep = 1;
        }
        
        coords.SetSize(dofs.Size()*dim);

        for (int i=0; i<dofs.Size(); i++)
        {
            int idof = dofs[i];
            for (int d=0; d<dim; d++)
            {
                int idx = (idof + nnodes*d)*nstep + (idof*dim + d)*vstep;
                coords(i*dim + d) = nodes(idx);
            }
        }
    }

    /**
       @brief Updates FindPoints finder if coordinates have changed.
     */
    void Update()
    {
        tar_dofs.DeleteAll();
        finder.FindPoints(coords);

        const Array<unsigned int> &code_out = finder.GetCode();
        for (int i = 0; i < code_out.Size(); i++)
        {
            if (code_out[i] != 2) // Only keep points found in the mesh
            {
                tar_dofs.Append(dofs[i]);
            }
        }
    }

    /**
       @brief Transfer from source to target  @a ParGridFunction.

       @param src Source @a ParGridFunction
       @param tar Target @a ParGridFunction
     */
    virtual void Transfer(const ParGridFunction &src, ParGridFunction &tar) override
    {
        finder.Interpolate(src,interp_vals);

        int dim = tar_fes->GetVDim();
        int src_nstep = 1, src_vstep = 0;
        int tar_nstep = 1, tar_vstep = 0;
        int src_nnodes = interp_vals.Size()/dim;
        int tar_nnodes = tar.Size()/dim;

        if(src_fes->GetOrdering() == Ordering::byVDIM){ src_nstep = 0; src_vstep = 1; }        
        if(tar_fes->GetOrdering() == Ordering::byVDIM){ tar_nstep = 0; tar_vstep = 1; }

        for (int i = 0; i < tar_dofs.Size(); i++)
        {
            int idof = tar_dofs[i];
            for (int d=0; d<dim; d++)
            {
                int src_idx = (i + src_nnodes*d)*src_nstep + (i*dim + d)*src_vstep;
                int tar_idx = (idof + tar_nnodes*d)*tar_nstep + (idof*dim + d)*tar_vstep;
                tar(tar_idx) = interp_vals(src_idx);
            }
        }
    }

    virtual ~GSLibTransfer(){ 
        finder.FreeData();
    }
#else
public:
    GSLibTransfer(ParMesh *src, Vector &coordinates)
    {
        MFEM_ABORT("MFEM is not built with GSLIB support.");
    }
    GSLibTransfer(ParFiniteElementSpace *src, ParFiniteElementSpace *tar,
                  Array<int> attr = Array<int>() )
    {
        MFEM_ABORT("MFEM is not built with GSLIB support.");
    }
    GSLibTransfer(ParFiniteElementSpace &src, ParFiniteElementSpace &tar,
                  Array<int> attr = Array<int>() )
    {
        MFEM_ABORT("MFEM is not built with GSLIB support.");
    }
    GSLibTransfer(ParGridFunction *src_gf, ParGridFunction *tar_gf,
                  Array<int> attr = Array<int>())
    {
        MFEM_ABORT("MFEM is not built with GSLIB support.");
    }
    GSLibTransfer(ParGridFunction &src_gf, ParGridFunction &tar_gf,
                  Array<int> attr = Array<int>())
    {
        MFEM_ABORT("MFEM is not built with GSLIB support.");
    }
    void Transfer(const ParGridFunction &src, ParGridFunction &tar) override
    {
        MFEM_ABORT("MFEM is not built with GSLIB support.");
    }
#endif // MFEM_USE_GSLIB
};

/// @brief Base class for storing fields (GridFunctions) and distinguishing
/// between types of fields (e.g., source vs target).
class Field
{
private:
    MPI_Comm comm = MPI_COMM_NULL;

    bool is_source = false;
    bool is_target = false;

protected:
    ParGridFunction *gf = nullptr;

public:
    Field(ParGridFunction *gf_, bool is_src, bool is_tar) : 
          is_source(is_src), is_target(is_tar), gf(gf_)
    {
        comm = gf->ParFESpace()->GetComm();
    }

    ///@brief Get the stored ParGridFunction
    virtual ParGridFunction* GetGridFunction() { return gf; }

    ///@brief Set the internal ParGridFunction
    virtual void SetGridFunction(ParGridFunction *gf_) { gf = gf_; }

    virtual bool IsSource() const { return is_source; }
    virtual bool IsTarget() const { return is_target; }
    virtual ~Field() {}
};

/// @brief Class for storing a ParGridFunction acting as a source field
class SourceField : public Field
{
public:
    SourceField(ParGridFunction *gf_) : Field(gf_, true, false){}
    ~SourceField() {}
};

/// @brief Class for storing a ParGridFunction acting as a target field
class TargetField : public Field
{
protected:
    FieldTransfer *transfer_map = nullptr;
    bool own_map = false;

public:
    TargetField(ParGridFunction *gf_, FieldTransfer *map, bool own=false) :
                Field(gf_, false, true), transfer_map(map), own_map(own) {}

    TargetField(ParGridFunction *gf_) : TargetField(gf_, nullptr, false) {}

    ///@brief Get the FieldTransfer associated with this target field
    virtual FieldTransfer* GetTransferMap() const { return transfer_map; }

    ~TargetField()
    {
        if(own_map && transfer_map) delete transfer_map;
    }
};


/**
   @brief A class to handle variables and transfers between sources and multiple
   target ParGridFunctions.
 */
class LinkedFields 
{
public:
    using Type = FieldTransfer::Type;
    using Pair = std::pair<TargetField*, bool>;
    Type default_transfer_type = Type::NATIVE;

protected:

    SourceField *source = nullptr;
    ParGridFunction *src_gf = nullptr;
    bool own_source = false;

    std::vector<Pair> targets;
    int ndest = 0;

public:

    LinkedFields() = default;

    /**
       @brief Constructor given a source @a ParGridFunction.

       @param src Source @a ParGridFunction
     */
    LinkedFields(ParGridFunction *src) : src_gf(src)
    {
        source = new SourceField(src_gf);
        own_source = true;
    }

    /**
     * @brief Construct a new LinkedFields with only a source and empty target
     */
    LinkedFields(SourceField *src, bool own=false) : source(src)
    {
        src_gf = source->GetGridFunction();
        own_source = own;
    }


    /**
       @brief Constructor given a source and target @a ParGridFunction.

       @param src Source  @a ParGridFunction
       @param tar Target  @a ParGridFunction
       @param transfer_map  @a FieldTransfer from src to tar (not owned)
     */
    LinkedFields(ParGridFunction *src, ParGridFunction *tar, 
                 FieldTransfer *transfer_map = nullptr) : LinkedFields(src)
    {
        AddTarget(tar, transfer_map);
    }

    /**
       @brief Set the source @a SourceField.

       @param src Source  @a SourceField
     */
    void SetSource(SourceField *src, bool own=false)
    {
        if(own_source && source) delete source;
        source = src;
        own_source = own;
        src_gf = source->GetGridFunction();
    }

    ///@brief Set the source @a ParGridFunction (does not own).
    void SetSource(ParGridFunction *src)
    {
        if(own_source && source) delete source;
        source = new SourceField(src);
        own_source = true;
        src_gf = source->GetGridFunction();
    }

    ///@brief Get the source @a ParGridFunction
    ParGridFunction *GetSource() const { return src_gf;}

    /**
       @brief Adds the @a ParGridFunction, @a tar, with a specified 
       FieldTransfer operator, @a transfer_map, to the list of targets
     */
    void AddTarget(ParGridFunction *tar, FieldTransfer *transfer_map = nullptr)
    {
        FieldTransfer *map = transfer_map;
        if(!map) map = FieldTransfer::Select( src_gf->ParFESpace(), tar->ParFESpace(),
                                              default_transfer_type);
        TargetField *target = new TargetField(tar, map, true);
        AddTarget(target, true);
    }

    ///@brief Adds the @a TargetField, @a tar, to the list of targets
    void AddTarget(TargetField *tar, bool own=false)
    {
        targets.push_back(std::make_pair(tar, own));
        ndest++;
    }

    ///@brief Get all target fields
    const std::vector<Pair>& GetTargets() const { return targets; }

    /**
       @brief Transfer from source to all the @a idest destination. 
       If the source vector is the same size.

       @param vsrc the source vector to transfer from; if the size 
       matches the source ParGridFunction, it is used to set the GridFunction.
       @param idest the index of the destination to transfer to;
       if <0 (default) transfer to all targets.
     */
    virtual void Transfer(const Vector &vsrc, const int idest=-1)
    {
        if(vsrc.Size() == src_gf->ParFESpace()->GetTrueVSize() )
        {   // Set GridFunction if the source vector is the same size.
            src_gf->SetFromTrueDofs(vsrc);
        }
        Transfer(idest);
    }

    /**
       @brief Transfer internally stored ParGridFunction from source to all the
       @a idest destination; if @a idest <0 (default) transfer to all targets
     */
    virtual void Transfer(const int idest = -1)
    {
        MFEM_VERIFY(idest < ndest, "Invalid destination index " 
                    << idest << " >= " << ndest);

        if(idest >= 0)
        {
            auto [target, owned] = targets[idest];
            FieldTransfer *transfer_map = target->GetTransferMap();
            ParGridFunction *tar_gf = target->GetGridFunction();
            transfer_map->Transfer(*src_gf,*tar_gf);
        }
        else
        {
            for (auto &destination : targets)
            {
                auto [target, owned] = destination;
                FieldTransfer *transfer_map = target->GetTransferMap();
                ParGridFunction *tar_gf = target->GetGridFunction();
                transfer_map->Transfer(*src_gf,*tar_gf);
            }
        }
    }

    virtual ~LinkedFields()
    {
        for (auto &dest : targets)
        {
            auto [target, owned] = dest;
            if(owned) delete target;
        }
        if(own_source && source) delete source;
    }
};

/// @brief A collection of LinkedFields, each identified by a name
class LinkedFieldsCollection
{
private:
   MPI_Comm comm = MPI_COMM_NULL;

    std::string name; /// Name of the collection
    Application *src_op = nullptr; /// Source Application (not owned)

    /// Fields for source Application. Contains all source fields and fields that
    /// may be targets of other applications.
    NamedFieldsMap<ParGridFunction> fields;

    /// LinkedFields for source Application.
    NamedFieldsMap<LinkedFields> linked_fields;

public:
    LinkedFieldsCollection(MPI_Comm comm_, std::string collection_name = "",
                           Application *src = nullptr) : name(collection_name),
                           src_op(src) {comm = comm_;}

    LinkedFieldsCollection() = default;

    /// @brief Constructor with collection name and optional source Application
    LinkedFieldsCollection(std::string collection_name,
                           Application *op = nullptr):
                           name(collection_name), 
                           src_op(op){}

    /// @brief Constructor with source Application
    LinkedFieldsCollection(Application *src):src_op(src){}

    /// @brief Get the number of linked fields in the collection
    int Size() const { return linked_fields.NumFields(); }

    /// @brief Set the name of the collection
    void SetName(const std::string &collection_name){ name = collection_name;}

    /// @brief Get the name of the collection
    std::string GetName() const { return name; }

    /// @brief Set the source Application
    void SetSource(Application *src){ src_op = src; }

    /// @brief Get the source Application
    const Application* GetSource() const { return src_op; }

    /// @brief Get the ParGridFunction for a given source name
    ParGridFunction *GetSourceField(const std::string &src_name) const
    {
        LinkedFields *lf = linked_fields.Get(src_name);
        if(!lf)
        {
            // MFEM_WARNING("LinkedFieldsCollection::GetSourceField: Source field "
            //              + src_name + " not found!");
            return nullptr;
        }
        return lf->GetSource();
    }

    /// @brief Get the ParGridFunction for a given field name
    ParGridFunction* GetField(const std::string &field_name) const
    {
        return fields.Get(field_name);
    }

    /// @brief Add a ParGridFunction as a field (does not specify source or target)
    void AddField(const std::string &field_name,
                  ParGridFunction *field_gf)
    {
        fields.Register(field_name, field_gf, false);
    }

    /// @brief Add a LinkedFields to the collection with name src_name
    void AddLinkedFields(const std::string &src_name,
                         LinkedFields *lf, bool own = false)
    {
        LinkedFields *lf_exist = linked_fields.Get(src_name);
        if(lf_exist)
        {
            for (auto &dest : lf->GetTargets()) {
                auto [target, owned] = dest;
                lf_exist->AddTarget(target, owned);
            }
            return;
        }
        linked_fields.Register(src_name, lf, own);
        fields.Register(src_name, lf->GetSource(), false);
    }

    /// @brief Add a source ParGridFunction to the collection. If src_name does not
    /// exist, a new LinkedFields is created and owned.
    void AddSourceField(const std::string &src_name, ParGridFunction *src_gf)
    {
        fields.Register(src_name, src_gf, false);        
        LinkedFields *lf = linked_fields.Get(src_name);
        if(!lf)
        {
            LinkedFields *lf = new LinkedFields(src_gf);
            linked_fields.Register(src_name, lf, true);
            return;
        }
        lf->SetSource(src_gf);        
    }

    /// @brief Add a target ParGridFunction (with an optional FieldTransfer) 
    /// to the source named src_name. If src_name does not exist, a new 
    /// LinkedFields is created and owned.
    void AddTargetField(const std::string &src_name,
                        ParGridFunction *tar_gf,
                        FieldTransfer *transfer_map = nullptr)
    {
        LinkedFields *lf = linked_fields.Get(src_name);
        if(!lf)
        {
            lf = new LinkedFields();
            linked_fields.Register(src_name, lf, true);
        }
        lf->AddTarget(tar_gf, transfer_map);
    }

    /// @brief Transfer from source named src_nameto target named tar_name. The source 
    /// GridFunction is set from vsrc. If src_name is empty and there is only one
    /// source in the collection, it is used.
    void Transfer(const std::string &src_name, const Vector &vsrc,
                  const std::string &tar_name = "")
    {
        std::string sname = src_name;
        if(sname.empty() && Size()>1)
        {
            MFEM_ABORT("LinkedFieldsCollection::Transfer: Source name is empty "
                       "and there are multiple source fields! Specify source name.");
        }
        else if(sname.empty() && Size()==1)
        {
            sname = linked_fields.begin()->first;
        }

        LinkedFields *lf = linked_fields.Get(sname);
        if(!lf)
        {
            MFEM_ABORT("LinkedFieldsCollection::Transfer: Source field "
                         + sname + " in collection " + name + " not found!");
        }
        lf->Transfer(vsrc);
    }

    /// @brief Transfer from source named src_name to target named tar_name. If
    /// src_name is empty, transfer from all sources, with internally stored 
    /// GridFunctions, to all targets.
    void Transfer(const std::string &src_name = "",
                  const std::string &tar_name = "")
    {
        if(src_name.empty())
        {
            for (auto &field : linked_fields)
            {
                auto [name, lf] = field;
                lf->Transfer();
            }
        }
        else
        {
            LinkedFields *lf = linked_fields.Get(src_name);
            if(!lf)
            {
                MFEM_ABORT("LinkedFieldsCollection::Transfer: Source field "
                            + src_name + " in collection " + name + " not found!");
            }
            // lf->Transfer(tar_name=="" ? -1 : linked_fields.GetIndex(tar_name));
            lf->Transfer();
        }
    }

    ~LinkedFieldsCollection(){}

};


/** @brief This class is used to define the interface for applications and miniapps.
 */
class Application : public TimeDependentOperator
{
public: 
    /**
       @brief Enum used to keep track of the type of operator
     */
    enum OperatorType {
        ANY_TYPE,       ///< Any MFEM Operator
        MFEM_TDO,       ///< MFEM TimeDependentOperator
        MFEM_SOLVER,    ///< MFEM Solver
        MFEM_ODESOLVER, ///< MFEM ODESolver
        NOT_MFEM_OBJECT ///< Object is not derived from an MFEM class 
    };

    /**
       @brief Used to determine which operation shoudl be performed by 
       PerformOperation(const int, const Vector &, Vector&).
     */
    struct Operations
    {
        enum OperationID {
            MULT, IMPLICIT_SOLVE, STEP, 
            IMPLICIT_MULT, EXPLICIT_MULT,
            SOLVE, DEFAULT, NONE
        };
    };
    using OperationID = Operations::OperationID;

protected:

    std::string name;
    int oper_index = std::numeric_limits<int>::max();
    OperatorType operator_type = OperatorType::ANY_TYPE; ///< Type of the operator
    OperationID operation_id = OperationID::NONE;        ///< Current operation ID
    bool is_coupled = false;

    std::function<void (Vector&)> pre_process_func; ///< Pre-process function
    std::function<void (Vector&)> post_process_func; ///< Post-process function

    LinkedFieldsCollection field_collection; ///< Collection of linked fields

public:
    /**
       @brief Construct a new Application object.
       @param n Size of the operator
     */
    Application(int n=0) : TimeDependentOperator(n) {}

    /**
       @brief Construct a new Application object.
       @param h Height of the operator
       @param w Width of the operator
     */
    Application(int h, int w) : TimeDependentOperator(h,w) {}

    /**
       @brief Initialize the operator.
     */
    virtual void Initialize()
    {
        mfem_error("Application::Initialize() is not overridden!");
    }

    /**
       @brief Assemble the operator.
     */
    virtual void Assemble()
    {
        mfem_error("Application::Assemble() is not overridden!");
    }

    /**
       @brief Finalize the operator.
     */
    virtual void Finalize()
    {
        mfem_error("Application::Finalize() is not overridden!");
    }

    /**
       @brief Set the index of the operator.
     */
    void SetOperatorIndex(int index){ oper_index = index; }

    /**
       @brief Get the index of the operator.
     */
    int GetOperatorIndex() const { return oper_index; }

    /**
       @brief Set the Operation ID to call the appropriate operation
       from Mult().
     */
    virtual void SetOperationID(OperationID id){ operation_id = id; }

    /**
       @brief Get the current OperationID.
     */
    OperationID GetOperationID() const { return operation_id; }

    /**
       @brief Set the Operator Type object. 
     */
    void SetOperatorType(OperatorType type) { operator_type = type; }

    /**
       @brief Get the Operator Type.
     */
    OperatorType GetOperatorType() const { return operator_type; }

    /**
       @brief Returns true if Application is coupled.
     */
    virtual bool IsCoupled() const {return is_coupled;}

    /**
       @brief Set whether the Application is coupled.
     */
    virtual void SetCoupled(bool coupled) {is_coupled = coupled;}

    /**
       @brief Set the pre-processing function.

       @param func Pre-processing function

       @note This is primarily used for type-erased objects
     */
    virtual void SetPreProcessFunction(std::function<void (Vector&)> func)
    {
        pre_process_func = std::move(func);
    }

    /**
       @brief Set the post-processing function.
       
       @param func Post-processing function
       
       @note This is primarily used for type-erased objects
     */
    virtual void SetPostProcessFunction(std::function<void (Vector&)> func)
    {
        post_process_func = std::move(func);
    }

    /**
       @brief Pre-process the input vector @a x before performing operations such as
       Mult, Solve, ImplicitSolve, Step, etc. when coupled with other operators.
       
       @note This method is always called before the main operation is performed, but must
       be either overridden or a pre-processing lambda shoudl be set, to have an effect.
       Currently, it does nothing.
     */
    virtual void PreProcess(Vector &x){
        if(pre_process_func) pre_process_func(x);
    }

    /**
       @brief Post-process the input vector @a x after performing operations such as
       Mult, Solve, ImplicitSolve, Step, etc. when coupled with other operators.
       
       @note This method is always called before the main operation is performed, but must
       be either overridden or a post-processing lambda shoudl be set, to have an effect.
       Currently, it does nothing.
     */
    virtual void PostProcess(Vector &x){
        if(post_process_func) post_process_func(x);
    }

    /**
       @brief Solve the problem defined by the operator, given an input @a x 
       and return the solution in @a y.
     */
    virtual void Solve(const Vector &x, Vector &y) const
    {
        MFEM_ABORT("Not implemented for this Application.");
    };

    /**
       @brief Apply the operator the vector @a x and return the result in @a y.
       For @a TimeDependentOperator, this computes (u,t) -> k(u,t).
     */
    virtual void Mult(const Vector &x, Vector &y) const override
    {
        MFEM_ABORT("Not implemented for this Application.");
    }

    /**
       @brief Update the operator state. 
     */
    virtual void Update()
    {
        mfem_error("Application::Update() is not overridden!");
    }
    
    /**
       @brief Perform an operation index by @a id on the vector
       @a x and return the result in @a y.
     */
    virtual void PerformOperation(const int id, const Vector &x, Vector &y)
    {
        mfem_error("Application::PerformOperation() is not overridden!");
    }

    /**
       @brief Get the LinkedFieldsCollection for this Application.
     */
    LinkedFieldsCollection& Fields() { return field_collection; }

    /**
       @brief Add a LinkedField @a field to the operator.

       @note The source field in the @a LinkedField should
       be from the current operator.
     */
    virtual void AddLinkedFields(const std::string &src_name, LinkedFields *field)
    {
        field_collection.AddLinkedFields(src_name, field);
    }

    /**
       @brief Perform a time step from time @a t [in] to time @a t [out] based
       on the requested step size @a dt [in].
       
       @note This function is called if the operator acts as an ODE Solver.
       
       @param x Approximate initial solution
       @param t Time associated with the approximate solution @a x
       @param dt Time step size
     */
    virtual void Step(Vector &x, real_t &t, real_t &dt)
    {
        mfem_error("Application::Step() is not overridden!");
    }

    /**
       @brief Transfer the internally stored data defined in the LinkedFields
       for current operator.
     */
    virtual void Transfer()
    {
        field_collection.Transfer();
    }

    /**
       @brief Transfer the data defined in the LinkedFields for current operator.

       @param x Information to transfer, e.g., current solution vector, stage update, etc.
     */
    virtual void Transfer(const Vector &x)
    {
        if(field_collection.Size() == 1)
        {
            field_collection.Transfer("", x);
        }
        else
        {
            Application::Transfer();
        }
    }

    /**
       @brief Transfer the stage 'u' and 'k' from multistage methods. Can be
       used to transfer the stage-updated solution, e.g., un = ui + dt*ki,
       to other applications.
     */
    virtual void Transfer(const Vector &u, const Vector &k, real_t dt = 0.0)
    {
        Application::Transfer(u);
    }
};


/**
   @brief An abstract, type-erased class to define the interface for 
   operators (not inherited from @a Application) to be used
   with applications. It performs SFINAE checks for stored operator's
   member functions and override the Mult, ImplicitSolve and Step methods
   to call the stored object's functions.
 */
template <typename OpType>
class AbstractOperator : public Application
{
protected:
    /// Define a template class 'check' to test for the existence of member functions
    template <typename C>
    class CheckMember{
        private:

        /// @brief A type trait to check if the erased class has the functions Step, Mult, and ImplicitSolve
        /// with the needed signatures.
        template<class T>
        using Step = decltype(std::declval<T&>().Step(std::declval<Vector&>(),
                                                      std::declval<real_t&>(),
                                                      std::declval<real_t&>()));

        template<class T>
        using StepPtr = decltype(std::declval<T&>().Step(std::declval<const int>(),
                                                         std::declval<real_t*>(),
                                                         std::declval<real_t&>(),
                                                         std::declval<real_t&>()));

        template<class T>
        using Mult = decltype(std::declval<T&>().Mult(std::declval<const Vector&>(),
                                                      std::declval<Vector&>()));

        template<class T>
        using MultPtr = decltype(std::declval<T&>().Mult(std::declval<const int>(),
                                                         std::declval<const real_t*>(),
                                                         std::declval<const int>(),
                                                         std::declval<real_t*>()));

        template<class T>
        using ImplicitSolve = decltype(std::declval<T&>().ImplicitSolve(std::declval<const real_t>(),
                                                                        std::declval<const Vector&>(),
                                                                        std::declval<Vector&>()));

        template<class T>
        using ImplicitSolvePtr = decltype(std::declval<T&>().ImplicitSolve(std::declval<const real_t>(),
                                                                           std::declval<const int>(),
                                                                           std::declval<const real_t*>(),
                                                                           std::declval<const int>(),
                                                                           std::declval<real_t*>()));

        // ---------------------------------------------------------------------
        
        template <typename T, template<typename> typename Func, typename R>
        static constexpr auto Check(T*) -> typename std::is_same< Func<T>, R>::type;

        template <typename, template<typename> typename, typename >
        static constexpr std::false_type Check(...);

        // --- Check for the existence of the member functions
        typedef decltype(Check<C,Mult,void>(0)) Has_Mult;
        typedef decltype(Check<C,Step,void>(0)) Has_Step;
        typedef decltype(Check<C,ImplicitSolve,void>(0)) Has_ImplicitSolve;

        typedef decltype(Check<C,MultPtr,void>(0)) Has_MultPtr;
        typedef decltype(Check<C,StepPtr,void>(0)) Has_StepPtr;
        typedef decltype(Check<C,ImplicitSolvePtr,void>(0)) Has_ImplicitSolvePtr;

    public:
        static constexpr bool HasMult  = Has_Mult::value;
        static constexpr bool HasStep  = Has_Step::value;
        static constexpr bool HasImplicitSolve  = Has_ImplicitSolve::value;
        static constexpr bool HasMultPtr  = Has_MultPtr::value;
        static constexpr bool HasStepPtr  = Has_StepPtr::value;
        static constexpr bool HasImplicitSolvePtr  = Has_ImplicitSolvePtr::value;
    };    

    OpType *op;  ///< Pointer to the operator
    Application *nested_op = nullptr; ///< Pointer to the nested operator, if any

public:

    constexpr bool HasMult(){return CheckMember<OpType>::HasMult;}
    constexpr bool HasStep(){return CheckMember<OpType>::HasStep;}
    constexpr bool HasImplicitSolve(){return CheckMember<OpType>::HasImplicitSolve;}

    /// @brief Constructor for the type-erased AbstractOperator class
    AbstractOperator(OpType *op_, int h, int w) : Application(h,w), op(op_)
    {
        // If the operator is an ODESolver, we need access to its time-dependent operator 
        if constexpr (std::is_base_of<ODESolver, OpType>::value)
        {
            TimeDependentOperator* tdo = op->GetTimeDependentOperator();
            this->height = tdo->Height();
            this->width  = tdo->Width();
            nested_op = dynamic_cast<Application*>(tdo);
        }
    }

    /// @brief Constructor for the type-erased AbstractOperator class.
    AbstractOperator(OpType *op_, int s = 0) : AbstractOperator(op_,s,s) {}

    /**
       @brief Perform Mult operation with the stored operator, if it exists.
     */
    void Mult(const Vector &x, Vector &y) const override
    {
        if constexpr (CheckMember<OpType>::HasMult)
        {
            op->Mult(x,y);
        }
        else if constexpr (CheckMember<OpType>::HasMultPtr)
        {
            op->Mult(x.Size(), x.GetData(), y.Size(), y.GetData());
        }
        else
        {
            MFEM_ABORT("The AbstractOperator does not have the function, "
                       "Mult(const Vector&, Vector&) or "
                       "Mult(int, double*, int, double*).");
        }
    }

    /**
       @brief Perform Step operation with the stored operator, if it exists.
     */
    void Step(Vector &x, real_t &t, real_t &dt) override
    {
        if constexpr (CheckMember<OpType>::HasStep)
        {
            op->Step(x,t,dt);
        }
        else if constexpr (CheckMember<OpType>::HasStepPtr)
        {
            op->Step(x.Size(), x.GetData(), t, dt);
        }
        else
        {
            MFEM_ABORT("The AbstractOperator does not have the function, "
                       "Step(Vector&, real_t&, real_t&) or "
                       "Step(int, double*, double&, double&).");
        }
    }

    /**
       @brief Perform ImplicitSolve operation with the stored operator, if it exists.
     */
    void ImplicitSolve(const real_t dt, const Vector &x, Vector &k) override
    {
        if constexpr (CheckMember<OpType>::HasImplicitSolve)
        {
            op->ImplicitSolve(dt,x,k);
        }
        else if constexpr (CheckMember<OpType>::HasImplicitSolvePtr)
        {
            op->ImplicitSolve(dt,x.Size(),x.GetData(),k.Size(),k.GetData());
        }
        else
        {
            MFEM_ABORT("The AbstractOperator does not have the function, "
                       "ImplicitSolve(const real_t , const Vector&, Vector&) or "
                       "ImplicitSolve(const real_t , int, double*, int, double*).");
        }
    }

    /**
       @brief Set the Operation ID to call the appropriate operation.
     */
    void SetOperationID(OperationID id) override
    {
        Application::SetOperationID(id);
        if (nested_op) nested_op->SetOperationID(id);
    }

    /**
       @brief Transfer the data defined in the LinkedFields for current operator.

       @param x Information to transfer, e.g., current solution vector, stage update, etc.
     */
    void Transfer(const Vector &x) override
    {
        if (nested_op && (field_collection.Size() == 0))
        {   // If this app does not have it's own linked_fields
            // call the nested_op's transfer
            nested_op->Transfer(x);
        }
        else
        {
            Application::Transfer(x);
        }
    }

    /**
       @brief Transfer the stage 'u' and 'k' from multistage methods. Can be used to transfer
       the stage-updated solution, e.g., un = ui + dt*ki, to other applications.
     */
    void Transfer(const Vector &u, const Vector &k, real_t dt = 0.0) override
    {
        if (nested_op && (field_collection.Size() == 0)) 
        {   // If this app does not have it's own linked_fields
            // call the nested_op's transfer
            nested_op->Transfer(u,k,dt);
        }
        else
        {
            Application::Transfer(u,k,dt);
        }
    }

    void Transfer() override
    {
        if (nested_op && (field_collection.Size() == 0)) 
        {   // If this app does not have it's own linked_fields
            // call the nested_op's transfer
            nested_op->Transfer();
        }
        else
        {
            Application::Transfer();
        }
    }
};


/**
   @brief A class to store and coupled multiple operators together.
 */
class CoupledOperator : public Application
{
public:
    /**
       @brief Flags to to define the coupling type between operators
     */
    struct CouplingTypes
    {
        enum class Types
        {
            NONE,                ///< No coupling, solve each operator independently
            MONOLITHIC,          ///< Solve all operators simultaneously
            ADDITIVE_SCHWARZ,    ///< Jacobi-type coupling
            ALTERNATING_SCHWARZ  ///< Gauss-Seidel-type coupling
        };
    };
    using Scheme = CouplingTypes::Types;

protected:
    std::vector<Application*> operators;  ///< List of individual operators
    Array<bool> operators_owned; ///< Whether the operators are owned
    Array<int> offsets;  ///< Block offsets for each operator
    int max_op_size=0;   ///< Largest operator size
    int nops = 0;        ///< The number of applications

    /// Operator for coupling type
    Scheme coupler_type = Scheme::NONE; ///< Current coupling type
    OperatorCoupler *op_coupler = nullptr;
    bool own_op_coupler = true;
    
    /// Solver for the coupling operator - For example: FPISolver for 
    /// partitioned and Newton/Krylov for monolithic
    Solver *solver = nullptr; 
    bool own_solver = false;

    mutable Vector b;

public:
    /**
       @brief Construct a new CoupledOperator object.
       @param nop Total number of operators to couple
     */
    CoupledOperator(const int nop) : Application()
    {
        operators.reserve(nop);
        offsets.Reserve(nop+1);
        offsets.Prepend(0);
        operators_owned.Reserve(nop);
    }

    /**
       @brief Construct a new CoupledOperator object for an 
       abstract non/mfem operator.
     */
    template <class OpType>
    CoupledOperator(const OpType &op) : CoupledOperator(1)
    {
        AddOperator(op);
    }

    /**
       @brief Add an operator to the list of coupled operator and
       return pointer to it. Not owned unless it's not derived from Application.
     */
    template <class OpType>
    Application* AddOperator(OpType *op_, int h, int w)
    {
        // Add operator to list of operators
        if constexpr(std::is_base_of<Application, OpType>::value)
        {
            operators.push_back(op_);
            operators_owned.Append(false);
        } 
        else
        {
            operators.push_back(new AbstractOperator<OpType>(op_,h,w));
            operators_owned.Append(true);
        }
        nops++;

        // Update size of the coupled operator and the block offsets
        Application* op = operators.back();
        op->SetOperatorIndex(nops-1); // Set the index of the operator

        int sum = offsets.Last();
        offsets.Append(sum + op->Width());

        this->width  += op->Width();
        this->height += op->Height();

        max_op_size = std::max(max_op_size, op->Width()); // Largest operator

        return op;
    }

    /// @brief Add an operator to the list of coupled operator and return pointer to it.
    template <class OpType>
    Application* AddOperator(OpType *op_, int s = 0) { return AddOperator(op_,s,s);}

    /// @brief Get the number of coupled operators
    int Size(){return nops;}

    /// @brief Get the size of the largest operator
    int Max(){return max_op_size;}

    /// @brief Get the operator at index @a i
    Application* GetOperator(const int i) { return operators[i]; }

    /// @brief Specify whether the operator at index @a i is owned.
    void OwnOperator(const int i, bool own = true)
    {
        MFEM_ASSERT(i >= 0 && i < nops,
               "index [" << i << "] is out of range [0," << nops << ")");
        operators_owned[i] = own;
    }

    /**
       @brief Set the operator coupling type.
       @note Currently supported options are provided in enum Scheme
     */
    virtual void SetCouplingScheme(Scheme type_) { coupler_type = type_; }


    /// @brief Get the current operator coupling type.
    Scheme GetCouplingScheme() const { return coupler_type; }

    /**
       @brief Set the offset used by BlockVector for the coupled operator.
       Checks if block offsets are consistent with operator sizes.

       @param off_sets Array of block offsets
     */
    void SetBlockOffsets(Array<int> off_sets)
    {
        bool same_size  = (off_sets.Size() == static_cast<int>(operators.size()));
        bool consistent = !same_size ? false : std::equal(
                                               operators.begin(), operators.end(), off_sets.begin(), 
                                               [](Application *a, int b){ return a->Width() == b;} 
                                               );
        if(!consistent)
        {
            // MFEM_WARNING("Inconsistent block offsets provided "
            //              "to CoupledOperator::SetBlockOffsets(). "
            //              "Using default offsets.");
            return;
        }
        this->offsets = off_sets;
    }

    /// @brief Return the offset used by BlockVector for the coupled operator.
    const Array<int> GetBlockOffsets(){ return offsets; }

    /**
       @brief Set the solver used for fixed point iterations or Newton solver in
       the implicit solve.

       @param own If 'true', own @a s
     */
    void SetSolver(Solver *s, bool own = false)
    { 
        if(own_solver && solver) delete solver;
        solver = s; own_solver = own;
    }

    /// @brief Get the current OperatorCoupler.
    const OperatorCoupler* GetOperatorCoupler(){ return op_coupler; }

    /**
       @brief Set the OperatorCoupler corresponding to the coupling type.

       @param op OperatorCoupler to use
       @param own If 'true', own @a op
     */
    void SetOperatorCoupler(OperatorCoupler* op, bool own=false);

    /**
       @brief Initialize the CoupledOperator.

       @param do_ops If 'true', call Initialize on all operator
     */
    virtual void Initialize(bool do_ops);
    void Initialize() override { Initialize(true); }

    /**
       @brief Assemble the CoupledOperator.

       @param do_ops If 'true', call Assemble on all operator
     */
    virtual void Assemble(bool do_ops);
    void Assemble() override { Assemble(true); }

    /**
       @brief Finalize the CoupledOperator.

       @param do_ops If 'true', call Finalize on all operator
     */
    virtual void Finalize(bool do_ops);
    void Finalize() override { Finalize(true); }

    /**
       @brief Pre-process the Vector @a x before an operation
       (e.g., Mult, ImplictSolve, etc.)
       
       @note See Application::PreProcess(Vector&) for when this can be used
       
       @param do_ops If 'true', call PreProcess for all operators
     */
    virtual void PreProcess(Vector &x, bool do_ops);
    void PreProcess(Vector &x) override { PreProcess(x, true); }

    /**
       @brief Post-process the Vector @a x after an operation
       (e.g., Mult, ImplictSolve, etc.)
       
       @note See Application::PostProcess(Vector&) for when this can be used
       
       @param do_ops If 'true', call PostProcess for all operators
     */
    virtual void PostProcess(Vector &x, bool do_ops);
    void PostProcess(Vector &x) override { PostProcess(x, true); }

    /// @brief Set the OperationID 
    virtual void SetOperationID(OperationID id, bool do_ops);
    void SetOperationID(OperationID id) override { SetOperationID(id, true); }

    /// @brief Transfer Vector @a x to operators via LinkedFields    
    virtual void Transfer(const Vector &x) override;

    /**
       @brief Transfer the stage 'u' and 'k' from multistage methods. Can be used 
       to transfer the stage-updated solution, e.g., un=ui+dt*ki, to other applications.
     */
    virtual void Transfer(const Vector &u, const Vector &k, real_t dt = 0.0) override;

    /// @brief Set the time for each operator.
    void SetTime(const real_t t_) override;

    /**
       @brief Perform operation, defined by @a op, on @a x and return in @a y.
       @note @a id is typically selected from enum Appliction::OperationID
     */
    void PerformOperation(const int id, const Vector &x, Vector &y)
    {
        for (auto &op : operators) {
            op->PerformOperation(id, x, y);
        }
    }

    /**
       @brief Apply the operator to the vector @a x 
       and return the result in @a y.
     */
    virtual void Mult(const Vector &x, Vector &y) const override;

    /**
       @brief Advance the time step for each operator. \Phi_i(x_i; t, dt) = x_i(t) -> x_i(t+dt).
       @note This is used when the operator to be coupled are of type ODESolver.
     */
    void Step(Vector &x, real_t &t, real_t &dt) override;

    /**
       @brief Solve for the unknown k, at the current time t, the following 
       equation: F(u + gamma k, k, t) = G(u + gamma k, t). The solution procedure 
       is determinied by the coupling type.
     */
    void ImplicitSolve(const real_t dt, const Vector &x, Vector &k ) override;

    /**
       @brief Perform the action of the implicit part of all operators, 
       F_i: v_i = F(u_i, k_i, t), where t is the current time. This function 
       performs an implicit solve for the monolithic system
     */
    void ImplicitMult(const Vector &u, const Vector &k, Vector &v) const override;

    /**
       @brief Perform the action of the explicit part of all operators, 
       G_i: v_i = G(u_i, t), where t is the current time.
     */
    void ExplicitMult(const Vector &u, Vector &v) const override;
    
    
    /// @brief Destroy the Coupled Application object
    ~CoupledOperator();
};



/**
   @brief Base class for coupling schemes between multiple operators.
 */
class OperatorCoupler : public TimeDependentOperator
{
public:
    using OperationID = Application::Operations::OperationID;
    using Scheme = CoupledOperator::CouplingTypes::Types;
private:
    Scheme type_ = Scheme::NONE;

protected:
    mutable CoupledOperator *coupled_op;
    mutable const Vector *input = nullptr; ///< Pointer to the input vector, used in ImplicitSolve()
    mutable real_t timestep = 0.0;               ///< Time step size, used for time-dependent applications
    OperationID operation_id = OperationID::NONE;  ///< Current operation ID

public:
    /**
       @brief Construct a new OperatorCoupler.
       @param op Pointer to the CoupledOperator; does not own the pointer.
     */
    OperatorCoupler(CoupledOperator *op) : TimeDependentOperator(op->Height(),op->Width()),
                                              coupled_op(op){}

    /**
       @brief Set the time step size, @a dt, used in ImplicitSolve and Step methods.
     */
    virtual void SetTimeStep(real_t dt){ timestep = dt;}

    /**
       @brief Set the input vector, @a inp, used in ImplicitSolve(dt,x,k) method.
       This is used when OperatorCoupler::Mult() const is called from Solver::Mult() to
       solver for @a k. Does not own the pointer.
     */
    virtual void SetInput(const Vector *x){input = x;}

    /**
       @brief Set the Operation ID to call the appropriate operation
       from Mult().
     */
    virtual void SetOperationID(OperationID id){ operation_id = id; }

    /**
       @brief Get the current OperationID.
     */
    OperationID GetOperationID() const { return operation_id; }

    /**
       @brief Implements the coupling scheme for function Application::Mult()
     */
    virtual void Mult(const Vector &x, Vector &y) const override {
        MFEM_ABORT("OperatorCoupler::Mult() const is not overridden!"
                   "Scheme may not be supported for this operation.");
    }

    /**
       @brief Implements the coupling scheme for function Application::Mult().
       This is a non-const version that calls the const version by default.
     */
    virtual void Mult(const Vector &x, Vector &y) {
        Mult(x,y);
    }

    /**
       @brief Implements the coupling scheme for function Application::ImplicitSolve()
     */
    virtual void ImplicitSolve(const real_t dt, const Vector &x, Vector &k ) const {
        MFEM_ABORT("OperatorCoupler::ImplicitSolve() const is not overridden!"
                   "Scheme may not be supported for this operation.");
    }

    /**
       @brief Implements the coupling scheme for function Application::ImplicitSolve().
       This is a non-const version that calls the const version by default, and preserves
       const-correctness when the function is called from Mult() const.
     */
    virtual void ImplicitSolve(const real_t dt, const Vector &x, Vector &k ) override {
        std::as_const(*this).ImplicitSolve(dt,x,k);
    }

    /**
       @brief Implements the coupling scheme for function Application::Step()
     */
    virtual void Step(Vector &x, real_t &t, real_t &dt) const {
        MFEM_ABORT("OperatorCoupler::Step() const is not overridden!"
                   "Scheme may not be supported for this operation.");
    }

    /**
       @brief Implements the coupling scheme for function Application::Step().
       This is a non-const version that calls the const version by default, and
       preserves const-correctness when the function is called from Mult() const.
     */
    virtual void Step(Vector &x, real_t &t, real_t &dt){
        std::as_const(*this).Step(x,t,dt);
    }

    virtual Scheme GetType() const { return type_; }
    
    /**
     * @brief Function for selecting the desired OperatorCoupler for coupling @a op
     * based on the specified coupling @a type. The called gets ownership of the
     * returned pointer and is responsible for deleting it.
     */
    static MFEM_EXPORT OperatorCoupler* Select(CoupledOperator *op, Scheme type);

};

/**
   @brief Base class for partitioned coupling schemes.
 */
class PartitionedOperatorCoupler : public OperatorCoupler
{
public:
    PartitionedOperatorCoupler(CoupledOperator *op) : OperatorCoupler(op){}
};

/**
   @brief Base class for monolithic coupling schemes.
 */
class MonolithicOperatorCoupler : public OperatorCoupler
{
public:
    MonolithicOperatorCoupler(CoupledOperator *op) : OperatorCoupler(op){}
};

/**
   @brief Additive Schwarz coupling, a Jacobi-type fixed-point scheme, between
   multiple operators.
 */
class AdditiveSchwarzCoupler : public PartitionedOperatorCoupler
{
private:
    Scheme type_ = Scheme::ADDITIVE_SCHWARZ;
public:
    /**
       @brief Construct a new Additive Schwarz coupler
     */
    AdditiveSchwarzCoupler(CoupledOperator *op) : PartitionedOperatorCoupler(op){}

    /**
       @brief Implements the coupling scheme for function Application::Mult()
       When called from Solver::Mult(), this can operate on ImplicitSolve or Step
       when OperationID is set appropriately.
     */
    void Mult(const Vector &x, Vector &y) const override;

    /**
       @brief Implements the coupling scheme for function Application::ImplicitSolve()
     */
    void ImplicitSolve(const real_t dt, const Vector &x, Vector &k ) const override;

    /**
       @brief Implements the coupling scheme for function Application::Step()
     */
    void Step(Vector &x, real_t &t, real_t &dt) const override;

    Scheme GetType() const override { return type_; }    
};

/**
   @brief Alternating Schwarz coupling, a Gauss-Seidel-type fixed-point scheme,
   between multiple operators.
 */
class AlternatingSchwarzCoupler : public PartitionedOperatorCoupler
{
private:
    Scheme type_ = Scheme::ALTERNATING_SCHWARZ;
public:
    /**
       @brief Construct a new Alternating Schwarz coupler
     */
    AlternatingSchwarzCoupler(CoupledOperator *op) : PartitionedOperatorCoupler(op){}

    /**
       @brief Implements the coupling scheme for function Application::Mult()
       When called from Solver::Mult(), this can operate on ImplicitSolve or Step
       when OperationID is set appropriately.
     */
    void Mult(const Vector &ki, Vector &k) const override;

    /**
       @brief Implements the coupling scheme for function Application::ImplicitSolve()
     */
    void ImplicitSolve(const real_t dt, const Vector &x, Vector &k ) const override;

    /**
       @brief Implements the coupling scheme for function Application::Step()
     */
    void Step(Vector &x, real_t &t, real_t &dt) const override;

    Scheme GetType() const override { return type_; }
};


/**
   @brief Inexact, monolithic coupling between multiple operators using
   Jacobian-free Newton-Krylov methods.
 */
class JacobianFreeFullCoupler : public MonolithicOperatorCoupler
{
private:
    Scheme type_ = Scheme::MONOLITHIC;
protected:
    mutable future::FDJacobian grad;
    mutable Vector u;

public:
    /**
       @brief Construct a new Inexact Full Coupler
       @param eps Finite difference perturbation size
     */
    JacobianFreeFullCoupler(CoupledOperator *op, real_t eps=1.0e-6) : 
                            MonolithicOperatorCoupler(op),
                            grad(*this,eps), u(Width()) {}

    /**
       @brief Computes the action of the implicit, monolithic operator on vector x.
              ImplicitMult assumes ODE of the form F(u,k,t) = M*k - g(u,t), G(u,t) = 0
              For fully implicit, monolithic solver, we take F(u+dt*k,k,t) = M*k - g(u+dt*k,t)
     */
    void Mult(const Vector &x, Vector &y) const override;

    /**
       @brief Update the current state/point of linearization for the nonlinear operator. For
              Jacobian-free methods, i.e. J(k)*v = (F(k+eps*v) - F(k))/eps, this sets the F(k)
     */
    Operator& GetGradient(const Vector &k) const override;

    Scheme GetType() const override { return type_; }
};


} //mfem namespace

#endif // MFEM_USE_MPI

#endif
