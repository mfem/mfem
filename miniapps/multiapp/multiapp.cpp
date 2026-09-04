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

#include "multiapp.hpp"

namespace mfem
{

#define PRINT_MESSAGE(msg) \
   { if (Mpi::Root()) { std::cout << msg << std::endl; } }

int GetValidID(int id, int& current, int lb, int ub)
{
    return (id >= lb && id <= ub) ? id : current++;
}

void BlockVectorToMultiVector(BlockVector &bv, MultiVector &mv)
{
    int nblocks = bv.NumBlocks();
    mv.SetNumBlocks(nblocks);
    for (int i = 0; i < nblocks; i++) { mv.MakeRef(i, bv.GetBlock(i)); }
}

GraphOperation *GraphOperation::GetGradient() const
{
    return new GraphOperationGradient(const_cast<GraphOperation&>(*this));
}

GraphOperationGradient::GraphOperationGradient(GraphOperation &oper):
                        GraphOperation(oper.GetOperator(), {}, {}, nullptr,
                                       oper.grad, oper.grad_transpose)
{
    inputs = oper.inputs;
    outputs = oper.outputs;
    execute_primal = oper.execute;
    execute = [func = grad, &x0 = pmv](const MultiVector &x, MultiVector &y)
                    { func(x0, x, y); };
}

void GraphOperationGradient::SetPrimal(MultiVector &x)
{
    int n = x.NumBlocks();
    pmv.SetNumBlocks(n);
    for(int i = 0; i < n; i++)
    {
        pmv.MakeRef(i, x[i]);
        // pmv.MakeRef(i, std::as_const(x[i]));
    }
}

void GraphOperationGradient::Execute(const MultiVector &x, MultiVector &y) const
{
    // if(execute) { grad(pmv, x, y); }
    if(execute) { execute(x, y); }
    else { MFEM_ABORT("Execute function not defined for this GraphOperationGradient."); }
}


void GraphNode::RegisterFields(std::initializer_list<Field*> inputs,
                               std::initializer_list<Field*> outputs,
                               GraphOperation::ExecuteFunc execute,
                               GraphOperation::GradFunc grad,
                               GraphOperation::GradFunc grad_transpose)
{
    // Tape operations
    // Check if all input fields have the same, non-null tape; if so, register this node with that tape
    bool has_same_tape = std::empty(inputs) || std::equal(inputs.begin(), inputs.end(), inputs.begin(),
                         [](const Field *a, const Field *b)
                         { return (a->GetTape() && (a->GetTape() == b->GetTape())); });

    auto in = inputs.begin();
    auto tape = (*in)->GetTape();
    if(!has_same_tape && tape)
    {
        MFEM_ABORT("Input fields are being recorded on different tapes. Cannot register operation.");
    }
    else if(tape->IsRecording()) // All tapes are the same and recording is active
    {
        // Register the operation on the tape
        auto op = new GraphOperation(*this, inputs, outputs, execute, grad, grad_transpose);
        tape->RegisterOperation(op);
    }
    else
    {   // Same tape but not recording, or no tape at all
        // Should we abort or do nothing?
        // MFEM_ABORT("Input fields are not being recorded on a tape. Cannot register operation.");
    }
}

void GraphNode::RegisterFields(std::initializer_list<Field*> inputs,
                               std::initializer_list<Field*> outputs)
{
    auto execute = [this](const MultiVector &x, MultiVector &y) { this->MultMV(x, y); };
    auto grad_mult = [this](const MultiVector &x, const MultiVector &dx, MultiVector &dy)
                           { this->GradientMultMV(x, dx, dy); };
    auto grad_mult_transpose = [this](const MultiVector &x, const MultiVector &dx, MultiVector &dy)
                                     { this->GradientMultTransposeMV(x, dx, dy); };
    RegisterFields(inputs, outputs, execute, grad_mult, grad_mult_transpose);
}

DAGraph::DAGraph(const int nops, const int nfields) : GraphNode()
{
    int reserve_ops = (nops > 0) ? nops : max_ops;
    operations.Reserve(reserve_ops);
    op_owned.Reserve(reserve_ops);

    int reserve_fields = (nfields > 0) ? nfields : max_fields;
    fields.Reserve(reserve_fields);

    // -- EXPERIMENTAL: Reserve state memory for each field
    state_memory.reserve(reserve_fields);
    // -- EXPERIMENTAL
}

DAGraph::~DAGraph()
{
    Reset();
}

void DAGraph::Reset()
{
    for(auto op : operations)
    {
        if(op) { delete op; op = nullptr; }
    }

    if(dag_op) { delete dag_op; dag_op = nullptr; }
    if(grad_dag) { delete grad_dag; grad_dag = nullptr; }

    fields.DeleteAll();
    operations.DeleteAll();
    op_depth.DeleteAll();
    op_owned.DeleteAll();
    id_to_op_index.clear();
    id_to_field_index.clear();

    is_sorted = false;
    is_assembled = false;
    if(fdj_op) { delete fdj_op; fdj_op = nullptr; }

    // -- EXPERIMENTAL: Clear state memory
    const int n = state_memory.size();
    for(int i = 0; i < n; i++)
    {
        const int ngrad = state_memory[i].Size();
        auto stmem = state_memory[i];
        for(int j = 0; j < ngrad; j++)
        {
            if(stmem[j]) { delete stmem[j]; stmem[j] = nullptr; }
        }
        stmem.DeleteAll();
    }
    state_memory.clear();
    // -- EXPERIMENTAL
}

void DAGraph::Assemble()
{
    // Sort graph nodes topologically to ensure correct execution order
    // Ordering is not unique, hence, id->op_index maps are needed
    Sort();

    // Compute depth of the graph nodes in reverse order
    const bool reverse = true;
    ComputeDepth(reverse);

    // Delete any existing gradient operator as node ordering may have changed
    if (grad_dag) { delete grad_dag; grad_dag = nullptr; }

    is_assembled = true;
}

void DAGraph::Sort()
{
    int nop = operations.Size();
    Array<int> sorted_indices;
    sorted_indices.Reserve(nop);

    Array<bool> visited(nop);
    visited = false; // Initialize all nodes as unvisited

    // Perform a depth-first search to sort the nodes topologically
    std::function<void(int)> DepthFirstSearch = [&](int op_index)
    {
        if(visited[op_index]) return;
        visited[op_index] = true;
        auto iop = operations[op_index];
        for(auto input_field : iop->inputs) // Visit all dependencies
        {
            const int in_id = input_field->ID();
            for(int j=0; j < nop; j++)
            {
                auto jop = operations[j];
                if(jop == iop) continue;
                for(auto output_field : jop->outputs)
                {
                    if(in_id == output_field->ID()) // Compare by unique ID
                    {
                        DepthFirstSearch(j);
                    }
                }
            }
        }
        sorted_indices.push_back(op_index);
    };

    for(int i=0; i < nop; i++) { DepthFirstSearch(i); }

    // check if already sorted
    bool already_sorted = true;
    for(int i=0; i < nop; i++)
    {
        if(sorted_indices[i] != i)
        {
            already_sorted = false;
            break;
        }
    }

    if(!already_sorted)
    {
        operations.Permute(sorted_indices);
        op_owned.Permute(sorted_indices); // Ownership not used
    }

    for(int i=0; i < nop; i++)
    {
        auto op = operations[i];
        for(auto *f : op->outputs)
        {
            id_to_op_index.Register(f->ID(), i);
        }
    }
    is_sorted = true;
}

void DAGraph::ComputeDepth(bool reverse)
{
    // Compute depth of ordered nodes
    int nop = operations.Size();
    op_depth.SetSize(nop);
    op_depth = 0;

    auto get_index = [nop, reverse](int i) { return reverse ? nop - 1 - i : i; };
    for(int i=0; i < nop; i++)
    {
        int maxdepth = 0;
        int idx = get_index(i);
        auto iop = operations[idx];
        auto ifields = (reverse) ? iop->outputs : iop->inputs;
        for(auto input_field : ifields)
        {
            const int in_id = input_field->ID();
            for(int j=0; j < i; j++)
            {
                int jdx = get_index(j);
                auto jop = operations[jdx];
                if(jop == iop) continue;
                auto jfields = (reverse) ? jop->inputs : jop->outputs;
                for(auto output_field : jfields)
                {
                    if(in_id == output_field->ID()) // Compare by unique ID
                    {
                        maxdepth = std::max(maxdepth, op_depth[jdx] + 1);
                    }
                }
            }
        }
        op_depth[idx] = maxdepth;
        max_depth = std::max(max_depth, maxdepth);
    }
}

void DAGraph::UpdateState(const MultiVector &x)
{
    MFEM_ABORT("Function not overridden for this class.")
}

void DAGraph::Watch(std::initializer_list<Field*> fields_list)
{
    // Clear the operation for the new recording
    Reset();

    dag_op = new GraphOperation(*this, fields_list, {});

    // TODO: // -- EXPERIMENTAL: allocate state memory for dag inputs

    // Set the tape for each watched field
    for(auto &f : dag_op->inputs) { f->SetTape(this); }
}

void DAGraph::AddOperation(GraphOperation *op)
{
    operations.push_back(op);
    // Register input fields and id->index mapping
    const int grad_order = GetGradientOrder() + 1; // Instead of grad_order of dag,
                                                  // we can use grad_order of operation
                                                  // For finitely differetiable operators

    for(auto *f : op->inputs)
    {
        if(!id_to_field_index.Has(f->ID()))
        {
            fields.push_back(f);
            id_to_field_index.Register(f->ID(), fields.Size() - 1);

            // -- EXPERIMENTAL: Reserve state memory for each field
            state_memory.push_back(Array<StateType>());
            auto &fmem = state_memory.back();
            fmem.Reserve(grad_order);
            for(int i = 0; i < grad_order; i++)
            {
                fmem.push_back(f->MakeNew()); // Allocate new memory (owned by dag)
            }
            // -- EXPERIMENTAL
        }
    }

    // Register outputs fields and id->index mapping and
    // id->operation that outputs the field
    for(auto *f : op->outputs)
    {
        id_to_op_index.Register(f->ID(), operations.Size() - 1);
        if(!id_to_field_index.Has(f->ID()))
        {
            fields.push_back(f);
            id_to_field_index.Register(f->ID(), fields.Size() - 1);

            // -- EXPERIMENTAL: Reserve state memory for each field
            state_memory.push_back(Array<StateType>());
            auto &fmem = state_memory.back();
            fmem.Reserve(grad_order);
            for(int i = 0; i < grad_order; i++)
            {
                fmem.push_back(f->MakeNew()); // Allocate new memory (owned by dag)
            }
            // -- EXPERIMENTAL
        }
    }
    op_owned.push_back(true); // For now, we own the operations
    is_sorted = false;
    is_assembled = false;
}

void DAGraph::RegisterOperation(GraphOperation *op)
{
    if(is_recording)
    {
        for(auto &f : op->inputs)
        {   // Check if this is the tape for input fields
            if(f->GetTape() != this)
            {
                MFEM_ABORT("Input field " << f->Name() << " (ID: " << f->ID()
                        << ") is not being recorded on this tape.");
            }
        }
        for(auto &f : op->outputs)
        {   // Check if output is already registered in the DAG
            bool has_output = id_to_field_index.Has(f->ID());
            if(has_output)
            {
                MFEM_ABORT("Output field " << f->Name() << " (ID: " << f->ID()
                        << ") is already registered in the DAG.");
            }
        }
    }

    AddOperation(op);

    if(is_recording)
    {   // Set this as tape for all outputs to be watched
        for(auto &f : op->outputs) { f->SetTape(this); }
    }
}

void DAGraph::StopRecording(std::initializer_list<Field*> outputs_list)
{
    int ninputs = dag_op->inputs.Size();
    int noutputs = outputs_list.size();

    MFEM_ASSERT((ninputs > 0) && (noutputs > 0),
                "Must have at least one input and one output field. Total inputs: "
                << ninputs << ", total outputs: " << noutputs);

    // Check if all output of DAG are output of operations in the DAG
    // by checking if the fields are registered
    for(auto &f : outputs_list)
    {
        if(!id_to_field_index.Has(f->ID()))
        {
            MFEM_ABORT("Output field " << f->Name() << " (ID: " << f->ID()
                        << ") is not registered in the DAG.");
        }
    }

    dag_op->outputs = Array<Field*>(outputs_list);

    // Clear the recorded fields after stopping recording and
    // nullifying the tape association for each tracked field
    is_recording = false;
    ClearTape();
}

void DAGraph::ClearTape()
{
    for (auto op : operations)
    {
        for (auto &f : op->inputs) { f->SetTape(nullptr); }
        for (auto &f : op->outputs) { f->SetTape(nullptr); }
    }
}

void DAGraph::Mult(const Vector &x, Vector &y) const
{
    MFEM_ASSERT(width == x.Size(), "Input vector size (" << x.Size()
                << ") must match matrix width (" << width << ")");

    MFEM_ASSERT(height == y.Size(), "Output vector size (" << y.Size()
                << ") must match matrix height (" << height << ")");

    auto [inoffsets, outoffsets] = GetOffsets();
    BlockVector xb(x.GetData(), inoffsets);
    BlockVector yb(y.GetData(), outoffsets);
    MultiVector xmv, ymv;

    BlockVectorToMultiVector(xb, xmv);
    BlockVectorToMultiVector(yb, ymv);

    MultMV(xmv, ymv);
}

void DAGraph::MultMV(const MultiVector &x, MultiVector &y) const
{
    MFEM_ASSERT(is_sorted, "Graph is not sorted. Please call Sort().");

    auto inputs = dag_op->inputs;
    auto outputs = dag_op->outputs;

    MFEM_ASSERT(inputs.Size() == x.NumBlocks(), "Number of input blocks (" << x.NumBlocks()
                << ") must match number of input fields (" << inputs.Size() << ")");

    MFEM_ASSERT(outputs.Size() == y.NumBlocks(), "Number of output blocks (" << y.NumBlocks()
                << ") must match number of output fields (" << outputs.Size() << ")");

    // -- EXPERIMENTAL: Pass input and output to Execute()
    // The memory for input and output does not need to be allocated
    // It comes as arguments to this function loop over inputs and outputs.
    int iin = 0, iout = 0;
    const int igrad = GetGradientOrder();
    for (auto &f : inputs)
    {
        bool has_state = id_to_field_index.Has(f->ID());
        int idx = id_to_field_index.Get(f->ID());
        MFEM_ASSERT(has_state, "Field with ID " << f->ID() << " is not registered in the DAG.");

        auto &fmem = state_memory[idx];
        MFEM_ASSERT(fmem.Size() > igrad, "State memory for field " << f->Name() << " (ID: " << f->ID()
                    << ") does not have enough derivatives. Expected at least "
                    << igrad + 1 << ", but got " << fmem.Size() << ".");
        if(fmem[igrad] != nullptr) { delete fmem[igrad]; } // Delete existing memory if any
        fmem[igrad] = const_cast<StateType>(&x[iin++]); // Point dag's memory to inputs
    }
    for (auto &f : outputs)
    {
        bool has_state = id_to_field_index.Has(f->ID());
        const int idx = id_to_field_index.Get(f->ID());
        MFEM_ASSERT(has_state, "Field with ID " << f->ID() << " is not registered in the DAG.");

        auto &fmem = state_memory[idx];
        MFEM_ASSERT(fmem.Size() > igrad, "State memory for field " << f->Name() << " (ID: " << f->ID()
                    << ") does not have enough derivatives. Expected at least "
                    << igrad + 1 << ", but got " << fmem.Size() << ".");

        if(fmem[igrad] != nullptr) { delete fmem[igrad]; } // Delete existing memory if any
        fmem[igrad] = &y[iout++]; // Point dag's memory to outputs
    }
    // -- EXPERIMENTAL

    int iop = 0;
    MultiVector xmv, ymv;
    for(auto op : operations)
    {
        auto [isz, osz] = op->Size();
        xmv.SetNumBlocks(isz);
        ymv.SetNumBlocks(osz);
        iin = 0;
        for (auto &f : op->inputs)
        {
            const int idx = id_to_field_index.Get(f->ID());
            xmv.MakeRef(iin++,*state_memory[idx][igrad]);
        }

        iout = 0;
        for (auto &f : op->outputs)
        {
            const int idx = id_to_field_index.Get(f->ID());
            ymv.MakeRef(iout++,*state_memory[idx][igrad]);
        }

        op->Execute(xmv, ymv);
        iop++;
    }

    // -- EXPERIMENTAL: Resetting the state memory for the current
    // gradient order to nullptr after execution
    for (auto & inout : {inputs, outputs}) {
        for (auto & f : inout) {
            const int idx = id_to_field_index.Get(f->ID());
            state_memory[idx][igrad] = nullptr; // Reset to nullptr after deletion
        }
    }
    // -- EXPERIMENTAL
}

void DAGraph::MultTranspose(const Vector &x, Vector &y) const
{
    MFEM_ASSERT(width == x.Size(), "Input vector size (" << x.Size()
                << ") must match matrix width (" << width << ")");

    MFEM_ASSERT(height == y.Size(), "Output vector size (" << y.Size()
                << ") must match matrix height (" << height << ")");

    auto [inoffsets, outoffsets] = GetOffsets();
    BlockVector xb(x.GetData(), outoffsets);
    BlockVector yb(y.GetData(), inoffsets);
    MultiVector xmv, ymv;

    BlockVectorToMultiVector(xb, xmv);
    BlockVectorToMultiVector(yb, ymv);

    MultTransposeMV(xmv, ymv);
}

// Backward pass not implemented yet
void DAGraph::MultTransposeMV(const MultiVector &x, MultiVector &y) const
{
    MFEM_ABORT("Function not overridden for this class.")
}

Operator& DAGraph::GetGradient(const Vector &x) const
{
    auto [inoffsets, outoffsets] = GetOffsets();
    BlockVector xb(x.GetData(), inoffsets);
    MultiVector xmv;
    BlockVectorToMultiVector(xb, xmv);

    // -- EXPERIMENTAL: Possibly support both modes
    // or remove the finite difference jacobian operator and use the gradient dag instead
    if(gradient_mode == GradientMode::FINITE_DIFFERENCE)
    {
        if(!fdj_op)
        {
            fdj_op = new future::FDJacobian(*this, x);
        }
        else
        {
            auto fd_op = dynamic_cast<future::FDJacobian*>(fdj_op);
            fd_op->GetGradient(x);
        }
        return *fdj_op;
    }
    else
    {
        return GetGradientMV(xmv);
    }
    // -- EXPERIMENTAL
}

Operator &DAGraph::GetGradientMV(const MultiVector &x) const
{
    if(!grad_dag)
    {
        grad_dag = new DualGraph(*this);
        grad_dag->Assemble();
        grad_dag->UpdateState(x);
    }
    else
    {
        grad_dag->UpdateState(x);
    }
    return *grad_dag;
}


DualGraph::DualGraph(const DAGraph &primal) : DAGraph(primal.Size()),
                                            primal_dag(&primal)
{
    SetGradientOrder(primal.GetGradientOrder() + 1);
    auto [inoffsets, outoffsets] = primal_dag->GetOffsets();
    SetOffsets(inoffsets, outoffsets);

    auto pdag_op = primal_dag->dag_op;
    auto inputs  = pdag_op->inputs;
    auto outputs = pdag_op->outputs;

    dag_op = new GraphOperation(*this, {}, {});
    dag_op->inputs = inputs;
    dag_op->outputs = outputs;
}

void DualGraph::Assemble()
{
    // -- EXPERIMENTAL: Should we use the tape feather for the dual graph?
    // For now, insert operations for the dual graph directly from the primal graph
    for (auto pop : primal_dag->operations)
    {
        auto dual_op = pop->GetGradient(); // Allocates new
        AddOperation(dual_op); // Transfer ownership
    }

    DAGraph::Assemble();
    const bool reverse = true;
    ComputeDepth(reverse); // Compute depth in reverse order to identify leaf nodes
}

void DualGraph::UpdateState(const MultiVector &x)
{
    auto inputs = dag_op->inputs;
    MFEM_ASSERT(inputs.Size() == x.NumBlocks(), "Number of input blocks (" << x.NumBlocks()
                << ") must match number of input fields (" << inputs.Size() << ")");

    // Temporary vector to hold output data
    // Not needed if we dont operate the leaf nodes
    auto outputs = dag_op->outputs;
    auto [inoffsets, outoffsets] = GetOffsets();

    int nop = primal_dag->operations.Size();
    auto grad_mode = GraphNode::ExecutionMode::GRADIENT_MODE;
    auto default_mode = GraphNode::ExecutionMode::DEFAULT_MODE;
    const int ipgrad = primal_dag->GetGradientOrder();

    // -- EXPERIMENTAL: Copy input into the memory for input field
    int iin = 0, iout = 0;
    for (auto &f : inputs)
    {
        bool has_state = id_to_field_index.Has(f->ID());
        int idx = id_to_field_index.Get(f->ID());
        MFEM_ASSERT(has_state, "Field with ID " << f->ID() << " is not registered in the DAG.");
        *state_memory[idx][ipgrad] = x[iin++];
    }

    MultiVector xmv, ymv;
    for (int iop = 0; iop < nop; iop++)
    {
        auto pop = primal_dag->operations[iop];
        auto [isz, osz] = pop->Size();
        xmv.SetNumBlocks(isz);
        ymv.SetNumBlocks(osz);

        iin = 0; iout = 0;
        for (auto &f : pop->inputs)
        {
            const int idx = id_to_field_index.Get(f->ID());
            xmv.MakeRef(iin++,*state_memory[idx][ipgrad]);
        }

        for (auto &f : pop->outputs)
        {
            const int idx = id_to_field_index.Get(f->ID());
            ymv.MakeRef(iout++,*state_memory[idx][ipgrad]);
        }
        // if(primal_dag->op_depth[iop] > 0) // Only execute nodes that are not leaves
        // {
            GraphNode *gop = dynamic_cast<GraphNode*>(&(pop->GetOperator()));
            if(gop) { gop->SetExecutionMode(grad_mode); }
            pop->Execute(xmv, ymv);
            if(gop) { gop->SetExecutionMode(default_mode); }
        // }
        operations[iop]->SetPrimal(xmv);
    }
}


} // namespace mfem
