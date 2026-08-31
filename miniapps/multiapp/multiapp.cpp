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

void GraphOperation::Execute() const
{
    auto get = [](const Field *f) -> Vector& { return *f->Data(); };
    IterableToMultiVector<Vector&>(inputs, xmv, get);
    IterableToMultiVector<Vector&>(outputs, ymv, get);
    execute(xmv, ymv);
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
    primal.SetSize(inputs.Size());
    pmv.SetNumBlocks(inputs.Size());
    for(int i = 0; i < inputs.Size(); i++)
    {
        primal[i] = new Vector(*inputs[i]->Data());
        pmv.MakeRef(i, *primal[i]);
    }

    execute_primal = oper.execute;
    execute = [func = grad, &x0 = pmv](const MultiVector &x, MultiVector &y)
                    { func(x0, x, y); };
}

void GraphOperationGradient::Execute() const
{
    auto get = [](const Field *f) -> Vector& { return *f->Adjoint(); };
    IterableToMultiVector<Vector&>(inputs, xmv, get);
    IterableToMultiVector<Vector&>(outputs, ymv, get);
    execute(xmv, ymv);
    // grad(pmv,xmv, ymv);
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
        auto op = new GraphOperation(*this, inputs, outputs, execute, grad, grad_transpose);
        tape->RegisterOperation(op);
    }
    else
    {   // Same tape but not recording, or no tape at all
        // MFEM_ABORT("Input fields are not being recorded on a tape. Cannot register operation.");
    }
}

void GraphNode::RegisterFields(std::initializer_list<Field*> inputs,
                               std::initializer_list<Field*> outputs)
{
    auto execute = [this](const MultiVector &x, MultiVector &y) { this->MultMV(x, y); };
    auto grad_mult = [this](const MultiVector &x, const MultiVector &dx, MultiVector &dy)
                           { this->GradientMult(x, dx, dy); };
    auto grad_mult_transpose = [this](const MultiVector &x, const MultiVector &dx, MultiVector &dy)
                                      { this->GradientMultTranspose(x, dx, dy); };
    RegisterFields(inputs, outputs, execute, grad_mult, grad_mult_transpose);
}

DAGraph::DAGraph(const int nops, const int nfields) : GraphNode()
{
    int reserve_ops = (nops > 0) ? nops : max_ops;
    operations.Reserve(reserve_ops);
    op_owned.Reserve(reserve_ops);

    int reserve_fields = (nfields > 0) ? nfields : max_fields;
    fields.Reserve(reserve_fields);
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
        // Visit all nodes that this node depends on
        for(auto input_field : iop->inputs)
        {
            for(int j=0; j < nop; j++)
            {
                auto jop = operations[j];
                if(jop == iop) continue;
                for(auto output_field : jop->outputs)
                {
                    if(input_field->ID() == output_field->ID()) // Compare by unique ID
                    {
                        DepthFirstSearch(j);
                    }
                }
            }
        }
        sorted_indices.push_back(op_index);
    };

    for(int i=0; i < nop; i++)
    {
        DepthFirstSearch(i);
    }

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
            for(int j=0; j < i; j++)
            {
                int jdx = get_index(j);
                auto jop = operations[jdx];
                if(jop == iop) continue;
                auto jfields = (reverse) ? jop->inputs : jop->outputs;
                for(auto output_field : jfields)
                {
                    if(input_field->ID() == output_field->ID()) // Compare by unique ID
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

    MultiVectorToIterable(x, inputs, get_inputs_func);
    MultiVectorToIterable(y, outputs, get_outputs_func);

    for(auto op : operations) { op->Execute(); }

    // Reset input and output data to null
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
    return GetGradientMV(xmv);
}

Operator &DAGraph::GetGradientMV(const MultiVector &x) const
{
    if(!grad_dag)
    {
        grad_dag = new DualGraph(*this);
        grad_dag->UpdateState(x);
        grad_dag->Assemble();
    }
    else
    {
        grad_dag->UpdateState(x);
    }
    return *grad_dag;
}

void DAGraph::Watch(std::initializer_list<Field*> fields_list)
{
    // Clear the operation for the new recording
    Reset();

    dag_op = new GraphOperation(*this, fields_list, {});

    // TODO: Experimental, allocate state memory for dag inputs

    // Set the tape for each watched field
    for(auto &f : dag_op->inputs) { f->SetTape(this); }
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
    }
    AddOperation(op);

    // TODO: Experimental, allocate state memory for outputs here

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

    dag_op->outputs = Array<Field*>(outputs_list);

    get_inputs_func = [](Field *f, const Vector &v) { f->SetData(const_cast<Vector*>(&v)); };
    get_outputs_func = [](Field *f, Vector &v) { f->SetData(&v); };

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


DualGraph::DualGraph(const DAGraph &primal) : DAGraph(primal.Size()),
                                            primal_dag(&primal)
{
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
    get_inputs_func = [](Field *f, const Vector &v) { f->SetAdjoint(const_cast<Vector*>(&v)); };
    get_outputs_func = [](Field *f, Vector &v) { f->SetAdjoint(&v); };

    for (auto pop : primal_dag->operations)
    {
        auto dual_op = pop->GetGradient(); // Allocates new
        AddOperation(dual_op); // Transfer ownership
    }

    DAGraph::Assemble();
    const bool reverse = true;
    ComputeDepth(reverse); // Compute depth in reverse order
}

void DualGraph::UpdateState(const MultiVector &x)
{
    auto inputs = dag_op->inputs;
    MFEM_ASSERT(inputs.Size() == x.NumBlocks(), "Number of input blocks (" << x.NumBlocks()
                << ") must match number of input fields (" << inputs.Size() << ")");

    for(int i=0; i < inputs.Size(); i++)
    {
        inputs[i]->SetData(const_cast<Vector*>(&x[i]));
    }

    // Temporary vector to hold output data
    // Not needed if we dont operate the leaf nodes
    auto outputs = dag_op->outputs;
    auto [inoffsets, outoffsets] = GetOffsets();
    Vector y(outoffsets.Last());
    MultiVector ymv(outputs.Size());
    for(int i=0; i < outputs.Size(); i++)
    {
        ymv.MakeRef(i, y);
        outputs[i]->SetData(&ymv[i]);
    }

    int nop = primal_dag->operations.Size();
    auto grad_mode = GraphNode::ExecutionMode::GRADIENT_MODE;
    auto default_mode = GraphNode::ExecutionMode::DEFAULT_MODE;
    for (int i = 0; i < nop; i++)
    {
        auto pop = primal_dag->operations[i];
        GraphNode *gop = dynamic_cast<GraphNode*>(&(pop->GetOperator()));
        if(gop) { gop->SetExecutionMode(grad_mode); }
        // if(primal_dag->op_depth[i] > 0)
        // {   // Only execute nodes that are not leaves
            pop->Execute(); 
        // }
        if(gop) { gop->SetExecutionMode(default_mode); }
    }
}


} // namespace mfem
