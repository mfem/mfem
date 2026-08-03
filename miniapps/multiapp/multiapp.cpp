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

#ifdef MFEM_USE_MPI

namespace mfem
{


void FieldBlockVectorTransfer(const Array<int> &offsets, Array<Field*> &fields, Vector &v,
                               std::function<void(Field&, Vector&)> assemble)
{
    int nblocks = offsets.Size() - 1;
    MFEM_ASSERT(nblocks == fields.Size(),
                "Number of blocks in offsets does not match number of fields");

    v.SetSize(offsets.Last());
    BlockVector vb(v.GetData(), offsets);
    for(int i=0; i < nblocks; i++)
    {
        auto field = fields[i];
        assemble(*field, vb.GetBlock(i));
    }
}



DAGraph::~DAGraph()
{
    for(int i=0; i < nnodes; i++)
    {
        if(node_owned[i] && nodes[i]) delete nodes[i];
    }
    if(grad) delete grad;
}

void DAGraph::Assemble()
{
    // Sort graph nodes topologically to ensure correct execution order
    TopologicalSort();

    // Compute depth of the graph nodes
    ComputeDepth();

    // Collect all fields from the nodes into the field map
    CollectFields();

    // Validate each node
    for (auto &node : nodes)
    {
        ValidateNode(*node);
    }

    // Update width and height of the DAG from offsets
    // Check that the input and output offsets are consistent
    ValidateOffsets();
    width  = input_offset.Last();
    height = output_offset.Last();

    // Delete any existing gradient operator as node ordering may have changed
    if (grad) delete grad;

    assembled = true;
}

void DAGraph::ValidateOffsets()
{
    // Check that the input and output offsets are consistent
    // with the number of inputs and outputs
    if(InputFields().Size() > 1)
    {
        MFEM_ASSERT(input_offset.Size() == InputFields().Size() + 1,
                    "Input offsets size inconsistent with number of input fields");
    }
    else
    {
        input_offset = Array<int>({0, nodes[0]->Width()});
    }

    if(OutputFields().Size() > 1)
    {
        MFEM_ASSERT(output_offset.Size() == OutputFields().Size() + 1,
                    "Output offsets size inconsistent with number of output fields");
    }
    else
    {
        output_offset = Array<int>({0, nodes.Last()->Height()});
    }
}

void DAGraph::ValidateNode(GraphNode &node)
{
    // Validate that the node's input and output fields are consistent with the graph's field map
    auto inputs = node.InputFields();
    auto outputs = node.OutputFields();

    // Check offsets match width and height of the node
    MFEM_ASSERT(node.InputOffsets().Last() == node.Width(),
                "Node ID: " << node.ID() << " input offsets do not match node width.");
    MFEM_ASSERT(node.OutputOffsets().Last() == node.Height(),
                "Node ID: " << node.ID() << " output offsets do not match node height.");

    // Check number of input and output fields match the offsets
    MFEM_ASSERT(node.InputOffsets().Size() == inputs.Size() + 1,
                "Node input offsets size inconsistent with number of input fields");
    MFEM_ASSERT(node.OutputOffsets().Size() == outputs.Size() + 1,
                "Node output offsets size inconsistent with number of output fields");

    // Check that all input and output fields are registered in the graph's field map
    for(auto input_field : inputs)
    {
        MFEM_ASSERT(id_to_index.Has(input_field->ID()),
                    "Input field ID " << input_field->ID() << " not found in graph's field map");
    }
    for(auto output_field : outputs)
    {
        MFEM_ASSERT(id_to_index.Has(output_field->ID()),
                    "Output field ID " << output_field->ID() << " not found in graph's field map");
    }
}

void DAGraph::TopologicalSort()
{
    Array<int> sorted_indices;
    sorted_indices.Reserve(nnodes);

    Array<bool> visited(nnodes);
    visited = false; // Initialize all nodes as unvisited

    // Perform a depth-first search to sort the nodes topologically
    std::function<void(int)> DepthFirstSearch = [&](int node_index)
    {
        if(visited[node_index]) return;
        visited[node_index] = true;
        auto node = nodes[node_index];
        // Visit all nodes that this node depends on
        for(auto input_field : node->InputFields())
        {
            for(int j=0; j < nnodes; j++)
            {
                auto other_node = nodes[j];
                if(other_node == node) continue;
                for(auto output_field : other_node->OutputFields())
                {
                    if(input_field->ID() == output_field->ID()) // Compare by unique ID
                    {
                        DepthFirstSearch(j);
                    }
                }
            }
        }
        sorted_indices.push_back(node_index);
    };

    for(int i=0; i < nnodes; i++)
    {
        DepthFirstSearch(i);
    }

    nodes.Permute(sorted_indices);
    node_owned.Permute(sorted_indices);

    sorted = true;
}

void DAGraph::ComputeDepth()
{
    // Compute depth of ordered nodes
    node_depth.SetSize(nnodes);
    node_depth = 0;
    for(int i=0; i < nnodes; i++)
    {
        int max_dep = 0;
        auto node = nodes[i];
        for(auto input_field : node->InputFields())
        {
            for(int j=0; j < i; j++)
            {
                auto other_node = nodes[j];
                if(other_node == node) continue;
                for(auto output_field : other_node->OutputFields())
                {
                    if(input_field->ID() == output_field->ID()) // Compare by unique ID
                    {
                        max_dep = std::max(max_dep, node_depth[j] + 1);
                    }
                }
            }
        }
        node_depth[i] = max_dep;
    }
}

void DAGraph::CollectFields()
{
    MFEM_ASSERT(sorted, "DAGraph must be topologically sorted before collecting fields");

    id_to_index.clear();
    id_to_field.clear();

    int nfields = 0;
    for (auto f : InputFields())
    {
        id_to_index.Register(f->ID(), nfields++);
        id_to_field.Register(f->ID(), f);
    }

    for (auto &node : nodes)
    {
        for (auto f : node->OutputFields())
        {
            if (!id_to_index.Has(f->ID()))
            {
                id_to_index.Register(f->ID(), nfields++);
            }
            if (!id_to_field.Has(f->ID()))
            {
                id_to_field.Register(f->ID(), f);
            }
        }
    }
}

void DAGraph::Mult(const Vector &x, Vector &y) const
{
    MFEM_ASSERT(assembled, "DAGraph must be assembled before calling Mult()");

    MFEM_ASSERT(width == x.Size(), "Input vector size (" << x.Size()
                << ") must match matrix width (" << width << ")");

    MFEM_ASSERT(height == y.Size(), "Output vector size (" << y.Size()
                << ") must match matrix height (" << height << ")");

    BlockVector xb(x.GetData(), input_offset);
    BlockVector yb(y.GetData(), output_offset);

    auto inputs  = InputFields();
    auto outputs = OutputFields();
    for(int i=0; i < inputs.Size(); i++)
    {
        inputs[i]->SetData(&xb.GetBlock(i));
    }

    for(int i=0; i < outputs.Size(); i++)
    {
        outputs[i]->SetData(&yb.GetBlock(i));
    }

    if(input_type == InputType::VECTOR)
    {
        auto f_to_vec = [](Field &f, Vector &v) { v = *f.Data(); };
        auto vec_to_f = [](Field &f, Vector &v) { *f.Data() = v; };

        x_node.SetSize(MaxWidth());
        y_node.SetSize(MaxHeight());

        for (auto node : nodes)
        {
            x_node.SetSize(node->Width());
            y_node.SetSize(node->Height());

            FieldBlockVectorTransfer(node->InputOffsets(), node->InputFields(), x_node, f_to_vec);
            node->Mult(x_node, y_node);
            FieldBlockVectorTransfer(node->OutputOffsets(), node->OutputFields(), y_node, vec_to_f);
        }
    }
    else if(input_type == InputType::MULTIVECTOR)
    {
        MFEM_ABORT("DAGraph::Mult() not implemented for input type: MULTIVECTOR");
    }
    else if(input_type == InputType::NONE)
    {
        Vector x_unused, y_unused;
        for (auto node : nodes)
        {
            node->Mult(x_unused, y_unused);
        }
    }
    else
    {
        MFEM_ABORT("DAGraph::Mult() not implemented for input type: "
                    << static_cast<int>(input_type));
    }

    for(auto &f : inputs)
    {
        f->SetData(nullptr);
    }
    for(auto &f : outputs)
    {
        f->SetData(nullptr);
    }
}

void DAGraph::Execute(const Vector &x, Vector &y)
{
    // Call Mult for now
    Mult(x, y);
}

Operator& DAGraph::GetGradient(const Vector &x) const
{
    if(grad_mode == GradMode::FINITE_DIFF)
    {
        if(!grad)
        {
            grad = new future::FDJacobian(*this, x, 1e-6);
        }
        else
        {
            grad->GetGradient(x); // Update the FDJacobian with new point x
        }
        return *grad;
    }

    if(grad_mode == GradMode::ASSEMBLED ||
       grad_mode == GradMode::MATRIX_FREE)
    {
        if(!grad)
        {
            grad = new GraphGradient(const_cast<DAGraph&>(*this)); // Create a new GraphGradient operator
        }
        if(grad_mode == GradMode::ASSEMBLED)
        {
            return grad->GetGradient(x); // Assembled the Jacobian
        }
        else // GradMode::MATRIX_FREE
        {
            dynamic_cast<GraphGradient*>(grad)->Update(x); // Update the GraphGradient with new point x
        }
        return *grad;
    }
    else
    {
        MFEM_ABORT("DAGraph::GetGradient() not implemented for gradient mode: "
                    << static_cast<int>(grad_mode));
    }

    return *grad;
}

GraphGradient::GraphGradient(DAGraph &dag) : Operator(dag.Height(), dag.Width()),
                                             graph(&dag)
{
    MFEM_ASSERT(graph->IsAssembled(), "GraphGradient requires an assembled DAGraph.");
    MFEM_ASSERT(graph->IsSorted(), "GraphGradient requires a topologically sorted DAGraph.");

    auto id_map = graph->GetIdToIndexMap();
    auto field_map = graph->GetIdToFieldMap();

    MFEM_ASSERT(id_map.NumFields() == field_map.NumFields(),
                "Mismatch in number of fields between id_map and field_map");

    x_fields.SetSize(id_map.NumFields());
    x_fields = nullptr; // Initialize all pointers to nullptr
}

void GraphGradient::Update(const Vector &x)
{
    MFEM_ASSERT(graph != nullptr, "GraphGradient operator requires a non-null DAGraph pointer.");

    auto set_exec_mode = [&](DAGraph::ExecutionMode mode)
    {
        for (auto &node : graph->Nodes())
        {
            node->SetExecutionMode(mode);
        }
    };

    set_exec_mode(DAGraph::ExecutionMode::GRADIENT_MODE); // Set execution mode for forward pass
    fx.SetSize(graph->Height());
    graph->Mult(x, fx); // Forward pass to populate fields for gradient computations
    set_exec_mode(DAGraph::ExecutionMode::DEFAULT_MODE); // Reset execution mode for forward pass

    BlockVector xb(x.GetData(), graph->InputOffsets());
    BlockVector yb(fx.GetData(), graph->OutputOffsets());

    auto inputs  = graph->InputFields();
    auto outputs = graph->OutputFields();

    for(int i=0; i < inputs.Size(); i++)
    {
        inputs[i]->SetData(&xb.GetBlock(i));
    }

    for(int i=0; i < outputs.Size(); i++)
    {
        outputs[i]->SetData(&yb.GetBlock(i));
    }

    auto id_map = graph->GetIdToIndexMap();
    auto field_map = graph->GetIdToFieldMap();

    for (auto const& [id, idx] : id_map)
    {
        MFEM_ASSERT(idx >= 0 && idx < x_fields.Size(), "Index out of bounds for field ID: " << id);
        MFEM_ASSERT(field_map.Has(id), "Field ID not found in field_map: " << id);

        auto field = field_map.Get(id);
        auto f_data = field->Data();

        MFEM_ASSERT(f_data != nullptr, "Field data pointer is null for field ID: " << id);

        if(x_fields[idx] == nullptr)
        {
            x_fields[idx] = new Vector(*f_data); // Store a copy
        }
        else
        {
            x_fields[idx]->SetSize(f_data->Size());
            *x_fields[idx] = *f_data; // Update the stored copy
        }
    }
}

void GraphGradient::Mult(const Vector &x, Vector &y) const
{
    Forward(x, y); // Forward mode: compute JVP, y = J(z) * x
}

void GraphGradient::MultTranspose(const Vector &x, Vector &y) const
{
    Backward(x, y); // Backward mode: compute VJP, y = J^T(z) * x
}

void GraphGradient::Forward(const Vector &x, Vector &y) const
{
    MFEM_ASSERT(graph != nullptr, "GraphGradient operator requires a non-null DAGraph pointer.");

    MFEM_ASSERT(x.Size() == graph->Width(), "Input vector size (" << x.Size()
                << ") must match graph width (" << graph->Width() << ")");

    MFEM_ASSERT(y.Size() == graph->Height(), "Output vector size (" << y.Size()
                << ") must match graph height (" << graph->Height() << ")");

    auto in_offsets  = graph->InputOffsets();
    auto out_offsets = graph->OutputOffsets();

    BlockVector xb(x.GetData(), in_offsets);
    BlockVector yb(y.GetData(), out_offsets);

    auto inputs  = graph->InputFields();
    auto outputs = graph->OutputFields();

    for(int i=0; i < inputs.Size(); i++)
    {
        inputs[i]->SetAdjoint(&xb.GetBlock(i));
    }

    for(int i=0; i < outputs.Size(); i++)
    {
        outputs[i]->SetAdjoint(&yb.GetBlock(i));
    }

    auto in_type = graph->GetInputType();
    auto id_map  = graph->GetIdToIndexMap();
    auto field_map = graph->GetIdToFieldMap();

    if(in_type == InputType::VECTOR)
    {
        auto fadj_to_vec = [](Field &f, Vector &v) { v = *f.Adjoint(); };
        auto vec_to_fadj = [](Field &f, Vector &v) { *f.Adjoint() = v; };

        x0.SetSize(graph->MaxWidth());
        dx.SetSize(graph->MaxWidth());
        dy.SetSize(graph->MaxHeight());

        auto nodes = graph->Nodes();
        for (auto node : nodes)
        {
            x0.SetSize(node->Width());
            dx.SetSize(node->Width());
            dy.SetSize(node->Height());

            int ioff = 0;
            auto node_offsets = node->InputOffsets();
            for(auto input_field : node->InputFields())
            {
                MFEM_ASSERT(id_map.Has(input_field->ID()), "Input field ID not found in id_map");
                int idx = id_map.Get(input_field->ID());
                x0.SetVector(*x_fields[idx], node_offsets[ioff]);
                ioff++;
            }

            FieldBlockVectorTransfer(node->InputOffsets(), node->InputFields(), dx, fadj_to_vec);

            node->GradientMult(x0, dx, dy); // Compute JVP for the node

            FieldBlockVectorTransfer(node->OutputOffsets(), node->OutputFields(), dy, vec_to_fadj);
        }
    }
    else if(in_type == InputType::MULTIVECTOR)
    {
        MFEM_ABORT("GraphGradient::Forward() not implemented for input type: MULTIVECTOR");
    }
    else if(in_type == InputType::NONE)
    {
        Vector x_unused, dx_unused, dy_unused;
        auto nodes = graph->Nodes();
        for (auto node : nodes)
        {
            node->GradientMult(x_unused, dx_unused, dy_unused);
        }
    }
    else
    {
        MFEM_ABORT("GraphGradient::Forward() not implemented for input type: "
                    << static_cast<int>(in_type));
    }

}

void GraphGradient::Backward(const Vector &x, Vector &y) const
{
    auto in_offsets  = graph->InputOffsets();
    auto out_offsets = graph->OutputOffsets();

    // Note the switch of input and output offsets for backward mode
    BlockVector xb(x.GetData(), out_offsets);
    BlockVector yb(y.GetData(), in_offsets);

    auto outputs = graph->OutputFields();
    int noutputs = outputs.Size();
    for(int i=0; i < noutputs; i++)
    {
        auto field = outputs[i];
        field->SetAdjoint(&xb.GetBlock(i));
    }

    auto inputs = graph->InputFields();
    int ninputs = inputs.Size();
    for(int i=0; i < ninputs; i++)
    {
        auto field = inputs[i];
        field->SetAdjoint(&yb.GetBlock(i));
    }

    // Vector x0, dx, dy;
    int nnodes  = graph->Size();
    for (int i=nnodes-1; i >= 0; i--)
    {
        auto node = graph->GetNode(i);
        node->GradientMultTranspose(x0, dx, dy); // Compute VJP for the node
    }
}

Operator& GraphGradient::GetGradient(const Vector &x) const
{
    // Used to build Jacobian matrix
    MFEM_ABORT("GraphGradient::GetGradient() not implemented");
}


} // namespace mfem
#endif // MFEM_USE_MPI
