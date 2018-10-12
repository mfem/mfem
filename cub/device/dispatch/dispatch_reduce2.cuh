
/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * cub::DeviceReduce2 provides device-wide, parallel operations for computing a reduction across a sequence of data items residing within device-accessible memory.
 */

#pragma once

#include <stdio.h>
#include <iterator>

#include "../../agent/agent_reduce.cuh"
#include "../../agent/agent_reduce2.cuh"
#include "../../iterator/arg_index_input_iterator.cuh"
#include "../../thread/thread_operators.cuh"
#include "../../grid/grid_even_share.cuh"
#include "../../iterator/arg_index_input_iterator.cuh"
#include "../../util_debug.cuh"
#include "../../util_device.cuh"
#include "../../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * Reduce region kernel entry point (multi-block).  Computes privatized reductions, one per thread block.
 */
template <
    typename                ChainedPolicyT,             ///< Chained tuning policy
    typename                InputIteratorT,             ///< Random-access input iterator type for reading input items \iterator
    typename                OutputIteratorT,            ///< Output iterator type for recording the reduced aggregate \iterator
    typename                OffsetT,                    ///< Signed integer type for global offsets
    typename                LoadOpT,
    typename                ReductionOpT>               ///< Binary reduction functor type having member <tt>T operator()(const T &a, const T &b)</tt>
__launch_bounds__ (int(ChainedPolicyT::ActivePolicy::ReducePolicy::BLOCK_THREADS))
__global__ void DeviceReduceKernel2i(
    InputIteratorT          d_in0,                      ///< [in] Pointer to the input sequence of data items
    InputIteratorT          d_in1,                      ///< [in] Pointer to the input sequence of data items
    OutputIteratorT         d_out,                      ///< [out] Pointer to the output aggregate
    OffsetT                 num_items,                  ///< [in] Total number of input data items
    GridEvenShare<OffsetT>  even_share,                 ///< [in] Even-share descriptor for mapping an equal number of tiles onto each thread block
    LoadOpT                 load_op,
    ReductionOpT            reduction_op)               ///< [in] Binary reduction functor
{
    // The output value type
    typedef typename If<(Equals<typename std::iterator_traits<OutputIteratorT>::value_type, void>::VALUE),  // OutputT =  (if output iterator's value type is void) ?
        typename std::iterator_traits<InputIteratorT>::value_type,                                          // ... then the input iterator's value type,
        typename std::iterator_traits<OutputIteratorT>::value_type>::Type OutputT;                          // ... else the output iterator's value type

    // Thread block type for reducing input tiles
    typedef AgentReduce2i<
            typename ChainedPolicyT::ActivePolicy::ReducePolicy,
            InputIteratorT,
            OutputIteratorT,
            OffsetT,
            LoadOpT,
            ReductionOpT>
        AgentReduce2iT;

    // Shared memory storage
    __shared__ typename AgentReduce2iT::TempStorage temp_storage;

    // Consume input tiles
    OutputT block_aggregate = AgentReduce2iT(temp_storage, d_in0,d_in1, load_op,reduction_op).ConsumeTiles(even_share);

    // Output result
    if (threadIdx.x == 0)
        d_out[blockIdx.x] = block_aggregate;
}


/**
 * Reduce a single tile kernel entry point (single-block).  Can be used to aggregate privatized thread block reductions from a previous multi-block reduction pass.
 */
template <
    typename                ChainedPolicyT,             ///< Chained tuning policy
    typename                InputIteratorT,             ///< Random-access input iterator type for reading input items \iterator
    typename                OutputIteratorT,            ///< Output iterator type for recording the reduced aggregate \iterator
    typename                OffsetT,                    ///< Signed integer type for global offsets
    typename                LoadOpT,
    typename                ReductionOpT,               ///< Binary reduction functor type having member <tt>T operator()(const T &a, const T &b)</tt>
    typename                OuputT>                     ///< Data element type that is convertible to the \p value type of \p OutputIteratorT
__launch_bounds__ (int(ChainedPolicyT::ActivePolicy::SingleTilePolicy::BLOCK_THREADS), 1)
__global__ void DeviceReduceSingleTileKernel2i(
    InputIteratorT          d_in0,                      ///< [in] Pointer to the input sequence of data items
    InputIteratorT          d_in1,                      ///< [in] Pointer to the input sequence of data items
    OutputIteratorT         d_out,                      ///< [out] Pointer to the output aggregate
    OffsetT                 num_items,                  ///< [in] Total number of input data items
    LoadOpT                 load_op,
    ReductionOpT            reduction_op,               ///< [in] Binary reduction functor
    OuputT                  init)                       ///< [in] The initial value of the reduction
{
    // Thread block type for reducing input tiles
    typedef AgentReduce<
            typename ChainedPolicyT::ActivePolicy::SingleTilePolicy,
            InputIteratorT,
            OutputIteratorT,
            OffsetT,
            ReductionOpT>
        AgentReduce1T;
    typedef AgentReduce2i<
            typename ChainedPolicyT::ActivePolicy::SingleTilePolicy,
            InputIteratorT,
            OutputIteratorT,
            OffsetT,
            LoadOpT,
            ReductionOpT>
      AgentReduce2iT;

    // Shared memory storage
    __shared__ typename AgentReduce1T::TempStorage temp1_storage;
    __shared__ typename AgentReduce2iT::TempStorage temp2_storage;

    // Check if empty problem
    if (num_items == 0)
    {
        if (threadIdx.x == 0)
            *d_out = init;
        return;
    }

    // Consume input tiles
    OuputT block_aggregate = (d_in1==NULL)?
      AgentReduce1T(temp1_storage, d_in0, reduction_op).ConsumeRange(OffsetT(0),num_items):
      AgentReduce2iT(temp2_storage, d_in0,d_in1, load_op,reduction_op).ConsumeRange(OffsetT(0),num_items);

    // Output result
    if (threadIdx.x == 0)
        *d_out = reduction_op(init, block_aggregate);
}



// *************************************************************************************************************************************************************
// *************************************************************************************************************************************************************
template <
    typename InputIteratorT,    ///< Random-access input iterator type for reading input items \iterator
    typename OutputIteratorT,   ///< Output iterator type for recording the reduced aggregate \iterator
    typename OffsetT,           ///< Signed integer type for global offsets
    typename LoadOpT,           ///< Binary load functor type having member <tt>T operator()(const T &a, const T &b)</tt> 
    typename ReductionOpT>      ///< Binary reduction functor type having member <tt>T operator()(const T &a, const T &b)</tt> 
struct DispatchReduce2i :
    DeviceReducePolicy<
        typename If<(Equals<typename std::iterator_traits<OutputIteratorT>::value_type, void>::VALUE),  // OutputT =  (if output iterator's value type is void) ?
            typename std::iterator_traits<InputIteratorT>::value_type,                                  // ... then the input iterator's value type,
            typename std::iterator_traits<OutputIteratorT>::value_type>::Type,                          // ... else the output iterator's value type
        OffsetT,
        ReductionOpT>
{
    //------------------------------------------------------------------------------
    // Constants
    //------------------------------------------------------------------------------

    // Data type of output iterator
    typedef typename If<(Equals<typename std::iterator_traits<OutputIteratorT>::value_type, void>::VALUE),  // OutputT =  (if output iterator's value type is void) ?
        typename std::iterator_traits<InputIteratorT>::value_type,                                          // ... then the input iterator's value type,
        typename std::iterator_traits<OutputIteratorT>::value_type>::Type OutputT;                          // ... else the output iterator's value type


    //------------------------------------------------------------------------------
    // Problem state
    //------------------------------------------------------------------------------

    void                *d_temp_storage;                ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
    size_t              &temp_storage_bytes;            ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
    InputIteratorT      d_in0;                          ///< [in] Pointer to the input sequence of data items
    InputIteratorT      d_in1;                          ///< [in] Pointer to the input sequence of data items
    OutputIteratorT     d_out;                          ///< [out] Pointer to the output aggregate
    OffsetT             num_items;                      ///< [in] Total number of input items (i.e., length of \p d_in)
    LoadOpT             load_op;                        ///< [in] Binary load functor 
    ReductionOpT        reduction_op;                   ///< [in] Binary reduction functor 
    OutputT             init;                           ///< [in] The initial value of the reduction
    cudaStream_t        stream;                         ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
    bool                debug_synchronous;              ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    int                 ptx_version;                    ///< [in] PTX version

    //------------------------------------------------------------------------------
    // Constructor
    //------------------------------------------------------------------------------

    /// Constructor
    CUB_RUNTIME_FUNCTION __forceinline__
    DispatchReduce2i(
        void*                   d_temp_storage,
        size_t                  &temp_storage_bytes,
        InputIteratorT          d_in0,
        InputIteratorT          d_in1,
        OutputIteratorT         d_out,
        OffsetT                 num_items,
        LoadOpT                 load_op,
        ReductionOpT            reduction_op,
        OutputT                 init,
        cudaStream_t            stream,
        bool                    debug_synchronous,
        int                     ptx_version)
    :
        d_temp_storage(d_temp_storage),
        temp_storage_bytes(temp_storage_bytes),
        d_in0(d_in0),d_in1(d_in1),
        d_out(d_out),
        num_items(num_items),
        load_op(load_op),
        reduction_op(reduction_op),
        init(init),
        stream(stream),
        debug_synchronous(debug_synchronous),
        ptx_version(ptx_version)
    {}


    //------------------------------------------------------------------------------
    // Small-problem (single tile) invocation
    //------------------------------------------------------------------------------

    /// Invoke a single block block to reduce in-core
    template <
        typename                ActivePolicyT,          ///< Umbrella policy active for the target device
        typename                SingleTileKernelT>      ///< Function type of cub::DeviceReduceSingleTileKernel
    CUB_RUNTIME_FUNCTION __forceinline__
    cudaError_t InvokeSingleTile2i(SingleTileKernelT single_tile_kernel2)  ///< [in] Kernel function pointer to parameterization of cub::DeviceReduceSingleTileKernel
    {
#ifndef CUB_RUNTIME_ENABLED
        (void)single_tile_kernel2;

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported );
#else
        cudaError error = cudaSuccess;
        do
        {
            // Return if the caller is simply requesting the size of the storage allocation
            if (d_temp_storage == NULL)
            {
                temp_storage_bytes = 1;
                break;
            }

            // Log single_reduce_sweep_kernel configuration
            if (debug_synchronous) _CubLog("Invoking DeviceReduceSingleTileKernel<<<1, %d, 0, %lld>>>(), %d items per thread\n",
                ActivePolicyT::SingleTilePolicy::BLOCK_THREADS,
                (long long) stream,
                ActivePolicyT::SingleTilePolicy::ITEMS_PER_THREAD);

            // Invoke single_reduce_sweep_kernel // DeviceReduceSingleTileKernel2i
            single_tile_kernel2<<<1, ActivePolicyT::SingleTilePolicy::BLOCK_THREADS, 0, stream>>>(
                d_in0, d_in1,
                d_out,
                num_items,
                load_op,
                reduction_op,
                init);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;
        }
        while (0);

        return error;

#endif // CUB_RUNTIME_ENABLED
    }


    //------------------------------------------------------------------------------
    // Normal problem size invocation (two-pass)
    //------------------------------------------------------------------------------

    /// Invoke two-passes to reduce
    template <
        typename                ActivePolicyT,              ///< Umbrella policy active for the target device
        typename                ReduceKernelT,              ///< Function type of cub::DeviceReduceKernel
        typename                SingleTileKernelT>          ///< Function type of cub::DeviceReduceSingleTileKernel
    CUB_RUNTIME_FUNCTION __forceinline__
    cudaError_t InvokePasses2i(
        ReduceKernelT           reduce_kernel,          ///< [in] Kernel function pointer to parameterization of cub::DeviceReduceKernel
        SingleTileKernelT       single_tile_kernel) // DeviceReduceSingleTileKernel2i  ///< [in] Kernel function pointer to parameterization of cub::DeviceReduceSingleTileKernel
    {
#ifndef CUB_RUNTIME_ENABLED
        (void)                  reduce_kernel;
        (void)                  single_tile_kernel;

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported );
#else

        cudaError error = cudaSuccess;
        do
        {
            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Get SM count
            int sm_count;
            if (CubDebug(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

            // Init regular kernel configuration
            KernelConfig reduce_config;
            if (CubDebug(error = reduce_config.Init<typename ActivePolicyT::ReducePolicy>(reduce_kernel))) break;
            int reduce_device_occupancy = reduce_config.sm_occupancy * sm_count;

            // Even-share work distribution
            int max_blocks = reduce_device_occupancy * CUB_SUBSCRIPTION_FACTOR(ptx_version);
            GridEvenShare<OffsetT> even_share;
            even_share.DispatchInit(num_items, max_blocks, reduce_config.tile_size);

            // Temporary storage allocation requirements
            void* allocations[1];
            size_t allocation_sizes[1] =
            {
                max_blocks * sizeof(OutputT)    // bytes needed for privatized block reductions
            };

            // Alias the temporary allocations from the single storage blob (or compute the necessary size of the blob)
            if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
            if (d_temp_storage == NULL)
            {
                // Return if the caller is simply requesting the size of the storage allocation
                return cudaSuccess;
            }

            // Alias the allocation for the privatized per-block reductions
            OutputT *d_block_reductions = (OutputT*) allocations[0];

            // Get grid size for device_reduce_sweep_kernel
            int reduce_grid_size = even_share.grid_size;

            // Log device_reduce_sweep_kernel configuration
            if (debug_synchronous) _CubLog("Invoking DeviceReduceKernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                reduce_grid_size,
                ActivePolicyT::ReducePolicy::BLOCK_THREADS,
                (long long) stream,
                ActivePolicyT::ReducePolicy::ITEMS_PER_THREAD,
                reduce_config.sm_occupancy);

            // Invoke DeviceReduceKernel2
            reduce_kernel<<<reduce_grid_size, ActivePolicyT::ReducePolicy::BLOCK_THREADS, 0, stream>>>(
                d_in0,d_in1,
                d_block_reductions,
                num_items,
                even_share,
                load_op,
                reduction_op);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

            // Log single_reduce_sweep_kernel configuration
            if (debug_synchronous) _CubLog("Invoking DeviceReduceSingleTileKernel<<<1, %d, 0, %lld>>>(), %d items per thread\n",
                ActivePolicyT::SingleTilePolicy::BLOCK_THREADS,
                (long long) stream,
                ActivePolicyT::SingleTilePolicy::ITEMS_PER_THREAD);

            // Invoke DeviceReduceSingleTileKernel
            single_tile_kernel<<<1, ActivePolicyT::SingleTilePolicy::BLOCK_THREADS, 0, stream>>>(
                d_block_reductions,NULL,
                d_out,
                reduce_grid_size,
                load_op,
                reduction_op,
                init);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;
        }
        while (0);

        return error;

#endif // CUB_RUNTIME_ENABLED

    }


    //------------------------------------------------------------------------------
    // Chained policy invocation
    //------------------------------------------------------------------------------

    /// Invocation
    template <typename ActivePolicyT>
    CUB_RUNTIME_FUNCTION __forceinline__
    cudaError_t Invoke()
    {
        typedef typename ActivePolicyT::SingleTilePolicy    SingleTilePolicyT;
        typedef typename DispatchReduce2i::MaxPolicy        MaxPolicyT;

        // Force kernel code-generation in all compiler passes
        if (num_items <= (SingleTilePolicyT::BLOCK_THREADS * SingleTilePolicyT::ITEMS_PER_THREAD))
        {
            // Small, single tile size
            return InvokeSingleTile2i<ActivePolicyT>(DeviceReduceSingleTileKernel2i<MaxPolicyT, InputIteratorT, OutputIteratorT, OffsetT, LoadOpT,ReductionOpT, OutputT>);
        }
        else
        {
            // Regular size
            return InvokePasses2i<ActivePolicyT>(DeviceReduceKernel2i<typename DispatchReduce2i::MaxPolicy, InputIteratorT, OutputT*, OffsetT, LoadOpT,ReductionOpT>,
                                                 DeviceReduceSingleTileKernel2i<MaxPolicyT, OutputT*, OutputIteratorT, OffsetT, LoadOpT,ReductionOpT, OutputT>);
        }
    }


    //------------------------------------------------------------------------------
    // Dispatch entrypoints
    //------------------------------------------------------------------------------

    /**
     * Internal dispatch routine for computing a device-wide reduction
     */
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void            *d_temp_storage,                    ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t          &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT  d_in0,                              ///< [in] Pointer to the input sequence of data items
        InputIteratorT  d_in1,                              ///< [in] Pointer to the input sequence of data items
        OutputIteratorT d_out,                              ///< [out] Pointer to the output aggregate
        OffsetT         num_items,                          ///< [in] Total number of input items (i.e., length of \p d_in)
        LoadOpT         load_op,                            ///< [in] Binary load functor 
        ReductionOpT    reduction_op,                       ///< [in] Binary reduction functor 
        OutputT         init,                               ///< [in] The initial value of the reduction
        cudaStream_t    stream,                             ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool            debug_synchronous)                  ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        typedef typename DispatchReduce2i::MaxPolicy MaxPolicyT;

        cudaError error = cudaSuccess;
        do
        {
            // Get PTX version
            int ptx_version;
            if (CubDebug(error = PtxVersion(ptx_version))) break;

            // Create dispatch functor
            DispatchReduce2i dispatch(
                d_temp_storage, temp_storage_bytes,
                d_in0, d_in1, d_out, num_items, load_op, reduction_op, init,
                stream, debug_synchronous, ptx_version);

            // Dispatch to chained policy
            if (CubDebug(error = MaxPolicyT::Invoke(ptx_version, dispatch))) break;
        }
        while (0);

        return error;
    }
};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


