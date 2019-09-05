Provides JIT feature for MFEM's kernels.

A MFEM preprocessor tool (`mpp`) is built to parse the files in the `*/kernels/` sub-directories.
It generates the calls to the memory manager, the prefix/postfix code for the JIT and avoids all the `d_` in the kernel body.

The keywords now available are: 
- `__kernel`: generates the calls to the memory manager as a prefix to the kernel
- `__jit` (as a `__kernel` qualifier): embeds the source and generates the prefix/postfix code to hash the source, the compiler/options used for MFEM and the kernel arguments. The kernel code needs to be standalone and not to rely yet on anything outside of it.
- `__template` (as a `__kernel` qualifier): transforms the kernel to a templated one, generating the instantiation table needed by the result of the cartesian product of the argument ranges.
- `__range` (as an argument qualifier of a templated kernel): sets the argument range used while building the instantiation table.

A `vector` kernel can now be written like:
```
__kernel void kVectorAssign(const size_t N, const double* v, double *data)
{
   MFEM_FORALL(i, N, data[i] = v[i];);
}
```

The `kGeom2D` kernel can also be called directly, without all the template instantiations:
``` 
__jit __kernel 
void kGeom2D(const int NUM_DOFS_1D,
             const int NUM_QUAD_1D,
             const int numElements,
             const double* __restrict dofToQuadD,
             const double* __restrict nodes,
             double* __restrict J,
             double* __restrict invJ,
             double* __restrict detJ)
{
   const int NUM_DOFS = NUM_DOFS_1D*NUM_DOFS_1D;
   const int NUM_QUAD = NUM_QUAD_1D*NUM_QUAD_1D;
   MFEM_FORALL(e, numElements,
...
```
And if we don't want/have the JIT support but still want some compiled-time performances, we can instantiate a range of templated kernels:

```
__template __kernel
void tGeom2D(const int __range(1,2-3) NUM_DOFS_1D,
             const int __range(2-16) NUM_QUAD_1D,
             const int numElements,
             const double* __restrict dofToQuadD,
             const double* __restrict nodes,
             double* __restrict J,
             double* __restrict invJ,
             double* __restrict detJ){...
```

To build:
`make config MFEM_USE_JIT=YES` and `MFEM_USE_MM=YES` to add the pointer translations.
