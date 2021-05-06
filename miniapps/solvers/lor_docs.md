# Low-order refined classes

MFEM features several low-order refined (LOR) classes intended to make working with LOR discretizations and solvers more convenient.
The interface is designed to be convenient, flexible, and powerful: solvers with "default" options can be created in only one line of code, but much more customization is available if desired.

## Creating solvers with default options

The most basic LOR solver can be created as a one liner. For example, the following creates a direct solver based on a serial LOR discretization:

```c++
// Given BilinearForm `a` and list of essential DOFs, create direct LOR solver
LORSolver<UMFPackSolver> prec(a, ess_tdof_list);
```
This will automatically create a LOR discretization corresponding to `a`, assemble the system, and create a solver whose type is given by the template parameter.
For example, in parallel, one could do the following to create an AMG preconditioner instead of direct solver:
```c++
LORSolver<HypreBoomerAMG> prec(a, ess_tdof_list);
```

### What does `LORSolver` do?

In the above examples, `LORSolver<T>` does a couple of things:

* Creates a LOR mesh and finite element space
* Assembles a LOR version of the bilinear form `a`
* Creates a preconditioner of type `<T>` using the LOR-assembled version
* Creates a DOF permutation (if required) to transition between the LOR and high-order DOF numbering

## Creating discretizations

If one is interested in using the underlying LOR discretization, it can be created as follows:
```c++
LORDiscretization lor(a, ess_tdof_list);
SparseMatrix &A_lor = lor.GetAssembledMatrix();
```
The low-order refined finite element space can be accessed by `lor.GetFESpace()`.
In parallel, there is an analogous `ParLORDiscretization` class:
```c++
ParLORDiscretization lor(a, ess_tdof_list);
HypreParMatrix &A_lor = lor.GetAssembledMatrix();
```

## Passing options to solvers

Some solvers require special options passed to the constructor.
For example, this can be useful when creating AMS preconditioners, which require access to the finite element space:
```c++
ParLORDiscretization lor(a, ess_tdof_list);
LORSolver<HypreAMS> ams(lor, &lor.GetParFESpace());
```
In this case, parameters passed to the `LORSolver` constructor will be forwarded to the `HypreAMS` constructor.
Also, note that the underlying solver object can be accessed with the `GetSolver` member function.

## Assembling custom bilinear forms

In all of the above examples, the LOR discretization is based off of the high-order bilinear form: the same form is assembled on a LOR finite element space.
In some circumstances, it is desirable to assemble a custom bilinear form on the LOR space and *not* to reuse the high-order form.
For example, one may want to assemble several LOR bilinear forms, and avoid the overhead of creating a mesh and finite element space for each of them.
These capabilities can be accomplished with the `LOR` and `LORSolver` classes:
```c++
LORDiscretization lor(fes); // Create LOR version of fes, don't assemble any forms
BilinearForm a_lor(&lor.GetFESpace());
a_lor.AddDomainIntegrator(new MassIntegrator);
a_lor.Assemble();
LORSolver<DSmoother> D(a_lor.SpMat(), lor);
```
In this case, the object `lor` does not assemble any forms, and does not represent any particular operator.
Instead, it simply represents a LOR mesh and finite element space, which can be used to create custom bilinear forms.
Once assembled, these operators can be passed to `LORSolver`, which will take care of any low-order-to-high-order DOF permutations, if needed.
