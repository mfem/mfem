# mfem Docker

We provide a [Dockerfile](Dockerfile) to build an ubuntu base image. You can use
this image for a demo of using mfem! 🎉️

Updated containers are built and deployed on merges to the main branch and releases.
If you want to request a build on demand, you can [manually run the workflow](https://docs.github.com/en/actions/managing-workflow-runs/manually-running-a-workflow) thanks to the workflow dispatch event.

### Usage

Here is how to build the container. Note that we build so it belongs to the same
namespace as the repository here. "ghcr.io" means "GitHub Container Registry" and
is the [GitHub packages](https://github.com/features/packages) registry that supports
 Docker images and other OCI artifacts. From the root of the repository:

```bash
$ docker build -f config/docker/Dockerfile -t ghcr.io/mfem/mfem-ubuntu-base .
```

or this directory:

```bash
$ docker build -f Dockerfile -t ghcr.io/mfem/mfem-ubuntu-base ../../
```

### Shell

To shell into a container (here is an example with ubuntu):

```bash
$ docker run -it ghcr.io/mfem/mfem-ubuntu-base bash
```

Off the bat, you can see mfem libraries are in your path so you can jump into development:

```bash
env | grep mfem
```
```bash
PKG_CONFIG_PATH=/opt/mfem-env/.spack-env/view/lib/pkgconfig:/opt/mfem-env/.spack-env/view/share/pkgconfig:/opt/mfem-env/.spack-env/view/lib64/pkgconfig
PWD=/opt/mfem-env
MANPATH=/opt/mfem-env/.spack-env/view/share/man:/opt/mfem-env/.spack-env/view/man:
CMAKE_PREFIX_PATH=/opt/mfem-env/.spack-env/view
SPACK_ENV=/opt/mfem-env
ACLOCAL_PATH=/opt/mfem-env/.spack-env/view/share/aclocal
LD_LIBRARY_PATH=/opt/mfem-env/.spack-env/view/lib:/opt/mfem-env/.spack-env/view/lib64
PATH=/opt/mfem-env/.spack-env/view/bin:/opt/view/bin:/opt/spack/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

#### Examples and MiniApps

If you want to develop a tool that _uses_ mfem, you can find the built libraries in:

```
$ ls /opt/mfem-env/.spack-env/view/
bin  etc  include  lib  libexec  sbin  share  var
```

And yes, this is the working directory when you shell into the container!
You can find the examples here:


```bash
cd share/mfem/examples
```
```bash
$ ./ex0
Options used:
   --mesh ../data/star.mesh
   --order 1
Number of unknowns: 101
   Iteration :   0  (B r, r) = 0.184259
   Iteration :   1  (B r, r) = 0.102754
   Iteration :   2  (B r, r) = 0.00558141
   Iteration :   3  (B r, r) = 1.5247e-05
   Iteration :   4  (B r, r) = 1.13807e-07
   Iteration :   5  (B r, r) = 6.27231e-09
   Iteration :   6  (B r, r) = 3.76268e-11
   Iteration :   7  (B r, r) = 6.07423e-13
   Iteration :   8  (B r, r) = 4.10615e-15
Average reduction factor = 0.140201
```

Try running a few, and look at the associated .cpp file for the source code!
You can also explore the "mini apps," also in share/mfem, but under miniapps.

```bash
# This is run from the examples directory
$ cd ../miniapps
```
```bash
$ ls
CMakeLists.txt  common            meshing  nurbs        shifted  toys
adjoint         electromagnetics  mtop     parelag      solvers
autodiff        gslib             navier   performance  tools
```

And an example in "toys"

```bash
cd toys
```
```bash
$ ./automata -no-vis
Options used:
   --num-steps 16
   --rule 90
   --no-visualization

Rule:
 111 110 101 100 011 010 001 000
  0   1   0   1   1   0   1   0

Applying rule...done.
```

Have fun!


#### Your own App
If you want to develop with your own code base
(and mfem as is in the container) you can bind to somewhere else in the container (e.g., src)

```bash
$ docker run -it ghcr.io/mfem/mfem-ubuntu-base -v $PWD:/src bash
```

In the above, we can pretend your project is in the present working directory (PWD) and we are
binding to source. You can then use the mfem in the container for development, and if you
want to distribute your library or app in a container, you can use the mfem container as the base.
