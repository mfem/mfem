# mfem Docker

We provide a [Dockerfile.base](Dockerfile.base) to build an ubuntu base image,
and a [Dockerfile](Dockerfile) to build a smaller one with a multi-stage build.
You can use this image for a demo of using mfem! 🎉️

Updated containers are built and deployed on merges to the main branch and releases.
If you want to request a build on demand, you can [manually run the workflow](https://docs.github.com/en/actions/managing-workflow-runs/manually-running-a-workflow) thanks to the workflow dispatch event.

## Usage

We provide two containers, which you can either build or use directly from
[GitHub packages](https://github.com/orgs/mfem/packages?repo_name=mfem).

 - `ghcr.io/mfem/mfem-ubuntu-base`: a "build from scratch" for mfem
 - `ghcr.io/mfem/mfem-ubuntu`: a quick build that uses the base container

In the above, "ghcr.io" means "GitHub Container Registry" and
is the [GitHub packages](https://github.com/features/packages) registry that supports
Docker images and other OCI artifacts.

### Ubuntu

> Use or build this container for a multi-stage, slimmer base to develop on top of mfem

Note that this container is provided on GitHub packages [here](https://github.com/mfem/mfem/pkgs/container/mfem-ubuntu)
so you don't need to build it. However, if you want to, you can do the following:

```bash
$ docker build -f config/docker/Dockerfile -t ghcr.io/mfem/mfem-ubuntu .
```

Note that this will pull the base image. If you want to rebuild it, see [ubuntu base](#ubuntu-base)
below. Once you have built (or prefer to pull) you can shell into the container as follows:

```bash
$ docker run -it ghcr.io/mfem/mfem-ubuntu
```

This smaller image has a view where everything is installed.

```bash
$ ls
bin  etc  include  lib  libexec  sbin  share  var
```

 - Examples are in share/mfem/examples
 - Examples are in share/mfem/miniapps

Using this container, if you want to develop a tool that _uses_ mfem, you can find the libraries / includes in:

```bash
$ ls include/ | grep mfem
mfem
mfem-performance.hpp
mfem.hpp
```

And yes, this is the working directory when you shell into the container!
You can find the examples here:


```bash
cd share/mfem/examples
```

Try quickly setting the `LD_LIBRARY_PATH` so we can see the shared libraries
we need:

```bash
export LD_LIBRARY_PATH=/opt/mfem-view/lib:$LD_LIBRARY_PATH
```

And then run:

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

Have fun! As a reminder, this container is ideal for developing your own
applications that might use mfem, or having a nice environment to test out
examples.


### Ubuntu Base

> Use this build for a development environment with spack and mfem

This container is also [provided on GitHub packages](https://github.com/mfem/mfem/pkgs/container/mfem-ubuntu-base),
however you can build it locally too:

```bash
$ docker build -f config/docker/Dockerfile.base -t ghcr.io/mfem/mfem-ubuntu-base .
```

To shell into the container:

```bash
$ docker run -it ghcr.io/mfem/mfem-ubuntu-base bash
```

Change directory to the mfem environment, setup spack, and activate the environment:

```bash
source /opt/spack/share/spack/setup-env.sh
cd /opt/mfem-env/
spack env activate .
```

Note that this environment is installing to the view at `/opt/view`. Since the environment
knows to install mfem from `/code` this means that you could make changes in the container (or bind
`/code` to your container) and then update spack:

```bash
# Note that concretization takes a hot minute!
$ spack install
```

And if you want to load mfem:

```bash
$ spack load mfem
$ env | grep mfem
```

In this development container, you can find the examples and miniapps alongside
mfem under `/code`:

```bash
cd /code/examples
```
```bash
$ ./ex0
```
```console
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

This container is likely ideal for someone that wants to develop mfem itself.
For other use cases, we recommend using the slimmer image. As an example,
if you want to develop with your own code base (and mfem as is in the container)
you can bind to somewhere else in the container (e.g., src)

```bash
$ docker run -it ghcr.io/mfem/mfem-ubuntu-base -v $PWD:/code bash
```

In the above, we can pretend your project is in the present working directory (PWD) and we are
binding to source. You can then use the mfem in the container for development, and if you
want to distribute your library or app in a container, you can use the mfem container as the base.
