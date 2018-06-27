<p align="center">
<a href="http://mfem.org/"><img alt="mfem" src="http://mfem.org/img/logo-300.png"></a>
</p>

<p align="center">
<a href="https://github.com/mfem/mfem/blob/master/COPYRIGHT"><img alt="License" src="https://img.shields.io/badge/License-LGPL--2.1-brightgreen.svg"></a>
<a href="https://travis-ci.org/mfem/mfem"><img alt="Build Status" src="https://travis-ci.org/mfem/mfem.svg?branch=master"></a>
<a href="https://ci.appveyor.com/project/mfem/mfem"><img alt="Build Status" src="https://ci.appveyor.com/api/projects/status/19non9sqm6msi2wy?svg=true"></a>
<a href="http://mfem.github.io/doxygen/html/index.html"><img alt="Doxygen" src="https://img.shields.io/badge/code-documented-brightgreen.svg"></a>
</p>


# How to Contribute

The MFEM team welcomes contributions at all levels: bugfixes; code
improvements; simplifications; new mesh, discretization or solver
capabilities; improved documentation; new examples and miniapps;
HPC performance improvements; ...

Use a pull request (PR) toward the `mfem:master` branch to propose your
contribution. If you are planning significant code changes, or have any
questions, you can also open an [issue](https://github.com/mfem/mfem/issues)
before issuing a PR.  We also welcome your [simulation
images](http://mfem.org/gallery/), which you can submit via a pull request in
[mfem/web](https://github.com/mfem/web).

See the [Quick Summary](#quick-summary) section for the main highlights of our
GitHub workflow. For more details, consult the following sections and refer
back to them before issuing pull requests:

- [Code Overview](#code-overview)
- [GitHub Workflow](#github-workflow)
  - [MFEM Organization](#mfem-organization)
  - [New Feature Development](#new-feature-development)
  - [Developer Guidelines](#developer-guidelines)
  - [Pull Requests](#pull-requests)
  - [Pull Request Checklist](#pull-request-checklist)
  - [Master/Next Workflow](#masternext-workflow)
  - [Releases](#releases)
  - [Release Checklist](#release-checklist)
- [LLNL Workflow](#llnl-workflow)
- [Automated Testing](#automated-testing)
- [Contact Information](#contact-information)

Contributing to MFEM requires knowledge of Git and, likely, finite elements. If
you are new to Git, see the [GitHub learning
resources](https://help.github.com/articles/git-and-github-learning-resources/).
To learn more about the finite element method, see our [FEM page](http://mfem.org/fem).

*By submitting a pull request, you are affirming the [Developer's Certificate of
Origin](#developers-certificate-of-origin-11) at the end of this file.*


## Quick Summary

- We encourage you to [join the MFEM organization](#mfem-organization) and create
  development branches off `mfem:master`.
- Please follow the [developer guidelines](#developer-guidelines), in particular
  with regards to documentation and code styling.
- Pull requests  should be issued toward `mfem:master`. Make sure
  to check the items off the [Pull Request Checklist](#pull-request-checklist).
- After approval, MFEM developers merge the PR manually in the [mfem:next branch](#masternext-workflow).
- After a week of testing in `mfem:next`, the original PR is merged in `mfem:master`.
- We use [milestones](https://github.com/mfem/mfem/milestones) to coordinate the
  work on different PRs toward a release.
- Don't hesitate to [contact us](#contact-information) if you have any questions.


### Code Overview

- The MFEM library uses object-orient design principles which reflect, in code,
  the independent mathematical concepts of meshing, linear algebra and finite
  element spaces and operators.

- The MFEM source code has the following structure:
  ```
  .
  ├── config
  │   └── cmake
  │       └── modules
  ├── data
  ├── doc
  │   └── web
  │       └── examples
  ├── examples
  │   ├── petsc
  │   ├── pumi
  │   └── sundials
  ├── fem
  ├── general
  ├── linalg
  ├── mesh
  └── miniapps
      ├── common
      ├── electromagnetics
      ├── meshing
      ├── nurbs
      ├── performance
      └── tools
  ```

- The main directories are `fem/`, `mesh/` and `linalg/` containing the C++
  classes implementing the finite element, mesh and linear algebra concepts
  respectively.

- The main mesh classes are:
  + [`Mesh`](http://mfem.github.io/doxygen/html/classmfem_1_1Mesh.html)
  + [`NCMesh`](http://mfem.github.io/doxygen/html/classmfem_1_1NCMesh.html)
  + [`Element`](http://mfem.github.io/doxygen/html/classmfem_1_1Element.html)
  + [`ElementTransformation`](http://mfem.github.io/doxygen/html/classmfem_1_1ElementTransformation.html)

- The main finite element classes are:
  + [`FiniteElement`](http://mfem.github.io/doxygen/html/classmfem_1_1FiniteElement.html)
  + [`FiniteElementCollection`](http://mfem.github.io/doxygen/html/classmfem_1_1FiniteElement.html)
  + [`FiniteElementSpace`](http://mfem.github.io/doxygen/html/classmfem_1_1FiniteElementSpace.html)
  + [`GridFunction`](http://mfem.github.io/doxygen/html/classmfem_1_1GridFunction.html)
  + [`BilinearFormIntegrator`](http://mfem.github.io/doxygen/html/classmfem_1_1BilinearFormIntegrator.html) and [`LinearFormIntegrator`](http://mfem.github.io/doxygen/html/classmfem_1_1LinearFormIntegrator.html)
  + [`LinearForm`](http://mfem.github.io/doxygen/html/classmfem_1_1LinearFormIntegrator.html), [`BilinearForm`](http://mfem.github.io/doxygen/html/classmfem_1_1BilinearForm.html) and [`MixedBilinearForm`](http://mfem.github.io/doxygen/html/classmfem_1_1MixedBilinearForm.html)

- The main linear algebra classes and sources are
  + [`Operator`](http://mfem.github.io/doxygen/html/classmfem_1_1Operator.html) and [`BilinearForm`](http://mfem.github.io/doxygen/html/classmfem_1_1BilinearForm.html)
  + [`Vector`](http://mfem.github.io/doxygen/html/classmfem_1_1BilinearForm.html) and [`LinearForm`](http://mfem.github.io/doxygen/html/classmfem_1_1LinearForm.html)
  + [`DenseMatrix`](http://mfem.github.io/doxygen/html/classmfem_1_1DenseMatrix.html) and [`SparseMatrix`](http://mfem.github.io/doxygen/html/classmfem_1_1SparseMatrix.html)
  + Sparse [smoothers](http://mfem.github.io/doxygen/html/sparsesmoothers_8hpp.html) and linear [solvers](http://mfem.github.io/doxygen/html/solvers_8hpp.html)

- Parallel MPI objects in MFEM inherit their serial counterparts, so a parallel
  mesh for example is just a serial mesh on each task plus the information on
  shared geometric entities between different tasks. The parallel source files
  have a `p` prefix, e.g. `pmesh.cpp` vs. the serial `mesh.cpp`.

- The main parallel classes are
  + [`ParMesh`](http://mfem.github.io/doxygen/html/solvers_8hpp.html)
  + [`ParNCMesh`](http://mfem.github.io/doxygen/html/classmfem_1_1ParMesh.html)
  + [`ParFiniteElementSpace`](http://mfem.github.io/doxygen/html/classmfem_1_1ParFiniteElementSpace.html)
  + [`ParGridFunction`](http://mfem.github.io/doxygen/html/classmfem_1_1ParGridFunction.html)
  + [`ParBilinearForm`](http://mfem.github.io/doxygen/html/classmfem_1_1ParBilinearForm.html) and [`ParLinearForm`](http://mfem.github.io/doxygen/html/classmfem_1_1ParLinearForm.html)
  + [`HypreParMatrix`](http://mfem.github.io/doxygen/html/classmfem_1_1HypreParMatrix.html) and [`HypreParVector`](http://mfem.github.io/doxygen/html/classmfem_1_1HypreParVector.html)
  + [`HypreSolver`](http://mfem.github.io/doxygen/html/classmfem_1_1HypreSolver.html) and other [hypre classes](http://mfem.github.io/doxygen/html/hypre_8hpp.html)

- The `general/` directory contains C++ classes that serve as utilities for
  communication, error handling, arrays, (Boolean) tables, timing, etc.

- The `config/` directory contains build-related files, both for the plain
  Makefile and the CMake build options.

- The `doc/` directory contains configuration for the Doxygen code documentation
  that can either be build locally, or browsed online at
  http://mfem.github.io/doxygen/html/index.html.

- The `data/` directory contains a collection of small mesh files, that are used
  in the simple example codes and more fully-featured mini applications in the
  `examples/` and `miniapps/` directories.

- See also the [code overview](http://mfem.org/code-overview/) section on the
  MFEM website.

## GitHub Workflow

The GitHub organization, https://github.com/mfem, is the main developer hub for
the MFEM project.

If you plan to make contributions or will like to stay up-to-date with changes
in the code, *we strongly encourage you to [join the MFEM organization](#mfem-organization)*.

This will simplify the workflow (by providing you additional permissions), and
will allow us to reach you directly with project announcements.


### MFEM Organization

- Before you can start, you need a GitHub account, here are a few suggestions:
  + Create the account at: github.com/join.
  + For easy identification, please add your name and maybe a picture of you at: https://github.com/settings/profile.
  + To receive notification, set a primary email at: https://github.com/settings/emails.
  + For password-less pull/push over SSH, add your SSH keys at: https://github.com/settings/keys.

- [Contact us](#contact-information) for an invitation to join the MFEM GitHub
  organization.

- You should receive an invitation email, which you can directly accept.
  Alternatively, *after logging into GitHub*, you can accept the invitation at
  the top of https://github.com/mfem.

- Consider making your membership public by going to https://github.com/orgs/mfem/people
  and clicking on the organization visibility dropbox next to your name.

- Project discussions and announcements will be posted at
  https://github.com/orgs/mfem/teams/everyone.

- The MFEM source code is in the [mfem](https://github.com/mfem/mfem)
  repository.

- The website and corresponding documentation are in the
  [web](https://github.com/mfem/web) repository.

- The [PyMFEM](https://github.com/mfem/PyMFEM) repository contains a Python
  wrapper for MFEM.

- The [data](https://github.com/mfem/data) repository contains additional
  (large) datafiles for MFEM.


### New Feature Development

- A new feature should be important enough that at least one person, the
  proposer, is willing to work on it and be its champion.

- The proposer creates a branch for the new feature (with suffix `-dev`), off
  the `master` branch, or another existing feature branch, for example:

  ```
  # Clone assuming you have setup your ssh keys on GitHub:
  git clone git@github.com:mfem/mfem.git

  # Alternatively, clone using the "https" protocol:
  git clone https://github.com/mfem/mfem.git

  # Create a new feature branch starting from "master":
  git checkout master
  git pull
  git checkout -b feature-dev

  # Work on "feature-dev", add local commits
  # ...

  # (One time only) push the branch to github and setup your local
  # branch to track the github branch (for "git pull"):
  git push -u origin feature-dev

  ```

- **We prefer that you create the new feature branch inside the MFEM organization
  as opposed to in a fork.** This allows everyone in the community to collaborate
  in one central place.

  - If you prefer to work in your fork, please [enable upstream edits](https://help.github.com/articles/allowing-changes-to-a-pull-request-branch-created-from-a-fork/).

  - Never use the `next` branch to start a new feature branch!

- The typical feature branch name is `new-feature-dev`, e.g. `pumi-dev`. While
  not frequent in MFEM, other suffixes are possible, e.g. `-fix`, `-doc`, etc.


### Developer Guidelines

- *Keep the code lean and as simple as possible*
  - Well-designed simple code is frequently more general and powerful.
  - Lean code base is easier to understand by new collaborators.
  - New features should be added only if they are necessary or generally useful.
  - Introduction of language constructions not currently used in MFEM should be
    justified and generally avoided (so we can build on cutting-edge systems).
  - We prefer basic C++ and the C++03 standard, to keep the code readable by
    a large audience and to make sure it compiles anywhere.

- *Keep the code general and reasonably efficient*
  - Main goal is fast prototyping for research.
  - When in doubt, generality wins over efficiency.
  - Respect the needs of different users (current and/or future).

- *Keep things separate and logically organized*
  - General usage features go in MFEM (implemented in as much generality as
    possible), non-general features go into external apps.
  - Inside MFEM, compartmentalize between linalg, fem, mesh, GLVis, etc.
  - Contributions that are project-specific or have external dependencies are
    allowed (if they are of broader interest), but should be `#ifdef`-ed and not
    change the code by default.

- Code specifics
  - All significant new classes, methods and functions have Doxygen-style
    documentation in source comments.
  - Consistent code styling is enforced with `make style` in the top-level
    directory. This requires [Artistic Style](http://astyle.sourceforge.net) (we
    specifically use version 2.05.1). See also the file `config/mfem.astylerc`.
  - Use `mfem::out` and `mfem::err` instead of `std::cout` and `std::cerr` in
    internal library code. (You can use `std` in examples and miniapps.)
  - When manually resolving conflicts during a merge, make sure to mention the
    conflicted files in the commit message.

### Pull Requests

- When your branch is ready for other developers to review / comment on
  the code, create a pull request towards `mfem:master`.

- Pull request typically have titles like:

     `Description [new-feature-dev]`

  for example:

     `Parallel Unstructured Mesh Infrastructure (PUMI) integration [pumi-dev]`

  Note the branch name suffix (in square brackets).

- Titles may contain a prefix in square brackets to emphasize the type of PR.
  Common choices are: `[DON'T MERGE]`, `[WIP]` and `[DISCUSS]`, for example:

     `[DISCUSS] Hybridized DG [hdg-dev]`

- Add a description, appropriate labels and assign yourself to the PR. The MFEM
  team will add reviewers as appropriate.

- List outstanding TODO items in the description, see PR #222 for an example.

- Track the Travis CI and Appveyor [continuous integration](#automated-testing)
  builds at the end of the PR. These should run clean, so address any errors as
  soon as possible.


### Pull Request Checklist

Before a PR can be merged, it should satisfy the following:

- [ ] Code builds.
- [ ] Code passes `make style`.
- [ ] Update `CHANGELOG`:
    - [ ] Is this a new feature users need to be aware of? New or updated example or miniapp?
    - [ ] Does it make sense to create a new section in the `CHANGELOG` to group with other related features?
- [ ] Update `INSTALL`:
    - [ ] Has a new optional library been added? (*Make sure the external library is licensed under LGPL, not GPL!*)
    - [ ] Does `make` or `cmake` have a new target?
    - [ ] Did the requirements or the installation process change? *(rare)*.
- [ ] Update `.gitignore`:
    - [ ] Check if `make distclean; git status` shows any files that are generated from the source but we don't want to track in the repository.
    - [ ] Add new patterns (just for the new files above) and re-run the above test.
- [ ] New examples:
    - [ ] All sample runs at the top of the example work.
    - [ ] Update `examples/makefile`:
      - [ ] Add the example code to the appropriate `SEQ_EXAMPLES` and `PAR_EXAMPLES` variables.
      - [ ] Add any files generated by it to the `clean` target.
      - [ ] Add the example binary and any files generated by it to the top-level `.gitignore` file.
    - [ ] Update `examples/CMakeLists.txt`:
      - [ ] Add the example code to the `ALL_EXE_SRCS` variable.
      - [ ] Make sure `THIS_TEST_OPTIONS` is set correctly for the new example.
   - [ ] List the new example in `doc/CodeDocumentation.dox`.
   - [ ] Companion pull request for documentation in [mfem/web](https://github.com/mfem/web) repo:
      - [ ] Update or add example-specific documentation, see e.g. the `src/examples.md`.
      - [ ] Add the description, labels and screenshots in `src/examples.md` and `src/img`.
      - [ ] In `examples.md`, list the example under the appropriate categories, add new categories if necessary.
      - [ ] Add a short description of the example in the "Extensive Examples" section of `features.md`.
- [ ] New miniapps:
   - [ ] All sample runs at the top of the miniapp work.
   - [ ] Update top-level `makefile` and `makefile` in corresponding miniapp directory.
   - [ ] Add the miniapp binary and any files generated by it to the top-level `.gitignore` file.
   - [ ] Update CMake build system:
      - [ ] Update the `CMakeLists.txt` file in the `miniapps` directory, if the new miniapp is in a new directory.
      - [ ] Add/update the `CMakeLists.txt` file in the new miniapp directory.
      - [ ] Consider adding a new test for the new miniapp.
   - [ ] List the new miniapp in `doc/CodeDocumentation.dox`
   - [ ] Companion pull request for documentation in [mfem/web](https://github.com/mfem/web) repo:
     - [ ] Update or add miniapp-specific documentation, see e.g. the `src/meshing.md` and `src/electromagnetics.md` files.
     - [ ] Add the description, labels and screenshots in `src/examples.md` and `src/img`.
     - [ ] The miniapps go at the end of the page, and are usually listed only under a specific "Application (PDE)" category.
     - [ ] Add a short description of the miniapp in the "Extensive Examples" section of `features.md`.
- [ ] New capability:
   - [ ] All significant new classes, methods and functions have Doxygen-style documentation in source comments.
   - [ ] Consider adding new sample runs in existing examples to highlight the new capability.
   - [ ] Consider saving cool simulation pictures with the new capability in the Confluence gallery (LLNL only) or submitting them, via pull request, to the gallery section of the `mfem/web` repo.
   - [ ] If this is a major new feature, consider mentioning in the short summary inside `README` *(rare)*.
   - [ ] List major new classes in `doc/CodeDocumentation.dox` *(rare)*.
- [ ] Update this checklist, if the new pull request affects it.
- [ ] (LLNL only) Clone the `tests` repository and run the following tests, see `mfem/tests/README.md`:
   - [ ] `compilers`
   - [ ] `memcheck`
   - [ ] `unit-test`
   - [ ] `documentation`
- [ ] (LLNL only) After merging:
   - [ ] Regenerate `README.html` files from companion documentation pull requests.
   - [ ] Update the `baseline` and `compiler` tests, add new tests if necessary.
   - [ ] Consider updating the script `mfem/tests/sample-runs` (`sample-runs-serial` and `sample-runs-parallel`).

### Master/Next Workflow

MFEM uses a `master`/`next`-branch workflow as described below:

- The `master` branch should always be of release quality and changes should not
  be merged until they have been fully tested. This branch is protected, and
  changes can only be made through pull requests.

- After approval, a pull request is merged manually (by MFEM developers) in the
  `next` branch for testing and the `in-next` label is added to the PR.
  This can be done as follows:

  ```
  # Pull the latest version of the "feature-dev" branch
  git checkout feature-dev
  git pull

  # Pull the latest version of the "next" branch
  git checkout next
  git pull

  # Merge "feature-dev" into "next", resolving conflicts, if necessary.
  # Use the "--no-ff" flag to create a new commit with merge message.
  git merge --no-ff feature-dev

  # Push the "next" branch to the server
  git push
  ```

- After a week of testing in `next` (excluding bugfixes), both on GitHub, as
  well as [internally](#tests-at-llnl) at LLNL, the original PR is merged into
  `master` (provided there are no issues).

- After the merge, the feature branch is deleted (unless it is a long-term
  project with periodic PRs).

- The `next` branch is used just for integrated testing of all PRs approved for
  merging into `master` to verify that each works individually and that all of
  them work as a group. This branch can be discarded at any time, though we
  typically do that only at the end of a [release cycle](#releases).


### Releases

- Releases are just tags in the `master` branch, e.g. https://github.com/mfem/mfem/releases/tag/v3.3.2,
  and have a version that ends in an even "patch" number, e.g. `v3.2.2` or
  `v3.4` (by convention `v3.4` is the same as `v3.4.0`.)  Between releases, the
  version ends in an odd "patch" number, e.g. `v3.3.3`.

- We use [milestones](https://github.com/mfem/mfem/milestones) to coordinate the
  work on different PRs toward a release, see for example the
  [v3.3.2 release](https://github.com/mfem/mfem/milestone/1?closed=1).

- After a release is complete, the `next` branch is recreated, e.g. as follows
  (replace `3.3.2` with current release):
  - Rename the current `next` branch to `next-pre-v3.3.2`.
  - Create a new `next` branch starting from the `v3.3.2` release.
  - Local copies of `next` can then be updated with `git checkout -B next origin/next`.

### Release Checklist

- [ ] Update the MFEM version in the following files:
    - [ ] `CHANGELOG`
    - [ ] `makefile`
    - [ ] `CMakeLists.txt`
    - [ ] `doc/CodeDocumentation.conf.in`
- [ ] (LLNL only) Make sure all `README.html` files in the source repo are up to date.
- [ ] Tag the repository:

  ```
  git tag -a v3.1 -m "Official release v3.1"
  git push origin v3.1
  ```
- [ ] Create the release tarball and push to `mfem/releases`.
- [ ] Recreate the `next` branch as described in previous section.
- [ ] Update and push documentation  to `mfem/doxygen`.
- [ ] Update URL shorlinks:
    - [ ] Create a shortlink at [https://goo.gl/](https://goo.gl/) for the release tarball, e.g. http://mfem.github.io/releases/mfem-3.1.tgz.
    - [ ] (LLNL only) Add and commit the new shorlink in the `links` and `links-mfem` files of the internal `mfem/downloads` repo.
    - [ ] Add the new shortlinks to the MFEM packages in `spack`, `homebrew/science`, `VisIt`, etc.
- [ ] Update website in `mfem/web` repo:
    - Update version and shortlinks in `src/index.md` and `src/download.md`.
    -  Use [cloc-1.62.pl](http://cloc.sourceforge.net/) and `ls -lh` to estimate the SLOC and the tarball size in `src/download.md`.


## LLNL Workflow

- The GitHub `master` and `next` branches are mirrored to the LLNL institutional
  Bitbucket repository as `gh-master` and `gh-next`.

- `gh-master` is merged into LLNL's internal `master` through pull requests; write
  permissions to `master` are restricted to ensure this is the only way in which it
  gets updated.

- We never push directly from LLNL to GitHub.

- Versions of the code on LLNL's internal server, from most to least stable:
  - MFEM official release on mfem.org -- Most stable, tested in many apps.
  - `mfem:master` -- Recent development version, guaranteed to work.
  - `mfem:gh-master` -- Stable development version, passed testing, you can use
     it to build your code between releases.
  - `mfem:gh-next` -- Bleeding-edge development version, may be broken, use at
     your own risk.


## Automated Testing

MFEM has several levels of automated testing running on GitHub, as well as on
local Mac and Linux workstations, and Livermore Computing clusters at LLNL.

### Linux and Mac smoke tests
We use Travis CI to drive the default tests on the `master` and `next`
branches. See the `.travis` file and the logs at
[https://travis-ci.org/mfem/mfem](https://travis-ci.org/mfem/mfem).

Testing using Travis CI should be kept lightweight, as there is a 50 minute time
constraint on jobs. Two virtual machines are configured - Mac (OS X) and Linux.

- Tests on the `master` branch are triggered whenever a PR is issued on this branch.
- Tests on the `next` branch are currently scheduled to run each night.

### Windows smoke test
We use Appveyor to test building with the MS Visual C++ compiler in a Windows
environment, as well as to test the CMake build. See the `.appveyor` file and the
build logs at
[https://ci.appveyor.com/project/mfem/mfem](https://ci.appveyor.com/project/mfem/mfem).

CMake is used to generate the MSVC Project files and drive the build.  A release
and debug build is performed with a simple run of `ex1` to verify the executable.

### Tests at LLNL
At LLNL, we mirror the `master` and `next` branches internally (to `gh-master`
and `gh-next`) and run longer nightly tests via cron. On the weekends, a more
extensive test is run which extracts and executes all the different sample runs
from each example.


## Contact Information

- Contact the MFEM team by posting to the [GitHub issue tracker](https://github.com/mfem/mfem).
  Please perform a search to make sure your question has not been answered already.

- Email communications should be sent to the MFEM developers mailing list,
  mfem-dev@llnl.gov.


## [Developer's Certificate of Origin 1.1](https://developercertificate.org/)

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I have the right
    to submit it under the open source license indicated in the file; or

(b) The contribution is based upon previous work that, to the best of my
    knowledge, is covered under an appropriate open source license and I have
    the right under that license to submit that work with modifications, whether
    created in whole or in part by me, under the same open source license
    (unless I am permitted to submit under a different license), as indicated in
    the file; or

(c) The contribution was provided directly to me by some other person who
    certified (a), (b) or (c) and I have not modified it.

(d) I understand and agree that this project and the contribution are public and
    that a record of the contribution (including all personal information I
    submit with it, including my sign-off) is maintained indefinitely and may be
    redistributed consistent with this project or the open source license(s)
    involved.
