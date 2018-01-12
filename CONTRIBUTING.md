# How to Contribute

The MFEM team welcomes contributions at all levels: from bugfixes, to code
improvements and simplification; to new mesh, discretization or solver
capabilities, improved documentation, new examples, miniapps, and more.

Use a pull request (PR) toward the `mfem:master` branch to propose your
contribution.  You can also open an [issue](https://github.com/mfem/mfem/issues)
to discuss the design and/or any significant changes.  We also welcome your
[simulation images](http://mfem.org/gallery/), for which you can submit a pull
request in the `web` repo.

Please refer back to this document as a checklist before issuing any pull
request and in particular consult the following sections:

- [GitHub Workflow](#github-workflow)
  - [MFEM Organization](#mfem-organization)
  - [New Feature Development](#new-feature-development)
  - [Developer Guidelines](#developer-guidelines)
  - [Pull Requests](#pull-requests)
  - [Pull Request Checklist](#pull-request-checklist)
  - [Master/Next Workflow](#masternext-workflow)
  - [Releases](#releases)
- [LLNL Workflow](#llnl-workflow)
- [Automated Testing](#automated-testing)
- [Contact Information](#contact-information)

Contributing to MFEM requires knowledge of Git and likely finite elements. If
you are new to Git, see the [GitHub learning
resources](https://help.github.com/articles/git-and-github-learning-resources/).
To learn more about the finite element method, see our [FEM page](http://mfem.org/fem).

By submitting a pull request, you are affirming the [Developer's Certificate of
Origin](#developers-certificate-of-origin-11) at the end of the file.


## GitHub Workflow

The GitHub organization, https://github.com/mfem, is the main developer hub for
the MFEM project.

If you plan to make contributions or will like to stay up-to-date with changes
in the code, we strongly encourage you to [join the MFEM organization](#mfem-organization).

This will simplify the workflow (by providing you additional permissions), and
will allow us to reach you directly with project announcements.


### MFEM Organization

- Please [contact us](#contact-information) for an invitation to join the MFEM
  GitHub organization.

- You should receive an invitation email, which you can directly accept.
  Alternatively, *after logging into GitHub*, you can accept the invitation at
  the top of https://github.com/mfem.

- Please consider making your membership public by going to
  https://github.com/orgs/mfem/people and clicking on the organization
  visibility dropbox next to your name.

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

- The new feature should be important enough that at least one person, the
  proposer, is willing to work on it and be its champion.

- The proposer creates a branch for the new feature (with suffix `-dev`) and
  implements it according to the discussed design.

- **We prefer that you create the new feature branch inside the MFEM organization
  as opposed to in a fork.** This allows everyone in the community to collaborate
  in one central place.

- If you prefer to work in your fork, please [enable upstream
  edits](https://help.github.com/articles/allowing-changes-to-a-pull-request-branch-created-from-a-fork/).

- The typical feature branch name is `new-feature-dev`, e.g. `pumi-dev`. While
  not frequent in MFEM, other suffixes are possible, e.g. `-fix`, `-doc`, etc.


### Developer Guidelines

- *Keep the code lean and as simple as possible*
  - Well-designed simple code is frequently more general and powerful.
  - Lean code base is easier to understand by new collaborators.
  - New features should be added only if they are necessary or generally useful.
  - Introduction of language constructions not currently used in MFEM should be
    justified and generally avoided (so we can build on cutting-edge systems).

- *Keep the code general and reasonably efficient*
  - Main goal is fast prototyping for research.
  - When in doubt generality wins over efficiency.
  - Respect the needs of different users (current and/or future).

- *Keep things separate and logically organized*
  - General usage features go in MFEM (implemented in as much generality as
    possible), non-general features go into the external apps.
  - Inside MFEM, compartmentalize between linalg, fem, mesh, GLVis, etc.
  - Contributions that are project-specific or have external dependencies are
    allowed (if they are of broader interest), but should be `#ifdef`-ed and not
    change the code by default.

- Code styling
  - Consistent code styling is enforced with `make style` in the top-level
    directory. This requires [Artistic Style](http://astyle.sourceforge.net) (we
    specifically use version 2.05.1). See also the file `config/mfem.astylerc`.


### Pull Requests

- When your branch is ready for other developer to start looking / commenting on
  the code, create a pull request towards `mfem:master`.

- Pull request titles typically have the form:

  `Description [new-feature-dev]`

  for example:

  `Parallel Unstructured Mesh Infrastructure (PUMI) integration [pumi-dev]`

  Note the branch name suffix (in square brackets).

- The title can contain a prefix in square brackets to emphasize the type of PR.
  Some common choices are: `[DON'T MERGE]`, `[WIP]` and `[DISCUSS]`, for example:

  `[DISCUSS] Hybridized DG [hdg-dev]`

- Add a description, appropriate labels and assign yourself to the PR. The MFEM
  team will add reviewers as appropriate.

- List outstanding TODO items in the description, see PR #222 for an example.

- Track the Travis CI and Appveyor [continuous integration](#automated-testing)
  builds at the end of the PR. These should run clean, so try to address any
  errors as soon as possible.


## Pull Request Checklist

Before a PR can be merged, it should satisfy the following:

- [ ] Code builds
- [ ] Code passes `make style`
- [ ] Update the `CHANGELOG` file:
    - [ ] Is this a new feature that the users need to be aware of? New or updated example or miniapp?
    - [ ] Does it make sense to create a new section in the CHANGELOG to group this with other related features?
- [ ] Update the `INSTALL` file :
    - [ ] Has a new optional library been added?
       - [ ] Make sure the external library is licensed under LGPL, not GPL!
    - [ ] Does `make` or `cmake` have a new target?
    - [ ] Did the requirements or the installation process change? (rare)
- [ ] Update README:
    - [ ] Does GLVis have new keys/functionality that needs to be documented?
    - [ ] Is this a really major feature that needs mentioning in the short summary here? (rare)
- [ ] New examples:
    - [ ] All sample runs at the top of the example work
    - [ ] Update examples/makefile
      - [ ] Add the example code to the appropriate SEQ_EXAMPLES and PAR_EXAMPLES variables.
      - [ ] Add any files generated by it to the clean target.
    - [ ] Update examples/CMakeLists.txt
      - [ ] Add the example code to the ALL_EXE_SRCS variable.
      - [ ] Make sure THIS_TEST_OPTIONS is set correctly for the new example.
- [ ] List the new example in doc/CodeDocumentation.dox
- [ ] Companion pull request for documentation in mfem/web repo
   - [ ] Update or add example-specific documentation, see e.g. the src/examples.md.
   - [ ] Add the description, labels and screenshots in src/examples.md and and src/img.
   - [ ] In examples.md, list the example under the appropriate categories, add new categories if necessary.
- [ ] New miniapps:
   - [ ] All sample runs at the top of the miniapp work
   - [ ] Update top-level makefile and makefile in corresponding miniapp directory
   - [ ] Update CMake build system
      - [ ] Update the CMakeLists.txt file in miniapps/ directory, if the new miniapp is in a new directory.
      - [ ] Add/update the CMakeLists.txt file in the new miniapp directory.
      - [ ] Consider adding a new test for the new miniapp.
   - [ ] List the new miniapp in doc/CodeDocumentation.dox
   - [ ] Companion pull request for documentation in mfem/web repo
     - [ ] Update or add miniapp-specific documentation, see e.g. the src/meshing.md and src/electromagnetics.md.
     - [ ] Add the description, labels and screenshots in src/examples.md and and src/img in the mfem/web repo.
     - [ ] The miniapps go at the end of the page, and are usually listed only under a specific "Application (PDE)" category.
- [ ] New capability:
   - [ ] All significant new classes, methods and functions have Doxygen-style documentation in source comments
   - [ ] Consider adding new sample runs in existing examples to highlight the new capability
   - [ ] Consider saving cool simulation pictures with the new capability in the Confluence gallery
   - [ ] List major new classes in doc/CodeDocumentation.dox (rare)
- [ ] Update this checklist, if the new pull request affects it


### Master/Next Workflow

MFEM uses a `master`/`next`-branch workflow as described below.

- The `master` branch should always be of release quality and changes should not
  be merged until they have been fully tested. This branch is protected and
  changes can be made only through pull requests.

- After approval, a pull request is merged manually (by MFEM developers) in the
  `next` branch for testing and the `in-next` label is added to the PR.

- After a week of testing in `next` (excluding bugfixes), both on GitHub, as
  well as [internally](#tests-at-llnl) at LLNL, the original PR is merged into
  `master` (provided there are no issues).

- After the merge, the feature branch is deleted (unless it is a long-term
  project with periodic PRs).

- The `next` branch is used just for integrated testing of all the PRs that have
  been approved for merging into `master` to make sure that each of them works
  individually and all of them work as a group. This branch can be discarded at
  any time, though we typically do that only at the end of a [release cycle](#releases).


### Releases

- Releases are just tags in the `master` branch, e.g. https://github.com/mfem/mfem/releases/tag/v3.3.2,
  and have a version that ends in an even "patch" number.  (By convention `v3.4`
  is the same as `v3.4.0`.)  Between releases, the version ends in an odd
  "patch" number, e.g. `v3.3.3`.

- We use [milestones](https://github.com/mfem/mfem/milestones) to coordinate the
  work on different PRs toward a release, see for example the
  [v3.3.2 release](https://github.com/mfem/mfem/milestone/1?closed=1).

- After the release is complete, the `next` branch is recreated e.g. as follows
  (replace `3.3.2` with current release):
  - Rename the current `next` branch to e.g. `next-pre-v3.3.2`.
  - Create a new `next` branch starting from the `v3.3.2` release.
  - Local copies of `next` can then be updated with `git checkout -B next
  origin/next`.


## LLNL Workflow

- The GitHub `master` and `next` branches are mirrored to the LLNL institutional
  Bitbucket repository as `gh-master` and `gh-next`.

- `gh-master` is merged into LLNL's internal `master` through pull requests, write
  permissions are restricted.

- We never push directly from LLNL to GitHub.

- Versions of the code on LLNL internal server, from most to least stable:
  - MFEM official release on mfem.org -- Most stable, tested in many apps.
  - `mfem:master` -- Recent development version, guaranteed to work.
  - `mfem:gh-master` -- Stable development version, passed testing, you can use
     it to build your code between releases.
  - `mfem:gh-next` -- Bleeding-edge development version, may be broken, use at
     your own risk.


## Automated Testing

MFEM has several levels of automated testing running on GitHub as well as on
local Mac and Linux workstations, and Livermore Computing clusters at LLNL.

### OSX and Linux smoke tests
We use Travis CI to drive the default tests on the `master` and `next`
branches. See the `.travis` file and the logs at
[https://travis-ci.org/mfem/mfem](https://travis-ci.org/mfem/mfem).

Any testing using Travis CI should be kept lightweight, as there is a 50 minute
time constraint on jobs. Two virtual machines are configured - OSX and Linux.

- Tests on the `master` branch are triggered whenever a PR is issued on this branch.
- Tests on the `next` branch are currently scheduled to run each night.

### Windows smoke test
We use Appveyor to test building with the MS Visual C++ compiler in a Windows
environment as well as to test the CMake build. See the `.appveyor` file and the
logs at
[https://ci.appveyor.com/project/tzanio/mfem](https://ci.appveyor.com/project/tzanio/mfem).

CMake is used to generate the MSVC Project files and drive the build.  A release
and debug build is performed, with a simple run of `ex1` to verify the
executable.

### Tests at LLNL
At LLNL we mirror the `master` and `next` branches internally (to `gh-master`
and `gh-next`) and run longer nightly tests via cron. On the weekends, a more
extensive test is run which extracts and executes all the different sample runs
from each example.


## Contact Information

- The best way to contact the MFEM team is by posting to the [GitHub issue
  tracker](https://github.com/mfem/mfem).

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
