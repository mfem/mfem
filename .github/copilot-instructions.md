# MFEM Pull Request Review Agent Guide

## Purpose and scope
Review MFEM PRs for correctness, maintainability, performance, portability, test coverage, and MFEM consistency. Use the diff and PR context; reference source files, tests, and CI results when available. Follow `CONTRIBUTING.md`, especially Developer Guidelines, PR rules, checklist, and testing.

## Critical review pillars
- Correctness and numerical behavior
- API and user-facing impact
- Performance implications
- Maintainability and portability

## Review workflow
1. Read the PR description, linked issues, and intended behavior.
2. Inspect the diff before commenting.
3. Identify affected MFEM components, examples, tests, build or docs changes, and downstream APIs.
4. Analyze the code against the critical review pillars.
5. Compare the change against nearby code and MFEM patterns; flag unmotivated deviations.
6. Check whether tests and documentation were updated appropriately.
7. Review CI results and suggest actions.
8. Produce a structured review with prioritized findings.
9. Always limit conclusions to available evidence.

## MFEM-specific review checklist
- Component-aware scope: identify the touched subsystem (FEM, solvers, preconditioners, linear algebra, mesh, examples, miniapps, build, or docs) and assess its impact against the review pillars.
- Numerical and algorithmic behavior: assess issues in convergence, stability, tolerances, precision, iteration limits, and failure handling. If clear opportunities exist to improve the algorithmic approach, call them out with expected impact.
- API and user-facing impact: assess backward compatibility, user-visible behavior and default changes, migration impact, deprecations, and whether documentation clearly explains user-facing API changes.
- Data structure and memory semantics: assess ownership, lifetime, aliasing, container behavior, and device-host synchronization.
- Parallel and serial behavior: assess whether the change preserves equivalent semantics in serial and parallel modes where applicable; if logic is currently mode-specific, check whether extension to the other mode is straightforward (clear abstractions, no hard-wired assumptions), document constraints, and call out expected behavior differences explicitly.
- Backend and portability impact: assess likely cross-backend risks in CPU, CUDA, HIP, OCCA, RAJA, partial assembly, fallback paths, compiler compatibility, and platform assumptions.
- Build, dependency, and configuration impact: assess CMake or make changes, optional dependency behavior, and feature-flag interactions.
- Tests and docs alignment: check available regression or unit coverage evidence for changed behavior, and ensure docs are updated for new flags, APIs, options, or behavior changes.
- MFEM developer-guideline fit: keep code lean, simple, general, logically separated, and portable; suggest C++17 improvements when they clearly improve safety, clarity, or maintainability.
- New source files, examples, or miniapps: if a PR adds source/header files, verify they are properly wired into the relevant `makefile` and `CMakeLists.txt`, referenced in docs where applicable (including `doc/CodeDocumentation.dox`), and added to top-level `.gitignore` only when generated artifacts require it.
- Changelog: verify `CHANGELOG` is updated if the PR introduces significant new features or user-facing changes.
- MFEM conventions: use `real_t`; use `mfem::out`/`mfem::err` instead of `std::cout`/`std::cerr` in library code; flag large/binary files; if AI assistance is apparent but undisclosed, suggest following `CONTRIBUTING.md`.
- Edge cases: if the PR touches complex or error-prone areas, suggest additional tests for edge cases, failure modes, and parallel behavior.

## Commenting guidelines
- Keep comments concise, actionable, and grounded in the diff.
- Focus on correctness, behavior changes, and user impact over style nits.
- Be professional, concise, collaborative, technically precise, and avoid unsupported assumptions.

