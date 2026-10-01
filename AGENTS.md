# Agents Manifest

## Purpose

This repository contains research code for running and analyzing bias experiments
in LLMs. Automated agents working here should prioritize reproducibility,
portable dependencies, clear experiment configuration, and narrowly scoped
changes.

## Guidelines

- Prefer standard Python tooling and public package indexes.
- Keep experiment code provider-neutral where possible.
- Avoid adding organization-specific service dependencies to core runtime paths.
- Keep data access isolated behind the database utilities.
- Preserve existing experiment outputs unless a task explicitly changes them.

## Commit Conventions

Use Conventional Commits:

```text
<type>: <description>
```

Common types: `feat`, `fix`, `docs`, `build`, `refactor`, `test`.
