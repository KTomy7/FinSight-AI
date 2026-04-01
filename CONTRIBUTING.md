# Contributing to FinSight-AI

Thank you for contributing! Please read the short guidelines below before opening a PR.

---

## Getting Started

1. Clone the repo and install dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

2. Run the tests to verify your environment:

   ```bash
   pytest
   ```

3. Open a feature branch from `main` and keep PRs focused on a single concern.

---

## Code Style and Architecture

- Follow the layer rules described in [`docs/architecture/overview.md`](docs/architecture/overview.md).
- Keep ML logic out of Streamlit views; views should only orchestrate input/output.
- Place new use-case DTOs in `src/finsight/application/dto.py` (see
  [`docs/conventions/naming-and-contracts.md`](docs/conventions/naming-and-contracts.md)).
- Implement new model or data-source adapters in `infrastructure/`, not in use-case code.

---

## When Documentation Updates Are Required

Update documentation in the **same PR** as the code change when:

| Change | Required doc update |
|---|---|
| New DTO added or renamed | Update the catalogue table in `docs/conventions/naming-and-contracts.md` |
| New port protocol added | Add to the ports table in `docs/conventions/naming-and-contracts.md` and note in `docs/architecture/overview.md` if the layer map changes |
| New infrastructure adapter added | Add an entry to the *Extension Points* section of `docs/architecture/overview.md` |
| Migration or breaking contract change | Create or update a migration note in `docs/architecture/` (e.g., following the pattern in `dto-migration.md`) |
| New use case or significant behaviour change | Update the relevant data flow diagram in `docs/architecture/overview.md` |

Documentation updates are **not required** for:
- Bug fixes that don't change public contracts.
- Test additions or refactors.
- Config value changes that don't alter how settings are accessed.

---

## Testing

- Write unit tests for all new use-case and domain logic under `tests/unit/`.
- Use port protocol fakes/mocks rather than real infrastructure in unit tests.
- Integration tests live in `tests/integration/` and may use real filesystem paths via
  `tmp_path` fixtures.
- Keep test coverage for `application/` and `domain/` layers high; UI adapter tests
  are valuable but lower priority.

---

## Pull Request Checklist

Before marking a PR ready for review, confirm:

- [ ] Tests pass (`pytest`).
- [ ] New code follows the layer rules (no Streamlit in use cases, no ML in views).
- [ ] New or changed DTOs follow the frozen-dataclass pattern with `to_dict` / `from_dict`.
- [ ] Documentation updated if required (see table above).
- [ ] PR description explains *what* changed and *why*.
