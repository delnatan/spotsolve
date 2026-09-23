# Building and distributing spotsolve

The root `pyproject.toml` builds one `spotsolve` distribution containing the
Python API and the `spotsolve_rs` extension. Users installing a matching wheel
need Python and the runtime dependencies, but no Rust compiler or separate
`spotsolve-rs` installation.

## GitHub Actions

[Build and test wheels](../.github/workflows/wheels.yml) runs on pull requests,
pushes to `main` and `codex/**` branches, manual dispatch, and published releases.

| Platform | Architecture | Wheel target |
|---|---|---|
| Linux (glibc) | x86-64 | manylinux2014 |
| Linux (glibc) | ARM64 | manylinux2014 |
| macOS | Intel | x86_64 |
| macOS | Apple Silicon | arm64 |
| Windows | x86-64 | win_amd64 |

Each platform runs the Rust tests, builds a release wheel, and runs the Python
suite against that installed wheel on CPython 3.10 and 3.14. The extension uses
CPython's stable ABI, so one wheel serves supported standard CPython versions;
the package metadata requires Python 3.10 or newer even though its ABI tag is
`cp39-abi3`. This does not include free-threaded Python or PyPy builds.

The source-distribution job installs from the generated archive and runs the
Python tests. It checks that release tags match both package versions.
Actions are pinned to commit revisions, with monthly Dependabot updates.
Build jobs have read-only repository permissions; only the release upload job
can write release assets.

Successful runs retain `wheels-*` and `sdist` artifacts for 14 days. Download
and unzip the artifact for your platform, then install its `.whl` file:

```sh
python -m pip install path/to/downloaded/filename.whl
```

The wheel filename must retain its original version, ABI and platform tags.
Dependencies are resolved by pip. Artifact and release downloads require
repository access while the repository is private.

## Releases

1. Update the version in `pyproject.toml`, `rust/Cargo.toml` and
   `src/spotsolve/__init__.py`, and refresh `rust/Cargo.lock` and `uv.lock`.
2. Publish a GitHub release tagged `v<version>` from the tested commit.
3. The workflow rebuilds and tests all five wheels plus the source archive.
   Only after every job passes does it attach the packages to that release.

A rerun replaces assets with the same filenames. Creating a tag without
publishing a release does not attach packages. Builds do not publish to PyPI;
that would require configuring a PyPI project and trusted publishing separately.

## Local builds

From an activated environment at the repository root, with Rust installed:

```sh
python -m pip install -e '.[dev,test]'
python -m pytest -q
cargo test --release --locked --workspace --manifest-path rust/Cargo.toml
maturin build --release --locked --out dist
maturin sdist --out dist
```

An editable install builds the bundled extension. After changing Rust code,
run `maturin develop --release` again. The old two-package install has been
retired; when migrating an existing environment, uninstall both `spotsolve`
and `spotsolve-rs` before installing the unified distribution, because the old
native distribution owns the same import path.

`src/spotsolve_rs/__init__.py` preserves the native import API. The source
archive includes the Rust workspace, lockfile, Python sources and test
fixtures. Neither the wheel nor source archive includes `WORKING.md` or local
build output.

Implementation references: [maturin mixed projects](https://www.maturin.rs/project_layout),
[maturin Action](https://github.com/PyO3/maturin-action), and
[GitHub runner platforms](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
