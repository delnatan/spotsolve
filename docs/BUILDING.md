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

For a tagged release in this private repository, authenticate with `gh auth login`
once, then download the wheel for your platform. For example, on Apple Silicon:

```sh
gh release download v0.1.0 --repo delnatan/spotsolve --pattern '*macosx_11_0_arm64.whl' --dir wheels
python -m pip install wheels/spotsolve-0.1.0-cp39-abi3-macosx_11_0_arm64.whl
```

Alternatively, download all five wheels and let pip select the compatible one:

```sh
gh release download v0.1.0 --repo delnatan/spotsolve --pattern '*.whl' --dir wheels
python -m pip install --find-links=./wheels --only-binary=spotsolve spotsolve==0.1.0
```

Pip still resolves dependencies from its configured package index. It does not
reuse GitHub CLI or browser authentication. If the repository becomes public,
you can install a wheel directly from its release asset URL:

```sh
python -m pip install "https://github.com/delnatan/spotsolve/releases/download/v0.1.0/spotsolve-0.1.0-cp39-abi3-macosx_11_0_arm64.whl"
```

Choose the asset matching your platform. A Git URL installs from source and
requires Rust; an Actions artifact is a ZIP that must be downloaded and unpacked.

## Releases

Release tags use `vMAJOR.MINOR.PATCH`, starting at `v0.1.0`. During the `0.x`
series, increment the minor version for new features or incompatible API changes,
and the patch version for compatible fixes. Record changes in `CHANGELOG.md`.
Keep published tags fixed; ship corrections under a new version.

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

On macOS with Xcode 27 (ld-27037), some Rust dylibs link with their symbol
string pool on a 4-byte boundary, and dyld refuses to load them ("mis-aligned
LINKEDIT string pool"). It depends on each binary's size, so it can hit the
extension or a proc-macro crate such as `pyo3_macros` (then reported as
"can't find crate"). Deployment targets and ld flags do not prevent it. Link
with Rust's bundled lld against the 26.5 SDK instead (lld cannot read the 27.0
SDK's library stubs):

```sh
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
export RUSTFLAGS="-C link-arg=-fuse-ld=$(rustc --print sysroot)/lib/rustlib/aarch64-apple-darwin/bin/gcc-ld/ld64.lld"
maturin develop --release
```

The linker does not change generated code (thin LTO runs inside rustc).
Release wheels are built on CI with an older Xcode.

`src/spotsolve_rs/__init__.py` preserves the native import API. The source
archive includes the Rust workspace, lockfile, Python sources and test
fixtures. Neither the wheel nor source archive includes `WORKING.md` or local
build output.

Implementation references: [maturin mixed projects](https://www.maturin.rs/project_layout),
[maturin Action](https://github.com/PyO3/maturin-action), and
[GitHub runner platforms](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
