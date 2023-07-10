# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2023-07-10

### Fixed

- Fixed a bug in clean_ground(). It caused that, during denoising step before dtm computation, the removed points (noise) were independent of the voxel resolution used.

## [0.2.0] - 2023-06-02

### Added

- This CHANGELOG file.
- Documentation is now live at `ReadTheDocs.io`.
- A `process_hook` in order to replace the current embedded progress logging.
- DTM interpolation

### Changed

- `3DFIN` organization was renamed `3DFin`. All links have been changed accordingly.

### Fixed

- Links to `PyPI` and documentation longer point to placeholders on `README.md`.

## [0.1.0] - 2023-03-27

First public version of dendromatics