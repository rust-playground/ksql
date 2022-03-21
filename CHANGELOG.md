# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2022-03-20
### Changed
- Cleaned up code.
- AND->&& and OR->|| and =->==in lexer.
- Expression signature to accept &[u8] instead of &str

### Added
- Not token type and support in lexer + parser.

### Fixed
- Identifier parsing when it ends in a parenthesis or bracket.
- Blank string parsing

## [0.1.2] - 2022-03-14
### Fixed
- Fixed README + leftover docs.

## [0.1.1] - 2022-03-14
### Fixed
- Fixed Cargo.toml for publishing.

## [0.1.0] - 2022-03-14
### Added
- Initial release.

[Unreleased]: https://github.com/rust-playground/ksql/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/rust-playground/ksql/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/rust-playground/ksql/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/rust-playground/ksql/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/rust-playground/ksql/commit/03fa237a7e2fe5f15de609e3da3c87f5dbed8805