# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2022-06-26
### Changed
- Display of Value not 50% faster for Objects and Arrays.

## [0.3.1] - 2022-06-08
### Fixed
- Missing commas between items is output JSON Objects.

### Changed
- Release profile for smaller binary size.
- Updated deps to latest.
- Linter suggested updates.

## [0.3.0] - 2022-04-06
### Changed
- Lexer token return signature reducing its size from 12->8 bytes

## [0.2.0] - 2022-04-06
### Changed
- Cleaned up code.
- AND->&& and OR->|| and =->==in lexer.
- Expression signature to accept &[u8] instead of &str
- Lexer token return signature reducing its size from 32->12 bytes
- Lexer now implements Iterator instead of returning a Vec of Tokens.
- Refactored Parser internals to use lexer iterator.

### Added
- Not token type and support in lexer + parser.
- More documentation.

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

[Unreleased]: https://github.com/rust-playground/ksql/compare/v0.4.0...HEAD
[0.3.1]: https://github.com/rust-playground/ksql/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/rust-playground/ksql/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/rust-playground/ksql/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/rust-playground/ksql/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/rust-playground/ksql/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/rust-playground/ksql/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/rust-playground/ksql/commit/03fa237a7e2fe5f15de609e3da3c87f5dbed8805