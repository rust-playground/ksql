# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2023-12-29
### Changed
- Updated deps.
- Releasing 1.0.0.

## [0.12.0] - 2023-06-17
### Changed
- Function signature of custom coercions to allow parsing more complex coercions types such as substr.

### Added
- Added new `_substr_[n:n]` COERCE identifier.

## [0.11.0] - 2023-06-10
### Added
- Ability to register new, remove existing and replace existing COERCE types.

### Changed
- Updated ksql binary logic to be more concise, no functional changes.

### Security
- Updated deps.

## [0.10.0] - 2023-05-31
### Added
- _string_ & _number_ COERCE types.

## [0.9.1] - 2023-05-25
### Fixed
- Fixed AND & OR not early exiting evaluation.

## [0.9.0] - 2023-01-05
### Added
- Added new `_uppercase_` & `_title_` COERCE identifiers.
- Added ability to use multiple COERCE identifiers at once separated by a comma.
- Added CLI ability to return original data if using an expression that returns a boolean.

### Changed
- Added Send + Sync restrictions to Expression trait for multithreaded use and async/await. 

## [0.8.0] - 2022-10-30
### Added
- Added new `_lowercase_` COERCE identifier.

## [0.7.0] - 2022-07-29
### Added
- The ability for CONTAINS_ANY and CONTAINS_ALL to check if a String contains any|all of the values
  within an Array. Any non-string data types return a false.

## [0.6.2] - 2022-07-29
### Fixed
- && and || expression chaining.

## [0.6.1] - 2022-07-19
### Fixed
- Fixed number parsing for exponential numbers eg. 1e10.

## [0.6.0] - 2022-07-19
### Added
- Added BETWEEN operator support <value> BETWEEN <value> <value>

### Fixed
- Missing Gt, Gte, Lt, Lte for DateTime data type.

## [0.5.0] - 2022-07-18
### Fixed
- Reworked Parsing algorithm fixing a bunch of scoping issues.
- Added COERCE to DateTime support.
- Added CONTAINS_ANY & CONTAINS_ALL operators.

## [0.4.1] - 2022-07-08
### Fixed
- Fixed Array parsing not handling Comma correctly.

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

[Unreleased]: https://github.com/rust-playground/ksql/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/rust-playground/ksql/compare/v0.12.0...v1.0.0
[0.12.0]: https://github.com/rust-playground/ksql/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/rust-playground/ksql/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/rust-playground/ksql/compare/v0.9.1...v0.10.0
[0.9.1]: https://github.com/rust-playground/ksql/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/rust-playground/ksql/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/rust-playground/ksql/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/rust-playground/ksql/compare/v0.6.2...v0.7.0
[0.6.2]: https://github.com/rust-playground/ksql/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/rust-playground/ksql/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/rust-playground/ksql/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/rust-playground/ksql/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/rust-playground/ksql/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/rust-playground/ksql/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/rust-playground/ksql/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/rust-playground/ksql/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/rust-playground/ksql/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/rust-playground/ksql/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/rust-playground/ksql/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/rust-playground/ksql/commit/03fa237a7e2fe5f15de609e3da3c87f5dbed8805