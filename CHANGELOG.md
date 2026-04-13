# Changelog

## v0.1.7

### Added

- `run` chat now supports more complete interactive workflow controls, including refined menu handling and model workflow improvements already shipped on `main`.

### Changed

- Improved long-response streaming in the terminal so visible output stays responsive for large replies.
- Improved narrow-terminal live rendering so long wrapped responses keep flowing instead of freezing mid-generation.
- Updated interactive terminal redraw behavior around resize and menu input handling.
- Refined model workflow and registry behavior included in the current `main` branch updates.

### Fixed

- Reduced live overlay stalls caused by large streamed bodies.
- Fixed small-window rendering cases where visible output could blank out or stop updating before the final repaint.
