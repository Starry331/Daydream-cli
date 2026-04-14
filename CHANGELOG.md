# Changelog

## v0.1.8

### Added

- Installer now supports optional guided model download during setup and can launch the downloaded model immediately after installation.

### Changed

- Improved installer UX with cleaner step execution, better menu handling, and clearer prompts for token setup and model selection.
- Repository setup, dependency install, and model download steps now stream output more predictably during installation.

### Fixed

- Reduced installer shell noise by disabling shell trace/verbose output during setup.
- Improved installer menu cleanup and cursor restoration so interactive screens leave the terminal in a cleaner state.

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
