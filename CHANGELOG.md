# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2026-01-24

### Added
- Exposed finetuning seed parameter for reproducible training
- Resource cleanup for mixed workflows

### Changed
- Improved memory management when switching between inference and training

## [1.3.0] - 2026-01-24

### Changed
- Updated model download management with improved caching and error handling
- Enhanced README documentation

## [1.2.0] - 2026-01-24

### Added
- RNG seed support for reproducible audio generation in inference nodes

## [1.1.0] - 2026-01-23

### Added
- Fine-tuning support with dataset handling (`dataset.py`)
- Simple finetuning example workflow
- MPS (Apple Silicon) device support

### Changed
- Use ComfyUI `model_management` for device detection
- Updated transformers dependency warning
- Updated requirements with safetensors

### Fixed
- Project name in pyproject.toml to match repository naming convention

## [1.0.0] - 2026-01-22

### Added
- Initial release
- Qwen3-TTS model loader node
- Text-to-speech generation node
- Custom voice workflow configuration
- Flash attention support with automatic fallback
