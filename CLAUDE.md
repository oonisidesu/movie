# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a video face-swapping tool that replaces faces in videos with different faces. The tool is designed for personal/educational use only and operates entirely locally without cloud dependencies.

## Architecture

The project uses:
- Python 3.8+ as the primary language
- Deep Learning frameworks (PyTorch/TensorFlow) for face swapping
- OpenCV for face detection and video processing
- FFmpeg for video encoding/decoding
- GUI framework (Tkinter/PyQt/Streamlit) for user interface

## Key Technical Requirements

### Core Functionality
- Support MP4, AVI, MOV video formats
- Automatic face detection in videos
- Face swapping with quality adjustment options
- Multiple person support with individual face selection
- Batch processing capability

### Performance Constraints
- Process 1-minute video in under 5 minutes (standard quality)
- Memory usage under 8GB
- GPU recommended (CUDA-enabled NVIDIA GPUs)
- Works on CPU but slower

### Development Phases
1. **MVP**: Single person face swap, basic GUI, MP4 support
2. **Extension**: Multiple people, quality adjustment, more formats
3. **Optimization**: Performance improvements, UI/UX, batch processing

## Security & Ethics

- Include usage warnings for personal/educational use only
- No network communication - all processing local
- Implement user agreement/consent before first use
- Option to auto-delete processing history

## Development Commands

Since this is a new project without existing code, common commands will be established as the codebase develops. Expected commands will include:
- Installation of dependencies (likely via pip)
- Running the main application
- Running tests
- Building standalone executables