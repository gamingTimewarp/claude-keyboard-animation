# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Video to QMK RGB Animation Converter - a Python utility that converts video files into C code for QMK-powered RGB keyboard animations.

## Dependencies

```bash
pip install opencv-python numpy
```

## Running the Script

Basic usage:
```bash
python video_to_qmk.py <video> -w <width> -y <height> [-o output.c] [-f fps] [--skip N] [--max-frames N] [--led-map layout.txt]
```

Example with included test video:
```bash
python video_to_qmk.py badapple.mp4 -w 15 -y 5 -f 10 --max-frames 100
```

## Architecture

Single-script architecture (`video_to_qmk.py`) with four main components:

1. **Video Info Extraction** (`get_video_info()`) - Reads video metadata via OpenCV
2. **LED Mapping** (`load_led_map()`) - Parses custom LED layout files with gap support (255 markers)
3. **Frame Processing** (`extract_and_resize_frames()`) - Extracts, resizes, and converts frames from BGR to RGB
4. **QMK Code Generation** (`generate_qmk_code()`) - Outputs C code with PROGMEM frame data and animation state machine

## LED Mapping Format

Custom LED layouts use comma-separated index files where:
- Numbers represent LED indices in the QMK LED array
- Value `255` marks gaps (non-LED positions)
- See `sample_RGB_layout.txt` for reference

## Generated Code Characteristics

- Frame data stored as `uint8_t PROGMEM` (flash memory)
- RGB888 format (24-bit color)
- Includes play/pause/toggle animation controls
- Supports both custom and sequential LED addressing
