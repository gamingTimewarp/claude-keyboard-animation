#!/usr/bin/env python3
"""
Video to QMK RGB Animation Converter
Converts a video file into a QMK-compatible RGB matrix animation.

Supports multiple output modes:
  rgb          - Full color, uncompressed (default)
  rgb-rle      - Full color with RLE compression
  mono-bitpack - Monochrome, 1 bit per pixel (24x compression)
  mono-rle     - Monochrome with RLE (50-100x compression)
"""

import sys
import cv2
import numpy as np
import argparse
from pathlib import Path


def get_video_info(video_path):
    """Extract basic video information."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    return fps, frame_count, duration


def load_led_map(led_map_file, width, height):
    """
    Load LED mapping from file.

    Format: One row per line, comma-separated LED indices, use 255 for gaps
    Example:
    0,1,2,3,4,5,6,7,8,9,10,11,12,255,13,14,15,16,17,18,19
    20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40
    """
    if not led_map_file:
        return None

    with open(led_map_file, 'r') as f:
        lines = f.readlines()

    led_map = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        row = [int(x.strip()) for x in line.split(',')]
        led_map.append(row)

    if len(led_map) != height:
        raise ValueError(f"LED map has {len(led_map)} rows, expected {height}")

    for i, row in enumerate(led_map):
        if len(row) != width:
            raise ValueError(f"LED map row {i} has {len(row)} columns, expected {width}")

    return led_map


def extract_and_resize_frames(video_path, width, height, max_frames=None, frame_skip=1,
                              skip_pattern=None, interp_method='area', bw_first=False, threshold=128):
    """
    Extract frames from video and resize to keyboard dimensions.

    Args:
        video_path: Path to video file
        width: Keyboard width in LEDs
        height: Keyboard height in LEDs
        max_frames: Maximum number of frames to extract (None for all)
        frame_skip: Skip every N frames (1 = use all frames, 2 = use every other frame)
        skip_pattern: Alternating skip pattern string (e.g., "2,3")
        interp_method: Resize interpolation method
        bw_first: Convert to B&W before resizing (best for pure B&W source)
        threshold: B&W threshold for bw_first mode

    Returns:
        List of resized frames as numpy arrays
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0
    extracted = 0

    # Map interpolation method
    interp_map = {
        'nearest': cv2.INTER_NEAREST,
        'area': cv2.INTER_AREA,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC
    }
    interp = interp_map.get(interp_method, cv2.INTER_AREA)

    # Parse skip pattern if provided
    pattern = None
    pattern_idx = 0
    frames_until_next = 0
    if skip_pattern:
        try:
            pattern = [int(x.strip()) for x in skip_pattern.split(',')]
            if not pattern or any(x < 1 for x in pattern):
                raise ValueError()
            print(f"Using skip pattern: {pattern}")
            frames_until_next = pattern[0]
        except:
            print(f"Warning: Invalid skip pattern '{skip_pattern}', using uniform skip")
            pattern = None

    print(f"Extracting and resizing frames to {width}x{height}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Determine if we should extract this frame
        should_extract = False

        if pattern:
            if frames_until_next == 0:
                should_extract = True
                pattern_idx = (pattern_idx + 1) % len(pattern)
                frames_until_next = pattern[pattern_idx]
            else:
                frames_until_next -= 1
        elif frame_skip > 1:
            if frame_idx % frame_skip == 0:
                should_extract = True
        else:
            should_extract = True

        if should_extract:
            if bw_first:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
                resized_bw = cv2.resize(bw, (width, height), interpolation=interp)
                rgb_frame = cv2.cvtColor(resized_bw, cv2.COLOR_GRAY2RGB)
            else:
                resized = cv2.resize(frame, (width, height), interpolation=interp)
                rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            frames.append(rgb_frame)
            extracted += 1

            if max_frames and extracted >= max_frames:
                break

            if extracted % 10 == 0:
                print(f"  Extracted {extracted} frames...", end='\r')

        frame_idx += 1

    cap.release()
    print(f"\nExtracted {len(frames)} frames total")
    return frames


# ---------------------------------------------------------------------------
# Color RLE compression (for rgb-rle mode)
# ---------------------------------------------------------------------------

def rle_compress_frame(frame):
    """Run-Length Encode a color frame."""
    pixels = frame.reshape(-1, 3)

    compressed = []
    if len(pixels) == 0:
        return compressed

    current_color = tuple(pixels[0])
    count = 1

    for pixel in pixels[1:]:
        pixel_tuple = tuple(pixel)
        if pixel_tuple == current_color and count < 255:
            count += 1
        else:
            compressed.append((count, *current_color))
            current_color = pixel_tuple
            count = 1

    compressed.append((count, *current_color))
    return compressed


# ---------------------------------------------------------------------------
# Monochrome helpers (for mono-bitpack and mono-rle modes)
# ---------------------------------------------------------------------------

def convert_to_monochrome(frame, threshold=128, colors=None, dither=False,
                          auto_threshold=False, adaptive_threshold=False):
    """
    Convert frame to pure black and white (1-bit per pixel).

    Returns array of booleans: True = white, False = black
    """
    if colors and colors >= 2 and colors <= 16:
        pixels = frame.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, palette = cv2.kmeans(pixels, colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        quantized = palette[labels.flatten()].reshape(frame.shape).astype(np.uint8)
        gray = cv2.cvtColor(quantized, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    if adaptive_threshold:
        mean_brightness = np.mean(gray)
        if mean_brightness > 160:
            threshold = 180
        elif mean_brightness > 120:
            threshold = 150
        elif mean_brightness > 80:
            threshold = 128
        elif mean_brightness > 40:
            threshold = 100
        else:
            threshold = 80
    elif auto_threshold:
        threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = int(threshold)

    if dither:
        height, width = gray.shape
        gray_float = gray.astype(np.float32)

        for y in range(height):
            for x in range(width):
                old_pixel = gray_float[y, x]
                new_pixel = 255 if old_pixel > threshold else 0
                gray_float[y, x] = new_pixel
                error = old_pixel - new_pixel

                if x + 1 < width:
                    gray_float[y, x + 1] += error * 7/16
                if y + 1 < height:
                    if x > 0:
                        gray_float[y + 1, x - 1] += error * 3/16
                    gray_float[y + 1, x] += error * 5/16
                    if x + 1 < width:
                        gray_float[y + 1, x + 1] += error * 1/16

        return gray_float > threshold
    else:
        return gray > threshold


def pack_bits(mono_frame):
    """Pack monochrome frame into bytes (8 pixels per byte)."""
    height, width = mono_frame.shape
    total_pixels = height * width
    flat = mono_frame.flatten()

    packed = []
    for i in range(0, total_pixels, 8):
        byte = 0
        for bit in range(8):
            if i + bit < total_pixels:
                if flat[i + bit]:
                    byte |= (1 << bit)
        packed.append(byte)

    return packed


def rle_compress_monochrome(mono_frame):
    """
    Run-Length Encode a monochrome frame.
    Returns list of (count, is_white) tuples.
    """
    flat = mono_frame.flatten()

    if len(flat) == 0:
        return []

    compressed = []
    current_value = flat[0]
    count = 1

    for pixel in flat[1:]:
        if pixel == current_value and count < 255:
            count += 1
        else:
            compressed.append((count, current_value))
            current_value = pixel
            count = 1

    compressed.append((count, current_value))
    return compressed


# ---------------------------------------------------------------------------
# Compression statistics
# ---------------------------------------------------------------------------

def calculate_compression_stats(frames, width, height, mode, threshold=128,
                                colors=None, dither=False, auto_threshold=False,
                                adaptive_threshold=False):
    """Calculate compression statistics for a given mode."""
    uncompressed_size = len(frames) * width * height * 3  # RGB888

    if mode == 'rgb':
        compressed_size = uncompressed_size
    elif mode == 'rgb-rle':
        compressed_size = 0
        for frame in frames:
            rle_data = rle_compress_frame(frame)
            compressed_size += len(rle_data) * 4
    elif mode == 'mono-bitpack':
        compressed_size = len(frames) * ((width * height + 7) // 8)
    elif mode == 'mono-rle':
        compressed_size = 0
        for frame in frames:
            mono = convert_to_monochrome(frame, threshold, colors, dither, auto_threshold, adaptive_threshold)
            rle_data = rle_compress_monochrome(mono)
            compressed_size += len(rle_data) * 2
    else:
        compressed_size = uncompressed_size

    return {
        'uncompressed': uncompressed_size,
        'compressed': compressed_size,
        'ratio': uncompressed_size / compressed_size if compressed_size > 0 else 0,
        'savings_percent': ((uncompressed_size - compressed_size) / uncompressed_size * 100) if uncompressed_size > 0 else 0
    }


# ---------------------------------------------------------------------------
# C code data writers
# ---------------------------------------------------------------------------

def write_uncompressed_data(f, frames, width, height):
    """Write uncompressed RGB frame data."""
    f.write("// Frame data stored as RGB888 (uncompressed)\n")
    f.write("static const uint8_t PROGMEM video_frames[VIDEO_FRAME_COUNT][VIDEO_HEIGHT][VIDEO_WIDTH][3] = {\n")

    for frame_idx, frame in enumerate(frames):
        f.write(f"    // Frame {frame_idx}\n")
        f.write("    {\n")

        for y in range(height):
            f.write("        {")
            for x in range(width):
                r, g, b = frame[y, x]
                f.write(f"{{{r},{g},{b}}}")
                if x < width - 1:
                    f.write(",")
            f.write("}")
            if y < height - 1:
                f.write(",")
            f.write("\n")

        f.write("    }")
        if frame_idx < len(frames) - 1:
            f.write(",")
        f.write("\n")

    f.write("};\n\n")


def write_color_rle_data(f, frames, width, height):
    """Write RLE compressed color frame data."""
    compressed_frames = []
    frame_offsets = [0]
    total_entries = 0

    print("Compressing frames with RLE...")
    for i, frame in enumerate(frames):
        rle_data = rle_compress_frame(frame)
        compressed_frames.append(rle_data)
        total_entries += len(rle_data)
        frame_offsets.append(total_entries)
        if (i + 1) % 10 == 0:
            print(f"  Compressed {i + 1}/{len(frames)} frames...", end='\r')
    print()

    # Frame offsets
    f.write("// Frame offsets in RLE data (uint32_t for large animations)\n")
    f.write(f"static const uint32_t PROGMEM frame_offsets[VIDEO_FRAME_COUNT + 1] = {{\n    ")
    for i, offset in enumerate(frame_offsets):
        f.write(f"{offset}")
        if i < len(frame_offsets) - 1:
            f.write(",")
        if (i + 1) % 10 == 0 and i < len(frame_offsets) - 1:
            f.write("\n    ")
        else:
            f.write(" ")
    f.write("\n};\n\n")

    # RLE data
    f.write("// RLE compressed frame data: [run_length, r, g, b, ...]\n")
    f.write(f"// Total entries: {total_entries}\n")
    f.write(f"static const uint8_t PROGMEM rle_data[{total_entries * 4}] = {{\n")

    entry_count = 0
    for frame_idx, rle_frame in enumerate(compressed_frames):
        f.write(f"    // Frame {frame_idx} ({len(rle_frame)} runs)\n    ")
        for run_idx, (count, r, g, b) in enumerate(rle_frame):
            f.write(f"{count},{r},{g},{b}")
            if frame_idx < len(compressed_frames) - 1 or run_idx < len(rle_frame) - 1:
                f.write(",")
            entry_count += 1
            if entry_count % 8 == 0:
                f.write("\n    ")
    f.write("\n};\n\n")


def write_bitpacked_data(f, frames, width, height, threshold, colors=None,
                         dither=False, auto_threshold=False, adaptive_threshold=False):
    """Write bit-packed monochrome data."""
    print("Converting to monochrome and bit-packing...")

    bytes_per_frame = (width * height + 7) // 8

    f.write(f"// Bit-packed monochrome frames ({bytes_per_frame} bytes per frame)\n")
    f.write(f"// Each bit: 0 = black, 1 = white\n")
    f.write(f"static const uint8_t PROGMEM mono_frames[VIDEO_FRAME_COUNT][{bytes_per_frame}] = {{\n")

    for frame_idx, frame in enumerate(frames):
        mono = convert_to_monochrome(frame, threshold, colors, dither, auto_threshold, adaptive_threshold)
        packed = pack_bits(mono)

        f.write(f"    // Frame {frame_idx}\n    {{")
        for i, byte in enumerate(packed):
            f.write(f"{byte}")
            if i < len(packed) - 1:
                f.write(",")
            if (i + 1) % 16 == 0 and i < len(packed) - 1:
                f.write("\n     ")
        f.write("}")
        if frame_idx < len(frames) - 1:
            f.write(",")
        f.write("\n")

        if (frame_idx + 1) % 10 == 0:
            print(f"  Packed {frame_idx + 1}/{len(frames)} frames...", end='\r')

    f.write("};\n\n")
    print()


def write_rle_monochrome_data(f, frames, width, height, threshold, colors=None,
                              dither=False, auto_threshold=False, adaptive_threshold=False):
    """Write RLE monochrome data."""
    print("Converting to monochrome and RLE compressing...")

    compressed_frames = []
    frame_offsets = [0]
    total_runs = 0

    for i, frame in enumerate(frames):
        mono = convert_to_monochrome(frame, threshold, colors, dither, auto_threshold, adaptive_threshold)
        rle_data = rle_compress_monochrome(mono)
        compressed_frames.append(rle_data)
        total_runs += len(rle_data)
        frame_offsets.append(total_runs)

        if (i + 1) % 10 == 0:
            print(f"  Compressed {i + 1}/{len(frames)} frames...", end='\r')
    print()

    # Frame offsets
    f.write("// Frame offsets in RLE data (uint32_t for large animations)\n")
    f.write(f"static const uint32_t PROGMEM frame_offsets[VIDEO_FRAME_COUNT + 1] = {{\n    ")
    for i, offset in enumerate(frame_offsets):
        f.write(f"{offset}")
        if i < len(frame_offsets) - 1:
            f.write(",")
        if (i + 1) % 10 == 0 and i < len(frame_offsets) - 1:
            f.write("\n    ")
        else:
            f.write(" ")
    f.write("\n};\n\n")

    # RLE data
    f.write("// RLE monochrome data: [count, is_white, ...]\n")
    f.write(f"// Total runs: {total_runs}\n")
    f.write(f"static const uint8_t PROGMEM rle_data[{total_runs * 2}] = {{\n")

    entry_count = 0
    for frame_idx, rle_frame in enumerate(compressed_frames):
        f.write(f"    // Frame {frame_idx} ({len(rle_frame)} runs)\n    ")
        for run_idx, (count, is_white) in enumerate(rle_frame):
            f.write(f"{count},{1 if is_white else 0}")
            if frame_idx < len(compressed_frames) - 1 or run_idx < len(rle_frame) - 1:
                f.write(",")
            entry_count += 1
            if entry_count % 10 == 0:
                f.write("\n    ")
        f.write("\n")
    f.write("};\n\n")


# ---------------------------------------------------------------------------
# C code function writers
# ---------------------------------------------------------------------------

def write_uncompressed_functions(f, led_map):
    """Write uncompressed update function."""
    if led_map:
        f.write("""// Update function - call this in rgb_matrix_indicators_advanced_user()
bool video_animation_update(uint8_t led_min, uint8_t led_max) {
    if (!animation_playing) {
        return false;
    }

    // Display current frame EVERY time to prevent blinking
    for (uint8_t y = 0; y < VIDEO_HEIGHT; y++) {
        for (uint8_t x = 0; x < VIDEO_WIDTH; x++) {
            uint8_t led_index = pgm_read_byte(&led_map[y][x]);
            if (led_index == 255) continue;
            if (led_index < led_min || led_index >= led_max) continue;

            uint8_t r = pgm_read_byte(&video_frames[current_frame][y][x][0]);
            uint8_t g = pgm_read_byte(&video_frames[current_frame][y][x][1]);
            uint8_t b = pgm_read_byte(&video_frames[current_frame][y][x][2]);

            rgb_matrix_set_color(led_index, r, g, b);
        }
    }

    uint32_t now = timer_read32();
    if (now - last_frame_time >= FRAME_DELAY_MS) {
        last_frame_time = now;
        current_frame++;
        if (current_frame >= VIDEO_FRAME_COUNT) {
            current_frame = 0;
        }
    }

    return true;
}

""")
    else:
        f.write("""// Update function - call this in rgb_matrix_indicators_advanced_user()
bool video_animation_update(uint8_t led_min, uint8_t led_max) {
    if (!animation_playing) {
        return false;
    }

    // Display current frame EVERY time to prevent blinking
    for (uint8_t y = 0; y < VIDEO_HEIGHT; y++) {
        for (uint8_t x = 0; x < VIDEO_WIDTH; x++) {
            uint8_t led_index = y * VIDEO_WIDTH + x;
            if (led_index < led_min || led_index >= led_max) continue;

            uint8_t r = pgm_read_byte(&video_frames[current_frame][y][x][0]);
            uint8_t g = pgm_read_byte(&video_frames[current_frame][y][x][1]);
            uint8_t b = pgm_read_byte(&video_frames[current_frame][y][x][2]);

            rgb_matrix_set_color(led_index, r, g, b);
        }
    }

    uint32_t now = timer_read32();
    if (now - last_frame_time >= FRAME_DELAY_MS) {
        last_frame_time = now;
        current_frame++;
        if (current_frame >= VIDEO_FRAME_COUNT) {
            current_frame = 0;
        }
    }

    return true;
}

""")


def write_color_rle_functions(f, led_map):
    """Write color RLE decompression functions."""
    f.write("""// Decode and display RLE compressed frame
static void decode_and_display_frame(uint16_t frame_idx, uint8_t led_min, uint8_t led_max) {
    uint32_t start_offset = pgm_read_dword(&frame_offsets[frame_idx]);
    uint32_t end_offset = pgm_read_dword(&frame_offsets[frame_idx + 1]);

    uint16_t pixel_idx = 0;

    for (uint32_t offset = start_offset; offset < end_offset; offset++) {
        // Read RLE entry: count, r, g, b
        uint8_t count = pgm_read_byte(&rle_data[offset * 4]);
        uint8_t r = pgm_read_byte(&rle_data[offset * 4 + 1]);
        uint8_t g = pgm_read_byte(&rle_data[offset * 4 + 2]);
        uint8_t b = pgm_read_byte(&rle_data[offset * 4 + 3]);

        // Draw 'count' pixels with this color
        for (uint8_t i = 0; i < count; i++) {
            uint8_t y = pixel_idx / VIDEO_WIDTH;
            uint8_t x = pixel_idx % VIDEO_WIDTH;
            pixel_idx++;

""")

    if led_map:
        f.write("""            uint8_t led_index = pgm_read_byte(&led_map[y][x]);
            if (led_index == 255) continue;
            if (led_index < led_min || led_index >= led_max) continue;

            rgb_matrix_set_color(led_index, r, g, b);
""")
    else:
        f.write("""            uint8_t led_index = y * VIDEO_WIDTH + x;
            if (led_index < led_min || led_index >= led_max) continue;

            rgb_matrix_set_color(led_index, r, g, b);
""")

    f.write("""        }
    }
}

// Update function - call this in rgb_matrix_indicators_advanced_user()
bool video_animation_update(uint8_t led_min, uint8_t led_max) {
    if (!animation_playing) {
        return false;
    }

    // Decode and display current frame EVERY time to prevent blinking
    decode_and_display_frame(current_frame, led_min, led_max);

    // Check if it's time to advance to next frame
    uint32_t now = timer_read32();
    if (now - last_frame_time >= FRAME_DELAY_MS) {
        last_frame_time = now;
        current_frame++;
        if (current_frame >= VIDEO_FRAME_COUNT) {
            current_frame = 0;
        }
    }

    return true;
}

""")


def write_bitpack_functions(f, led_map):
    """Write bit-unpacking and display functions."""
    f.write("""// Unpack and display bit-packed frame
static void display_bitpacked_frame(uint16_t frame_idx, uint8_t led_min, uint8_t led_max) {
    uint16_t pixel_idx = 0;
    uint16_t byte_idx = 0;
    uint8_t current_byte = pgm_read_byte(&mono_frames[frame_idx][0]);

    for (uint8_t y = 0; y < VIDEO_HEIGHT; y++) {
        for (uint8_t x = 0; x < VIDEO_WIDTH; x++) {
            // Get bit position in current byte
            uint8_t bit_pos = pixel_idx % 8;

            // Load next byte if needed
            if (bit_pos == 0 && pixel_idx > 0) {
                byte_idx++;
                current_byte = pgm_read_byte(&mono_frames[frame_idx][byte_idx]);
            }

            // Check if pixel is white (bit = 1)
            bool is_white = (current_byte >> bit_pos) & 1;

""")

    if led_map:
        f.write("""            uint8_t led_index = pgm_read_byte(&led_map[y][x]);
            if (led_index != 255 && led_index >= led_min && led_index < led_max) {
                if (is_white) {
                    rgb_matrix_set_color(led_index, WHITE_R, WHITE_G, WHITE_B);
                } else {
                    rgb_matrix_set_color(led_index, 0, 0, 0);
                }
            }

            pixel_idx++;
""")
    else:
        f.write("""            uint8_t led_index = y * VIDEO_WIDTH + x;
            if (led_index >= led_min && led_index < led_max) {
                if (is_white) {
                    rgb_matrix_set_color(led_index, WHITE_R, WHITE_G, WHITE_B);
                } else {
                    rgb_matrix_set_color(led_index, 0, 0, 0);
                }
            }

            pixel_idx++;
""")

    f.write("""        }
    }
}

// Update function
bool video_animation_update(uint8_t led_min, uint8_t led_max) {
    if (!animation_playing) {
        return false;
    }

    display_bitpacked_frame(current_frame, led_min, led_max);

    uint32_t now = timer_read32();
    if (now - last_frame_time >= FRAME_DELAY_MS) {
        last_frame_time = now;
        current_frame++;
        if (current_frame >= VIDEO_FRAME_COUNT) {
            current_frame = 0;
        }
    }

    return true;
}

""")


def write_rle_monochrome_functions(f, led_map):
    """Write RLE monochrome decompression functions."""
    f.write("""// Decode and display RLE monochrome frame
static void decode_and_display_mono_rle(uint16_t frame_idx, uint8_t led_min, uint8_t led_max) {
    uint32_t start_offset = pgm_read_dword(&frame_offsets[frame_idx]);
    uint32_t end_offset = pgm_read_dword(&frame_offsets[frame_idx + 1]);

    uint32_t pixel_idx = 0;

    for (uint32_t offset = start_offset; offset < end_offset; offset++) {
        uint8_t count = pgm_read_byte(&rle_data[offset * 2]);
        bool is_white = pgm_read_byte(&rle_data[offset * 2 + 1]);

        for (uint8_t i = 0; i < count; i++) {
            uint8_t y = pixel_idx / VIDEO_WIDTH;
            uint8_t x = pixel_idx % VIDEO_WIDTH;

""")

    if led_map:
        f.write("""            uint8_t led_index = pgm_read_byte(&led_map[y][x]);
            if (led_index != 255 && led_index >= led_min && led_index < led_max) {
                if (is_white) {
                    rgb_matrix_set_color(led_index, WHITE_R, WHITE_G, WHITE_B);
                } else {
                    rgb_matrix_set_color(led_index, 0, 0, 0);
                }
            }

            pixel_idx++;
""")
    else:
        f.write("""            uint8_t led_index = y * VIDEO_WIDTH + x;
            if (led_index >= led_min && led_index < led_max) {
                if (is_white) {
                    rgb_matrix_set_color(led_index, WHITE_R, WHITE_G, WHITE_B);
                } else {
                    rgb_matrix_set_color(led_index, 0, 0, 0);
                }
            }

            pixel_idx++;
""")

    f.write("""        }
    }
}

// Update function
bool video_animation_update(uint8_t led_min, uint8_t led_max) {
    if (!animation_playing) {
        return false;
    }

    decode_and_display_mono_rle(current_frame, led_min, led_max);

    uint32_t now = timer_read32();
    if (now - last_frame_time >= FRAME_DELAY_MS) {
        last_frame_time = now;
        current_frame++;
        if (current_frame >= VIDEO_FRAME_COUNT) {
            current_frame = 0;
        }
    }

    return true;
}

""")


# ---------------------------------------------------------------------------
# Main code generation
# ---------------------------------------------------------------------------

def generate_qmk_code(frames, output_path, fps, keyboard_width, keyboard_height,
                      led_map=None, mode='rgb', threshold=128, white_color=(255,255,255),
                      colors=None, dither=False, auto_threshold=False, adaptive_threshold=False):
    """Generate QMK C code for the animation."""
    is_mono = mode.startswith('mono')

    with open(output_path, 'w') as f:
        stats = calculate_compression_stats(frames, keyboard_width, keyboard_height, mode,
                                            threshold, colors, dither, auto_threshold, adaptive_threshold)

        # Header
        if is_mono:
            f.write("// Generated MONOCHROME video animation for QMK RGB Matrix\n")
        else:
            f.write("// Generated video animation for QMK RGB Matrix\n")
        f.write(f"// Total frames: {len(frames)}\n")
        f.write(f"// Target FPS: {fps}\n")
        f.write(f"// Dimensions: {keyboard_width}x{keyboard_height}\n")
        f.write(f"// Mode: {mode}\n")
        f.write(f"// Uncompressed size: {stats['uncompressed']} bytes\n")
        f.write(f"// Compressed size: {stats['compressed']} bytes\n")
        f.write(f"// Compression ratio: {stats['ratio']:.1f}x\n")
        f.write(f"// Space saved: {stats['savings_percent']:.1f}%\n\n")

        f.write("#include QMK_KEYBOARD_H\n")
        f.write("#include \"rgb_matrix.h\"\n\n")

        # Constants
        f.write(f"#define VIDEO_FRAME_COUNT {len(frames)}\n")
        f.write(f"#define VIDEO_WIDTH {keyboard_width}\n")
        f.write(f"#define VIDEO_HEIGHT {keyboard_height}\n")
        f.write(f"#define VIDEO_FPS {fps}\n")
        f.write(f"#define FRAME_DELAY_MS {int(1000/fps)}\n")

        if is_mono:
            f.write(f"#define TOTAL_PIXELS {keyboard_width * keyboard_height}\n")
            f.write(f"#define WHITE_R {white_color[0]}\n")
            f.write(f"#define WHITE_G {white_color[1]}\n")
            f.write(f"#define WHITE_B {white_color[2]}\n")

        f.write("\n")

        # LED mapping
        if led_map:
            f.write("// LED mapping array - matches keyboard matrix\n")
            f.write("static const uint8_t PROGMEM led_map[VIDEO_HEIGHT][VIDEO_WIDTH] = {\n")
            for y, row in enumerate(led_map):
                f.write("    {")
                for x, led_idx in enumerate(row):
                    f.write(f"{led_idx:3d}")
                    if x < len(row) - 1:
                        f.write(",")
                f.write("}")
                if y < len(led_map) - 1:
                    f.write(",")
                f.write("\n")
            f.write("};\n\n")

        # Frame data
        if mode == 'rgb':
            write_uncompressed_data(f, frames, keyboard_width, keyboard_height)
        elif mode == 'rgb-rle':
            write_color_rle_data(f, frames, keyboard_width, keyboard_height)
        elif mode == 'mono-bitpack':
            write_bitpacked_data(f, frames, keyboard_width, keyboard_height, threshold,
                                 colors, dither, auto_threshold, adaptive_threshold)
        elif mode == 'mono-rle':
            write_rle_monochrome_data(f, frames, keyboard_width, keyboard_height, threshold,
                                      colors, dither, auto_threshold, adaptive_threshold)

        # Animation state
        f.write("""// Animation state
static uint16_t current_frame = 0;
static uint32_t last_frame_time = 0;
static bool animation_playing = false;

""")

        # Mode-specific update functions
        if mode == 'rgb':
            write_uncompressed_functions(f, led_map)
        elif mode == 'rgb-rle':
            write_color_rle_functions(f, led_map)
        elif mode == 'mono-bitpack':
            write_bitpack_functions(f, led_map)
        elif mode == 'mono-rle':
            write_rle_monochrome_functions(f, led_map)

        # Common control functions
        f.write("""// Start the animation
void video_animation_start(void) {
    current_frame = 0;
    last_frame_time = timer_read32();
    animation_playing = true;

    rgb_matrix_mode_noeeprom(RGB_MATRIX_SOLID_COLOR);
    rgb_matrix_sethsv_noeeprom(0, 0, 0);
}

// Stop the animation
void video_animation_stop(void) {
    animation_playing = false;
    rgb_matrix_mode_noeeprom(RGB_MATRIX_CYCLE_ALL);
}

// Toggle animation on/off
void video_animation_toggle(void) {
    if (animation_playing) {
        video_animation_stop();
    } else {
        video_animation_start();
    }
}
""")

        # Usage examples
        f.write("""
// Example usage in keymap.c:
//
// #include "video_animation.h"
//
// enum custom_keycodes {
//     VIDEO_TOGGLE = SAFE_RANGE,
// };
//
// bool process_record_user(uint16_t keycode, keyrecord_t *record) {
//     switch (keycode) {
//         case VIDEO_TOGGLE:
//             if (record->event.pressed) {
//                 video_animation_toggle();
//             }
//             return false;
//     }
//     return true;
// }
//
// void keyboard_post_init_user(void) {
//     rgb_matrix_mode_noeeprom(RGB_MATRIX_SOLID_COLOR);
//     rgb_matrix_sethsv_noeeprom(0, 0, 0);
//     rgb_matrix_set_speed_noeeprom(255);
// }
//
// bool rgb_matrix_indicators_advanced_user(uint8_t led_min, uint8_t led_max) {
//     if (video_animation_update(led_min, led_max)) {
//         return true;  // Skip other effects when video is playing
//     }
//     return false;
// }
""")

    print(f"\nGenerated QMK code: {output_path}")
    if mode != 'rgb':
        print(f"Mode: {mode}, Compression: {stats['ratio']:.1f}x")


# ====================== Keymap Setup ======================

HEADER_TEMPLATE = """\
#pragma once

#include QMK_KEYBOARD_H

void video_animation_start(void);
void video_animation_stop(void);
void video_animation_toggle(void);
bool video_animation_update(uint8_t led_min, uint8_t led_max);
"""


def find_animation_file(keymap_dir):
    """Auto-detect the animation .c file by scanning for video_animation_update."""
    candidates = []
    for f in keymap_dir.iterdir():
        if f.suffix == '.c' and f.name != 'keymap.c':
            try:
                text = f.read_text(errors='ignore')
                if 'video_animation_update' in text:
                    candidates.append(f)
            except OSError:
                continue

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = ', '.join(c.name for c in candidates)
        print(f"Multiple animation files found: {names}")
        return None

    default = keymap_dir / 'video_animation.c'
    if default.exists():
        return default

    return None


def generate_header(keymap_dir, stem):
    """Write the .h header file."""
    header_path = keymap_dir / f'{stem}.h'
    header_path.write_text(HEADER_TEMPLATE)
    return header_path


def update_rules_mk(keymap_dir, c_filename):
    """Create or update rules.mk with the SRC line."""
    rules_path = keymap_dir / 'rules.mk'
    src_line = f'SRC += {c_filename}'

    if rules_path.exists():
        content = rules_path.read_text()
        if src_line in content:
            return rules_path, False
        if content and not content.endswith('\n'):
            content += '\n'
        content += src_line + '\n'
        rules_path.write_text(content)
    else:
        rules_path.write_text(src_line + '\n')

    return rules_path, True


def setup_keymap(keymap_dir, c_file_path):
    """Generate .h and rules.mk for the given animation .c file in keymap_dir.

    If c_file_path is not inside keymap_dir, it is copied there first.
    """
    import shutil

    keymap_dir = Path(keymap_dir).resolve()
    c_file_path = Path(c_file_path).resolve()

    if not keymap_dir.is_dir():
        print(f"Error: {keymap_dir} is not a directory")
        return

    # Copy .c file into keymap dir if it isn't already there
    dest = keymap_dir / c_file_path.name
    if dest.resolve() != c_file_path:
        shutil.copy2(c_file_path, dest)
        print(f"Copied {c_file_path.name} -> {keymap_dir}")

    stem = c_file_path.stem

    header_path = generate_header(keymap_dir, stem)
    print(f"Created {header_path}")

    rules_path, rules_changed = update_rules_mk(keymap_dir, c_file_path.name)
    if rules_changed:
        print(f"Updated {rules_path}")
    else:
        print(f"{rules_path} already includes {c_file_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert video to QMK RGB matrix animation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output Modes:
  rgb          - Full color, uncompressed (default)
  rgb-rle      - Full color with RLE compression
  mono-bitpack - Monochrome, 1 bit per pixel (24x compression)
  mono-rle     - Monochrome with RLE (50-100x compression, best for Bad Apple!)

Examples:
  # Basic RGB conversion
  python video_to_qmk.py input.mp4 -w 15 -y 5 -f 10

  # RGB with RLE compression
  python video_to_qmk.py input.mp4 -w 21 -y 6 -f 10 --mode rgb-rle --max-frames 200

  # Monochrome RLE (extreme compression!)
  python video_to_qmk.py badapple.mp4 -w 21 -y 6 --mode mono-rle --max-frames 1000

  # Monochrome bit-packed with custom white color (cyan)
  python video_to_qmk.py video.mp4 -w 21 -y 6 --mode mono-bitpack --white 0,255,255

  # Use custom LED mapping
  python video_to_qmk.py input.mp4 -w 21 -y 6 --led-map keychron_v6_map.txt

  # Use every 3rd frame and limit to 100 frames
  python video_to_qmk.py input.mp4 -w 15 -y 5 --skip 3 --max-frames 100
        """
    )

    parser.add_argument('video', type=str, help='Input video file')
    parser.add_argument('-w', '--width', type=int, required=True, help='Keyboard width in LEDs')
    parser.add_argument('-y', '--height', type=int, required=True, help='Keyboard height in LEDs')
    parser.add_argument('-o', '--output', type=str, default='video_animation.c',
                        help='Output C file (default: video_animation.c)')
    parser.add_argument('-f', '--fps', type=int, default=10,
                        help='Target frames per second (default: 10)')
    parser.add_argument('--skip', type=int, default=1,
                        help='Frame skip - use every Nth frame (default: 1, no skip)')
    parser.add_argument('--skip-pattern', type=str, default=None,
                        help='Alternating skip pattern (e.g., "2,3" = skip 2, then 3, repeat)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to extract (default: all)')
    parser.add_argument('--led-map', type=str, default=None,
                        help='Path to file containing LED mapping (one row per line, comma-separated, use 255 for gaps)')
    parser.add_argument('--mode', choices=['rgb', 'rgb-rle', 'mono-bitpack', 'mono-rle'],
                        default='rgb',
                        help='Output mode (default: rgb)')
    parser.add_argument('--interp', choices=['nearest', 'area', 'linear', 'cubic'], default='area',
                        help='Resize interpolation method (default: area)')

    # Monochrome-specific options
    mono_group = parser.add_argument_group('monochrome options', 'Options for mono-bitpack and mono-rle modes')
    mono_group.add_argument('--threshold', type=int, default=128,
                            help='B&W threshold 0-255 (default: 128)')
    mono_group.add_argument('--auto-threshold', action='store_true',
                            help='Automatically find optimal threshold using Otsu method')
    mono_group.add_argument('--adaptive-threshold', action='store_true',
                            help='Dynamic threshold that adapts to frame brightness')
    mono_group.add_argument('--white', type=str, default='255,255,255',
                            help='White color as R,G,B (default: 255,255,255)')
    mono_group.add_argument('--colors', type=int, default=None,
                            help='Reduce to N colors before B&W conversion (2-16)')
    mono_group.add_argument('--dither', action='store_true',
                            help='Apply Floyd-Steinberg dithering for smoother B&W conversion')
    mono_group.add_argument('--bw-first', action='store_true',
                            help='Convert to B&W before resizing (best for pure B&W source videos)')

    # Keymap integration
    parser.add_argument('--setup-keymap', type=str, default=None, metavar='DIR',
                        help='Also generate .h and rules.mk in the given keymap directory')

    args = parser.parse_args()
    run_conversion(args)

    if args.setup_keymap and Path(args.output).exists():
        print()
        setup_keymap(args.setup_keymap, args.output)


def run_conversion(args, interactive=True):
    """
    Run the video-to-QMK conversion.

    Args:
        args: Namespace with conversion parameters (from argparse or GUI).
        interactive: If True, prompt the user on large uncompressed output.
    """
    # Validate input
    if not Path(args.video).exists():
        print(f"Error: Video file '{args.video}' not found")
        return

    if args.width <= 0 or args.height <= 0:
        print("Error: Width and height must be positive")
        return

    # Parse white color
    white_color = (255, 255, 255)
    if args.mode.startswith('mono'):
        try:
            white_color = tuple(int(x) for x in args.white.split(','))
            if len(white_color) != 3 or any(c < 0 or c > 255 for c in white_color):
                raise ValueError()
        except:
            print("Error: --white must be R,G,B (e.g., 255,255,255)")
            return

    # Load LED mapping if provided
    led_map = None
    if args.led_map:
        try:
            led_map = load_led_map(args.led_map, args.width, args.height)
            print(f"Loaded LED mapping from {args.led_map}")
        except Exception as e:
            print(f"Error loading LED map: {e}")
            return

    # Get video info
    print(f"Processing video: {args.video}")
    original_fps, frame_count, duration = get_video_info(args.video)
    print(f"Original video: {original_fps:.2f} FPS, {frame_count} frames, {duration:.2f}s")

    # Calculate effective frame count
    effective_frames = frame_count // args.skip
    if args.max_frames:
        effective_frames = min(effective_frames, args.max_frames)

    print(f"\nTarget settings:")
    print(f"  Keyboard dimensions: {args.width}x{args.height} LEDs")
    print(f"  Target FPS: {args.fps}")
    print(f"  Frame skip: every {args.skip} frame(s)")
    print(f"  Estimated frames to extract: ~{effective_frames}")
    print(f"  Mode: {args.mode}")
    if led_map:
        print(f"  LED mapping: Custom mapping loaded")
    else:
        print(f"  LED mapping: Sequential (0, 1, 2, 3...)")

    # Warn about large file sizes
    estimated_size_kb = (effective_frames * args.width * args.height * 3) / 1024
    print(f"  Estimated uncompressed size: ~{estimated_size_kb:.1f} KB")

    if args.mode != 'rgb':
        if args.mode == 'mono-bitpack':
            estimated_compressed = estimated_size_kb / 24
        elif args.mode == 'mono-rle':
            estimated_compressed = estimated_size_kb / 50
        else:
            estimated_compressed = estimated_size_kb / 2.5
        print(f"  Estimated compressed size: ~{estimated_compressed:.1f} KB (rough estimate)")

    if estimated_size_kb > 1000 and args.mode == 'rgb' and interactive:
        print("\nWARNING: Output file will be very large!")
        print("    Consider using --mode rgb-rle or --mode mono-rle to reduce size")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled")
            return

    # Extract frames
    frames = extract_and_resize_frames(
        args.video,
        args.width,
        args.height,
        max_frames=args.max_frames,
        frame_skip=args.skip,
        skip_pattern=args.skip_pattern,
        interp_method=args.interp,
        bw_first=args.bw_first,
        threshold=args.threshold
    )

    # Show compression stats
    if args.mode != 'rgb' and frames:
        stats = calculate_compression_stats(frames, args.width, args.height, args.mode,
                                            args.threshold, args.colors, args.dither,
                                            args.auto_threshold, args.adaptive_threshold)
        print(f"\nCompression statistics:")
        print(f"  Uncompressed: {stats['uncompressed']/1024:.1f} KB")
        print(f"  Compressed: {stats['compressed']/1024:.1f} KB")
        print(f"  Compression ratio: {stats['ratio']:.2f}x")
        print(f"  Space saved: {stats['savings_percent']:.1f}%")

    # Generate QMK code
    generate_qmk_code(frames, args.output, args.fps, args.width, args.height,
                      led_map, args.mode, args.threshold, white_color,
                      args.colors, args.dither, args.auto_threshold, args.adaptive_threshold)

    print("\nConversion complete!")
    print(f"\nNext steps:")
    print(f"1. Copy {args.output} to your QMK keyboard's keymap directory")
    print(f"2. Create video_animation.h with function declarations")
    print(f"3. Add 'SRC += video_animation.c' to your keymap's rules.mk")
    print(f"4. Include the header and integrate with your keymap.c (see examples in generated file)")
    print(f"5. Compile and flash!")


def launch_gui():
    """Launch the tkinter GUI for video-to-QMK conversion."""
    import tkinter as tk
    from tkinter import ttk, filedialog, scrolledtext
    import threading
    import queue
    import io

    root = tk.Tk()
    root.title("Video to QMK RGB Animation Converter")
    root.resizable(True, True)

    # --- stdout redirect via queue ---
    log_queue = queue.Queue()

    class QueueWriter(io.TextIOBase):
        """Redirect writes to a thread-safe queue."""
        def write(self, text):
            if text:
                log_queue.put(text)
            return len(text) if text else 0

        def flush(self):
            pass

    def poll_log():
        """Drain the queue into the log widget."""
        while True:
            try:
                msg = log_queue.get_nowait()
            except queue.Empty:
                break
            log_text.configure(state='normal')
            log_text.insert(tk.END, msg)
            log_text.see(tk.END)
            log_text.configure(state='disabled')
        root.after(100, poll_log)

    # --- Variables ---
    video_var = tk.StringVar()
    output_var = tk.StringVar(value='video_animation.c')
    width_var = tk.IntVar(value=15)
    height_var = tk.IntVar(value=5)
    fps_var = tk.IntVar(value=10)
    mode_var = tk.StringVar(value='rgb')
    skip_var = tk.IntVar(value=1)
    skip_pattern_var = tk.StringVar()
    max_frames_var = tk.StringVar()
    interp_var = tk.StringVar(value='area')
    led_map_var = tk.StringVar()
    threshold_var = tk.IntVar(value=128)
    adaptive_var = tk.BooleanVar()
    auto_thresh_var = tk.BooleanVar()
    dither_var = tk.BooleanVar()
    bw_first_var = tk.BooleanVar()
    white_var = tk.StringVar(value='255,255,255')
    colors_var = tk.StringVar()
    setup_keymap_var = tk.BooleanVar()
    keymap_dir_var = tk.StringVar()

    # --- Helpers ---
    mono_widgets = []
    keymap_widgets = []

    def browse_video():
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.webm"), ("All files", "*.*")]
        )
        if path:
            video_var.set(path)

    def browse_output():
        path = filedialog.asksaveasfilename(
            title="Save Output As",
            defaultextension=".c",
            filetypes=[("C source files", "*.c"), ("All files", "*.*")]
        )
        if path:
            output_var.set(path)

    def browse_led_map():
        path = filedialog.askopenfilename(
            title="Select LED Map File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if path:
            led_map_var.set(path)

    def browse_keymap_dir():
        path = filedialog.askdirectory(title="Select QMK Keymap Directory")
        if path:
            keymap_dir_var.set(path)

    def on_mode_change(*_args):
        is_mono = mode_var.get().startswith('mono')
        state = 'normal' if is_mono else 'disabled'
        for w in mono_widgets:
            try:
                w.configure(state=state)
            except tk.TclError:
                pass

    def on_setup_keymap_change(*_args):
        state = 'normal' if setup_keymap_var.get() else 'disabled'
        for w in keymap_widgets:
            try:
                w.configure(state=state)
            except tk.TclError:
                pass

    def do_convert():
        """Gather form values and run conversion in a background thread."""
        video = video_var.get().strip()
        if not video:
            log_queue.put("Error: No video file selected.\n")
            return

        w = width_var.get()
        h = height_var.get()
        if w <= 0 or h <= 0:
            log_queue.put("Error: Width and height must be positive.\n")
            return

        # Build an argparse-like namespace
        args = argparse.Namespace(
            video=video,
            output=output_var.get().strip() or 'video_animation.c',
            width=w,
            height=h,
            fps=fps_var.get(),
            mode=mode_var.get(),
            skip=skip_var.get(),
            skip_pattern=skip_pattern_var.get().strip() or None,
            max_frames=None,
            interp=interp_var.get(),
            led_map=led_map_var.get().strip() or None,
            threshold=threshold_var.get(),
            adaptive_threshold=adaptive_var.get(),
            auto_threshold=auto_thresh_var.get(),
            dither=dither_var.get(),
            bw_first=bw_first_var.get(),
            white=white_var.get().strip(),
            colors=None,
        )

        # Parse optional integers
        mf = max_frames_var.get().strip()
        if mf:
            try:
                args.max_frames = int(mf)
            except ValueError:
                log_queue.put("Error: Max Frames must be a number.\n")
                return

        cv = colors_var.get().strip()
        if cv:
            try:
                args.colors = int(cv)
            except ValueError:
                log_queue.put("Error: Colors must be a number.\n")
                return

        # Capture keymap setup options from main thread
        do_setup = setup_keymap_var.get()
        keymap_dir = keymap_dir_var.get().strip()

        if do_setup and not keymap_dir:
            log_queue.put("Error: Keymap directory not set.\n")
            return

        convert_btn.configure(state='disabled')

        # Clear log
        log_text.configure(state='normal')
        log_text.delete('1.0', tk.END)
        log_text.configure(state='disabled')

        def worker():
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = QueueWriter()
            sys.stderr = QueueWriter()
            try:
                run_conversion(args, interactive=False)
                if do_setup and Path(args.output).exists():
                    print()
                    setup_keymap(keymap_dir, args.output)
            except Exception as e:
                print(f"\nError during conversion: {e}")
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                # Re-enable button from main thread
                root.after(0, lambda: convert_btn.configure(state='normal'))

        threading.Thread(target=worker, daemon=True).start()

    # ====================== Build UI ======================
    frame = ttk.Frame(root, padding=10)
    frame.grid(sticky='nsew')
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    frame.columnconfigure(1, weight=1)

    row = 0

    # --- Video Input ---
    ttk.Label(frame, text="Video File:").grid(row=row, column=0, sticky='w', pady=2)
    entry_video = ttk.Entry(frame, textvariable=video_var)
    entry_video.grid(row=row, column=1, sticky='ew', padx=(5, 0), pady=2)
    ttk.Button(frame, text="Browse...", command=browse_video).grid(row=row, column=2, padx=(5, 0), pady=2)
    row += 1

    # --- Output File ---
    ttk.Label(frame, text="Output File:").grid(row=row, column=0, sticky='w', pady=2)
    ttk.Entry(frame, textvariable=output_var).grid(row=row, column=1, sticky='ew', padx=(5, 0), pady=2)
    ttk.Button(frame, text="Browse...", command=browse_output).grid(row=row, column=2, padx=(5, 0), pady=2)
    row += 1

    # --- Dimensions ---
    dim_frame = ttk.Frame(frame)
    dim_frame.grid(row=row, column=0, columnspan=3, sticky='w', pady=2)
    ttk.Label(dim_frame, text="Width:").pack(side='left')
    ttk.Spinbox(dim_frame, textvariable=width_var, from_=1, to=200, width=5).pack(side='left', padx=(5, 15))
    ttk.Label(dim_frame, text="Height:").pack(side='left')
    ttk.Spinbox(dim_frame, textvariable=height_var, from_=1, to=200, width=5).pack(side='left', padx=(5, 0))
    row += 1

    # --- FPS ---
    fps_frame = ttk.Frame(frame)
    fps_frame.grid(row=row, column=0, columnspan=3, sticky='w', pady=2)
    ttk.Label(fps_frame, text="FPS:").pack(side='left')
    ttk.Spinbox(fps_frame, textvariable=fps_var, from_=1, to=60, width=5).pack(side='left', padx=(5, 0))
    row += 1

    # --- Mode ---
    mode_frame = ttk.Frame(frame)
    mode_frame.grid(row=row, column=0, columnspan=3, sticky='w', pady=2)
    ttk.Label(mode_frame, text="Mode:").pack(side='left')
    mode_combo = ttk.Combobox(mode_frame, textvariable=mode_var,
                              values=['rgb', 'rgb-rle', 'mono-bitpack', 'mono-rle'],
                              state='readonly', width=15)
    mode_combo.pack(side='left', padx=(5, 0))
    mode_var.trace_add('write', on_mode_change)
    row += 1

    # --- Frame Options ---
    fopt_frame = ttk.Frame(frame)
    fopt_frame.grid(row=row, column=0, columnspan=3, sticky='w', pady=2)
    ttk.Label(fopt_frame, text="Skip:").pack(side='left')
    ttk.Spinbox(fopt_frame, textvariable=skip_var, from_=1, to=100, width=5).pack(side='left', padx=(5, 15))
    ttk.Label(fopt_frame, text="Skip Pattern:").pack(side='left')
    ttk.Entry(fopt_frame, textvariable=skip_pattern_var, width=10).pack(side='left', padx=(5, 15))
    ttk.Label(fopt_frame, text="Max Frames:").pack(side='left')
    ttk.Entry(fopt_frame, textvariable=max_frames_var, width=8).pack(side='left', padx=(5, 0))
    row += 1

    # --- Interpolation ---
    interp_frame = ttk.Frame(frame)
    interp_frame.grid(row=row, column=0, columnspan=3, sticky='w', pady=2)
    ttk.Label(interp_frame, text="Interpolation:").pack(side='left')
    ttk.Combobox(interp_frame, textvariable=interp_var,
                 values=['area', 'nearest', 'linear', 'cubic'],
                 state='readonly', width=10).pack(side='left', padx=(5, 0))
    row += 1

    # --- LED Map ---
    ttk.Label(frame, text="LED Map:").grid(row=row, column=0, sticky='w', pady=2)
    ttk.Entry(frame, textvariable=led_map_var).grid(row=row, column=1, sticky='ew', padx=(5, 0), pady=2)
    ttk.Button(frame, text="Browse...", command=browse_led_map).grid(row=row, column=2, padx=(5, 0), pady=2)
    row += 1

    # --- Monochrome Options (separator) ---
    ttk.Separator(frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='ew', pady=6)
    row += 1

    mono_label = ttk.Label(frame, text="Monochrome Options")
    mono_label.grid(row=row, column=0, columnspan=3, sticky='w', pady=(0, 2))
    row += 1

    # Threshold
    mono_row1 = ttk.Frame(frame)
    mono_row1.grid(row=row, column=0, columnspan=3, sticky='w', pady=2)
    lbl_thresh = ttk.Label(mono_row1, text="Threshold:")
    lbl_thresh.pack(side='left')
    sp_thresh = ttk.Spinbox(mono_row1, textvariable=threshold_var, from_=0, to=255, width=5)
    sp_thresh.pack(side='left', padx=(5, 15))
    mono_widgets.extend([lbl_thresh, sp_thresh])

    lbl_colors = ttk.Label(mono_row1, text="Colors:")
    lbl_colors.pack(side='left')
    ent_colors = ttk.Entry(mono_row1, textvariable=colors_var, width=5)
    ent_colors.pack(side='left', padx=(5, 0))
    mono_widgets.extend([lbl_colors, ent_colors])
    row += 1

    # Checkboxes
    mono_row2 = ttk.Frame(frame)
    mono_row2.grid(row=row, column=0, columnspan=3, sticky='w', pady=2)
    cb_adaptive = ttk.Checkbutton(mono_row2, text="Adaptive Threshold", variable=adaptive_var)
    cb_adaptive.pack(side='left', padx=(0, 10))
    cb_auto = ttk.Checkbutton(mono_row2, text="Auto Threshold", variable=auto_thresh_var)
    cb_auto.pack(side='left', padx=(0, 10))
    cb_dither = ttk.Checkbutton(mono_row2, text="Dither", variable=dither_var)
    cb_dither.pack(side='left', padx=(0, 10))
    cb_bw = ttk.Checkbutton(mono_row2, text="BW First", variable=bw_first_var)
    cb_bw.pack(side='left')
    mono_widgets.extend([cb_adaptive, cb_auto, cb_dither, cb_bw])
    row += 1

    # White color
    mono_row3 = ttk.Frame(frame)
    mono_row3.grid(row=row, column=0, columnspan=3, sticky='w', pady=2)
    lbl_white = ttk.Label(mono_row3, text="White (R,G,B):")
    lbl_white.pack(side='left')
    ent_white = ttk.Entry(mono_row3, textvariable=white_var, width=12)
    ent_white.pack(side='left', padx=(5, 0))
    mono_widgets.extend([lbl_white, ent_white])
    row += 1

    # Grey out mono options initially
    on_mode_change()

    # --- Keymap Integration ---
    ttk.Separator(frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='ew', pady=6)
    row += 1

    ttk.Label(frame, text="Keymap Integration").grid(row=row, column=0, columnspan=3, sticky='w', pady=(0, 2))
    row += 1

    cb_setup = ttk.Checkbutton(frame, text="Generate keymap files (.h + rules.mk)",
                                variable=setup_keymap_var, command=on_setup_keymap_change)
    cb_setup.grid(row=row, column=0, columnspan=3, sticky='w', pady=2)
    row += 1

    lbl_keymap = ttk.Label(frame, text="Keymap Dir:")
    lbl_keymap.grid(row=row, column=0, sticky='w', pady=2)
    ent_keymap = ttk.Entry(frame, textvariable=keymap_dir_var)
    ent_keymap.grid(row=row, column=1, sticky='ew', padx=(5, 0), pady=2)
    btn_keymap = ttk.Button(frame, text="Browse...", command=browse_keymap_dir)
    btn_keymap.grid(row=row, column=2, padx=(5, 0), pady=2)
    keymap_widgets.extend([lbl_keymap, ent_keymap, btn_keymap])
    row += 1

    # Grey out keymap dir initially
    on_setup_keymap_change()

    # --- Convert Button ---
    ttk.Separator(frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='ew', pady=6)
    row += 1

    convert_btn = ttk.Button(frame, text="Convert", command=do_convert)
    convert_btn.grid(row=row, column=0, columnspan=3, pady=4)
    row += 1

    # --- Log Output ---
    log_text = scrolledtext.ScrolledText(frame, height=14, state='disabled', wrap='word')
    log_text.grid(row=row, column=0, columnspan=3, sticky='nsew', pady=(4, 0))
    frame.rowconfigure(row, weight=1)

    # Start polling the log queue
    poll_log()

    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        launch_gui()
