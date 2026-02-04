#!/usr/bin/env python3
"""
Video to QMK RGB Animation Converter - Monochrome Optimized
Converts black & white videos into highly compressed QMK animations using bit-packing.
Each pixel = 1 bit instead of 24 bits = 24x compression for pure B&W!
"""

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
    """Load LED mapping from file."""
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


def convert_to_monochrome(frame, threshold=128):
    """
    Convert frame to pure black and white (1-bit per pixel).
    Returns array of booleans: True = white, False = black
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Threshold to pure black/white
    return gray > threshold


def pack_bits(mono_frame):
    """
    Pack monochrome frame into bytes (8 pixels per byte).
    Returns bytes array.
    """
    height, width = mono_frame.shape
    total_pixels = height * width
    
    # Flatten to 1D
    flat = mono_frame.flatten()
    
    # Pack 8 pixels into each byte
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
    Much more efficient than color RLE for B&W!
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


def calculate_compression_stats(frames, width, height, mode):
    """Calculate compression statistics."""
    uncompressed_size = len(frames) * width * height * 3  # RGB888
    
    if mode == 'bitpack':
        # 1 bit per pixel, packed into bytes
        compressed_size = len(frames) * ((width * height + 7) // 8)
    elif mode == 'rle':
        # RLE: 2 bytes per run (count + is_white flag)
        compressed_size = 0
        for frame in frames:
            mono = convert_to_monochrome(frame)
            rle_data = rle_compress_monochrome(mono)
            compressed_size += len(rle_data) * 2
    else:  # normal RLE
        compressed_size = 0
        for frame in frames:
            # Simulate RLE on color data
            pixels = frame.reshape(-1, 3)
            runs = 1
            for i in range(1, len(pixels)):
                if not np.array_equal(pixels[i], pixels[i-1]):
                    runs += 1
            compressed_size += runs * 4
    
    return {
        'uncompressed': uncompressed_size,
        'compressed': compressed_size,
        'ratio': uncompressed_size / compressed_size if compressed_size > 0 else 0,
        'savings_percent': ((uncompressed_size - compressed_size) / uncompressed_size * 100) if uncompressed_size > 0 else 0
    }


def extract_and_resize_frames(video_path, width, height, max_frames=None, frame_skip=1):
    """Extract frames from video and resize to keyboard dimensions."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0
    extracted = 0
    
    print(f"Extracting and resizing frames to {width}x{height}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue
        
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        frames.append(rgb_frame)
        extracted += 1
        
        if max_frames and extracted >= max_frames:
            break
        
        frame_idx += 1
        
        if extracted % 10 == 0:
            print(f"  Extracted {extracted} frames...", end='\r')
    
    cap.release()
    print(f"\nExtracted {len(frames)} frames total")
    return frames


def generate_qmk_code(frames, output_path, fps, keyboard_width, keyboard_height, 
                     led_map=None, mode='bitpack', threshold=128, white_color=(255,255,255)):
    """
    Generate QMK C code for monochrome animation.
    
    Modes:
    - 'bitpack': 1 bit per pixel, packed into bytes (24x compression)
    - 'rle': Run-length encoding of black/white runs (variable, often 50-100x!)
    """
    with open(output_path, 'w') as f:
        # Header
        stats = calculate_compression_stats(frames, keyboard_width, keyboard_height, mode)
        
        f.write("// Generated MONOCHROME video animation for QMK RGB Matrix\n")
        f.write(f"// Total frames: {len(frames)}\n")
        f.write(f"// Target FPS: {fps}\n")
        f.write(f"// Dimensions: {keyboard_width}x{keyboard_height}\n")
        f.write(f"// Mode: {mode.upper()}\n")
        f.write(f"// Uncompressed size: {stats['uncompressed']} bytes\n")
        f.write(f"// Compressed size: {stats['compressed']} bytes\n")
        f.write(f"// Compression ratio: {stats['ratio']:.1f}x\n")
        f.write(f"// Space saved: {stats['savings_percent']:.1f}%\n\n")
        
        f.write("#include QMK_KEYBOARD_H\n")
        f.write("#include \"rgb_matrix.h\"\n\n")
        
        # Define constants
        f.write(f"#define VIDEO_FRAME_COUNT {len(frames)}\n")
        f.write(f"#define VIDEO_WIDTH {keyboard_width}\n")
        f.write(f"#define VIDEO_HEIGHT {keyboard_height}\n")
        f.write(f"#define VIDEO_FPS {fps}\n")
        f.write(f"#define FRAME_DELAY_MS {int(1000/fps)}\n")
        f.write(f"#define WHITE_R {white_color[0]}\n")
        f.write(f"#define WHITE_G {white_color[1]}\n")
        f.write(f"#define WHITE_B {white_color[2]}\n\n")
        
        # LED mapping
        if led_map:
            f.write("// LED mapping array\n")
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
        
        # Generate frame data based on mode
        if mode == 'bitpack':
            write_bitpacked_data(f, frames, keyboard_width, keyboard_height, threshold)
            write_bitpack_functions(f, led_map)
        elif mode == 'rle':
            write_rle_monochrome_data(f, frames, keyboard_width, keyboard_height, threshold)
            write_rle_monochrome_functions(f, led_map)
        
        # Animation state
        f.write("""// Animation state
static uint16_t current_frame = 0;
static uint32_t last_frame_time = 0;
static bool animation_playing = false;

// Start the animation
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
    
    print(f"\nGenerated monochrome QMK code: {output_path}")
    print(f"Mode: {mode}, Compression: {stats['ratio']:.1f}x")


def write_bitpacked_data(f, frames, width, height, threshold):
    """Write bit-packed monochrome data."""
    print("Converting to monochrome and bit-packing...")
    
    bytes_per_frame = (width * height + 7) // 8
    
    f.write(f"// Bit-packed monochrome frames ({bytes_per_frame} bytes per frame)\n")
    f.write(f"// Each bit: 0 = black, 1 = white\n")
    f.write(f"static const uint8_t PROGMEM mono_frames[VIDEO_FRAME_COUNT][{bytes_per_frame}] = {{\n")
    
    for frame_idx, frame in enumerate(frames):
        mono = convert_to_monochrome(frame, threshold)
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
            if (led_index == 255) {
                pixel_idx++;
                continue;
            }
            if (led_index < led_min || led_index >= led_max) {
                pixel_idx++;
                continue;
            }
            
            if (is_white) {
                rgb_matrix_set_color(led_index, WHITE_R, WHITE_G, WHITE_B);
            } else {
                rgb_matrix_set_color(led_index, 0, 0, 0);
            }
""")
    else:
        f.write("""            uint8_t led_index = y * VIDEO_WIDTH + x;
            if (led_index < led_min || led_index >= led_max) {
                pixel_idx++;
                continue;
            }
            
            if (is_white) {
                rgb_matrix_set_color(led_index, WHITE_R, WHITE_G, WHITE_B);
            } else {
                rgb_matrix_set_color(led_index, 0, 0, 0);
            }
""")
    
    f.write("""            pixel_idx++;
        }
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


def write_rle_monochrome_data(f, frames, width, height, threshold):
    """Write RLE monochrome data."""
    print("Converting to monochrome and RLE compressing...")
    
    compressed_frames = []
    frame_offsets = [0]
    total_runs = 0
    
    for i, frame in enumerate(frames):
        mono = convert_to_monochrome(frame, threshold)
        rle_data = rle_compress_monochrome(mono)
        compressed_frames.append(rle_data)
        total_runs += len(rle_data)
        frame_offsets.append(total_runs)
        
        if (i + 1) % 10 == 0:
            print(f"  Compressed {i + 1}/{len(frames)} frames...", end='\r')
    print()
    
    # Frame offsets
    f.write("// Frame offsets in RLE data\n")
    f.write(f"static const uint16_t PROGMEM frame_offsets[VIDEO_FRAME_COUNT + 1] = {{\n    ")
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


def write_rle_monochrome_functions(f, led_map):
    """Write RLE monochrome decompression functions."""
    f.write("""// Decode and display RLE monochrome frame
static void decode_and_display_mono_rle(uint16_t frame_idx, uint8_t led_min, uint8_t led_max) {
    uint16_t start_offset = pgm_read_word(&frame_offsets[frame_idx]);
    uint16_t end_offset = pgm_read_word(&frame_offsets[frame_idx + 1]);
    
    uint16_t pixel_idx = 0;
    
    for (uint16_t offset = start_offset; offset < end_offset; offset++) {
        uint8_t count = pgm_read_byte(&rle_data[offset * 2]);
        bool is_white = pgm_read_byte(&rle_data[offset * 2 + 1]);
        
        for (uint8_t i = 0; i < count; i++) {
            uint8_t y = pixel_idx / VIDEO_WIDTH;
            uint8_t x = pixel_idx % VIDEO_WIDTH;
            pixel_idx++;
            
""")
    
    if led_map:
        f.write("""            uint8_t led_index = pgm_read_byte(&led_map[y][x]);
            if (led_index == 255) continue;
            if (led_index < led_min || led_index >= led_max) continue;
            
            if (is_white) {
                rgb_matrix_set_color(led_index, WHITE_R, WHITE_G, WHITE_B);
            } else {
                rgb_matrix_set_color(led_index, 0, 0, 0);
            }
""")
    else:
        f.write("""            uint8_t led_index = y * VIDEO_WIDTH + x;
            if (led_index < led_min || led_index >= led_max) continue;
            
            if (is_white) {
                rgb_matrix_set_color(led_index, WHITE_R, WHITE_G, WHITE_B);
            } else {
                rgb_matrix_set_color(led_index, 0, 0, 0);
            }
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


def main():
    parser = argparse.ArgumentParser(
        description="Convert B&W video to highly compressed QMK animation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Monochrome Compression Modes:
  bitpack - 1 bit per pixel, 24x compression (best for simple playback)
  rle     - Run-length encoding, 50-100x compression (best for Bad Apple!)

Examples:
  # Bad Apple with RLE (extreme compression!)
  python video_to_qmk_mono.py badapple.mp4 -w 21 -y 6 --mode rle --max-frames 1000
  
  # Bit-packed monochrome
  python video_to_qmk_mono.py video.mp4 -w 21 -y 6 --mode bitpack --max-frames 500
  
  # Custom white color (e.g., cyan)
  python video_to_qmk_mono.py video.mp4 -w 21 -y 6 --mode rle --white 0,255,255
        """
    )
    
    parser.add_argument('video', type=str, help='Input video file')
    parser.add_argument('-w', '--width', type=int, required=True, help='Keyboard width in LEDs')
    parser.add_argument('-y', '--height', type=int, required=True, help='Keyboard height in LEDs')
    parser.add_argument('-o', '--output', type=str, default='video_animation.c', 
                        help='Output C file')
    parser.add_argument('-f', '--fps', type=int, default=10, help='Target FPS')
    parser.add_argument('--skip', type=int, default=1, help='Frame skip')
    parser.add_argument('--max-frames', type=int, default=None, help='Max frames to extract')
    parser.add_argument('--led-map', type=str, default=None, help='LED mapping file')
    parser.add_argument('--mode', choices=['bitpack', 'rle'], default='rle',
                        help='Compression mode (rle recommended for Bad Apple)')
    parser.add_argument('--threshold', type=int, default=128,
                        help='B&W threshold 0-255 (default: 128)')
    parser.add_argument('--white', type=str, default='255,255,255',
                        help='White color as R,G,B (default: 255,255,255)')
    
    args = parser.parse_args()
    
    # Validate
    if not Path(args.video).exists():
        print(f"Error: Video file '{args.video}' not found")
        return
    
    # Parse white color
    try:
        white_color = tuple(int(x) for x in args.white.split(','))
        if len(white_color) != 3 or any(c < 0 or c > 255 for c in white_color):
            raise ValueError()
    except:
        print("Error: --white must be R,G,B (e.g., 255,255,255)")
        return
    
    # Load LED map
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
    
    # Extract frames
    frames = extract_and_resize_frames(
        args.video, 
        args.width, 
        args.height,
        max_frames=args.max_frames,
        frame_skip=args.skip
    )
    
    # Show compression preview
    stats = calculate_compression_stats(frames, args.width, args.height, args.mode)
    print(f"\nCompression preview:")
    print(f"  Mode: {args.mode}")
    print(f"  Uncompressed: {stats['uncompressed']/1024:.1f} KB")
    print(f"  Compressed: {stats['compressed']/1024:.1f} KB")
    print(f"  Ratio: {stats['ratio']:.1f}x")
    print(f"  Savings: {stats['savings_percent']:.1f}%")
    
    # Generate
    generate_qmk_code(frames, args.output, args.fps, args.width, args.height, 
                     led_map, args.mode, args.threshold, white_color)
    
    print("\n✓ Conversion complete!")
    print(f"\nWith {stats['ratio']:.1f}x compression, you could fit {int(stats['ratio'])} times")
    print(f"as many frames compared to uncompressed RGB!")


if __name__ == "__main__":
    main()
