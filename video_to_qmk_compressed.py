#!/usr/bin/env python3
"""
Video to QMK RGB Animation Converter with Compression
Converts a video file into a QMK-compatible RGB matrix animation with optional RLE compression.
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


def rle_compress_frame(frame):
    """Run-Length Encode a frame."""
    height, width, _ = frame.shape
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


def calculate_compression_stats(frames, width, height):
    """Calculate compression statistics."""
    uncompressed_size = len(frames) * width * height * 3
    
    compressed_sizes = []
    for frame in frames:
        rle_data = rle_compress_frame(frame)
        compressed_sizes.append(len(rle_data) * 4)
    
    total_compressed = sum(compressed_sizes)
    avg_compressed = total_compressed / len(frames) if frames else 0
    
    return {
        'uncompressed': uncompressed_size,
        'compressed': total_compressed,
        'ratio': uncompressed_size / total_compressed if total_compressed > 0 else 0,
        'savings_percent': ((uncompressed_size - total_compressed) / uncompressed_size * 100) if uncompressed_size > 0 else 0,
        'avg_per_frame': avg_compressed
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


def generate_qmk_code(frames, output_path, fps, keyboard_width, keyboard_height, led_map=None, use_compression=False):
    """Generate QMK C code for the animation."""
    with open(output_path, 'w') as f:
        # Header
        f.write("// Generated video animation for QMK RGB Matrix\n")
        f.write(f"// Total frames: {len(frames)}\n")
        f.write(f"// Target FPS: {fps}\n")
        f.write(f"// Dimensions: {keyboard_width}x{keyboard_height}\n")
        
        if use_compression:
            stats = calculate_compression_stats(frames, keyboard_width, keyboard_height)
            f.write(f"// Compression: RLE (Run-Length Encoding)\n")
            f.write(f"// Uncompressed size: {stats['uncompressed']} bytes\n")
            f.write(f"// Compressed size: {stats['compressed']} bytes\n")
            f.write(f"// Compression ratio: {stats['ratio']:.2f}x\n")
            f.write(f"// Space saved: {stats['savings_percent']:.1f}%\n")
        else:
            total_size = len(frames) * keyboard_width * keyboard_height * 3
            f.write(f"// Compression: None\n")
            f.write(f"// Total size: {total_size} bytes\n")
        
        f.write("\n")
        f.write("#include QMK_KEYBOARD_H\n")
        f.write("#include \"rgb_matrix.h\"\n\n")
        
        # Define constants
        f.write(f"#define VIDEO_FRAME_COUNT {len(frames)}\n")
        f.write(f"#define VIDEO_WIDTH {keyboard_width}\n")
        f.write(f"#define VIDEO_HEIGHT {keyboard_height}\n")
        f.write(f"#define VIDEO_FPS {fps}\n")
        f.write(f"#define FRAME_DELAY_MS {int(1000/fps)}\n\n")
        
        # Add LED mapping if provided
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
        
        # Generate frame data
        if use_compression:
            write_compressed_data(f, frames, keyboard_width, keyboard_height)
        else:
            write_uncompressed_data(f, frames, keyboard_width, keyboard_height)
        
        # Animation state - MUST be declared here, before functions
        f.write("""// Animation state
static uint16_t current_frame = 0;
static uint32_t last_frame_time = 0;
static bool animation_playing = false;

""")
        
        # Write functions
        if use_compression:
            write_compressed_functions(f, led_map)
        else:
            write_uncompressed_functions(f, led_map)
        
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
    
    print(f"\nGenerated QMK code: {output_path}")


def write_uncompressed_data(f, frames, width, height):
    """Write uncompressed frame data."""
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


def write_compressed_data(f, frames, width, height):
    """Write RLE compressed frame data."""
    # Compress all frames
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
    
    # Write frame offsets
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
    
    # Write RLE compressed data
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


def write_compressed_functions(f, led_map):
    """Write RLE decompression functions."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Convert video to QMK RGB matrix animation with optional compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python video_to_qmk.py input.mp4 -w 21 -y 6 -f 10 --max-frames 100
  
  # With RLE compression (recommended for longer animations)
  python video_to_qmk.py input.mp4 -w 21 -y 6 -f 10 --max-frames 200 --compress
  
  # With custom LED mapping and compression
  python video_to_qmk.py input.mp4 -w 21 -y 6 --led-map v6_map.txt --compress
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
                        help='Frame skip - use every Nth frame (default: 1)')
    parser.add_argument('--max-frames', type=int, default=None, 
                        help='Maximum number of frames to extract (default: all)')
    parser.add_argument('--led-map', type=str, default=None,
                        help='Path to LED mapping file')
    parser.add_argument('--compress', action='store_true',
                        help='Use RLE compression to reduce file size')
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.video).exists():
        print(f"Error: Video file '{args.video}' not found")
        return
    
    if args.width <= 0 or args.height <= 0:
        print("Error: Width and height must be positive")
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
    print(f"  Compression: {'RLE' if args.compress else 'None'}")
    
    # Warn about large file sizes
    estimated_size_kb = (effective_frames * args.width * args.height * 3) / 1024
    print(f"  Estimated uncompressed size: ~{estimated_size_kb:.1f} KB")
    
    if args.compress:
        estimated_compressed = estimated_size_kb / 2.5
        print(f"  Estimated compressed size: ~{estimated_compressed:.1f} KB (rough estimate)")
    
    if estimated_size_kb > 1000 and not args.compress:
        print("\n⚠️  WARNING: Output file will be very large!")
        print("    Consider using --compress for RLE compression")
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
        frame_skip=args.skip
    )
    
    # Show actual compression stats if using compression
    if args.compress and frames:
        stats = calculate_compression_stats(frames, args.width, args.height)
        print(f"\nCompression statistics:")
        print(f"  Uncompressed: {stats['uncompressed']/1024:.1f} KB")
        print(f"  Compressed: {stats['compressed']/1024:.1f} KB")
        print(f"  Compression ratio: {stats['ratio']:.2f}x")
        print(f"  Space saved: {stats['savings_percent']:.1f}%")
    
    # Generate QMK code
    generate_qmk_code(frames, args.output, args.fps, args.width, args.height, led_map, args.compress)
    
    print("\n✓ Conversion complete!")
    print(f"\nNext steps:")
    print(f"1. Copy {args.output} to your QMK keyboard's keymap directory")
    print(f"2. Create video_animation.h with function declarations")
    print(f"3. Add 'SRC += video_animation.c' to rules.mk")
    print(f"4. Integrate with your keymap.c")
    print(f"5. Compile and flash!")


if __name__ == "__main__":
    main()
