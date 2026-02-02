#!/usr/bin/env python3
"""
Video to QMK RGB Animation Converter
Converts a video file into a QMK-compatible RGB matrix animation.
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


def extract_and_resize_frames(video_path, width, height, max_frames=None, frame_skip=1):
    """
    Extract frames from video and resize to keyboard dimensions.
    
    Args:
        video_path: Path to video file
        width: Keyboard width in LEDs
        height: Keyboard height in LEDs
        max_frames: Maximum number of frames to extract (None for all)
        frame_skip: Skip every N frames (1 = use all frames, 2 = use every other frame)
    
    Returns:
        List of resized frames as numpy arrays
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0
    extracted = 0
    
    print(f"Extracting and resizing frames to {width}x{height}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames if needed
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue
        
        # Resize frame to keyboard dimensions
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        
        # Convert BGR to RGB
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


def generate_qmk_code(frames, output_path, fps, keyboard_width, keyboard_height, led_map=None):
    """
    Generate QMK C code for the animation.
    
    Args:
        frames: List of RGB frames
        output_path: Output file path
        fps: Target frames per second
        keyboard_width: Keyboard width in LEDs
        keyboard_height: Keyboard height in LEDs
        led_map: Optional LED mapping array (2D list)
    """
    with open(output_path, 'w') as f:
        # Header
        f.write("// Generated video animation for QMK RGB Matrix\n")
        f.write(f"// Total frames: {len(frames)}\n")
        f.write(f"// Target FPS: {fps}\n")
        f.write(f"// Dimensions: {keyboard_width}x{keyboard_height}\n\n")
        
        f.write("#include QMK_KEYBOARD_H\n")
        f.write("#include \"rgb_matrix.h\"\n\n")
        
        # Define constants
        f.write(f"#define VIDEO_FRAME_COUNT {len(frames)}\n")
        f.write(f"#define VIDEO_WIDTH {keyboard_width}\n")
        f.write(f"#define VIDEO_HEIGHT {keyboard_height}\n")
        f.write(f"#define VIDEO_FPS {fps}\n")
        f.write(f"#define FRAME_DELAY_MS {int(1000/fps)}\n\n")
        
        # Generate frame data
        f.write("// Frame data stored as RGB888\n")
        f.write("static const uint8_t PROGMEM video_frames[VIDEO_FRAME_COUNT][VIDEO_HEIGHT][VIDEO_WIDTH][3] = {\n")
        
        for frame_idx, frame in enumerate(frames):
            f.write(f"    // Frame {frame_idx}\n")
            f.write("    {\n")
            
            for y in range(keyboard_height):
                f.write("        {")
                for x in range(keyboard_width):
                    r, g, b = frame[y, x]
                    f.write(f"{{{r},{g},{b}}}")
                    if x < keyboard_width - 1:
                        f.write(",")
                f.write("}")
                if y < keyboard_height - 1:
                    f.write(",")
                f.write("\n")
            
            f.write("    }")
            if frame_idx < len(frames) - 1:
                f.write(",")
            f.write("\n")
        
        f.write("};\n\n")
        
        # Add LED mapping if provided
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
        
        # Generate animation control code
        f.write("""// Animation state
static uint16_t current_frame = 0;
static uint32_t last_frame_time = 0;
static bool animation_playing = false;

// Start the animation
void video_animation_start(void) {
    current_frame = 0;
    last_frame_time = timer_read32();
    animation_playing = true;
    
    // Disable RGB matrix effects
    rgb_matrix_mode_noeeprom(RGB_MATRIX_SOLID_COLOR);
    rgb_matrix_sethsv_noeeprom(0, 0, 0);
}

// Stop the animation
void video_animation_stop(void) {
    animation_playing = false;
    
    // Re-enable RGB matrix (optional - choose your preferred effect)
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
        
        # Generate the correct update function based on whether LED mapping is provided
        if led_map:
            # Version with LED mapping
            f.write("""// Update function - call this in rgb_matrix_indicators_advanced_user()
bool video_animation_update(uint8_t led_min, uint8_t led_max) {
    if (!animation_playing) {
        return false;
    }
    
    // CRITICAL: Display current frame EVERY time to prevent blinking
    for (uint8_t y = 0; y < VIDEO_HEIGHT; y++) {
        for (uint8_t x = 0; x < VIDEO_WIDTH; x++) {
            // Get LED index from mapping array
            uint8_t led_index = pgm_read_byte(&led_map[y][x]);
            
            // Skip if no LED at this position (255) or out of range
            if (led_index == 255) continue;
            if (led_index < led_min || led_index >= led_max) continue;
            
            uint8_t r = pgm_read_byte(&video_frames[current_frame][y][x][0]);
            uint8_t g = pgm_read_byte(&video_frames[current_frame][y][x][1]);
            uint8_t b = pgm_read_byte(&video_frames[current_frame][y][x][2]);
            
            rgb_matrix_set_color(led_index, r, g, b);
        }
    }
    
    // Check if it's time to advance to next frame
    uint32_t now = timer_read32();
    if (now - last_frame_time >= FRAME_DELAY_MS) {
        last_frame_time = now;
        
        // Advance to next frame
        current_frame++;
        if (current_frame >= VIDEO_FRAME_COUNT) {
            current_frame = 0; // Loop the animation
        }
    }
    
    return true;
}
""")
        else:
            # Version without LED mapping (sequential addressing)
            f.write("""// Update function - call this in rgb_matrix_indicators_advanced_user()
bool video_animation_update(uint8_t led_min, uint8_t led_max) {
    if (!animation_playing) {
        return false;
    }
    
    // CRITICAL: Display current frame EVERY time to prevent blinking
    for (uint8_t y = 0; y < VIDEO_HEIGHT; y++) {
        for (uint8_t x = 0; x < VIDEO_WIDTH; x++) {
            // Calculate LED index (sequential: 0, 1, 2, 3...)
            uint8_t led_index = y * VIDEO_WIDTH + x;
            
            if (led_index < led_min || led_index >= led_max) continue;
            
            uint8_t r = pgm_read_byte(&video_frames[current_frame][y][x][0]);
            uint8_t g = pgm_read_byte(&video_frames[current_frame][y][x][1]);
            uint8_t b = pgm_read_byte(&video_frames[current_frame][y][x][2]);
            
            rgb_matrix_set_color(led_index, r, g, b);
        }
    }
    
    // Check if it's time to advance to next frame
    uint32_t now = timer_read32();
    if (now - last_frame_time >= FRAME_DELAY_MS) {
        last_frame_time = now;
        
        // Advance to next frame
        current_frame++;
        if (current_frame >= VIDEO_FRAME_COUNT) {
            current_frame = 0; // Loop the animation
        }
    }
    
    return true;
}
""")
        
        # Add usage examples
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


def main():
    parser = argparse.ArgumentParser(
        description="Convert video to QMK RGB matrix animation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert video to 15x5 keyboard with 10 FPS
  python video_to_qmk.py input.mp4 -w 15 -y 5 -f 10
  
  # Use every 3rd frame and limit to 100 frames
  python video_to_qmk.py input.mp4 -w 15 -y 5 --skip 3 --max-frames 100
  
  # Use custom LED mapping
  python video_to_qmk.py input.mp4 -w 21 -y 6 --led-map keychron_v6_map.txt
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
    parser.add_argument('--max-frames', type=int, default=None, 
                        help='Maximum number of frames to extract (default: all)')
    parser.add_argument('--led-map', type=str, default=None,
                        help='Path to file containing LED mapping (one row per line, comma-separated, use 255 for gaps)')
    
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
    if led_map:
        print(f"  LED mapping: Custom mapping loaded")
    else:
        print(f"  LED mapping: Sequential (0, 1, 2, 3...)")
    
    # Warn about large file sizes
    estimated_size_kb = (effective_frames * args.width * args.height * 3) / 1024
    print(f"  Estimated output size: ~{estimated_size_kb:.1f} KB")
    
    if estimated_size_kb > 1000:
        print("\n⚠️  WARNING: Output file will be very large!")
        print("    Consider using --skip or --max-frames to reduce size")
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
    
    # Generate QMK code
    generate_qmk_code(frames, args.output, args.fps, args.width, args.height, led_map)
    
    print("\n✓ Conversion complete!")
    print(f"\nNext steps:")
    print(f"1. Copy {args.output} to your QMK keyboard's keymap directory")
    print(f"2. Create video_animation.h with function declarations")
    print(f"3. Add 'SRC += video_animation.c' to your keymap's rules.mk")
    print(f"4. Include the header and integrate with your keymap.c (see examples in generated file)")
    print(f"5. Compile and flash!")


if __name__ == "__main__":
    main()
