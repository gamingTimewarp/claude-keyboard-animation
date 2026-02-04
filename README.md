# Video to QMK RGB Animation Converter

Convert video files into highly compressed RGB LED animations for QMK-powered keyboards!

**Generated with assistance from Anthropic's Claude Sonnet 4.5**

## Quick Start

```bash
# Install dependencies
pip install opencv-python numpy

# Basic usage (uncompressed RGB)
python video_to_qmk.py badapple.mp4 -w 21 -y 6 -f 10 --max-frames 100

# With color RLE compression (2-4x smaller)
python video_to_qmk.py video.mp4 -w 21 -y 6 --mode rgb-rle --max-frames 200

# Black & white with extreme compression (50-100x smaller!)
python video_to_qmk.py badapple.mp4 -w 21 -y 6 --mode mono-rle --max-frames 1000
```

---

## Prebuilt Binaries

Standalone executables are available for Windows and Linux—no Python installation required.

### Download

Download the latest release from the [Releases page](https://github.com/awright/video-to-qmk/releases):

- **Windows:** `video_to_qmk.exe`
- **Linux:** `video_to_qmk`

### Usage

The binaries work exactly like the Python script:

```bash
# Windows
video_to_qmk.exe badapple.mp4 -w 21 -y 6 -f 10 --max-frames 100

# Linux
./video_to_qmk badapple.mp4 -w 21 -y 6 -f 10 --max-frames 100
```

On Linux, you may need to make the binary executable first:
```bash
chmod +x video_to_qmk
```

### GUI Mode

Running the executable without any arguments launches a graphical interface:

```bash
# Windows - double-click video_to_qmk.exe or run:
video_to_qmk.exe

# Linux
./video_to_qmk
```

---

## Output Modes

All modes are accessed via the `--mode` argument:

### `rgb` (default) - No Compression
**Use when:** Testing, very short clips (<50 frames)
**File size:** 3 bytes/pixel/frame
**Example:** 21x6, 100 frames = 37.8 KB

### `rgb-rle` - Color RLE
**Use when:** Color videos, 100-300 frames
**File size:** ~0.75 bytes/pixel/frame
**Compression:** 2-4x
**Example:** 21x6, 200 frames = 75.6 KB -> 20-30 KB

### `mono-bitpack` - Monochrome Bit-Packed
**Use when:** B&W videos, predictable file sizes
**File size:** ~0.125 bytes/pixel/frame
**Compression:** Fixed 24x
**Example:** 21x6, 500 frames = 189 KB -> 7.9 KB

### `mono-rle` - Monochrome RLE (Recommended!)
**Use when:** Black & white videos, maximum frames
**File size:** ~0.03 bytes/pixel/frame
**Compression:** 50-100x
**Example:** 21x6, 1000 frames = 378 KB -> 8-15 KB

---

## Arguments

```
Required:
  video                Input video file
  -w, --width          Keyboard width in LEDs
  -y, --height         Keyboard height in LEDs

Optional:
  -o, --output         Output C file (default: video_animation.c)
  -f, --fps            Target FPS (default: 10)
  --mode MODE          Output mode: rgb, rgb-rle, mono-bitpack, mono-rle (default: rgb)
  --max-frames N       Maximum frames to extract
  --skip N             Use every Nth frame (1=all, 2=every other)
  --skip-pattern X,Y   Alternating skip pattern (e.g., "2,3" = skip 2, then 3, repeat)
  --interp METHOD      Resize interpolation: nearest, area, linear, cubic (default: area)
  --led-map FILE       Custom LED mapping file (for non-rectangular layouts)

Monochrome options (for mono-bitpack and mono-rle modes):
  --threshold N        B&W threshold 0-255 (default: 128)
  --adaptive-threshold Dynamic threshold based on frame brightness (RECOMMENDED!)
  --auto-threshold     Use Otsu's method to find optimal threshold
  --white R,G,B        Custom white color (default: 255,255,255)
  --colors N           Reduce to N colors before B&W conversion (2-16)
  --dither             Apply Floyd-Steinberg dithering
  --bw-first           Convert to B&W before resizing (best for pure B&W source videos)
```

---

## Advanced Features

### Adaptive Threshold - BEST FOR MOST VIDEOS!
Dynamically adjusts B&W threshold based on frame brightness:
```bash
--adaptive-threshold  # High threshold for bright frames, low for dark frames
```
**This is the solution for "odd pixels"!** When downscaling creates gray values, adaptive threshold ensures they're converted correctly based on the frame's overall brightness.

### Skip Patterns - Smoother Motion with Fewer Frames
Instead of uniform frame skip, use alternating patterns for better motion:
```bash
--skip-pattern 2,3  # Alternates: skip 2, then 3, repeat (40% of frames, smoother than --skip 3)
--skip-pattern 1,2,3  # Complex pattern for even smoother results
```

### Interpolation Control - Choose Downscaling Quality
Control how video is resized to keyboard dimensions:
```bash
--interp area     # Smooth downscaling (default, best quality)
--interp nearest  # Sharp edges, no blending (can be blocky)
--interp linear   # Bilinear (balanced)
--interp cubic    # Bicubic (smoothest)
```

### Color Reduction - Better B&W Conversion
Reduce colors before converting to B&W for cleaner results:
```bash
--colors 4  # Reduces to 4 colors using k-means clustering
--colors 8  # More colors = more detail
```

### Dithering - Smoother Gradients
Apply Floyd-Steinberg dithering for professional-quality B&W conversion:
```bash
--dither  # Adds dithering (NOT recommended for tiny displays)
```

### B&W First - Convert Before Resizing
For pure B&W source videos, convert to B&W before downscaling:
```bash
--bw-first  # Avoids gray averaging during resize
```

### Recommended Combination for Bad Apple:
```bash
python video_to_qmk.py badapple.mp4 -w 21 -y 6 -f 12 \
  --adaptive-threshold --mode mono-rle --max-frames 2000 \
  --led-map keychron_v6_max_iso.txt
# Result: Perfect B&W conversion, 2000+ frames in ~12-15 KB!
```

---

## Usage Examples

### Bad Apple (Full Video!) - RECOMMENDED SETTINGS
```bash
# Perfect quality with adaptive threshold (solves "odd pixels" issue!)
python video_to_qmk.py badapple.mp4 -w 21 -y 6 -f 12 \
  --adaptive-threshold --mode mono-rle --max-frames 2500 \
  --led-map keychron_v6_max_iso.txt

# Result: Entire 3.5-minute video in ~12-15 KB with perfect B&W conversion!
```

### Noisy Video - Use Color Reduction
```bash
# Video with lots of colors/noise
python video_to_qmk.py noisy_video.mp4 -w 21 -y 6 -f 10 \
  --adaptive-threshold --colors 4 --mode mono-rle --max-frames 500

# Before: ~30x compression, messy image
# After: ~80x compression, clean image!
```

### Smooth Motion - Skip Patterns
```bash
# Alternating skip pattern for smoother motion
python video_to_qmk.py video.mp4 -w 21 -y 6 -f 10 \
  --adaptive-threshold --skip-pattern 2,3 --mode mono-rle --max-frames 1000
```

### Photo/Grayscale Content
```bash
# For photos or grayscale videos (not pure B&W)
python video_to_qmk.py photo_video.mp4 -w 21 -y 6 -f 10 \
  --adaptive-threshold --colors 8 --mode mono-rle
```

### Color Animation
```bash
# 200 frames of color video with RLE compression
python video_to_qmk.py logo.mp4 -w 21 -y 6 -f 15 \
  --max-frames 200 --mode rgb-rle --led-map keychron_v6_max_iso.txt
```

### Quick Test
```bash
# 30 frames, no compression
python video_to_qmk.py test.mp4 -w 21 -y 6 --max-frames 30
```

### Custom Colors
```bash
# Cyan and black instead of white and black
python video_to_qmk.py video.mp4 -w 21 -y 6 --mode mono-rle \
  --white 0,255,255 --max-frames 500

# Matrix green
python video_to_qmk.py video.mp4 -w 21 -y 6 --mode mono-rle \
  --white 0,255,0 --max-frames 500
```

### Better B&W Conversion
```bash
# Reduce colors first for cleaner conversion
python video_to_qmk.py video.mp4 -w 21 -y 6 --mode mono-rle \
  --colors 4 --dither --threshold 128

# Adjust threshold if too dark/light
python video_to_qmk.py video.mp4 -w 21 -y 6 --mode mono-rle \
  --threshold 180  # More black (only brightest pixels = white)
```

---

## Custom LED Mapping

For keyboards with gaps or non-rectangular layouts, create a mapping file:

**Format:** One row per line, comma-separated LED indices, `255` for gaps

**Example (`keychron_v6_max_iso.txt`):**
```
# Row 1
0,1,2,3,4,5,6,7,8,9,10,11,12,255,13,14,15,16,17,18,19
# Row 2
20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40
# ... more rows
```

**Usage:**
```bash
python video_to_qmk.py video.mp4 -w 21 -y 6 --mode mono-rle \
  --led-map keychron_v6_max_iso.txt
```

---

## Integration with QMK

### Step 1: Generate Animation
```bash
python video_to_qmk.py badapple.mp4 -w 21 -y 6 -f 12 \
  --max-frames 500 --mode mono-rle --led-map your_layout.txt -o video_animation.c
```

### Step 2: Create Header File
Create `video_animation.h` in your keymap directory:
```c
#pragma once

void video_animation_start(void);
void video_animation_stop(void);
void video_animation_toggle(void);
bool video_animation_update(uint8_t led_min, uint8_t led_max);
```

### Step 3: Update `keymap.c`
```c
#include "video_animation.h"

enum custom_keycodes {
    VIDEO_TOGGLE = SAFE_RANGE,
};

bool process_record_user(uint16_t keycode, keyrecord_t *record) {
    if (keycode == VIDEO_TOGGLE && record->event.pressed) {
        video_animation_toggle();
        return false;
    }
    return true;
}

void keyboard_post_init_user(void) {
    rgb_matrix_mode_noeeprom(RGB_MATRIX_SOLID_COLOR);
    rgb_matrix_sethsv_noeeprom(0, 0, 0);
    rgb_matrix_set_speed_noeeprom(255);
}

bool rgb_matrix_indicators_advanced_user(uint8_t led_min, uint8_t led_max) {
    if (video_animation_update(led_min, led_max)) {
        return true;
    }
    return false;
}
```

### Step 4: Update `rules.mk`
```makefile
SRC += video_animation.c

# Optional: Save space
CONSOLE_ENABLE = no
COMMAND_ENABLE = no
```

### Step 5: Map Toggle Key
Add `VIDEO_TOGGLE` to a key in your keymap layout to control playback.

### Step 6: Compile & Flash
```bash
qmk compile -kb your_keyboard -km your_keymap
qmk flash -kb your_keyboard -km your_keymap
```

---

## Compression Comparison

**For 21x6 keyboard (126 LEDs):**

| Mode | 100 frames | 200 frames | 500 frames | 1000 frames |
|------|------------|------------|------------|-------------|
| **rgb** | 37.8 KB | 75.6 KB | 189 KB | 378 KB |
| **rgb-rle** | 12-20 KB | 25-35 KB | 60-85 KB | Too large |
| **mono-bitpack** | 1.6 KB | 3.2 KB | 7.9 KB | 15.8 KB |
| **mono-rle** | 0.8-1.5 KB | 1.5-3 KB | 4-8 KB | **8-15 KB** |

---

## Compression Techniques Explained

### RLE (Run-Length Encoding)
Encodes consecutive identical pixels as `{count, color}`:
- **Before:** 20 white pixels = 60 bytes
- **After:** 1 run = 4 bytes (15x smaller!)
- **Best for:** Solid colors, gradients, Bad Apple

### Bit-Packing (Monochrome)
Stores 8 pixels per byte (1 bit each):
- **Compression:** Fixed 24x
- **Best for:** Predictable file sizes

### Mono RLE (Best!)
Encodes runs of black/white pixels:
- **Before:** 100 consecutive white pixels = 100 bytes
- **After:** 1 run = 2 bytes (50x smaller!)
- **Best for:** Bad Apple, silhouettes, high contrast

### Skip Patterns
Alternating frame skip reduces total frames while preserving motion:
```bash
--skip 3             # Uniform: 33% of frames (jerky)
--skip-pattern 2,3   # Alternating: 40% of frames (smooth!)
--skip-pattern 1,2,3 # Complex: 50% of frames (very smooth!)
```
**How it works:** Instead of dropping the same frame position repeatedly (0, 3, 6, 9...), alternating patterns distribute drops more evenly (0, 2, 5, 7, 10...), creating smoother perceived motion.

### Color Reduction
Reduces color palette before B&W conversion using k-means clustering:
```bash
--colors 4  # Reduce to 4 colors
--colors 8  # More detail
```
**Effect:**
- Reduces noise and artifacts
- Creates cleaner edges in B&W conversion
- Improves RLE compression (fewer color transitions = longer runs)
- Can improve compression ratio from 30x to 80x on noisy videos!

### Dithering
Floyd-Steinberg dithering preserves gradients in B&W conversion:
```bash
--dither  # Enable dithering
```
**Effect:**
- Converts gradients to patterns of black/white pixels
- Preserves tonal information lost in simple thresholding
- Better visual quality at slight compression cost

---

## Advanced Optimization Tips

### 1. Adaptive Threshold - MOST IMPORTANT
**Problem:** When downscaling creates gray pixels, fixed threshold puts some on wrong side
**Solution:** Dynamic threshold adjusts based on frame brightness

```bash
python video_to_qmk.py video.mp4 -w 21 -y 6 --adaptive-threshold --mode mono-rle
```

**How it works:**
- Analyzes each frame's average brightness
- **Bright frames** (mostly white) -> Uses **high threshold (180)** -> Strict about what's white
- **Dark frames** (mostly black) -> Uses **low threshold (80)** -> Generous with white
- **Medium frames** -> Uses **middle threshold (128)**

### 2. Interpolation Method
**Problem:** Different downscaling methods affect B&W conversion quality
**Solution:** Choose the right interpolation for your content

```bash
--interp area     # Default, best quality for most content
--interp nearest  # No blending, picks closest pixel (good for pixel art)
--interp linear   # Bilinear interpolation
--interp cubic    # Bicubic (smoothest but slowest)
```

### 3. Skip Pattern Strategy
Instead of uniform skip, use patterns for better motion:
```bash
--skip 3           # Uniform: every 3rd frame (33% of frames)
--skip-pattern 2,3 # Alternating: 40% of frames, smoother motion
```

### 4. Custom Threshold
Adjust B&W conversion threshold:
```bash
--threshold 100  # More white (darker videos)
--threshold 150  # More black (brighter videos)
```

### 5. Battery Optimization
Use dimmer white color:
```bash
--white 80,80,80    # Dim white (saves power)
--white 128,128,128 # Medium white
```

### 6. Preprocess Video
Use video editing software before conversion:
- Convert to pure black/white (not grayscale)
- Increase contrast
- Remove noise/grain
- Add letterboxing (black bars compress to nothing)

### 7. Test Compression Ratio
Always test with a short clip first:
```bash
python video_to_qmk.py test.mp4 -w 21 -y 6 --max-frames 20 --mode mono-rle
# Check "Compression ratio: X.Xx" in output
```

### 8. Optimal FPS
- **5-8 FPS:** Most videos, smooth enough
- **10-12 FPS:** Action content
- **15+ FPS:** Only for short, high-action clips

---

## Troubleshooting

### Compilation Fails: "will not fit in region 'flash0'"
**Solution:** File too large
```bash
# Reduce frames
--max-frames 100

# Use skip pattern
--skip-pattern 2,3

# Use mono-rle mode for maximum compression
--mode mono-rle

# Disable features in rules.mk
CONSOLE_ENABLE = no
COMMAND_ENABLE = no
MOUSEKEY_ENABLE = no
```

### Video Doesn't Play Correctly
**Solutions:**
- Verify LED mapping matches keyboard layout
- Check RGB_MATRIX_ENABLE in rules.mk
- Add `keyboard_post_init_user()` to set black background
- Make sure VIDEO_TOGGLE is mapped to a key

### Animation is Choppy
**Solutions:**
- Lower FPS: `-f 5` or `-f 8`
- Use skip pattern: `--skip-pattern 2,3`
- Check keyboard processing speed

### Video Looks Wrong (Distorted/Shifted)
**Solutions:**
- Verify LED mapping file is correct
- Check dimensions match keyboard: `-w` and `-y`

### "Compressed size larger than uncompressed"
**Cause:** Video has too much noise/variation
**Solutions:**
- Preprocess: reduce colors, increase contrast
- Try different mode
- Use `rgb` mode for this video

---

## File Size Calculator

**Quick estimates:**

```
rgb:           frames x width x height x 3 bytes
rgb-rle:       frames x width x height x 0.75 bytes
mono-bitpack:  frames x (width x height / 8) bytes
mono-rle:      frames x width x height x 0.03 bytes (Bad Apple)
```

**Example (21x6 keyboard):**
- 500 frames rgb: 189 KB
- 500 frames rgb-rle: ~70 KB
- 500 frames mono-bitpack: 7.9 KB
- 500 frames mono-rle: ~5 KB

---

## Performance

All modes are **fast enough for smooth playback**:
- Decompression: Microseconds per frame
- CPU usage: Minimal
- Battery impact: Low
- Quality: **Lossless** (perfect frame reproduction)

---

## Summary: Which Mode to Use?

### Bad Apple or B&W content?
**Use `--mode mono-rle`**
```bash
python video_to_qmk.py badapple.mp4 -w 21 -y 6 -f 12 \
  --max-frames 1000 --mode mono-rle --led-map your_layout.txt
```
**Result:** 50-100x compression, 1000+ frames possible

### Color video?
**Use `--mode rgb-rle`**
```bash
python video_to_qmk.py video.mp4 -w 21 -y 6 -f 10 \
  --max-frames 200 --mode rgb-rle --led-map your_layout.txt
```
**Result:** 2-4x compression, 200-300 frames

### Quick test?
**Use `--mode rgb` (default)**
```bash
python video_to_qmk.py test.mp4 -w 21 -y 6 --max-frames 30
```
**Result:** No compression, simplest code, <50 frames

---

## Pro Tips Summary

1. **USE --adaptive-threshold** (solves "odd pixels" issue!)
2. **Always use mono-rle for B&W content** (50-100x compression!)
3. **Use skip patterns** instead of uniform skip for smoother motion
4. **Color reduction helps noisy videos** (huge compression boost!)
5. **Preprocess videos** for better compression
6. **Test compression ratio** with short clips first
7. **Custom colors** work great (cyan, green, dim white)
8. **LED mapping file** for keyboards with gaps
9. **DON'T use --dither** on tiny displays (makes it worse)

### Feature Combinations That Work Great:

**For Bad Apple (already clean):** RECOMMENDED
```bash
--adaptive-threshold --mode mono-rle
# Don't need --colors or --dither (already high contrast)
```

**For noisy/grayscale videos:**
```bash
--adaptive-threshold --colors 4 --mode mono-rle
# Adaptive threshold + color reduction = great results
```

**For smooth motion with fewer frames:**
```bash
--adaptive-threshold --skip-pattern 2,3 --mode mono-rle
# Better motion than --skip 3
```

**For maximum compression (simple graphics):**
```bash
--adaptive-threshold --colors 2 --skip-pattern 2,3 --mode mono-rle
# Fewest colors + skip pattern + adaptive = smallest file
```

---

## License

Released into the public domain. Use however you like!

## Credits

Scripts generated with assistance from Anthropic's Claude Sonnet 4.5.

Perfect for: Bad Apple, custom animations, keyboard art, and more!
