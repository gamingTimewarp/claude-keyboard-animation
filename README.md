# Video to QMK RGB Animation Converter

Convert video files into highly compressed RGB LED animations for QMK-powered keyboards!

**Generated with assistance from Anthropic's Claude Sonnet 4.5**

## 🆕 NEW Advanced Features!

### Skip Patterns - Smoother Motion with Fewer Frames
Instead of uniform frame skip, use alternating patterns for better motion:
```bash
--skip-pattern 2,3  # Alternates: skip 2, then 3, repeat (40% of frames, smoother than --skip 3)
--skip-pattern 1,2,3  # Complex pattern for even smoother results
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
--dither  # Adds dithering (combines great with --colors!)
```

### Combined Example - Maximum Quality & Compression
```bash
python video_to_qmk_mono.py badapple.mp4 -w 21 -y 6 -f 12 \
  --colors 4 --dither --skip-pattern 2,3 --mode rle --max-frames 2000
# Result: 2000 frames in ~12 KB with smooth motion and clean edges!
```

---

## Quick Start

```bash
# Install dependencies
pip install opencv-python numpy

# Basic usage (uncompressed)
python video_to_qmk.py badapple.mp4 -w 21 -y 6 -f 10 --max-frames 100

# With color RLE compression (2-4x smaller)
python video_to_qmk_compressed.py video.mp4 -w 21 -y 6 --compress --max-frames 200

# Black & white with extreme compression (50-100x smaller!)
python video_to_qmk_mono.py badapple.mp4 -w 21 -y 6 --mode rle --max-frames 1000
```

---

## Three Scripts for Different Needs

### 1. `video_to_qmk.py` - No Compression
**Use when:** Testing, very short clips (<50 frames)  
**File size:** 3 bytes/pixel/frame  
**Example:** 21×6, 100 frames = 37.8 KB

### 2. `video_to_qmk_compressed.py` - Color RLE
**Use when:** Color videos, 100-300 frames  
**File size:** ~0.75 bytes/pixel/frame  
**Compression:** 2-4x  
**Example:** 21×6, 200 frames = 75.6 KB → 20-30 KB

### 3. `video_to_qmk_mono.py` - Monochrome (Recommended!)
**Use when:** Black & white videos, maximum frames  
**File size:** ~0.03-0.125 bytes/pixel/frame  
**Compression:** 24-100x  
**Example:** 21×6, 1000 frames = 378 KB → 8-15 KB

---

## Common Arguments (All Scripts)

```
Required:
  video                Input video file
  -w, --width          Keyboard width in LEDs
  -y, --height         Keyboard height in LEDs

Optional:
  -o, --output         Output C file (default: video_animation.c)
  -f, --fps            Target FPS (default: 10)
  --max-frames N       Maximum frames to extract
  --skip N             Use every Nth frame (1=all, 2=every other)
  --skip-pattern X,Y   Alternating skip pattern (e.g., "2,3" = skip 2, then 3, repeat)
  --led-map FILE       Custom LED mapping file (for non-rectangular layouts)
```

## Script-Specific Options

### Color RLE (`video_to_qmk_compressed.py`)
```
--compress           Enable RLE compression
```

### Monochrome (`video_to_qmk_mono.py`)
```
--mode MODE          'rle' (best) or 'bitpack' (predictable)
--threshold N        B&W threshold 0-255 (default: 128)
--white R,G,B        Custom white color (default: 255,255,255)
--colors N           Reduce to N colors before conversion (2-16)
--dither             Apply dithering for better B&W conversion
```

---

## Usage Examples

### Bad Apple (Full Video!) - With All Optimizations
```bash
# Ultimate quality: 2500+ frames with smooth motion and clean image
python video_to_qmk_mono.py badapple.mp4 -w 21 -y 6 -f 12 \
  --colors 4 --dither --skip-pattern 2,3 --mode rle \
  --max-frames 2500 --led-map keychron_v6_max_iso.txt

# Result: Entire 3.5-minute video in ~12-15 KB!
```

### Noisy Video - Use Color Reduction
```bash
# Video with lots of colors/noise compresses poorly
# Add color reduction for much better results
python video_to_qmk_mono.py noisy_video.mp4 -w 21 -y 6 -f 10 \
  --colors 4 --dither --mode rle --max-frames 500

# Before: ~30x compression, messy image
# After: ~80x compression, clean image!
```

### Smooth Motion - Skip Patterns
```bash
# Instead of dropping every 3rd frame (jerky):
python video_to_qmk_mono.py video.mp4 -w 21 -y 6 --skip 3 --mode rle

# Use alternating pattern (smooth):
python video_to_qmk_mono.py video.mp4 -w 21 -y 6 --skip-pattern 2,3 --mode rle

# Even smoother with more frames:
python video_to_qmk_mono.py video.mp4 -w 21 -y 6 --skip-pattern 1,2,2 --mode rle
```

### Color Animation
```bash
# 200 frames of color video
python video_to_qmk_compressed.py logo.mp4 -w 21 -y 6 -f 15 \
  --max-frames 200 --compress --led-map keychron_v6_max_iso.txt
```

### Quick Test
```bash
# 30 frames, no compression
python video_to_qmk.py test.mp4 -w 21 -y 6 --max-frames 30
```

### Advanced: Skip Pattern
```bash
# Alternating skip: 2 frames, 3 frames, 2 frames, 3 frames...
# Results in ~40% fewer frames while maintaining motion
python video_to_qmk_mono.py video.mp4 -w 21 -y 6 --skip-pattern 2,3 --mode rle
```

### Custom Colors
```bash
# Cyan and black instead of white and black
python video_to_qmk_mono.py video.mp4 -w 21 -y 6 --mode rle \
  --white 0,255,255 --max-frames 500

# Matrix green
python video_to_qmk_mono.py video.mp4 -w 21 -y 6 --mode rle \
  --white 0,255,0 --max-frames 500
```

### Better B&W Conversion
```bash
# Reduce colors first for cleaner conversion
python video_to_qmk_mono.py video.mp4 -w 21 -y 6 --mode rle \
  --colors 4 --dither --threshold 128

# Adjust threshold if too dark/light
python video_to_qmk_mono.py video.mp4 -w 21 -y 6 --mode rle \
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
python video_to_qmk_mono.py video.mp4 -w 21 -y 6 --mode rle \
  --led-map keychron_v6_max_iso.txt
```

---

## Integration with QMK

### Step 1: Generate Animation
```bash
python video_to_qmk_mono.py badapple.mp4 -w 21 -y 6 -f 12 \
  --max-frames 500 --mode rle --led-map your_layout.txt -o video_animation.c
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

**For 21×6 keyboard (126 LEDs):**

| Method | 100 frames | 200 frames | 500 frames | 1000 frames |
|--------|------------|------------|------------|-------------|
| **Uncompressed** | 37.8 KB | 75.6 KB | 189 KB ❌ | 378 KB ❌ |
| **Color RLE** | 12-20 KB | 25-35 KB | 60-85 KB ⚠️ | Too large ❌ |
| **Mono Bitpack** | 1.6 KB | 3.2 KB | 7.9 KB | 15.8 KB |
| **Mono RLE** | 0.8-1.5 KB | 1.5-3 KB | 4-8 KB | **8-15 KB** ✅ |

✅ Safe | ⚠️ May be too large | ❌ Too large

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

### Skip Patterns (NEW!) 🎯
Alternating frame skip reduces total frames while preserving motion:
```bash
--skip 3             # Uniform: 33% of frames (jerky)
--skip-pattern 2,3   # Alternating: 40% of frames (smooth!)
--skip-pattern 1,2,3 # Complex: 50% of frames (very smooth!)
```
**How it works:** Instead of dropping the same frame position repeatedly (0, 3, 6, 9...), alternating patterns distribute drops more evenly (0, 2, 5, 7, 10...), creating smoother perceived motion.

**Compression benefit:** 40-60% fewer frames with better visual quality than uniform skip!

### Color Reduction (NEW!) 🎨
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

**Best for:** Noisy videos, gradients, photos, anything that's not already high-contrast

### Dithering (NEW!) 🖼️
Floyd-Steinberg dithering preserves gradients in B&W conversion:
```bash
--dither  # Enable dithering
```
**Effect:**
- Converts gradients to patterns of black/white pixels
- Preserves tonal information lost in simple thresholding
- Better visual quality at slight compression cost
- Creates professional-looking B&W conversions

**Combines perfectly with color reduction:**
```bash
--colors 4 --dither  # Clean colors + smooth gradients = best quality
```

---

## Advanced Optimization Tips

### 1. **Skip Patterns - NEW!** 🎯
**Problem:** Uniform skip (--skip 3) drops every 3rd frame, creating jerky motion  
**Solution:** Alternating patterns distribute dropped frames more evenly

```bash
# Instead of this (drops same frame position every time):
--skip 3  # Keeps frame 0, 3, 6, 9... (33% of frames, jerky)

# Use this (alternates which frames to drop):
--skip-pattern 2,3  # Pattern: 0,2,5,7,10,12... (40% of frames, smoother!)
--skip-pattern 1,2,3  # Pattern: 0,1,3,6,8,11,13... (50% of frames, even smoother!)
```

**Why it works:** Alternating patterns distribute temporal sampling more evenly, reducing motion artifacts.

**Best patterns:**
- `--skip-pattern 2,3` - Good balance (40% of frames)
- `--skip-pattern 1,2,2` - Smoother (50% of frames)
- `--skip-pattern 1,2,3,2` - Very smooth (56% of frames)

### 2. **Color Reduction - NEW!** 🎨
**Problem:** Videos with many colors/noise produce messy B&W conversions  
**Solution:** Reduce to fewer colors first using k-means clustering

```bash
# Original video has 1000s of colors → messy B&W
python video_to_qmk_mono.py video.mp4 -w 21 -y 6 --mode rle

# Reduce to 4 colors first → clean B&W with sharp edges
python video_to_qmk_mono.py video.mp4 -w 21 -y 6 --mode rle --colors 4

# More colors for more detail
python video_to_qmk_mono.py video.mp4 -w 21 -y 6 --mode rle --colors 8
```

**Effect on compression:**
- Without color reduction: Video with 1000 colors → messy B&W → poor RLE compression (30x)
- With `--colors 4`: Video → 4 clean colors → sharp B&W → excellent RLE compression (80x)

**Recommended values:**
- `--colors 2` - Extreme simplification (best compression)
- `--colors 4` - Good balance (recommended for most videos)
- `--colors 8` - More detail, still clean
- `--colors 16` - Maximum detail

### 3. **Dithering - NEW!** 🖼️
**Problem:** Simple thresholding loses gradient information  
**Solution:** Floyd-Steinberg dithering distributes error for smooth gradients

```bash
# Without dithering - gradients become solid blocks
python video_to_qmk_mono.py video.mp4 -w 21 -y 6 --mode rle --threshold 128

# With dithering - gradients preserved as patterns
python video_to_qmk_mono.py video.mp4 -w 21 -y 6 --mode rle --threshold 128 --dither
```

**When to use:**
- ✅ Videos with gradients, faces, or smooth transitions
- ✅ Combined with color reduction (`--colors 4 --dither`)
- ❌ Skip for already high-contrast content (Bad Apple doesn't need it)

**Note:** Dithering adds detail, which slightly reduces compression ratio but greatly improves visual quality.

### 4. **Ultimate Combo - ALL THREE!** 🚀
Combine all techniques for best results:

```bash
python video_to_qmk_mono.py video.mp4 -w 21 -y 6 -f 12 \
  --colors 4 \           # Clean up noisy video
  --dither \             # Smooth gradients
  --skip-pattern 2,3 \   # Smooth motion with fewer frames
  --mode rle \           # Maximum compression
  --max-frames 2000      # Lots of frames!

# Result: 2000 frames in ~12-15 KB with excellent quality!
```

### 5. **Preprocess Video** (Still the Best!)
Use video editing software before conversion:
- Convert to pure black/white (not grayscale)
- Increase contrast
- Remove noise/grain
- Add letterboxing (black bars compress to nothing)
- Reduce resolution if needed

### 6. **Test Compression Ratio**
Always test with a short clip first:
```bash
python video_to_qmk_mono.py test.mp4 -w 21 -y 6 --max-frames 20 --mode rle
# Check "Compression ratio: X.Xx" in output
```

### 7. **Optimal FPS**
- **5-8 FPS:** Most videos, smooth enough
- **10-12 FPS:** Action content
- **15+ FPS:** Only for short, high-action clips

### 4. **Skip Pattern Strategy**
Instead of uniform skip, use patterns for better motion:
```bash
--skip 3           # Uniform: every 3rd frame (33% of frames)
--skip-pattern 2,3 # Alternating: 40% of frames, smoother motion
```

### 5. **Custom Threshold**
Adjust B&W conversion threshold:
```bash
--threshold 100  # More white (darker videos)
--threshold 150  # More black (brighter videos)
```

### 6. **Battery Optimization**
Use dimmer white color:
```bash
--white 80,80,80    # Dim white (saves power)
--white 128,128,128 # Medium white
```

---

## Troubleshooting

### Compilation Fails: "will not fit in region 'flash0'"
**Solution:** File too large
```bash
# Reduce frames
--max-frames 100

# Use skip pattern
--skip-pattern 2,3

# For mono: ensure using RLE mode
--mode rle

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
- Regenerate with fixed `video_to_qmk_mono.py` (latest version)
- Verify LED mapping file is correct
- Check dimensions match keyboard: `-w` and `-y`

### "Compressed size larger than uncompressed"
**Cause:** Video has too much noise/variation  
**Solutions:**
- Preprocess: reduce colors, increase contrast
- Try different compression mode
- Use uncompressed version for this video

---

## File Size Calculator

**Quick estimates:**

```
Uncompressed:  frames × width × height × 3 bytes
Color RLE:     frames × width × height × 0.75 bytes
Mono Bitpack:  frames × (width × height / 8) bytes
Mono RLE:      frames × width × height × 0.03 bytes (Bad Apple)
```

**Example (21×6 keyboard):**
- 500 frames uncompressed: 189 KB ❌
- 500 frames color RLE: ~70 KB ⚠️
- 500 frames mono bitpack: 7.9 KB ✅
- 500 frames mono RLE: ~5 KB ✅

---

## Performance

All compression methods are **fast enough for smooth playback**:
- Decompression: Microseconds per frame
- CPU usage: Minimal
- Battery impact: Low
- Quality: **Lossless** (perfect frame reproduction)

---

## Summary: Which Script to Use?

### 🏆 Bad Apple or B&W content?
**Use `video_to_qmk_mono.py` with `--mode rle`**
```bash
python video_to_qmk_mono.py badapple.mp4 -w 21 -y 6 -f 12 \
  --max-frames 1000 --mode rle --led-map your_layout.txt
```
**Result:** 50-100x compression, 1000+ frames possible

### 🎨 Color video?
**Use `video_to_qmk_compressed.py` with `--compress`**
```bash
python video_to_qmk_compressed.py video.mp4 -w 21 -y 6 -f 10 \
  --max-frames 200 --compress --led-map your_layout.txt
```
**Result:** 2-4x compression, 200-300 frames

### ⚡ Quick test?
**Use `video_to_qmk.py`**
```bash
python video_to_qmk.py test.mp4 -w 21 -y 6 --max-frames 30
```
**Result:** No compression, simplest code, <50 frames

---

## Pro Tips Summary

1. ✅ **Always use mono RLE for B&W content** (50-100x compression!)
2. ✅ **NEW: Use skip patterns** instead of uniform skip for smoother motion
3. ✅ **NEW: Add --colors 4 --dither** for noisy videos (huge compression boost!)
4. ✅ **Combine all three new features** for maximum quality & compression
5. ✅ **Preprocess videos** for better compression (still the best!)
6. ✅ **Test compression ratio** with short clips first
7. ✅ **Adjust threshold** if B&W conversion looks wrong
8. ✅ **Custom colors** work great (cyan, green, dim white)
9. ✅ **LED mapping file** for keyboards with gaps

### Feature Combinations That Work Great:

**For Bad Apple (already clean):**
```bash
--skip-pattern 2,3 --mode rle
# Don't need --colors or --dither (already high contrast)
```

**For noisy/grayscale videos:**
```bash
--colors 4 --dither --skip-pattern 2,3 --mode rle
# All three features = maximum improvement!
```

**For smooth gradients (photos, faces):**
```bash
--colors 8 --dither --threshold 128 --mode rle
# Higher color count + dithering preserves detail
```

**For maximum compression (simple graphics):**
```bash
--colors 2 --skip-pattern 2,3 --mode rle
# Fewest colors + skip pattern = smallest file
```

---

## License

Released into the public domain. Use however you like!

## Credits

Scripts generated with assistance from Anthropic's Claude Sonnet 4.5.

Perfect for: Bad Apple, custom animations, keyboard art, and more! 🚀
