# Video to QMK RGB Animation Converter

Convert video files into RGB LED animations for QMK-powered keyboards!

## Requirements

```bash
pip install opencv-python numpy
```

## Usage

Basic command:
```bash
python video_to_qmk.py input.mp4 -w WIDTH -y HEIGHT
```

### Arguments

- `video` - Input video file (required)
- `-w, --width` - Keyboard width in LEDs (required)
- `-y, --height` - Keyboard height in LEDs (required)
- `-o, --output` - Output C file name (default: video_animation.c)
- `-f, --fps` - Target frames per second (default: 10)
- `--skip` - Use every Nth frame (default: 1, no skip)
- `--max-frames` - Maximum frames to extract (default: all)

### Examples

**Convert a video for a 15x5 keyboard at 10 FPS:**
```bash
python video_to_qmk.py badapple.mp4 -w 15 -y 5 -f 10
```

**Use every 3rd frame to reduce file size:**
```bash
python video_to_qmk.py video.mp4 -w 15 -y 5 --skip 3
```

**Limit to first 100 frames (10 seconds at 10 FPS):**
```bash
python video_to_qmk.py video.mp4 -w 15 -y 5 --max-frames 100
```

**High quality, short clip:**
```bash
python video_to_qmk.py short_clip.mp4 -w 20 -y 6 -f 15 --max-frames 50
```

## Integration with QMK

### Step 1: Copy the generated file

Copy the generated `.c` file to your QMK keyboard directory:
```
keyboards/your_keyboard/keymaps/your_keymap/
```

### Step 2: Create a header file

Create `video_animation.h` in the same directory:
```c
#pragma once

void video_animation_start(void);
void video_animation_stop(void);
void video_animation_toggle(void);
bool video_animation_update(uint8_t led_min, uint8_t led_max);
```

### Step 3: Update keymap.c

Add to your `keymap.c`:

```c
#include "video_animation.h"

// Define custom keycode
enum custom_keycodes {
    VIDEO_TOGGLE = SAFE_RANGE,
};

// Handle the keycode
bool process_record_user(uint16_t keycode, keyrecord_t *record) {
    switch (keycode) {
        case VIDEO_TOGGLE:
            if (record->event.pressed) {
                video_animation_toggle();
            }
            return false;
    }
    return true;
}

// Update the RGB matrix
bool rgb_matrix_indicators_advanced_user(uint8_t led_min, uint8_t led_max) {
    video_animation_update(led_min, led_max);
    return false;
}
```

### Step 4: Update rules.mk

Add to your `rules.mk`:
```makefile
RGB_MATRIX_ENABLE = yes
SRC += video_animation.c
```

### Step 5: Map a key

In your keymap, add `VIDEO_TOGGLE` to a key to control the animation.

## Tips for Best Results

### File Size Management

The output file size depends on:
- Number of frames
- Keyboard dimensions (width × height)
- Frame skip

**Formula:** `Size (KB) ≈ frames × width × height × 3 / 1024`

For a 15x5 keyboard with 100 frames: ~22 KB

### Reducing File Size

1. **Use frame skip:** `--skip 2` or `--skip 3` to use fewer frames
2. **Limit frames:** `--max-frames 100` for shorter clips
3. **Lower FPS:** Use `-f 5` or `-f 8` instead of 10+
4. **Choose shorter videos:** A few seconds at 10 FPS is often enough

### Content Recommendations

Best results with:
- **High contrast videos** (black and white works great!)
- **Simple graphics** (complex details get lost at low resolution)
- **Intentional animations** (not random footage)
- **Bad Apple!!** (classic choice for RGB matrix displays)

### Keyboard Layout Mapping

The generated code assumes LEDs are mapped in a row-major order:
```
LED 0,  LED 1,  LED 2,  ... LED (width-1)
LED width, LED width+1, ...
```

If your keyboard has a different layout, you'll need to modify the LED index calculation in the `video_animation_update()` function:

```c
// Change this line:
uint8_t led_index = y * VIDEO_WIDTH + x;

// To match your keyboard's layout
// Example for column-major:
uint8_t led_index = x * VIDEO_HEIGHT + y;
```

## Advanced: Custom LED Mapping

For keyboards with non-rectangular layouts or custom LED orders, create a mapping array:

```c
// Define your custom LED positions
static const uint8_t PROGMEM led_map[VIDEO_HEIGHT][VIDEO_WIDTH] = {
    {0, 1, 2, 3, 4, 5},
    {6, 7, 8, 9, 10, 11},
    // ... etc
};

// Then use it:
uint8_t led_index = pgm_read_byte(&led_map[y][x]);
```

## Troubleshooting

**"Output file will be very large" warning:**
- Use `--skip` to reduce frames
- Use `--max-frames` to limit duration
- Lower the `-f` (FPS) value

**Video doesn't play correctly:**
- Check your LED mapping matches your keyboard
- Verify RGB_MATRIX_ENABLE is set in rules.mk
- Make sure you're calling the update function

**Compilation fails - file too large:**
- Reduce frame count with `--max-frames`
- Use higher `--skip` value
- Your MCU might not have enough storage for large animations

**Animation is choppy:**
- Try lower FPS (`-f 5` or `-f 8`)
- Your keyboard's processing speed may limit smooth playback

## Examples

### Bad Apple (Classic!)
```bash
# Extract 200 frames at 10 FPS for a 15x5 keyboard
python video_to_qmk.py badapple.mp4 -w 15 -y 5 -f 10 --max-frames 200
```

### Short Logo Animation
```bash
# High quality, short 3-second loop
python video_to_qmk.py logo.mp4 -w 20 -y 6 -f 15 --max-frames 45
```

### Long Music Video (Optimized)
```bash
# Use every 5th frame to keep file size reasonable
python video_to_qmk.py music.mp4 -w 15 -y 5 -f 10 --skip 5 --max-frames 500
```

## License

This script is released into the public domain. Use it however you like!
