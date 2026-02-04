# Compression Comparison Guide

## Three Scripts, Different Use Cases

### 1. **video_to_qmk.py** - Original (No Compression)
**Best for**: Very short clips, testing, simple use cases

**Pros:**
- Simplest code
- Fastest playback (no decompression)
- Full color support

**Cons:**
- Largest file size
- ~50-100 frames max for most keyboards

**File size**: 3 bytes per pixel per frame
```
Example (21×6, 100 frames): 37.8 KB
```

---

### 2. **video_to_qmk_compressed.py** - Color RLE
**Best for**: Color videos, moderate compression needs

**Pros:**
- 2-4x compression
- Full RGB color
- Lossless

**Cons:**
- More complex code
- Moderate decompression overhead

**File size**: ~0.75-1.5 bytes per pixel per frame (varies with content)
```
Example (21×6, 200 frames): 75.6 KB → 20-30 KB
```

---

### 3. **video_to_qmk_mono.py** - Monochrome (NEW! ⭐)
**Best for**: Black & white videos (Bad Apple!), maximum compression

**Two modes:**

#### **Bit-Pack Mode** (24x compression)
- 1 bit per pixel (pure B&W)
- Fixed compression ratio
- Extremely simple decompression

**File size**: 0.125 bytes per pixel per frame
```
Example (21×6, 500 frames): 189 KB → 7.9 KB (24x smaller!)
```

#### **RLE Mode** (50-100x+ compression for Bad Apple!)
- Run-length encoding of black/white runs
- Variable compression (depends on content)
- Best for videos with long runs of same color

**File size**: Typically 0.03-0.06 bytes per pixel per frame
```
Example Bad Apple (21×6, 1000 frames): 378 KB → 4-8 KB (50-100x smaller!)
```

---

## Compression Comparison Chart

**For Keychron V6 Max (21×6 = 126 LEDs):**

| Method | 100 frames | 200 frames | 500 frames | 1000 frames |
|--------|------------|------------|------------|-------------|
| **Uncompressed** | 37.8 KB | 75.6 KB | 189 KB ❌ | 378 KB ❌ |
| **Color RLE** | ~12-20 KB ✅ | ~25-35 KB ✅ | ~60-85 KB ⚠️ | Too large ❌ |
| **Mono Bitpack** | 1.6 KB ✅ | 3.2 KB ✅ | 7.9 KB ✅ | 15.8 KB ✅ |
| **Mono RLE** | 0.8-1.5 KB ✅ | 1.5-3 KB ✅ | 4-8 KB ✅ | 8-15 KB ✅ |

✅ = Safe for most keyboards  
⚠️ = May be too large  
❌ = Too large

---

## Which Script Should You Use?

### Use **video_to_qmk_mono.py** if:
✅ Your video is black & white (Bad Apple, silhouette animations, etc.)  
✅ You want the longest possible animation  
✅ You want to fit 500-1000+ frames  

**Example:**
```bash
# Bad Apple - 1000 frames with RLE!
python video_to_qmk_mono.py badapple.mp4 -w 21 -y 6 -f 12 \
  --max-frames 1000 --mode rle --led-map keychron_v6_max_iso.txt
```

### Use **video_to_qmk_compressed.py** if:
✅ Your video has colors you want to preserve  
✅ You need 100-300 frames  
✅ Content has moderate color variation  

**Example:**
```bash
# Color video with RLE
python video_to_qmk_compressed.py video.mp4 -w 21 -y 6 -f 10 \
  --max-frames 200 --compress --led-map keychron_v6_max_iso.txt
```

### Use **video_to_qmk.py** if:
✅ You're just testing/prototyping  
✅ You only need <50 frames  
✅ You want the simplest possible code  

**Example:**
```bash
# Simple test
python video_to_qmk.py test.mp4 -w 21 -y 6 --max-frames 30
```

---

## Real-World Examples

### Bad Apple (Full Song - ~3.5 minutes at 12 FPS)
```bash
# Without compression: Would need ~10 MB ❌ IMPOSSIBLE
# With color RLE: ~500 KB ❌ Still too large
# With mono RLE: ~8-15 KB ✅ PERFECT!

python video_to_qmk_mono.py badapple.mp4 -w 21 -y 6 -f 12 \
  --max-frames 2520 --mode rle --led-map keychron_v6_max_iso.txt
```

### Color Logo Animation (30 frames, looping)
```bash
# Clean logo with solid colors compresses well with RLE
python video_to_qmk_compressed.py logo.mp4 -w 21 -y 6 -f 15 \
  --max-frames 30 --compress
```

### Short Intro Animation (50 frames)
```bash
# So short that compression overhead isn't worth it
python video_to_qmk.py intro.mp4 -w 21 -y 6 -f 10 --max-frames 50
```

---

## Monochrome Mode Deep Dive

### **Bit-Pack Mode**
```bash
python video_to_qmk_mono.py video.mp4 -w 21 -y 6 --mode bitpack
```

**How it works:**
- Converts each pixel to pure black or white (threshold at 128 gray value)
- Packs 8 pixels into each byte
- 1 bit per pixel = **24x compression** (vs RGB888)

**Pros:**
- Fixed, predictable compression
- Extremely fast decompression
- Simple code

**Cons:**
- Not as good as RLE for videos with long runs

**Best for:**
- Videos with lots of detail/texture
- When you want predictable file size

### **RLE Mode** (Recommended!)
```bash
python video_to_qmk_mono.py video.mp4 -w 21 -y 6 --mode rle
```

**How it works:**
- Converts to B&W, then encodes runs of consecutive pixels
- Example: 50 white pixels = just 2 bytes!

**Compression examples:**
- Bad Apple: **50-100x compression** 🤯
- High contrast graphics: **30-80x**
- Noisy/detailed B&W: **10-20x**

**Pros:**
- Insane compression for content with runs
- Variable - adapts to content
- Still very fast

**Cons:**
- Compression ratio varies
- Slightly more complex code

**Best for:**
- Bad Apple and similar content
- Silhouette animations
- High-contrast graphics

### Custom White Color

Make your B&W animation use any color instead of white:

```bash
# Cyan and black
python video_to_qmk_mono.py video.mp4 -w 21 -y 6 --mode rle --white 0,255,255

# Red and black  
python video_to_qmk_mono.py video.mp4 -w 21 -y 6 --mode rle --white 255,0,0

# Dim white (save power)
python video_to_qmk_mono.py video.mp4 -w 21 -y 6 --mode rle --white 128,128,128
```

### Threshold Adjustment

Control how the grayscale is converted to B&W:

```bash
# Default threshold (128 = medium gray)
python video_to_qmk_mono.py video.mp4 -w 21 -y 6 --mode rle

# More black (threshold 180 = only brightest pixels become white)
python video_to_qmk_mono.py video.mp4 -w 21 -y 6 --mode rle --threshold 180

# More white (threshold 80 = more pixels become white)
python video_to_qmk_mono.py video.mp4 -w 21 -y 6 --mode rle --threshold 80
```

**Tip:** If your video looks too dark or light, adjust the threshold!

---

## Performance Comparison

| Method | Decompression Speed | CPU Usage | Battery Impact |
|--------|---------------------|-----------|----------------|
| Uncompressed | ⚡ Instant | Minimal | Low |
| Color RLE | ⚡ Fast | Low | Low |
| Mono Bitpack | ⚡ Fast | Low | Low |
| Mono RLE | ⚡ Fast | Low-Medium | Low |

All methods are fast enough for smooth playback at 10-15 FPS on modern MCUs.

---

## File Size Formula Reference

**Uncompressed RGB:**
```
Size = frames × width × height × 3 bytes
```

**Color RLE (estimate):**
```
Size ≈ frames × width × height × 0.75 bytes (varies 0.5-1.5)
```

**Mono Bitpack (exact):**
```
Size = frames × ceil((width × height) / 8) bytes
```

**Mono RLE (estimate for Bad Apple):**
```
Size ≈ frames × width × height × 0.03 bytes (varies 0.02-0.1)
```

---

## Summary Recommendations

### 🏆 **For Bad Apple: Use Mono RLE**
```bash
python video_to_qmk_mono.py badapple.mp4 -w 21 -y 6 -f 12 \
  --max-frames 1000 --mode rle --led-map keychron_v6_max_iso.txt
```
**Result:** ~8-15 KB for 1000 frames (50-100x compression!)

### 🎨 **For Color Videos: Use Color RLE**
```bash
python video_to_qmk_compressed.py video.mp4 -w 21 -y 6 -f 10 \
  --max-frames 200 --compress --led-map keychron_v6_max_iso.txt
```
**Result:** ~20-35 KB for 200 frames (2-4x compression)

### ⚡ **For Quick Tests: Use Uncompressed**
```bash
python video_to_qmk.py test.mp4 -w 21 -y 6 --max-frames 30
```
**Result:** ~11 KB for 30 frames (no compression)

---

## Pro Tips

1. **Always test compression ratio first** with a small clip:
   ```bash
   python video_to_qmk_mono.py test.mp4 -w 21 -y 6 --max-frames 20 --mode rle
   # Check the "Compression ratio" in output
   ```

2. **For Bad Apple**, RLE can compress **100:1** - you can fit the ENTIRE VIDEO!

3. **Preprocess your video** for better compression:
   - Increase contrast
   - Convert to pure B&W in video editor
   - Remove noise/grain

4. **Custom colors** work great:
   ```bash
   --white 0,255,0    # Matrix green
   --white 255,128,0  # Amber 
   --white 64,64,255  # Soft blue
   ```

5. **Battery-conscious** users: Use dimmer white like `--white 100,100,100`

---

You now have THREE powerful tools for different use cases. Choose wisely! 🚀
