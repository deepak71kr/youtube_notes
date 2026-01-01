# 🎓 LectureLens AI

**Turn video lectures into clean, distraction-free PDF notes.**

LectureLens AI automatically detects slide transitions in lecture
videos, uses AI to erase the presenter, and stitches the background
together to create unobstructed PDF notes.

## ✨ Features

- **Smart Detection:** Auto-detects slide changes using SSIM.
- **AI Eraser:** Removes the presenter using DeepLabV3.
- **Auto-Healing:** Fills \"holes\" where the person stood by merging
  frames.
- **PDF Output:** Compiles clean slides into a ready-to-study PDF.

## 🚀 Quick Start

### 1. Install Dependencies

pip install opencv-python numpy torch torchvision scikit-image pillow

### 2. Required Files

Ensure these three files are in the same folder:

- `main.py` (Run this one)
- `slide_change_detector.py`
- `create_composite_image.py`

### 3. Run

python main.py

## 🎮 How to Use

1. **Run the script** and select your video file (`.mp4`, `.avi`, etc.)
   via the popup.
2. **Adjust settings** if needed (defaults are usually fine):
   - **Min Slide Duration:** Increase (e.g., 40s) if you see
     duplicate slides.
   - **Sampling Interval:** Decrease (e.g., 1s) for better accuracy.
3. **Wait for processing.** The first run downloads the AI model
   (\~100MB).
4. **Enter a name** for your output PDF when prompted.

**Raw Picture
<img width="1156" height="576" alt="frame_bgonly_0" src="https://github.com/S-A-U-R-A-V-48/slidelens.ai/blob/main/original_img2.jpeg" />
**Person detected an removed at intervals until we found the whole complete picture
<img width="1156" height="576" alt="frame_bgonly_0" src="https://github.com/user-attachments/assets/ce68fd01-113b-41db-918d-2d1650faa38f" />
<img width="1156" height="576" alt="frame_bgonly_0" src="https://github.com/S-A-U-R-A-V-48/slidelens.ai/blob/main/frame_bgonly_30.png" />
<img width="1156" height="576" alt="frame_bgonly_0" src="https://github.com/S-A-U-R-A-V-48/slidelens.ai/blob/main/frame_bgonly_60.png" />
<img width="1156" height="576" alt="frame_bgonly_0" src="https://github.com/S-A-U-R-A-V-48/slidelens.ai/blob/main/frame_bgonly_90.png" />
<img width="1156" height="576" alt="frame_bgonly_0" src="https://github.com/S-A-U-R-A-V-48/slidelens.ai/blob/main/frame_bgonly_120.png" />
<img width="1156" height="576" alt="frame_bgonly_0" src="https://github.com/S-A-U-R-A-V-48/slidelens.ai/blob/main/frame_bgonly_180.png" />
<img width="1156" height="576" alt="frame_bgonly_0" src="https://github.com/S-A-U-R-A-V-48/slidelens.ai/blob/main/frame_bgonly_270.png" />


**final Composition after merging

<img width="1156" height="576" alt="frame_bgonly_0" src="https://github.com/S-A-U-R-A-V-48/slidelens.ai/blob/main/composite_image.png" />
## ⚙️ Troubleshooting

- **Process too slow?** Increase *Sampling Interval* or *Frame Step*.
- **Presenter ghosting?** Increase *Person Mask Dilation* (e.g., to 20
  or 25).
- **No slides found?** Your *Min Slide Duration* might be longer than
  the actual slides. Lower it.
