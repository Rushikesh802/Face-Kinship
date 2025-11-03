# Visual Validation Guide

## What You'll See

### 1. Upload Image
```
┌─────────────────────────────┐
│   📤 Upload Face 1          │
│   Drag & drop or click      │
│   Supports: JPG, PNG (5MB)  │
└─────────────────────────────┘
```

### 2. Validating (Immediate)
```
┌─────────────────────────────┐
│   [Your Image]              │
│                             │
│   🔄 Validating face...     │ ← Blue border, spinner
└─────────────────────────────┘
```

### 3a. Valid Face ✅
```
┌─────────────────────────────┐
│   [Your Image]              │
│                             │
│   ✓ Face detected ✓         │ ← GREEN border
└─────────────────────────────┘

[Analyze Kinship] ← Button ENABLED
```

### 3b. Invalid - No Face ❌
```
┌─────────────────────────────┐
│   [Car/Screenshot Image]    │
│                             │
│   ⚠ No face detected...     │ ← RED border
└─────────────────────────────┘

[Analyze Kinship] ← Button DISABLED
```

### 3c. Invalid - Multiple Faces ❌
```
┌─────────────────────────────┐
│   [Group Photo]             │
│                             │
│   ⚠ Multiple faces...       │ ← RED border
└─────────────────────────────┘

[Analyze Kinship] ← Button DISABLED
```

---

## Color Coding

| Status | Border Color | Message Color | Icon |
|--------|-------------|---------------|------|
| Validating | 🔵 Blue (pulsing) | Blue | 🔄 Spinner |
| Valid | 🟢 Green | Green | ✓ Check |
| Invalid | 🔴 Red | Red | ⚠ Alert |

---

## Button States

### Enabled (Both Images Valid)
```
┌───────────────────────────────┐
│  🔍 Analyze Kinship           │ ← Clickable, gradient
└───────────────────────────────┘
```

### Disabled (Validating)
```
┌───────────────────────────────┐
│  🔄 Validating faces...       │ ← Grayed out, spinner
└───────────────────────────────┘
```

### Disabled (Invalid Image)
```
┌───────────────────────────────┐
│  🔍 Analyze Kinship           │ ← Grayed out, not clickable
└───────────────────────────────┘
```

---

## Example Workflow

### ✅ Success Flow
```
1. User uploads face1.jpg
   → "Validating face..." (blue)
   → "Face detected ✓" (green)

2. User uploads face2.jpg
   → "Validating face..." (blue)
   → "Face detected ✓" (green)

3. Analyze button enabled
   → User clicks "Analyze Kinship"
   → Results shown
```

### ❌ Error Flow (Immediate Feedback)
```
1. User uploads car.jpg
   → "Validating face..." (blue)
   → "No face detected..." (red) ← IMMEDIATE!

2. User sees red border and error
   → Removes car.jpg
   → Uploads face1.jpg instead
   → "Face detected ✓" (green)

3. Continues with valid images
```

---

## Error Messages Reference

| Error Type | Message Shown |
|-----------|---------------|
| No face | "No face detected in Image. Please upload a clear face image." |
| Multiple faces | "Multiple faces detected in Image. Please upload an image with only one face." |
| Low quality | "Face quality too low in Image. Please upload a clearer image." |
| Wrong file type | "Please upload a valid image file" |
| File too large | "Image size should be less than 5MB" |
| Network error | "Validation failed. Please try again." |

---

## Quick Tips

### ✅ DO:
- Upload clear, well-lit face photos
- Use single-person photos
- JPG, PNG, JPEG formats
- Keep files under 5MB

### ❌ DON'T:
- Upload group photos (multiple faces)
- Upload cars, screenshots, text
- Upload blurry or dark images
- Upload files over 5MB

---

## Troubleshooting

### "No face detected" but it's a face photo?
- Image might be too small (< 64x64)
- Face might be too dark or blurry
- Try a clearer, better-lit photo

### "Multiple faces detected" but only one person?
- MediaPipe might detect reflections
- Try a different photo
- Ensure only one clear face visible

### Validation taking too long?
- Check internet connection
- Ensure backend server is running
- Try refreshing the page

---

## Technical Notes

- Validation happens **client-side** (calls backend API)
- Uses **MediaPipe Face Detection**
- Validation time: **< 1 second**
- No data stored during validation
- Same validation used in final analysis
