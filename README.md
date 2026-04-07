# AI Image Detector (Synthetic Image Detection)

This project uses a **ResNet18** deep learning model to detect whether an image is AI-generated (FAKE) or a real photograph (REAL).

## 📁 Project Structure

- **Backend/**: FastAPI server that handles image uploads and runs predictions.
- **frontend/**: Modern React (Vite) UI for interactive testing.
- **DATA/**: Placeholder for image datasets (`train` and `test`).
- **Train.py**: Script to train the ResNet18 model locally.

---

## 🚀 Getting Started

### 1️⃣ Backend Setup
1. Navigate to the `Backend` directory.
2. Install Python dependencies:
   ```bash
   pip install -r Requirements.txt
   ```
3. Start the API server:
   ```bash
   uvicorn main:app --reload
   ```

### 2️⃣ Frontend Setup
1. Navigate to the `frontend` directory.
2. Install Node.js dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

### 3️⃣ Training the Model (Optional)
If you wish to retrain the model, ensure your dataset is placed in the `DATA/` directory:
- `DATA/train/FAKE/` (synthetic images)
- `DATA/train/REAL/` (real photos)
- `DATA/test/FAKE/`
- `DATA/test/REAL/`

Then run:
```bash
python Train.py
```
This will generate a `model.pth` file. Copy this file into the `Backend/` folder to use it for live predictions.

---

## ✅ Recent Improvements
- **Backend Performance**: Enabled CUDA support if available for faster GPU inference.
- **UI Data Binding**: Updated API to return full class scores, allowing the frontend to display detailed probability breakdowns.
- **Bug Fix**: Fixed `Train.py` hardcoded Kaggle paths and data leak issues.
- **Dependencies**: Populated `Requirements.txt` with necessary libraries.
