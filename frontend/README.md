# Deepfake Detection & Explainable Digital Forensics Dashboard

**Project:** Multimodal Deepfake Detection and Explainable Digital Forensics System for Images and Videos  
**Academic Context:** Final-Year Computer Science & Engineering Capstone Project  
**Tech Stack:** React 18, Vite 5, Tailwind CSS, Lucide Icons, FastAPI (Backend)

---

## 1. Overview

The Deepfake Forensics Dashboard is a modern, high-performance web interface designed for digital forensics investigators and academic examiners. It interfaces with an **EfficientNet-B4** deep learning backend to authenticate facial photographs and video sequences, delivering spatial explainability via **Grad-CAM** saliency maps and cryptographically anchored examination reports.

### Key Capabilities
* **System Dashboard:** Live telemetry showing total scans, high-risk detection ratios, and backend heartbeat status.
* **Single-Image Face Forgery Detection:** File drag-and-drop, instant client-side SHA-256 calculation, and calibrated continuous logit scoring ($P(\text{Fake}) = \sigma(s)$).
* **Video Deepfake Analysis:** Uniform 16-frame temporal sampling, OpenCV YuNet face tracking with IoU continuity, peak anomaly keyframe isolation ($j^* = \arg\max s_j$), and mean logit aggregation.
* **Grad-CAM Saliency Inspector:** 3-view comparative inspector (Original Face Crop, Raw Normalized Heatmap, and Blended Jet Overlay) targeting `model.conv_head` (1,792 feature maps).
* **Forensic Examination Sheets:** Formal evidence sheets recording Evidence ID, SHA-256 hash, timestamps, verdict, and limitations, with one-click JSON and printable HTML export.
* **Analysis History:** Session evidence audit log with search, verdict filters, and explicit labeling of baseline benchmark records (`[DEMO]`).

---

## 2. Prerequisites

* **Node.js:** v18.0.0 or newer (v24.19.0 verified).
* **Package Manager:** `pnpm` (v11.x verified) or `npm`.
* **Backend:** Python 3.11 with FastAPI server running on `http://localhost:8000`.

---

## 3. Installation & Local Setup

### Step 1: Install Frontend Dependencies
From the `frontend/` directory:
```bash
# Using pnpm (recommended)
pnpm install

# Or using npm
npm install
```

### Step 2: Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Default configuration:
```env
VITE_API_BASE_URL=http://localhost:8000
```

---

## 4. Running the Complete System Locally

### Step 1: Start the FastAPI Backend
Open a terminal in the root repository directory (`DEEPFAKE`):
```powershell
.\.venv\Scripts\python.exe -m uvicorn src.backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Verify the backend is live by opening: `http://localhost:8000/api/health`

### Step 2: Start the Vite Frontend Development Server
Open a second terminal in the `frontend/` directory:
```powershell
pnpm run dev
# Or: npm run dev
```
Open your browser to: `http://localhost:5173`

---

## 5. Building for Production

To create an optimized production build:
```bash
pnpm run build
# Or: npm run build
```
Build output will be bundled into `frontend/dist/`.

To preview the production bundle locally:
```bash
pnpm run preview
# Or: npm run preview
```

---

## 6. Evidentiary & Ethical Disclaimer

Model classification probabilities represent statistical pattern matching and do NOT constitute definitive legal proof of manipulation or personal wrongdoing. Saliency maps highlight spatial receptive fields influencing neural activations, not certified physical tampering boundaries. Formal legal custody must be established through independent evidentiary documentation.
