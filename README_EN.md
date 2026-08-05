# Lens Model (棱镜模型)

> **Project Goal:** Thoroughly analyze and clarify the dynamical system of a specific single network model, and study model parameter equivalence classes from two distinct perspectives: measure-theoretic information compression and dimensionality reduction.

A **conditional diffusion model** for predicting 3D prism optical path mappings. Given any triangular mesh model, the system predicts the exit light displacement field (2D→2D mapping) after parallel light passes through a prism, applicable to inverse optical design, optical simulation visualization, and related scenarios.

## Principle

```
Parallel light (z=+∞, direction 0,0,-1)
   ↓ ↓ ↓ ↓ ↓
┌─────────────────┐  ← Entrance plane
│   Air (n=1.0)    │
│  ┌───────────┐  │  ← Glass prism (n=1.43)
│  │  Snell refr. │  │
│  │  Internal    │  │
│  └───────────┘  │
│   Air (n=1.0)    │
└─────────────────┘  ← Exit recording plane (z=-0.5)
```

- **Input:** Layered Depth Encoding of 3D mesh (up to 8 penetration surfaces + cumulative thickness + occupancy mask)
- **Output:** Normalized 2D displacement field `(H, W, 2)` = exit coords − entrance coords
- **Model:** U-Net conditional diffusion (DDPM) with DDIM accelerated sampling and guided completion

## Directory Structure

```
LensModel/
├── config.py              # Shared configuration (paths, model/training/physics params)
├── requirements.txt       # Python dependencies
│
├── train_diffusion.py     # Diffusion model training
├── inference_diffusion.py # Single-shape inference (DDIM sampling)
├── evaluate_diffusion.py  # Model evaluation & metrics
├── visualize_diffusion.py # Result visualization
│
├── generate_shapes.py     # Batch 3D shape generation (PLY format)
├── shape2map.py           # Optical path mapping (Snell refraction ray tracing)
├── download_models.py     # Open-source model downloader
├── organize_modelnet.py   # ModelNet dataset organizer
│
├── shapes/                # 3D models (categories 01-10, PLY format)
│   ├── 01_Platonic/       # Tetrahedron, cube, octahedron, dodecahedron, icosahedron
│   ├── 02_Convex/         # Truncated tetrahedron, cuboctahedron, etc.
│   ├── 03_Spheres_Cones/  # Sphere, cone, pyramids
│   ├── 04_Multi-genus/    # Torus (genus 1/2/3), Klein bottle, Möbius strip
│   ├── 05_Animals/        # Rabbit, cat, bird, fish, deer, turtle
│   ├── 06_Plants/         # Broadleaf tree, pine, cactus, flower, mushroom
│   ├── 07_Tools/          # Pliers, hammer, axe, screwdriver, wrench, handsaw
│   ├── 08_Architecture/   # House, castle, tower, arch bridge, pyramid
│   ├── 09_Characters/     # Standing, walking, running, sitting, waving
│   └── 10_Misc/           # Heart, boat, airplane, car, chair, table, etc.
│
├── maps/                  # Optical path maps (matching shapes, .npz + .txt)
├── figures/               # Visualization comparison charts (PNG)
├── results/               # Model prediction outputs (.npz)
├── models/                # Trained weights + data split
│   ├── best_model.pt
│   ├── final_model.pt
│   └── data_split.json
│
└── README.md
```

## Environment Setup

### 1. Clone the Repository

```bash
git clone git@github.com:Darkhe55/LensModel.git
cd LensModel
```

### 2. Create Python Environment

Python 3.10+ recommended. Use Conda or venv:

```bash
# Conda
conda create -n lensmodel python=3.10
conda activate lensmodel

# Or venv
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies:
- `torch >= 2.0.0` — Deep learning framework
- `numpy >= 1.24.0` — Numerical computing
- `scipy >= 1.10.0` — Scientific computing
- `matplotlib >= 3.7.0` — Visualization

Optional:
- `trimesh >= 3.20.0` — 3D model loading acceleration

### 4. GPU Support (Recommended)

Training is recommended with CUDA GPU:

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Step 1: Generate 3D Shapes (Optional)

If `shapes/` is empty or needs regeneration:

```bash
python generate_shapes.py
```

Generates PLY model files across 10 categories under `shapes/`.

### Step 2: Compute Optical Path Maps

Ray-trace each 3D model using Snell's law to generate training data:

```bash
# Process all shapes (128×128 resolution)
python shape2map.py

# Higher resolution
python shape2map.py --resolution 256

# Specific folder only
python shape2map.py --folder "01_Platonic"

# Single model
python shape2map.py --model "shapes/01_Platonic/tetrahedron.ply"

# Custom refractive index (default 1.43 for glass)
python shape2map.py --n_refract 1.52

# Skip existing maps (incremental)
python shape2map.py --skip_existing

# Limit divergence distance and path length (complex models)
python shape2map.py --max_diverge 30 --max_path 100

# Multi-threaded
python shape2map.py --workers 8
```

Generated `.npz` file contents:
| Field | Description |
|------|------|
| `input_grid` | Entrance ray coordinates `(res, res, 2)` |
| `output_grid` | Exit ray coordinates `(res, res, 2)` |
| `valid_mask` | Valid ray boolean mask |
| `inverse_map` | Inverse mapping: exit → entrance |
| `bbox_min/max` | Model bounding box |

### Step 3: Train the Diffusion Model

```bash
# Full pipeline (train + sample + evaluate)
python train_diffusion.py

# Train only
python train_diffusion.py --mode train

# Custom parameters
python train_diffusion.py --mode train --epochs 2000 --batch_size 4

# Adjust KL divergence weight (latent space regularization)
python train_diffusion.py --mode train --kl_weight 0.01   # Stronger regularization
python train_diffusion.py --mode train --kl_weight 0.0    # Pure MSE, no KL

# Sample only (generate predictions from trained model)
python train_diffusion.py --mode sample

# Evaluate only
python train_diffusion.py --mode evaluate
```

Weights are saved to `models/best_model.pt` (best validation) and `models/final_model.pt` (final).

### Step 4: Inference

Predict optical path mapping for any 3D model:

```bash
# Basic inference
python inference_diffusion.py shapes/01_Platonic/tetrahedron.ply

# Specify output path
python inference_diffusion.py shapes/05_Animals/rabbit.ply --output results/rabbit_pred.npz

# DDIM acceleration (default 50 steps, 10-20× faster than DDPM)
python inference_diffusion.py shapes/04_Multi-genus/torus_genus1.ply --ddim_steps 50
```

### Step 5: Evaluation & Visualization

```bash
# Evaluate model on test set
python evaluate_diffusion.py

# Generate comparison plots, error histograms, prediction field visualizations
python visualize_diffusion.py
```

Results saved to `figures/` (PNG charts) and `results/` (NPZ data).

## Key Parameters Reference

Centralized in `config.py`:

| Parameter | Default | Description |
|------|--------|------|
| `IMG_SIZE` | 64 | Working resolution |
| `K_LAYERS` | 8 | Layered depth encoding layers |
| `HIDDEN_DIM` | 64 | U-Net base channels |
| `T_STEPS` | 1000 | Total diffusion steps |
| `LEARNING_RATE` | 2e-4 | Learning rate |
| `DEFAULT_EPOCHS` | 1500 | Default training epochs |
| `GUIDANCE_STRENGTH` | 0.7 | Guided sampling strength |
| `N_REFRACT` | 1.43 | Glass refractive index |
| `TRAIN_RATIO` | 0.7 | Training set ratio |
| `SEED` | 42 | Random seed |

## Model Architecture

- **Encoder:** Layered Depth Encoding — records up to 8 surface penetration depths along −z + cumulative thickness + occupancy mask
- **Diffusion Model:** U-Net conditional DDPM, conditioned on depth encoding
- **Sampler:** Supports both DDPM and DDIM (deterministic acceleration)
- **Missing Data Handling:**
  - Confidence-weighted loss — full weight for valid regions, low weight for missing regions
  - Random mask augmentation — additionally mask some valid points during training
  - Guided sampling — known regions pulled toward ground truth

## FAQ

### Q: CUDA out of memory
Reduce batch size:
```bash
python train_diffusion.py --mode train --batch_size 2
```

### Q: Model loading error
Ensure `shapes` and `maps` directory structures match. Use incremental generation:
```bash
python shape2map.py --skip_existing
```

### Q: Poor optical path quality
- Increase resolution: `shape2map.py --resolution 256`
- Increase training epochs: `train_diffusion.py --epochs 3000`
- Adjust KL weight: `--kl_weight 0.005`

## License

This project is for academic research purposes only.
