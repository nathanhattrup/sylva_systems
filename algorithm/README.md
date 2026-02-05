# Grounded-SAM-2 Local Repro (WSL Ubuntu) — GroundingDINO → SAM2 “trees only”

This README documents the exact steps used in this session to set up and run **Grounded-SAM-2** locally (GroundingDINO box proposals → SAM2 masks), based on the official repository:

Source: https://github.com/IDEA-Research/Grounded-SAM-2

---

## What this produces

Given an image and a text prompt like `tree.`:

1. **GroundingDINO** proposes bounding boxes that match the text prompt.
2. **SAM2** converts those boxes into segmentation masks.
3. Outputs include:
   - Annotated image with boxes
   - Annotated image with masks
   - Optional JSON (RLE masks) for downstream use

**Important prompt rule:** text prompts must be **lowercase** and end with a **period** (e.g., `tree.` or `tree canopy.`).

---

## Environment used (this session)

- Ubuntu 22.04 under **WSL** (CUDA toolkit installed via NVIDIA WSL repo)
- CUDA Toolkit: **11.8**
- PyTorch: **cu118** build (Torch CUDA reports 11.8)
- Repo: `IDEA-Research/Grounded-SAM-2`

---

## 1) Clone the repo

```bash
mkdir -p SGIL
cd SGIL
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd Grounded-SAM-2
```

---

## 2) Download checkpoints (SAM2 + GroundingDINO)

### SAM2 checkpoints

```bash
cd checkpoints
bash download_ckpts.sh
cd ..
```

### GroundingDINO checkpoints

```bash
cd gdino_checkpoints
bash download_ckpts.sh
cd ..
```

---

## 3) Install CUDA Toolkit 11.8 on WSL (matches cu118)

This session originally had a CUDA 11.5 toolkit via `nvidia-cuda-toolkit`, but PyTorch was installed as `+cu128` (CUDA 12.8). To build the GroundingDINO CUDA extension reliably, we aligned everything to **CUDA 11.8**.

### Install CUDA 11.8 (WSL Ubuntu)

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb

sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

### Set CUDA_HOME to 11.8

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Verify:

```bash
nvcc --version
```

Expected: `release 11.8`

---

## 4) Install PyTorch (cu118) to match CUDA 11.8

Uninstall any existing torch stack:

```bash
pip uninstall -y torch torchvision torchaudio
```

Install cu118 wheels:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify torch CUDA alignment:

```bash
python - <<EOF
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
EOF
```

Expected:

* `torch.version.cuda` prints `11.8`
* `cuda_available` is `True`

---

## 5) Install Grounded-SAM-2 + build GroundingDINO extension

From the repo root:

```bash
pip install -e .
pip install --no-build-isolation -e grounding_dino
```

If `pip install --no-build-isolation -e grounding_dino` fails, it is almost always due to CUDA/torch mismatch or missing build deps. Confirm `nvcc --version` and `torch.version.cuda` are aligned.

---

## 6) Install remaining Python dependencies (used in the demos)

These were installed during the session to satisfy imports for the demo scripts:

```bash
pip install opencv-python
pip install supervision
pip install pycocotools
pip install transformers --upgrade
pip install addict
pip install yapf
pip install timm
```

---

## 7) Run the Hugging Face demo (downloads models on first run)

This demo uses Hugging Face (`AutoProcessor.from_pretrained`, etc.). It will download the model weights the first time if not cached.

Example (trees only):

```bash
python grounded_sam2_hf_model_demo.py
```

Examples used in this session with personal city_2.png example:

```bash
python grounded_sam2_hf_model_demo.py \
  --img-path "notebooks/images/city_2.png" \
  --text-prompt "tree."
```

Outputs are written to:

```text
outputs/grounded_sam2_hf_demo/
```

Check:

```bash
ls -R outputs/grounded_sam2_hf_demo | head -50
```

Hugging Face cache location:

```bash
ls -lh ~/.cache/huggingface/hub/ | head
```

---

## 8) Run the fully local demo (no Hugging Face required)

This demo uses the checkpoints downloaded in `checkpoints/` and `gdino_checkpoints/`.

```bash
python grounded_sam2_local_demo.py \
  --img-path "notebooks/images/city_2.png" \
  --text-prompt "tree."
```

Outputs are written under:

```text
outputs/grounded_sam2_local_demo/
```

---

## 9) Prompting tips for better “trees only” results

Prompts must be lowercase and end with a dot:

* `tree.`
* `trees.`
* `tree canopy.`
* `tree crown.`
* `conifer tree.`
* `palm tree.`

If the model grabs bushes/shrubs:

* prefer more specific prompts like `tree canopy.`
* increase detection thresholds inside the script (`threshold`, `text_threshold`)
* filter by box size/score before passing boxes to SAM2

---

## Reference

All steps and scripts derive from the official repository:

[https://github.com/IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)