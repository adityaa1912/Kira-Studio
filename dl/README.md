# Kira Studio

A local anime-face generation studio based on the 64x64 DCGAN in [Anime-Face-Generator](https://github.com/NeilGeorge1/Anime-Face-Generator). The Jupyter workflow is split into a FastAPI backend, a React/Vite frontend, and a standalone training script.

The project includes the repository's pretrained generator at `backend/models/animeface_generator_model.h5`. The dataset archive stays in Downloads; it is not copied into the project. Training uses the archive after extracting it into `backend/data`.

## Requirements

- Windows 10/11
- Python 3.12 x64 (the standard TensorFlow CPU setup supports Python 3.10–3.13; Python 3.15 is not supported)
- Node.js 20 or later
- About 1 GB of free disk space to extract the image dataset; training is faster with a supported NVIDIA GPU, but CPU training is available

### Experimental Intel GPU training on Windows

The standard environment uses TensorFlow 2.20 CPU and cannot use the Intel integrated GPU. An optional, separate environment uses Microsoft's discontinued/preview TensorFlow DirectML plugin. It requires 64-bit Python 3.10 and `tensorflow-cpu==2.10.0`; it may not work with every graphics driver and is not a production-supported TensorFlow path. Keep the normal `.venv` for the frontend API or CPU use.

Install the Python 3.10 runtime and create this separate environment from the project root:

```powershell
uv python install 3.10
uv venv --python 3.10 ..\.venv-gpu
uv pip install --python ..\.venv-gpu\Scripts\python.exe -r backend\requirements-gpu.txt
..\.venv-gpu\Scripts\python.exe -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Continue only if the last command lists a GPU. Train a fixed 10,000-image subset for 25 epochs with:

```powershell
..\.venv-gpu\Scripts\python.exe backend\train.py --epochs 25 --batch-size 16 --max-images 10000 --seed 1337 --require-gpu --output backend\models\intel-gpu-run\animeface-generator.keras --preview-dir backend\training_previews\intel-gpu-run --checkpoint-dir backend\models\intel-gpu-run\checkpoints
```

Training prints the detected device. If no GPU is listed, do not expect acceleration; update the Intel graphics driver and recheck. The plugin is experimental and may have missing operator support. The candidate model is saved separately and does not replace the active model.

## Install

From the project root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
Set-Location frontend
npm install
Set-Location ..
```

The included H5 model can be used without extracting the dataset. Extract the archive only if you plan to train:

```powershell
python backend\prepare_dataset.py "$env:USERPROFILE\Downloads\archive (4).zip"
```

Run a small trial first. Training uses a DCGAN architecture with hinge loss, seeded horizontal-flip augmentation, deterministic shuffling, and a bounded 64-image batch by default. It saves a candidate model separately from the active model, a 256px 16-image preview after each epoch, loss metrics, and resumable checkpoints:

```powershell
python backend\train.py --epochs 1 --batch-size 16 --max-images 256
```

Preview images are saved in `backend/training_previews`, epoch losses in `backend/models/training_metrics.csv`, checkpoints in `backend/models/checkpoints`, and the experimental generator in `backend/models/animeface-generator-experiment.keras`. The API continues using the existing model; it will not switch automatically. For a longer run, omit `--max-images` and increase `--epochs`. To continue from the latest checkpoint, use `--resume` and request a larger total epoch count, for example `python backend\train.py --epochs 20 --resume`. Keep the dataset, seed, and output/checkpoint paths unchanged when resuming. Review previews and losses and compare the candidate against the current model before switching.

## Run

Open two PowerShell terminals from the project root. In the first:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --app-dir backend --reload --port 8000
```

In the second:

```powershell
Set-Location frontend
npm run dev
```

Open the URL printed by Vite, usually `http://localhost:5173`. The frontend proxies API calls to the local FastAPI server. The API docs are at `http://127.0.0.1:8000/docs`.
Generated 64x64 model outputs are first resized to 256x256 with Lanczos. Optional AI super-resolution uses the ONNX Model Zoo Sub-Pixel CNN (Apache-2.0, trained on BSD300) to produce a 672x672 image; it enhances luminance and resizes color channels, and may introduce or miss details because it was not trained specifically for this anime dataset. The frontend enables AI upscaling by default and offers a checkbox to compare the regular 256x256 result. AI inference currently runs on CPU and takes longer. The model file is `backend/models/super-resolution-10.onnx`; Python dependencies are listed in both requirements files. Each generation response includes the seed, image count, variation, upscale choice, and output size; Repeat batch reuses those settings.

## Useful commands

Extract to another location:

```powershell
python backend\prepare_dataset.py "C:\path\to\archive (4).zip" --destination backend\data
```

Train with a limited dataset sample and smaller batch for a quick CPU trial:

```powershell
python backend\train.py --epochs 1 --batch-size 16 --max-images 256
```

Set a custom model or dataset location with `ANIME_GENERATOR_MODEL` and `ANIME_DATASET_ARCHIVE` / `ANIME_DATASET_DIR` environment variables.