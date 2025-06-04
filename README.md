<p align="center">
  <img src="assets/Banner.png" alt="ScaleX-CLI Banner">
  <!-- Make sure 'assets/Banner.png' exists in your project or update the path -->
</p>

# 🌟 ScaleX: AI Face Restoration & Enhancement CLI

ScaleX is a powerful command-line tool for enhancing and restoring faces in images using advanced AI. Built on the robust foundations of GFPGAN and Real-ESRGAN, ScaleX has been modernized for Python 3.12, features a user-friendly interactive CLI, and includes smarts for easier setup and smoother operation.

<!-- Optional: Add a before/after image or a short demo GIF -->
<!--
<p align="center">
  <img src="assets/scalex_before_after_example.png" width="70%" alt="ScaleX Before and After">
</p>
-->

## 🗂 Table of Contents

- 📖 [Introduction](#-introduction)
- ✨ [Features](#-features)
- 🛠️ [Installation](#️-installation)
  - 📋 [Prerequisites](#-prerequisites)
  - 🚀 [Automated Installation (Recommended)](#-automated-installation-recommended)
  - 🔩 [Manual Installation](#-manual-installation)
- 💻 [Usage](#-usage)
  - ▶️ [Running ScaleX](#️-running-scalex)
  - ⚙️ [Key Options](#️-key-options)
- 💡 [Troubleshooting](#-troubleshooting)
- 🤝 [Contributing](#-contributing)
- 📜 [Acknowledgments & License](#-acknowledgments--license)

---

## 📖 Introduction

ScaleX brings cutting-edge AI face restoration to your command line. Whether you're dealing with old family photos, blurry portraits, or low-resolution images, ScaleX aims to revitalize them with remarkable clarity and detail. It intelligently combines face-specific enhancement using **GFPGAN** models with optional background upscaling via **Real-ESRGAN**.

This project focuses on delivering a refined and accessible experience by:

- Upgrading the core to **Python 3.12**.
- Providing an **enhanced CLI** with clear options and rich progress feedback (thanks to Typer and Rich).
- Automating fixes for common **dependency issues**.
- Streamlining **model downloads** with user-friendly progress bars.

---

## ✨ Features

- 🚀 **High-Quality Face Restoration:** Leverages **GFPGAN v1.3 & v1.4** models.
- 🖼️ **Background Enhancement:** Optional background upscaling with **Real-ESRGAN x2plus & x4plus**.
- 📈 **Overall Image Upscaling:** Control the final output resolution (e.g., 2x, 4x).
- 💻 **Intuitive CLI:** User-friendly command-line interface that's easy to learn and use.
- 📂 **Batch Processing:** Process individual images or entire folders of images.
- 🎨 **Comprehensive Outputs:** Saves restored full images, cropped faces, individually restored faces, and side-by-side comparison images.
- 🎯 **Aligned Input Support:** Option for pre-aligned 512x512 face inputs for specialized workflows.
- ⚙️ **Flexible Device Control:** Supports CPU, NVIDIA CUDA GPUs, and Apple Silicon (MPS).
- 📝 **Customizable Outputs:** Control output file extensions and add custom suffixes.
- 📊 **Rich Progress Indicators:** Clear, single-line progress bars for model downloads and detailed progress for image processing steps.
- 🔧 **Automated Dependency Patching:** Resolves known `torchvision` compatibility issues at runtime.

---

## 🛠️ Installation

### 📋 Prerequisites

- 🐍 [Python](https://www.python.org/) 3.12
- 🐉 [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) (Required for the automated installation scripts and recommended for PyTorch)
- ➕ [Git](https://git-scm.com/) (To clone the repository)
- ❗ **For GPU Acceleration (Optional but Recommended):**
  - NVIDIA GPU + Up-to-date [CUDA Drivers](https://www.nvidia.com/Download/index.aspx)

### 🚀 Automated Installation (Recommended)

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/Md-Siam-Mia-Code/ScaleX-CLI.git
    cd ScaleX-CLI
    ```

2.  **Run the Installation Script:**
    The scripts will guide you through creating a Conda environment, choosing a PyTorch version (CPU/GPU), and installing all dependencies.

    - **For Windows:**
      Open a Command Prompt (cmd.exe) or PowerShell in the `ScaleX-CLI` directory and run:

      ```batch
      install_scalex_windows.bat
      ```

    - **For Linux / macOS:**
      Open your terminal in the `ScaleX-CLI` directory, make the script executable, then run it:
      `bash
    chmod +x install_scalex.sh
    ./install_scalex_linux.sh
    `
      Follow the on-screen prompts provided by the script.

### 🔩 Manual Installation

If you prefer to set up the environment manually:

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/Md-Siam-Mia-Code/ScaleX-CLI.git
    cd ScaleX-CLI
    ```

2.  **Create and Activate Conda Environment:**

    ```bash
    conda create -n ScaleX-CLI python=3.12 -y
    conda activate ScaleX-CLI
    ```

3.  **Install PyTorch:**
    Visit the [PyTorch official website](https://pytorch.org/get-started/locally/) to get the correct `conda install` command for your system (OS, package manager, CUDA version if applicable).

    _Example for CPU-only (conda):_

    ```bash
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    ```

    _Example for CUDA (conda):_

    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=<your-cuda-version> -c pytorch -c nvidia -y
    ```

    _(Adjust the CUDA version as needed for your setup)_

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    
5.  **Model Downloads:**
    🚀 _Required models (GFPGAN, Real-ESRGAN, facexlib components) will be automatically downloaded with a progress bar when you run the application for the first time if they are not found in the `models/pretrained/` directory._

---

## 💻 Usage

### ▶️ Running ScaleX

After successful installation:

1.  Open your terminal or Anaconda Prompt.
2.  Activate the Conda environment: `conda activate ScaleX-CLI`
3.  Navigate to the `ScaleX-CLI` directory.
4.  Run the main script `inference_scalex.py`.

**Basic Command Structure:**

```bash
python inference_scalex.py --input <path_to_input> --output <path_to_output_folder> [OPTIONS]
```

**Example:**
To process all images in a folder named `MyPhotos` and save results to `EnhancedPhotos`, using GFPGAN v1.4, RealESRGAN x4 for background, and upscaling the final image by 4x:

````bash
python inference_scalex.py -i MyPhotos/ -o EnhancedPhotos/ -f v1.4 -b x4 -s 4
```

### ⚙️ Key Options
*   `-i, --input PATH`: **Required.** Path to your input image or a folder containing images.
*   `-o, --output PATH`: Folder where results will be saved. (Default: `results_scalex`)
*   `-f, --face-enhance [v1.3|v1.4]`: Choose the GFPGAN model for face restoration. (Default: `v1.4`)
*   `-b, --bg-enhance [none|x2|x4]`: Background enhancement model.
    *   `none`: No background processing.
    *   `x2`: RealESRGAN x2plus.
    *   `x4`: RealESRGAN x4plus.
    (Default: `x2`)
*   `-s, --upscale INTEGER`: The final desired upscaling factor for the *entire image*. (Default: `2`)
*   `--bg-tile INTEGER`: Tile size for background upsampler to manage memory. 0 disables tiling. (Default: `400`)
*   `--device [auto|cpu|cuda|mps]`: Select computation device. (Default: `auto`)
*   `--aligned`: Use this if your input images are already 512x512 aligned faces.
*   `--ext [auto|png|jpg]`: Extension for saved output images. (Default: `auto`)
*   `--no-save-cropped`: Do not save the initial cropped faces.
*   `--no-save-restored`: Do not save the individually restored faces.
*   `--no-save-comparison`: Do not save the side-by-side comparison images.

**Get Full Help:**
For a comprehensive list of all available options and their descriptions:
```bash
python inference_scalex.py --help
````

---

## 💡 Troubleshooting

- **`ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'`**:
  ScaleX includes an automated patch (`patches.py`) for this common `basicsr`/`torchvision` issue. It should be applied automatically. If you see this, ensure `patches.py` is present and correctly imported at the start of `inference_scalex.py`.
- **Model Download Issues**:
  - Check your internet connection.
  - If a download seems stuck or corrupt, try deleting the specific `.pth` file from the `models/pretrained/` directory within your ScaleX project folder and re-run.
- **GPU (CUDA / MPS) Errors**:
  - **NVIDIA CUDA:** Ensure your NVIDIA drivers are up-to-date and that you installed the PyTorch version matching your system's CUDA toolkit version.
  - **Apple MPS:** Ensure you are on a compatible macOS and PyTorch version.
  - Test with `--device cpu` to isolate if the problem is GPU-specific.
- **Installation Script Issues**:
  - Ensure Conda is installed and its `bin` or `Scripts` directory is in your system's PATH, or that Conda has been initialized for your shell (e.g., via `conda init bash` or `conda init cmd.exe`).
- **Other `ModuleNotFoundError`s**:
  Make sure you have activated the correct conda environment (`conda activate ScaleX-CLI`) and that all packages from `requirements.txt` were installed successfully.

---

## 🤝 Contributing

🎉 Contributions, issues, and feature requests are welcome!
Feel free to check the [issues page](https://github.com/Md-Siam-Mia-Code/ScaleX-CLI/issues). You can also fork the repository and submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## 📜 Acknowledgments & License

ScaleX is built upon and inspired by the incredible work of the open-source community. Special thanks to the creators and maintainers of:

- **GFPGAN:** [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN)
- **Real-ESRGAN:** [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- **BasicSR:** [xinntao/BasicSR](https://github.com/xinntao/BasicSR)
- **facexlib:** [xinntao/facexlib](https://github.com/xinntao/facexlib)
- **Typer & Rich:** For the excellent CLI and console UI libraries.

This project is licensed under the **MIT License**. Please see the `LICENSE` file for more details.
_(Note: The underlying models and libraries (GFPGAN, Real-ESRGAN, etc.) have their own licenses which should also be respected.)_

---

# ❤️ _Happy Enhancing with ScaleX!_
