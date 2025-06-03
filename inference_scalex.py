# inference_scalex.py

# Apply patches FIRST!
try:
    import patches

    patches.apply_torchvision_patches()
except ImportError:
    print(
        "ScaleX WARNING: patches.py not found. Proceeding without patches, which may lead to errors if you're using a newer torchvision with older dependencies."
    )
except Exception as e:
    print(
        f"ScaleX WARNING: Failed to apply patches: {e}. Proceeding, but errors may occur."
    )

# --- Add Warning Filtering HERE ---
import warnings

warnings.filterwarnings(
    action="ignore",
    category=FutureWarning,
    message=".*You are using `torch.load` with `weights_only=False`.*",
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.models._utils"
)
# --- End Warning Filtering ---

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict as TypingDict
from enum import Enum
import typer
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
    SpinnerColumn,
)
from rich.status import Status
from rich.text import Text

import threading
import queue as thread_queue
import time

import io
import sys
import re
from contextlib import redirect_stdout
import traceback

from basicsr.utils import imwrite
from basicsr.utils.download_util import load_file_from_url
from scalex.utils import ScaleXEnhancer

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

app = typer.Typer(
    name="scalex",
    help="ScaleX: AI Face Restoration and Enhancement Tool.",
    add_completion=True,
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

console = Console()

STYLE_INFO = "bright_white"
STYLE_PATH = "bright_cyan"
STYLE_VALUE = "bright_white"
STYLE_SUCCESS = "bright_green"
STYLE_ERROR = "bright_red"
STYLE_WARNING = "bright_yellow"
STYLE_HEADER = "bold bright_white"


class FaceModelEnum(str, Enum):
    v1_3 = "v1.3"
    v1_4 = "v1.4"


class BGModelEnum(str, Enum):
    none = "none"
    x2plus = "x2"
    x4plus = "x4"


SCALEX_MODEL_CONFIGS: TypingDict[str, TypingDict[str, Any]] = {
    "v1.3": {
        "arch": "clean",
        "channel_multiplier": 2,
        "model_name": "GFPGANv1.3",
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
    },
    "v1.4": {
        "arch": "clean",
        "channel_multiplier": 2,
        "model_name": "GFPGANv1.4",
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    },
}
REALESRGAN_MODELS: TypingDict[str, TypingDict[str, Any]] = {
    "x2": {
        "internal_name": "RealESRGAN-x2plus",
        "model_path": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "netscale": 2,
        "model_class_params": {
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 64,
            "num_block": 23,
            "num_grow_ch": 32,
            "scale": 2,
        },
    },
    "x4": {
        "internal_name": "RealESRGAN-x4plus",
        "model_path": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "netscale": 4,
        "model_class_params": {
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 64,
            "num_block": 23,
            "num_grow_ch": 32,
            "scale": 4,
        },
    },
}


class TileProgressStream(io.StringIO):
    def __init__(self, progress_queue: thread_queue.Queue, image_name_for_debug: str):
        super().__init__()
        self.progress_queue = progress_queue
        self.image_name = image_name_for_debug
        self.line_buffer = ""
        self.tile_regex = re.compile(r"Tile\s+(\d+)/(\d+)")

    def write(self, s: str):
        self.line_buffer += s
        terminators = ["\n", "\r"]
        while any(t in self.line_buffer for t in terminators):
            first_terminator_pos = -1
            for t_char in terminators:
                pos = self.line_buffer.find(t_char)
                if pos != -1:
                    if first_terminator_pos == -1 or pos < first_terminator_pos:
                        first_terminator_pos = pos
            if first_terminator_pos == -1:
                break
            line_to_process = self.line_buffer[:first_terminator_pos]
            self.line_buffer = self.line_buffer[first_terminator_pos + 1 :]
            if line_to_process.strip():
                match = self.tile_regex.search(line_to_process.strip())
                if match:
                    self.progress_queue.put(
                        {
                            "type": "tile_update",
                            "current": int(match.group(1)),
                            "total": int(match.group(2)),
                        }
                    )
        return len(s)

    def flush(self):
        pass


def get_scalex_model_display_name_and_path(
    face_model_cli_value: str,
) -> Tuple[str, str]:
    if face_model_cli_value not in SCALEX_MODEL_CONFIGS:
        console.print(
            f"[{STYLE_ERROR}]Error: Invalid face model version '{face_model_cli_value}'.[/{STYLE_ERROR}]"
        )
        raise typer.Exit(code=1)
    config = SCALEX_MODEL_CONFIGS[face_model_cli_value]
    model_display_name = config["model_name"]
    script_dir = Path(__file__).resolve().parent
    base_model_dir = script_dir / "models"
    local_path_pretrained = base_model_dir / "pretrained" / f"{model_display_name}.pth"
    if local_path_pretrained.is_file():
        return model_display_name, str(local_path_pretrained)
    local_path_weights = base_model_dir / "weights" / f"{model_display_name}.pth"
    if local_path_weights.is_file():
        return model_display_name, str(local_path_weights)
    return model_display_name, config["url"]


def enhancement_worker(
    restorer_instance: ScaleXEnhancer,
    image_np: np.ndarray,
    kwargs_for_enhance: TypingDict[str, Any],
    result_queue: thread_queue.Queue,
    image_name_for_debug: str,
    progress_q_for_stream: Optional[thread_queue.Queue] = None,
    is_bg_active: bool = False,
):
    try:
        if is_bg_active and progress_q_for_stream:
            tile_stream = TileProgressStream(
                progress_q_for_stream, image_name_for_debug
            )
            with redirect_stdout(tile_stream):
                output = restorer_instance.enhance(image_np, **kwargs_for_enhance)
        else:
            output = restorer_instance.enhance(image_np, **kwargs_for_enhance)
        result_queue.put({"data": output, "exception": None})
    except Exception as e:
        result_queue.put({"data": None, "exception": e})


@app.command()
def main(
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to input image or folder.",
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    output_path: Path = typer.Option(
        "results_scalex",
        "--output",
        "-o",
        help="Path to output folder.",
        file_okay=False,
        writable=True,
        resolve_path=True,
    ),
    face_enhance_model: FaceModelEnum = typer.Option(
        FaceModelEnum.v1_4.value,
        "--face-enhance",
        "-f",
        help=f"Face model. Choices: {[e.value for e in FaceModelEnum]}",
        case_sensitive=False,
    ),
    bg_enhance_model: BGModelEnum = typer.Option(
        BGModelEnum.x2plus.value,
        "--bg-enhance",
        "-b",
        help=f"Background model. Choices: {[e.value for e in BGModelEnum]}",
        case_sensitive=False,
    ),
    overall_upscale: int = typer.Option(
        2, "--upscale", "-s", help="Final upscale factor.", min=1, max=16
    ),
    bg_tile_size: int = typer.Option(
        400, "--bg-tile", help="Tile size for BG upsampler (0 for no tiling).", min=0
    ),
    output_suffix: Optional[str] = typer.Option(
        None, "--suffix", help="Suffix for output filenames."
    ),
    center_face_only: bool = typer.Option(
        False, "--center-face", help="Restore only center face.", is_flag=True
    ),
    aligned_input: bool = typer.Option(
        False, "--aligned", help="Inputs are aligned faces.", is_flag=True
    ),
    output_ext: str = typer.Option(
        "auto", "--ext", help="Output image extension ('png', 'jpg', 'auto')."
    ),
    fidelity_weight: Optional[float] = typer.Option(
        None,
        "--fidelity-weight",
        "-w",
        help="Fidelity weight (0-1, advanced/older models).",
        min=0.0,
        max=1.0,
        hidden=True,
    ),
    device: str = typer.Option(
        "auto", "--device", help="Device ('cpu', 'cuda', 'mps', 'auto')."
    ),
    save_cropped: bool = typer.Option(
        True, "--save-cropped/--no-save-cropped", help="Save cropped faces."
    ),
    save_restored: bool = typer.Option(
        True, "--save-restored/--no-save-restored", help="Save restored faces."
    ),
    save_comparison: bool = typer.Option(
        True, "--save-comparison/--no-save-comparison", help="Save comparisons."
    ),
):
    selected_face_model_str = face_enhance_model.value
    selected_bg_model_str = bg_enhance_model.value
    face_model_display_name, final_gfpgan_model_path_str = (
        get_scalex_model_display_name_and_path(selected_face_model_str)
    )

    bg_model_display_name = "None"
    if selected_bg_model_str != "none" and selected_bg_model_str in REALESRGAN_MODELS:
        bg_model_display_name = REALESRGAN_MODELS[selected_bg_model_str].get(
            "internal_name", selected_bg_model_str.upper()
        )

    console.rule(
        f"[{STYLE_HEADER}]ScaleX Face Restoration[/{STYLE_HEADER}]", style=STYLE_SUCCESS
    )
    console.print(
        f" L Input Path : [{STYLE_PATH}]{input_path.name if input_path.is_file() else input_path}[/{STYLE_PATH}]"
    )
    console.print(f" L Output Path: [{STYLE_PATH}]{output_path}[/{STYLE_PATH}]")
    console.print(
        f" L Face Model : [{STYLE_VALUE}]{face_model_display_name}[/{STYLE_VALUE}]"
    )
    console.print(
        f" L BG Model   : [{STYLE_VALUE}]{bg_model_display_name}[/{STYLE_VALUE}]"
    )
    console.print(f" L Upscale    : [{STYLE_VALUE}]x{overall_upscale}[/{STYLE_VALUE}]")

    torch_device_str: str
    if device.lower() == "auto":
        if torch.cuda.is_available():
            torch_device_str = "cuda"
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        ):
            torch_device_str = "mps"
        else:
            torch_device_str = "cpu"
    else:
        torch_device_str = device.lower()
    try:
        selected_torch_device = torch.device(torch_device_str)
        if selected_torch_device.type == "cuda":
            torch.cuda.current_device()
    except Exception as e:
        console.print(
            f"[{STYLE_ERROR}]Error setting device '{torch_device_str}': {e}. Falling back to CPU.[/{STYLE_ERROR}]"
        )
        selected_torch_device = torch.device("cpu")
    console.print(
        f" L Device     : [{STYLE_VALUE}]{selected_torch_device.type.upper()}[/{STYLE_VALUE}]"
    )

    output_path.mkdir(parents=True, exist_ok=True)
    if not aligned_input:
        (output_path / "restored_imgs").mkdir(exist_ok=True)
    if save_cropped:
        (output_path / "cropped_faces").mkdir(exist_ok=True)
    if save_restored:
        (output_path / "restored_faces").mkdir(exist_ok=True)
    if save_comparison:
        (output_path / "cmp").mkdir(exist_ok=True)

    img_list: List[Path]
    if input_path.is_file():
        img_list = [input_path]
    else:
        img_list = sorted(
            list(input_path.glob("*.png"))
            + list(input_path.glob("*.jpg"))
            + list(input_path.glob("*.jpeg"))
            + list(input_path.glob("*.bmp"))
            + list(input_path.glob("*.tif"))
            + list(input_path.glob("*.tiff"))
        )
    if not img_list:
        console.print(
            f"[{STYLE_ERROR}]Error: No images found in {input_path}.[/{STYLE_ERROR}]"
        )
        raise typer.Exit(code=1)

    bg_upsampler_instance: Optional[RealESRGANer] = None
    if selected_bg_model_str != "none":
        if selected_bg_model_str in REALESRGAN_MODELS:
            bg_config = REALESRGAN_MODELS[selected_bg_model_str]

            final_bg_model_path = bg_config["model_path"]
            if final_bg_model_path.startswith("https://"):
                project_model_pretrained_dir = (
                    Path(__file__).resolve().parent / "models" / "pretrained"
                )
                project_model_pretrained_dir.mkdir(parents=True, exist_ok=True)
                try:
                    final_bg_model_path = load_file_from_url(
                        url=final_bg_model_path,
                        model_dir=str(project_model_pretrained_dir),
                        progress=True,
                        file_name=None,
                    )
                    if not Path(final_bg_model_path).is_file():
                        console.print(
                            f"[{STYLE_ERROR}]BG Model file not found after download attempt: {final_bg_model_path}[/{STYLE_ERROR}]"
                        )
                        final_bg_model_path = None
                except Exception as lf_err:
                    console.print(
                        f"[{STYLE_ERROR}]Error downloading BG model: {lf_err}[/{STYLE_ERROR}]"
                    )
                    console.print(traceback.format_exc(), style=STYLE_ERROR)
                    final_bg_model_path = None
            elif not Path(final_bg_model_path).is_file():
                console.print(
                    f"[{STYLE_ERROR}]Local BG Model path specified but file not found: {final_bg_model_path}[/{STYLE_ERROR}]"
                )
                final_bg_model_path = None

            if final_bg_model_path:
                with Status(
                    f"Initializing BG Upsampler ({bg_config['internal_name']})...",
                    console=console,
                    spinner="dots",
                ) as status:
                    try:
                        model_params = bg_config["model_class_params"]
                        realesrgan_model_instance = RRDBNet(**model_params)
                        use_half_precision = selected_torch_device.type == "cuda"

                        bg_upsampler_instance = RealESRGANer(
                            scale=bg_config["netscale"],
                            model_path=str(final_bg_model_path),
                            model=realesrgan_model_instance,
                            tile=bg_tile_size if bg_tile_size > 0 else 0,
                            tile_pad=10,
                            pre_pad=0,
                            half=use_half_precision,
                            device=selected_torch_device,
                        )
                        status.update(
                            Text(
                                f"Initialized BG Upsampler ({bg_config['internal_name']})",
                                style=STYLE_SUCCESS,
                            )
                        )
                    except Exception as e:
                        console.print(traceback.format_exc(), style=STYLE_ERROR)
                        console.print(
                            f"[{STYLE_WARNING}] L Warning: Could not initialize {bg_config['internal_name']}: {e}. BG upsampling disabled.[/{STYLE_WARNING}]"
                        )
                        bg_upsampler_instance = None
            else:
                console.print(
                    f"[{STYLE_WARNING}]Skipping BG Upsampler initialization because its model path is invalid or download failed.[/{STYLE_WARNING}]"
                )
                bg_upsampler_instance = None
        else:
            console.print(
                f"[{STYLE_WARNING}] L Warning: BG model '{selected_bg_model_str}' not recognized. BG upsampling disabled.[/{STYLE_WARNING}]"
            )

    console.print(
        f" L ScaleX Model: Arch: [{STYLE_VALUE}]{SCALEX_MODEL_CONFIGS[selected_face_model_str]['arch'].upper()}[/{STYLE_VALUE}], Loaded: [{STYLE_VALUE}]{face_model_display_name}[/{STYLE_VALUE}]"
    )
    with Status(
        f"Initializing ScaleX ({face_model_display_name})...",
        console=console,
        spinner="dots",
    ) as status:
        try:
            scalex_arch = SCALEX_MODEL_CONFIGS[selected_face_model_str]["arch"]
            scalex_cm = SCALEX_MODEL_CONFIGS[selected_face_model_str][
                "channel_multiplier"
            ]
            restorer = ScaleXEnhancer(
                model_path=final_gfpgan_model_path_str,
                upscale=float(overall_upscale),
                arch=scalex_arch,
                channel_multiplier=scalex_cm,
                bg_upsampler=bg_upsampler_instance,
                device=selected_torch_device,
            )
            status.update(
                Text(
                    f"Initialized ScaleX ({face_model_display_name})",
                    style=STYLE_SUCCESS,
                )
            )
        except Exception as e:
            console.print(
                f"[{STYLE_ERROR}]Error initializing ScaleX restorer: {e}[/{STYLE_ERROR}]"
            )
            console.print(traceback.format_exc())
            raise typer.Exit(code=1)

    console.rule(
        f"[{STYLE_HEADER}]Image Processing ({len(img_list)} image{'s' if len(img_list) > 1 else ''})[/{STYLE_HEADER}]",
        style=STYLE_INFO,
    )
    total_images = len(img_list)
    for idx, img_p in enumerate(img_list):
        console.line()
        console.print(
            f"Image Name: [{STYLE_PATH}]{img_p.name}[/{STYLE_PATH}] ({idx + 1}/{total_images})"
        )
        img_name_stem = img_p.stem
        try:
            input_img_np: np.ndarray = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
            if input_img_np is None:
                console.print(
                    f"  L [{STYLE_WARNING}]Status: Failed (Could not read image)[/{STYLE_WARNING}]"
                )
                continue
        except Exception as e:
            console.print(
                f"  L [{STYLE_WARNING}]Status: Failed (Error reading image: {e})[/{STYLE_WARNING}]"
            )
            continue

        enhance_kwargs_for_call = {
            "has_aligned": aligned_input,
            "only_center_face": center_face_only,
            "paste_back": not aligned_input,
        }
        if fidelity_weight is not None:
            enhance_kwargs_for_call["weight"] = fidelity_weight
        processing_successful = False
        bg_upsampling_active_for_this_image = (
            bg_upsampler_instance is not None and bg_tile_size > 0 and not aligned_input
        )

        with Progress(
            TextColumn("  Progress:", style=STYLE_INFO),
            SpinnerColumn(spinner_name="dots", style=STYLE_VALUE),
            TextColumn("{task.description}", style="blue"),
            BarColumn(
                bar_width=30,
                style=STYLE_INFO,
                complete_style=STYLE_SUCCESS,
                finished_style=STYLE_SUCCESS,
            ),
            TaskProgressColumn(style=STYLE_VALUE),
            TimeElapsedColumn(),
            console=console,
            transient=False,
            refresh_per_second=10,
        ) as per_image_progress:
            initial_total_progress = 100 if bg_upsampling_active_for_this_image else 2
            img_task_id = per_image_progress.add_task(
                "Preparing...", total=initial_total_progress
            )
            result_q_thread = thread_queue.Queue()
            tile_progress_q_thread = (
                thread_queue.Queue() if bg_upsampling_active_for_this_image else None
            )
            worker_args = (
                restorer,
                input_img_np,
                enhance_kwargs_for_call,
                result_q_thread,
                img_p.name,
                tile_progress_q_thread,
                bg_upsampling_active_for_this_image,
            )
            enhancement_thread = threading.Thread(
                target=enhancement_worker, args=worker_args, daemon=True
            )
            enhancement_thread.start()
            animation_chars = ["   ", ".  ", ".. ", "..."]
            anim_idx = 0
            is_tiling_desc_active = False
            while enhancement_thread.is_alive():
                has_tile_update_this_iter = False
                if bg_upsampling_active_for_this_image and tile_progress_q_thread:
                    try:
                        tile_info = tile_progress_q_thread.get_nowait()
                        if tile_info["type"] == "tile_update":
                            current_t, total_t = (
                                tile_info["current"],
                                tile_info["total"],
                            )
                            task = per_image_progress.tasks[img_task_id]
                            if task.total != total_t and total_t > 0:
                                per_image_progress.update(img_task_id, total=total_t)
                            completed_val = min(
                                current_t, task.total if task.total else current_t
                            )
                            per_image_progress.update(
                                img_task_id,
                                completed=completed_val,
                                description=Text(
                                    f"BG Tiling ({current_t}/{task.total if task.total else '?'})",
                                    style="blue",
                                ),
                            )
                            has_tile_update_this_iter = True
                            is_tiling_desc_active = True
                    except thread_queue.Empty:
                        pass
                if not has_tile_update_this_iter:
                    desc_to_animate = ""
                    current_task_state = per_image_progress.tasks[img_task_id]
                    if bg_upsampling_active_for_this_image:
                        if not is_tiling_desc_active:
                            desc_to_animate = f"Upscaling BG{animation_chars[anim_idx]}"
                    elif not current_task_state.finished:
                        desc_to_animate = f"Enhancing Faces{animation_chars[anim_idx]}"
                    if desc_to_animate and not current_task_state.finished:
                        per_image_progress.update(
                            img_task_id, description=Text(desc_to_animate, style="blue")
                        )
                    anim_idx = (anim_idx + 1) % len(animation_chars)
                time.sleep(0.1)
            enhancement_thread.join()
            final_result = result_q_thread.get()
            task = per_image_progress.tasks[img_task_id]
            if final_result["exception"]:
                console.print(
                    f"  L [{STYLE_ERROR}]Status: Failed (Error: {final_result['exception']})[/{STYLE_ERROR}]"
                )
                per_image_progress.update(
                    img_task_id,
                    description=Text("Failed!", style=STYLE_ERROR),
                    completed=task.total or initial_total_progress,
                )
            else:
                cropped_faces, restored_faces, restored_output_img = final_result[
                    "data"
                ]
                final_description_text = "Complete!"
                per_image_progress.update(
                    img_task_id,
                    description=Text(final_description_text, style=STYLE_SUCCESS),
                    completed=task.total or initial_total_progress,
                )
                processing_successful = True

        if processing_successful:
            console.print(f"  L Status: [{STYLE_SUCCESS}]Successful[/{STYLE_SUCCESS}]")
            with Status(
                f"  L Saving outputs for [{STYLE_PATH}]{img_p.name}[/{STYLE_PATH}]...",
                console=console,
                spinner="earth",
            ) as save_status:
                output_ext_final: str = (
                    output_ext.lower()
                    if output_ext.lower() != "auto"
                    else img_p.suffix[1:].lower()
                )
                if not output_ext_final:
                    output_ext_final = "png"
                if save_cropped and cropped_faces:
                    for i_face, cropped_face_np in enumerate(cropped_faces):
                        face_tag = f"_face_{i_face:02d}"
                        save_path = (
                            output_path
                            / "cropped_faces"
                            / f"{img_name_stem}{face_tag}.png"
                        )
                        imwrite(cropped_face_np, str(save_path))
                if save_restored and restored_faces:
                    for i_face, restored_face_np in enumerate(restored_faces):
                        face_tag = f"_face_{i_face:02d}"
                        fn = f"{img_name_stem}{face_tag}"
                        if output_suffix:
                            fn += f"_{output_suffix}"
                        save_path = (
                            output_path / "restored_faces" / f"{fn}.{output_ext_final}"
                        )
                        imwrite(restored_face_np, str(save_path))
                if (
                    save_comparison
                    and cropped_faces
                    and restored_faces
                    and len(cropped_faces) == len(restored_faces)
                ):
                    for i_face, (cropped_face_np, restored_face_np) in enumerate(
                        zip(cropped_faces, restored_faces)
                    ):
                        face_tag = f"_face_{i_face:02d}"
                        try:
                            if (
                                cropped_face_np.shape[0] != restored_face_np.shape[0]
                                or cropped_face_np.shape[1] != restored_face_np.shape[1]
                            ):
                                target_height = restored_face_np.shape[0]
                                target_width = restored_face_np.shape[1]
                                cropped_face_np_resized = cv2.resize(
                                    cropped_face_np,
                                    (target_width, target_height),
                                    interpolation=cv2.INTER_AREA,
                                )
                            else:
                                cropped_face_np_resized = cropped_face_np
                            comp_img = np.concatenate(
                                (cropped_face_np_resized, restored_face_np), axis=1
                            )
                            save_path = (
                                output_path
                                / "cmp"
                                / f"{img_name_stem}{face_tag}_cmp.png"
                            )
                            imwrite(comp_img, str(save_path))
                        except cv2.error as e:
                            console.print(
                                f"    L [{STYLE_WARNING}]Cmp fail for {img_name_stem}{face_tag}: {e}[/{STYLE_WARNING}]"
                            )
                        except Exception as e_gen:
                            console.print(
                                f"    L [{STYLE_WARNING}]General Cmp fail for {img_name_stem}{face_tag}: {e_gen}[/{STYLE_WARNING}]"
                            )
                if not aligned_input and restored_output_img is not None:
                    fn = img_name_stem
                    if output_suffix:
                        fn += f"_{output_suffix}"
                    save_path = (
                        output_path / "restored_imgs" / f"{fn}.{output_ext_final}"
                    )
                    imwrite(restored_output_img, str(save_path))
                elif (
                    not aligned_input
                    and not restored_output_img
                    and (cropped_faces or restored_faces)
                ):
                    console.print(
                        f"  L [{STYLE_WARNING}]Note: No final composite image for {img_p.name}, though faces were processed.[/{STYLE_WARNING}]"
                    )
                save_status.update(
                    Text(
                        f"  L Outputs saved for [{STYLE_PATH}]{img_p.name}[/{STYLE_PATH}]",
                        style=STYLE_SUCCESS,
                    )
                )

    console.line(2)
    console.print(
        f"[{STYLE_SUCCESS}]Processing complete! Results saved in:[/{STYLE_SUCCESS}] [link=file://{output_path.resolve()}]{output_path.resolve()}[/link]"
    )
    console.rule(style=STYLE_SUCCESS)


if __name__ == "__main__":
    app()
