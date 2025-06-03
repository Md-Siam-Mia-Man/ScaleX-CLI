# scalex/utils.py
import cv2
import os
import torch
import numpy as np
from torch import nn, Tensor
from typing import (
    Literal,
    Tuple,
    List,
    Optional,
    Union,
    OrderedDict,
)

from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize

from .archs.gfpgan_bilinear_arch import GFPGANBilinear
from .archs.gfpganv1_arch import GFPGANv1
from .archs.gfpganv1_clean_arch import GFPGANv1Clean

PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ArchType = Literal["clean", "original", "bilinear", "RestoreFormer"]


class ScaleXEnhancer:
    def __init__(
        self,
        model_path: str,
        upscale: float = 2.0,
        arch: ArchType = "clean",
        channel_multiplier: int = 2,
        bg_upsampler: Optional[nn.Module] = None,
        device: Optional[Union[torch.device, str]] = None,
    ):
        self.upscale: float = upscale
        self.bg_upsampler: Optional[nn.Module] = bg_upsampler

        if device is None:
            self.device: torch.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.gfpgan: nn.Module
        if arch == "clean":
            self.gfpgan = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True,
            )
        elif arch == "bilinear":
            self.gfpgan = GFPGANBilinear(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True,
            )
        elif arch == "original":
            self.gfpgan = GFPGANv1(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=True,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True,
            )
        elif arch == "RestoreFormer":
            try:
                from .archs.restoreformer_arch import RestoreFormer

                self.gfpgan = RestoreFormer()
            except ImportError:
                raise ImportError(
                    "RestoreFormer architecture selected but its module could not be imported. Ensure '.archs.restoreformer_arch' is available."
                )
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        facexlib_model_root_default = os.path.join(
            PROJECT_ROOT_DIR, "models", "weights"
        )
        if not os.path.isdir(facexlib_model_root_default):
            print(
                f"ScaleX Info: Pre-defined facexlib model rootpath '{facexlib_model_root_default}' not found. "
                "Face detection/parsing will rely on facexlib's default caching or download mechanisms."
            )
            facexlib_model_path_for_helper = facexlib_model_root_default  # Pass it anyway, FaceRestoreHelper handles it
        else:
            facexlib_model_path_for_helper = facexlib_model_root_default

        self.face_helper = FaceRestoreHelper(
            upscale_factor=int(upscale),
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=self.device,
            model_rootpath=facexlib_model_path_for_helper,
        )

        if model_path.startswith("https://"):
            model_download_dir = os.path.join(PROJECT_ROOT_DIR, "models", "pretrained")
            os.makedirs(model_download_dir, exist_ok=True)
            model_name_from_url = model_path.split("/")[-1]
            local_model_path_check = os.path.join(
                model_download_dir, model_name_from_url
            )

            if os.path.isfile(local_model_path_check):
                print(
                    f"ScaleX Info: Model '{model_name_from_url}' found locally at '{local_model_path_check}'. Using local copy."
                )
                model_path = local_model_path_check
            else:
                print(f"ScaleX Info: Downloading model from URL: {model_path}")
                model_path = load_file_from_url(
                    url=model_path,
                    model_dir=model_download_dir,
                    progress=True,
                    file_name=None,
                )

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model weights not found at the resolved path: {model_path}"
            )

        loadnet = torch.load(
            model_path, map_location=lambda storage, loc: storage, weights_only=True
        )

        keyname = (
            "params_ema"
            if "params_ema" in loadnet
            else (
                "params"
                if "params" in loadnet
                else "g_ema" if "g_ema" in loadnet else None
            )
        )

        loaded_successfully = False
        if keyname and keyname in loadnet:
            try:
                self.gfpgan.load_state_dict(loadnet[keyname], strict=False)
                loaded_successfully = True
            except RuntimeError as e:
                print(
                    f"ScaleX Warning: Failed to load state_dict with key '{keyname}' (strict=False). Error: {e}"
                )

        if not loaded_successfully and isinstance(loadnet, (OrderedDict, dict)):
            try:
                new_state_dict = OrderedDict()
                has_module_prefix = any(k.startswith("module.") for k in loadnet.keys())
                if has_module_prefix:
                    print(
                        "ScaleX Info: Removing 'module.' prefix from model state_dict keys."
                    )
                    for k, v in loadnet.items():
                        name = k[7:] if k.startswith("module.") else k
                        new_state_dict[name] = v
                    self.gfpgan.load_state_dict(new_state_dict, strict=False)
                else:
                    self.gfpgan.load_state_dict(loadnet, strict=False)
                loaded_successfully = True
            except RuntimeError as e:
                print(
                    f"ScaleX Warning: Failed to load state_dict directly (strict=False). Error: {e}"
                )

        if not loaded_successfully:
            raise ValueError(
                f"Cannot load model weights from {model_path}: Unknown format {type(loadnet)} or failed to match keys."
            )

        self.gfpgan.eval()
        self.gfpgan = self.gfpgan.to(self.device)

    @torch.no_grad()
    def enhance(
        self,
        img: np.ndarray,
        has_aligned: bool = False,
        only_center_face: bool = False,
        paste_back: bool = True,
        weight: Optional[float] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[np.ndarray]]:
        self.face_helper.clean_all()

        if has_aligned:
            if img.shape[0:2] != (512, 512):
                print(
                    f"ScaleX Info: Resizing aligned input from {img.shape[0:2]} to (512, 512)."
                )
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            self.face_helper.cropped_faces = [img]
        else:
            self.face_helper.read_image(img)
            self.face_helper.get_face_landmarks_5(
                only_center_face=only_center_face, eye_dist_threshold=5
            )
            self.face_helper.align_warp_face()

        if not self.face_helper.cropped_faces:
            print(
                "ScaleX Info: No faces detected or aligned successfully in the input image."
            )
            if paste_back:
                if self.bg_upsampler is not None:
                    try:
                        print(
                            "ScaleX Info: No faces processed, attempting background upsampling only."
                        )
                        bg_output = self.bg_upsampler.enhance(
                            img, outscale=self.upscale
                        )
                        if (
                            isinstance(bg_output, tuple)
                            and len(bg_output) > 0
                            and isinstance(bg_output[0], np.ndarray)
                        ):
                            return [], [], bg_output[0]
                        elif isinstance(bg_output, np.ndarray):
                            return [], [], bg_output
                        else:
                            print(
                                f"    [Warning] Unexpected output from bg_upsampler (no faces): {type(bg_output)}. Using original image."
                            )
                            return [], [], img.copy()
                    except Exception as bg_err:
                        print(
                            f"    [Error] Background upsampling failed (no faces): {bg_err}. Using original image."
                        )
                        return [], [], img.copy()
                else:
                    return [], [], img.copy()
            else:
                return [], [], None

        for cropped_face_np in self.face_helper.cropped_faces:
            cropped_face_tensor: Tensor = img2tensor(
                cropped_face_np / 255.0, bgr2rgb=True, float32=True
            )
            normalize(
                cropped_face_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True
            )
            cropped_face_tensor = cropped_face_tensor.unsqueeze(0).to(self.device)

            gfpgan_call_kwargs = {"return_rgb": False}
            if weight is not None and isinstance(self.gfpgan, GFPGANv1):
                gfpgan_call_kwargs["weight"] = weight
            elif weight is not None and not isinstance(self.gfpgan, GFPGANv1):
                print(
                    f"ScaleX Warning: 'weight' argument provided but the current architecture ('{type(self.gfpgan).__name__}') does not support it. Ignoring 'weight'."
                )

            try:
                output_from_gfpgan = self.gfpgan(
                    cropped_face_tensor, **gfpgan_call_kwargs
                )
                if (
                    isinstance(output_from_gfpgan, tuple)
                    and len(output_from_gfpgan) > 0
                ):
                    output_tensor = output_from_gfpgan[0]
                elif isinstance(output_from_gfpgan, torch.Tensor):
                    output_tensor = output_from_gfpgan
                else:
                    raise ValueError(
                        f"Unexpected output type from GFPGAN model: {type(output_from_gfpgan)}"
                    )
                restored_face_np = tensor2img(
                    output_tensor.squeeze(0), rgb2bgr=True, min_max=(-1, 1)
                )
            except Exception as e:
                print(
                    f"    [Error] ScaleX inference failed for a face: {str(e)}. Using original cropped face for this instance."
                )
                restored_face_np = cropped_face_np

            restored_face_np = restored_face_np.astype(np.uint8)
            self.face_helper.add_restored_face(restored_face_np)

        final_restored_img: Optional[np.ndarray] = None
        if not has_aligned and paste_back:
            # This block assumes self.face_helper.restored_faces is now populated if faces were processed.
            # If, for some reason, add_restored_face failed or wasn't called for all cropped_faces,
            # restored_faces might be empty or shorter than cropped_faces.
            # The FaceRestoreHelper's paste_faces_to_input_image should handle cases where restored_faces is empty.

            background_img_for_paste: Optional[np.ndarray] = None
            if self.bg_upsampler is not None:
                try:
                    print("ScaleX Info: Upsampling background for face pasting.")
                    # Assuming bg_upsampler.enhance matches RealESRGANer's signature
                    bg_output = self.bg_upsampler.enhance(img, outscale=self.upscale)
                    if (
                        isinstance(bg_output, tuple)
                        and len(bg_output) > 0
                        and isinstance(bg_output[0], np.ndarray)
                    ):
                        background_img_for_paste = bg_output[0]
                    elif isinstance(bg_output, np.ndarray):
                        background_img_for_paste = bg_output
                    else:
                        print(
                            f"    [Warning] Unexpected output from bg_upsampler: {type(bg_output)}"
                        )
                except Exception as bg_err:
                    print(f"    [Error] Background upsampling failed: {bg_err}")

            # It's crucial that self.face_helper.restored_faces is correctly populated before this step.
            if not self.face_helper.restored_faces and self.face_helper.cropped_faces:
                print(
                    "ScaleX Warning: Faces were cropped, but no faces were successfully restored. Pasting may use original faces or fail."
                )

            self.face_helper.get_inverse_affine(None)  # Prepare for pasting
            final_restored_img = self.face_helper.paste_faces_to_input_image(
                upsample_img=background_img_for_paste
            )
        elif (
            has_aligned and self.face_helper.restored_faces
        ):  # Input was aligned and faces were restored
            final_restored_img = self.face_helper.restored_faces[0]
        elif (
            not paste_back and self.face_helper.restored_faces
        ):  # Not pasting back, but faces were restored (e.g., for --aligned input)
            # This case might be redundant if has_aligned covers it, but can be a fallback.
            # If only one face, return it. If multiple, this might need different handling or just return None for the composite.
            if len(self.face_helper.restored_faces) == 1:
                final_restored_img = self.face_helper.restored_faces[0]
            # else: more than one restored face with no paste_back and not has_aligned is ambiguous for 'final_restored_img'.
            # The individual restored faces are still available in self.face_helper.restored_faces.

        return (
            self.face_helper.cropped_faces,
            self.face_helper.restored_faces,
            final_restored_img,
        )
