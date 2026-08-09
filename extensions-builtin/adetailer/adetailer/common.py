from __future__ import annotations

import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from rich import print  # noqa: A004  Shadowing built-in 'print'
from torchvision.transforms.functional import to_pil_image

# Try to import InsightFace for complementary face detection
try:
    import insightface
    from insightface.app import FaceAnalysis
    from .insightface_detector import InsightFaceDetector, get_insightface_detector
    INSIGHTFACE_AVAILABLE = True
except (ImportError, AttributeError) as e:
    INSIGHTFACE_AVAILABLE = False
    InsightFaceDetector = None
    get_insightface_detector = None
    print(f"[!] InsightFace not available: {e}")

REPO_ID = "Bingsu/adetailer"

T = TypeVar("T", int, float)


@dataclass
class PredictOutput(Generic[T]):
    bboxes: list[list[T]] = field(default_factory=list)
    masks: list[Image.Image] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    preview: Optional[Image.Image] = None


def hf_download(file: str, repo_id: str = REPO_ID, check_remote: bool = True) -> str:
    # face_yolov8n.pt excluded (replaced by YOLOv11)
    if file == "face_yolov8n.pt":
        return "INVALID"
    
    # Check local paths first (builtin + legacy + cwd-relative)
    _builtin_models = Path(__file__).resolve().parents[1] / "models"
    possible_paths = [
        _builtin_models / file,
        Path("extensions-builtin/adetailer/models") / file,
        Path("extensions/adetailer/models") / file,
        Path("models/adetailer") / file,
    ]

    for local_path in possible_paths:
        if local_path.exists():
            return str(local_path.resolve())
    
    if check_remote:
        with suppress(Exception):
            return hf_hub_download(repo_id, file, etag_timeout=1)

        with suppress(Exception):
            return hf_hub_download(
                repo_id, file, etag_timeout=1, endpoint="https://hf-mirror.com"
            )

    with suppress(Exception):
        return hf_hub_download(repo_id, file, local_files_only=True)

    if check_remote:
        msg = f"[-] ADetailer: Failed to load model {file!r} from huggingface"
        print(msg)
    return "INVALID"


def safe_mkdir(path: str | os.PathLike[str]) -> None:
    path = Path(path)
    if not path.exists() and path.parent.exists() and os.access(path.parent, os.W_OK):
        path.mkdir()


def scan_model_dir(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return [p for p in path.rglob("*") if p.is_file() and p.suffix == ".pt"]


def download_models(*names: str, check_remote: bool = True) -> dict[str, str]:
    models = OrderedDict()
    with ThreadPoolExecutor() as executor:
        for name in names:
            if "-world" in name:
                models[name] = executor.submit(
                    hf_download,
                    name,
                    repo_id="Bingsu/yolo-world-mirror",
                    check_remote=check_remote,
                )
            else:
                models[name] = executor.submit(
                    hf_download,
                    name,
                    check_remote=check_remote,
                )
    # Preserve order by using OrderedDict
    result = OrderedDict()
    for name, future in models.items():
        result[name] = future.result()
    return result


def get_models(
    *dirs: str | os.PathLike[str], huggingface: bool = True
) -> OrderedDict[str, str]:
    model_paths = []

    for dir_ in dirs:
        if not dir_:
            continue
        model_paths.extend(scan_model_dir(Path(dir_)))

    models = OrderedDict()
    to_download = [
        "face_yolo11n.pt",  # YOLOv11n face detection (smaller, faster) - First priority
        "face_yolo11s.pt",  # YOLOv11s face detection (enhanced accuracy)
        "face_yolov8s.pt",
        "hand_yolov8n.pt",
        "person_yolov8n-seg.pt",
        "person_yolov8s-seg.pt",
        "yolov8x-worldv2.pt",
    ]

    # Prefer already-scanned local files before any Hugging Face attempt
    local_by_name = {path.name: str(path) for path in model_paths}
    need_remote = [name for name in to_download if name not in local_by_name]
    if need_remote:
        models.update(download_models(*need_remote, check_remote=huggingface))
    for name in to_download:
        if name in local_by_name:
            models[name] = local_by_name[name]

    # MediaPipe models removed - use YOLO models instead for Python 3.13+ compatibility

    invalid_keys = [k for k, v in models.items() if v == "INVALID"]
    for key in invalid_keys:
        models.pop(key)

    # Add remaining local models while preserving order
    for path in model_paths:
        if path.name in models:
            continue
        models[path.name] = str(path)

    # Reorder to ensure YOLOv11n is first, then YOLOv11s
    ordered_models = OrderedDict()
    priority_models = ["face_yolo11n.pt", "face_yolo11s.pt", "face_yolov8s.pt"]
    
    # Add priority models first
    for model in priority_models:
        if model in models:
            ordered_models[model] = models[model]
    
    # Add remaining models
    for name, path in models.items():
        if name not in ordered_models:
            ordered_models[name] = path

    return ordered_models


def create_mask_from_bbox(
    bboxes: list[list[float]], shape: tuple[int, int]
) -> list[Image.Image]:
    """
    Parameters
    ----------
        bboxes: list[list[float]]
            list of [x1, y1, x2, y2]
            bounding boxes
        shape: tuple[int, int]
            shape of the image (width, height)

    Returns
    -------
        masks: list[Image.Image]
        A list of masks

    """
    masks = []
    for bbox in bboxes:
        mask = Image.new("L", shape, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(bbox, fill=255)
        masks.append(mask)
    return masks


def create_bbox_from_mask(
    masks: list[Image.Image], shape: tuple[int, int]
) -> list[list[int]]:
    """
    Parameters
    ----------
        masks: list[Image.Image]
            A list of masks
        shape: tuple[int, int]
            shape of the image (width, height)

    Returns
    -------
        bboxes: list[list[float]]
        A list of bounding boxes

    """
    bboxes = []
    for mask in masks:
        mask = mask.resize(shape)  # noqa: PLW2901
        bbox = mask.getbbox()
        if bbox is not None:
            bboxes.append(list(bbox))
    return bboxes


def ensure_pil_image(image: Any, mode: str = "RGB") -> Image.Image:
    if not isinstance(image, Image.Image):
        image = to_pil_image(image)
    if image.mode != mode:
        image = image.convert(mode)
    return image


def insightface_predict(
    image: Image.Image, 
    model_name: str, 
    confidence: float = 0.5
) -> PredictOutput:
    """
    Predict faces using InsightFace for enhanced accuracy.
    
    Parameters
    ----------
    image : Image.Image
        Input image
    model_name : str
        InsightFace model name (buffalo_l, buffalo_m, buffalo_s)
    confidence : float
        Detection confidence threshold
        
    Returns
    -------
    PredictOutput
        Detection results with bounding boxes and confidence scores
    """
    if not INSIGHTFACE_AVAILABLE:
        print("[-] ADetailer: InsightFace not available, returning empty results")
        return PredictOutput(bboxes=[], masks=[], preview=None)
    
    detector = get_insightface_detector()
    if detector is None:
        raise RuntimeError("Failed to initialize InsightFace detector")
    
    # Extract model type from model_name
    if model_name.startswith("insightface_"):
        model_type = model_name.replace("insightface_", "")
    else:
        model_type = "buffalo_l"  # Default
    
    # Create detector with specific model and dynamic detection size
    try:
        from .insightface_detector import InsightFaceDetector
        # Use larger detection size for SDXL/Pony (1024+) images
        img_max_size = max(image.width, image.height)
        if img_max_size >= 1024:
            det_size = (1280, 1280)  # Higher resolution for SDXL
        else:
            det_size = (640, 640)  # Standard for SD1.5
        
        detector = InsightFaceDetector(model_type, det_size=det_size)
        print(f"[-] InsightFace: Using detection size {det_size} for {img_max_size}px image")
    except Exception as e:
        print(f"[-] InsightFace: Failed to create detector: {e}")
        return PredictOutput(bboxes=[], masks=[], preview=None)
    
    # Detect faces
    faces = detector.detect_faces(image, confidence)
    
    if not faces:
        return PredictOutput(bboxes=[], masks=[], preview=None)
    
    # Convert to PredictOutput format
    bboxes = []
    masks = []
    
    for face in faces:
        bbox = face['bbox']
        conf = face['confidence']
        
        # Convert bbox format [x1, y1, x2, y2] to [x1, y1, x2, y2, conf]
        bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3], conf])
        
        # Create mask from bbox
        mask = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(bbox, fill=255)
        masks.append(mask)
    
    return PredictOutput(bboxes=bboxes, masks=masks, preview=None)
