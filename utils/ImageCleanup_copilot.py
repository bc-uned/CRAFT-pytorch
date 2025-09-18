from __future__ import annotations

import io
import time
import yaml
import math
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover - environment dependent
    raise ImportError("OpenCV (cv2) is required for ImageCleanup.") from exc

try:
    from PIL import Image, ImageOps  # type: ignore
except Exception as exc:  # pragma: no cover - environment dependent
    raise ImportError("Pillow (PIL) is required for ImageCleanup.") from exc

# Optional dependencies from scikit-image. Gracefully degrade if unavailable.
try:
    from skimage import exposure, filters, morphology, measure, util  # type: ignore
    SKIMAGE_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    SKIMAGE_AVAILABLE = False


# ---- Types --------------------------------------------------------------------

ArrayLike = np.ndarray
ImageInput = Union[str, ArrayLike, Image.Image]
StepType = Dict[str, Any]
BBox = Tuple[int, int, int, int]  # x, y, w, h
Polygon = np.ndarray  # Nx2 float/int array
Region = Union[BBox, Polygon]
RegionMode = Literal["bbox", "polygon"]
CanvasMode = Literal["overlay", "canvas"]


# ---- Utilities ----------------------------------------------------------------


def _pil_to_cv2(img: Image.Image) -> ArrayLike:
    """Convert a PIL image to OpenCV BGR numpy array."""
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _cv2_to_pil(arr: ArrayLike) -> Image.Image:
    """Convert an OpenCV BGR numpy array to PIL image."""
    if len(arr.shape) == 2:
        return Image.fromarray(arr)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def _ensure_uint8(img: ArrayLike) -> ArrayLike:
    """Ensure image is uint8 with range [0,255]."""
    if img.dtype == np.uint8:
        return img
    # Normalize to 0-255
    imin, imax = float(img.min()), float(img.max())
    if imax <= imin:
        return img.astype(np.uint8)
    scaled = (img.astype(np.float32) - imin) / (imax - imin)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def _to_gray(img: ArrayLike) -> ArrayLike:
    """Convert BGR/RGB to single-channel grayscale."""
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _estimate_char_height(gray: ArrayLike) -> Optional[float]:
    """
    Heuristic estimation of character height using connected components on a binarized image.
    Returns average component height (excluding small noise), or None if not enough data.
    """
    g = gray if len(gray.shape) == 2 else _to_gray(gray)
    # Try Otsu threshold
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Assume dark text on light bg; if inverted, flip
    if bw.mean() < 127:
        bw = 255 - bw
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(255 - bw, connectivity=8)
    heights = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if h < 5 or w < 2:
            continue
        if area < 10:
            continue
        heights.append(h)
    if not heights:
        return None
    return float(np.median(heights))


def _invert_if_white_on_dark(gray: ArrayLike) -> ArrayLike:
    """
    Invert image if it appears to be light text on dark background.
    Uses a simple heuristic on mean intensity inside likely text strokes.
    """
    g = gray if len(gray.shape) == 2 else _to_gray(gray)
    # Thin edges to guess stroke polarity
    edges = cv2.Canny(g, 50, 150)
    # Sample intensities at edge pixels
    vals = g[edges > 0]
    if vals.size == 0:
        return g
    # If edge pixels are mostly bright, likely white text on black => invert
    if np.median(vals) > 127:
        return 255 - g
    return g


def _safe_kernel(size: int) -> ArrayLike:
    """Create a square kernel for morphology with odd size >= 1."""
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))


def _homography_from_resize(src_shape: Tuple[int, int], dst_shape: Tuple[int, int]) -> np.ndarray:
    """Compute homography induced by uniform resize from src_shape (h,w) to dst_shape (h,w)."""
    sh, sw = src_shape
    dh, dw = dst_shape
    sx, sy = dw / float(sw), dh / float(sh)
    H = np.array([[sx, 0, 0],
                  [0, sy, 0],
                  [0,  0, 1]], dtype=np.float64)
    return H


def _homography_from_border(top: int, bottom: int, left: int, right: int) -> np.ndarray:
    """Homography for adding border (a translation)."""
    tx, ty = float(left), float(top)
    H = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0,  1]], dtype=np.float64)
    return H


# ---- Timing Decorator ---------------------------------------------------------


def time_step(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to time a transformation step and attach the elapsed ms to the step dict."""

    def wrapper(self: "ImageCleanup", step: StepType, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(self, step, *args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        step["elapsed_ms"] = round(elapsed_ms, 3)
        return result

    return wrapper


# ---- Transform Registry -------------------------------------------------------


class TransformError(RuntimeError):
    """Raised when a transformation fails or is unknown."""


@dataclass
class TransformRecord:
    """Record of a shape-changing transform for back-projection."""
    name: str
    H: np.ndarray  # 3x3 homography mapping from previous to current image coordinates


# ---- ImageCleanup -------------------------------------------------------------


class ImageCleanup:
    """
    Image preprocessing pipeline for OCR.

    Features:
    - Normalize input (filename, cv2 image, PIL image) to numpy ndarray (BGR).
    - Sequential transformation pipeline with timing, per-whole-image or per-region execution.
    - Text-detection steps switch the pipeline into region (bbox/polygon) mode.
    - Optionally keep intermediate images for both whole-image and region-level histories.
    - Basic homography tracking to back-project results into original coordinates.
    - YAML load/save for global config and multiple pipelines.

    Notes on text detection:
    - Detection step types: "EAST", "DB50", "DB18", "CRAFT", "PP-OCRv5".
      These require external model files. This implementation provides the interface and hooks,
      and sets bbox/polygon mode. You can integrate actual detectors by extending the
      _detect_* methods.
    """

    SUPPORTED_TYPES: Tuple[str, ...] = (
        # Detection
        "EAST", "DB50", "DB18", "CRAFT", "PP-OCRv5",
        # Resizing & resolution
        "resize_to_dpi", "resize_to_char_height", "resize_scale",
        # Grayscale & polarity
        "grayscale", "invert_if_white_on_dark",
        # Contrast
        "equalize_hist", "clahe",
        # Noise reduction
        "gaussian_blur", "median_blur", "bilateral_filter", "nl_means",
        # Thresholding
        "otsu", "otsu_after_gaussian", "niblack", "sauvola",
        "global_threshold", "adaptive_mean", "adaptive_gaussian", "try_all_threshold",
        # Skew & perspective
        "deskew_moments", "deskew_hough", "perspective_warp",
        # Morphology
        "dilate", "erode", "open", "close", "skeletonize",
        # Edge & border enhancements
        "canny_dilate", "mser_enhance", "unsharp_mask",
        # Border & margin
        "add_border", "remove_border",
        # Color channels
        "extract_channel",
        # Adaptive filtering
        "local_equalize_adapthist",
        # Segmentation
        "connected_components",
        # OCR-specific
        "kasar_text_extraction",
        # Utility
        "noop",
    )

    DETECTION_TYPES: Tuple[str, ...] = ("EAST", "DB50", "DB18", "CRAFT", "PP-OCRv5")

    def __init__(
        self,
        image: ImageInput,
        steps: Optional[List[StepType]] = None,
        keep_details: bool = False,
        work_per_region: bool = False,
        region_mode: RegionMode = "bbox",
        global_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the ImageCleanup pipeline.

        Args:
            image: cv2 image (ndarray), PIL.Image, or filename (str).
            steps: List of transformation steps dicts.
            keep_details: If True, store intermediate images after each step.
            work_per_region: If True, apply transformations per region rather than whole image.
            region_mode: "bbox" or "polygon" (only used when work_per_region is True).
            global_config: Optional global configuration dict for defaults.
        """
        self.original_bgr: ArrayLike = self._normalize_input(image)
        self.image: ArrayLike = self.original_bgr.copy()
        self.keep_details: bool = bool(keep_details)
        self.steps: List[StepType] = steps[:] if steps else []
        self.global_config: Dict[str, Any] = global_config.copy() if global_config else {}

        # Region processing
        self.bbox_mode: bool = bool(work_per_region)
        self.region_mode: RegionMode = region_mode
        self.regions: List[Region] = []  # bboxes or polygons

        # Histories
        self.history_image_steps: List[Tuple[str, ArrayLike]] = []
        self.history_region_steps: List[Tuple[str, List[ArrayLike]]] = []

        # Transform chain for back-projection
        self._transform_chain: List[TransformRecord] = []
        self._current_H: np.ndarray = np.eye(3, dtype=np.float64)  # original -> current

        # Results captured per step when keep_details is enabled
        self._results_per_step: List[Any] = []

    # ---- Public API -----------------------------------------------------------

    def run(self) -> ArrayLike:
        """
        Execute the configured pipeline in order. Mutates self.image and internal state.

        Returns:
            The final processed image (whole image if not in region mode, or a composed canvas
            if in region mode where regions were transformed and placed back).
        """
        self._results_per_step.clear()
        for step in self.steps:
            self._execute_step(step)
        return self.image

    def add_steps(self, steps: Iterable[StepType]) -> None:
        """Append additional steps to the pipeline."""
        self.steps.extend(steps)

    def return_details(self) -> List[StepType]:
        """
        Return the steps list with added "result" entry per step.

        - For whole-image steps: "result" is the transformed image (ndarray).
        - For region steps: "result" is the list of transformed region images.
        """
        # Attach results back into self.steps, preserving order
        for s, r in zip(self.steps, self._results_per_step):
            s["result"] = r
        return self.steps

    def download_all(self) -> None:
        """
        Download/setup all required OCR detection models and language packs.

        This is a stub for integration with EAST, DBNet (DB50/DB18), CRAFT, and PP-OCRv5 models.
        Implementations typically:
        - Download model files (e.g., .pb, .onnx, .pth) to a cache directory.
        - Verify checksums and versions.
        - Prepare language packs for OCR engines if needed.

        Raises:
            NotImplementedError: Until integrated with actual model downloaders.
        """
        raise NotImplementedError("Model downloader is not implemented in this reference class.")

    def save_yaml(self, path: str, pipelines: Optional[Dict[str, List[StepType]]] = None) -> None:
        """
        Save global config and (optionally) multiple pipelines to a YAML file.

        Args:
            path: Output YAML path.
            pipelines: A mapping like {"default": steps, "variantA": stepsA, ...}.
                       If None, saves only the current instance pipeline under 'current'.
        """
        data = {
            "global_config": self.global_config,
            "pipelines": pipelines if pipelines is not None else {"current": self.steps},
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: str, pipeline_key: str = "current") -> Tuple[Dict[str, Any], List[StepType]]:
        """
        Load global config and a specific pipeline from YAML.

        Args:
            path: YAML file path.
            pipeline_key: Which pipeline key to load from 'pipelines'.

        Returns:
            (global_config, steps) tuple.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        global_config = data.get("global_config", {}) or {}
        pipelines = data.get("pipelines", {}) or {}
        steps = pipelines.get(pipeline_key, []) or []
        return global_config, steps

    def back_project(
        self,
        img: Optional[ArrayLike] = None,
        mode: CanvasMode = "overlay",
        background_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> ArrayLike:
        """
        Reproject the current image (or provided img) into the original image coordinates.

        Args:
            img: If provided, back-project this image; otherwise use self.image.
            mode: "overlay" places the warped image over original, "canvas" renders on a blank canvas.
            background_color: Background color used for "canvas" mode (BGR).

        Returns:
            The reprojected image aligned to the original coordinate system.
        """
        if img is None:
            img = self.image
        H_total = self._current_H  # original -> current
        # We need inverse: current -> original
        H_inv = np.linalg.inv(H_total)
        h0, w0 = self.original_bgr.shape[:2]
        warped = cv2.warpPerspective(img, H_inv, (w0, h0), flags=cv2.INTER_CUBIC)
        if mode == "canvas":
            canvas = np.full_like(self.original_bgr, background_color, dtype=np.uint8)
            mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped.ndim == 3 else warped
            _, m = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            if warped.ndim == 2:
                warped_bgr = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
            else:
                warped_bgr = warped
            canvas[m > 0] = warped_bgr[m > 0]
            return canvas
        else:
            base = self.original_bgr.copy()
            if warped.ndim == 2:
                warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
            mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            _, m = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            base[m > 0] = warped[m > 0]
            return base

    # ---- Internal: normalization ---------------------------------------------

    def _normalize_input(self, image: ImageInput) -> ArrayLike:
        if isinstance(image, str):
            arr = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise FileNotFoundError(f"Cannot read image from path: {image}")
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            return _ensure_uint8(arr)
        if isinstance(image, Image.Image):
            return _ensure_uint8(_pil_to_cv2(image))
        if isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            return _ensure_uint8(arr.copy())
        raise TypeError("Unsupported input type. Use filename, PIL.Image.Image, or numpy.ndarray.")

    # ---- Internal: execution --------------------------------------------------

    def _execute_step(self, step: StepType) -> None:
        ttype = step.get("type")
        if ttype not in self.SUPPORTED_TYPES:
            raise TransformError(f"Unknown transformation type: {ttype}")

        if ttype in self.DETECTION_TYPES:
            self._run_detection(step)
            # keep details record for detection (regions visualization or crops)
            if self.keep_details:
                crops = self._extract_region_crops(self.image, self.regions, self.region_mode)
                self._results_per_step.append(crops)
                self.history_region_steps.append((step.get("name", ttype), crops))
            else:
                self._results_per_step.append(None)
            return

        # Decide execution scope
        if self.bbox_mode:
            # Apply per region
            region_results = self._apply_per_region(step)
            if self.keep_details:
                self._results_per_step.append(region_results)
                self.history_region_steps.append((step.get("name", ttype), region_results))
            else:
                self._results_per_step.append(None)
        else:
            # Apply on the whole image
            out = self._apply_whole(step)
            if self.keep_details:
                self._results_per_step.append(out.copy())
                self.history_image_steps.append((step.get("name", ttype), out.copy()))

    # ---- Internal: detection --------------------------------------------------

    @time_step
    def _run_detection(self, step: StepType) -> None:
        """
        Switches to region mode and populates self.regions.

        By default, this provides a simple connected-component based fallback to yield bounding boxes.
        Integrate actual detectors by replacing _detect_* methods.
        """
        det_type = step["type"]
        mode: RegionMode = step.get("kwargs", {}).get("region_mode", "bbox")
        self.region_mode = mode if mode in ("bbox", "polygon") else "bbox"
        self.bbox_mode = True

        # Placeholder routing (replace with real model inference as needed)
        if det_type == "EAST":
            regions = self._detect_fallback(self.image)
        elif det_type == "DB50":
            regions = self._detect_fallback(self.image)
        elif det_type == "DB18":
            regions = self._detect_fallback(self.image)
        elif det_type == "CRAFT":
            regions = self._detect_fallback(self.image)
        elif det_type == "PP-OCRv5":
            regions = self._detect_fallback(self.image)
        else:
            raise TransformError(f"Unsupported detection type: {det_type}")

        self.regions = regions

    def _detect_fallback(self, img: ArrayLike) -> List[BBox]:
        """
        Minimal fallback detector using connected components to yield candidate text boxes.
        Produces bounding boxes.
        """
        g = _to_gray(img)
        # Boost contrast a bit
        g = cv2.equalizeHist(g)
        # Otsu -> assume dark text
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if bw.mean() < 127:  # text bright? invert
            bw = 255 - bw
        # Close small gaps
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, _safe_kernel(3), iterations=1)

        contours, _ = cv2.findContours(255 - bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[BBox] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 10:
                continue
            aspect = w / float(h)
            if aspect < 0.2 or aspect > 25:
                continue
            boxes.append((x, y, w, h))
        # Sort left-to-right, top-to-bottom
        boxes.sort(key=lambda b: (b[1] // 10, b[0]))
        return boxes

    # ---- Internal: per-region execution --------------------------------------

    def _apply_per_region(self, step: StepType) -> List[ArrayLike]:
        """
        Apply a whole-image transform to each region crop independently, then paste back.
        Returns the list of transformed region crops.
        """
        crops = self._extract_region_crops(self.image, self.regions, self.region_mode)
        out_crops: List[ArrayLike] = []
        new_image = self.image.copy()

        for idx, (crop, region) in enumerate(zip(crops, self.regions)):
            temp_img = crop.copy()
            # Run the same transform but on temp_img
            temp = self._apply_whole(step, override_image=temp_img)
            out_crops.append(temp.copy())
            # Paste back
            if self.region_mode == "bbox":
                x, y, w, h = region  # type: ignore
                # Align sizes in case filtering changed dimensions
                th, tw = temp.shape[:2]
                if (tw, th) != (w, h):
                    temp = cv2.resize(temp, (w, h), interpolation=cv2.INTER_CUBIC)
                new_image[y:y + h, x:x + w] = temp if temp.ndim == 3 else cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
            else:
                # Polygon paste via mask
                poly: Polygon = region  # type: ignore
                mask = np.zeros(new_image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
                # Fit temp into polygon's bounding box and warp via minimal quad if available
                x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
                temp_resized = cv2.resize(temp, (w, h), interpolation=cv2.INTER_CUBIC)
                patch = np.zeros_like(new_image)
                roi = temp_resized if temp_resized.ndim == 3 else cv2.cvtColor(temp_resized, cv2.COLOR_GRAY2BGR)
                patch[y:y + h, x:x + w] = roi
                new_image[mask > 0] = patch[mask > 0]

        self.image = new_image
        return out_crops

    def _extract_region_crops(
        self,
        img: ArrayLike,
        regions: List[Region],
        mode: RegionMode,
    ) -> List[ArrayLike]:
        crops: List[ArrayLike] = []
        if mode == "bbox":
            for (x, y, w, h) in regions:  # type: ignore
                crops.append(img[y:y + h, x:x + w].copy())
        else:
            for poly in regions:  # type: ignore
                x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
                crops.append(img[y:y + h, x:x + w].copy())
        return crops

    # ---- Internal: whole-image transforms ------------------------------------

    @time_step
    def _apply_whole(self, step: StepType, override_image: Optional[ArrayLike] = None) -> ArrayLike:
        img = self.image if override_image is None else override_image
        ttype = step["type"]
        kw = step.get("kwargs", {}) or {}

        # Map each type to a handler
        if ttype == "noop":
            out = img.copy()

        elif ttype == "resize_to_dpi":
            out = self._resize_to_dpi(img, **kw)

        elif ttype == "resize_to_char_height":
            out = self._resize_to_char_height(img, **kw)

        elif ttype == "resize_scale":
            out = self._resize_scale(img, **kw)

        elif ttype == "grayscale":
            out = _to_gray(img)

        elif ttype == "invert_if_white_on_dark":
            out = _invert_if_white_on_dark(_to_gray(img))

        elif ttype == "equalize_hist":
            gray = _to_gray(img)
            out = cv2.equalizeHist(gray)

        elif ttype == "clahe":
            out = self._clahe(img, **kw)

        elif ttype == "gaussian_blur":
            out = self._gaussian_blur(img, **kw)

        elif ttype == "median_blur":
            out = self._median_blur(img, **kw)

        elif ttype == "bilateral_filter":
            out = self._bilateral(img, **kw)

        elif ttype == "nl_means":
            out = self._nl_means(img, **kw)

        elif ttype == "otsu":
            out = self._otsu(img)

        elif ttype == "otsu_after_gaussian":
            out = self._otsu_after_gaussian(img, **kw)

        elif ttype == "niblack":
            out = self._niblack(img, **kw)

        elif ttype == "sauvola":
            out = self._sauvola(img, **kw)

        elif ttype == "global_threshold":
            out = self._global_threshold(img, **kw)

        elif ttype == "adaptive_mean":
            out = self._adaptive_threshold(img, method="mean", **kw)

        elif ttype == "adaptive_gaussian":
            out = self._adaptive_threshold(img, method="gaussian", **kw)

        elif ttype == "try_all_threshold":
            out = self._try_all_threshold(img)

        elif ttype == "deskew_moments":
            out = self._deskew_moments(img, **kw)

        elif ttype == "deskew_hough":
            out = self._deskew_hough(img, **kw)

        elif ttype == "perspective_warp":
            out = self._perspective_warp(img, **kw)

        elif ttype == "dilate":
            out = self._morph(img, op="dilate", **kw)

        elif ttype == "erode":
            out = self._morph(img, op="erode", **kw)

        elif ttype == "open":
            out = self._morph(img, op="open", **kw)

        elif ttype == "close":
            out = self._morph(img, op="close", **kw)

        elif ttype == "skeletonize":
            out = self._skeletonize(img)

        elif ttype == "canny_dilate":
            out = self._canny_dilate(img, **kw)

        elif ttype == "mser_enhance":
            out = self._mser_enhance(img, **kw)

        elif ttype == "unsharp_mask":
            out = self._unsharp(img, **kw)

        elif ttype == "add_border":
            out = self._add_border(img, **kw)

        elif ttype == "remove_border":
            out = self._remove_border(img, **kw)

        elif ttype == "extract_channel":
            out = self._extract_channel(img, **kw)

        elif ttype == "local_equalize_adapthist":
            out = self._local_adapthist(img, **kw)

        elif ttype == "connected_components":
            out = self._connected_components(img, **kw)

        elif ttype == "kasar_text_extraction":
            out = self._kasar(img, **kw)

        else:
            raise TransformError(f"Unknown transformation type: {ttype}")

        # Update current image and transform chain if whole-image call
        if override_image is None:
            self.image = out if out.ndim == 3 else cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        return out

    # ---- Transform implementations -------------------------------------------

    def _resize_to_dpi(
        self,
        img: ArrayLike,
        target_dpi: int = 300,
        assumed_screen_dpi: int = 96,
        interpolation: str = "cubic",
    ) -> ArrayLike:
        """
        Resize image assuming original physical size derived from screen DPI to target DPI.
        """
        interp = self._cv2_interp(interpolation)
        scale = target_dpi / float(assumed_screen_dpi)
        out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)
        # Update homography
        H = _homography_from_resize((img.shape[0], img.shape[1]), (out.shape[0], out.shape[1]))
        self._push_transform("resize_to_dpi", H)
        return out

    def _resize_to_char_height(
        self,
        img: ArrayLike,
        target_char_height: int = 26,
        min_scale: float = 0.5,
        max_scale: float = 4.0,
        interpolation: str = "cubic",
    ) -> ArrayLike:
        """
        Resize image so that median character height ~ target_char_height.
        If estimation fails, leaves image unchanged.
        """
        gray = _to_gray(img)
        est = _estimate_char_height(gray)
        if est is None or est <= 0:
            return img
        scale = float(target_char_height) / float(est)
        scale = float(np.clip(scale, min_scale, max_scale))
        interp = self._cv2_interp(interpolation)
        out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)
        H = _homography_from_resize((img.shape[0], img.shape[1]), (out.shape[0], out.shape[1]))
        self._push_transform("resize_to_char_height", H)
        return out

    def _resize_scale(self, img: ArrayLike, fx: float = 1.0, fy: Optional[float] = None,
                      interpolation: str = "cubic") -> ArrayLike:
        """Resize by explicit scales."""
        fy = fy if fy is not None else fx
        interp = self._cv2_interp(interpolation)
        out = cv2.resize(img, None, fx=fx, fy=fy, interpolation=interp)
        H = _homography_from_resize((img.shape[0], img.shape[1]), (out.shape[0], out.shape[1]))
        self._push_transform("resize_scale", H)
        return out

    def _clahe(self, img: ArrayLike, clip_limit: float = 2.0, tile_grid_size: int = 8) -> ArrayLike:
        gray = _to_gray(img)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        return clahe.apply(gray)

    def _gaussian_blur(self, img: ArrayLike, ksize: int = 3, sigma: float = 0.0) -> ArrayLike:
        ksize = max(1, int(ksize))
        if ksize % 2 == 0:
            ksize += 1
        return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)

    def _median_blur(self, img: ArrayLike, ksize: int = 3) -> ArrayLike:
        ksize = max(1, int(ksize))
        if ksize % 2 == 0:
            ksize += 1
        if img.ndim == 2:
            return cv2.medianBlur(img, ksize)
        # Apply channel-wise to avoid color shift
        channels = cv2.split(img)
        channels = [cv2.medianBlur(c, ksize) for c in channels]
        return cv2.merge(channels)

    def _bilateral(self, img: ArrayLike, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> ArrayLike:
        return cv2.bilateralFilter(img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    def _nl_means(self, img: ArrayLike, h: float = 10.0, template_window_size: int = 7,
                  search_window_size: int = 21) -> ArrayLike:
        if img.ndim == 2:
            return cv2.fastNlMeansDenoising(img, None, h=h, templateWindowSize=template_window_size,
                                            searchWindowSize=search_window_size)
        return cv2.fastNlMeansDenoisingColored(img, None, h=h, hColor=h,
                                               templateWindowSize=template_window_size,
                                               searchWindowSize=search_window_size)

    def _otsu(self, img: ArrayLike) -> ArrayLike:
        gray = _to_gray(img)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th

    def _otsu_after_gaussian(self, img: ArrayLike, ksize: int = 3, sigma: float = 0.0) -> ArrayLike:
        gray = _to_gray(img)
        blur = self._gaussian_blur(gray, ksize=ksize, sigma=sigma)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th

    def _niblack(self, img: ArrayLike, window_size: int = 25, k: float = -0.2) -> ArrayLike:
        gray = _to_gray(img)
        if not SKIMAGE_AVAILABLE:
            raise TransformError("Niblack requires scikit-image.")
        thresh = filters.threshold_niblack(gray, window_size=window_size, k=k)
        return (gray > thresh).astype(np.uint8) * 255

    def _sauvola(self, img: ArrayLike, window_size: int = 25, k: float = 0.2) -> ArrayLike:
        gray = _to_gray(img)
        if not SKIMAGE_AVAILABLE:
            raise TransformError("Sauvola requires scikit-image.")
        thresh = filters.threshold_sauvola(gray, window_size=window_size, k=k)
        return (gray > thresh).astype(np.uint8) * 255

    def _global_threshold(self, img: ArrayLike, thresh: int = 127, maxval: int = 255,
                          invert: bool = False) -> ArrayLike:
        gray = _to_gray(img)
        tflag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, out = cv2.threshold(gray, thresh, maxval, tflag)
        return out

    def _adaptive_threshold(self, img: ArrayLike, method: Literal["mean", "gaussian"] = "mean",
                            block_size: int = 31, C: int = 10, invert: bool = False) -> ArrayLike:
        gray = _to_gray(img)
        if block_size % 2 == 0:
            block_size += 1
        method_flag = cv2.ADAPTIVE_THRESH_MEAN_C if method == "mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        out = cv2.adaptiveThreshold(gray, 255, method_flag,
                                    cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY,
                                    block_size, C)
        return out

    def _try_all_threshold(self, img: ArrayLike) -> ArrayLike:
        if not SKIMAGE_AVAILABLE:
            raise TransformError("try_all_threshold requires scikit-image.")
        gray = _to_gray(img)
        # Use skimage to compute Otsu as a representative best; display montage is not needed here
        val = filters.threshold_otsu(gray)
        return (gray > val).astype(np.uint8) * 255

    def _deskew_moments(self, img: ArrayLike, max_angle: float = 15.0) -> ArrayLike:
        gray = _to_gray(img)
        # Binarize
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if bw.mean() < 127:
            bw = 255 - bw
        coords = np.column_stack(np.where(255 - bw > 0))
        if coords.size == 0:
            return img
        rect = cv2.minAreaRect(coords.astype(np.float32))
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) > max_angle:
            return img
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        # Upgrade to homography
        H = np.array([[M[0, 0], M[0, 1], M[0, 2]],
                      [M[1, 0], M[1, 1], M[1, 2]],
                      [0, 0, 1]], dtype=np.float64)
        self._push_transform("deskew_moments", H)
        return out

    def _deskew_hough(self, img: ArrayLike, max_angle: float = 20.0) -> ArrayLike:
        gray = _to_gray(img)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        if lines is None:
            return img
        angles = []
        for rho_theta in lines[:100]:
            rho, theta = rho_theta[0]
            angle = (theta - np.pi / 2) * (180.0 / np.pi)
            if abs(angle) <= max_angle:
                angles.append(angle)
        if not angles:
            return img
        angle = float(np.median(angles))
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        H = np.array([[M[0, 0], M[0, 1], M[0, 2]],
                      [M[1, 0], M[1, 1], M[1, 2]],
                      [0, 0, 1]], dtype=np.float64)
        self._push_transform("deskew_hough", H)
        return out

    def _perspective_warp(
        self,
        img: ArrayLike,
        src_quad: Optional[List[Tuple[float, float]]] = None,
        dst_size: Optional[Tuple[int, int]] = None,
    ) -> ArrayLike:
        """
        Apply perspective warp using a source quadrilateral to a rectified rectangle of dst_size (w,h).
        """
        if src_quad is None or dst_size is None:
            # Nothing to do
            return img
        src = np.array(src_quad, dtype=np.float32)
        w, h = dst_size
        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_CUBIC)
        self._push_transform("perspective_warp", H)
        return out

    def _morph(self, img: ArrayLike, op: Literal["dilate", "erode", "open", "close"], ksize: int = 3,
               iterations: int = 1) -> ArrayLike:
        kernel = _safe_kernel(ksize)
        if op == "dilate":
            return cv2.dilate(img, kernel, iterations=iterations)
        if op == "erode":
            return cv2.erode(img, kernel, iterations=iterations)
        if op == "open":
            return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
        if op == "close":
            return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        raise TransformError(f"Unknown morphology op: {op}")

    def _skeletonize(self, img: ArrayLike) -> ArrayLike:
        gray = _to_gray(img)
        if not SKIMAGE_AVAILABLE:
            raise TransformError("Skeletonize requires scikit-image.")
        bw = (gray > filters.threshold_otsu(gray)).astype(np.uint8)
        skel = morphology.skeletonize(bw > 0)
        return (skel.astype(np.uint8)) * 255

    def _canny_dilate(self, img: ArrayLike, low: int = 50, high: int = 150,
                      ksize: int = 3, iterations: int = 1) -> ArrayLike:
        gray = _to_gray(img)
        edges = cv2.Canny(gray, low, high)
        kernel = _safe_kernel(ksize)
        return cv2.dilate(edges, kernel, iterations=iterations)

    def _mser_enhance(self, img: ArrayLike, delta: int = 5, min_area: int = 60, max_area: int = 14400) -> ArrayLike:
        gray = _to_gray(img)
        mser = cv2.MSER_create(_delta=delta, _min_area=min_area, _max_area=max_area)
        regions, _ = mser.detectRegions(gray)
        mask = np.zeros_like(gray)
        for p in regions:
            cv2.fillPoly(mask, [p], 255)
        # Use mask to boost text regions
        boosted = gray.copy()
        boosted[mask > 0] = cv2.equalizeHist(boosted[mask > 0])
        return boosted

    def _unsharp(self, img: ArrayLike, radius: float = 1.0, amount: float = 1.5) -> ArrayLike:
        if SKIMAGE_AVAILABLE:
            from skimage.filters import unsharp_mask  # type: ignore
            if img.ndim == 2:
                us = unsharp_mask(img, radius=radius, amount=amount)
                return _ensure_uint8(us)
            # Convert to RGB for skimage
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            us = unsharp_mask(rgb, radius=radius, amount=amount, channel_axis=-1)
            bgr = cv2.cvtColor(_ensure_uint8(us), cv2.COLOR_RGB2BGR)
            return bgr
        # OpenCV fallback
        gaussian = cv2.GaussianBlur(img, (0, 0), radius)
        return cv2.addWeighted(img, 1 + amount, gaussian, -amount, 0)

    def _add_border(
        self,
        img: ArrayLike,
        top: int = 10,
        bottom: int = 10,
        left: int = 10,
        right: int = 10,
        color: Tuple[int, int, int] = (255, 255, 255),
    ) -> ArrayLike:
        out = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        H = _homography_from_border(top, bottom, left, right)
        self._push_transform("add_border", H)
        return out

    def _remove_border(self, img: ArrayLike, margin: int = 3) -> ArrayLike:
        gray = _to_gray(img)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2 * margin)
        h = min(img.shape[0] - y, h + 2 * margin)
        out = img[y:y + h, x:x + w].copy()
        # Crop is also a translation homography
        H = np.array([[1, 0, -float(x)],
                      [0, 1, -float(y)],
                      [0, 0, 1]], dtype=np.float64)
        self._push_transform("remove_border", H)
        return out

    def _extract_channel(self, img: ArrayLike, channel: Literal["r", "g", "b", "gray"] = "gray") -> ArrayLike:
        if channel == "gray":
            return _to_gray(img)
        b, g, r = cv2.split(img)
        if channel == "r":
            return r
        if channel == "g":
            return g
        if channel == "b":
            return b
        raise TransformError(f"Unknown channel: {channel}")

    def _local_adapthist(self, img: ArrayLike, clip_limit: float = 0.01, kernel_size: Optional[int] = None) -> ArrayLike:
        if not SKIMAGE_AVAILABLE:
            raise TransformError("Adaptive histogram equalization requires scikit-image.")
        gray = _to_gray(img)
        out = exposure.equalize_adapthist(gray, clip_limit=clip_limit, kernel_size=kernel_size)
        return _ensure_uint8(out)

    def _connected_components(self, img: ArrayLike, connectivity: int = 8, min_area: int = 10) -> ArrayLike:
        gray = _to_gray(img)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(255 - bw, connectivity=connectivity)
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area >= min_area:
                cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 1)
        return out

    def _kasar(self, img: ArrayLike, window_size: int = 15, morph_ksize: int = 3) -> ArrayLike:
        """
        Kasar algorithm inspired text extraction (simplified).
        Reference idea: stroke width emphasis and background suppression.
        """
        gray = _to_gray(img)
        blur = cv2.medianBlur(gray, 3)
        grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, _safe_kernel(3))
        # Local threshold
        if SKIMAGE_AVAILABLE:
            thr = filters.threshold_sauvola(grad, window_size=window_size)
            bw = (grad > thr).astype(np.uint8) * 255
        else:
            bw = self._adaptive_threshold(grad, method="gaussian", block_size=35, C=5)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, _safe_kernel(morph_ksize), iterations=1)
        return bw

    # ---- Helpers --------------------------------------------------------------

    def _cv2_interp(self, name: str) -> int:
        name = name.lower()
        return {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "area": cv2.INTER_AREA,
            "cubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }.get(name, cv2.INTER_CUBIC)

    def _push_transform(self, name: str, H_step: np.ndarray) -> None:
        """Accumulate homographies to track mapping from original -> current."""
        self._transform_chain.append(TransformRecord(name=name, H=H_step.copy()))
        self._current_H = H_step @ self._current_H


# ---- Example usage (for reference/testing) ------------------------------------
# steps = [
#     {"name": "Detect text", "type": "EAST", "kwargs": {"region_mode": "bbox"}},
#     {"name": "Per-box grayscale", "type": "grayscale", "kwargs": {}},
#     {"name": "Per-box CLAHE", "type": "clahe", "kwargs": {"clip_limit": 2.0, "tile_grid_size": 8}},
#     {"name": "Per-box Otsu", "type": "otsu", "kwargs": {}},
#     {"name": "Global deskew", "type": "deskew_moments", "kwargs": {"max_angle": 15}},
# ]
# ic = ImageCleanup("input.jpg", steps=steps, keep_details=True)
# final_img = ic.run()
# details = ic.return_details()
# ic.save_yaml("pipeline.yml", pipelines={"default": steps})
# cfg, loaded_steps = ImageCleanup.load_yaml("pipeline.yml", "default")
# back = ic.back_project(mode="overlay")
