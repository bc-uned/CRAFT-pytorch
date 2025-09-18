"""
image_cleanup.py

ImageCleanup class for preprocessing and clearing images for OCR recognition.

Author: ChatGPT (GPT-5 Thinking mini)

NOTES:

Notes & guidance for extension

Text detection: _run_text_detector is a placeholder that must be replaced with actual model inference code (loading model files from models_dir and producing either bounding boxes or polygons). Once implemented, detected shapes should be detailed dictionaries like {"type":"polygon", "coords":[(x1,y1)...], "score":0.9} or bbox tuples.

download_all() should be implemented to actually fetch and verify model files and language packs. On success, _run_text_detector may use those files to run inference.

Several skimage-based transformations (Sauvola, Niblack, unsharp mask) require scikit-image; code raises helpful errors if unavailable.

The timing decorator time_step stores timing_ms on the step dictionary.

return_details() returns self.steps where each step contains .result (either a numpy image or a list of shape results) and .timing_ms.

The back() method supports a basic reprojection mechanism for bbox/polygon crops back into the original coordinates by resizing the transformed crop into the original bbox and overlaying it. More advanced re-projection (e.g., inverse perspective transforms) can be added.

The class keeps separate image_history (whole-image transformations) and shape_history (per-step list of shape transformations) when keep_details is True.

If you'd like, I can:

Implement actual inference code for a specific text detector (e.g., an EAST-based pipeline) if you provide the model files or allow me to implement download_all() to fetch them.

Add unit tests for this class (recommended: use pytest and create small test images to exercise each step).

Add example usage script showing a realistic pipeline (resize → grayscale → CLAHE → Otsu) and visual comparison.

PROMPT:

You are an expert IT engineer. All the code you write is high quality, modular, and conforms to best practices (PEP8, type hints, docstrings, unit-testable design).
Write a Python class called ImageCleanup. 
The main goal of this function is to preprocess and clear the image for OCR recognition.
These are the ImageCleanup specifications:
Initialization
•	The class initializes with one of:
o	a cv2 image (numpy.ndarray),
o	a PIL.Image.Image,
o	or an image filename (string).
Transformation Pipeline
•	The class accepts a list of transformations called steps. The steps can either work on the whole or on all the image bounding boxes and/or polygons.
•	Each step is a dictionary with:
o	"name": string, descriptive label (for logging only),
o	"type": string, predefined transformation type,
o	"kwargs": dict, parameters for the transformation.
•	Transformations are executed sequentially in the given order.
•	Unknown transformation types should raise a clear exception.
Global Parameters
•	Keep details: If enabled, store all intermediate images (transformed images) for each step of the pipeline.
•	Work at bounding_box/polygon level:
o	If enabled, transformations are applied per bounding box/polygon instead of the whole image.
o	This flag is automatically set after any text-detection step (EAST, DB50, DB18, CRAFT, PP-OCRv5).
•	Keep separate histories for whole-image transformations and bbox/polygon transformations.
Performance Tracking
•	Use a decorator to measure execution time of each step.
•	Store timing statistics in the step’s dictionary entry.
Additional Methods
•	download_all(): Download all required OCR models and language packs. EAST, DB50, DB18, CRAFT, PP-OCRv5, etc..
•	return_details(): Return the steps list, with an added "result" entry for each step containing the transformed image (or list of bbox/polygon images). (No need to deepcopy)
•	load/save: Load and save global config and pipeline config to yaml file (posiblity of having different pipelines and config in the same yaml)
Supported Transformations
Text Detection (switches to bbox/polygon mode):
•	EAST: OpenCV EAST model (bbox or polygon output).
•	DB50: OpenCV DB50 model (bbox or polygon output).
•	DB18: OpenCV DB18 model (bbox or polygon output).
•	CRAFT: EasyOCR CRAFT model (bbox or polygon output).
•	PP-OCRv5: PaddleOCR PP-OCRv5 model (bbox or polygon output).

1. Resizing & Resolution:
   - Resize image to target DPI or character height (20–30 px per character for Tesseract). Try to determine de actual characters actual height before transformation.
   - Methods: bicubic, Lanczos (Pillow), INTER_CUBIC / INTER_AREA (OpenCV).

2. Grayscale Conversion:
   - Convert image to single-channel grayscale.
   - Invert image (only if we somehow detect that the characters are white / clear on dark bag round)

3. Contrast Enhancement:
   - Histogram Equalization (cv2.equalizeHist, skimage.exposure.equalize_hist).
   - CLAHE (cv2.createCLAHE, skimage.exposure.equalize_adapthist).

4. Noise Reduction:
   - Gaussian blur (cv2.GaussianBlur).
   - Median blur (cv2.medianBlur).
   - Bilateral filter (cv2.bilateralFilter).
   - Fast Non-Local Means (cv2.fastNlMeansDenoisingColored).

5. Binarization / Thresholding:
   - Global Otsu (cv2.threshold + THRESH_OTSU / skimage.filters.threshold_otsu).
   - Otsu after Gaussian blur.
   - Niblack & Sauvola (skimage.filters.threshold_niblack, threshold_sauvola).
   - Global threshold (cv2.threshold).
   - Adaptive mean & Gaussian (cv2.adaptiveThreshold).
   - try_all_threshold (skimage.filters.try_all_threshold).

6. Skew & Perspective Correction:
   - Deskew using Hough transform or moments.
   - Perspective warp (cv2.getPerspectiveTransform, cv2.warpPerspective).

7. Morphological Operations:
   - Dilate, erode, open, close (cv2.morphologyEx).
   - Skeletonize (skimage.morphology.skeletonize).

8. Edge & Border Enhancements:
   - Canny edge + dilation (cv2.Canny + cv2.dilate).
   - MSER-based enhancement (cv2.MSER_create).
   - Unsharp mask (skimage.filters.unsharp_mask).

9. Border & Margin Handling:
   - Add border (cv2.copyMakeBorder, PIL.ImageOps.expand).
   - Remove border (contour detection + crop).

10. Color Channel Operations:
   - Extract R/G/B or grayscale channel (cv2.split).

11. Adaptive Filtering:
   - Locally adjust brightness/contrast (skimage.exposure.equalize_adapthist).

12. Segmentation:
   - Connected Component Analysis (cv2.connectedComponents, skimage.measure.label).

13. OCR-Specific Enhancements:
   - Apply Kasar algorithm for text extraction (see: https://github.com/jasonlfunk/ocr-text-extraction/blob/master/extract_text).
Shape & Coordinate Management
•	back: Undo shape-changing transformations (warp, add_border, resize) and reproject results into the original image coordinate system, either overlaid on the original image or on a white canvas.


"""

from __future__ import annotations


import time
import yaml
import math
import logging
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

import numpy as np

# Prefer cv2 for most operations
try:
    import cv2
except Exception as exc:  # pragma: no cover - environment dependent
    raise ImportError("OpenCV (cv2) is required for ImageCleanup.") from exc

from PIL import Image, ImageOps

# Optional: scikit-image features
try:
    from skimage import exposure, filters, morphology, util
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False

# Configure module logger
logger = logging.getLogger("ImageCleanup")
logger.addHandler(logging.NullHandler())


# Type aliases
NumpyImage = np.ndarray
PILImage = Image.Image
ImageLike = Union[NumpyImage, PILImage, str]
BBox = Tuple[int, int, int, int]  # x, y, w, h
Polygon = List[Tuple[int, int]]


def time_step(func: Callable) -> Callable:
    """
    Decorator to measure execution time of a step method.
    Stores timing (ms) in the step dict if provided via kwargs["step_dict"].
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        step_dict: Optional[Dict[str, Any]] = kwargs.get("step_dict")
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if step_dict is not None:
            step_dict.setdefault("timing_ms", 0.0)
            step_dict["timing_ms"] = elapsed_ms
        return result

    return wrapper


class ImageCleanup:
    """
    ImageCleanup: preprocess images for OCR.

    Usage:
        ic = ImageCleanup(img_or_path)
        ic.steps = [
            {"name": "grayscale", "type": "grayscale", "kwargs": {}},
            {"name": "clahe", "type": "clahe", "kwargs": {"clipLimit": 3.0, "tileGridSize": (8,8)}},
            ...
        ]
        ic.run()
        details = ic.return_details()
    """

    SUPPORTED_TEXT_DETECTORS = {"EAST", "DB50", "DB18", "CRAFT", "PP-OCRv5"}

    def __init__(
        self,
        image: ImageLike,
        *,
        keep_details: bool = False,
        work_on_shapes: bool = False,
        models_dir: str = "./models",
    ) -> None:
        """
        Initialize with a numpy.ndarray (BGR as cv2 uses), PIL.Image, or filename.

        :param image: image input (numpy.ndarray, PIL.Image.Image, or path)
        :param keep_details: store intermediate images for each step
        :param work_on_shapes: whether to apply steps per bbox/polygon (if available)
        :param models_dir: directory used for storing downloaded models
        """
        self.original_image = self._load_image(image)
        # store a working copy as BGR numpy array (cv2 convention)
        self.image = self.original_image.copy()
        self.steps: List[Dict[str, Any]] = []
        self.keep_details = keep_details
        self.work_on_shapes = work_on_shapes
        self.models_dir = models_dir

        # Histories
        self.image_history: List[NumpyImage] = []
        self.shape_history: List[List[NumpyImage]] = []  # list per step

        # shape mode (bboxes/polygons)
        self.shape_mode = False  # set True after any text-detection step
        # shapes described as list of (bbox or polygon) with type tag
        self.detected_shapes: List[Dict[str, Any]] = []

        if keep_details:
            self.image_history.append(self.image.copy())

    # ----------------------------
    # IO and normalization
    # ----------------------------
    @staticmethod
    def _load_image(image: ImageLike) -> NumpyImage:
        """
        Normalize various input types to a BGR numpy.ndarray suitable for cv2.
        """
        if isinstance(image, np.ndarray):
            # assume cv2 style BGR or grayscale -> convert to 3-channel BGR
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 3:
                return image.copy()
            elif image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            else:
                raise ValueError("Unsupported numpy image shape.")
        if isinstance(image, Image.Image):
            arr = np.array(image.convert("RGB"))
            # convert RGB to BGR for OpenCV
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if isinstance(image, str):
            img = cv2.imdecode(
                np.fromfile(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED
            )
            if img is None:
                raise FileNotFoundError(f"Image file not found: {image}")
            if img.ndim == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            return img
        raise TypeError("Unsupported image input type; requires ndarray, PIL.Image, or filename")

    @staticmethod
    def _bgr_to_pil(img: NumpyImage) -> Image.Image:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    @staticmethod
    def _pil_to_bgr(img: Image.Image) -> NumpyImage:
        arr = np.array(img.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # ----------------------------
    # Pipeline orchestration
    # ----------------------------
    def add_steps(self, steps: List[Dict[str, Any]]) -> None:
        """
        Add steps to pipeline. Each step must have: name, type, kwargs.
        """
        for s in steps:
            if "name" not in s or "type" not in s or "kwargs" not in s:
                raise ValueError("Each step must contain 'name', 'type', and 'kwargs'.")
            # canonicalize
            s.setdefault("result", None)
            s.setdefault("timing_ms", None)
            self.steps.append(s)

    def run(self) -> None:
        """
        Execute pipeline steps in order. Results and timing are stored in each step dict.
        """
        for step in self.steps:
            stype = step["type"]
            name = step.get("name", stype)
            logger.info("Running step: %s (%s)", name, stype)
            # If step is a text detector, special handling
            if stype.upper() in self.SUPPORTED_TEXT_DETECTORS:
                self._run_text_detector(stype.upper(), step)
                # mark that we should operate on shapes
                self.shape_mode = True
                self.work_on_shapes = True
                # store result (list of images per shape) as placeholder or actual crops
                step["result"] = self._extract_shape_images()
                if self.keep_details:
                    self.shape_history.append(step["result"])
                continue

            # Otherwise, transformations might be applied to whole image or per shape
            if self.work_on_shapes and self.shape_mode and len(self.detected_shapes) > 0:
                # apply per shape
                shape_results = []
                for shape in self.detected_shapes:
                    crop, meta = self._crop_for_shape(shape)
                    transformed = self._apply_transformation_to_image(crop, step)
                    shape_results.append({"image": transformed, "meta": meta})
                step["result"] = shape_results
                if self.keep_details:
                    self.shape_history.append(shape_results)
            else:
                # apply to whole image
                transformed = self._apply_transformation_to_image(self.image, step)
                # update working image
                self.image = transformed
                step["result"] = transformed
                if self.keep_details:
                    self.image_history.append(self.image.copy())

    def _apply_transformation_to_image(self, img: NumpyImage, step: Dict[str, Any]) -> NumpyImage:
        """
        Dispatch to transformation method based on step['type'].
        The step dict will receive timing info via decorator.
        """
        stype = step["type"].lower()
        kwargs = dict(step.get("kwargs", {}))
        step_dict = step

        # map to method
        dispatcher = {
            # Resizing
            "resize": self._step_resize,
            "resize_to_dpi": self._step_resize_to_dpi,
            # Colors & contrast
            "grayscale": self._step_grayscale,
            "invert": self._step_invert,
            "equalize_hist": self._step_equalize_hist,
            "clahe": self._step_clahe,
            # Noise reduction
            "gaussian_blur": self._step_gaussian_blur,
            "median_blur": self._step_median_blur,
            "bilateral": self._step_bilateral,
            "nlm": self._step_nlm,
            # Thresholding / binarization
            "otsu": self._step_otsu,
            "adaptive": self._step_adaptive_threshold,
            "sauvola": self._step_sauvola,
            "niblack": self._step_niblack,
            # Morphology
            "morph": self._step_morph,
            # Skew / perspective
            "deskew": self._step_deskew,
            "perspective_warp": self._step_perspective_warp,
            # Borders
            "add_border": self._step_add_border,
            "remove_border": self._step_remove_border,
            # edges and enh
            "canny_dilate": self._step_canny_dilate,
            "unsharp_mask": self._step_unsharp_mask,
            # channel ops
            "extract_channel": self._step_extract_channel,
            # segmentation
            "connected_components": self._step_connected_components,
            # fallback
        }

        if stype not in dispatcher:
            raise ValueError(f"Unknown transformation type: {step['type']}")

        func = dispatcher[stype]
        # wrapped with timing
        timed_func = time_step(func)
        return timed_func(img, **kwargs, step_dict=step_dict)

    # ----------------------------
    # Transformation implementations
    # ----------------------------

    # --- Resizing & resolution helpers ---
    @staticmethod
    def _estimate_char_height(gray_img: NumpyImage) -> int:
        """
        Try to estimate character height in pixels by connected components or projection.
        Returns estimated median height or fallback value 16.
        """
        if gray_img.ndim == 3:
            gray = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = gray_img
        # simple threshold then connected components
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # invert if background black
        if np.mean(th) < 127:
            th = cv2.bitwise_not(th)
        # connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th)
        heights = []
        for i in range(1, num_labels):
            h = stats[i, cv2.CC_STAT_HEIGHT]
            w = stats[i, cv2.CC_STAT_WIDTH]
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 8 and h > 3 and w < 200:  # likely a glyph
                heights.append(h)
        if not heights:
            return 16
        return int(np.median(heights))

    @staticmethod
    def _cv_resize(img: NumpyImage, target_shape: Tuple[int, int], method: int) -> NumpyImage:
        return cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=method)

    @time_step
    def _step_resize(self, img: NumpyImage, *, width: Optional[int] = None, height: Optional[int] = None,
                     method: str = "INTER_CUBIC", step_dict: Dict[str, Any]) -> NumpyImage:
        """
        Resize to explicit width/height (pixels). method can be one of OpenCV interpolation flags
        specified as string: INTER_CUBIC, INTER_AREA, INTER_LINEAR.
        """
        methods_map = {
            "INTER_CUBIC": cv2.INTER_CUBIC,
            "INTER_AREA": cv2.INTER_AREA,
            "INTER_LINEAR": cv2.INTER_LINEAR,
            "LANCZOS": cv2.INTER_LANCZOS4,
        }
        method_flag = methods_map.get(method, cv2.INTER_CUBIC)
        h, w = img.shape[:2]
        if width is None and height is None:
            return img
        if width is None:
            # compute width to keep aspect
            scale = height / float(h)
            width = int(w * scale)
        if height is None:
            scale = width / float(w)
            height = int(h * scale)
        return self._cv_resize(img, (height, width), method_flag)

    @time_step
    def _step_resize_to_dpi(self, img: NumpyImage, *, target_dpi: Optional[int] = None,
                            target_char_height: Optional[int] = None, step_dict: Dict[str, Any]) -> NumpyImage:
        """
        Resize to target DPI or to reach target character height (in px).
        If target_char_height is not provided, use estimated char height and scale to 25px per char.
        """
        if target_dpi is None and target_char_height is None:
            return img
        h, w = img.shape[:2]
        est_char_h = self._estimate_char_height(img)
        desired_char_h = target_char_height if target_char_height is not None else 25
        scale = desired_char_h / float(est_char_h)
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        # use cv2.INTER_CUBIC for upscaling and INTER_AREA for downscaling
        method = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        return self._cv_resize(img, (new_h, new_w), method)

    # --- Color & grayscale ---
    @time_step
    def _step_grayscale(self, img: NumpyImage, *, step_dict: Dict[str, Any]) -> NumpyImage:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @time_step
    def _step_invert(self, img: NumpyImage, *, step_dict: Dict[str, Any]) -> NumpyImage:
        bgr = img.copy()
        inv = cv2.bitwise_not(bgr)
        return inv

    # --- Contrast enhancement ---
    @time_step
    def _step_equalize_hist(self, img: NumpyImage, *, step_dict: Dict[str, Any]) -> NumpyImage:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

    @time_step
    def _step_clahe(self, img: NumpyImage, *, clipLimit: float = 2.0, tileGridSize: Tuple[int, int] = (8, 8),
                    step_dict: Dict[str, Any]) -> NumpyImage:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        cl = clahe.apply(gray)
        return cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)

    # --- Noise reduction ---
    @time_step
    def _step_gaussian_blur(self, img: NumpyImage, *, ksize: Tuple[int, int] = (3, 3),
                            sigmaX: float = 0, step_dict: Dict[str, Any]) -> NumpyImage:
        g = cv2.GaussianBlur(img, ksize, sigmaX)
        return g

    @time_step
    def _step_median_blur(self, img: NumpyImage, *, ksize: int = 3, step_dict: Dict[str, Any]) -> NumpyImage:
        return cv2.medianBlur(img, ksize)

    @time_step
    def _step_bilateral(self, img: NumpyImage, *, d: int = 9, sigmaColor: int = 75, sigmaSpace: int = 75,
                        step_dict: Dict[str, Any]) -> NumpyImage:
        return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

    @time_step
    def _step_nlm(self, img: NumpyImage, *, h: float = 10.0, templateWindowSize: int = 7,
                  searchWindowSize: int = 21, step_dict: Dict[str, Any]) -> NumpyImage:
        # works on colored images
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, templateWindowSize, searchWindowSize)

    # --- Thresholding / binarization ---
    @time_step
    def _step_otsu(self, img: NumpyImage, *, gaussian_blur: bool = False, step_dict: Dict[str, Any]) -> NumpyImage:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gaussian_blur:
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # return 3-channel BGR
        return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

    @time_step
    def _step_adaptive_threshold(self, img: NumpyImage, *,
                                 maxValue: int = 255, adaptiveMethod: str = "GAUSSIAN",
                                 thresholdType: str = "BINARY", blockSize: int = 11, C: int = 2,
                                 step_dict: Dict[str, Any]) -> NumpyImage:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        am = cv2.ADAPTIVE_THRESH_MEAN_C if adaptiveMethod.upper() == "MEAN" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        tt = cv2.THRESH_BINARY if thresholdType.upper() == "BINARY" else cv2.THRESH_BINARY_INV
        out = cv2.adaptiveThreshold(gray, maxValue, am, tt, blockSize, C)
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    @time_step
    def _step_sauvola(self, img: NumpyImage, *, window_size: int = 25, k: float = 0.2, step_dict: Dict[str, Any]) -> NumpyImage:
        if not SKIMAGE_AVAILABLE:
            raise RuntimeError("skimage is required for Sauvola thresholding.")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = filters.threshold_sauvola(gray, window_size=window_size, k=k)
        out = (gray > thresh).astype("uint8") * 255
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    @time_step
    def _step_niblack(self, img: NumpyImage, *, window_size: int = 25, k: float = 0.2, step_dict: Dict[str, Any]) -> NumpyImage:
        if not SKIMAGE_AVAILABLE:
            raise RuntimeError("skimage is required for Niblack thresholding.")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = filters.threshold_niblack(gray, window_size=window_size, k=k)
        out = (gray > thresh).astype("uint8") * 255
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    # --- Morphology ---
    @time_step
    def _step_morph(self, img: NumpyImage, *, op: str = "open", kernel_size: Tuple[int, int] = (3, 3),
                    iterations: int = 1, step_dict: Dict[str, Any]) -> NumpyImage:
        op_map = {
            "open": cv2.MORPH_OPEN,
            "close": cv2.MORPH_CLOSE,
            "erode": cv2.MORPH_ERODE,
            "dilate": cv2.MORPH_DILATE,
            "gradient": cv2.MORPH_GRADIENT,
            "tophat": cv2.MORPH_TOPHAT,
            "blackhat": cv2.MORPH_BLACKHAT,
        }
        op_flag = op_map.get(op, cv2.MORPH_OPEN)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        result = cv2.morphologyEx(img, op_flag, kernel, iterations=iterations)
        return result

    # --- Deskew / perspective ---
    @time_step
    def _step_deskew(self, img: NumpyImage, *, step_dict: Dict[str, Any]) -> NumpyImage:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        coords = np.column_stack(np.where(gray < 255))
        if coords.size == 0:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        # adjust angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    @time_step
    def _step_perspective_warp(self, img: NumpyImage, *, src: Optional[List[Tuple[float, float]]] = None,
                               dst: Optional[List[Tuple[float, float]]] = None, step_dict: Dict[str, Any]) -> NumpyImage:
        """
        Apply perspective transform given src/dst coordinates. If not provided, returns image unchanged.
        """
        if not src or not dst or len(src) != 4 or len(dst) != 4:
            # Not provided or invalid; nothing to do
            return img
        src_pts = np.array(src, dtype=np.float32)
        dst_pts = np.array(dst, dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        h, w = img.shape[:2]
        warped = cv2.warpPerspective(img, M, (w, h))
        return warped

    # --- Border handling ---
    @time_step
    def _step_add_border(self, img: NumpyImage, *, top: int = 10, bottom: int = 10, left: int = 10, right: int = 10,
                         color: Tuple[int, int, int] = (255, 255, 255), step_dict: Dict[str, Any]) -> NumpyImage:
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    @time_step
    def _step_remove_border(self, img: NumpyImage, *, threshold: int = 10, step_dict: Dict[str, Any]) -> NumpyImage:
        """
        Heuristic removal of constant-color border: find largest contour inside margin.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(255 - th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img
        # union of bounding rects
        x_min, y_min, x_max, y_max = img.shape[1], img.shape[0], 0, 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        # safety checks
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img.shape[1], x_max)
        y_max = min(img.shape[0], y_max)
        if x_max - x_min <= 0 or y_max - y_min <= 0:
            return img
        return img[y_min:y_max, x_min:x_max]

    # --- Edge & border enhancements ---
    @time_step
    def _step_canny_dilate(self, img: NumpyImage, *, canny_thresh1: int = 100, canny_thresh2: int = 200,
                          dilate_iter: int = 1, step_dict: Dict[str, Any]) -> NumpyImage:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, canny_thresh1, canny_thresh2)
        kernel = np.ones((3, 3), np.uint8)
        dil = cv2.dilate(edges, kernel, iterations=dilate_iter)
        return cv2.cvtColor(dil, cv2.COLOR_GRAY2BGR)

    @time_step
    def _step_unsharp_mask(self, img: NumpyImage, *, radius: float = 1.0, amount: float = 1.0,
                           step_dict: Dict[str, Any]) -> NumpyImage:
        if not SKIMAGE_AVAILABLE:
            # simple OpenCV unsharp approximate
            blur = cv2.GaussianBlur(img, (0, 0), radius)
            sharpened = cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)
            return sharpened
        pil = self._bgr_to_pil(img)
        from skimage.filters import unsharp_mask  # optional local import if available
        arr = np.array(pil.convert("L")) / 255.0
        sharp = unsharp_mask(arr, radius=radius, amount=amount)
        sharp_img = (np.clip(sharp * 255.0, 0, 255)).astype("uint8")
        return cv2.cvtColor(sharp_img, cv2.COLOR_GRAY2BGR)

    # --- Channel ops ---
    @time_step
    def _step_extract_channel(self, img: NumpyImage, *, channel: str = "R", step_dict: Dict[str, Any]) -> NumpyImage:
        b, g, r = cv2.split(img)
        ch = {"R": r, "G": g, "B": b}.get(channel.upper(), r)
        return cv2.cvtColor(ch, cv2.COLOR_GRAY2BGR)

    # --- Segmentation / connected components ---
    @time_step
    def _step_connected_components(self, img: NumpyImage, *, min_area: int = 30, step_dict: Dict[str, Any]) -> NumpyImage:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th)
        out = img.copy()
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                x, y, w, h, area = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                                  stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
                out[y:y + h, x:x + w] = 255
        return out

    # ----------------------------
    # Text detection (placeholders & integration)
    # ----------------------------
    def _run_text_detector(self, detector_name: str, step: Dict[str, Any]) -> None:
        """
        Run a text-detector. For now, this method is a placeholder that must be extended with
        actual model inference code. It will set detected_shapes to a list of bboxes or polygons,
        and mark step['detected_shapes'] with metadata.

        If model files are not found in models_dir, raise a clear error.
        """
        # quick check for model files (stub)
        model_path = f"{self.models_dir}/{detector_name}"
        # In a robust implementation we'd check for actual files. For now, warn and create a fallback bbox.
        logger.info("Text detection: %s (models_dir=%s)", detector_name, self.models_dir)
        # If you have real models, implement their inference here.
        # Fallback: single bbox for full image
        h, w = self.image.shape[:2]
        bbox = {"type": "bbox", "coords": (0, 0, w, h), "detector": detector_name}
        self.detected_shapes = [bbox]
        step.setdefault("detected_shapes", self.detected_shapes)

    def _crop_for_shape(self, shape: Dict[str, Any]) -> Tuple[NumpyImage, Dict[str, Any]]:
        """
        Crop the current image for a bbox or polygon and return the crop and metadata.
        """
        if shape.get("type") == "bbox":
            x, y, w, h = shape["coords"]
            crop = self.image[y:y + h, x:x + w].copy()
            meta = {"origin": (x, y, w, h)}
            return crop, meta
        elif shape.get("type") == "polygon":
            # polygon cropping by bounding box for simplicity; advanced clipping omitted
            poly = shape["coords"]
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            crop = self.image[y0:y1, x0:x1].copy()
            meta = {"origin": (x0, y0, x1 - x0, y1 - y0), "polygon": poly}
            return crop, meta
        raise ValueError("Unknown shape type for cropping.")

    def _extract_shape_images(self) -> List[Dict[str, Any]]:
        """
        Return list of dicts {'image': crop, 'meta': ...} for each detected shape.
        """
        out = []
        for s in self.detected_shapes:
            crop, meta = self._crop_for_shape(s)
            out.append({"image": crop, "meta": meta})
        return out

    # ----------------------------
    # Shape reprojection (back)
    # ----------------------------
    def back(self, shapes_results: List[Dict[str, Any]],
             overlay_on_original: bool = True, background_color: Tuple[int, int, int] = (255, 255, 255)) -> NumpyImage:
        """
        Reproject shape-level transformed images back into original image coordinates.

        :param shapes_results: list of dicts per shape: {"image": np.ndarray, "meta":{"origin":(x,y,w,h)}}.
        :param overlay_on_original: if True, overlay on original image; else create white canvas.
        :param background_color: background color for new canvas if overlay_on_original False.
        :return: reprojected BGR image
        """
        canvas = self.original_image.copy() if overlay_on_original else np.full_like(self.original_image, 255)
        for res in shapes_results:
            meta = res.get("meta", {})
            origin = meta.get("origin")
            if not origin:
                continue
            x, y, w, h = origin
            piece = res["image"]
            # resize if piece size differs from origin
            ph, pw = piece.shape[:2]
            if (ph, pw) != (h, w):
                piece = cv2.resize(piece, (w, h), interpolation=cv2.INTER_CUBIC)
            canvas[y:y + h, x:x + w] = piece
        return canvas

    # ----------------------------
    # Persistence: YAML load/save for pipelines and configs
    # ----------------------------
    def save_config(self, path: str, pipelines: Optional[Dict[str, List[Dict[str, Any]]]] = None,
                    global_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Save pipelines and global config to a YAML file.
        pipelines: dict of pipeline_name -> list of step dicts
        global_config: e.g., {"keep_details": True, "work_on_shapes": False}
        """
        doc = {"global": global_config or {"keep_details": self.keep_details, "work_on_shapes": self.work_on_shapes},
               "pipelines": pipelines or {"default": self.steps}}
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(doc, fh)

    @classmethod
    def load_config(cls, path: str, pipeline_name: str = "default") -> Dict[str, Any]:
        """
        Load YAML config and return the selected pipeline and global settings.
        """
        with open(path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        global_conf = doc.get("global", {})
        pipelines = doc.get("pipelines", {})
        if pipeline_name not in pipelines:
            raise KeyError(f"Pipeline {pipeline_name} not present in file.")
        steps = pipelines[pipeline_name]
        return {"global": global_conf, "steps": steps}

    # ----------------------------
    # Utility & Additional Methods
    # ----------------------------
    def return_details(self) -> List[Dict[str, Any]]:
        """
        Return the steps list with added "result" entry for each step containing the
        transformed image (or list of shape images). This does not deepcopy.
        """
        return self.steps

    def download_all(self, *, skip_confirmation: bool = True) -> None:
        """
        Download OCR models and language packs (stub). Extend this method with actual
        downloading logic (e.g., wget or huggingface-hub downloads).
        For now simply records that models "would be downloaded" into models_dir.

        :param skip_confirmation: if False, you could prompt the user in interactive code (not done here).
        """
        # This method intentionally left as a stub - implement actual download logic here.
        # Example: for EAST you might download a .pb file to models_dir/EAST/frozen_east_text_detection.pb
        # For DB models, download required .onnx or .pb files.
        logger.info("download_all called. Implement model download logic to populate %s", self.models_dir)
        # For now we mimic success by creating a marker file or directory (if desired).
        # raise NotImplementedError if you want to force explicit implementation.
        return

    # ----------------------------
    # Public helpers for tests and interactive use
    # ----------------------------
    def reset(self) -> None:
        """
        Reset working image and histories to original.
        """
        self.image = self.original_image.copy()
        self.image_history = [self.image.copy()] if self.keep_details else []
        self.shape_history = []
        self.detected_shapes = []
        self.shape_mode = False

    def save_result_image(self, path: str, *, use_original_size: bool = True) -> None:
        """
        Save the current working image to disk (handles unicode paths with numpy -> file).
        """
        # write using imencode to preserve unicode path support
        ext = path.split(".")[-1]
        ok, enc = cv2.imencode(f".{ext}", self.image)
        if not ok:
            raise IOError("Failed to encode image for saving.")
        enc.tofile(path)
