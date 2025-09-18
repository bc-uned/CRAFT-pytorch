# image_cleanup.py

import cv2
import numpy as np
import yaml
import time
from PIL import Image
from skimage import filters
from typing import List, Dict, Union, Any, Tuple, Callable

# --- Helper Decorator for Performance Tracking ---

def time_it(func: Callable) -> Callable:
    """Decorator to measure and record the execution time of a transformation."""
    def wrapper(instance: 'ImageCleanup', image: np.ndarray, step: Dict, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = func(instance, image, step, **kwargs)
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        step['duration_ms'] = duration_ms
        return result
    return wrapper

# --- Main Class ---

class ImageCleanup:
    """
    A class for preprocessing and cleaning images for OCR recognition.

    This class provides a flexible pipeline to apply a series of transformations
    to an image, either globally or on specific regions of interest (bounding boxes).
    It supports various operations from resizing and denoising to text detection
    and perspective correction.
    """

    def __init__(self, image: Union[np.ndarray, Image.Image, str], keep_details: bool = False):
        """
        Initializes the ImageCleanup instance.

        Args:
            image (Union[np.ndarray, Image.Image, str]): The input image, which can be
                an OpenCV image (numpy.ndarray), a Pillow image, or a file path.
            keep_details (bool): If True, stores intermediate images for each step.
        """
        self.original_image: np.ndarray = self._normalize_image(image)
        self.current_image: np.ndarray = self.original_image.copy()
        self.keep_details: bool = keep_details
        self.config: Dict[str, Any] = {'keep_details': self.keep_details}

        # Pipeline and state
        self.steps: List[Dict] = []
        self.work_on_bboxes: bool = False
        self.bboxes: List[np.ndarray] = []
        self.processed_bboxes: List[np.ndarray] = []

        # History tracking
        self.whole_image_history: List[Dict] = []
        self.bbox_history: List[Dict] = []
        self.coordinate_history: List[Dict] = [] # For invertible transformations

        # Method mapping from type string to function
        self._transformations: Dict[str, Callable] = self._map_transformations()
        
        # Cache for heavy models (e.g., DNNs)
        self._models: Dict[str, Any] = {}

    def _normalize_image(self, image_input: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        """Converts any supported input format into a consistent OpenCV numpy.ndarray (BGR)."""
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise FileNotFoundError(f"Image file not found at: {image_input}")
            return img
        elif isinstance(image_input, Image.Image):
            # Convert PIL image (often RGB) to OpenCV's BGR format
            return cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            return image_input.copy()
        else:
            raise TypeError("Unsupported image type. Use numpy.ndarray, PIL.Image, or file path (str).")

    def _map_transformations(self) -> Dict[str, Callable]:
        """Maps transformation type strings to their corresponding methods for dispatch."""
        return {
            # Text Detection
            "east": self._detect_text_east,
            # Resizing
            "resize": self._resize,
            # Grayscale
            "grayscale": self._grayscale,
            # Contrast
            "histogram_equalization": self._contrast_histogram_equalization,
            "clahe": self._contrast_clahe,
            # Noise Reduction
            "gaussian_blur": self._denoise_gaussian,
            "median_blur": self._denoise_median,
            "bilateral_filter": self._denoise_bilateral,
            # Binarization
            "otsu_threshold": self._binarize_otsu,
            "adaptive_gaussian_threshold": self._binarize_adaptive_gaussian,
            "niblack_threshold": self._binarize_niblack,
            "sauvola_threshold": self._binarize_sauvola,
            # Skew & Perspective
            "deskew": self._correct_skew,
            # Morphological
            "morphology": self._morphology_op,
            # Border
            "add_border": self._add_border,
            # Back projection
            "back": self._back_project
        }
        # Note: This framework is extensible. Other transformations (CRAFT, DB50, Canny, etc.)
        # would be added to this dictionary following the same pattern.

    ##
    ## Core Pipeline Logic
    ##
    
    def process(self, steps: List[Dict]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Executes a pipeline of transformations sequentially.

        Args:
            steps (List[Dict]): A list of transformation dictionaries. Each dictionary
                                should contain "name", "type", and an optional "kwargs".

        Returns:
            The final processed image or, if in bounding box mode, a list of
            processed bounding box images.
        """
        self.steps = steps
        for step in self.steps:
            step_type = step.get("type")
            if not step_type:
                raise ValueError(f"Step '{step.get('name')}' is missing a 'type'.")

            transform_func = self._transformations.get(step_type)
            if not transform_func:
                raise ValueError(f"Unknown transformation type: '{step_type}'")

            kwargs = step.get("kwargs", {})
            
            if self.work_on_bboxes:
                # Apply transformation to each detected bounding box
                source_images = [self._crop_polygon(self.current_image, bbox) for bbox in self.bboxes] if not self.processed_bboxes else self.processed_bboxes
                
                self.processed_bboxes = [transform_func(img, step, **kwargs) for img in source_images]
                
                if self.keep_details:
                    self.bbox_history.append({"name": step["name"], "result": self.processed_bboxes})
            else:
                # Apply transformation to the whole image
                self.current_image = transform_func(self.current_image, step, **kwargs)
                if self.keep_details:
                    self.whole_image_history.append({"name": step["name"], "result": self.current_image})

        return self.processed_bboxes if self.work_on_bboxes else self.current_image

    def _crop_polygon(self, image: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        """Crops an image using a polygon, handling non-rectangular shapes."""
        rect = cv2.boundingRect(polygon)
        x, y, w, h = rect
        cropped = image[y:y+h, x:x+w].copy()
        
        # Create a mask to handle rotated rectangles or polygons
        pts = polygon - polygon.min(axis=0)
        mask = np.zeros(cropped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        
        # Apply mask to isolate the polygon area
        dst = cv2.bitwise_and(cropped, cropped, mask=mask)
        
        # Create a white background and combine it with the foreground
        bg = np.ones_like(cropped, np.uint8) * 255
        cv2.bitwise_not(bg, bg, mask=np.bitwise_not(mask))
        return bg + dst

    ##
    ## Public Utility Methods
    ##
    
    @staticmethod
    def download_all():
        """
        Downloads all required models and language packs.
        
        Note: This is a placeholder. A production implementation would download
        specific model files from official URLs. Libraries like EasyOCR and
        PaddleOCR handle their own downloads upon initialization.
        """
        print("🤖 Simulating model download...")
        try:
            import easyocr
            print("Initializing EasyOCR to download models (en)...")
            _ = easyocr.Reader(['en'])
            print("EasyOCR models are ready.")
        except ImportError:
            print("EasyOCR not installed. Skipping its models.")
        
        print("To use OpenCV's EAST model, download 'frozen_east_text_detection.pb'.")
        print("Model download simulation complete.")

    def return_details(self) -> List[Dict]:
        """
        Returns the steps list with the results of each transformation.
        
        'keep_details' must have been set to True during initialization.
        """
        if not self.keep_details:
            print("Warning: 'keep_details' was False. No intermediate results were stored.")
            return self.steps
        
        whole_idx, bbox_idx = 0, 0
        returned_steps = []
        was_on_bbox_mode = False

        for step in self.steps:
            new_step = step.copy()
            is_detection_step = step['type'] in ["east"] # Expand with other detector types

            if was_on_bbox_mode:
                if bbox_idx < len(self.bbox_history):
                    new_step['result'] = self.bbox_history[bbox_idx]['result']
                    bbox_idx += 1
            else:
                if whole_idx < len(self.whole_image_history):
                    new_step['result'] = self.whole_image_history[whole_idx]['result']
                    whole_idx += 1
            
            returned_steps.append(new_step)
            
            if is_detection_step:
                was_on_bbox_mode = True
                
        return returned_steps

    def save(self, filepath: str, config_name: str):
        """Saves the current pipeline and configuration to a YAML file."""
        data_to_save = {config_name: {"config": self.config, "pipeline": self.steps}}
        try:
            with open(filepath, 'r') as f:
                existing_data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            existing_data = {}
        
        existing_data.update(data_to_save)
        
        with open(filepath, 'w') as f:
            yaml.dump(existing_data, f, sort_keys=False, indent=2)

    def load(self, filepath: str, config_name: str) -> List[Dict]:
        """Loads a pipeline and configuration from a YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        config_data = data.get(config_name)
        if not config_data:
            raise ValueError(f"Configuration '{config_name}' not found in '{filepath}'.")
            
        self.config = config_data.get("config", {})
        self.steps = config_data.get("pipeline", [])
        
        self.keep_details = self.config.get('keep_details', False)
        print(f"Loaded pipeline '{config_name}' with {len(self.steps)} steps.")
        return self.steps

    ##
    ## Transformation Implementations (by Category)
    ##
    
    # --- Text Detection ---
    @time_it
    def _detect_text_east(self, image: np.ndarray, step: Dict, **kwargs) -> np.ndarray:
        """Detects text using OpenCV's EAST model and switches to bbox mode."""
        net = self._models.get("east_net")
        if not net:
            model_path = kwargs.get("model_path", "frozen_east_text_detection.pb")
            try:
                net = cv2.dnn.readNet(model_path)
                self._models["east_net"] = net
            except cv2.error:
                raise FileNotFoundError(f"EAST model not found at '{model_path}'. Call download_all() or provide a valid path.")
        
        h, w = image.shape[:2]
        new_w, new_h = 320, 320 # EAST requires multiples of 32
        r_w, r_h = w / new_w, h / new_h

        blob = cv2.dnn.blobFromImage(image, 1.0, (new_w, new_h), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

        # Decode EAST predictions to get rotated bounding boxes
        (num_rows, num_cols) = scores.shape[2:4]
        rects, confidences = [], []
        for y in range(num_rows):
            scores_data, geo_data = scores[0, 0, y], geometry[0, :, y]
            for x in range(num_cols):
                if scores_data[x] < kwargs.get("min_confidence", 0.5):
                    continue
                
                offset_x, offset_y = x * 4.0, y * 4.0
                angle = geo_data[4][x]
                h_box, w_box = geo_data[0][x] + geo_data[2][x], geo_data[1][x] + geo_data[3][x]
                
                end_x = int(offset_x + (np.cos(angle) * geo_data[1][x]) + (np.sin(angle) * geo_data[2][x]))
                end_y = int(offset_y - (np.sin(angle) * geo_data[1][x]) + (np.cos(angle) * geo_data[2][x]))
                start_x, start_y = int(end_x - w_box), int(end_y - h_box)
                
                rects.append((start_x, start_y, end_x, end_y))
                confidences.append(float(scores_data[x]))
        
        # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(rects, confidences, kwargs.get("min_confidence", 0.5), kwargs.get("nms_threshold", 0.4))
        
        polygons = []
        if len(indices) > 0:
            for i in indices.flatten():
                (start_x, start_y, end_x, end_y) = rects[i]
                # Scale box back to original image size
                box = np.array([
                    [start_x * r_w, start_y * r_h],
                    [end_x * r_w, start_y * r_h],
                    [end_x * r_w, end_y * r_h],
                    [start_x * r_w, end_y * r_h]
                ]).astype(int)
                polygons.append(box)

        self.bboxes = polygons
        self.work_on_bboxes = True # Switch to bbox processing for subsequent steps
        print(f"EAST detected {len(self.bboxes)} text regions.")
        
        # This step modifies state but returns the original image for consistency
        return image

    # --- Resizing & Resolution ---
    @time_it
    def _resize(self, image: np.ndarray, step: Dict, **kwargs) -> np.ndarray:
        """Resizes an image using specified method and parameters."""
        h, w = image.shape[:2]
        scale_factor = kwargs.get("scale_factor")
        target_w, target_h = kwargs.get("width"), kwargs.get("height")
        
        if scale_factor:
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        elif target_w and target_h:
            new_w, new_h = target_w, target_h
        else:
            return image

        interpolation_map = {"bicubic": cv2.INTER_CUBIC, "area": cv2.INTER_AREA, "lanczos": cv2.INTER_LANCZOS4}
        interpolation = interpolation_map.get(kwargs.get("method", "lanczos").lower(), cv2.INTER_LANCZOS4)
        resized_img = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        
        self.coordinate_history.append({"type": "resize", "from_shape": (h, w), "to_shape": (new_h, new_w)})
        return resized_img

    # --- Color & Contrast ---
    @time_it
    def _grayscale(self, image: np.ndarray, step: Dict, **kwargs) -> np.ndarray:
        """Converts image to grayscale and optionally inverts it based on background color."""
        gray = self._ensure_grayscale(image)
        if kwargs.get("invert", False) and np.mean(gray) < 128: # Invert if background is dark
            gray = cv2.bitwise_not(gray)
        return gray

    @time_it
    def _contrast_histogram_equalization(self, image: np.ndarray, step: Dict, **kwargs) -> np.ndarray:
        """Applies global histogram equalization to enhance contrast."""
        if len(image.shape) > 2: # Color image
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else: # Grayscale
            return cv2.equalizeHist(image)

    @time_it
    def _contrast_clahe(self, image: np.ndarray, step: Dict, **kwargs) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
        clahe = cv2.createCLAHE(clipLimit=kwargs.get("clip_limit", 2.0), tileGridSize=kwargs.get("tile_grid_size", (8, 8)))
        if len(image.shape) > 2:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            return clahe.apply(image)

    # --- Noise Reduction ---
    @time_it
    def _denoise_gaussian(self, image: np.ndarray, step: Dict, **kwargs) -> np.ndarray:
        return cv2.GaussianBlur(image, kwargs.get("ksize", (5, 5)), 0)

    @time_it
    def _denoise_median(self, image: np.ndarray, step: Dict, **kwargs) -> np.ndarray:
        return cv2.medianBlur(image, kwargs.get("ksize", 5))

    @time_it
    def _denoise_bilateral(self, image: np.ndarray, step: Dict, **kwargs) -> np.ndarray:
        return cv2.bilateralFilter(image, kwargs.get("d", 9), kwargs.get("sigma_color", 75), kwargs.get("sigma_space", 75))
        
    # --- Binarization / Thresholding ---
    @time_it
    def _binarize_otsu(self, image: np.ndarray, step: Dict, **kwargs) -> np.ndarray:
        gray = self._ensure_grayscale(image)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    @time_it
    def _binarize_adaptive_gaussian(self, image: np.ndarray, step: Dict, **kwargs) -> np.ndarray:
        gray = self._ensure_grayscale(image)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, kwargs.get("block_size", 11), kwargs.get("C", 2))

    @time_it
    def _binarize_niblack(self, image: np.ndarray, step: Dict, **kwargs) -> np.ndarray:
        gray = self._ensure_grayscale(image)
        thresh_val = filters.threshold_niblack(gray, window_size=kwargs.get("window_size", 25), k=kwargs.get("k", 0.8))
        return (gray > thresh_val).astype(np.uint8) * 255

    @time_it
    def _binarize_sauvola(self, image: np.ndarray, step: Dict, **kwargs) -> np.ndarray:
        gray = self._ensure_grayscale(image)
        thresh_val = filters.threshold_sauvola(gray, window_size=kwargs.get("window_size", 25), k=kwargs.get("k", 0.2))
        return (gray > thresh_val).astype(np.uint8) * 255
    
    # --- Skew & Perspective Correction ---
    @time_it
    def _correct_skew(self, image: np.ndarray, step: Dict, **kwargs) -> np.ndarray:
        """Corrects text skew by rotating the image."""
        gray = self._ensure_grayscale(image)
        thresh = cv2.bitwise_not(self._binarize_otsu(gray, {}))
        
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        self.coordinate_history.append({"type": "affine", "matrix": M})
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
    # --- Morphological Operations ---
    @time_it
    def _morphology_op(self, image: np.ndarray, step: Dict, **kwargs) -> np.ndarray:
        """Applies a morphological operation like OPEN, CLOSE, DILATE, or ERODE."""
        op_map = {"ERODE": cv2.MORPH_ERODE, "DILATE": cv2.MORPH_DILATE, "OPEN": cv2.MORPH_OPEN, "CLOSE": cv2.MORPH_CLOSE}
        op_type = kwargs.get("op_type", "open").upper()
        if op_type not in op_map:
            raise ValueError(f"Unsupported morphology op_type: {op_type}")
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kwargs.get("ksize", (3, 3)))
        return cv2.morphologyEx(image, op_map[op_type], kernel)
        
    # --- Border & Margin Handling ---
    @time_it
    def _add_border(self, image: np.ndarray, step: Dict, **kwargs) -> np.ndarray:
        """Adds a border to the image."""
        size = kwargs.get("size", 10)
        bordered = cv2.copyMakeBorder(image, size, size, size, size, cv2.BORDER_CONSTANT, value=kwargs.get("color", [255, 255, 255]))
        self.coordinate_history.append({"type": "add_border", "size": size})
        return bordered

    # --- Shape & Coordinate Management ---
    @time_it
    def _back_project(self, image: np.ndarray, step: Dict, **kwargs) -> np.ndarray:
        """
        Projects results back to original image coordinates for visualization.
        This is a finalizer step and draws the initial detections on the original image.
        """
        print("Back projection: Visualizing final bboxes on original image.")
        output_image = self.original_image.copy()
        if self.bboxes:
            cv2.drawContours(output_image, self.bboxes, -1, (0, 255, 0), 2)
        return output_image
        
    # --- Helper Methods ---
    def _ensure_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Ensures an image is single-channel grayscale for processing."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image