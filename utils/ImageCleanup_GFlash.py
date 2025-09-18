import time
import functools
import os
import yaml
import logging
from typing import List, Dict, Any, Union, Optional, Callable
import numpy as np
import cv2
from PIL import Image
from skimage.exposure import equalize_hist, equalize_adapthist
from skimage.filters import (
    threshold_otsu, threshold_niblack, threshold_sauvola, try_all_threshold)
from skimage.morphology import skeletonize
from skimage.measure import label
from scipy.ndimage import rotate

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageCleanup:
    """
    A class to preprocess and clean an image for OCR recognition.

    The pipeline handles various transformations and can operate on the whole image
    or on detected bounding boxes/polygons. It also tracks performance and
    stores intermediate results.
    """

    SUPPORTED_TRANSFORMATIONS = {
        "text_detection_east": "Text Detection",
        "text_detection_db50": "Text Detection",
        "text_detection_db18": "Text Detection",
        "text_detection_craft": "Text Detection",
        "text_detection_ppocv5": "Text Detection",
        "resize": "Resizing & Resolution",
        "grayscale": "Grayscale Conversion",
        "invert": "Grayscale Conversion",
        "equalize_hist": "Contrast Enhancement",
        "clahe": "Contrast Enhancement",
        "gaussian_blur": "Noise Reduction",
        "median_blur": "Noise Reduction",
        "bilateral_filter": "Noise Reduction",
        "fast_nl_means_denoising": "Noise Reduction",
        "threshold_otsu": "Binarization / Thresholding",
        "threshold_otsu_gaussian": "Binarization / Thresholding",
        "threshold_niblack": "Binarization / Thresholding",
        "threshold_sauvola": "Binarization / Thresholding",
        "threshold_global": "Binarization / Thresholding",
        "adaptive_threshold_mean": "Binarization / Thresholding",
        "adaptive_threshold_gaussian": "Binarization / Thresholding",
        "try_all_thresholds": "Binarization / Thresholding",
        "deskew": "Skew & Perspective Correction",
        "perspective_warp": "Skew & Perspective Correction",
        "morphology": "Morphological Operations",
        "skeletonize": "Morphological Operations",
        "canny_edge_dilate": "Edge & Border Enhancements",
        "unsharp_mask": "Edge & Border Enhancements",
        "add_border": "Border & Margin Handling",
        "remove_border": "Border & Margin Handling",
        "extract_channel": "Color Channel Operations",
        "adaptive_filtering": "Adaptive Filtering",
        "connected_components": "Segmentation",
        "kasar_algorithm": "OCR-Specific Enhancements",
    }
    
    TEXT_DETECTION_STEPS = [
        "text_detection_east", "text_detection_db50", "text_detection_db18",
        "text_detection_craft", "text_detection_ppocv5"
    ]

    def __init__(self, image_input: Union[np.ndarray, Image.Image, str], keep_details: bool = False):
        """
        Initializes the ImageCleanup class with an image.

        Args:
            image_input (Union[np.ndarray, Image.Image, str]): The input image,
                which can be a cv2 image (numpy.ndarray), a PIL Image, or a
                file path (string).
            keep_details (bool): If True, all intermediate images are stored.
        """
        self.keep_details = keep_details
        self.steps: List[Dict[str, Any]] = []
        self.image = self._normalize_image(image_input)
        self.original_image = self.image.copy()
        self.bounding_boxes: Optional[List[Dict[str, Any]]] = None
        self.work_on_bboxes = False
        self.history: Dict[str, List[np.ndarray]] = {"whole_image": [], "bboxes": []}

    def _normalize_image(self, image_input: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        """
        Normalizes the input image to a consistent numpy.ndarray format.

        Args:
            image_input (Union[np.ndarray, Image.Image, str]): The input image.

        Returns:
            np.ndarray: The normalized image as a NumPy array.

        Raises:
            TypeError: If the input type is not supported.
            FileNotFoundError: If the input is a string and the file does not exist.
        """
        if isinstance(image_input, np.ndarray):
            return image_input
        elif isinstance(image_input, Image.Image):
            return cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Could not read image from path: {image_input}")
            return img
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def _timing_decorator(self, func: Callable) -> Callable:
        """
        Decorator to measure the execution time of a function.
        """
        @functools.wraps(func)
        def wrapper(step: Dict[str, Any], *args, **kwargs) -> Any:
            start_time = time.time()
            result = func(step, *args, **kwargs)
            end_time = time.time()
            step['timing'] = end_time - start_time
            return result
        return wrapper

    def _get_transform_method(self, step: Dict[str, Any]) -> Callable:
        """
        Returns the appropriate transformation method based on the step type.

        Args:
            step (Dict[str, Any]): The step dictionary.

        Returns:
            Callable: The method to execute.

        Raises:
            ValueError: If the transformation type is unknown.
        """
        transform_type = step['type']
        method_name = f"_{transform_type.replace('-', '_').replace(' ', '_').lower()}_transform"
        method = getattr(self, method_name, None)
        if not method:
            raise ValueError(f"Unknown transformation type: {transform_type}")
        return self._timing_decorator(method)
    
    def process_pipeline(self, steps: List[Dict[str, Any]]):
        """
        Executes a series of transformations on the image.

        Args:
            steps (List[Dict[str, Any]]): A list of transformation steps, each
                represented by a dictionary.
        """
        self.steps = steps
        for step in self.steps:
            logging.info(f"Executing step: {step['name']} ({step['type']})")
            
            try:
                transform_method = self._get_transform_method(step)
                
                if step['type'] in self.TEXT_DETECTION_STEPS:
                    self.work_on_bboxes = True
                    self.bounding_boxes = transform_method(step)
                    if self.keep_details:
                        self.history["bboxes"] = [
                            self._apply_transform_to_bbox(self.image, box, step) for box in self.bounding_boxes
                        ]
                else:
                    if self.work_on_bboxes and self.bounding_boxes:
                        processed_bboxes = []
                        for box in self.bounding_boxes:
                            bbox_image = self._crop_bbox(self.image, box)
                            processed_bbox = transform_method(step, bbox_image)
                            processed_bboxes.append(processed_bbox)
                        self.image = self._reconstruct_image_from_bboxes(self.image, processed_bboxes, self.bounding_boxes)
                        
                        if self.keep_details:
                            self.history["bboxes"].append(processed_bboxes)
                    else:
                        self.image = transform_method(step, self.image)
                        if self.keep_details:
                            self.history["whole_image"].append(self.image.copy())
            
            except Exception as e:
                logging.error(f"Error in step '{step['name']}': {e}")
                # Optional: re-raise or handle gracefully

    def _resize_transform(self, step: Dict[str, Any], image: np.ndarray) -> np.ndarray:
        """
        Resizes the image based on DPI or character height.
        """
        kwargs = step.get('kwargs', {})
        target_dpi = kwargs.get('target_dpi')
        target_char_height = kwargs.get('target_char_height')
        interpolation_method = kwargs.get('interpolation', 'INTER_CUBIC')

        h, w = image.shape[:2]
        new_w, new_h = w, h

        if target_dpi:
            # Simple DPI scaling (requires known source DPI, not implemented here)
            # Placeholder: assume a simple scale factor
            scale = target_dpi / 96.0  # Common assumption
            new_w, new_h = int(w * scale), int(h * scale)
        elif target_char_height:
            # This is a complex problem, simplified here
            # A real implementation would require text detection first
            # Placeholder: assume average char height is proportional to image height
            avg_char_height = h / 50  # Arbitrary assumption
            scale = target_char_height / avg_char_height
            new_w, new_h = int(w * scale), int(h * scale)

        inter_code = getattr(cv2, interpolation_method.upper(), cv2.INTER_CUBIC)
        return cv2.resize(image, (new_w, new_h), interpolation=inter_code)

    def _grayscale_transform(self, step: Dict[str, Any], image: np.ndarray) -> np.ndarray:
        """
        Converts the image to grayscale.
        """
        if len(image.shape) > 2:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def _invert_transform(self, step: Dict[str, Any], image: np.ndarray) -> np.ndarray:
        """
        Inverts the colors of a grayscale image.
        """
        # A real implementation would detect if characters are light on a dark background
        return cv2.bitwise_not(image)
        
    def _equalize_hist_transform(self, step: Dict[str, Any], image: np.ndarray) -> np.ndarray:
        """
        Performs global histogram equalization.
        """
        if len(image.shape) > 2:
            image = self._grayscale_transform({}, image)
        return equalize_hist(image) * 255
    
    def _clahe_transform(self, step: Dict[str, Any], image: np.ndarray) -> np.ndarray:
        """
        Applies CLAHE for local contrast enhancement.
        """
        if len(image.shape) > 2:
            image = self._grayscale_transform({}, image)
        
        clahe = cv2.createCLAHE(
            clipLimit=step['kwargs'].get('clip_limit', 2.0),
            tileGridSize=step['kwargs'].get('tile_grid_size', (8, 8))
        )
        return clahe.apply(image)
        
    def _gaussian_blur_transform(self, step: Dict[str, Any], image: np.ndarray) -> np.ndarray:
        """
        Applies Gaussian blur for noise reduction.
        """
        ksize = step['kwargs'].get('ksize', (5, 5))
        sigmaX = step['kwargs'].get('sigmaX', 0)
        return cv2.GaussianBlur(image, ksize, sigmaX)

    def _median_blur_transform(self, step: Dict[str, Any], image: np.ndarray) -> np.ndarray:
        """
        Applies median blur for noise reduction.
        """
        ksize = step['kwargs'].get('ksize', 5)
        return cv2.medianBlur(image, ksize)

    def _bilateral_filter_transform(self, step: Dict[str, Any], image: np.ndarray) -> np.ndarray:
        """
        Applies a bilateral filter for edge-preserving noise reduction.
        """
        d = step['kwargs'].get('d', 9)
        sigmaColor = step['kwargs'].get('sigma_color', 75)
        sigmaSpace = step['kwargs'].get('sigma_space', 75)
        return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

    def _fast_nl_means_denoising_transform(self, step: Dict[str, Any], image: np.ndarray) -> np.ndarray:
        """
        Applies Fast Non-Local Means Denoising.
        """
        h = step['kwargs'].get('h', 10)
        if len(image.shape) > 2:
            return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
        return cv2.fastNlMeansDenoising(image, None, h, 7, 21)
    
    def _threshold_otsu_transform(self, step: Dict[str, Any], image: np.ndarray) -> np.ndarray:
        """
        Applies Otsu's binarization.
        """
        if len(image.shape) > 2:
            image = self._grayscale_transform({}, image)
        _, binarized = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binarized
        
    def _deskew_transform(self, step: Dict[str, Any], image: np.ndarray) -> np.ndarray:
        """
        Corrects image skew using moments.
        """
        if len(image.shape) > 2:
            image = self._grayscale_transform({}, image)
            
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return deskewed

    def _morphology_transform(self, step: Dict[str, Any], image: np.ndarray) -> np.ndarray:
        """
        Performs morphological operations (erode, dilate, open, close).
        """
        op_type = step['kwargs'].get('operation', 'erode')
        kernel_size = step['kwargs'].get('kernel_size', (3, 3))
        iterations = step['kwargs'].get('iterations', 1)
        kernel = np.ones(kernel_size, np.uint8)

        op_map = {
            'erode': cv2.MORPH_ERODE,
            'dilate': cv2.MORPH_DILATE,
            'open': cv2.MORPH_OPEN,
            'close': cv2.MORPH_CLOSE,
        }
        op = op_map.get(op_type)
        if not op:
            raise ValueError(f"Unknown morphological operation: {op_type}")
        
        return cv2.morphologyEx(image, op, kernel, iterations=iterations)
    
    def _add_border_transform(self, step: Dict[str, Any], image: np.ndarray) -> np.ndarray:
        """
        Adds a border to the image.
        """
        top = step['kwargs'].get('top', 0)
        bottom = step['kwargs'].get('bottom', 0)
        left = step['kwargs'].get('left', 0)
        right = step['kwargs'].get('right', 0)
        border_type = step['kwargs'].get('border_type', cv2.BORDER_CONSTANT)
        value = step['kwargs'].get('value', [255, 255, 255])
        
        return cv2.copyMakeBorder(image, top, bottom, left, right, border_type, value=value)
        
    def _text_detection_east_transform(self, step: Dict[str, Any], image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects text using the EAST model and returns bounding boxes.
        
        This is a placeholder function. A real implementation would require
        the EAST model files and complex processing.
        """
        logging.warning("EAST text detection is a placeholder and requires model files.")
        # Dummy bounding boxes for demonstration
        h, w = image.shape[:2]
        dummy_bboxes = [
            {"bbox": (10, 10, w - 20, h // 2 - 10)},
            {"bbox": (10, h // 2 + 10, w - 20, h - 20)}
        ]
        return dummy_bboxes

    def _crop_bbox(self, image: np.ndarray, bbox: Dict[str, Any]) -> np.ndarray:
        """
        Crops a single bounding box from the image.
        """
        x, y, w, h = bbox['bbox']
        return image[y:y+h, x:x+w]

    def _reconstruct_image_from_bboxes(self, original_image: np.ndarray, bbox_images: List[np.ndarray], bboxes: List[Dict[str, Any]]) -> np.ndarray:
        """
        Reconstructs the image by placing the processed bboxes back.
        """
        reconstructed_image = original_image.copy()
        for bbox_img, bbox in zip(bbox_images, bboxes):
            x, y, w, h = bbox['bbox']
            reconstructed_image[y:y+h, x:x+w] = bbox_img
        return reconstructed_image

    def download_all(self):
        """
        Downloads all required OCR models and language packs.
        
        This is a placeholder function. A real implementation would use
        model downloaders for each library (e.g., PaddleOCR, EasyOCR, etc.).
        """
        logging.info("Downloading all required OCR models and language packs...")
        # Placeholder for actual download logic
        # e.g., paddleocr.download_paddleocr() or easyocr.Reader(['en'], download_langs=['en'])
        logging.info("Download complete (placeholder).")

    def return_details(self) -> Dict[str, Any]:
        """
        Returns the steps list with results and performance metrics.
        
        Returns:
            Dict[str, Any]: A dictionary containing the pipeline configuration
                and results.
        """
        if self.keep_details:
            history = self.history
        else:
            history = {"whole_image": [self.image], "bboxes": []}
            
        return {
            "steps": self.steps,
            "final_image": self.image,
            "intermediate_results": history,
            "work_on_bboxes": self.work_on_bboxes,
            "bounding_boxes": self.bounding_boxes
        }

    def save_config(self, filename: str, pipeline_name: str, config: Dict[str, Any]):
        """
        Saves a pipeline configuration to a YAML file.

        Args:
            filename (str): The YAML file path.
            pipeline_name (str): A name for the pipeline.
            config (Dict[str, Any]): The configuration to save.
        """
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}
            
        data[pipeline_name] = config
        
        with open(filename, 'w') as f:
            yaml.safe_dump(data, f, sort_keys=False)
        logging.info(f"Configuration '{pipeline_name}' saved to {filename}")

    @staticmethod
    def load_config(filename: str) -> Dict[str, Any]:
        """
        Loads all configurations from a YAML file.
        
        Args:
            filename (str): The YAML file path.
            
        Returns:
            Dict[str, Any]: A dictionary of all pipelines and configurations.
        """
        with open(filename, 'r') as f:
            data = yaml.safe_load(f)
        logging.info(f"Configurations loaded from {filename}")
        return data

    def back(self, result_image: np.ndarray, history: List[np.ndarray], canvas: bool = False) -> np.ndarray:
        """
        Re-projects a result image back into the original coordinate system.
        This is a conceptual method for complex transformations like warp/resize.
        
        Args:
            result_image (np.ndarray): The transformed image to be reprojected.
            history (List[np.ndarray]): The list of intermediate images.
            canvas (bool): If True, reproject on a white canvas.
            
        Returns:
            np.ndarray: The reprojected image.
        """
        logging.warning("The 'back' method is a conceptual placeholder for complex reprojection.")
        
        # In a real implementation, you would need to store transformation matrices
        # for `perspective_warp` and `resize` to perform the inverse operation.
        # This placeholder returns the original image.
        
        if canvas:
            h, w = self.original_image.shape[:2]
            return np.ones((h, w, 3), np.uint8) * 255
        
        return self.original_image

# Example Usage
if __name__ == '__main__':
    # Create a dummy image for demonstration
    dummy_image = np.zeros((200, 400, 3), dtype=np.uint8)
    cv2.putText(dummy_image, "Hello, OCR!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite('dummy_image.png', dummy_image)
    
    # Define a sample pipeline
    sample_pipeline = [
        {"name": "Step 1: Grayscale", "type": "grayscale", "kwargs": {}},
        {"name": "Step 2: Otsu Thresholding", "type": "threshold_otsu", "kwargs": {}},
        {"name": "Step 3: Noise Reduction", "type": "median_blur", "kwargs": {"ksize": 3}},
        {"name": "Step 4: Text Detection (EAST)", "type": "text_detection_east", "kwargs": {}}
    ]
    
    # Save the pipeline to a YAML file
    test_config = {"pipeline": sample_pipeline, "global_settings": {"keep_details": True}}
    ImageCleanup(dummy_image).save_config("ocr_pipelines.yaml", "basic_text_pipeline", test_config)
    
    # Load the pipeline from the YAML file
    loaded_config = ImageCleanup.load_config("ocr_pipelines.yaml")
    pipeline_steps = loaded_config['basic_text_pipeline']['pipeline']
    
    # Initialize the class with a filename and keep_details=True
    cleanup = ImageCleanup('dummy_image.png', keep_details=True)
    
    # Process the pipeline
    cleanup.process_pipeline(pipeline_steps)
    
    # Get the detailed results
    results = cleanup.return_details()
    
    # Print out results
    print("\n--- Pipeline Details ---")
    for step in results['steps']:
        print(f"Step '{step['name']}' ({step['type']}):")
        if 'timing' in step:
            print(f"  - Execution Time: {step['timing']:.4f} seconds")
    
    print(f"\nWork on BBoxes: {results['work_on_bboxes']}")
    if results['work_on_bboxes']:
        print(f"Detected Bounding Boxes: {results['bounding_boxes']}")
        
    print("\n--- Intermediate Results ---")
    print(f"Number of Whole-Image History states: {len(results['intermediate_results']['whole_image'])}")
    print(f"Number of Bbox History states: {len(results['intermediate_results']['bboxes'])}")
    
    # Save final image for inspection
    if results['work_on_bboxes']:
        # For bbox history, you would need a more complex way to display it.
        # Here we just save the final reconstructed image.
        cv2.imwrite("final_reconstructed_image.png", results['final_image'])
        print("\nFinal reconstructed image saved as final_reconstructed_image.png")
    else:
        cv2.imwrite("final_processed_image.png", results['final_image'])
        print("\nFinal processed image saved as final_processed_image.png")
    
    # Clean up dummy files
    os.remove('dummy_image.png')
    os.remove('ocr_pipelines.yaml')
    os.remove('final_reconstructed_image.png')