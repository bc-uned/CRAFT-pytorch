import cv2
import numpy as np
from PIL import Image
import time
import yaml
from typing import Union, List, Dict, Any, Callable
import os
import requests
import tarfile
from io import BytesIO
import skimage.filters as filters
import skimage.exposure as exposure
import skimage.morphology as morphology
import easyocr
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt

class ImageCleanup:
    """
    A class for preprocessing images for OCR recognition.

    This class handles image loading, transformation pipelines, performance tracking,
    and additional utilities like model downloading and configuration management.
    """

    TEXT_DETECTION_TYPES = ['EAST', 'DB50', 'DB18', 'CRAFT', 'PP-OCRv5']
    SHAPE_CHANGING_TYPES = ['resize', 'add_border', 'perspective_warp', 'deskew']

    def __init__(
        self,
        image: Union[np.ndarray, Image.Image, str],
        steps: List[Dict[str, Any]],
        keep_details: bool = False,
    ):
        """
        Initialize the ImageCleanup class.

        :param image: Input image as cv2 image, PIL image, or filename.
        :param steps: List of transformation steps.
        :param keep_details: Flag to store intermediate results.
        """
        self.original_image = self._load_image(image)
        self.current_whole = self.original_image.copy()
        self.steps = steps
        self.keep_details = keep_details
        self.work_at_bbox_level = False
        self.bboxes: List[Dict[str, Any]] | None = None
        self.current_images = [self.current_whole]
        self.details: List[Dict[str, Any]] = []
        self.history_whole: List[np.ndarray] = []
        self.history_bbox: List[List[np.ndarray]] = []
        self.inverse_chain: List[Callable[[np.ndarray], np.ndarray]] = []
        self.inverse_chains: List[List[Callable[[np.ndarray], np.ndarray]]] | None = None

    def _load_image(self, image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        """
        Load and normalize the input image to numpy.ndarray.

        :param image: Input image.
        :return: Normalized cv2 image.
        """
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image from file: {image}")
            return img
        elif isinstance(image, Image.Image):
            return np.array(image)
        elif isinstance(image, np.ndarray):
            if image.ndim not in (2, 3):
                raise ValueError("Invalid image array dimension")
            return image.copy()
        else:
            raise TypeError("Invalid image type")

    def process(self) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Apply the transformation pipeline.

        :return: Final processed image or list of bbox images.
        """
        for step in self.steps:
            step['time'] = 0.0
            start = time.time()
            self._apply_step(step)
            step['time'] = time.time() - start
            if self.keep_details:
                result = (
                    self.current_whole.copy()
                    if not self.work_at_bbox_level
                    else [img.copy() for img in self.current_images]
                )
                self.details.append({'step': step.copy(), 'result': result})
            if self.work_at_bbox_level:
                self.history_bbox.append([img.copy() for img in self.current_images])
            else:
                self.history_whole.append(self.current_whole.copy())
        return self.current_whole if not self.work_at_bbox_level else self.current_images

    def _apply_step(self, step: Dict[str, Any]) -> None:
        type_ = step['type']
        kwargs = step.get('kwargs', {})
        if type_ in self.TEXT_DETECTION_TYPES:
            self._apply_text_detection(type_, **kwargs)
        else:
            if not self.work_at_bbox_level:
                self.current_whole = self._apply_transformation(type_, self.current_whole, **kwargs)
            else:
                for i in range(len(self.current_images)):
                    self.current_images[i] = self._apply_transformation(
                        type_, self.current_images[i], **kwargs
                    )

    def _apply_text_detection(self, type_: str, **kwargs) -> None:
        output = kwargs.get('output', 'bbox')
        model_path = kwargs.get('model_path', None)
        image = self.current_whole
        bboxes_list = []

        if type_ == 'EAST':
            if model_path is None:
                model_path = 'frozen_east_text_detection.pb'
            model = cv2.dnn_TextDetectionModel_EAST(model_path)
            model.setConfidenceThreshold(kwargs.get('conf', 0.5))
            model.setNMSThreshold(kwargs.get('nms', 0.4))
            model.setInputSize(kwargs.get('size', (320, 320)))
            rects, _ = model.detect(image)
            for rect in rects:
                if output == 'bbox':
                    x, y, w, h = cv2.boundingRect(rect)
                    bboxes_list.append({'type': 'bbox', 'coords': (x, y, w, h)})
                else:
                    bboxes_list.append({'type': 'polygon', 'coords': rect.tolist()})
        elif type_ == 'DB50' or type_ == 'DB18':
            if type_ == 'DB50':
                if model_path is None:
                    model_path = 'DB_TD500_resnet50.onnx'
            else:
                if model_path is None:
                    model_path = 'DB_TD500_resnet18.onnx'
            model = cv2.dnn_TextDetectionModel_DB(model_path)
            model.setBinaryThreshold(kwargs.get('bin_thresh', 0.3))
            model.setPolygonThreshold(kwargs.get('poly_thresh', 0.5))
            model.setMaxCandidates(kwargs.get('max_candidates', 200))
            model.setUnclipRatio(kwargs.get('unclip_ratio', 2.0))
            size = kwargs.get('size', 1024)
            model.setInputSize((size, size))
            polys, _ = model.detect(image)
            for poly in polys:
                if output == 'bbox':
                    pts = poly
                    x, y = pts.min(axis=0).astype(int)
                    w, h = (pts.max(axis=0) - pts.min(axis=0) + 1).astype(int)
                    bboxes_list.append({'type': 'bbox', 'coords': (x, y, w, h)})
                else:
                    bboxes_list.append({'type': 'polygon', 'coords': poly.tolist()})
        elif type_ == 'CRAFT':
            reader = easyocr.Reader(['en'], detector=True, recognizer=False)
            horizontal_list, free_list = reader.detect(image)
            all_bboxes = horizontal_list[0] if horizontal_list else [] + free_list[0] if free_list else []
            for bbox in all_bboxes:
                if output == 'bbox':
                    pts = np.array(bbox)
                    x, y = pts.min(0).astype(int)
                    w, h = (pts.max(0) - pts.min(0) + 1).astype(int)
                    bboxes_list.append({'type': 'bbox', 'coords': (x, y, w, h)})
                else:
                    bboxes_list.append({'type': 'polygon', 'coords': bbox})
        elif type_ == 'PP-OCRv5':
            pocr = PaddleOCR(lang='en', use_gpu=False, use_angle_cls=False, rec=False)
            result = pocr.ocr(image, cls=False, rec=False)
            if result is not None:
                for line in result:
                    for box in line:
                        poly, _ = box
                        if output == 'bbox':
                            pts = np.array(poly)
                            x, y = pts.min(0).astype(int)
                            w, h = (pts.max(0) - pts.min(0) + 1).astype(int)
                            bboxes_list.append({'type': 'bbox', 'coords': (x, y, w, h)})
                        else:
                            bboxes_list.append({'type': 'polygon', 'coords': poly})

        crops = []
        for bbox in bboxes_list:
            if bbox['type'] == 'bbox':
                x, y, w, h = bbox['coords']
                crop = image[y:y+h, x:x+w]
            else:
                pts = np.array(bbox['coords'])
                x, y = pts.min(0).astype(int)
                w, h = (pts.max(0) - pts.min(0) + 1).astype(int)
                crop = image[y:y+h, x:x+w]
            crops.append(crop)
        self.bboxes = bboxes_list
        self.current_images = crops
        self.work_at_bbox_level = True
        self.inverse_chains = [[] for _ in bboxes_list]

    def _apply_transformation(self, type_: str, image: np.ndarray, **kwargs) -> np.ndarray:
        original_shape = image.shape[:2]
        if type_ == 'resize':
            target_char_height = kwargs.get('target_char_height', 25)
            estimate = kwargs.get('estimate_char_height', False)
            if estimate:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
                _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                heights = [cv2.boundingRect(cnt)[3] for cnt in contours if 0.5 < cv2.boundingRect(cnt)[2]/cv2.boundingRect(cnt)[3] < 10 and cv2.boundingRect(cnt)[3] > 5]
                if heights:
                    current_height = np.median(heights)
                    scale = target_char_height / current_height
                    dsize = (int(image.shape[1] * scale), int(image.shape[0] * scale))
                else:
                    dsize = image.shape[:2][::-1]
            else:
                dsize = kwargs.get('size', image.shape[:2][::-1])
            interp_str = kwargs.get('method', 'bicubic')
            interps = {'bicubic': cv2.INTER_CUBIC, 'area': cv2.INTER_AREA, 'lanczos': Image.LANCZOS}
            interp = interps.get(interp_str, cv2.INTER_CUBIC)
            if interp_str == 'lanczos':
                pil_img = Image.fromarray(image)
                pil_img = pil_img.resize(dsize, Image.LANCZOS)
                image = np.array(pil_img)
            else:
                image = cv2.resize(image, dsize, interpolation=interp)
            inv = lambda img: cv2.resize(img, original_shape[::-1], interpolation=interp)
            self._add_inverse(inv)
        elif type_ == 'grayscale':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            if kwargs.get('invert', False) or (kwargs.get('auto_invert', False) and np.mean(image) < 127):
                image = 255 - image
        elif type_ == 'histogram_equalization':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            method = kwargs.get('method', 'cv2')
            if method == 'cv2':
                image = cv2.equalizeHist(gray)
            else:
                image = (exposure.equalize_hist(gray) * 255).astype(np.uint8)
        elif type_ == 'clahe':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            clahe = cv2.createCLAHE(clipLimit=kwargs.get('clip', 2.0), tileGridSize=kwargs.get('tile', (8, 8)))
            image = clahe.apply(gray)
        elif type_ == 'gaussian_blur':
            image = cv2.GaussianBlur(image, kwargs.get('kernel', (5, 5)), kwargs.get('sigma', 0))
        elif type_ == 'median_blur':
            image = cv2.medianBlur(image, kwargs.get('kernel', 5))
        elif type_ == 'bilateral_filter':
            image = cv2.bilateralFilter(
                image,
                kwargs.get('d', 9),
                kwargs.get('sigma_color', 75),
                kwargs.get('sigma_space', 75),
            )
        elif type_ == 'fast_nlmeans':
            image = cv2.fastNlMeansDenoisingColored(
                image,
                None,
                kwargs.get('h', 10),
                kwargs.get('h', 10),
                kwargs.get('template', 7),
                kwargs.get('search', 21),
            )
        elif type_ == 'otsu':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            _, image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif type_ == 'otsu_gaussian':
            image = self._apply_transformation('gaussian_blur', image, **kwargs)
            image = self._apply_transformation('otsu', image, **kwargs)
        elif type_ == 'niblack':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            image = (filters.threshold_niblack(gray, kwargs.get('window', 15), kwargs.get('k', -0.2)) * 255).astype(np.uint8)
        elif type_ == 'sauvola':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            image = (filters.threshold_sauvola(gray, kwargs.get('window', 15), kwargs.get('k', 0.5), kwargs.get('r', 128)) * 255).astype(np.uint8)
        elif type_ == 'global_threshold':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            _, image = cv2.threshold(gray, kwargs.get('thresh', 127), kwargs.get('maxval', 255), kwargs.get('thresh_type', cv2.THRESH_BINARY))
        elif type_ == 'adaptive_mean':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, kwargs.get('block', 11), kwargs.get('c', 2))
        elif type_ == 'adaptive_gaussian':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, kwargs.get('block', 11), kwargs.get('c', 2))
        elif type_ == 'try_all_threshold':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            fig, _ = filters.try_all_threshold(gray, figsize=(10, 8), verbose=False)
            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            plt.close(fig)
        elif type_ == 'deskew':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            method = kwargs.get('method', 'hough')
            if method == 'hough':
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)
                lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
                angle = 0
                if lines is not None:
                    angles = [line[0][1] * 180 / np.pi - 90 for line in lines]
                    angle = np.median(angles)
            else:
                _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                moments = cv2.moments(bin_img)
                angle = -0.5 * np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02']) * 180 / np.pi if moments['mu20'] - moments['mu02'] != 0 else 0
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            inv = lambda img: cv2.warpAffine(img, cv2.getRotationMatrix2D(center, -angle, 1.0), (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            self._add_inverse(inv)
        elif type_ == 'perspective_warp':
            src_points = np.array(kwargs['src_points'], dtype=np.float32)
            dst_points = np.array(kwargs['dst_points'], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            dsize = kwargs.get('size', image.shape[:2][::-1])
            image = cv2.warpPerspective(image, M, dsize)
            inv_M = np.linalg.inv(M)
            inv = lambda img: cv2.warpPerspective(img, inv_M, original_shape[::-1])
            self._add_inverse(inv)
        elif type_ in ['dilate', 'erode', 'open', 'close']:
            ops = {'dilate': cv2.MORPH_DILATE, 'erode': cv2.MORPH_ERODE, 'open': cv2.MORPH_OPEN, 'close': cv2.MORPH_CLOSE}
            op = ops[type_]
            kernel = kwargs.get('kernel', cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            image = cv2.morphologyEx(image, op, kernel, iterations=kwargs.get('iterations', 1))
        elif type_ == 'skeletonize':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bin_img = bin_img / 255.0
            skeleton = morphology.skeletonize(bin_img)
            image = (skeleton * 255).astype(np.uint8)
        elif type_ == 'canny_dilate':
            edges = cv2.Canny(image, kwargs.get('low', 100), kwargs.get('high', 200))
            kernel = kwargs.get('kernel', cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            image = cv2.dilate(edges, kernel)
        elif type_ == 'mser':
            mser = cv2.MSER_create(delta=kwargs.get('delta', 5))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            regions, _ = mser.detectRegions(gray)
            mask = np.zeros_like(gray)
            for pts in regions:
                cv2.polylines(mask, [pts], True, 255, 2)
            image = mask
        elif type_ == 'unsharp_mask':
            image = (filters.unsharp_mask(image, kwargs.get('sigma', 1.0), kwargs.get('amount', 1.0), kwargs.get('threshold', 0), multichannel=image.ndim == 3) * 255).astype(np.uint8)
        elif type_ == 'add_border':
            top, bottom, left, right = kwargs.get('top', 0), kwargs.get('bottom', 0), kwargs.get('left', 0), kwargs.get('right', 0)
            color = kwargs.get('color', [255, 255, 255])
            border_type = kwargs.get('type', cv2.BORDER_CONSTANT)
            image = cv2.copyMakeBorder(image, top, bottom, left, right, border_type, value=color)
            inv = lambda img: img[top:-bottom if bottom else None, left:-right if right else None]
            self._add_inverse(inv)
        elif type_ == 'remove_border':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            _, bin_img = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                image = image[y:y + h, x:x + w]
        elif type_ == 'extract_channel':
            channel = kwargs.get('channel', 'gray')
            if channel == 'gray':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                idx = 'rgb'.find(channel)
                if idx >= 0:
                    image = image[..., idx]
                else:
                    raise ValueError("Invalid channel")
        elif type_ == 'adaptive_hist_equal':
            image = (exposure.equalize_adapthist(image, **kwargs) * 255).astype(np.uint8)
        elif type_ == 'connected_components':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, labels = cv2.connectedComponents(bin_img)
            min_size = kwargs.get('min_size', 10)
            for label in np.unique(labels)[1:]:
                if (labels == label).sum() < min_size:
                    bin_img[labels == label] = 0
            image = bin_img
        elif type_ == 'kasar':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            edges = cv2.Canny(gray, kwargs.get('canny_low', 50), kwargs.get('canny_high', 150))
            edge_pixels = gray[edges > 0]
            if edge_pixels.size == 0:
                return gray
            m, s = np.mean(edge_pixels), np.std(edge_pixels)
            T = m - kwargs.get('k', 0.3) * s
            image = ((gray < T) * 255).astype(np.uint8)
        else:
            raise ValueError(f"Unknown transformation type: {type_}")
        return image

    def _add_inverse(self, inv: Callable[[np.ndarray], np.ndarray]) -> None:
        if not self.work_at_bbox_level:
            self.inverse_chain.append(inv)
        else:
            assert self.inverse_chains is not None
            for chain in self.inverse_chains:
                chain.append(inv)

    def return_details(self) -> List[Dict[str, Any]]:
        """
        Return the steps with results if keep_details is enabled.

        :return: List of step details.
        """
        return self.details

    def back(self, overlay_on_original: bool = True) -> np.ndarray:
        """
        Reproject the processed images back to the original coordinate system.

        :param overlay_on_original: Overlay on original image or white canvas.
        :return: Reprojected image.
        """
        if not self.work_at_bbox_level:
            img = self.current_whole
            for inv in reversed(self.inverse_chain):
                img = inv(img)
            return img
        else:
            canvas = self.original_image.copy() if overlay_on_original else np.full_like(self.original_image, 255)
            assert self.bboxes is not None and self.inverse_chains is not None
            for i, crop in enumerate(self.current_images):
                crop_inv = crop
                for inv in reversed(self.inverse_chains[i]):
                    crop_inv = inv(crop_inv)
                bbox = self.bboxes[i]
                if bbox['type'] == 'bbox':
                    x, y, w, h = bbox['coords']
                else:
                    pts = np.array(bbox['coords'])
                    x = int(pts[:, 0].min())
                    y = int(pts[:, 1].min())
                    w = int(pts[:, 0].max() - x + 1)
                    h = int(pts[:, 1].max() - y + 1)
                if crop_inv.shape[:2] != (h, w):
                    crop_inv = cv2.resize(crop_inv, (w, h))
                if crop_inv.ndim == 2:
                    crop_inv = cv2.cvtColor(crop_inv, cv2.COLOR_GRAY2BGR)
                canvas[y:y + h, x:x + w] = crop_inv
            return canvas

    @staticmethod
    def download_all() -> None:
        """
        Download all required OCR models and language packs.
        """
        # EAST
        east_url = 'https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1'
        response = requests.get(east_url)
        with BytesIO(response.content) as f:
            with tarfile.open(fileobj=f) as tar:
                tar.extractall()
        # DB50
        db50_id = '19YWhArrNccaoSza0CfkXlA8im4-lAGsR'
        db50_url = f'https://drive.google.com/uc?export=download&id={db50_id}'
        response = requests.get(db50_url)
        with open('DB_TD500_resnet50.onnx', 'wb') as f:
            f.write(response.content)
        # DB18
        db18_id = '1sZszH3pEt8hliyBlTmB-iulxHP1dCQWV'
        db18_url = f'https://drive.google.com/uc?export=download&id={db18_id}'
        response = requests.get(db18_url)
        with open('DB_TD500_resnet18.onnx', 'wb') as f:
            f.write(response.content)
        # CRAFT
        easyocr.Reader(['en'])
        # PP-OCRv5 (assuming PaddleOCR handles v5 by 2025)
        PaddleOCR(lang='en')

    def save(self, filename: str, pipeline_name: str = 'default') -> None:
        """
        Save global config and pipeline to YAML.

        :param filename: YAML file path.
        :param pipeline_name: Name of the pipeline.
        """
        data = {
            'global': {
                'keep_details': self.keep_details,
            },
            'pipelines': {
                pipeline_name: self.steps,
            },
        }
        with open(filename, 'w') as f:
            yaml.dump(data, f)

    @classmethod
    def load(
        cls,
        filename: str,
        image: Union[np.ndarray, Image.Image, str],
        pipeline_name: str = 'default',
    ) -> 'ImageCleanup':
        """
        Load config and pipeline from YAML.

        :param filename: YAML file path.
        :param image: Input image.
        :param pipeline_name: Name of the pipeline.
        :return: ImageCleanup instance.
        """
        with open(filename, 'r') as f:
            data = yaml.safe_load(f)
        global_config = data['global']
        steps = data['pipelines'][pipeline_name]
        return cls(image, steps, **global_config)