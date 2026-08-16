"""
Meme OCR Pipeline — PaddleOCR v4 backbone.

Replaces the old CRAFT + EasyOCR pipeline with:
  Stage 0: Multi-scale preprocessing (upscale, CLAHE, bilateral, 3 image versions)
  Stage 1: Multi-path PaddleOCR v4 detection with NMS dedup + confidence gate
  Stage 2: DBSCAN spatial clustering for top/bottom caption separation
  Stage 3: (external) BiLSTM classification via result['all_text']

Backward-compatible exports:
  - extract_and_clean_text(pil_image) -> (cleaned, raw)
  - OCR_LAST_DEBUG dict

Usage:
    from ocr_pipeline import MemeOCRPipeline
    pipeline = MemeOCRPipeline()
    result = pipeline.extract("meme.jpg")
    # result = {top_text, bottom_text, all_text, num_regions, raw_regions}
"""

import os
# Disable oneDNN for PaddlePaddle 3.0 on Windows CPU (prevents PIR attribute error)
os.environ.setdefault("FLAGS_use_mkldnn", "0")

import logging
from typing import List, Dict, Optional, NamedTuple
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TextRegion:
    """A detected text region with bounding-box centre, text, and confidence."""
    text: str
    confidence: float
    x_center: float
    y_center: float
    bbox: list = field(default_factory=list)
    source_version: str = ""


# ---------------------------------------------------------------------------
# Debug info exported for backward compatibility with app.py
# ---------------------------------------------------------------------------
OCR_LAST_DEBUG: Dict = {}


# ---------------------------------------------------------------------------
# MemeOCRPipeline
# ---------------------------------------------------------------------------

class MemeOCRPipeline:
    """PaddleOCR v4–based meme text extraction pipeline."""

    # ── construction ────────────────────────────────────────────────────

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self._paddle_ocr = None  # lazy init
        self._easyocr_reader = None  # lazy fallback

    # Lazy-load PaddleOCR so import doesn't block
    def _get_paddle_ocr(self):
        if self._paddle_ocr is None:
            from paddleocr import PaddleOCR
            self._paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                use_gpu=self.use_gpu,
                show_log=False,
                # PP-OCRv4 models (auto-downloaded on first run ~50 MB)
                ocr_version="PP-OCRv4",
            )
        return self._paddle_ocr

    # Lazy-load EasyOCR for optional fallback
    def _get_easyocr_reader(self):
        if self._easyocr_reader is None:
            try:
                import easyocr
                self._easyocr_reader = easyocr.Reader(
                    ["en"], gpu=self.use_gpu, verbose=False
                )
            except Exception as exc:
                logger.warning("EasyOCR fallback unavailable: %s", exc)
        return self._easyocr_reader

    # ── Stage 0: Multi-Scale Preprocessing ──────────────────────────────

    def preprocess(self, image: np.ndarray) -> dict:
        """Return dict with keys 'original', 'inverted', 'enhanced_gray'.

        Key insight: PaddleOCR has its own internal text detector that works
        best on full-detail images, NOT pre-binarized ones.  We feed it three
        complementary views so it can pick up both dark-on-light AND
        white-on-dark (Impact font) text:

          - original     — upscaled + CLAHE-enhanced colour image
                           (handles dark text on light backgrounds)
          - inverted     — colour-inverted so white meme text becomes dark
                           on a light background (critical for Impact font)
          - enhanced_gray — high-contrast grayscale with bilateral filter
                           (fallback that sometimes catches faint text)
        """
        img = image.copy()

        # --- Upscale if too small ---
        h, w = img.shape[:2]
        if w < 800:
            scale = 800 / w
            img = cv2.resize(
                img, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_LANCZOS4,
            )

        # --- Version 1: Enhanced original colour (good for dark text) ---
        # Convert to LAB, apply CLAHE to lightness channel only
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_ch = clahe.apply(l_ch)
            lab = cv2.merge([l_ch, a_ch, b_ch])
            original = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            original = clahe.apply(img)

        # --- Version 2: Colour-inverted (white text → dark text) ---
        inverted = cv2.bitwise_not(original)

        # --- Version 3: Enhanced grayscale ---
        if len(img.shape) == 3:
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            gray = original.copy()
        enhanced_gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        # Convert back to 3-channel so PaddleOCR accepts it
        enhanced_gray = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

        return {
            "original": original,
            "inverted": inverted,
            "enhanced_gray": enhanced_gray,
        }

    # ── Stage 1: Multi-Path Detection ───────────────────────────────────

    def detect_text_regions(self, image_versions: dict) -> List[TextRegion]:
        """Run PaddleOCR on all 3 image versions, deduplicate, and
        filter by confidence >= 0.6.

        Returns a list of TextRegion objects.
        """
        ocr = self._get_paddle_ocr()
        all_regions: List[TextRegion] = []

        for version_name, img in image_versions.items():
            try:
                # PaddleOCR expects BGR or grayscale numpy array
                result = ocr.ocr(img, cls=True)
                if result is None:
                    continue

                for line_group in result:
                    if line_group is None:
                        continue
                    for line in line_group:
                        if line is None or len(line) < 2:
                            continue
                        bbox_points = line[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                        text_info = line[1]     # ("text", confidence)
                        if text_info is None or len(text_info) < 2:
                            continue

                        text = str(text_info[0]).strip()
                        conf = float(text_info[1])

                        if not text:
                            continue

                        # Compute centre of bounding box
                        xs = [p[0] for p in bbox_points]
                        ys = [p[1] for p in bbox_points]
                        x_center = sum(xs) / len(xs)
                        y_center = sum(ys) / len(ys)

                        all_regions.append(TextRegion(
                            text=text,
                            confidence=conf,
                            x_center=x_center,
                            y_center=y_center,
                            bbox=bbox_points,
                            source_version=version_name,
                        ))
            except Exception as exc:
                logger.warning("PaddleOCR failed on '%s' version: %s", version_name, exc)
                continue

        # --- NMS deduplication ---
        all_regions = self._nms_dedup(all_regions)

        # --- Confidence gate ---
        filtered = [r for r in all_regions if r.confidence >= 0.6]

        # --- Optional EasyOCR fallback for low-confidence regions ---
        low_conf = [r for r in all_regions if r.confidence < 0.6]
        if low_conf:
            reader = self._get_easyocr_reader()
            if reader is not None:
                for region in low_conf:
                    try:
                        # Try EasyOCR on a small crop around the region
                        # (best-effort; skipped if anything fails)
                        easy_results = reader.readtext(
                            image_versions.get("normal", list(image_versions.values())[0]),
                            paragraph=False,
                        )
                        for er in easy_results:
                            if len(er) >= 3:
                                easy_conf = float(er[2])
                                easy_text = str(er[1]).strip()
                                if easy_conf >= 0.6 and easy_text:
                                    bbox_pts = er[0]
                                    exs = [p[0] for p in bbox_pts]
                                    eys = [p[1] for p in bbox_pts]
                                    filtered.append(TextRegion(
                                        text=easy_text,
                                        confidence=easy_conf,
                                        x_center=sum(exs) / len(exs),
                                        y_center=sum(eys) / len(eys),
                                        bbox=bbox_pts,
                                        source_version="easyocr_fallback",
                                    ))
                        break  # only run EasyOCR once, not per-region
                    except Exception as exc:
                        logger.warning("EasyOCR fallback failed: %s", exc)
                        break

                # Deduplicate again after fallback additions
                filtered = self._nms_dedup(filtered)
                filtered = [r for r in filtered if r.confidence >= 0.6]

        return filtered

    @staticmethod
    def _nms_dedup(regions: List[TextRegion]) -> List[TextRegion]:
        """Non-maximum suppression: if two boxes have |dx| < 50 AND |dy| < 20,
        keep the one with higher confidence.

        Also handles text containment: if a higher-confidence region's text
        contains a lower-confidence region's text (or vice versa) and they
        are on the same vertical band (|dy| < 40), the longer text wins.
        """
        if not regions:
            return regions

        # Sort by confidence descending so we keep the best one first
        sorted_regions = sorted(regions, key=lambda r: r.confidence, reverse=True)
        kept: List[TextRegion] = []

        for region in sorted_regions:
            is_duplicate = False
            for existing in kept:
                dx = abs(region.x_center - existing.x_center)
                dy = abs(region.y_center - existing.y_center)

                # Standard spatial NMS (PRD spec)
                if dx < 50 and dy < 20:
                    is_duplicate = True
                    break

                # Text containment check: same vertical band, text overlap
                if dy < 40:
                    r_text = region.text.strip().lower()
                    e_text = existing.text.strip().lower()
                    if r_text in e_text or e_text in r_text:
                        is_duplicate = True
                        break

            if not is_duplicate:
                kept.append(region)

        return kept

    # ── Stage 2: Spatial Clustering ─────────────────────────────────────

    def cluster_by_position(self, regions: List[TextRegion]) -> dict:
        """DBSCAN clustering on y-coordinates to separate top/bottom captions.

        Returns:
            {
                'top_text':    str,
                'bottom_text': str,
                'all_text':    str,
                'num_regions': int,
                'raw_regions': [{'text': ..., 'confidence': ...}, ...]
            }
        """
        if not regions:
            return {
                "top_text": "",
                "bottom_text": "",
                "all_text": "",
                "num_regions": 0,
                "raw_regions": [],
            }

        from sklearn.cluster import DBSCAN

        # Cluster on y-coordinates only
        y_coords = np.array([[r.y_center] for r in regions])
        clustering = DBSCAN(eps=50, min_samples=1).fit(y_coords)
        labels = clustering.labels_

        # Group regions by cluster label
        clusters: Dict[int, List[TextRegion]] = {}
        for region, label in zip(regions, labels):
            clusters.setdefault(label, []).append(region)

        # Sort clusters by mean y-position (top → bottom)
        sorted_cluster_ids = sorted(
            clusters.keys(),
            key=lambda cid: np.mean([r.y_center for r in clusters[cid]])
        )

        # Build text per cluster (within each cluster, sort L→R by x_center)
        cluster_texts = []
        for cid in sorted_cluster_ids:
            members = sorted(clusters[cid], key=lambda r: r.x_center)
            text = " ".join(r.text for r in members)
            cluster_texts.append(text)

        # Assign top / bottom
        top_text = cluster_texts[0] if len(cluster_texts) >= 1 else ""
        bottom_text = cluster_texts[-1] if len(cluster_texts) >= 2 else ""
        all_text = " ".join(cluster_texts)

        raw_regions = [
            {"text": r.text, "confidence": round(r.confidence, 4)}
            for r in regions
        ]

        return {
            "top_text": top_text,
            "bottom_text": bottom_text,
            "all_text": all_text,
            "num_regions": len(regions),
            "raw_regions": raw_regions,
        }

    # ── Main entrypoint ─────────────────────────────────────────────────

    def extract(self, image_path: str) -> dict:
        """Full pipeline: path in → structured dict out.

        Args:
            image_path: Path to the meme image file.

        Returns:
            dict with keys: top_text, bottom_text, all_text, num_regions, raw_regions
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            # Try with PIL as fallback (handles more formats)
            try:
                pil_img = Image.open(image_path).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as exc:
                logger.warning("Failed to load image '%s': %s", image_path, exc)
                return {
                    "top_text": "", "bottom_text": "", "all_text": "",
                    "num_regions": 0, "raw_regions": [],
                }

        # Stage 0
        versions = self.preprocess(img)

        # Stage 1
        regions = self.detect_text_regions(versions)

        # Stage 2
        result = self.cluster_by_position(regions)

        return result

    def extract_from_pil(self, pil_image: Image.Image) -> dict:
        """Convenience: accept a PIL Image directly (used by app.py).

        Args:
            pil_image: PIL Image object.

        Returns:
            Same structured dict as extract().
        """
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        versions = self.preprocess(img)
        regions = self.detect_text_regions(versions)
        result = self.cluster_by_position(regions)
        return result


# ---------------------------------------------------------------------------
# Backward-compatible wrapper used by app.py
# ---------------------------------------------------------------------------

_pipeline_instance: Optional[MemeOCRPipeline] = None


def _get_pipeline() -> MemeOCRPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = MemeOCRPipeline(use_gpu=False)
    return _pipeline_instance


def extract_and_clean_text(pil_image: Image.Image):
    """Backward-compatible function matching the old pipeline's signature.

    Returns:
        (cleaned_text: str, raw_merged_text: str)
    """
    # Check environment variable for rollback to old pipeline
    if os.environ.get("MEME_OCR_BACKEND", "").lower() == "craft":
        try:
            from craft_pipeline import extract_and_clean_text as craft_extract
            return craft_extract(pil_image)
        except ImportError:
            logger.warning("MEME_OCR_BACKEND=craft but craft_pipeline.py not found, using PaddleOCR")

    pipeline = _get_pipeline()
    result = pipeline.extract_from_pil(pil_image)

    cleaned = result.get("all_text", "")
    raw = cleaned  # PaddleOCR output is already clean

    # Populate OCR_LAST_DEBUG for the debug panel in app.py
    OCR_LAST_DEBUG.clear()
    OCR_LAST_DEBUG["variants_tested"] = [
        ("normal", 0), ("inverted", 0), ("dilated", 0)
    ]
    OCR_LAST_DEBUG["best_variant"] = "paddle_multi_path"
    OCR_LAST_DEBUG["best_count"] = len(cleaned)
    OCR_LAST_DEBUG["top_text"] = result.get("top_text", "")
    OCR_LAST_DEBUG["bottom_text"] = result.get("bottom_text", "")
    OCR_LAST_DEBUG["num_regions"] = result.get("num_regions", 0)
    OCR_LAST_DEBUG["raw_regions"] = result.get("raw_regions", [])

    return cleaned, raw


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python ocr_pipeline.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    pipeline = MemeOCRPipeline(use_gpu=False)
    result = pipeline.extract(path)
    print(json.dumps(result, indent=2, ensure_ascii=False))