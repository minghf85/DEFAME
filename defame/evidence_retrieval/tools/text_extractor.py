from paddleocr import PaddleOCR
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from PIL.Image import Image as PILImage
import numpy as np
from ezmm import Image, MultimodalSequence

from defame.common import Action, logger
from defame.common.results import Results
from defame.evidence_retrieval.tools.tool import Tool


class OCR(Action):
    """Performs Optical Character Recognition to extract text from an image."""
    name = "ocr"
    requires_image = True

    def __init__(self, image: str):
        """
        @param image: The reference of the image to extract text from.
        """
        self._save_parameters(locals())
        self.image = Image(reference=image)

    def __str__(self):
        return f'{self.name}({self.image.reference})'

    def __eq__(self, other):
        return isinstance(other, OCR) and self.image == other.image

    def __hash__(self):
        return hash((self.name, self.image))


@dataclass
class TextBlock:
    """Represents a detected text block with position and confidence information."""
    text: str
    bbox: List[List[float]]  # Bounding box coordinates [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    confidence: float
    
    @property
    def center_y(self) -> float:
        """Calculate the vertical center of the text block for sorting."""
        y_coords = [point[1] for point in self.bbox]
        return sum(y_coords) / len(y_coords)
    
    @property 
    def center_x(self) -> float:
        """Calculate the horizontal center of the text block for sorting."""
        x_coords = [point[0] for point in self.bbox]
        return sum(x_coords) / len(x_coords)


@dataclass
class OCRResults(Results):
    source: str
    extracted_text: str
    markdown_text: str = ""
    text_blocks: List[TextBlock] = field(default_factory=list)
    model_output: Optional[list] = None
    text: str = field(init=False)  # This will be assigned in __post_init__

    def __post_init__(self):
        self.text = str(self)

    def __str__(self):
        if self.markdown_text:
            return f'## OCR Results from {self.source}\n\n{self.markdown_text}\n\n**Confidence:** {self._get_average_confidence():.2f}'
        return f'From [Source]({self.source}):\nExtracted Text: {self.extracted_text}'

    def _get_average_confidence(self) -> float:
        """Calculate average confidence across all text blocks."""
        if not self.text_blocks:
            return 0.0
        return sum(block.confidence for block in self.text_blocks) / len(self.text_blocks)

    def is_useful(self) -> Optional[bool]:
        return self.model_output is not None and len(self.text_blocks) > 0


class TextExtractor(Tool):
    """Employs OCR to get all the text visible in the image and format it as markdown."""
    name = "text_extractor"
    actions = [OCR]
    summarize = False

    def __init__(self, 
                 lang='en',
                 ocr_version='PP-OCRv5',
                 use_textline_orientation=True,
                 text_det_limit_side_len=960,
                 text_det_thresh=0.3,
                 text_det_box_thresh=0.6,
                 text_det_unclip_ratio=1.5,
                 text_rec_score_thresh=0.5,
                 return_word_box=False,
                 use_doc_orientation_classify=False,
                 use_doc_unwarping=False,
                 **kwargs):
        super().__init__(**kwargs)
        """
        Initialize the OCR tool with PaddleOCR optimized for fact-checking tasks.

        :param lang: Language for OCR recognition ('ch', 'en', 'korean', 'japan', etc.)
        :param ocr_version: PP-OCR version to use ('PP-OCRv3', 'PP-OCRv4', 'PP-OCRv5')
        :param use_textline_orientation: Whether to use text line orientation classification
        :param text_det_limit_side_len: Limit on the side length of input image for text detection
        :param text_det_thresh: Detection pixel threshold (lower = more sensitive)
        :param text_det_box_thresh: Detection box threshold (lower = more boxes)
        :param text_det_unclip_ratio: Text detection expansion coefficient
        :param text_rec_score_thresh: Text recognition threshold (lower = more text kept)
        :param return_word_box: Whether to return word-level bounding boxes
        :param use_doc_orientation_classify: Whether to use document orientation classification
        :param use_doc_unwarping: Whether to use document unwarping
        """
        self.model = None  # For future trainable OCR model integration
        self.lang = lang
        self.ocr_version = ocr_version
        
        # Store OCR parameters
        self.ocr_params = {
            'use_textline_orientation': use_textline_orientation,
            'text_det_limit_side_len': text_det_limit_side_len,
            'text_det_thresh': text_det_thresh,
            'text_det_box_thresh': text_det_box_thresh,
            'text_det_unclip_ratio': text_det_unclip_ratio,
            'text_rec_score_thresh': text_rec_score_thresh,
            'return_word_box': return_word_box,
            'use_doc_orientation_classify': use_doc_orientation_classify,
            'use_doc_unwarping': use_doc_unwarping
        }
        
        # Initialize PaddleOCR with optimized parameters for fact-checking
        try:
            self.reader = PaddleOCR(
                lang=lang,
                ocr_version=ocr_version,
                use_textline_orientation=use_textline_orientation,
                text_det_limit_side_len=text_det_limit_side_len,
                text_det_thresh=text_det_thresh,
                text_det_box_thresh=text_det_box_thresh,
                text_det_unclip_ratio=text_det_unclip_ratio,
                text_rec_score_thresh=text_rec_score_thresh,
                return_word_box=return_word_box,
                use_doc_orientation_classify=use_doc_orientation_classify,
                use_doc_unwarping=use_doc_unwarping
            )
            logger.log(f"PaddleOCR initialized successfully - Language: {lang}, Version: {ocr_version}")
        except Exception as e:
            logger.error(f"Error initializing PaddleOCR: {e}")
            self.reader = None

    def _perform(self, action: OCR) -> Results:
        """Perform OCR action on the specified image."""
        # Get PIL Image from the ezmm Image object
        try:
            return self.extract_text(action.image.image)
        except Exception as e:
            logger.error(f"Error getting PIL image from ezmm Image: {e}")
            return OCRResults(
                source="PaddleOCR", 
                extracted_text="", 
                markdown_text="",
                text_blocks=[],
                model_output=None
            )

    def extract_text(self, image) -> OCRResults:
        """
        Perform OCR on an image using PaddleOCR and format as markdown.
        Optimized for fact-checking with better parameter configuration.

        :param image: A PIL image or image array.
        :return: An OCRResult object containing the extracted text and markdown.
        """
        if self.reader is None:
            logger.error("PaddleOCR not properly initialized")
            return OCRResults(
                source="PaddleOCR", 
                extracted_text="", 
                markdown_text="",
                text_blocks=[],
                model_output=None
            )
        
        try:
            # Convert PIL image to numpy array if needed
            if hasattr(image, 'mode'):  # PIL Image
                img_array = np.array(image)
            else:
                img_array = image
            
            # Use PaddleOCR predict method with dynamic parameter passing
            ocr_kwargs = {}
            # Only pass non-None parameters to avoid overriding defaults
            for key, value in self.ocr_params.items():
                if value is not None:
                    ocr_kwargs[key] = value
            
            # Call PaddleOCR with optimized parameters
            results = self.reader.predict(img_array, **ocr_kwargs)
            
            if not results:
                logger.warning("PaddleOCR returned empty results")
                return OCRResults(
                    source="PaddleOCR", 
                    extracted_text="", 
                    markdown_text="",
                    text_blocks=[],
                    model_output=results
                )
            
            logger.log(f"PaddleOCR processed image successfully, found {len(results)} result objects")
            
            # Process results and create text blocks
            text_blocks = self._parse_ocr_results(results)
            
            # Sort text blocks by reading order (top to bottom, left to right)
            if text_blocks:
                text_blocks.sort(key=lambda block: (block.center_y, block.center_x))
            
            # Generate different text formats
            extracted_text = self._generate_plain_text(text_blocks)
            markdown_text = self._generate_markdown_text(text_blocks)
            
            result = OCRResults(
                source="PaddleOCR", 
                extracted_text=extracted_text,
                markdown_text=markdown_text,
                text_blocks=text_blocks,
                model_output=results
            )
            
            logger.log(f"OCR completed. Detected {len(text_blocks)} text blocks.")
            if extracted_text:
                preview = extracted_text[:100] + "..." if len(extracted_text) > 100 else extracted_text
                logger.log(f"Sample extracted text: {preview}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during OCR processing: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return OCRResults(
                source="PaddleOCR", 
                extracted_text="", 
                markdown_text="",
                text_blocks=[],
                model_output=None
            )

    def _parse_ocr_results(self, results) -> List[TextBlock]:
        """
        Parse PaddleOCR results into TextBlock objects.
        Handles both legacy and new PaddleX result formats.
        """
        text_blocks = []
        
        if not results or len(results) == 0:
            return text_blocks
        
        first_result = results[0]
        
        # Check for new PaddleX OCRResult format with json attribute
        if hasattr(first_result, 'json'):
            text_blocks.extend(self._parse_paddlex_format(first_result))
        
        # Check for predictions attribute (alternative new format)
        elif hasattr(first_result, 'predictions'):
            text_blocks.extend(self._parse_predictions_format(first_result))
        
        # Check for dictionary format with predictions key
        elif isinstance(first_result, dict) and 'predictions' in first_result:
            text_blocks.extend(self._parse_dict_predictions_format(first_result))
        
        # Fallback to legacy format parsing
        elif isinstance(first_result, list):
            text_blocks.extend(self._parse_legacy_format(first_result))
        
        # Try to handle any other list-like structure
        else:
            logger.warning(f"Unexpected OCR result format: {type(first_result)}")
            # Try to iterate through results as if they're line results
            for line_idx, line in enumerate(results):
                try:
                    if self._is_legacy_line_format(line):
                        text_blocks.extend(self._parse_single_legacy_line(line, line_idx))
                except Exception as e:
                    logger.warning(f"Error parsing result line {line_idx}: {e}")
                    continue
        
        return text_blocks

    def _parse_paddlex_format(self, result) -> List[TextBlock]:
        """Parse new PaddleX format with json attribute."""
        text_blocks = []
        try:
            result_data = getattr(result, 'json')
            res = result_data.get('res', {}) if isinstance(result_data, dict) else {}
            
            # Extract text information
            rec_texts = res.get('rec_texts', [])
            rec_scores = res.get('rec_scores', [])
            rec_polys = res.get('rec_polys', [])
            
            logger.log(f"PaddleX format: Found {len(rec_texts)} texts")
            
            # Create text blocks
            for i, text in enumerate(rec_texts):
                if text and text.strip():
                    confidence = rec_scores[i] if i < len(rec_scores) else 1.0
                    bbox = rec_polys[i] if i < len(rec_polys) else []
                    
                    # Convert numpy array to list if necessary
                    try:
                        if hasattr(bbox, 'tolist') and callable(getattr(bbox, 'tolist')):
                            bbox = getattr(bbox, 'tolist')()
                        elif not isinstance(bbox, list):
                            bbox = list(bbox) if bbox else []
                    except:
                        bbox = []
                    
                    text_block = TextBlock(
                        text=text.strip(),
                        bbox=bbox,
                        confidence=float(confidence)
                    )
                    text_blocks.append(text_block)
        except Exception as e:
            logger.error(f"Error parsing PaddleX format: {e}")
        
        return text_blocks

    def _parse_predictions_format(self, result) -> List[TextBlock]:
        """Parse format with predictions attribute."""
        text_blocks = []
        try:
            predictions = getattr(result, 'predictions')
            
            for pred in predictions:
                if hasattr(pred, 'text') and hasattr(pred, 'bbox') and hasattr(pred, 'score'):
                    text = pred.text.strip()
                    bbox = pred.bbox
                    confidence = pred.score
                    
                    if text:  # Only add non-empty text
                        text_block = TextBlock(
                            text=text,
                            bbox=bbox,
                            confidence=confidence
                        )
                        text_blocks.append(text_block)
                elif isinstance(pred, dict):
                    text_blocks.extend(self._parse_dict_prediction(pred))
        except Exception as e:
            logger.error(f"Error parsing predictions format: {e}")
        
        return text_blocks

    def _parse_dict_predictions_format(self, result) -> List[TextBlock]:
        """Parse dictionary format with predictions key."""
        text_blocks = []
        try:
            predictions = result['predictions']
            
            for pred in predictions:
                if isinstance(pred, dict):
                    text_blocks.extend(self._parse_dict_prediction(pred))
        except Exception as e:
            logger.error(f"Error parsing dict predictions format: {e}")
        
        return text_blocks

    def _parse_dict_prediction(self, pred) -> List[TextBlock]:
        """Parse a single prediction dictionary."""
        text_blocks = []
        try:
            text = pred.get('text', '').strip()
            bbox = pred.get('bbox', [])
            confidence = pred.get('score', 1.0)
            
            if text:
                text_block = TextBlock(
                    text=text,
                    bbox=bbox,
                    confidence=confidence
                )
                text_blocks.append(text_block)
        except Exception as e:
            logger.warning(f"Error parsing dict prediction: {e}")
        
        return text_blocks

    def _parse_legacy_format(self, result) -> List[TextBlock]:
        """Parse legacy PaddleOCR format."""
        text_blocks = []
        try:
            for line_idx, line in enumerate(result):
                text_blocks.extend(self._parse_single_legacy_line(line, line_idx))
        except Exception as e:
            logger.error(f"Error parsing legacy format: {e}")
        
        return text_blocks

    def _is_legacy_line_format(self, line) -> bool:
        """Check if a line follows the legacy format."""
        return (isinstance(line, (list, tuple)) and 
                len(line) >= 2 and 
                isinstance(line[0], list))  # bbox should be a list

    def _parse_single_legacy_line(self, line, line_idx) -> List[TextBlock]:
        """Parse a single line in legacy format."""
        text_blocks = []
        try:
            if not self._is_legacy_line_format(line):
                return text_blocks
            
            bbox = line[0]
            text_info = line[1]
            
            # Extract text and confidence
            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                text = str(text_info[0])
                confidence = float(text_info[1])
            elif isinstance(text_info, str):
                text = text_info
                confidence = 1.0
            else:
                logger.warning(f"Unexpected text_info format at line {line_idx}: {type(text_info)}")
                return text_blocks
            
            # Validate bbox format
            if bbox and isinstance(bbox, list) and len(bbox) >= 4:
                if text.strip():  # Only add non-empty text
                    text_block = TextBlock(
                        text=text.strip(),
                        bbox=bbox,
                        confidence=confidence
                    )
                    text_blocks.append(text_block)
            else:
                logger.warning(f"Invalid bbox format for line {line_idx}: {bbox}")
                
        except Exception as e:
            logger.warning(f"Error processing legacy line {line_idx}: {e}")
        
        return text_blocks

    def _generate_plain_text(self, text_blocks: List[TextBlock]) -> str:
        """Generate plain text from text blocks."""
        return ' '.join(block.text for block in text_blocks)

    def _generate_markdown_text(self, text_blocks: List[TextBlock]) -> str:
        """
        Generate markdown formatted text from text blocks.
        Optimized for fact-checking with better structure detection.
        """
        if not text_blocks:
            return ""
        
        markdown_lines = []
        current_paragraph = []
        
        # Analyze text blocks to detect structure
        for i, block in enumerate(text_blocks):
            text = block.text.strip()
            confidence = block.confidence
            
            # Add confidence annotation for low-confidence text (for fact-checking quality assessment)
            text_with_confidence = text
            if confidence < 0.7:
                text_with_confidence = f"*{text}* `[conf: {confidence:.2f}]`"
            elif confidence < 0.9:
                text_with_confidence = f"{text} `[conf: {confidence:.2f}]`"
            
            # Enhanced structure detection for fact-checking
            is_potential_header = self._is_header_text(text, confidence)
            is_list_item = self._is_list_item(text)
            is_date_or_number = self._is_date_or_number(text)
            is_url_or_handle = self._is_url_or_handle(text)
            is_paragraph_end = self._is_paragraph_end(text_blocks, i)
            
            # Handle different text types
            if is_potential_header:
                # Flush current paragraph
                if current_paragraph:
                    markdown_lines.append(' '.join(current_paragraph))
                    current_paragraph = []
                
                # Determine header level based on text characteristics
                header_level = self._determine_header_level(text, confidence)
                markdown_lines.append(f"\n{'#' * header_level} {text_with_confidence}\n")
                
            elif is_list_item:
                # Flush current paragraph
                if current_paragraph:
                    markdown_lines.append(' '.join(current_paragraph))
                    current_paragraph = []
                
                # Format as list item
                if not text.startswith(('•', '-', '*', '1.', '2.', '3.', '4.', '5.')):
                    text_with_confidence = f"- {text_with_confidence}"
                markdown_lines.append(text_with_confidence)
                
            elif is_date_or_number:
                # Dates and numbers often important for fact-checking
                markdown_lines.append(f"**{text_with_confidence}**")
                
            elif is_url_or_handle:
                # URLs and social media handles are crucial for fact-checking
                markdown_lines.append(f"`{text_with_confidence}`")
                
            else:
                # Regular text - add to current paragraph
                current_paragraph.append(text_with_confidence)
                
                # Check if this might be end of paragraph
                if is_paragraph_end:
                    markdown_lines.append(' '.join(current_paragraph))
                    current_paragraph = []
        
        # Add remaining paragraph
        if current_paragraph:
            markdown_lines.append(' '.join(current_paragraph))
        
        # Join all lines and clean up
        markdown_text = '\n\n'.join(line.strip() for line in markdown_lines if line.strip())
        
        # Add structured metadata section for fact-checking
        if text_blocks:
            avg_confidence = sum(block.confidence for block in text_blocks) / len(text_blocks)
            low_conf_count = sum(1 for block in text_blocks if block.confidence < 0.7)
            
            metadata_lines = [
                "\n---",
                "### OCR Analysis Summary",
                f"- **Text blocks detected:** {len(text_blocks)}",
                f"- **Average confidence:** {avg_confidence:.3f}",
                f"- **Low confidence blocks:** {low_conf_count}",
                f"- **Language:** {self.lang}",
                f"- **OCR Version:** {self.ocr_version}",
            ]
            
            # Add confidence distribution
            high_conf = sum(1 for block in text_blocks if block.confidence >= 0.9)
            med_conf = sum(1 for block in text_blocks if 0.7 <= block.confidence < 0.9)
            
            metadata_lines.extend([
                f"- **High confidence (≥0.9):** {high_conf}",
                f"- **Medium confidence (0.7-0.9):** {med_conf}",
                f"- **Low confidence (<0.7):** {low_conf_count}",
            ])
            
            markdown_text += '\n'.join(metadata_lines)
        
        return markdown_text

    def _is_header_text(self, text: str, confidence: float) -> bool:
        """Detect if text is likely a header."""
        return (
            len(text.split()) <= 6 and 
            (text.isupper() or text.istitle() or 
             any(word in text.lower() for word in ['news', 'report', 'article', 'headline', 'breaking'])) and
            confidence > 0.8 and
            len(text) > 3
        )

    def _is_list_item(self, text: str) -> bool:
        """Detect if text is a list item."""
        return (
            text.startswith(('•', '-', '*', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')) or
            text.endswith(':') and len(text.split()) <= 4 or
            any(text.lower().startswith(prefix) for prefix in ['step ', 'point ', 'item '])
        )

    def _is_date_or_number(self, text: str) -> bool:
        """Detect dates, numbers, or statistical information important for fact-checking."""
        import re
        # Date patterns
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY or MM-DD-YYYY
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',    # YYYY/MM/DD or YYYY-MM-DD
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}',
            r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}',
        ]
        
        # Number/statistical patterns
        number_patterns = [
            r'\d+%',  # Percentages
            r'\$\d+',  # Money
            r'\d+\s*(million|billion|thousand|k|M|B)',  # Large numbers
            r'\d+:\d+',  # Ratios or times
        ]
        
        all_patterns = date_patterns + number_patterns
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in all_patterns)

    def _is_url_or_handle(self, text: str) -> bool:
        """Detect URLs, email addresses, or social media handles."""
        import re
        patterns = [
            r'https?://[^\s]+',  # URLs
            r'www\.[^\s]+',      # www domains
            r'@[a-zA-Z0-9_]+',   # Social media handles
            r'#[a-zA-Z0-9_]+',   # Hashtags
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Email addresses
        ]
        
        return any(re.search(pattern, text) for pattern in patterns)

    def _is_paragraph_end(self, text_blocks: List[TextBlock], current_idx: int) -> bool:
        """Determine if current block ends a paragraph based on spacing."""
        if current_idx >= len(text_blocks) - 1:
            return True
        
        current_block = text_blocks[current_idx]
        next_block = text_blocks[current_idx + 1]
        
        # Calculate vertical gap
        current_y = current_block.center_y
        next_y = next_block.center_y
        line_height = abs(next_y - current_y)
        
        # Significant gap indicates paragraph break
        return line_height > 40  # Adjustable threshold

    def _determine_header_level(self, text: str, confidence: float) -> int:
        """Determine appropriate header level (1-3) based on text characteristics."""
        if any(word in text.lower() for word in ['breaking', 'urgent', 'alert']):
            return 1  # H1 for urgent/breaking news
        elif any(word in text.lower() for word in ['headline', 'title', 'news']):
            return 2  # H2 for headlines
        elif confidence > 0.95 and len(text.split()) <= 3:
            return 2  # H2 for short, high-confidence headers
        else:
            return 3  # H3 for other headers

    def _summarize(self, result: OCRResults, **kwargs) -> Optional[MultimodalSequence]:
        """
        Summarize OCR results with enhanced analysis for fact-checking.
        """
        if not result.is_useful():
            return MultimodalSequence("No readable text detected in the image.")
        
        # Create comprehensive summary for fact-checking
        summary_parts = []
        
        # Add extraction summary with quality metrics
        summary_parts.append("**OCR Text Extraction Analysis:**")
        summary_parts.append(f"- **Text blocks detected:** {len(result.text_blocks)}")
        summary_parts.append(f"- **Average confidence:** {result._get_average_confidence():.3f}")
        summary_parts.append(f"- **OCR Language:** {self.lang}")
        summary_parts.append(f"- **OCR Version:** {self.ocr_version}")
        
        # Quality assessment for fact-checking reliability
        high_conf_blocks = [b for b in result.text_blocks if b.confidence > 0.9]
        medium_conf_blocks = [b for b in result.text_blocks if 0.7 <= b.confidence <= 0.9]
        low_conf_blocks = [b for b in result.text_blocks if b.confidence < 0.7]
        
        summary_parts.append(f"- **High confidence blocks (≥0.9):** {len(high_conf_blocks)}")
        summary_parts.append(f"- **Medium confidence blocks (0.7-0.9):** {len(medium_conf_blocks)}")
        summary_parts.append(f"- **Low confidence blocks (<0.7):** {len(low_conf_blocks)}")
        
        # Reliability assessment
        reliability_score = len(high_conf_blocks) / len(result.text_blocks) if result.text_blocks else 0
        if reliability_score >= 0.8:
            reliability = "HIGH"
        elif reliability_score >= 0.6:
            reliability = "MEDIUM"
        else:
            reliability = "LOW"
        
        summary_parts.append(f"- **Overall reliability:** {reliability} ({reliability_score:.2f})")
        
        # Key content preview (prioritize high-confidence text)
        if high_conf_blocks:
            preview_text = ' '.join(b.text for b in high_conf_blocks[:5])
            if len(preview_text) > 150:
                preview_text = preview_text[:150] + "..."
            summary_parts.append(f"- **High-confidence text preview:** \"{preview_text}\"")
        
        # Detect potential fact-checking indicators
        all_text = result.extracted_text.lower()
        fact_check_indicators = []
        
        if any(term in all_text for term in ['date', 'time', '2023', '2024', '2025']):
            fact_check_indicators.append("temporal references")
        if any(term in all_text for term in ['%', 'percent', 'million', 'billion', 'thousand']):
            fact_check_indicators.append("numerical claims")
        if any(term in all_text for term in ['said', 'according', 'reported', 'claimed']):
            fact_check_indicators.append("attributed statements")
        if any(term in all_text for term in ['http', 'www', '@', '#']):
            fact_check_indicators.append("digital references")
        
        if fact_check_indicators:
            summary_parts.append(f"- **Fact-check indicators:** {', '.join(fact_check_indicators)}")
        
        # Add structured content if available
        if result.markdown_text and len(result.markdown_text.strip()) > 0:
            summary_parts.append("\n**Structured Text Content:**")
            # Truncate very long content for summary
            content = result.markdown_text
            if len(content) > 1000:
                content = content[:1000] + "\n\n*[Content truncated for summary]*"
            summary_parts.append(content)
        
        # Add usage recommendations for fact-checkers
        summary_parts.append("\n**Fact-Checking Recommendations:**")
        if reliability_score >= 0.8:
            summary_parts.append("- ✅ High confidence OCR results - suitable for automated fact-checking")
        elif reliability_score >= 0.6:
            summary_parts.append("- ⚠️ Medium confidence OCR results - manual review recommended")
        else:
            summary_parts.append("- ❌ Low confidence OCR results - manual verification required")
        
        if low_conf_blocks:
            summary_parts.append(f"- 🔍 {len(low_conf_blocks)} low-confidence text blocks need verification")
        
        if fact_check_indicators:
            summary_parts.append("- 📊 Text contains verifiable claims and references")
        
        summary_text = '\n'.join(summary_parts)
        return MultimodalSequence(summary_text)
