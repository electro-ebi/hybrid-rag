"""
Multi-modal ingestion pipeline for text, images, charts, and complex documents.
Handles PDFs with images, OCR, vision analysis, and data extraction.
"""
from __future__ import annotations

import os
from typing import Dict, Any, List, Optional

from logger import get_logger
from .loaders.loader_factory import load_file
from .preprocess import clean_text
from .pdf_image_extractor import get_pdf_extractor
from .ocr_processor import get_ocr_processor
from .vision_analyzer import get_vision_analyzer
from .data_analyzer import get_data_analyzer

logger = get_logger("multimodal_ingest")

class MultiModalIngestor:
    """Advanced ingestion for multi-modal document processing."""
    
    def __init__(self):
        self.pdf_extractor = get_pdf_extractor()
        self.ocr_processor = get_ocr_processor()
        self.vision_analyzer = get_vision_analyzer()
        self.data_analyzer = get_data_analyzer()
    
    def ingest_file(
        self, 
        file_path: str, 
        extraction_level: str = "comprehensive",
        include_vision_analysis: bool = True,
        include_data_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest file with multi-modal content extraction.
        
        Args:
            file_path: Path to file
            extraction_level: "basic", "standard", or "comprehensive"
            include_vision_analysis: Include vision model analysis
            include_data_analysis: Include data analysis and insights
            
        Returns:
            Dictionary with extracted content and metadata
        """
        try:
            logger.info("Multi-modal ingestion: %s (level: %s)", file_path, extraction_level)
            
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Route to appropriate processor
            if file_ext == '.pdf':
                return self._ingest_pdf(
                    file_path, extraction_level, include_vision_analysis, include_data_analysis
                )
            elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                return self._ingest_image(
                    file_path, extraction_level, include_vision_analysis, include_data_analysis
                )
            else:
                return self._ingest_text_file(
                    file_path, extraction_level, include_data_analysis
                )
                
        except Exception as e:
            logger.error("Multi-modal ingestion failed for %s: %s", file_path, e)
            return {"error": str(e), "file_path": file_path}
    
    def _ingest_pdf(
        self,
        pdf_path: str,
        extraction_level: str,
        include_vision_analysis: bool,
        include_data_analysis: bool
    ) -> Dict[str, Any]:
        """Ingest PDF with full multi-modal processing."""
        
        # Extract images and text from PDF
        pdf_result = self.pdf_extractor.extract_from_pdf_path(
            pdf_path,
            extract_images=True,
            extract_text=True,
            analyze_structure=True
        )
        
        if "error" in pdf_result:
            return pdf_result
        
        # Prepare result structure
        result = {
            "file_type": "pdf",
            "file_path": pdf_path,
            "extraction_level": extraction_level,
            "content": {
                "text": "",
                "pages": [],
                "images": [],
                "structured_data": []
            },
            "metadata": {
                "total_pages": pdf_result["total_pages"],
                "content_summary": pdf_result["summary"]
            },
            "analysis": {}
        }
        
        # Process each page
        all_text = []
        charts_and_graphs = []
        
        for page in pdf_result["pages"]:
            page_content = {
                "page_number": page["page_number"],
                "text": page.get("text", ""),
                "content_type": page.get("content_type", "text"),
                "word_count": page.get("word_count", 0),
                "confidence": page.get("ocr_confidence", 0.0)
            }
            
            # Add image if available
            if page.get("image_base64"):
                page_content["image_base64"] = page["image_base64"]
                result["content"]["images"].append({
                    "page_number": page["page_number"],
                    "image_base64": page["image_base64"]
                })
                
                # Vision analysis for images
                if include_vision_analysis and extraction_level in ["standard", "comprehensive"]:
                    vision_result = self.vision_analyzer.analyze_chart_or_graph(
                        image_base64=page["image_base64"],
                        content_type=page.get("content_type", "general")
                    )
                    page_content["vision_analysis"] = vision_result
                    
                    # Track charts and graphs
                    if page.get("content_type") in ["chart_data", "table"]:
                        charts_and_graphs.append({
                            "page_number": page["page_number"],
                            "analysis": vision_result
                        })
            
            # Add structured data
            if page.get("structured_data"):
                page_content["structured_data"] = page["structured_data"]
                result["content"]["structured_data"].extend(page["structured_data"])
            
            result["content"]["pages"].append(page_content)
            all_text.append(page.get("text", ""))
        
        # Combine all text
        result["content"]["text"] = "\n\n".join(all_text)
        
        # Data analysis
        if include_data_analysis and extraction_level == "comprehensive":
            data_analysis = self.data_analyzer.analyze_document_data(
                text_content=result["content"]["text"],
                analysis_depth="comprehensive"
            )
            result["analysis"]["data_analysis"] = data_analysis
        
        # Vision analysis summary
        if include_vision_analysis and charts_and_graphs:
            result["analysis"]["charts_and_graphs"] = charts_and_graphs
            result["analysis"]["visualization_summary"] = {
                "total_charts": len(charts_and_graphs),
                "chart_types": list(set(
                    chart["analysis"].get("content_type", "unknown") 
                    for chart in charts_and_graphs
                ))
            }
        
        # Clean main text
        result["content"]["text"] = clean_text(result["content"]["text"])
        
        logger.info(
            "PDF ingestion complete: %d pages, %d images, %d charts",
            len(result["content"]["pages"]),
            len(result["content"]["images"]),
            len(charts_and_graphs)
        )
        
        return result
    
    def _ingest_image(
        self,
        image_path: str,
        extraction_level: str,
        include_vision_analysis: bool,
        include_data_analysis: bool
    ) -> Dict[str, Any]:
        """Ingest image with OCR and vision analysis."""
        
        result = {
            "file_type": "image",
            "file_path": image_path,
            "extraction_level": extraction_level,
            "content": {
                "text": "",
                "image_base64": ""
            },
            "metadata": {},
            "analysis": {}
        }
        
        # OCR extraction
        ocr_result = self.ocr_processor.extract_text_from_image(
            image_path=image_path,
            preprocess=True,
            dense_text=False
        )
        
        result["content"]["text"] = clean_text(ocr_result.get("text", ""))
        result["metadata"]["ocr_confidence"] = ocr_result.get("confidence", 0.0)
        result["metadata"]["word_count"] = ocr_result.get("word_count", 0)
        
        # Load image for vision analysis
        with open(image_path, 'rb') as f:
            import base64
            result["content"]["image_base64"] = base64.b64encode(f.read()).decode('utf-8')
        
        # Vision analysis
        if include_vision_analysis:
            vision_result = self.vision_analyzer.analyze_chart_or_graph(
                image_path=image_path,
                content_type="general"
            )
            result["analysis"]["vision_analysis"] = vision_result
            
            # Data extraction if it looks like a chart
            if vision_result.get("structured_insights", {}).get("data_mentioned"):
                data_result = self.vision_analyzer.extract_data_from_visualization(
                    image_path=image_path
                )
                result["analysis"]["data_extraction"] = data_result
        
        # Data analysis
        if include_data_analysis and extraction_level == "comprehensive":
            data_analysis = self.data_analyzer.analyze_document_data(
                text_content=result["content"]["text"],
                image_path=image_path,
                analysis_depth="comprehensive"
            )
            result["analysis"]["data_analysis"] = data_analysis
        
        logger.info(
            "Image ingestion complete: %d words, confidence: %.1f%%",
            result["metadata"]["word_count"],
            result["metadata"]["ocr_confidence"]
        )
        
        return result
    
    def _ingest_text_file(
        self,
        file_path: str,
        extraction_level: str,
        include_data_analysis: bool
    ) -> Dict[str, Any]:
        """Ingest text file with data analysis."""
        
        # Use existing loader
        basic_result = load_file(file_path)
        
        result = {
            "file_type": "text",
            "file_path": file_path,
            "extraction_level": extraction_level,
            "content": {
                "text": clean_text(basic_result["content"])
            },
            "metadata": basic_result["metadata"],
            "analysis": {}
        }
        
        # Data analysis
        if include_data_analysis and extraction_level == "comprehensive":
            data_analysis = self.data_analyzer.analyze_document_data(
                text_content=result["content"]["text"],
                analysis_depth="comprehensive"
            )
            result["analysis"]["data_analysis"] = data_analysis
        
        logger.info("Text file ingestion complete: %d characters", len(result["content"]["text"]))
        
        return result
    
    def create_searchable_chunks(
        self, 
        ingestion_result: Dict[str, Any],
        chunk_size: int = 1000,
        overlap: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Create searchable chunks from multi-modal ingestion result.
        
        Args:
            ingestion_result: Result from ingest_file
            chunk_size: Maximum characters per chunk
            overlap: Character overlap between chunks
            
        Returns:
            List of searchable chunks with rich metadata
        """
        chunks = []
        
        # Get main text
        text = ingestion_result["content"]["text"]
        file_path = ingestion_result["file_path"]
        file_type = ingestion_result["file_type"]
        
        # Create text chunks
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            
            chunk = {
                "text": chunk_text,
                "file_path": file_path,
                "file_type": file_type,
                "chunk_index": i // (chunk_size - overlap),
                "chunk_type": "text",
                "metadata": {
                    "extraction_level": ingestion_result["extraction_level"],
                    "has_images": len(ingestion_result["content"].get("images", [])) > 0,
                    "has_structured_data": len(ingestion_result["content"].get("structured_data", [])) > 0
                }
            }
            
            # Add page-specific metadata for PDFs
            if file_type == "pdf" and "pages" in ingestion_result["content"]:
                # Find which page this chunk likely belongs to
                page_num = self._find_page_for_chunk(chunk_text, ingestion_result["content"]["pages"])
                if page_num:
                    chunk["metadata"]["page_number"] = page_num
            
            chunks.append(chunk)
        
        # Add image-based chunks
        for img in ingestion_result["content"].get("images", []):
            img_chunk = {
                "text": f"[Image content from page {img['page_number']}]",
                "file_path": file_path,
                "file_type": file_type,
                "chunk_index": len(chunks),
                "chunk_type": "image",
                "metadata": {
                    "page_number": img["page_number"],
                    "image_base64": img["image_base64"],
                    "has_visual_content": True
                }
            }
            
            # Add vision analysis if available
            if "analysis" in ingestion_result:
                vision_analysis = ingestion_result["analysis"].get("charts_and_graphs", [])
                for chart in vision_analysis:
                    if chart["page_number"] == img["page_number"]:
                        img_chunk["metadata"]["vision_analysis"] = chart["analysis"]
                        break
            
            chunks.append(img_chunk)
        
        logger.info("Created %d searchable chunks from %s", len(chunks), file_path)
        return chunks
    
    def _find_page_for_chunk(self, chunk_text: str, pages: List[Dict[str, Any]]) -> Optional[int]:
        """Find which page a chunk most likely belongs to."""
        if not pages:
            return None
        
        # Simple matching - look for unique text snippets
        for page in pages:
            page_text = page.get("text", "")
            if page_text and chunk_text[:100] in page_text:
                return page["page_number"]
        
        return None

# Global instance
_multimodal_ingestor_instance = None

def get_multimodal_ingestor() -> MultiModalIngestor:
    """Get or create multi-modal ingestor instance."""
    global _multimodal_ingestor_instance
    if _multimodal_ingestor_instance is None:
        _multimodal_ingestor_instance = MultiModalIngestor()
    return _multimodal_ingestor_instance

# Convenience function
def ingest_file_multimodal(file_path: str, extraction_level: str = "comprehensive") -> Dict[str, Any]:
    """
    Convenience function for multi-modal file ingestion.
    
    Args:
        file_path: Path to file
        extraction_level: "basic", "standard", or "comprehensive"
        
    Returns:
        Dictionary with extracted content and analysis
    """
    ingestor = get_multimodal_ingestor()
    return ingestor.ingest_file(
        file_path=file_path,
        extraction_level=extraction_level,
        include_vision_analysis=True,
        include_data_analysis=True
    )
