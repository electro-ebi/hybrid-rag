"""
Advanced data analysis module for extracting insights from structured and unstructured data.
Combines OCR, vision analysis, and statistical analysis.
"""
from __future__ import annotations

import re
import statistics
from typing import List, Dict, Any, Union, Tuple
from dataclasses import dataclass

import numpy as np

from logger import get_logger

logger = get_logger("data_analyzer")

@dataclass
class DataPoint:
    """Structured data point extracted from content."""
    label: str
    value: Union[float, int, str]
    confidence: float
    source: str
    context: str

@dataclass
class DataInsight:
    """Insight extracted from data analysis."""
    insight_type: str
    description: str
    confidence: float
    supporting_data: List[DataPoint]

class DataAnalyzer:
    """Advanced data analysis for multi-modal content."""
    
    def __init__(self):
        # Simplified data analyzer without legacy dependencies
        self.name = "DataAnalyzer"
        
        # Pattern matching for different data types
        self.patterns = {
            'percentage': r'(\d+\.?\d*)\s*%|percent',
            'currency': r'\$?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            'number': r'(\d+\.?\d*)',
            'date': r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})',
            'year': r'\b(19|20)\d{2}\b'
        }
    
    def analyze_document_data(
        self,
        text_content: str = "",
        image_path: str = None,
        image_base64: str = None,
        analysis_depth: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Comprehensive data analysis of document content.
        
        Args:
            text_content: Text from document
            image_path: Path to document image
            image_base64: Base64 encoded image
            analysis_depth: "quick", "standard", or "comprehensive"
            
        Returns:
            Dictionary with analysis results
        """
        try:
            logger.info("Starting %s data analysis", analysis_depth)
            
            results = {
                "analysis_depth": analysis_depth,
                "extracted_data": [],
                "insights": [],
                "statistics": {},
                "data_quality": {},
                "recommendations": []
            }
            
            # Analyze text content
            if text_content:
                text_analysis = self._analyze_text_data(text_content)
                results["extracted_data"].extend(text_analysis["data_points"])
                results["insights"].extend(text_analysis["insights"])
            
            # Analyze image content if provided
            if image_path or image_base64:
                image_analysis = self._analyze_image_data(
                    image_path, image_base64, analysis_depth
                )
                results["extracted_data"].extend(image_analysis["data_points"])
                results["insights"].extend(image_analysis["insights"])
                results["visual_analysis"] = image_analysis["visual_analysis"]
            
            # Compute statistics
            results["statistics"] = self._compute_statistics(results["extracted_data"])
            
            # Assess data quality
            results["data_quality"] = self._assess_data_quality(results["extracted_data"])
            
            # Generate recommendations
            results["recommendations"] = self._generate_recommendations(results)
            
            logger.info(
                "Data analysis complete: %d data points, %d insights",
                len(results["extracted_data"]),
                len(results["insights"])
            )
            
            return results
            
        except Exception as e:
            logger.error("Data analysis failed: %s", e)
            return {"error": str(e)}
    
    def _analyze_text_data(self, text: str) -> Dict[str, Any]:
        """Extract and analyze data from text content."""
        data_points = []
        insights = []
        
        # Extract different types of data
        for data_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                try:
                    if data_type == 'percentage':
                        value = float(match.group(1)) / 100.0
                        label = f"Percentage_{len(data_points)}"
                    elif data_type == 'currency':
                        value_str = match.group(1).replace(',', '')
                        value = float(value_str)
                        label = f"Currency_{len(data_points)}"
                    elif data_type == 'number':
                        value = float(match.group(1))
                        label = f"Number_{len(data_points)}"
                    elif data_type == 'date':
                        value = match.group(1)
                        label = f"Date_{len(data_points)}"
                    elif data_type == 'year':
                        value = int(match.group(1))
                        label = f"Year_{len(data_points)}"
                    else:
                        continue
                    
                    # Get context around the match
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end].strip()
                    
                    data_point = DataPoint(
                        label=label,
                        value=value,
                        confidence=0.8,  # Base confidence for regex extraction
                        source="text_analysis",
                        context=context
                    )
                    
                    data_points.append(data_point)
                    
                except (ValueError, IndexError) as e:
                    logger.debug("Failed to parse match: %s", e)
                    continue
        
        # Generate insights from extracted data
        insights = self._generate_text_insights(data_points, text)
        
        return {
            "data_points": [self._data_point_to_dict(dp) for dp in data_points],
            "insights": insights
        }
    
    def _analyze_image_data(
        self, 
        image_path: str, 
        image_base64: str,
        analysis_depth: str
    ) -> Dict[str, Any]:
        """Extract and analyze data from image content."""
        data_points = []
        insights = []
        visual_analysis = {}
        
        try:
            # OCR extraction
            ocr_result = self.ocr_processor.extract_structured_data(
                image_path=image_path,
                image_base64=image_base64
            )
            
            # Convert OCR results to data points
            for element in ocr_result.get("structured_data", []):
                if element["type"] in ["number", "percentage"]:
                    try:
                        value = float(re.sub(r'[^\d.]', '', element["text"]))
                        data_point = DataPoint(
                            label=f"OCR_{element['type']}_{len(data_points)}",
                            value=value,
                            confidence=element["confidence"] / 100.0,
                            source="ocr_analysis",
                            context=element["text"]
                        )
                        data_points.append(data_point)
                    except ValueError:
                        continue
            
            # Simplified vision analysis (placeholder)
            if analysis_depth in ["standard", "comprehensive"]:
                # Placeholder for vision analysis
                visual_analysis = {
                    "analysis_type": "basic",
                    "extracted_data": {},
                    "confidence": 0.5
                }
                
                # Create a basic data point for vision analysis
                data_point = DataPoint(
                    label="vision_analysis",
                    value="basic_vision_processing",
                    confidence=0.5,
                    source="vision_analysis",
                    context="Image processed with basic vision analysis"
                )
                data_points.append(data_point)
            
            # Generate insights
            insights = self._generate_image_insights(data_points, visual_analysis)
            
        except Exception as e:
            logger.error("Image data analysis failed: %s", e)
        
        return {
            "data_points": [self._data_point_to_dict(dp) for dp in data_points],
            "insights": insights,
            "visual_analysis": visual_analysis
        }
    
    def _compute_statistics(self, data_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute statistical measures from extracted data."""
        stats = {
            "total_data_points": len(data_points),
            "numeric_data_points": 0,
            "data_types": {},
            "value_ranges": {},
            "averages": {}
        }
        
        # Separate numeric data
        numeric_values = []
        for dp in data_points:
            if isinstance(dp["value"], (int, float)):
                numeric_values.append(dp["value"])
                stats["numeric_data_points"] += 1
            
            # Count data types by source
            source = dp["source"]
            stats["data_types"][source] = stats["data_types"].get(source, 0) + 1
        
        # Compute statistics for numeric data
        if numeric_values:
            stats["value_ranges"] = {
                "min": min(numeric_values),
                "max": max(numeric_values),
                "range": max(numeric_values) - min(numeric_values)
            }
            
            stats["averages"] = {
                "mean": statistics.mean(numeric_values),
                "median": statistics.median(numeric_values),
                "std_dev": statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0
            }
        
        return stats
    
    def _assess_data_quality(self, data_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess quality of extracted data."""
        quality = {
            "overall_score": 0.0,
            "completeness": 0.0,
            "accuracy": 0.0,
            "consistency": 0.0,
            "issues": []
        }
        
        if not data_points:
            quality["issues"].append("No data points extracted")
            return quality
        
        # Completeness: ratio of successful extractions
        quality["completeness"] = len(data_points) / max(len(data_points), 1)
        
        # Accuracy: average confidence
        confidences = [dp["confidence"] for dp in data_points]
        quality["accuracy"] = sum(confidences) / len(confidences) if confidences else 0
        
        # Consistency: check for duplicate or conflicting data
        labels = [dp["label"] for dp in data_points]
        unique_labels = set(labels)
        quality["consistency"] = len(unique_labels) / len(labels) if labels else 0
        
        # Overall score
        quality["overall_score"] = (
            quality["completeness"] * 0.3 +
            quality["accuracy"] * 0.4 +
            quality["consistency"] * 0.3
        )
        
        # Identify issues
        if quality["accuracy"] < 0.7:
            quality["issues"].append("Low extraction accuracy")
        if quality["consistency"] < 0.8:
            quality["issues"].append("Data inconsistency detected")
        if len(data_points) < 3:
            quality["issues"].append("Limited data extracted")
        
        return quality
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Data quality recommendations
        quality = analysis_results.get("data_quality", {})
        if quality.get("accuracy", 0) < 0.7:
            recommendations.append("Consider manual verification of extracted data due to low accuracy")
        
        if quality.get("completeness", 0) < 0.5:
            recommendations.append("Document may contain additional data that wasn't extracted")
        
        # Statistical insights
        stats = analysis_results.get("statistics", {})
        if stats.get("numeric_data_points", 0) > 5:
            std_dev = stats.get("averages", {}).get("std_dev", 0)
            if std_dev > stats.get("averages", {}).get("mean", 0):
                recommendations.append("High data variability detected - investigate outliers")
        
        # Content-specific recommendations
        if "visual_analysis" in analysis_results:
            chart_type = analysis_results["visual_analysis"].get("extracted_data", {}).get("chart_type", "")
            if chart_type in ["line", "bar"]:
                recommendations.append(f"Consider time series analysis for {chart_type} chart data")
        
        return recommendations
    
    def _generate_text_insights(self, data_points: List[DataPoint], text: str) -> List[Dict[str, Any]]:
        """Generate insights from text-based data extraction."""
        insights = []
        
        # Look for patterns in the data
        numeric_data = [dp for dp in data_points if isinstance(dp.value, (int, float))]
        
        if numeric_data:
            # Trend analysis
            values = [dp.value for dp in numeric_data]
            if len(values) > 2:
                if values[-1] > values[0]:
                    insights.append({
                        "type": "trend",
                        "description": "Upward trend detected in numerical data",
                        "confidence": 0.6,
                        "data_points": [self._data_point_to_dict(dp) for dp in numeric_data[-3:]]
                    })
        
        # Context-based insights
        if any("revenue" in dp.context.lower() for dp in data_points):
            revenue_data = [dp for dp in data_points if "revenue" in dp.context.lower()]
            if revenue_data:
                insights.append({
                    "type": "financial",
                    "description": "Financial data detected in document",
                    "confidence": 0.8,
                    "data_points": [self._data_point_to_dict(dp) for dp in revenue_data]
                })
        
        return insights
    
    def _generate_image_insights(
        self, 
        data_points: List[DataPoint], 
        visual_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate insights from image-based data extraction."""
        insights = []
        
        # Chart type insights
        chart_type = visual_analysis.get("extracted_data", {}).get("chart_type", "")
        if chart_type != "unknown":
            insights.append({
                "type": "chart_identification",
                "description": f"Document contains {chart_type} visualization",
                "confidence": 0.7,
                "data_points": []
            })
        
        # Data volume insights
        if len(data_points) > 10:
            insights.append({
                "type": "data_richness",
                "description": "Rich data visualization with multiple data points",
                "confidence": 0.8,
                "data_points": [self._data_point_to_dict(dp) for dp in data_points[:5]]
            })
        
        return insights
    
    def _data_point_to_dict(self, data_point: DataPoint) -> Dict[str, Any]:
        """Convert DataPoint to dictionary for serialization."""
        return {
            "label": data_point.label,
            "value": data_point.value,
            "confidence": data_point.confidence,
            "source": data_point.source,
            "context": data_point.context
        }

# Global instance
_data_analyzer_instance = None

def get_data_analyzer() -> DataAnalyzer:
    """Get or create data analyzer instance."""
    global _data_analyzer_instance
    if _data_analyzer_instance is None:
        _data_analyzer_instance = DataAnalyzer()
    return _data_analyzer_instance
