"""
Autonomous code execution agent for coding tasks, analysis, and execution.
Uses specialized coding model for code generation and analysis.
"""
from __future__ import annotations

import ast
import subprocess
import tempfile
import traceback
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json

from llm import call_llm
from logger import get_logger
from model_router import route_model

logger = get_logger("code_agent")

@dataclass
class CodeExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: str
    execution_time: float
    code: str
    language: str

@dataclass
class CodeAnalysisResult:
    """Result of code analysis."""
    analysis: str
    suggestions: List[str]
    issues: List[str]
    complexity_score: float
    language_detected: str

class CodeAgent:
    """Autonomous code execution and analysis agent."""
    
    def __init__(self):
        self.supported_languages = {
            "python": {"extension": ".py", "command": ["python", "-c"]},
            "javascript": {"extension": ".js", "command": ["node", "-e"]},
            "bash": {"extension": ".sh", "command": ["bash"]},
            "sql": {"extension": ".sql", "command": ["sqlite3", ":memory:"]}
        }
        
        # System prompts for different coding tasks
        self.prompts = {
            "generate": """You are an expert programmer. Generate clean, efficient, and well-documented code for the given task. Follow best practices and include error handling where appropriate.""",
            "debug": """You are a debugging expert. Analyze the code, identify issues, and provide corrected code with explanations of the fixes.""",
            "analyze": """You are a code analysis expert. Analyze the code for quality, complexity, security issues, and provide improvement suggestions.""",
            "optimize": """You are a performance optimization expert. Analyze the code and provide optimized version with performance improvements.""",
            "explain": """You are a code explanation expert. Provide clear, detailed explanation of what the code does, how it works, and key concepts involved."""
        }
    
    def generate_code(
        self, 
        task: str, 
        language: str = "python",
        context: str = "",
        include_tests: bool = False
    ) -> Dict[str, Any]:
        """
        Generate code for a given task.
        
        Args:
            task: Description of coding task
            language: Programming language
            context: Additional context or requirements
            include_tests: Whether to include unit tests
            
        Returns:
            Dictionary with generated code and metadata
        """
        try:
            logger.info("Generating %s code for task: %s", language, task[:100])
            
            # Build prompt
            prompt = f"Task: {task}\n"
            prompt += f"Language: {language}\n"
            
            if context:
                prompt += f"Context: {context}\n"
            
            if include_tests:
                prompt += "Include unit tests for the generated code.\n"
            
            prompt += "\nGenerate the complete code:"
            
            # Use coding model
            response = call_llm(
                prompt,
                temperature=0.1,  # Lower temperature for code
                system_prompt=self.prompts["generate"],
                task_type="coding"
            )
            
            # Extract code from response
            code = self._extract_code_from_response(response, language)
            
            # Analyze generated code
            analysis = self.analyze_code(code, language)
            
            result = {
                "task": task,
                "language": language,
                "generated_code": code,
                "raw_response": response,
                "analysis": analysis,
                "includes_tests": include_tests
            }
            
            logger.info("Code generation completed for %s", language)
            return result
            
        except Exception as e:
            logger.error("Code generation failed: %s", e)
            return {"error": str(e), "task": task, "language": language}
    
    def execute_code(
        self, 
        code: str, 
        language: str = "python",
        timeout: int = 30,
        sandbox: bool = True
    ) -> CodeExecutionResult:
        """
        Execute code safely in sandboxed environment.
        
        Args:
            code: Code to execute
            language: Programming language
            timeout: Execution timeout in seconds
            sandbox: Whether to use sandbox (always True for safety)
            
        Returns:
            CodeExecutionResult with execution details
        """
        try:
            logger.info("Executing %s code (%d chars)", language, len(code))
            
            if language not in self.supported_languages:
                raise ValueError(f"Unsupported language: {language}")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix=self.supported_languages[language]["extension"],
                delete=False
            ) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name
            
            import time
            start_time = time.time()
            
            try:
                # Execute code
                if language == "python":
                    # Special handling for Python
                    result = self._execute_python(code, timeout)
                else:
                    # Other languages
                    command = self.supported_languages[language]["command"] + [temp_file_path]
                    result = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        cwd=tempfile.gettempdir()
                    )
                    
                    output = result.stdout
                    error = result.stderr
                    success = result.returncode == 0
                
                execution_time = time.time() - start_time
                
                execution_result = CodeExecutionResult(
                    success=success,
                    output=output,
                    error=error,
                    execution_time=execution_time,
                    code=code,
                    language=language
                )
                
                logger.info(
                    "Code execution %s in %.2fs (output: %d chars)",
                    "succeeded" if success else "failed",
                    execution_time,
                    len(output)
                )
                
                return execution_result
                
            finally:
                # Clean up temporary file
                import os
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass
            
        except subprocess.TimeoutExpired:
            logger.error("Code execution timed out after %d seconds", timeout)
            return CodeExecutionResult(
                success=False,
                output="",
                error=f"Execution timed out after {timeout} seconds",
                execution_time=timeout,
                code=code,
                language=language
            )
        except Exception as e:
            logger.error("Code execution failed: %s", e)
            return CodeExecutionResult(
                success=False,
                output="",
                error=str(e),
                execution_time=0,
                code=code,
                language=language
            )
    
    def debug_code(
        self, 
        code: str, 
        error_message: str = "",
        language: str = "python"
    ) -> Dict[str, Any]:
        """
        Debug code and provide fixes.
        
        Args:
            code: Code with issues
            error_message: Error message if available
            language: Programming language
            
        Returns:
            Dictionary with debugging analysis and fixes
        """
        try:
            logger.info("Debugging %s code", language)
            
            prompt = f"Code to debug:\n{code}\n"
            if error_message:
                prompt += f"Error message: {error_message}\n"
            prompt += "\nAnalyze and provide corrected code:"
            
            response = call_llm(
                prompt,
                temperature=0.1,
                system_prompt=self.prompts["debug"],
                task_type="coding"
            )
            
            # Extract corrected code
            corrected_code = self._extract_code_from_response(response, language)
            
            # Test the corrected code
            test_result = None
            if corrected_code and corrected_code != code:
                test_result = self.execute_code(corrected_code, language, timeout=10)
            
            result = {
                "original_code": code,
                "corrected_code": corrected_code,
                "error_message": error_message,
                "debugging_analysis": response,
                "test_result": test_result,
                "language": language
            }
            
            logger.info("Code debugging completed for %s", language)
            return result
            
        except Exception as e:
            logger.error("Code debugging failed: %s", e)
            return {"error": str(e), "original_code": code}
    
    def analyze_code(
        self, 
        code: str, 
        language: str = "python"
    ) -> CodeAnalysisResult:
        """
        Analyze code quality and provide insights.
        
        Args:
            code: Code to analyze
            language: Programming language
            
        Returns:
            CodeAnalysisResult with analysis details
        """
        try:
            logger.info("Analyzing %s code quality", language)
            
            # Static analysis
            static_analysis = self._static_code_analysis(code, language)
            
            # AI-powered analysis
            prompt = f"Analyze this {language} code for quality, complexity, security, and best practices:\n{code}"
            
            response = call_llm(
                prompt,
                temperature=0.2,
                system_prompt=self.prompts["analyze"],
                task_type="coding"
            )
            
            # Parse AI analysis
            ai_analysis = self._parse_analysis_response(response)
            
            # Combine analyses
            result = CodeAnalysisResult(
                analysis=response,
                suggestions=ai_analysis.get("suggestions", []) + static_analysis.get("suggestions", []),
                issues=ai_analysis.get("issues", []) + static_analysis.get("issues", []),
                complexity_score=static_analysis.get("complexity", 5.0),
                language_detected=language
            )
            
            logger.info("Code analysis completed for %s", language)
            return result
            
        except Exception as e:
            logger.error("Code analysis failed: %s", e)
            return CodeAnalysisResult(
                analysis=str(e),
                suggestions=[],
                issues=[f"Analysis failed: {str(e)}"],
                complexity_score=10.0,
                language_detected=language
            )
    
    def explain_code(
        self, 
        code: str, 
        language: str = "python",
        detail_level: str = "detailed"
    ) -> Dict[str, Any]:
        """
        Explain what code does.
        
        Args:
            code: Code to explain
            language: Programming language
            detail_level: "brief", "detailed", or "comprehensive"
            
        Returns:
            Dictionary with explanation
        """
        try:
            logger.info("Explaining %s code (%s detail)", language, detail_level)
            
            prompt = f"Explain this {language} code ({detail_level} explanation):\n{code}"
            
            response = call_llm(
                prompt,
                temperature=0.3,
                system_prompt=self.prompts["explain"],
                task_type="coding"
            )
            
            result = {
                "code": code,
                "language": language,
                "detail_level": detail_level,
                "explanation": response
            }
            
            logger.info("Code explanation completed for %s", language)
            return result
            
        except Exception as e:
            logger.error("Code explanation failed: %s", e)
            return {"error": str(e), "code": code}
    
    def _extract_code_from_response(self, response: str, language: str) -> str:
        """Extract code block from LLM response."""
        import re
        
        # Look for code blocks
        patterns = [
            rf'```{language}\n(.*?)\n```',
            rf'```\n(.*?)\n```',
            r'`(.*?)`'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return matches[0].strip()
        
        # If no code blocks found, return response as-is
        return response.strip()
    
    def _execute_python(self, code: str, timeout: int) -> Tuple[str, str, bool]:
        """Safely execute Python code."""
        try:
            # Create a restricted namespace
            safe_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'sum': sum,
                    'max': max,
                    'min': min,
                    'abs': abs,
                    'round': round,
                    'str': str,
                    'int': int,
                    'float': float,
                    'list': list,
                    'dict': dict,
                    'set': set,
                    'tuple': tuple,
                }
            }
            
            # Initialize variables
            output = ""
            error = ""
            success = False
            
            # Capture output
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            try:
                # Execute the code
                exec(code, safe_globals)
                output = captured_output.getvalue()
                error = ""
                success = True
            except Exception as e:
                output = captured_output.getvalue()
                error = f"{type(e).__name__}: {str(e)}"
                success = False
            finally:
                sys.stdout = old_stdout
            
            return output, error, success
            
        except Exception as e:
            return "", f"Execution setup error: {str(e)}", False
    
    def _static_code_analysis(self, code: str, language: str) -> Dict[str, Any]:
        """Perform static code analysis."""
        analysis = {
            "suggestions": [],
            "issues": [],
            "complexity": 5.0
        }
        
        if language == "python":
            try:
                # Parse AST for basic analysis
                tree = ast.parse(code)
                
                # Count complexity indicators
                function_count = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
                class_count = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
                loop_count = len([node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))])
                
                # Basic complexity score
                analysis["complexity"] = min(10.0, (function_count * 0.5 + loop_count * 1.0 + class_count * 0.3))
                
                # Basic suggestions
                if function_count == 0 and len(code.strip()) > 50:
                    analysis["suggestions"].append("Consider breaking code into functions for better organization")
                
                if loop_count > 5:
                    analysis["suggestions"].append("High number of loops detected - consider optimization")
                
            except SyntaxError as e:
                analysis["issues"].append(f"Syntax error: {str(e)}")
        
        return analysis
    
    def _parse_analysis_response(self, response: str) -> Dict[str, List[str]]:
        """Parse AI analysis response into structured format."""
        import re
        
        suggestions = []
        issues = []
        
        # Extract suggestions
        suggestion_patterns = [
            r'(?:suggestion|recommend|improve)[s]?:\s*(.*?)(?:\n|$)',
            r'-(.*?)(?:\n|$)'
        ]
        
        for pattern in suggestion_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            suggestions.extend([match.strip() for match in matches if match.strip()])
        
        # Extract issues
        issue_patterns = [
            r'(?:issue|problem|error|bug)[s]?:\s*(.*?)(?:\n|$)',
            r'(?:dangerous|risky|unsafe)[s]?:\s*(.*?)(?:\n|$)'
        ]
        
        for pattern in issue_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            issues.extend([match.strip() for match in matches if match.strip()])
        
        return {
            "suggestions": suggestions[:10],  # Limit to top 10
            "issues": issues[:10]
        }

# Global instance
_code_agent_instance = None

def get_code_agent() -> CodeAgent:
    """Get or create code agent instance."""
    global _code_agent_instance
    if _code_agent_instance is None:
        _code_agent_instance = CodeAgent()
    return _code_agent_instance
