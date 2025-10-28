"""
COMPILER AS REFERENCE MODEL - Wild Idea Test
Using Python's native intelligence to guide LLM learning
"""

import ast
import subprocess
import tempfile
import os

class CompilerReference:
    def __init__(self):
        print("🚀 Testing: Python Compiler as Reference Model")
        
    def analyze_code_difficulty(self, code_snippet):
        """Use Python compiler to analyze code complexity"""
        difficulties = []
        
        try:
            # Parse AST to understand code structure
            tree = ast.parse(code_snippet)
            
            # Analyze complexity metrics
            complexity_score = self.calculate_ast_complexity(tree)
            
            # Try to compile (basic syntax check)
            compile(code_snippet, '<string>', 'exec')
            compile_success = True
            
        except SyntaxError as e:
            complexity_score = 100  # High difficulty if syntax errors
            compile_success = False
        except Exception as e:
            complexity_score = 50   # Medium difficulty for other issues
            compile_success = False
            
        return {
            "complexity": complexity_score,
            "compiles": compile_success,
            "tokens": code_snippet.split()  # Simple tokenization
        }
    
    def calculate_ast_complexity(self, tree):
        """Calculate complexity from AST structure"""
        complexity = 0
        
        for node in ast.walk(tree):
            # Different node types = different complexity
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                complexity += 10
            elif isinstance(node, (ast.For, ast.While, ast.If)):
                complexity += 5
            elif isinstance(node, (ast.ListComp, ast.DictComp)):
                complexity += 8
            elif isinstance(node, ast.Call):
                complexity += 3
                
        return complexity
    
    def test_compiler_reference(self):
        """Test the compiler reference concept"""
        test_codes = [
            "x = 5",  # Simple
            "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",  # Complex
            "for i in range(10): print(i)",  # Medium
            "invalid code syntax error here"  # Should fail
        ]
        
        print("🧪 Testing Compiler Reference Model")
        for code in test_codes:
            analysis = self.analyze_code_difficulty(code)
            print(f"   '{code[:30]}...' -> Complexity: {analysis['complexity']}, Compiles: {analysis['compiles']}")

# Test it
if __name__ == "__main__":
    compiler_ref = CompilerReference()
    compiler_ref.test_compiler_reference()