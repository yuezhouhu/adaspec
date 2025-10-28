"""
REFERENCE COMMITTEE: Domain-Intelligent Ensemble
A swarm of specialized experts that vote on token difficulty
"""

import ast
import subprocess
import tempfile
import re
from typing import Dict, List

class ReferenceCommittee:
    def __init__(self, domain="python"):
        self.domain = domain
        self.experts = self._initialize_experts()
        print(f"🚀 INITIALIZING REFERENCE COMMITTEE for {domain.upper()}")
        
    def _initialize_experts(self):
        """Initialize domain-specific expert analyzers"""
        return {
            'syntax_judge': self.syntax_expert,
            'complexity_analyst': self.complexity_expert, 
            'runtime_profiler': self.runtime_expert,
            'style_critic': self.style_expert,
            'security_auditor': self.security_expert
        }
    
    def syntax_expert(self, code: str) -> Dict:
        """Expert: Python syntax validity"""
        try:
            ast.parse(code)
            compile(code, '<string>', 'exec')
            return {'score': 0, 'confidence': 1.0, 'reason': 'Flawless syntax'}
        except SyntaxError as e:
            return {'score': 100, 'confidence': 1.0, 'reason': f'Syntax error: {e}'}
        except Exception as e:
            return {'score': 50, 'confidence': 0.8, 'reason': f'Complex syntax: {e}'}
    
    def complexity_expert(self, code: str) -> Dict:
        """Expert: Code structural complexity"""
        try:
            tree = ast.parse(code)
            complexity = 0
            
            # Advanced complexity analysis
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity += 15  # Functions are complex
                    # Analyze function body complexity
                    for stmt in node.body:
                        if isinstance(stmt, (ast.For, ast.While, ast.If)):
                            complexity += 8
                elif isinstance(node, (ast.ClassDef)):
                    complexity += 25  # Classes are very complex
                elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp)):
                    complexity += 12  # Comprehensions are complex
                elif isinstance(node, ast.Call):
                    complexity += 3   # Function calls
            
            # Normalize score
            normalized = min(100, complexity)
            confidence = 0.9 if complexity > 0 else 0.7
            
            return {'score': normalized, 'confidence': confidence, 'reason': f'Structural complexity: {complexity}'}
            
        except Exception:
            return {'score': 100, 'confidence': 0.6, 'reason': 'Cannot analyze structure'}
    
    def runtime_expert(self, code: str) -> Dict:
        """Expert: Execution behavior analysis"""
        try:
            # Create a safe execution environment
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(f"""
import time
def safe_execution():
    start = time.time()
    try:
        {code}
        return "SUCCESS", time.time() - start
    except Exception as e:
        return f"ERROR: {{e}}", time.time() - start

result, exec_time = safe_execution()
print(f"EXEC_RESULT:|{{result}}|TIME:|{{exec_time}}|")
""")
                temp_file = f.name
            
            # Execute with timeout
            result = subprocess.run(['python', temp_file], 
                                  capture_output=True, text=True, timeout=2)
            
            # Clean up
            import os
            os.unlink(temp_file)
            
            # Parse results
            if "SUCCESS" in result.stdout:
                exec_time = float(result.stdout.split('TIME:|')[1].split('|')[0])
                time_score = min(100, exec_time * 50)  # Scale time to score
                return {'score': time_score, 'confidence': 0.8, 'reason': f'Execution time: {exec_time:.3f}s'}
            else:
                return {'score': 80, 'confidence': 0.9, 'reason': 'Runtime error'}
                
        except subprocess.TimeoutExpired:
            return {'score': 100, 'confidence': 1.0, 'reason': 'Execution timeout'}
        except Exception as e:
            return {'score': 90, 'confidence': 0.7, 'reason': f'Execution failed: {e}'}
    
    def style_expert(self, code: str) -> Dict:
        """Expert: Code style and best practices"""
        violations = 0
        
        # PEP 8 style checks
        lines = code.split('\n')
        for i, line in enumerate(lines):
            # Check line length
            if len(line) > 79:
                violations += 1
            
            # Check for trailing whitespace
            if line.rstrip() != line:
                violations += 1
            
            # Check improper indentation
            if line and line[0] == ' ' and len(line) - len(line.lstrip()) % 4 != 0:
                violations += 2
        
        # Check variable naming (basic)
        bad_names = re.findall(r'[a-z]_[a-z]', code)  # snake_case check
        violations += len(bad_names)
        
        score = min(100, violations * 10)
        confidence = 0.7 if violations > 0 else 0.5
        
        return {'score': score, 'confidence': confidence, 'reason': f'Style violations: {violations}'}
    
    def security_expert(self, code: str) -> Dict:
        """Expert: Security vulnerability analysis"""
        red_flags = 0
        
        # Basic security checks
        dangerous_patterns = [
            r'eval\s*\(', r'exec\s*\(', r'__import__', r'open\s*\(.*[rw]\)',
            r'os\.system', r'subprocess\.call', r'pickle\.loads'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                red_flags += 3  # High risk
        
        # Input validation checks
        if 'input()' in code and not any(keyword in code for keyword in ['int(', 'str(', 'try:', 'except:']):
            red_flags += 2
        
        score = min(100, red_flags * 20)
        confidence = 0.9 if red_flags > 0 else 0.6
        
        return {'score': score, 'confidence': confidence, 'reason': f'Security concerns: {red_flags}'}
    
    def deliberate(self, code: str) -> Dict:
        """Committee deliberation - all experts vote on difficulty"""
        print(f"🧠 COMMITTEE DELIBERATION: Analyzing '{code[:50]}...'")
        
        votes = []
        total_confidence = 0
        
        for expert_name, expert_func in self.experts.items():
            verdict = expert_func(code)
            votes.append({
                'expert': expert_name,
                'score': verdict['score'],
                'confidence': verdict['confidence'],
                'reason': verdict['reason']
            })
            total_confidence += verdict['confidence']
            
            # FIXED: Use proper formatting for floats
            print(f"   {expert_name.upper():<20}: {int(verdict['score']):3d} (conf: {verdict['confidence']:.1f}) - {verdict['reason']}")
        
        # Weighted average based on confidence
        if total_confidence > 0:
            weighted_score = sum(v['score'] * v['confidence'] for v in votes) / total_confidence
        else:
            weighted_score = 50  # Default if no confidence
        
        final_score = min(100, weighted_score)
        
        # Committee reasoning
        primary_concern = max(votes, key=lambda x: x['score'])
        
        print(f"🎯 FINAL VERDICT: {final_score:.1f}/100")
        print(f"   PRIMARY CONCERN: {primary_concern['expert']} - {primary_concern['reason']}")
        
        return {
            'final_score': final_score,
            'breakdown': votes,
            'learnability': max(0, 100 - final_score),  # Inverse for learnability
            'primary_issue': primary_concern['expert']
        }

# DEMO: Show the power
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 DOMAIN-INTELLIGENT REFERENCE COMMITTEE DEMO")
    print("=" * 70)
    
    committee = ReferenceCommittee()
    
    test_cases = [
        "x = 5",  # Trivial
        "for i in range(10): print(i)",  # Simple loop
        "def fib(n): return 1 if n <= 1 else fib(n-1) + fib(n-2)",  # Complex recursion
        "eval(input('Enter code: '))",  # Dangerous!
        "result = [x for x in range(1000) if x % 2 == 0 and x % 3 == 0]",  # Complex comprehension
    ]
    
    for code in test_cases:
        print(f"\n{' TEST CASE ':~^70}")
        print(f"CODE: {code}")
        result = committee.deliberate(code)
        print(f"LEARNABILITY: {result['learnability']:.1f}%")
        
        if result['learnability'] > 70:
            print("🎯 RECOMMENDATION: EXCELLENT learning candidate!")
        elif result['learnability'] > 40:
            print("🎯 RECOMMENDATION: Good learning candidate")
        else:
            print("🎯 RECOMMENDATION: Consider filtering out - too complex/dangerous")