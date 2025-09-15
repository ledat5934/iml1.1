# src/iML/prompts/comparison_prompt.py
import json
from typing import Dict, Any, List

from .base_prompt import BasePrompt

class ComparisonPrompt(BasePrompt):
    """
    Prompt handler for intelligent comparison and ranking of multiple iteration results.
    """

    def default_template(self) -> str:
        """Default template for LLM to analyze and rank iterations."""
        return """You are an expert Machine Learning engineer and model evaluator. Your task is to analyze multiple AutoML iteration results and intelligently select the best performing solution.

## Original Task Context:
**Dataset**: {dataset_name}
**Task Type**: {task_type}
**Problem Description**: {task_description}
**Target Variable**: {target_variable}

## Iteration Results to Compare:
{iteration_results_formatted}

## Your Analysis Task:
1. **Extract Performance Metrics**: Find and extract validation scores, CV scores, accuracy, F1, RMSE, etc. from each iteration's execution output
2. **Evaluate Performance**: Compare the extracted metrics for each iteration
3. **Consider Task Type**: Weight metrics appropriately based on the ML task (classification vs regression)
4. **Assess Reliability**: Consider execution stability, error rates, and consistency
5. **Evaluate Complexity**: Balance performance with code complexity and maintainability
6. **Make Final Ranking**: Select the best iteration with detailed reasoning based on extracted scores

## Evaluation Criteria (in order of importance):
1. **Primary Performance**: Task-relevant metrics (accuracy/F1 for classification, RMSE/MAE for regression)
2. **Cross-Validation Stability**: Consistent performance across CV folds
3. **Execution Reliability**: Low error rate, successful completion
4. **Model Complexity**: Simpler models preferred when performance is similar
5. **Interpretability**: More interpretable models preferred for business applications

## Important Considerations:
- For **Classification**: Prioritize accuracy, F1-score, precision/recall balance, ROC-AUC
- For **Regression**: Prioritize RMSE, MAE, R² score
- **Imbalanced datasets**: Favor F1-score and precision/recall over raw accuracy
- **Business context**: Consider if interpretability vs performance trade-off is important
- **Small performance differences**: If metrics are very close (<2% difference), prefer simpler, more reliable solutions

## Output Format:
Provide your analysis in the following JSON format:

```json
{{
    "analysis_summary": {{
        "task_type_detected": "classification/regression",
        "primary_metrics_for_task": ["metric1", "metric2"],
        "total_iterations_analyzed": 3,
        "successful_iterations": 2
    }},
    "iteration_rankings": [
        {{
            "rank": 1,
            "iteration_name": "iteration_3_pretrained",
            "key_metrics": {{"accuracy": 0.95, "f1_score": 0.94}},
            "strengths": ["High accuracy", "Stable execution", "Good F1 score"],
            "weaknesses": ["Higher complexity", "Longer execution time"],
            "reliability_score": 9.5,
            "performance_score": 9.8
        }},
        {{
            "rank": 2,
            "iteration_name": "iteration_1_traditional", 
            "key_metrics": {{"accuracy": 0.92, "f1_score": 0.91}},
            "strengths": ["Fast execution", "Simple and interpretable"],
            "weaknesses": ["Lower accuracy than pretrained"],
            "reliability_score": 9.8,
            "performance_score": 9.2
        }},
        {{
            "rank": 3,
            "iteration_name": "iteration_2_custom_nn",
            "key_metrics": {{"accuracy": 0.89, "f1_score": 0.87}},
            "strengths": ["Custom architecture"],
            "weaknesses": ["Lower performance", "Higher complexity", "Execution errors"],
            "reliability_score": 7.5,
            "performance_score": 8.9
        }}
    ],
    "best_iteration": {{
        "name": "iteration_3_pretrained",
        "final_score": 9.65,
        "selection_reasoning": "Despite higher complexity, the pretrained model achieves significantly better performance with stable execution. The 3% accuracy improvement and 3% F1-score improvement over traditional ML justify the added complexity.",
        "confidence_level": "high",
        "alternative_recommendation": "iteration_1_traditional for production environments requiring high interpretability"
    }},
    "reasoning_summary": "Selected iteration_3_pretrained due to superior performance metrics with acceptable reliability. The pretrained model demonstrates clear performance advantages that outweigh the complexity cost.",
    "recommendations": [
        "Deploy iteration_3_pretrained for maximum performance",
        "Consider iteration_1_traditional for interpretability-critical applications",
        "Monitor iteration_3_pretrained for production stability"
    ]
}}
```

## Critical Instructions:
- Be objective and data-driven in your analysis
- Provide specific numeric comparisons when available
- If two iterations have very similar performance, prefer the simpler one
- Always provide reasoning for your ranking decisions
- Consider the business context of the ML task
- Ensure your JSON output is perfectly valid (no comments, proper escaping)
"""

    def build(self, iteration_results: List[Dict[str, Any]], 
              task_description: Dict[str, Any]) -> str:
        """Build comparison prompt with iteration results and task context."""
        
        # Extract task information
        dataset_name = task_description.get('name', 'Unknown Dataset')
        task_type = task_description.get('task', 'Unknown Task')
        target_variable = task_description.get('target_variable', 'Unknown')
        task_desc = task_description.get('description', 'No description available')
        
        # Format iteration results for LLM
        formatted_results = self._format_iteration_results(iteration_results)
        
        prompt = self.template.format(
            dataset_name=dataset_name,
            task_type=task_type,
            task_description=task_desc,
            target_variable=target_variable,
            iteration_results_formatted=formatted_results
        )
        
        self.manager.save_and_log_states(prompt, "comparison_prompt.txt")
        return prompt
    
    def _format_iteration_results(self, iteration_results: List[Dict[str, Any]]) -> str:
        """Format iteration results for LLM consumption."""
        formatted_sections = []
        
        for i, result in enumerate(iteration_results, 1):
            name = result.get('iteration_name', f'iteration_{i}')
            status = result.get('status', 'unknown')
            scores = result.get('scores', {})
            exec_stats = result.get('execution_stats', {})
            complexity = result.get('code_complexity', {})
            error = result.get('error')
            
            section = f"""
### {name.upper()}
**Status**: {status}
**Raw Output Analysis**:
Please extract validation scores, accuracy, F1, RMSE, etc. from the execution output below.

**Execution Statistics**:
  - Number of attempts: {exec_stats.get('num_attempts', 'unknown')}
  - Final attempt: {exec_stats.get('final_attempt', 'unknown')}
  - Error count: {exec_stats.get('error_count', 0)}
  - Status: {status}

**Code Complexity**:
  - Lines of code: {complexity.get('lines_of_code', 'unknown')}
  - Model type: {complexity.get('model_type', 'unknown')}
  - Functions count: {complexity.get('functions_count', 'unknown')}
  - Imports count: {complexity.get('imports_count', 'unknown')}
"""
            
            if error:
                section += f"\n**Error Details**: {error}\n"
            
            # Add full stdout for LLM analysis
            if result.get('full_stdout'):
                section += f"\n**Complete Execution Output**:\n```\n{result['full_stdout']}\n```\n"
            elif result.get('stdout_excerpt'):
                section += f"\n**Output Excerpt**:\n```\n{result['stdout_excerpt']}\n```\n"
            
            # Add stderr excerpt if available
            if result.get('stderr_excerpt'):
                section += f"\n**Error Output**:\n```\n{result['stderr_excerpt']}\n```\n"
            
            formatted_sections.append(section)
        
        return "\n".join(formatted_sections)

    def parse(self, response: str) -> Dict[str, Any]:
        """Parse LLM comparison response."""
        try:
            # Clean the response
            cleaned_response = response.strip()
            
            # Remove markdown code blocks if present
            if "```json" in cleaned_response:
                start = cleaned_response.find("```json") + 7
                end = cleaned_response.rfind("```")
                if end > start:
                    cleaned_response = cleaned_response[start:end].strip()
            elif "```" in cleaned_response:
                start = cleaned_response.find("```") + 3
                end = cleaned_response.rfind("```")
                if end > start:
                    cleaned_response = cleaned_response[start:end].strip()
            
            parsed_result = json.loads(cleaned_response)
            
            # Validate required fields
            required_fields = ['best_iteration', 'iteration_rankings', 'reasoning_summary']
            for field in required_fields:
                if field not in parsed_result:
                    return {
                        "error": f"Missing required field: {field}",
                        "raw_response": response
                    }
            
            # Ensure best_iteration has required fields
            best_iter = parsed_result.get('best_iteration', {})
            if not best_iter.get('name'):
                return {
                    "error": "Best iteration name not specified",
                    "raw_response": response
                }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM comparison response: {e}")
            return {
                "error": f"Invalid JSON response: {str(e)}",
                "raw_response": response
            }
        except Exception as e:
            logger.error(f"Unexpected error parsing comparison response: {e}")
            return {
                "error": f"Parsing error: {str(e)}",
                "raw_response": response
            }
        
        # Save parsed response
        self.manager.save_and_log_states(
            json.dumps(parsed_result, indent=2, ensure_ascii=False),
            "comparison_response.json"
        )
        
        return parsed_result
