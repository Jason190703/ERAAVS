"""
AI Integration Module for Enhanced Requirement Analysis

This module integrates with OpenAI and Anthropic APIs to provide advanced
natural language understanding capabilities for the requirement analysis system.
"""

import os
import logging
import json
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if OpenAI API key is available
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

def has_valid_api_key() -> bool:
    """Check if a valid API key is available."""
    return bool(OPENAI_API_KEY or ANTHROPIC_API_KEY)

def analyze_requirement_with_ai(requirement: str, product_text: str) -> Optional[Dict]:
    """
    Use AI to evaluate if a requirement is met in the product documentation.
    
    Args:
        requirement (str): The requirement text to analyze
        product_text (str): The product documentation text
        
    Returns:
        dict: Analysis results including status, confidence, and evidence
    """
    # Return early if no API keys are available
    if not has_valid_api_key():
        logger.warning("No AI API keys available for enhanced analysis")
        return None
    
    try:
        # Try OpenAI first
        if OPENAI_API_KEY:
            return analyze_with_openai(requirement, product_text)
        # Fall back to Anthropic if OpenAI not available
        elif ANTHROPIC_API_KEY:
            return analyze_with_anthropic(requirement, product_text)
    except Exception as e:
        logger.error(f"AI analysis error: {str(e)}")
        return None
    
    return None

def analyze_with_openai(requirement: str, product_text: str) -> Dict:
    """Use OpenAI API to analyze a requirement."""
    from openai import OpenAI
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Truncate product text if too long (OpenAI has token limits)
    max_length = 14000  # Keeping well below token limits
    if len(product_text) > max_length:
        product_text = product_text[:max_length] + "... [truncated]"
    
    prompt = f"""
    Your task is to determine if the requirement below is met in the provided product documentation.
    
    REQUIREMENT:
    {requirement}
    
    PRODUCT DOCUMENTATION:
    {product_text}
    
    Analyze whether the requirement is fully met, partially met, or not met in the documentation.
    Extract key terms from the requirement and check if they are present in the documentation.
    Provide evidence from the documentation that supports your conclusion.
    
    Respond with a JSON object in the following format:
    {{
        "status": "Met", "Partially Met", or "Unmet",
        "confidence": A float between 0 and 1 indicating your confidence,
        "key_terms": ["list", "of", "key", "terms", "from", "requirement"],
        "found_terms": ["list", "of", "terms", "found", "in", "documentation"],
        "missing_terms": ["list", "of", "terms", "not", "found"],
        "evidence": "The most relevant excerpt from the documentation",
        "explanation": "Brief explanation of your reasoning"
    }}
    """
    
    try:
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",  # Using the most capable model
            messages=[{"role": "system", "content": "You are an expert in requirements analysis and verification."},
                     {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        analysis = json.loads(response.choices[0].message.content)
        
        # Add source field to indicate this came from AI
        analysis['source'] = 'ai_openai'
        
        return analysis
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise

def analyze_with_anthropic(requirement: str, product_text: str) -> Dict:
    """Use Anthropic API to analyze a requirement."""
    from anthropic import Anthropic
    
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    
    # Truncate product text if too long (Anthropic has token limits)
    max_length = 20000  # Keeping well below token limits
    if len(product_text) > max_length:
        product_text = product_text[:max_length] + "... [truncated]"
    
    prompt = f"""
    Your task is to determine if the requirement below is met in the provided product documentation.
    
    REQUIREMENT:
    {requirement}
    
    PRODUCT DOCUMENTATION:
    {product_text}
    
    Analyze whether the requirement is fully met, partially met, or not met in the documentation.
    Extract key terms from the requirement and check if they are present in the documentation.
    Provide evidence from the documentation that supports your conclusion.
    
    Respond with a JSON object in the following format:
    {{
        "status": "Met", "Partially Met", or "Unmet",
        "confidence": A float between 0 and 1 indicating your confidence,
        "key_terms": ["list", "of", "key", "terms", "from", "requirement"],
        "found_terms": ["list", "of", "terms", "found", "in", "documentation"],
        "missing_terms": ["list", "of", "terms", "not", "found"],
        "evidence": "The most relevant excerpt from the documentation",
        "explanation": "Brief explanation of your reasoning"
    }}
    
    Your response should only be the JSON object, nothing else.
    """
    
    try:
        # Call the Anthropic API
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",  # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
            max_tokens=1024,
            system="You are an expert in requirements analysis and verification. Provide accurate, evidence-based evaluations of requirement fulfillment.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract JSON from response
        content = response.content[0].text
        # Find the JSON part in the response
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        json_str = content[json_start:json_end]
        
        # Parse the response
        analysis = json.loads(json_str)
        
        # Add source field to indicate this came from AI
        analysis['source'] = 'ai_anthropic'
        
        return analysis
    except Exception as e:
        logger.error(f"Anthropic API error: {str(e)}")
        raise

def enhance_existing_analysis(analysis_results: List[Dict], product_text: str) -> List[Dict]:
    """
    Enhance existing analysis results with AI insights when possible.
    
    Args:
        analysis_results: List of requirement analysis results
        product_text: Full product documentation text
        
    Returns:
        List: Enhanced analysis results
    """
    # Skip if no API keys available
    if not has_valid_api_key():
        return analysis_results
    
    # Randomly select a few requirements to enhance with AI (for demo purposes)
    # In a production system, you might enhance all or use criteria to select
    import random
    sample_size = min(3, len(analysis_results))
    sample_indices = random.sample(range(len(analysis_results)), sample_size)
    
    # For each sampled requirement, try to enhance with AI
    enhanced_results = analysis_results.copy()
    for idx in sample_indices:
        requirement = analysis_results[idx]['requirement']
        try:
            ai_analysis = analyze_requirement_with_ai(requirement, product_text)
            if ai_analysis:
                # Update with AI analysis but preserve original requirement
                enhanced_results[idx].update(ai_analysis)
                # Add a marker that this was AI-enhanced
                enhanced_results[idx]['ai_enhanced'] = True
        except Exception as e:
            logger.error(f"Error enhancing analysis for requirement {idx}: {str(e)}")
    
    return enhanced_results