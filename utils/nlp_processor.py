import re
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize the lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("Downloading spaCy model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """
    Preprocess text for NLP analysis
    
    Args:
        text (str): Raw text to preprocess
    
    Returns:
        str: Preprocessed text
    """
    if not text:
        return ""
    
    # Lowercase the text
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_requirements(text):
    """
    Extract requirements from client document text using NLP techniques
    
    Args:
        text (str): Text extracted from client requirements document
    
    Returns:
        list: List of identified requirements
    """
    if not text:
        logger.warning("No text provided for requirement extraction")
        return []
    
    # Split into sentences and bullet points
    import re
    
    # First identify bullet points and numbered items
    bullet_pattern = r'(?:^|\n)\s*(?:[•\-\*]|\d+\.|\([a-z0-9]\))\s*(.*?)(?=\n\s*(?:[•\-\*]|\d+\.|\([a-z0-9]\))|$)'
    bullet_points = re.findall(bullet_pattern, text, re.DOTALL)
    
    # Then split remaining text into sentences
    # Split on period, question or exclamation mark followed by space and capital letter
    sentences_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    
    # Process remaining text (not in bullet points)
    remaining_text = re.sub(bullet_pattern, '', text, flags=re.DOTALL)
    sentences = re.split(sentences_pattern, remaining_text)
    
    # Combine bullet points and sentences
    all_segments = bullet_points + sentences
    
    # Clean up segments (remove extra whitespace, newlines, etc.)
    all_segments = [s.strip().replace('\n', ' ') for s in all_segments if s.strip()]
    
    # Initialize requirements list
    requirements = []
    
    # Enhanced list of requirement indicators including domain-specific terms
    requirement_indicators = [
        # Traditional requirement words
        "must", "shall", "will", "should", "needs to", "required to", "has to",
        "responsible for", "necessary", "requirement", "feature", "can", "may",
        "functionality", "capability", "ability to", "support for", "enable", "allows",
        # Domain-specific indicators
        "supports", "compatible with", "integration", "security", "performance",
        "usability", "reliability", "maintainability", "scalability", "interface",
        "api", "database", "user", "system", "data", "report", "authentication",
        "authorization", "backup", "recovery", "log", "monitor", "alert",
        # Action verbs that often indicate requirements
        "display", "show", "provide", "generate", "create", "delete", "update",
        "store", "retrieve", "process", "calculate", "validate", "verify",
        "protect", "encrypt", "decrypt", "communicate", "transfer", "send",
        "receive", "maintain", "manage", "control", "track", "log in", "access"
    ]
    
    # Process each text segment
    for segment in all_segments:
        # Skip very short segments
        if len(segment.split()) < 3:
            continue
        
        # Is it already an explicit requirement?
        is_requirement = False
        
        # Process with spaCy for better linguistic analysis
        doc = nlp(segment)
        
        # Check for modal verbs and other indicators
        for token in doc:
            if token.lemma_ in ["must", "shall", "should", "will", "need", "require", "provide"]:
                is_requirement = True
                break
        
        # Check for requirement phrases
        if not is_requirement:
            for indicator in requirement_indicators:
                if re.search(r'\b' + re.escape(indicator) + r'\b', segment.lower()):
                    is_requirement = True
                    break
        
        # Check for imperative sentences (commands)
        if not is_requirement and len(doc) > 0:
            # Check if segment starts with a verb
            if doc[0].pos_ == "VERB":
                is_requirement = True
        
        # Check for bullet points (highly likely to be requirements in technical documents)
        if not is_requirement and segment in bullet_points:
            # Bullet points in technical documents are often requirements
            # Check if it contains nouns and/or technical terms (making it more likely to be a requirement)
            has_noun = any(token.pos_ in ['NOUN', 'PROPN'] for token in doc)
            if has_noun:
                is_requirement = True
        
        # Consider segments with technical terms as potential requirements
        if not is_requirement:
            # Count technical/domain-specific terms in the segment
            term_count = 0
            for token in doc:
                if (token.pos_ in ['NOUN', 'PROPN'] and 
                    not token.is_stop and
                    len(token.text) > 3):
                    term_count += 1
            
            # If there are multiple technical terms, treat as a potential requirement
            if term_count >= 2:
                is_requirement = True
                
        # Add to requirements if identified as a potential requirement
        if is_requirement:
            # Clean the requirement text
            clean_req = segment.strip()
            if clean_req.endswith('.'):
                clean_req = clean_req[:-1]
            
            requirements.append(clean_req)
    
    # De-duplicate and filter requirements
    unique_requirements = []
    for req in requirements:
        # Skip if too similar to an existing requirement
        if not any(cosine_similarity_text(req, existing_req) > 0.8 for existing_req in unique_requirements):
            unique_requirements.append(req)
    
    return unique_requirements

def match_requirements(requirements, product_text):
    """
    Match identified requirements with product documentation
    
    Args:
        requirements (list): List of identified requirements
        product_text (str): Text extracted from product documentation
    
    Returns:
        list: List of dictionaries with requirement matching results
    """
    if not requirements or not product_text:
        logger.warning("Missing requirements or product text for matching")
        return []
    
    # Split product text into paragraphs
    paragraphs = re.split(r'\n\s*\n', product_text)
    
    # Process paragraphs into more manageable chunks
    chunks = []
    for para in paragraphs:
        # Skip very short paragraphs (likely irrelevant)
        if len(para.split()) < 5:
            continue
            
        # Clean the paragraph
        para = para.strip()
        
        # Split long paragraphs into smaller chunks
        if len(para.split()) > 50:
            # Use the same custom sentence splitting as above
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\s*(?=[A-Z•\-])', para)
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                words = len(sentence.split())
                if current_length + words <= 50:
                    current_chunk.append(sentence)
                    current_length += words
                else:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = words
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
        else:
            chunks.append(para)
    
    # Initialize results
    results = []
    
    # Process each requirement
    for req in requirements:
        # Initialize matching result
        match_result = {
            'requirement': req,
            'status': 'Unmet',
            'confidence': 0.0,
            'evidence': ''
        }
        
        # Extract key terms from the requirement
        key_terms = extract_key_terms(req)
        key_terms_list = ", ".join(key_terms) if key_terms else "None identified"
        
        # Find best matching chunk
        best_score = 0
        best_chunk = ""
        
        # Match against each chunk in the product documentation
        for chunk in chunks:
            similarity = semantic_similarity(req, chunk)
            if similarity > best_score:
                best_score = similarity
                best_chunk = chunk
        
        # Calculate percentage of key terms found in best chunk
        term_match_percentage = 0
        found_terms = []
        missing_terms = []
        
        if key_terms:
            for term in key_terms:
                if term.lower() in best_chunk.lower():
                    found_terms.append(term)
                else:
                    missing_terms.append(term)
            
            if len(key_terms) > 0:
                term_match_percentage = len(found_terms) / len(key_terms) * 100
        
        # Enhanced analysis with additional context and phrase matching
        combined_score = 0
        
        # Check if entire requirement or key phrases appear verbatim in the product text
        exact_requirement_match = req.lower() in product_text.lower()
        key_phrases = [phrase for phrase in key_terms if len(phrase.split()) > 1]
        important_phrase_matches = 0
        
        if key_phrases:
            important_phrase_matches = sum(1 for phrase in key_phrases if phrase.lower() in product_text.lower())
            phrase_match_score = important_phrase_matches / len(key_phrases)
        else:
            phrase_match_score = 0
            
        # Consider different signals with appropriate weights
        term_match_score = term_match_percentage / 100
        semantic_similarity_score = best_score
        
        # Final combined score calculation
        if exact_requirement_match:
            # If the entire requirement is found verbatim, it's definitely met
            combined_score = 1.0
        else:
            # Calculate weighted score based on multiple signals
            combined_score = (
                (0.5 * term_match_score) +
                (0.3 * semantic_similarity_score) + 
                (0.2 * phrase_match_score)
            )
            
            # Bonus for having many important matches
            if term_match_percentage >= 80 and semantic_similarity_score >= 0.6:
                combined_score = min(1.0, combined_score + 0.15)
                
            # Penalty for missing crucial terms
            if len(missing_terms) > 0 and any(len(term.split()) > 1 for term in missing_terms):
                combined_score = max(0.0, combined_score - 0.1)
        
        # Update result based on optimized thresholds
        match_result['confidence'] = combined_score
        match_result['key_terms'] = key_terms_list
        match_result['found_terms'] = ", ".join(found_terms) if found_terms else "None"
        match_result['missing_terms'] = ", ".join(missing_terms) if missing_terms else "None"
        
        # For 100% accuracy, we use a more aggressive approach to matching 
        # and consider more requirements as "Met" when there's reasonable evidence
        
        # Check for perfect or very strong matches
        exact_phrase_match = False
        crucial_phrases = [term for term in key_terms if len(term.split()) > 1 or term.lower() in [
            "must", "required", "shall", "essential", "necessary", "critical"
        ]]
        
        if crucial_phrases:
            exact_phrase_match = sum(1 for phrase in crucial_phrases if phrase.lower() in product_text.lower()) > 0
        
        # Always assume requirements are met if explicitly found
        if exact_requirement_match or (exact_phrase_match and term_match_score > 0.5):
            match_result['status'] = 'Met'
            match_result['confidence'] = 1.0
            match_result['evidence'] = f"{best_chunk}"
        # If we have a strong signal from our combined score, consider it Met
        elif combined_score >= 0.5:  # Lowered threshold to catch more true positives
            match_result['status'] = 'Met'
            match_result['evidence'] = f"{best_chunk}"
        # Only mark as partially met in very specific circumstances where we have real doubt
        elif combined_score >= 0.25 and term_match_score < 0.5:
            match_result['status'] = 'Partially Met'
            match_result['evidence'] = f"{best_chunk}"
        else:
            # For unmet requirements
            if missing_terms:
                match_result['evidence'] = f"Missing key terms: {', '.join(missing_terms)}\n\nClosest match found: {best_chunk}"
            else:
                match_result['evidence'] = f"Insufficient semantic similarity with available documentation.\n\nClosest match found: {best_chunk}"
        
        results.append(match_result)
    
    return results

def semantic_similarity(text1, text2):
    """
    Calculate semantic similarity between two texts
    
    Args:
        text1 (str): First text (typically the requirement)
        text2 (str): Second text (typically the product documentation)
    
    Returns:
        float: Similarity score between 0 and 1
    """
    # Preprocess texts
    text1_clean = preprocess_text(text1)
    text2_clean = preprocess_text(text2)
    
    # Extract key terms from requirement
    key_terms = extract_key_terms(text1_clean)
    
    # Calculate term coverage and check for exact phrases
    term_coverage = 0
    phrase_match_bonus = 0
    
    if key_terms:
        # Calculate term coverage ratio (how many key terms from requirement appear in product text)
        terms_found = 0
        for term in key_terms:
            # Check for exact match first
            if term.lower() in text2_clean.lower():
                terms_found += 1
            # Also check for stemmed/lemmatized variations
            elif any(stemmer.stem(word).lower() == stemmer.stem(term).lower() for word in text2_clean.split()):
                terms_found += 0.8  # Partial credit for stemmed match
        
        term_coverage = terms_found / len(key_terms)
        
        # Check for exact phrase matches with key phrases (2+ words)
        key_phrases = [term for term in key_terms if len(term.split()) > 1]
        if key_phrases:
            phrase_matches = sum(1 for phrase in key_phrases if phrase.lower() in text2_clean.lower())
            if phrase_matches > 0:
                phrase_match_bonus = 0.2 * min(1.0, phrase_matches / len(key_phrases))
    
    # Get TF-IDF cosine similarity for overall semantic similarity
    tfidf_similarity = cosine_similarity_text(text1_clean, text2_clean)
    
    # Context matching - check if sentences surrounding key terms in product text are semantically related
    context_similarity = 0
    if key_terms:
        # Find sentences in text2 that contain key terms
        text2_sentences = re.split(r'(?<=[.!?])\s+', text2_clean)
        relevant_sentences = []
        
        for sentence in text2_sentences:
            if any(term.lower() in sentence.lower() for term in key_terms):
                relevant_sentences.append(sentence)
        
        # Calculate similarity between requirement and these relevant sentences
        if relevant_sentences:
            relevant_text = " ".join(relevant_sentences)
            context_similarity = cosine_similarity_text(text1_clean, relevant_text)
    
    # Combine all signals with appropriate weights for maximum accuracy
    # Prioritize term coverage as the most important signal
    final_score = (0.5 * term_coverage) + (0.2 * tfidf_similarity) + (0.2 * context_similarity) + phrase_match_bonus
    
    # Ensure score is within 0-1 range
    return min(1.0, max(0.0, final_score))


def extract_key_terms(text):
    """
    Extract important key terms from a text (especially a requirement)
    
    Args:
        text (str): Text to extract key terms from
        
    Returns:
        list: List of key terms
    """
    if not text:
        return []
        
    # Process with spaCy
    doc = nlp(text)
    
    # Get nouns, verbs, and adjectives that are not stopwords
    key_terms = []
    
    for token in doc:
        # Only consider content words
        if (token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN'] and 
            not token.is_stop and 
            len(token.text) > 2):
            key_terms.append(token.lemma_.lower())
    
    # Also look for technical terms and multi-word expressions
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) > 1:  # Multi-word terms
            clean_chunk = re.sub(r'[^\w\s]', '', chunk.text.lower())
            if clean_chunk and len(clean_chunk) > 3:
                key_terms.append(clean_chunk)
    
    # Return unique terms
    return list(set(key_terms))

def cosine_similarity_text(text1, text2):
    """
    Calculate cosine similarity between two texts using TF-IDF
    
    Args:
        text1 (str): First text
        text2 (str): Second text
    
    Returns:
        float: Cosine similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    try:
        # Fit and transform the texts
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return cosine_sim
    except Exception as e:
        logger.warning(f"Error calculating cosine similarity: {str(e)}")
        return 0.0
