def normalize_transcript(transcript):
    """
    Normalize the given transcript by applying consistent formatting.
    
    Args:
        transcript (str): The raw transcript to be normalized.
        
    Returns:
        str: The normalized transcript.
    """
    # Convert to lowercase
    normalized = transcript.lower()
    
    # Remove extra whitespace
    normalized = ' '.join(normalized.split())
    
    # Normalize punctuation (example: ensure single spaces after punctuation)
    normalized = normalized.replace('.', '. ').replace('?', '? ').replace('!', '! ')
    
    # Additional normalization rules can be added here
    
    return normalized.strip()

def check_normalization_consistency(transcript, reference_transcript):
    """
    Check the consistency of normalization between the given transcript and a reference transcript.
    
    Args:
        transcript (str): The normalized transcript to check.
        reference_transcript (str): The reference transcript for comparison.
        
    Returns:
        bool: True if the normalization is consistent, False otherwise.
    """
    normalized_transcript = normalize_transcript(transcript)
    normalized_reference = normalize_transcript(reference_transcript)
    
    return normalized_transcript == normalized_reference

def apply_normalization_rules(transcript):
    """
    Apply a set of normalization rules to the transcript.
    
    Args:
        transcript (str): The raw transcript to be normalized.
        
    Returns:
        str: The transcript after applying normalization rules.
    """
    # Example of applying specific normalization rules
    # This can be expanded based on requirements
    normalized = normalize_transcript(transcript)
    
    # Further rules can be applied here
    
    return normalized