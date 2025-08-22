from src.pipeline import evaluate_pipeline

def test_evaluate_pipeline():
    # Sample audio and transcript data for testing
    audio_data = "path/to/sample_audio.wav"
    transcript_data = "path/to/sample_transcript.txt"
    
    # Run the evaluation pipeline
    result = evaluate_pipeline(audio_data, transcript_data)
    
    # Check if the result meets expected criteria
    assert result['suitability_score'] >= 0.5, "Suitability score should be at least 0.5"
    assert 'classification' in result, "Classification should be present in the result"
    assert result['classification']['L0'] is not None, "L0 classification should not be None"
    assert result['classification']['L1'] is not None, "L1 classification should not be None"
    assert result['classification']['L2'] is not None, "L2 classification should not be None"