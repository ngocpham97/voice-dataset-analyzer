import numpy as np


def estimate_noise_floor(signal, percentile=10):
    """
    Estimate noise floor from quietest parts of signal.

    Args:
        signal: Audio signal array
        percentile: Percentile to use for noise estimation (default 10%)

    Returns:
        Estimated noise power
    """
    # Use absolute values to get signal magnitude
    signal_abs = np.abs(signal)
    # Get the quietest portions of the signal
    noise_threshold = np.percentile(signal_abs, percentile)
    # Extract samples below this threshold as noise
    noise_samples = signal[signal_abs <= noise_threshold]

    # If we don't have enough noise samples, use a minimum noise floor
    if len(noise_samples) < 100:
        # Assume a digital noise floor of approximately -60 dB
        min_noise_power = 10 ** (-60/10) * np.mean(signal ** 2)
        return min_noise_power

    # Calculate noise power from the quiet samples
    noise_power = np.mean(noise_samples ** 2)

    # Ensure noise power is not zero
    if noise_power == 0:
        min_noise_power = 10 ** (-60/10) * np.mean(signal ** 2)
        return min_noise_power

    return noise_power


def calculate_snr_with_estimation(signal):
    """
    Calculate SNR by estimating noise from the signal itself.

    Args:
        signal: Audio signal array

    Returns:
        SNR in dB
    """
    # Calculate signal power
    signal_power = np.mean(signal ** 2)

    # Estimate noise power from quietest parts
    noise_power = estimate_noise_floor(signal)

    # Handle edge cases
    if noise_power == 0 or signal_power == 0:
        return float('inf') if noise_power == 0 else float('-inf')

    return 10 * np.log10(signal_power / noise_power)


def calculate_snr(signal, noise_signal):
    """Calculate Signal-to-Noise Ratio (SNR) in dB."""
    # If noise_signal is a scalar (like 0), use estimation method
    if np.isscalar(noise_signal) or len(np.atleast_1d(noise_signal)) == 1:
        return calculate_snr_with_estimation(signal)

    # Original method for when we have actual noise signal
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise_signal ** 2)

    # Handle edge cases
    if noise_power == 0 or signal_power == 0:
        return float('inf') if noise_power == 0 else float('-inf')

    return 10 * np.log10(signal_power / noise_power)


def calculate_clipping_percentage(audio_signal):
    """Return fraction of clipped samples (0–1)."""
    clipping_threshold = 1.0
    clipped_samples = np.sum(np.abs(audio_signal) > clipping_threshold)
    return clipped_samples / len(audio_signal) if len(audio_signal) > 0 else 0


def calculate_loudness(audio_signal):
    """Calculate the loudness of the audio signal using EBU R128 standard."""
    # Convert to RMS
    rms = np.sqrt(np.mean(audio_signal ** 2))
    # Convert RMS to LUFS (Loudness Units Full Scale)
    loudness = 20 * np.log10(rms) + 94  # Reference level is -23 LUFS
    return loudness


def analyze_audio_metrics(audio_signal, noise_signal):
    """Analyze and return acoustic metrics for the given audio signal."""
    snr = calculate_snr(audio_signal, noise_signal)
    clipping_pct = calculate_clipping_percentage(audio_signal)
    loudness = calculate_loudness(audio_signal)

    return {
        'SNR': snr,
        'clipping_percentage': clipping_pct,
        'loudness': loudness
    }
