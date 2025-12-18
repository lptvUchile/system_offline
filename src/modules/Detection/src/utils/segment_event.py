import scipy
import numpy as np
import obspy


def segment_event(stream: obspy.Stream, first_umbral, second_umbral):
    f, t, Sxx = scipy.signal.spectrogram(stream.select(component="Z")[0].data, 
                                fs=stream[0].stats.sampling_rate,
                                nperseg=256,
                                    noverlap=128)
    signal_mean_sxx = np.mean(Sxx, axis=0)
    max_sxx = np.max(signal_mean_sxx)
    
    # a schmitt trigger starting from 5% and ending at 1%

    min_umbral_start = max_sxx*first_umbral + np.min(signal_mean_sxx)
    min_umbral_end = max_sxx*second_umbral + np.min(signal_mean_sxx)
    
    trigger_start = signal_mean_sxx > min_umbral_start
    trigger_end = signal_mean_sxx > min_umbral_end
    
    trigger_start = trigger_start.astype(int)
    trigger_end = trigger_end.astype(int)
    
    # where trigger starts its the first 1
    first_one_idx = np.where(trigger_start == 1)[0][0] if len(np.where(trigger_start == 1)[0]) > 0 else None
    
    # then we search for the first 0 after the 1, using the next threshold
    if first_one_idx is not None:
        # Get remaining trigger values after first 1
        remaining_trigger = trigger_end[first_one_idx+1:]
        # Find first 0 after the 1
        first_zero_after_one = np.where(remaining_trigger == 0)[0][0] if len(np.where(remaining_trigger == 0)[0]) > 0 else None
        if first_zero_after_one is not None:
            first_zero_after_one += first_one_idx + 1
    
    if first_zero_after_one is None:
        first_zero_after_one = len(trigger_end)-1
    return t[first_one_idx], t[first_zero_after_one], t, signal_mean_sxx, min_umbral_start, min_umbral_end