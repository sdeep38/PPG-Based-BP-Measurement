
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PPG SIGNAL PRE-PROCESSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, welch
import pywt

# Load the data from CSV
file_path = r"C:\Users\Soumyadeep\Desktop\Local Docs\Healthcare\PPG DATA\arduino_data_with_time_7.xlsx"
data = pd.read_excel(file_path)

# Extract Red LED values
red_values = data['Data2:RED'].values

# Normalize the Red LED values
normalized_red = (red_values - np.min(red_values)) / (np.max(red_values) - np.min(red_values))

# Define a bandpass filter
def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, signal)
    return y

# Sampling frequency (samples per second)
fs = 10

# Apply bandpass filter to remove noise
filtered_red = bandpass_filter(normalized_red, 0.5, 3.5, fs)

# Smooth the signal using a moving average filter
def moving_average(signal, window_size):
    cumsum = np.cumsum(np.insert(signal, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

smoothed_red = moving_average(filtered_red, window_size=5)

# Plot the original, filtered, and smoothed signals
time = np.arange(0, len(red_values)) / fs

plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(time, red_values, label='Original Red LED Signal')
plt.title('Original Red LED Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(time, filtered_red, label='Filtered Red LED Signal')
plt.title('Filtered Red LED Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(time[:len(smoothed_red)], smoothed_red, label='Smoothed Red LED Signal')
plt.title('Smoothed Red LED Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()

plt.tight_layout()
plt.show()


## ------------------------------------- SEGMENT-WISE ANALYSIS -----------------------------

# Specify the segment to analyze (in seconds)
start_time = 20
end_time = 50       # 30s segment for better estimation

# Convert the segment time to sample indices
start_sample = int(start_time * fs)
end_sample = int(end_time * fs)

# Extract the segment
segment = normalized_red[start_sample:end_sample]

# Apply bandpass filter to remove noise from the segment
filtered_segment = bandpass_filter(segment, 0.5, 3.5, fs)

smoothed_segment = moving_average(filtered_segment, window_size=5)

# Plot the original, filtered, and smoothed signals
time_segment = np.arange(start_sample, end_sample) / fs

plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(time_segment, segment, label='Original Red LED Segment')
plt.title('Original Red LED Segment')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(time_segment, filtered_segment, label='Filtered Red LED Segment')
plt.title('Filtered Red LED Segment')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(time_segment[:len(smoothed_segment)], smoothed_segment, label='Smoothed Red LED Segment')
plt.title('Smoothed Red LED Segment')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()

plt.tight_layout()
plt.show()


# --------------------------------------- WAVELET DECOMPOSITION ----------------------------------------

# Perform wavelet decomposition using 'db4' wavelet
coeffs = pywt.wavedec(filtered_segment, 'db4', level=6)

# Reconstruct approximate (low-frequency) and detail (high-frequency) coefficients
approx = pywt.waverec(coeffs[:-1] + [None] * 1, 'db4')
details = pywt.waverec([None] + coeffs[1:], 'db4')

# Extract heart rate and breathing signals
heart_rate_signal = coeffs[-2]  # Choose appropriate level based on frequency range
breathing_signal = coeffs[-4]   # Choose appropriate level based on frequency range

# Plot the original, filtered, and decomposed signals
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(time_segment, segment, label='Original Red LED Segment')
plt.title('Original Red LED Segment')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(4, 1, 2)
plt.plot(time_segment, filtered_segment, label='Filtered Red LED Segment')
plt.title('Filtered Red LED Segment')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(time_segment[:len(heart_rate_signal)], heart_rate_signal, label='Heart Rate Signal')
plt.title('Heart Rate Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(4, 1, 4)
plt.plot(time_segment[:len(breathing_signal)], breathing_signal, label='Breathing Signal')
plt.title('Breathing Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()

plt.tight_layout()
plt.show()


# --------------------------------- CALCULATE HEART RATE --------------------------------------

# Detect peaks in the heart rate signal
peaks, _ = find_peaks(heart_rate_signal, distance=fs*0.6)

# Calculate time intervals between peaks
peak_intervals = np.diff(peaks) / fs

# Calculate heart rate frequency in Hz
heart_rate_frequency_hz = 1 / peak_intervals.mean()

# Convert heart rate frequency to BPM (beats per minute)
heart_rate_bpm = heart_rate_frequency_hz * 60

print(f'Estimated Heart Rate Frequency: {heart_rate_frequency_hz:.2f} Hz')
print(f'Estimated Heart Rate: {heart_rate_bpm:.2f} BPM')

# Plot the original, filtered, and heart rate signals
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(time_segment, segment, label='Original Red LED Segment')
plt.title('Original Red LED Segment')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(4, 1, 2)
plt.plot(time_segment, filtered_segment, label='Filtered Red LED Segment')
plt.title('Filtered Red LED Segment')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(time_segment[:len(heart_rate_signal)], heart_rate_signal, label='Heart Rate Signal')
plt.plot(time_segment[peaks], heart_rate_signal[peaks], 'rx')
plt.title('Heart Rate Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(4, 1, 4)
plt.plot(time_segment[peaks[:-1]], peak_intervals, 'bo-')
plt.title('Time Intervals Between Peaks')
plt.xlabel('Peak Index')
plt.ylabel('Time Interval [s]')
plt.grid()

plt.tight_layout()
plt.show()

# ------------------------------ DERIVATIVE ON PPG SIGNAL -----------------------------

# First derivative of the PPG signal
first_derivative = np.diff(filtered_segment)
time_first_derivative = np.arange(start_sample + 1, end_sample) / fs

# Second derivative of the PPG signal
second_derivative = np.diff(first_derivative)
time_second_derivative = np.arange(start_sample + 2, end_sample) / fs

# Plot the original, filtered, and derivative signals
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(time_segment, segment, label='Original Red LED Segment')
plt.title('Original Red LED Segment')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(4, 1, 2)
plt.plot(time_segment, filtered_segment, label='Filtered Red LED Segment')
plt.title('Filtered Red LED Segment')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(time_first_derivative, first_derivative, label='First Derivative of PPG Signal')
plt.title('First Derivative of PPG Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(4, 1, 4)
plt.plot(time_second_derivative, second_derivative, label='Second Derivative of PPG Signal')
plt.title('Second Derivative of PPG Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()

plt.tight_layout()
plt.show()

# ------------------------------ FFT ANALYSIS -----------------------------

# FFT Analysis on the PPG signal
fft_ppg = np.fft.fft(filtered_segment)
fft_freqs_ppg = np.fft.fftfreq(len(filtered_segment), 1/fs)

# FFT Analysis on the heart rate signal
fft_heart_rate = np.fft.fft(heart_rate_signal)
fft_freqs_heart_rate = np.fft.fftfreq(len(heart_rate_signal), 1/fs)


# Plot the original, filtered, and FFT results
plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
plt.plot(fft_freqs_ppg[:len(fft_freqs_ppg)//2], np.abs(fft_ppg)[:len(fft_ppg)//2])
plt.title('FFT of Filtered Red LED Segment')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(fft_freqs_heart_rate[:len(fft_freqs_heart_rate)//2], np.abs(fft_heart_rate)[:len(fft_heart_rate)//2])
plt.title('FFT of Heart Rate Signal')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.grid()

plt.tight_layout()
plt.show()


# ------------------------------ POWER SPECTRAL DENSITY -----------------------------

# Calculate PSD using Welch's method
f_ppg, psd_ppg = welch(filtered_segment, fs, nperseg=256)
f_hr, psd_hr = welch(heart_rate_signal, fs, nperseg=256)

# Plot the PSD results
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.semilogy(f_ppg, psd_ppg)
plt.title('PSD of Filtered Red LED Segment')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power/Frequency [dB/Hz]')
plt.grid()

plt.subplot(2, 1, 2)
plt.semilogy(f_hr, psd_hr)
plt.title('PSD of Heart Rate Signal')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power/Frequency [dB/Hz]')
plt.grid()

plt.tight_layout()
plt.show()
