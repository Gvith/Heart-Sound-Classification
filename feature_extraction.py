import csv
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fftpack import dct
import sys
import pandas as pd
import argparse

# parse the command line arguments and store the value in its respective variable
parser = argparse.ArgumentParser()

# --train_state: path to the file containing heart sound states
# --train_amps: path to the file containing heart sound amplitudes
# --feat_names: path to the list of feature names
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument("--train_state", type=str, help="path to the heart sound states file", required=True)
requiredNamed.add_argument("--train_amps", type=str, help="path to the heart sound amplitudes file", required=True)
requiredNamed.add_argument("--feat_names", type=str, help="path to the feature names file", required=True)
requiredNamed.add_argument("--out", type=str, help="path to the output file", required=True)

args = parser.parse_args()

NFFT = 256
frequency_ranges = [(25, 45), (45, 65), (65, 85), (85, 105), (105, 125), (125, 150), (150, 200), (200, 300), (300, 500)]
freq = np.array([i/float(NFFT/2)*500 for i in range(0,(NFFT/2))])
skew_list = []
kurtosis_list = []
interval_len_list = []
mel_list = []
power_list = []
power_freq = []

# computes the mel coefficients for the sound signal.
# The theoretical underpinnings are as explained in http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/#computing-the-mel-filterbank
def mel_coefficients(sample_rate, nfilt, pow_frames):
    low_freq_mel = 0
    num_mel_coeff = 12
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2.0) / 700.0))  # Convert Hz to Mel

    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz

    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    mfcc = dct(filter_banks, type=2, axis=0, norm='ortho')[1:(num_mel_coeff+1)]
    (ncoeff,) = mfcc.shape
    cep_lifter = ncoeff
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    return mfcc

# This function puts the different frequency components into one of 9 bins
def check_freq_num():
    q = 0
    frequency_index_list = []
    for each_freq_band in frequency_ranges:
        index = np.where(np.logical_and(freq>=each_freq_band[0], freq<each_freq_band[1]))
        frequency_index_list.append(index)
    return frequency_index_list

# This function computes the time-domain features of the signal
# as explained in the paper http://ieeexplore.ieee.org/abstract/document/7868819/
def time_features(data_sequence, row_index, hs_state):

    # computes the length of the sequence
    interval_length = len(data_sequence)

    # computes the mean value of the sequence
    mean_value_per_beat = np.mean(np.around(np.absolute(data_sequence), decimals=4))

    # computes the skewness of the sequence
    seq_skew = skew(data_sequence)

    # computes the kurtosis of the sequence
    seq_kurtosis = kurtosis(data_sequence, fisher=False)

    if row_index < 4:
        interval_len_list.append([interval_length])
        skew_list.append([seq_skew])
        kurtosis_list.append([seq_kurtosis])
    else:
        interval_len_list[hs_state].append(interval_length)
        skew_list[hs_state].append(seq_skew)
        kurtosis_list[hs_state].append(seq_kurtosis)

    return interval_length, mean_value_per_beat

# This function computes the frequency domain features of the signal
# as explained in the paper http://ieeexplore.ieee.org/abstract/document/7868819/
def frequency_features(data_sequence, row_index, hs_state):

    global f_indices
    power_freq = []

    # computes the power spectrum of the signal
    hamming_distance = data_sequence*np.hamming(len(data_sequence))
    fft = np.absolute(np.fft.rfft(hamming_distance, NFFT))
    power_spec = np.around(fft[:NFFT/2], decimals=4)
    p_spec = ((1.0 / NFFT) * ((fft) ** 2))

    # computes the mel frequency cepstral coefficient of the sound signal
    mel_coeff = mel_coefficients(1000, 40, p_spec)

    for r, each_f_index in enumerate(f_indices):
        median_power = power_spec[each_f_index]
        if row_index < 4:
            power_freq.append([median_power])
        else:
            power_list[hs_state][r].append(median_power)

    if row_index < 4:
        mel_list.append([mel_coeff])
        power_list.append(power_freq)
    else:
        mel_list[hs_state].append(mel_coeff)

# This function extracts the different features from the heart sound signal.
# Some of these features include statistical measures like mean, std. deviation, etc.
def feature_extraction(hs_state, hs_amps):
    amp_dictionary = {}
    feature_list = []
    f_indices = check_freq_num()

    # map the heart sound state with its corresponding amplitude
    for j, each_amp in enumerate(hs_amps):
        amp_dictionary[j] = each_amp

    # iterate over each heart sound recording
    for i, each_row in enumerate(hs_state):
        # print i
        aggregate_features = []
        rr_list = []
        ratio_systolic_rr = []
        ratio_diastolic_rr = []
        ratio_systole_diastole = []
        ratio_systole_s1 = []
        ratio_diastole_s2 = []
        power_spectrum_s1 = []
        power_spectrum_systole = []
        power_spectrum_s2 = []
        power_spectrum_diastole = []
        rr_flag = False

        each_row = map(int, each_row)
        segment_indices = list(np.where(np.diff(each_row) != 0)[0]+1)

        amp = amp_dictionary[i]
        amp = map(float, amp)

        # iterate over different groups of heart sound state
        for m, each_index in enumerate(segment_indices):
            if m == len(segment_indices)-1:

                # at the end of each heart sound recording compute the mean and std. deviation for the below
                # metrics over the entire recording
                mean_RR = np.around(np.mean(rr_list), decimals=4)
                std_RR = np.around(np.std(rr_list), decimals=4)
                mean_s1 = np.around(np.mean(interval_len_list[0]), decimals=4)
                std_s1 = np.around(np.std(interval_len_list[0]), decimals=4)
                mean_systole = np.around(np.mean(interval_len_list[1]), decimals=4)
                std_systole = np.around(np.std(interval_len_list[1]), decimals=4)
                mean_s2 = np.around(np.mean(interval_len_list[2]), decimals=4)
                std_s2 = np.around(np.std(interval_len_list[2]), decimals=4)
                mean_diastole = np.around(np.mean(interval_len_list[3]), decimals=4)
                std_diastole = np.around(np.std(interval_len_list[3]), decimals=4)
                mean_ratio_systolic_rr = np.around(np.mean(ratio_systolic_rr), decimals=4)
                std_ratio_systolic_rr = np.around(np.std(ratio_systolic_rr), decimals=4)
                mean_ratio_diastolic_rr = np.around(np.mean(ratio_diastolic_rr), decimals=4)
                std_ratio_diastolic_rr = np.around(np.std(ratio_diastolic_rr), decimals=4)
                mean_ratio_systole_diastole = np.around(np.mean(ratio_systole_diastole), decimals=4)
                std_ratio_systole_diastole = np.around(np.std(ratio_systole_diastole), decimals=4)

                mean_ratio_systole_s1 = np.around(np.mean(ratio_systole_s1), decimals=4)
                std_ratio_systole_s1 = np.around(np.std(ratio_systole_s1), decimals=4)
                mean_ratio_diastole_s2 = np.around(np.mean(ratio_diastole_s2), decimals=4)
                std_ratio_diastole_s2 = np.around(np.std(ratio_diastole_s2), decimals=4)
                mean_s1_skew = np.around(np.mean(skew_list[0]), decimals=4)
                std_s1_skew = np.around(np.std(skew_list[0]), decimals=4)
                mean_systole_skew = np.around(np.mean(skew_list[1]), decimals=4)
                std_systole_skew = np.around(np.std(skew_list[1]), decimals=4)
                mean_s2_skew = np.around(np.mean(skew_list[2]), decimals=4)
                std_s2_skew = np.around(np.std(skew_list[2]), decimals=4)
                mean_diastole_skew = np.around(np.mean(skew_list[3]), decimals=4)
                std_diastole_skew = np.around(np.std(skew_list[3]), decimals=4)
                mean_s1_kurtosis = np.around(np.mean(kurtosis_list[0]), decimals=4)
                std_s1_kurtosis = np.around(np.std(kurtosis_list[0]), decimals=4)
                mean_systole_kurtosis = np.around(np.mean(kurtosis_list[1]), decimals=4)
                std_systole_kurtosis = np.around(np.std(kurtosis_list[1]), decimals=4)
                mean_s2_kurtosis = np.around(np.mean(kurtosis_list[2]), decimals=4)
                std_s2_kurtosis = np.around(np.std(kurtosis_list[2]), decimals=4)
                mean_diastole_kurtosis = np.around(np.mean(kurtosis_list[3]), decimals=4)
                std_diastole_kurtosis = np.around(np.std(kurtosis_list[3]), decimals=4)

                aggregate_features = [mean_RR, std_RR, mean_s1, std_s1, mean_systole, std_systole, mean_s2, std_s2, \
                                      mean_diastole, std_diastole, mean_ratio_systolic_rr, std_ratio_systolic_rr, \
                                      mean_ratio_diastolic_rr, std_ratio_diastolic_rr, mean_ratio_systole_diastole, \
                                      std_ratio_systole_diastole, mean_ratio_systole_s1, std_ratio_systole_s1, \
                                      mean_ratio_diastole_s2, std_ratio_diastole_s2, mean_s1_skew, std_s1_skew, \
                                      mean_systole_skew, std_systole_skew, mean_s2_skew, std_s2_skew, mean_diastole_skew, \
                                      std_diastole_skew, mean_s1_kurtosis, std_s1_kurtosis, mean_systole_kurtosis, \
                                      std_systole_kurtosis, mean_s2_kurtosis, std_s2_kurtosis, mean_diastole_kurtosis, \
                                      std_diastole_kurtosis]

                # compute the power spectrum for the four different heart states divided into 9 frequency bins
                for t in range(len(f_indices)):
                    power_spectrum_s1.append(np.around(np.median(np.median(power_list[0][t], axis=0)), decimals=4))
                    power_spectrum_systole.append(np.around(np.median(np.median(power_list[1][t], axis=0)), decimals=4))
                    power_spectrum_s2.append(np.around(np.median(np.median(power_list[2][t], axis=0)), decimals=4))
                    power_spectrum_diastole.append(np.around(np.median(np.median(power_list[3][t], axis=0)), decimals=4))

                aggregate_features.extend(power_spectrum_s1)
                aggregate_features.extend(power_spectrum_systole)
                aggregate_features.extend(power_spectrum_s2)
                aggregate_features.extend(power_spectrum_diastole)

                # compute the first 12 mel frequency cepstral coefficients for each of the 4 different heart sound state
                mfcc_s1 = list(np.around(np.median(mel_list[0], axis=0), decimals=4))
                mfcc_systole = list(np.around(np.median(mel_list[1], axis=0), decimals=4))
                mfcc_s2 = list(np.around(np.median(mel_list[2], axis=0), decimals=4))
                mfcc_diastole = list(np.around(np.median(mel_list[3], axis=0), decimals=4))

                aggregate_features.extend(mfcc_s1)
                aggregate_features.extend(mfcc_systole)
                aggregate_features.extend(mfcc_s2)
                aggregate_features.extend(mfcc_diastole)

                feature_list.append(aggregate_features)

                continue

            # perform the calculation for S1
            if each_row[each_index+1] == 1:
                if len(segment_indices[m:]) < 5:
                    continue

                # extract the time and frequency domain features
                s1_amplitude = amp[each_index+1:segment_indices[m+1]+1]
                s1_interval_length, mean_s1_per_beat = time_features(s1_amplitude, m, 0)
                frequency_features(s1_amplitude, m, 0)

                rr_flag = True

            # perform the calculation for Systole
            elif each_row[each_index+1] == 2 and rr_flag==True:
                if len(segment_indices[m:]) < 4:
                    continue

                # extract the time and frequency domain features
                systole_amplitude = amp[each_index+1:segment_indices[m+1]+1]
                sys_interval_length, mean_sys_per_beat = time_features(systole_amplitude, m, 1)
                frequency_features(systole_amplitude, m, 1)

            # perform the calculation for S2
            elif each_row[each_index+1] == 3 and rr_flag==True:
                if len(segment_indices[m:]) < 3:
                    continue

                # extract the time and frequency domain features
                s2_amplitude = amp[each_index+1:segment_indices[m+1]+1]
                s2_interval_length, mean_s2_per_beat = time_features(s2_amplitude, m, 2)
                frequency_features(s2_amplitude, m, 2)

            # perform the calculation for Diastole
            elif each_row[each_index+1] == 4 and rr_flag==True:
                if len(segment_indices[m:]) < 2:
                    continue

                # extract the time and frequency domain features
                diastole_amplitude = amp[each_index+1:segment_indices[m+1]+1]
                diastole_interval_length, mean_diastole_per_beat = time_features(diastole_amplitude, m, 3)
                frequency_features(diastole_amplitude, m, 3)

                if rr_flag == True:

                    # compute the below aggregate measures at the end of each heart cycle
                    ratio_systole_s1.append(mean_sys_per_beat/float(mean_s1_per_beat))
                    ratio_diastole_s2.append(mean_diastole_per_beat/float(mean_s2_per_beat))
                    rr_interval = s1_interval_length+sys_interval_length+s2_interval_length+diastole_interval_length
                    rr_list.append(rr_interval)
                    ratio_systolic_rr.append(sys_interval_length/float(rr_interval))
                    ratio_diastolic_rr.append(diastole_interval_length/float(rr_interval))
                    ratio_systole_diastole.append(sys_interval_length/float(diastole_interval_length))
                    rr_flag = False

    return feature_list

if __name__=="__main__":
    feature_names = []
    if args.train_state is None:
        print "Please specify the path to heart sound state file. Usage: <python feature_extraction.py -h>"
        sys.exit(0)
    if args.train_amps is None:
        print "Please specify the path to heart sound amplitude file. Usage: <python feature_extraction.py -h>"
        sys.exit(0)
    try:
        fp_hs_state = open(args.train_state, 'r')
    except FileNotFoundError:
        print "The specified path to heart sound state file does not exist."
        sys.exit(0)

    read_hs_state = csv.reader(fp_hs_state)

    try:
        fp_hs_amps = open(args.train_amps, 'r')
    except FileNotFoundError:
        print "The specified path to heart sound amplitude file does not exist."
        sys.exit(0)
    read_hs_amps = csv.reader(fp_hs_amps)

    try:
        fp_feat_list = open(args.feat_names, 'r')
    except FileNotFoundError:
        print "The specified path to feature names file does not exist."
        sys.exit(0)

    if args.out is None:
        print "Output path file not specified. Usage: <python feature_extraction.py -h> for more information"
        sys.exit(0)


    # load feature names into a list
    for each_line in fp_feat_list:
        feature_names.append(str(each_line).strip())


    extracted_features = feature_extraction(read_hs_state, read_hs_amps)
    df_feat_ext = pd.DataFrame(extracted_features, columns=feature_names)

    try:
        df_feat_ext.to_csv(args.out, index=False)
    except Exception:
        print "Output path does not exist"
