import yaml
import mne
import numpy as np
import os


'''
To check in preproc:
* valid electrode config
* Valid sampling rate
* length of record is at least 30s + 30s + 2* segment_length(itself equals to 5 or 10s => total 20s. ). Use 1min30 limit. 
'''

y = yaml.safe_load(open('CONFIG.yaml'))
VALID_CH_LIST = y['VALID_CH_LIST']
VALID_SAMPLING_RATE = y['VALID_SAMPLING_RATE']
MAX_REC_DURATION_IN_STACK = y['MAX_REC_DURATION_IN_STACK']


def get_valid_invalid_channel(channels):
    # take a list of channel names and determine which ones are valid.
    # Return idxs and also names.

    # todo (optional): also return new (standardized) names such that we can rename
    # todo reordering list (within valid) for slicing such that channels will always be in the same order (if needed)

    # Dummy list for debug:
    if channels:  # todo: condition
        valid = [(i, chan) for i, chan in enumerate(channels) if chan in VALID_CH_LIST]
        valid_idxs, valid_names = zip(*valid)
        return {'valid:idxs': valid_idxs,
                'valid_names': valid_names}
    else:
        # if available channels do not correspond to a valid config. Should not happen if preproc was done correctly.
        raise ValueError('Provided channels do not correspond to a valid config...')


def cut_into_segments(data, resampled_rate, segment_duration):

    # data: np array of size (nb_chan, nb_samples)
    # resampled_rate: int
    # segment duration: in seconds
    if not float(segment_duration).is_integer():
        raise ValueError('Segment_duration should be a whole number')
    samples_per_segment = int(resampled_rate) * segment_duration
    if not data.shape[1] >= 2 * samples_per_segment:
        raise ValueError('Record is too short! Check preprocessing. ')
    nb_segments = int((data.shape[1] - samples_per_segment) // samples_per_segment)
    begin_offset = np.random.randint(0, samples_per_segment)
    begin_idxs = samples_per_segment * np.arange(nb_segments) + begin_offset
    end_idxs = samples_per_segment * np.arange(1, nb_segments + 1) + begin_offset
    begin_idxs = np.asarray(begin_idxs, dtype=np.int)
    end_idxs = np.asarray(end_idxs, dtype=np.int)
    segments = [data[:, t[0]:t[1]] for t in zip(begin_idxs, end_idxs)]
    return segments


def load_edf(filename,
             resample_rate=64.,
             resample_method='naive',
             segment_duration=5.,
             info_only=False):
    '''
    * Read edf into np array
    * Resample to resample_rate (in Hz)
    * Remove first ? and last ? seconds.
    * Cut into segments of segment_duration (in seconds, whole number),
    (such that segment_duration corresponds to an integer number of samples)
    '''
    # todo: look at some recordings using edfViewer and see whether we should get rid of the first _ and the last _ seconds ?

    if not float(resample_rate).is_integer():
        raise ValueError('Please provide integer resampling rate. ')

    if not resample_method in ['mne', 'naive']:
        raise ValueError('Resample method should be mne or naive. ')

    if not info_only:
        # 1) load using MNE. Disk-bound.
        raw = mne.io.read_raw_edf(filename, preload=True)
        sfreq = raw.info['sfreq']
        if not float(sfreq).is_integer() and float(sfreq) == VALID_SAMPLING_RATE:
            raise ValueError('Invalid sampling rate. ')
        ch_names = raw.ch_names

        # remove right now the channels that we do not need
        d_chan = get_valid_invalid_channel(ch_names)
        valid_ch_names = d_chan['valid_names']
        raw.pick_channels(valid_ch_names)

        # If record is longer than max_rec_duration, use only part of it.
        rec_duration = int(raw.n_times / sfreq)
        if not type(MAX_REC_DURATION_IN_STACK) == int:
            raise ValueError('Please provide an int value (in seconds) for max duration. ')
        if rec_duration > MAX_REC_DURATION_IN_STACK + 1: # +1 to avoid rounding errors.
            begin_offset_seconds = np.random.random_sample() * int(rec_duration - MAX_REC_DURATION_IN_STACK)
            raw.crop(tmin=begin_offset_seconds, tmax=begin_offset_seconds+MAX_REC_DURATION_IN_STACK)

        # 2) Resample (using MNE or naive method) and keep only the
        # channels that we need. The format is slightly different
        # depending on the method.
        # Note: Resampling will be far too slow to use real time in the case where
        # resample_rate does not divide sampling frequency:
        if not (sfreq / resample_rate).is_integer() and float(resample_rate).is_integer():
            raise ValueError('For this example, resample_rate %.1f does not divide sampling rate %.1f. \n'
                             '(or resample_rate is not a whole number). '
                             'In this case resampling is too slow. Please filter. '
                             'Example: %s' % (float(resample_rate), float(sfreq), filename))
        if resample_method == 'mne':
            raw.resample(float(resample_rate), n_jobs=os.cpu_count()) # if faster than needed, add hanning window.
        # get data array.
        data = raw[:, :][0]
        if resample_method == 'naive':
            resample_factor = int(sfreq / resample_rate) # already checked that it s a whole nb
            data = data[:, ::resample_factor]

        # 3) Cut beginning and end. Default: first and last 30s.
        data = data[:, 30*int(resample_rate):-30*int(resample_rate)]
        # 4) cut into segments. It could also happen that the record is too short.
        return cut_into_segments(data, resample_rate, segment_duration)

    else:
        raw = mne.io.read_raw_edf(filename, preload=False)
        return {'ch_names': raw.ch_names, 'sfreq': raw.info['sfreq'],
                'n_times': raw.n_times}




