import numpy as np
import pandas as pd
import scipy.stats as scp
import os
import math
import matplotlib.pyplot as plt
import cmlreaders as cml
from ptsa.data.filters import ButterworthFilter
from ptsa.data.filters import MorletWaveletFilter
from ptsa.data.timeseries import TimeSeries
from scipy.stats.distributions import chi2
from sklearn.metrics import r2_score


class P_episode(object):
    '''An object for handling the Better Oscillation Detection Methods

    Parameters:
        events - a pandas DataFrame containing the event or events (from the same session) being analyzed
        eeg - a ptsa timeseries object containing EEG data. Should contain data for ONE ELECTRODE ONLY, 
        		and therefore should not have a "channel" dimension. Should contain contiguous EEG for an entire session

        event_type - 'WORD' for encoding, 'REC_WORD' for retrieval.

        relstart = beginning of event in ms relative to word onset
        relstop = end of event in ms relative to word onset

        width - the length of the MorletWaveletFilter
        sr - the sample rate for the EEG signal. If none, it uses CMLReader to find it 
        lowfreq - the lowest frequency to be calculated
        highfreq - the highest frequency to be calculated
        numfreqs - the number of frequencies logspaced between lowfreq and highfreq to be calculated
        
        percentthresh - Percentile (default 95th) of chi-squared distribution of powers. Describes how much power is deemed a 
        				significant oscillation. 
        numcyclesthresh - how many cycles the power threshold must last to be considered significant. 3 by deault
        
    Attributes: 
        detected - a 3d list/array containing which frequencies have significant osccilations at which times 
            (lists, frequencies, times)
        pepisode -  a 2d array containing the pepisode for each frequency (significant osccilations over time) 
            (events, frequencies)
        freqs - an array of the frequencies calculated
        meanpower - a 2d array containing the mean power at each frequency
            (event, frequencies)
        tfm - a 3d array containing the origional time frequency matrix for each list
            (lists, frequencies, times)
        lists - lists for which BOSC was computed
    Methods:
    	background_fit - plots power spectrum with background fit
    	raw_trace - visualization of EEG trace with true oscillations highlighted

    
    '''
    
    def __init__(self, events, eeg, event_type = 'WORD', relstart = 300, relstop = 1301, width=5, sr=None, lowfreq=2, highfreq=120, numfreqs=30, 
    			percentthresh=.95, numcyclesthresh=3):
        
        self.events = events
        self.event_type = event_type
        self.interest_events = events[events.type == event_type]
        self.width = width
        if sr is None:
            self.sr = eeg.samplerate.values
        else:
            self.sr = sr
        self.freqs = np.logspace(np.log10(lowfreq), np.log10(highfreq), numfreqs)
        self.percentthresh = percentthresh
        self.numcyclesthresh = numcyclesthresh
        self.relstart = relstart
        self.relstop = relstop
        
        #Get the time freqency matrix for the events
        self.eeg = eeg
        self.__BOSC_tf()
        
        self.detected = [[[] for i in range(len(self.freqs))] for j in range(len(self.lists))] # will be list, freqs, time
        self.Pepisode = np.zeros((len(self.interest_events), len(self.freqs))) # will be events, freqs
        
        self.meanpower = []# will be events by frequency
        # NB: meanpower is the estimated background power spectrum (average power as a function of frequency)
        
        self.powerthresh = []
        self.durthresh = [] # will be events by frequency

        
        for i, lst in enumerate(self.tfm):
            #Estimate the background spectrum for a power frequency plot with a linear regression 
            #in log space. 
            self.__BOSC_bgfit(lst)
            
            #STEP THREE: Calculate the threshold values to use for detection
            self.__BOSC_threshholds(self.meanpower[i])
            # *** Hint: At this stage, it is a good idea to cross-check the background power spectrum fit 
            #(see PLOT #1: Power spectrum and background spectrum fit)
            for num, freq in enumerate(self.freqs):
                self.detected[i][num] = np.zeros(len(self.tfm[i][num]))
                
        for ev_idx, event in enumerate(self.interest_events.iterrows()):
            event = event[1]
            #STEP FOUR: Set the target signal in which oscillations will be detected.
            lst_idx = self.lists.index(event.list)
            mat = self.tfm[lst_idx]
            bools = np.logical_and(self.list_times[lst_idx] >= (event.eegoffset*(1000/self.sr) + relstart), 
                                   self.list_times[lst_idx] < (event.eegoffset*(1000/self.sr) + relstop))
            mat = mat[:, bools]
            for num, freq in enumerate(self.freqs):
                self.detected[lst_idx][num][bools] = self.__BOSC_detect(
                    mat[num], 
                    self.powerthresh[lst_idx][num], 
                    self.durthresh[lst_idx][num])
                self.Pepisode[ev_idx][num] = np.mean(self.detected[lst_idx][num][bools]) 
                #Pepisode as a function of frequency.
        
        #convert all nested lists to numpy arrays
        self.detected = np.array(self.detected)
        self.meanpower = np.array(self.meanpower)
        self.powerthresh = np.array(self.powerthresh)
        self.durthresh = np.array(self.durthresh)
    

    def __BOSC_tf(self):
        '''
        Gets the time frequency matrix for events

        This function computes a continuous wavelet (Morlet) transform on 
        the events of the BOSC object; this can be used to estimate the
        background spectrum (BOSC_bgfit) or to apply the BOSC method to
        detect oscillatory episodes in signal of interest (BOSC_detect).
        '''
        
        from ptsa.data.filters import MorletWaveletFilter
        
        wf = MorletWaveletFilter(timeseries=self.eeg, freqs=self.freqs, width=self.width, output='power')
        pows = wf.filter().data  #output is freqs, events, and time
        
        # inconsistent event labeling
        
        start_type, end_type = self.__get_event_keywords()
        
        list_events = self.events[np.logical_or(self.events.type==start_type, self.events.type==end_type)]
        while list_events.type.iloc[0] != start_type:
            list_events = list_events.iloc[1:]
        lists = list(list_events.list.unique())
        self.tfm = []
        self.list_times = []
        self.lists = []
        for lst in lists:
            start = list_events[(list_events.type == start_type) & (list_events.list == lst)].eegoffset.values
            end = list_events[(list_events.type == end_type) & (list_events.list == lst)].eegoffset.values
            
            #account for differences in samplerate - eegoffset is in samples, so convert to time in ms (same as eeg.time)
            start = int(start*(1000/self.sr))
            end = int(end*(1000/self.sr))
            
            if start > self.eeg.time.values[-1]:
                print('No corresponding EEG data for list {}'.format(lst))
                continue
            tfm = pows[:, (self.eeg.time >= start) & (self.eeg.time <= end)]
            if tfm == []:
                raise Exception('Empty powers')
            self.tfm.append(tfm)
            self.list_times.append(self.eeg[(self.eeg.time >= start) & (self.eeg.time <= end)].time.data)
            #only record successful lists
            self.lists.append(lst)
        self.interest_events = self.interest_events[np.isin(self.interest_events.list, self.lists)]
    
    def __get_event_keywords(self):
        if self.event_type == 'WORD': 
            if np.any(np.isin(self.events.type, ['ORIENT'])):
                start_type = 'ORIENT'
                end_type = 'DISTRACT_START'
            else:
                start_type = 'ENCODING_START'
                end_type = 'ENCODING_END'
        elif self.event_type == 'REC_WORD':
            start_type = 'REC_START'
            end_type = 'REC_END'
        else:
            raise Exception('event_type' + self.event_type + 'not supported. Use either WORD or REC_WORD.')
        return start_type, end_type   
 
    def __BOSC_bgfit(self, tfm):
        '''
        This function estimates the background power spectrum via a linear regression fit to the power
        spectrum in log-log coordinates

        parameters: 
            tfm - the time-frequency matrix for the event to be analyzed
        '''
        #linear regression
        fit = np.polyfit(np.log10(self.freqs),np.mean(np.log10(tfm),1),1) 
        #transform back to natural units (power; usually uV^2/Hz)
        self.meanpower.append(10.**(np.polyval(fit, np.log10(self.freqs))))
        
    def __BOSC_threshholds(self, meanpower):
        '''
        This function calculates all the power thresholds and duration
        thresholds for use with BOSC_detect to detect oscillatory episodes

        returns:
        power thresholds and duration thresholds
        '''
        from scipy.stats.distributions import chi2
        self.powerthresh.append(chi2.ppf(self.percentthresh, 2)*meanpower/2)

        #duration threshold is simply a certain number of cycles, so it scales with frequency
        self.durthresh.append(self.numcyclesthresh*self.sr/self.freqs)
        
    def __BOSC_detect(self, timecourse, powthresh, durthresh):
        '''
        This function detects oscillations based on a wavelet power
        timecourse, b, a power threshold (powthresh) and duration
        threshold (durthresh) returned from BOSC_thresholds.m.

        It now returns the detected vector which is already episode-detected.

        timecourse - the power timecourse (at one frequency of interest)

        durthresh - duration threshold in  required to be deemed oscillatory
        powthrseh - power threshold

        returns:
        detected - a binary vector containing the value 1 for times at
                   which oscillations (at the frequency of interest) were
                   detected and 0 where no oscillations were detected.

        To calculate Pepisode:
        Pepisode=length(find(detected))/(length(detected));
        '''
        # def diff(x):
        #     y = [np.subtract(x[num+1],x[num], dtype=np.float32) for num in range(len(x)-1)]
        #     return np.array(y)

        nT=len(timecourse)-1 #number of time points, used as the last time point

        x = timecourse > powthresh #Step 1: power threshold
        dx = np.diff(x)

        # pos represents the times where the timecourse goes above the threshhold
        # neg represents the times where the timecourse dips below the threshhold
        pos = np.nonzero(dx == 1)[0] # np.nonzero returns a tuple, we only care about the first value
        neg = np.nonzero(dx == -1)[0] # show the +1 and -1 edges

        detected = np.zeros(len(timecourse))#this will be the returned list
        # now do all the special cases to handle the edges
        # h is the start time and the end time of episodes
        if pos.size==0 and neg.size==0: #no positive or negative changes
            H = np.array([[0], [nT]]) if x.any()>0 else np.array([[]])
            # all values are above the threshhold or no values are above the threshhold
        elif pos.size==0: H=np.stack([[0], [neg[0]]], axis=0) # the timecourse starts on an episode, and dips below
        elif neg.size==0: H=np.stack([[pos[0]],[nT]], axis=0) # ends on its only episode    
        else:
            if pos[0]>neg[0]: pos = np.insert(pos, 0, 0) # starts with an episode
            if pos[-1]>neg[-1]: neg=np.insert(neg, len(neg), nT) # ends with an episode
            H = np.stack([pos, neg], axis=0) # NOTE: by this time, length(pos)==length(neg), necessarily
        if H.size!=0: #if there is at least one episode then find epochs lasting longer than minNcycles*period
            goodep = np.nonzero((H[1]-H[0])>=durthresh)[0]#the episodes with a long enough duration
            H = H[:, goodep] if goodep.size else []
            for episode in np.array(H).T:
                detected[episode[0]:episode[1]]=1#from start to end there is an episode
        return detected

    def background_fit(self, plot_type = 'list', list_idx = None):
        """
        Visualize the data's fit to the 1/f background spectrum
        
        Parameters:
        plot_type - if 'session', average over all lists. If list, plot for specified list.
        list_idx - index of list to plot. Only required if plot_type is 'list'.
        """


        def plot_curve(power, fit):
            plt.plot(self.freqs, power, 'ko-', linewidth=2, alpha = 0.5)
            plt.plot(self.freqs,  fit, 'r', linewidth=2)
            plt.ylabel(r'Log(Power) [$\mu V^2 / Hz$]')
            plt.xlabel('Frequency [Hz]')
            plt.title('Power spectrum and background spectrum fit')
            plt.legend(['Power spectrum', 'background fit'])
            
        if plot_type == 'session':
            log_power = []
            fit = []
            # average over lists so that plot is more readable
            for list_idx, l in enumerate(self.lists):
                log_power.append(np.mean(np.log10(self.tfm[list_idx]),1))
                fit.append(np.log10(self.meanpower[list_idx]))
            r2 = r2_score(np.array(log_power).T, np.array(fit).T, multioutput = 'uniform_average')
            plot_curve(np.mean(log_power, 0), np.mean(fit, 0))
            return r2
        if plot_type == 'list':
            if list_idx is None:
                for list_idx, l in enumerate(self.lists):
                    plot_curve(np.mean(np.log10(self.tfm[list_idx]),1), np.log10(self.meanpower[list_idx]))
            else:
                plot_curve(np.mean(np.log10(self.tfm[list_idx]),1), np.log10(self.meanpower[list_idx]))
            return r2_score(np.mean(np.log10(self.tfm[list_idx]),1), np.log10(self.meanpower[list_idx]))
        
    def raw_trace(self, freq_idx, list_idx=0, filtered = False, ax = None):
        """
        Visualize the raw_eeg at a specific frequency, with significant oscillations highlighted.

        Parameters:
        freq_idx - index of 1D array in P_episode().freqs attribute
        list_idx - index of list to visualize

        """
        if ax is None:
            ax = plt.subplot(1,1,1)
        bools = np.logical_and(self.eeg.time >= self.list_times[list_idx][0], 
                               self.eeg.time < self.list_times[list_idx][-1])
        if filtered:
            complex_mat = MorletWaveletFilter(timeseries=self.eeg,
                             freqs= self.freqs,
                             width = self.width, output = 'complex'                  
                             ).filter()
            target_signal = np.real(complex_mat)[freq_idx][bools]
        else:
            target_signal = self.eeg[bools]
        osc = np.copy(target_signal)
        #where the oscilations are
        osc[np.nonzero(self.detected[list_idx, freq_idx][:-1]==0)[0]]=None #TODO: incompatible length with self.detected necessitates [:-1]
        time = range(len(target_signal))/self.sr
        #plot the normal graph
        ax.plot(time, target_signal,'k', linewidth=.25, label = 'EEG Time Series')
        #highlight the oscilations
        ax.plot(time, osc, 'r', linewidth=2, label = 'Detected Oscillations')
        #shade word presentation events
        if (self.event_type == 'WORD') & np.any(np.isin(self.events.type, ['WORD_OFF'])):
            local_events = self.events[np.logical_or(self.events.type =='WORD', self.events.type =='WORD_OFF')]
            local_events = local_events[np.logical_and(
                                            local_events.eegoffset*(1000/self.sr)>=self.list_times[list_idx][0],
                                            local_events.eegoffset*(1000/self.sr)<=self.list_times[list_idx][-1]
                                        )]
            list_start = self.list_times[list_idx][0]
            for start, end in zip(local_events[local_events.type == 'WORD'].eegoffset*(1000/self.sr),
                                 local_events[local_events.type == 'WORD_OFF'].eegoffset*(1000/self.sr)):       
                ax.axvspan((start - list_start)/1000, (end - list_start)/1000, alpha = 0.2)
        else:
            local_events = self.interest_events[np.logical_and(
                                            self.interest_events.eegoffset*(1000/self.sr)>=self.list_times[list_idx][0],
                                            self.interest_events.eegoffset*(1000/self.sr)<=self.list_times[list_idx][-1]
                                        )]
            list_start = self.list_times[list_idx][0]
            for start in local_events.eegoffset.values*(1000/self.sr):
                ax.axvspan((start-list_start)/1000, (start + self.relstop - list_start)/1000, alpha = 0.2)
        ax.set_ylabel(r'Voltage [$\mu V$]'); ax.set_xlabel('Time [s]')
        ax.set_title('Frequency: {} Hz'.format(round(self.freqs[freq_idx],2)))
        if ax is None:
            ax.legend(['EEG Time Series', 'Detected Oscillations', 'Word Presentation'])

## END OF CLASS

def calc_subj_pep(subj, elecs = None, method = 'bip', relstart = 300, relstop = 1301, freq_specs = (2, 120, 30), 
    percentthresh=.95, numcyclesthresh=3, load_eeg = False, save_eeg = False, save_result = False, plot = False, 
    kind = 'r1', experiment = 'FR1', eeg_path = '~/', result_path = '~/'):
    """
    Inputs:
    subj - subject string
    elecs - list of electrode pairs (strings)
    method - bip or avg depending on referencing scheme
    freq_specs - tuple of (low_freq, high_freq, num_freqs) for background fitting in BOSC.
    
    Returns:
    pep_all - average Pepisode for all words at each frequency
    pep_rec - average Pepisode for recalled words at each frequency
    pep_nrec - average Pepisode for non-recalled words at each frequency
    subj_tscores - t-score at each frequency, comparing rec and nrec across events
    ** Note that tscore is not itself meaningful because events are not independent. Comparing these 
        tscores across subjects, however, is valid.
    """
    import numpy as np
    import scipy.stats as scp
    import pandas as pd
    import cmlreaders as cml
    import matplotlib.pyplot as plt
    from ptsa.data.filters import ButterworthFilter
    import P_episode as pep
    
    if save_eeg and load_eeg:
        raise('Cannot save and load eeg simultaneously.')

    print('Subject: ', subj)
    if elecs is None:
        good_subj = pd.read_pickle('hippo_subject_pairs.csv')
        elecs = good_subj[good_subj['Subject']==subj]['hippo_pairs'].iloc[0]
    subj_pepisode = None
    subj_recalled = None
    subj_tscores = None
    if plot:
        plt.figure(figsize = (12, 6))
    lowfreq, highfreq, numfreqs = freq_specs
    print(elecs)
    for pair_str in elecs:
        chans = pair_str.split('-')
        print(chans)
        data = cml.get_data_index(kind = kind); data = data[data['experiment'] ==experiment]
        sessions = data[data['subject']==subj]['session'].unique()
        pepisodes = None # events, freqs
        recalled = None # events, freqs   
        tscore = None
        for sess in sessions:
            try:
                print('Loading session {} EEG'.format(sess))
                reader = cml.CMLReader(subject = subj, experiment = experiment, session = sess)
                all_events = reader.load('task_events')
                # path = '/home1/jrudoler/Saved_files/bosc_referencing/'+subj+'/'+method+'/eeg/'
                if not os.path.exists(eeg_path):
                    os.makedirs(eeg_path)
                if load_eeg:
                    eeg = TimeSeries.from_hdf(eeg_path + 'session_' + str(sess) + '_' + pair_str)
                    bosc = P_episode(all_events, eeg, sr = eeg.samplerate.values,
                                    lowfreq = lowfreq, highfreq=highfreq, numfreqs = numfreqs)
                elif method == 'bip': 
                    pairs = reader.load("pairs")
                    #bipolar eeg
                    bip = reader.load_eeg(scheme = pairs[pairs.label ==pair_str]).to_ptsa().mean(['event', 'channel'])
                    bip = ButterworthFilter(bip, freq_range=[58., 62.], filt_type='stop', order=4).filter()
                    print("Applying BOSC method!")
                    if save_eeg:
                    	bip.to_hdf(eeg_path + 'session_' + str(sess) + '_' + pair_str)
                    bosc = P_episode(all_events, bip, sr = bip.samplerate.values, 
                                    lowfreq = lowfreq, highfreq=highfreq, numfreqs = numfreqs)

                elif method == 'avg':
                    contacts = reader.load("contacts")
                    #average eeg
                    eeg = reader.load_eeg(scheme = contacts).to_ptsa().mean('event')
                    #all zeros from a broken lead leads to -inf power, which results in a LinAlg error for log-log fit
                    #TODO: verify this channel exclusion doesn't cause any problems. Maybe print a message or raise an error?
                    bad_chan_mask = ~np.all(eeg.values==0, axis = 1)
                    contacts = contacts[bad_chan_mask]
                    eeg = eeg[bad_chan_mask, :]
                    avg = (eeg[contacts.label.str.contains(chans[0]) | contacts.label.str.contains(chans[1]), :] \
                           - eeg.mean('channel')).mean('channel')
                    avg = ButterworthFilter(avg, freq_range=[58., 62.], filt_type='stop', order=4).filter()
                    if save_eeg:
                    	avg.to_hdf(eeg_path + '/session_' + str(sess) + '_' + pair_str)
                    bosc = P_episode(all_events, avg, sr = avg.samplerate.values,
                                    lowfreq = lowfreq, highfreq=highfreq, numfreqs = numfreqs)
                    
                if plot:
                    r2 = bosc.background_fit(plot_type = 'session')

                if pepisodes is None:
                    pepisodes = bosc.Pepisode
                    #be careful to only use events from lists that have eeg data.
                    recalled = bosc.interest_events.recalled.values #[np.isin(bosc.interest_events.list, self.lists)]
                    tscore, _ = scp.ttest_ind(pepisodes[recalled], pepisodes[~recalled], axis = 0)
                elif np.isnan(tscore).all():
                    tscore, _ = scp.ttest_ind(pepisodes[recalled], pepisodes[~recalled], axis = 0)
                else:
                    pepisodes = np.vstack([pepisodes, bosc.Pepisode])
                    recalled = np.hstack([recalled, bosc.interest_events.recalled.values])
                    t, _ = scp.ttest_ind(pepisodes[recalled], pepisodes[~recalled], axis = 0)
                    tscore = np.vstack([tscore, t])
                print(recalled.mean())
                print(np.shape(tscore))
            except IndexError:
                print('Error for subject {} session {}'.format(subj, sess))
        if pepisodes is None:
            raise Exception('No working sessions')
        subj_pepisode = pepisodes if subj_pepisode is None else np.dstack([subj_pepisode, pepisodes])
        subj_recalled = recalled if subj_recalled is None else np.vstack([subj_recalled, recalled])
        subj_tscores = tscore if subj_tscores is None else np.vstack([subj_tscores, tscore])
        if np.isnan(subj_tscores).all():
            raise Exception('Too many nan in T-scores')
    if subj_pepisode.ndim > 2: #if multiple electrode pairs, average over pairs
        print("Averaging over {} electrodes for subject {}".format(subj_pepisode.shape[2], subj))
        subj_pepisode = subj_pepisode.mean(2)
        subj_recalled = subj_recalled.mean(0)
    subj_recalled = subj_recalled.astype(bool)
    if subj_tscores.ndim > 1:
        print(len(sessions), 'sessions')
        subj_tscores = np.nanmean(subj_tscores, axis = 0)
    
    
    print('{} total events: {} recalled and {} non-recalled'.format(len(subj_recalled), 
                                                                    sum(subj_recalled), 
                                                                    sum(~subj_recalled)))
    
    pep_rec = subj_pepisode[subj_recalled, :].mean(0)
    pep_nrec = subj_pepisode[~subj_recalled, :].mean(0)
    pep_all = subj_pepisode.mean(0)

    if save_result:
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        np.save(result_path + '{}_all_{}'.format(subj, method), pep_all)
        np.save(result_path + '{}_rec_{}'.format(subj, method), pep_rec)
        np.save(result_path + '{}_nrec_{}'.format(subj, method), pep_nrec)
        np.save(result_path + '{}_tscore_{}'.format(subj, method), subj_tscores)
    
    
    return pep_all, pep_rec, pep_nrec, subj_tscores


## PLOTTING ##

def plot_pepisode(pep_all, pep_rec, pep_nrec, freqs = np.round(np.logspace(np.log10(2), np.log10(120), 30), 2), ax = None, title = ''):
    """
    Function for plotting P_episode by frequency, averaged separately across events for recalled and not recalled words.

    Inputs:
    pep_rec - 1D array-like containing P_episode for recalled events. Length equal to number of frequencies
    pep_nrec - 1D array-like containing P_episode for non-recalled events. Length equal to number of frequencies
    freqs - Frequencies for which P_episode is calculated. Length must equal pep_rec and pep_nrec.
    
    """
    if ax is None:
        ax = plt.subplot(1,1,1)

    ax.plot(freqs, pep_all, '-o', alpha = 0.5, label = 'All events')
    ax.plot(freqs, pep_rec, '-o', label = 'Recalled')
    ax.plot(freqs, pep_nrec, '-o', label = 'Not recalled')

    ax.set_title(title, fontsize = 16) 


    ax.set_xlabel('Frequency (Hz)', fontsize = 14)
    ax.set_ylabel('P-episode', fontsize = 14)

    ax.legend()
    plt.tight_layout()


def plot_pepisode_multi_subj(all_subj_pep_all = None, all_subj_pep_rec = None, all_subj_pep_nrec = None, all_subj_ttest = None, 
    subjects = None, method = 'bip', figsize = (12,10), 
    freqs = np.round(np.logspace(np.log10(2), np.log10(120), 30), 2), pep_path = ''):
    
    """
    Inputs:
    all_subj_pep_all - 2D array (subjects, frequencies) of P_episode values for all events
    all_subj_pep_rec - 2D array (subjects, frequencies) of P_episode values for all events
    all_subj_pep_nrec - 2D array (subjects, frequencies) of P_episode values for all events
    all_subj_ttest - 2D array (subjects, frequencies) of tscores comparing successful and unsuccesful encoding/recall

    Alternatively:
    subjects - list of subject ID's
    method - referencing method, either 'avg' or 'bip'
    pep_path - path specifying location of files with P_episode information
    """


    if all_subj_pep_all is None and subjects is None:
        raise('Must provide either P_episode statistics directly or specify a list of subjects \
                along with a referencing method and path')
    elif subjects is not None:
        all_subj_pep_all = []
        all_subj_pep_rec = []
        all_subj_pep_nrec = []
        all_subj_ttest = []
        success = []
        for subj in subjects:
            try:
                pep_all = np.load(pep_path + '{}_all_{}.npy'.format(subj, method))
                pep_rec = np.load(pep_path + '{}_rec_{}.npy'.format(subj, method))
                pep_nrec = np.load(pep_path + '{}_nrec_{}.npy'.format(subj, method))
                tscore = np.load(pep_path + '{}_tscore_{}.npy'.format(subj, method))
                success.append(subj)
            except:
                continue
            all_subj_pep_all.append(pep_all)
            all_subj_pep_rec.append(pep_rec)
            all_subj_pep_nrec.append(pep_nrec)
            all_subj_ttest.append(tscore)
            
        all_subj_pep_all = np.vstack(all_subj_pep_all)
        all_subj_pep_rec = np.vstack(all_subj_pep_rec)
        all_subj_pep_nrec = np.vstack(all_subj_pep_nrec)
        all_subj_ttest = np.vstack(all_subj_ttest)

    t_all, p_all = scp.ttest_1samp(all_subj_ttest, popmean = 0, nan_policy='omit') #TODO: is 'omit' an acceptable policy?
    plt.figure(figsize = figsize)
    ax1 = plt.subplot(311)

    ax1.plot(freqs, all_subj_pep_all.mean(0), '-o', c = 'slateblue', label = 'All word presentation events')
    ax1.fill_between(freqs, 
                     all_subj_pep_all.mean(0)+all_subj_pep_all.std(0)/np.sqrt(len(all_subj_pep_all)),
                     all_subj_pep_all.mean(0)-all_subj_pep_all.std(0)/np.sqrt(len(all_subj_pep_all)),
                     color = 'b', alpha = 0.1)
    
    ax2 = plt.subplot(312)
    ax2.plot(freqs, (all_subj_pep_rec-all_subj_pep_nrec).mean(0), c = 'slateblue', label = 'Recalled - Not Recalled')
    
    sig = p_all<0.05
    ax3 = plt.subplot(313)
    ax3.scatter(freqs[sig], t_all[sig], marker = '*', c = 'r', s = 50, label = 'Significant (p<0.05)')
    ax3.plot(freqs, t_all, '-', c = 'slateblue')
    
    ax1.set_title('{} Subjects'.format(len(success)), fontsize = 16) 


    ax3.set_xlabel('Frequency (Hz)', fontsize = 14)
    ax1.set_ylabel('P-episode', fontsize = 14)
    ax2.set_ylabel('P-episode Difference', fontsize = 14)
    ax3.set_ylabel('T-score', fontsize = 14)

    ax1.set_ybound(-0.005)

    ax1.legend()
    ax2.legend()

    plt.tight_layout()
    return success, all_subj_pep_rec, all_subj_pep_nrec, all_subj_ttest