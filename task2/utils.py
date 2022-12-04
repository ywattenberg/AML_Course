import biosppy
import numpy as np
import pandas as pd
import pywt
import torch
from biosppy.signals import ecg
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler as OverSampler
from sklearn.preprocessing import normalize
from torch.nn import Softmax


def remove_starting_period(data):
    start = 700
    data = data.drop(data.columns[range(start)], axis=1)
    return data

def check_flipped_ecg(data):
    a = (np.max(data, axis=1) <= -0.75*np.min(data, axis=1))
    idx = np.where(a)[0]
    data.loc[idx,:] = - data.loc[idx,:]
    return data

def extract_peaks_intervals(tmpl):
    norm_tmpl = normalize(tmpl)
    wave = np.array(sum(norm_tmpl) / len(norm_tmpl))
    idx = np.where(wave == np.max(wave))
    R_location = idx[0][0]
    before_R = wave[:R_location]
    Q_location = np.where(wave == np.min(before_R[R_location-35:R_location]))
    Q_location = Q_location[0][0]
    P_location = np.where(before_R[0:Q_location] == np.max(before_R[0:Q_location]))
    P_location = P_location[0][0]
    after_R = wave[R_location:]
    S_location = np.where(wave == np.min(after_R[:25]))[0][0]
    T_location = np.where(wave == np.max(after_R[S_location-R_location+1:]))[0][0]

    PR_interval = R_location - P_location
    QRS_interval = S_location - Q_location
    ST_interval = T_location - S_location
    
    return R_location, Q_location, P_location, S_location, T_location, PR_interval, QRS_interval, ST_interval, np.ptp(wave), np.average(wave)

def features(data):
    num_row = data.shape[0]
    num_col = 118  # num of features defined below
      
    output = np.empty((num_row, num_col))
    
    for i in range(num_row):
        # extract ecg information
        idx = np.isfinite(data.loc[i,:])
        ecg_output = biosppy.signals.ecg.ecg(data.loc[i,idx],sampling_rate=300, show=False)
        
        # rpeaks => 5 features
        rpeaks = ecg_output['rpeaks']
        # relative to number of observations
        output[i,0] = rpeaks.shape[0] / np.sum(idx==True)
        #average and std of delta betweeen rpeaks
        output[i,1] = np.mean(np.diff(rpeaks))   
        output[i,2] = np.std(np.diff(rpeaks))        
        output[i,3] = np.quantile(np.diff(rpeaks),0.05) 
        output[i,4] = np.quantile(np.diff(rpeaks),0.95)
        
        # heart rate => 7 features
        hr = ecg_output['heart_rate']
        if hr.shape[0] != 0:
            output[i,5] = np.min(hr)
            output[i,6] = np.max(hr)
            output[i,7] = np.quantile(hr, 0.05)
            output[i,8] = np.quantile(hr, 0.95)
            output[i,9] = np.std(hr)
            output[i,10] = np.std(hr)
        else:
            output[i,5:11] = float("nan")
            
        # templates => 
        tmpl = ecg_output['templates']
        # median, mean and std across templates (rows) for each time point (col)
        data_mean = np.mean(tmpl, axis=0)
        data_median = np.median(tmpl, axis=0)
        data_std = np.std(tmpl, axis=0)
        data_max = np.max(tmpl, axis=0)
        data_min = np.min(tmpl, axis=0)
        output[i,11] = np.mean(np.abs(data_mean - data_median))
        output[i,12] = np.mean(np.abs(data_max))
        output[i,13] = np.mean(np.abs(data_min))
        output[i,14] = np.mean(data_std)
        output[i,15] = output[i, 14] / output[i, 12]  # like a normalized volatility (peak=1)
        
        # wavelet transform
        cA, cD = pywt.dwt(ecg_output['templates'], 'db2')
        # calc stats per bucket (covering 20pts)
        k=16
        for j in range(9):
            idx = range(j*10,(j+1)*10)
            data_mean = np.mean(cD[:,idx], axis=0)
            data_median = np.median(cD[:,idx], axis=0)
            data_std = np.std(cD[:,idx], axis=0)
            data_max = np.max(cD[:,idx], axis=0)
            data_min = np.min(cD[:,idx], axis=0)
            output[i,k] = np.mean(data_median)
            output[i,k+1] = np.mean(data_max)
            output[i,k+2] = np.mean(data_min)
            output[i,k+3] = np.mean(data_mean)   
            output[i,k+4] = np.mean(data_std)       
            # same with data normalized by peak value
            data_mean = np.mean(cA[:,idx], axis=0)
            data_median = np.median(cA[:,idx], axis=0)
            data_std = np.std(cA[:,idx], axis=0)
            data_max = np.max(cA[:,idx], axis=0)
            data_min = np.min(cA[:,idx], axis=0)
            output[i,k+5] = np.mean(data_median)
            output[i,k+6] = np.mean(data_max)
            output[i,k+7] = np.mean(data_min)
            output[i,k+8] = np.mean(data_mean)   
            output[i,k+9] = np.mean(data_std)               
            k = k+10
            
        
        # add R, Q, P, S, T and their intervals
        peaks_intervals = extract_peaks_intervals(tmpl)
        for feature in peaks_intervals:
            output[i, k] = feature
            k += 1
            
        # ptp = np.ptp(data.loc[i,idx])
        # avg = np.avg(data.loc[i,idx])
        # output[i, k] = ptp
        # output[i, k+1] = avg
        
            
        if i % 500 == 0:
            print(i)
            
    col_mean = np.nanmean(output, axis=0)
    idx = np.where(np.isnan(output))
    print('Nr of NaNs: ', idx[1].shape[0])
    print(idx)
    output[idx] = np.take(col_mean, idx[1])
    
    return output

def process_data(data):
    data = remove_starting_period(data)
    data = check_flipped_ecg(data)
    return features(data)

def mean_sqrd_diff(rpeaks):
    diff = np.diff(rpeaks)
    mean_sqrd = np.mean(diff*diff)
    return mean_sqrd

def obtain_features(signal, sampling_rate):
    
    # features obtained from biosppy
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg.ecg(signal, sampling_rate, show = False)
    
    # Correct R-peak locations to the maximum --- introduce some tolerance level
    rpeaks = ecg.correct_rpeaks(signal = signal, rpeaks = rpeaks, sampling_rate = sampling_rate, tol = 0.01)  
    
    # extracting values of R-peaks -- Note: rpeaks gives only indices for R-peaks location
    peak_values = signal[rpeaks]
    
    # Set heart rates to array of nans if contains no elements, otherwise min and max are not defined
    if len(heart_rate) == 0:
        heart_rate = np.array([np.nan, np.nan])
    if len(heart_rate_ts) == 0:
        heart_rate_ts = np.array([np.nan, np.nan])
    
    # Add a bunch of features
    feats = np.array([])
    feats = np.append(feats, np.mean(peak_values))
    feats = np.append(feats, np.median(peak_values))
    feats = np.append(feats, np.min(peak_values))
    feats = np.append(feats, np.max(peak_values))
    feats = np.append(feats, np.std(peak_values))
    feats = np.append(feats, np.mean(rpeaks))
    feats = np.append(feats, np.median(rpeaks))
    feats = np.append(feats, np.min(rpeaks))
    feats = np.append(feats, np.max(rpeaks))
    feats = np.append(feats, np.std(rpeaks))
    feats = np.append(feats, np.sqrt(mean_sqrd_diff(rpeaks)))
    feats = np.append(feats, np.mean(np.diff(rpeaks)))
    feats = np.append(feats, np.median(np.diff(rpeaks)))
    feats = np.append(feats, np.min(np.diff(rpeaks)))
    feats = np.append(feats, np.max(np.diff(rpeaks)))
    feats = np.append(feats, np.std(np.diff(rpeaks)))
    feats = np.append(feats, np.mean(templates, axis = 0))
    feats = np.append(feats, np.median(templates, axis = 0))
    feats = np.append(feats, np.min(templates, axis=0))
    feats = np.append(feats, np.max(templates, axis=0))
    feats = np.append(feats, np.std(templates, axis = 0))
    feats = np.append(feats, np.mean(heart_rate))
    feats = np.append(feats, np.median(heart_rate))
    feats = np.append(feats, np.min(heart_rate))
    feats = np.append(feats, np.max(heart_rate))
    feats = np.append(feats, np.std(heart_rate))
    feats = np.append(feats, np.mean(heart_rate_ts))
    feats = np.append(feats, np.median(heart_rate_ts))
    feats = np.append(feats, np.min(heart_rate_ts))
    feats = np.append(feats, np.max(heart_rate_ts))
    feats = np.append(feats, np.std(heart_rate_ts))
    # Once again check, if heart_rate arrays contain one element min and max of differences will return error
    if len(heart_rate) == 1:
        heart_rate = np.array([np.nan, np.nan])
    if len(heart_rate_ts) == 1:
        heart_rate_ts = np.array([np.nan, np.nan])
    feats = np.append(feats, np.mean(np.diff(heart_rate)))
    feats = np.append(feats, np.median(np.diff(heart_rate)))
    feats = np.append(feats, np.min(np.diff(heart_rate)))
    feats = np.append(feats, np.max(np.diff(heart_rate)))
    feats = np.append(feats, np.std(np.diff(heart_rate)))
    feats = np.append(feats, np.mean(np.diff(heart_rate_ts)))
    feats = np.append(feats, np.median(np.diff(heart_rate_ts)))
    feats = np.append(feats, np.min(np.diff(heart_rate_ts)))
    feats = np.append(feats, np.max(np.diff(heart_rate_ts)))
    feats = np.append(feats, np.std(np.diff(heart_rate_ts)))
    
    #feats = np.append(feats, np.abs(np.fft.rfft(np.mean(templates, axis=0), axis=0))[0:45] # adding FFT (choose only half of entries)
    '''removed fft -- no improvements by adding it'''

    return feats

def process_data2(data):
    data = remove_starting_period(data)
    data = check_flipped_ecg(data)
    for i in np.arange(data.shape[0]):
        if i == 0:
            row = np.array(data.iloc[i].dropna())
            X_train = [obtain_features(row, 300)]
        else: 
            row = np.array(data.iloc[i].dropna())
            X_train = np.append(X_train, [obtain_features(row, 300)], axis = 0)
    return X_train


def train_loop(dataloader, model, loss_fn, optimizer):
    sm = Softmax(dim=1)
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        pred = model(x).type(torch.float)
        pred_sm = sm(pred)
        y = y.long()
    
        loss = loss_fn(pred_sm, y.squeeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 20 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    sm = Softmax(dim=1)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    not_correct = [0,0,0,0]
    classes = [0,0,0,0]
    
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            pred = model(x).type(torch.float)
            pred_sm = sm(pred)

            # print(pred_sm)
            loss = loss_fn(pred_sm, y.long().squeeze(1))
            test_loss += loss
            pred_sm = np.argmax(pred_sm.numpy(), axis=1)
            correct += np.sum(pred_sm == y.numpy().squeeze(1))

           
            for (i, j) in zip(pred_sm, y.numpy().squeeze(1)):
                classes[int (j)] += 1
                if (i != j):
                    not_correct[int (j)] += 1
                    
    percentage = [0,0,0,0]
    for i in range(4):
        percentage[i] = 100 * (not_correct[i] / classes[i])
        print(f"Class {i} not predicted correctly: {percentage[i]:>0.1f}%")
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error \n Accuracy: {(100 * correct):>1f}%, Avg loss: {test_loss:>8f}")
    return correct * 100, percentage


def oversample_data(X, y):
    
    def ratio_multiplier(y):
        from collections import Counter

        multiplier = {0: 1, 1: 2, 2: 1.5, 3: 3}
        target_stats = Counter(y)
        for key, value in target_stats.items():
            if key in multiplier:
                target_stats[key] = int(value * multiplier[key])
        return target_stats

    # oversample the data
    oversampler = OverSampler(sampling_strategy=ratio_multiplier)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    return X_resampled, y_resampled

def smote(X, y):
    # oversample the data
    oversampler = SMOTE()
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    return X_resampled, y_resampled