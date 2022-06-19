import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import librosa
from sklearn import preprocessing
import pickle
dir_ = 'Data'
df_3sec = pd.read_csv(f'{dir_}/features_3_sec.csv')
df_3sec = df_3sec.iloc[0:, 1:]
# Breaking Up  X and Y   Independent  and Target Variables )
y = df_3sec['label']
X = df_3sec.loc[:, df_3sec.columns != 'label']
# MinMAX Scaling implementation:
cols = X.columns
scaler = preprocessing.MinMaxScaler()
np_scaled = scaler.fit_transform(X)
X = pd.DataFrame(np_scaled, columns = cols)
# train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#model training
GBC = GradientBoostingClassifier(learning_rate=0.3, n_estimators=2000)
GBCmodel=GBC.fit(X_train, y_train)


def predict_Genre(song):
    y, s = librosa.load(song)
    trim_y, _ = librosa.effects.trim(y)
    '''It will trim leading and trailing silence from an audio signal. 
    In this code, we will remove audio signal that is lower than 10db'''
    chroma_stft = librosa.feature.chroma_stft(y=trim_y, sr=s,n_fft=2048, hop_length=512).flatten()
    rmse = librosa.feature.rms(y=trim_y, frame_length=2048, hop_length=512).flatten()
    spec_cent = librosa.feature.spectral_centroid(y=trim_y, sr=s, n_fft=2048,hop_length=512).flatten()
    spec_bw = librosa.feature.spectral_bandwidth(y=trim_y, sr=s,n_fft=2048, hop_length=512).flatten()
    rolloff = librosa.feature.spectral_rolloff(y=trim_y + 0.01, sr=s,n_fft=2048, hop_length=512).flatten()
    zcr = librosa.feature.zero_crossing_rate(trim_y,frame_length=2048, hop_length=512).flatten()
    y_harmonic, y_percep = librosa.effects.hpss(trim_y)
    tempo,beats = librosa.beat.beat_track(y, sr = s)
    mfcc = librosa.feature.mfcc(y=trim_y, sr=s,win_length=2048, hop_length=512)
    mfcc_mean=mfcc.T.mean(axis=0)
    mfcc_var=mfcc.T.var(axis=0)

    length=y.shape[0]
    chroma_stft_mean=chroma_stft.mean()
    chroma_stft_var=chroma_stft.var()
    rms_mean=rmse.mean()
    rms_var=rmse.var()
    spectral_centroid_mean=spec_cent.mean()
    spectral_centroid_var=spec_cent.var()
    spectral_bandwidth_mean=spec_bw.mean()
    spectral_bandwidth_var=spec_bw.var()
    rolloff_mean=rolloff.mean()
    rolloff_var=rolloff.var()
    zero_crossing_rate_mean=zcr.mean()
    zero_crossing_rate_var=zcr.var()
    harmony_mean=y_harmonic.mean()
    harmony_var=y_harmonic.var()
    perceptr_mean=y_percep.mean()
    perceptr_var=y_percep.var()
    mfcc1_mean=mfcc_mean[0]
    mfcc1_var=mfcc_var[0]
    mfcc2_mean=mfcc_mean[1]
    mfcc2_var=mfcc_var[1]
    mfcc3_mean=mfcc_mean[2]
    mfcc3_var=mfcc_var[2]
    mfcc4_mean=mfcc_mean[3]
    mfcc4_var=mfcc_var[3]
    mfcc5_mean=mfcc_mean[4]
    mfcc5_var=mfcc_var[4]
    mfcc6_mean=mfcc_mean[5]
    mfcc6_var=mfcc_var[5]
    mfcc7_mean=mfcc_mean[6]
    mfcc7_var=mfcc_var[6]
    mfcc8_mean=mfcc_mean[7]
    mfcc8_var=mfcc_var[7]
    mfcc9_mean=mfcc_mean[8]
    mfcc9_var=mfcc_var[8]
    mfcc10_mean=mfcc_mean[9]
    mfcc10_var=mfcc_var[9]
    mfcc11_mean=mfcc_mean[10]
    mfcc11_var=mfcc_var[10]
    mfcc12_mean=mfcc_mean[11]
    mfcc12_var=mfcc_var[11]
    mfcc13_mean=mfcc_mean[12]
    mfcc13_var=mfcc_var[12]
    mfcc14_mean=mfcc_mean[13]
    mfcc14_var=mfcc_var[13]
    mfcc15_mean=mfcc_mean[14]
    mfcc15_var=mfcc_var[14]
    mfcc16_mean=mfcc_mean[15]
    mfcc16_var=mfcc_var[15]
    mfcc17_mean=mfcc_mean[16]
    mfcc17_var=mfcc_var[16]
    mfcc18_mean=mfcc_mean[17]
    mfcc18_var=mfcc_var[17]
    mfcc19_mean=mfcc_mean[18]
    mfcc19_var=mfcc_var[18]
    mfcc20_mean=mfcc_mean[19]
    mfcc20_var=mfcc_var[19]
    feature_array=np.array([length, chroma_stft_mean, chroma_stft_var, rms_mean, rms_var,
           spectral_centroid_mean, spectral_centroid_var,spectral_bandwidth_mean,spectral_bandwidth_var,
               rolloff_mean,rolloff_var,zero_crossing_rate_mean, zero_crossing_rate_var,
           harmony_mean,harmony_var,perceptr_mean,perceptr_var,tempo,
           mfcc1_mean, mfcc1_var, mfcc2_mean,mfcc2_var,mfcc3_mean,
           mfcc3_var,mfcc4_mean, mfcc4_var, mfcc5_mean, mfcc5_var,
           mfcc6_mean,mfcc6_var,mfcc7_mean,mfcc7_var,mfcc8_mean,
           mfcc8_var, mfcc9_mean, mfcc9_var,mfcc10_mean,mfcc10_var,
           mfcc11_mean,mfcc11_var, mfcc12_mean,mfcc12_var,mfcc13_mean,
           mfcc13_var, mfcc14_mean, mfcc14_var, mfcc15_mean, mfcc15_var,
           mfcc16_mean, mfcc16_var, mfcc17_mean, mfcc17_var, mfcc18_mean,
           mfcc18_var,mfcc19_mean, mfcc19_var, mfcc20_mean, mfcc20_var])
    song_test=feature_array.reshape(1,-1)
    Song = pd.DataFrame(song_test, columns = ['length', 'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
       'spectral_centroid_mean', 'spectral_centroid_var',
       'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean',
       'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var',
       'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo',
       'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean',
       'mfcc3_var', 'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean', 'mfcc5_var',
       'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean',
       'mfcc8_var', 'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean', 'mfcc10_var',
       'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var', 'mfcc13_mean',
       'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var',
       'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean',
       'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var'])
    pred=GBCmodel.predict(Song)
    print('This song belongs to ' , pred[0])

if __name__== '__main__':

      #pickling our Model:
    mymusicmodel='MyMGCmodel.pkl'
    fileobj=open(mymusicmodel,'wb')
    pickle.dump(GBCmodel,fileobj)
    fileobj.close()
