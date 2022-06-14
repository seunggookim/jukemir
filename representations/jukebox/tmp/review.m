clear
cd ~/usrfolder/jukemir/representations/jukebox
[y,fs] = audioread('input/classical.00000.wav');
times_wav = (0:numel(y)-1)'/fs;
% fs = 22050 Hz

clf
subplot(511)
plot(times_wav, y); colorbar
% sample_length = 1048576 
% hps.sr = 44100 Hz
xlim([1 1048576/44100]); xlabel('Time [s]')

subplot(512)
load('output/classical.00000_depth1.mat')
% 8192 pnts x 4800 dimensions
[~,Scores] = pca(Representation,'numComp',10);
imagesc(Scores'); colorbar; caxis([-50 50]); axis xy; xlabel('Smpl')

subplot(513)
load('output/classical.00000_depth9.mat')
[~,Scores] = pca(Representation,'numComp',10);
imagesc(Scores'); colorbar; caxis([-50 50]); axis xy; xlabel('Smpl')

subplot(514)
load('output/classical.00000_depth18.mat')
[~,Scores] = pca(Representation,'numComp',10);
imagesc(Scores'); colorbar; caxis([-50 50]); axis xy; xlabel('Smpl')

subplot(515)
load('output/classical.00000_depth36.mat')
[~,Scores] = pca(Representation,'numComp',10);
imagesc(Scores'); colorbar; caxis([-50 50]); axis xy; xlabel('Smpl')

% 1048576/8192 = 128 samples per activation
% 1048576/44100/8192*1000 = 2.9025 msec (one hopping step)
% 8192/(1048576/44100) = 344.5312 Hz 

%%
clear
cd ~/usrfolder/jukemir/representations/jukebox
[y,fs] = audioread('input/classical.00000.wav');
times_wav = (0:numel(y)-1)'/fs;
% fs = 22050 Hz

clf
subplot(411)
plot(times_wav, y); colorbar
xlim([1 1048576/44100]); xlabel('Time [s]')

subplot(412)
load('output/classical.00000_level1_depth36.mat')
% 8192 pnts x 4800 dimensions
[~,Scores] = pca(Representation,'numComp',20);
imagesc(Scores'); colorbar; caxis([-50 50]); axis xy

subplot(413)
load('output/classical.00000_level2_depth36.mat')
[~,Scores] = pca(Representation,'numComp',20);
imagesc(Scores'); colorbar; caxis([-50 50]); axis xy

subplot(414)
load('output/classical.00000_level3_depth36.mat')
[~,Scores] = pca(Representation,'numComp',20);
imagesc(Scores'); colorbar; caxis([-50 50]); axis xy


