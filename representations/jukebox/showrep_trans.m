function showrep_trans(prefix, upto)
% showrep_trans(prefix, [upto])
if ~exist('upto','var'), upto = 10; end
if upto>72, error('the highest layer = 72'); end
fs = 44100/128;
% Context length = 8192 [tokens]
t = (0:8192-1)/fs;
figure
for i = 1:upto
    fname = sprintf('output/%s_depth%02i.mat',prefix,i);
    if ~isfile(fname)
      fprintf('"%s" not found\n',fname);
      return; 
    else
      load(fname)
    end
    subplot(10,ceil(upto/10),i)
    [~,Pcs] = pca(Acts);
    imagesc(t, 1:30, Pcs(:,1:30)', [-50 50]); axis xy;
    ylabel(sprintf('[Lay%02i]PCs',i));
end 
xlabel('Time [s]')
end
