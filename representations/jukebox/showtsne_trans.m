function showtnse_trans(prefix, upto)
% showtnse_trans(prefix, [upto])
if ~exist('upto','var'), upto = 10; end
if upto>72, error('the highest layer = 72'); end
fs = 44100/128;
% Context length = 8192 [tokens]

%{
VQ-VAE embedding for 1102500 samples (25 sec):
Bottom-level: 137,812 tokens
Middle-level:  34,453 tokens
Top-level:      8,613 tokens

This makes sense:
[8:8:1102500] -> 137812 points
[32:32:1102500] -> 34453 points
[128:128:1102500] -> 8613 points
%}



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
    subplot(5,ceil(upto/5),i)
    [~,Pcs] = pca(Acts);
    imagesc(t, 1:30, Pcs(:,1:30)', [-50 50]); axis xy;
    ylabel(sprintf('[Lay%02i]PCs',i));
end
xlabel('Time [s]')
end


