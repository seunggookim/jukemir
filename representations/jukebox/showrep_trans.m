function showrep_trans(prefix, layers)
% showrep_trans(prefix, [layers])
if ~exist('layers','var'), layers = round(linspace(1,72,5)); end
% fs = 44100/128;
% Context length = 8192 [tokens]
% t = (0:8192-1)/fs;
figure
iSubplot = 1;
for iLayer = layers
    fname = sprintf('output/%s_depth%02i.mat',prefix,iLayer);
    if ~isfile(fname)
      fprintf('"%s" not found\n',fname);
      return; 
    else
      load(fname)
    end
    subplot(numel(layers),1,iSubplot)
    [~,Pcs,~,~,Expl] = pca(Acts);
    imagesc(Pcs(:,1:30)', [-50 50]); axis xy;
    ylabel(sprintf('[L%02i]PC(%.0f%%)',iLayer, sum(Expl(1:30))))
    iSubplot = iSubplot + 1;
end 
xlabel('Tokens')
end
