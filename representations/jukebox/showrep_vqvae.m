function showrep_vqvae(prefix)
Hops = [8 32 128];
figure
for i=1:3
    load(sprintf('output/%s_z%02i.mat', prefix, i-1))
    token_length = 1/44100*Hops(i)*1000;
    subplot(2,3,i); 
    plot(Times, double(Z),'.','LineStyle','none'); 
    xlabel('Time [s]'); ylabel('Code index')
    title(sprintf('VV%i-code| %.3f ms/tkn', i-1, token_length))
    hold on
    plot([0 0.1],[0 0],'color','r', 'LineWidth',8)
    
    subplot(2,3,i+3); 
    plot(Times*1000, double(Z),'.-'); 
    xlabel('Time [ms]'); ylabel('Code index')
    xlim([0 100])
end

