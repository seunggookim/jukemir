function showrep_vqvae(prefix)
Codes = load(['output/',prefix,'_z.mat']);
Hops = [8 32 128];
figure
for i=1:3
    token_length = 1/44100*Hops(i)*1000;
    subplot(2,3,i); 
    plot(double(Codes.(['Z',num2str(i-1)])),'.','LineStyle','none'); 
    xlabel('Tokens'); ylabel('Code index')
    title(sprintf('VV%i-code| %.3f ms/tkn', i-1, token_length))
    hold on
    plot([1 100],[0 0],'color','r', 'LineWidth',8)
    
    subplot(2,3,i+3); 
    plot(double(Codes.(['Z',num2str(i-1)])),'.-'); 
    xlabel('Tokens'); ylabel('Code index')
    title(['VQ-VAE: lvl',num2str(i-1)])
    xlim([0 100])
end

