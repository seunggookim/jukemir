X={};
for i=1:3
    X{i} = load(['output/codes_lvl',num2str(i-1),'.mat']);
    subplot(3,1,i); 
    plot(double(X{i}.Codes),'.','LineStyle','none'); 
    axis([0 0.1*5512/4^(i-1) 0 2047]);
    xlabel('Tokens'); ylabel('Codes')
    title(['VQ-VAE: scale',num2str(i-1)])
end

