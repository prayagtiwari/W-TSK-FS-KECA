function [fuzzy_membership_scores,residuls]=computing_fuzzy_scores_keca(x,C,sigma)

center=0;
n=0;

[K] = kernel(x',sigma);
[Phi,newData] = kernelECA(x,K,C,center,n);

sn = size(x,1);
residuls = zeros(sn,1);

for i=1:sn
	res_1 = newData(i,:) - x(i,:);
	res_1 =(norm(res_1,2))^2;
	residuls(i) = res_1;

end
LD = line_map(residuls);
fuzzy_membership_scores = (1 - LD).^2 + 0.001;


end


