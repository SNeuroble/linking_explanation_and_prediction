% explanatory model - power (false pos) & inaccuracy

%% no correlation

it=1;
n=40;

for it=1:100
% for n=10:10:100
    
    T=rand(n,1);
    t_y=rand(n,1);
    
    % explanatory
    [r(it),p(it)]=corr(T,t_y);
    
    % cross-validation predictive
    for sub=1:n
        
        X_new=T;
        X_new(sub)=[];
        y_new=t_y;
        y_new(sub)=[];
        
        X_leftout=T(sub);
        y_leftout=t_y(sub);
        
        mdl=fitlm(X_new,y_new);
        y_predict(sub)=mdl.Coefficients.Estimate(1)+mdl.Coefficients.Estimate(2)*X_leftout;
        
    end
    
    [r_predict(it),p_predict(it)]=corr(y_predict',t_y);
    
%     sq_err[it]=(y-y_predict)^2;
%     it=it+1;
    

end

fp=sum(+p<0.05)/it
fp_predict=sum(+p_predict<0.05 & r_predict>0)/it % 54% of the time, r is significantly negative - should there be a different interpretation

% MSE=mean(sq_err);

%% Some correlation - thx https://www.mathworks.com/matlabcentral/answers/101802-how-can-i-generate-two-correlated-random-vectors-with-values-drawn-from-a-normal-distribution
% right now this is demonstrating relationships between reliability and validity

n=100;
mu = 0;
sigma = 5;
ICC_target=0.1; % ICC=var_p/var_p+var_w
r_xy_target=0.6;
% rep_measurements=10;

it=1;
for n=[100,1000]
for i=1:100
T = mu + sigma*randn(n,2);
R = [1 r_xy_target; r_xy_target 1];
L = chol(R);
T = T*L;
t_x=T(:,1);
t_y=T(:,2);
r_Ty=corr(t_x,t_y);

ids_low=t_y<median(t_y); % for group contrast

sigma_sq_p=var(t_x,1);
sigma_sq_w=sigma_sq_p/ICC_target-sigma_sq_p;
sigma_w=sqrt(sigma_sq_w);
% X=repmat(t_x,1,rep_measurements);
% X=X+sigma_w*randn(n,rep_measurements);
X=repmat(t_x,1,2);
e_w=sigma_w*randn(n,1);
X=X+[e_w, -e_w];
var_tot=var(X(:));
var_p=var(mean(X,2));
var_w=var_tot-var_p;
r_icc_xx=var_p/var_tot;
% r_xx=corr(X(:,1),X(:,2));

%r_xy=r_Ty*sqrt(icc) ?
r_xy_expected(i)=r_Ty*sqrt(r_icc_xx);
[r_xy(i),p_xy(i)]=corr(X(:,1),t_y);

[~,p_ttest(i),ci_ttest(:,i)]=ttest2(X(~ids_low,1),X(ids_low,1),'Tail','right');
d(i)=(mean(X(~ids_low,1))-mean(X(ids_low,1)))/ std_pooled(X(~ids_low,1),X(ids_low,1));

end

tpr_r(it)=mean(p_xy<0.05);
r_mean(:,it)=[mean(r_xy);var(r_xy)];
tpr_ttest(it)=mean(p_ttest<0.05);
ci_ttest_mean(:,it)=mean(ci,2);
d_mean(:,it)=[mean(d);var(d)];

it=it+1;

end

% test: E(r(x,y)) = r(T,y) * E(r(x,x'))
% test what this looks like for t-test

% 100 subs
% p < 0.05 : 39%
% mean(r_xy_100)=0.1818 +/- var 0.0070
% mean lower ci: -0.7313 (upper is inf)
% 1000 subs
% p <0.05: 100%
% mean(r_xy_100)=0.1847 +/- var 0.0008
% mean lower ci: 3.0610

