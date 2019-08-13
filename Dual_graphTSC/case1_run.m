clear;
addpath(genpath('D:\matcode\Dual_graphTSC\ml'));

load('jaffe_fea2.mat');
load('CK_fea.mat');
k =256;    % number of basis vectors
mu =1e3;       % MMD regularization
alpha=0.2;            % graph regularization
beta=0.8;  %1;
lambda = 0.02;           % sparsity regularization
nIters = 10;            % number of iterations per TSC
n=1;
fea1=jaffe_fea2;
fea2=CK_fea;
train_label= fea1(1:end,1);
Xs= fea1(1:end,2:end);
test_label= fea2(1:end,1);
Xt= fea2(1:end,2:end);
label=[train_label;test_label];

     Xs = diag(sparse(1./sqrt(sum(Xs.^2,2))))*Xs;
    Xt = diag(sparse(1./sqrt(sum(Xt.^2,2))))*Xt;
mod= svmtrain(train_label, Xs,'-t 0  -c 1000');
[predict3, accuracy1, ~] = svmpredict(test_label,Xt, mod);

X=[Xs;Xt];
[COEFF, SCORE, latent]=pca(X);
u=cumsum(latent)./sum(latent);
h=length(find(u<0.96));
newX=SCORE(:,1:h);
newXs=newX(1:size(Xs,1),:);
newXt=newX(size(Xs,1)+1:end,:);
newXs = diag(sparse(1./sqrt(sum(newXs.^2,2))))*newXs;
   newXt = diag(sparse(1./sqrt(sum(newXt.^2,2))))*newXt;
    mod= svmtrain(train_label, newXs,'-t 0 -c 1000');
[predict2, accuracy2,~] = svmpredict(test_label,newXt, mod);
for i=1:n

[B,Ss,St,stat] = DGTSC(newXs',newXt',k,alpha,beta,lambda,mu,nIters);
mod= svmtrain(train_label, Ss','-t 0 -c 1000');
[predict0, accuracy, ~] = svmpredict(test_label,St', mod);
acc_dgtsc=[acc_dgtsc accuracy(1)];
end
fprintf('>>svm=%0.4f \n\n',accuracy1(1));
fprintf('>>PCA+svm=%0.4f \n\n',accuracy2(1));
fprintf('>>DGTSC+svm=%0.4f \n\n',accuracy(1));

