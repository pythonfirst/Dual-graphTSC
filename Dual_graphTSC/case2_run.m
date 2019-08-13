clear;
addpath(genpath('D:\matcode\Dual_graphTSC\ml'));
addpath(genpath('D:\matcode\Dual_graphTSC\GraphSC'));
load('jaffe_fea2.mat');
load('CK_fea.mat');

k =256;                % number of basis vectors
mu =1e2;               % MMD regularization
alpha =0.2;%;0.1;            % graph regularization
beta=0.8;
gama=0.1;                     % discriminative criterion
oumiga=0.55;
lambda = 0.02;           % sparsity regularization
nIters = 10;            % number of iterations or TSC
n=1;

acc_dgtsc=[];
acc_dgdtsc=[];
fea1=jaffe_fea2;
fea2=CK_fea;
train1_label= fea1(1:end,1);
Xls= fea1(1:end,2:end);
train2_label= fea2(1:30,1);
Xlt= fea2(1:30,2:end);
%Xl=[Xls;Xlt];
r=size(Xlt,1);
test_label= fea2(31:end,1);
Xut=fea2(31:end,2:end);
label=[train1_label;train2_label];
Xt=[Xlt;Xut];
%Normalization of original data
   Xls = diag(sparse(1./sqrt(sum(Xls.^2,2))))*Xls;
    Xt = diag(sparse(1./sqrt(sum(Xt.^2,2))))*Xt;
     Xlt=Xt(1:r,:);
     Xu=Xt(r+1:end,:);
Xl=[Xls;Xlt];
mod= svmtrain(label, Xl,'-t 0 -c 64');
[predict3, accuracy1, ~] = svmpredict(test_label,Xu, mod);


X=[Xls;Xt];
[COEFF SCORE latent]=pca(X);
u=cumsum(latent)./sum(latent);
h=length(find(u<0.96));
newX=SCORE(:,1:h);
newXs=newX(1:size(Xls,1),:);
newXt=newX(size(Xls,1)+1:end,:);
newXs = diag(sparse(1./sqrt(sum(newXs.^2,2))))*newXs;
    newXt = diag(sparse(1./sqrt(sum(newXt.^2,2))))*newXt;
    
 newXls= newXs;
newXlt = newXt(1:r,:);   
newXut =newXt(r+1:end,:); 
newXl=[newXls;newXlt];
    mod= svmtrain(label, newXl,'-t 0 -c 64');
[predict2, accuracy2,~] = svmpredict(test_label,newXut, mod);

for i=1:n
[B,Sl,Su,stat] = DGDTSC(newXls',newXlt',label,k,alpha,beta,gama,oumiga,lambda,mu,nIters);
train=[Sl';Su'];
W=constructW(newXut);
[~ ,S,~] = GraphSC(newXut', W, k, 0, lambda, 1, B);% SC: donot use the graph item of GraphSC
mod= svmtrain(label, train,'-t 0 -c 64');
 [predict, accuracy, pre] = svmpredict(test_label,S', mod);
 acc_dgdtsc=[acc_dgdtsc accuracy(1)];
end

fprintf('>>svm=%0.4f \n\n',accuracy1(1));
fprintf('>>PCA+svm=%0.4f \n\n',accuracy2(1));
fprintf('>>DGDTSC+svm=%0.4f \n\n',accuracy(1)); 
