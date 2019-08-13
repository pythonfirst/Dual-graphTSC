function [B, Ss, St, stat] = DGTSC(Xs, Xt, num_bases, alpha,beta, lambda, mu, num_iters, Binit, pars)
% Transfer Sparse Coding (DGTSC) Algorithm
%
%    minimize_B,S   0.5*||X - B*S||^2 + alpha*Tr(SLS') + mu*Tr(SMS') + lambda*sum(abs(S(:)))
%    subject to   ||B(:,j)||_2 <= l2norm, forall j=1...size(S,1)
% 
% Input:
% Xs: labeled data matrix, each column is a sample vector
% Xt: unlabeled data matrix, each column is a sample vector
% M: MMD matrix between the labeled and unlabeled data
% num_bases: number of bases
% alpha: graph Laplician parameter
% beta:  sub-domian graph parameter 
% lambda: sparsity penalty parameter
% mu: MMD regularization parameter
% num_iters: number of iteration
% Binit: initial B matrix
% pars: additional parameters to specify (see the code)
%
% Output:
% B: dictionary matrix
% Ss: coding matrix for labeled data
% St: coding matrix for unlabeled data
% stat: other statistics
%

diff = 1e-7;
X = [Xs,Xt];

pars.mFea = size(X,1);
pars.nSmp = size(X,2);
pars.num_bases = num_bases;
pars.num_iters = num_iters;
pars.lambda = lambda;
pars.noise_var = 1;
pars.sigma = 1;
pars.VAR_basis = 1;


% Sparsity parameters
if ~isfield(pars,'tol')
    pars.tol = 0.005;
end

% initialize basis
if ~exist('Binit','var') || isempty(Binit)
    B = rand(pars.mFea,pars.num_bases)-0.5;
	B = B - repmat(mean(B,1), size(B,1),1);
    B = B*diag(1./sqrt(sum(B.*B)));
else
    disp('Using Binit...');
    B = Binit;
end


% initialize t only if it does not exist
t=0;
% statistics variable
stat= [];
stat.fobj_avg = [];
stat.fresidue_avg = [];
stat.fsparsity_avg = [];
stat.fmmd_avg = [];
stat.var_tot = [];
stat.svar_tot = [];
stat.elapsed_time=0;


% Construct the k-NN Graph
%L=constructLs(Xs,Xt,beta,label);
L=constructL(Xs,Xt,beta);
%W = constructW(X');
%DCol = full(sum(W,2));
%D = spdiags(DCol,0,speye(size(W,1)));
%L = D - W;

% Construct the MMD matrix
ns = size(Xs,2);
nt = size(Xt,2);
e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
M = e*e';

% optimization loop
while t < pars.num_iters
    t=t+1;
    start_time= tic;
    
    stat.fobj_total=0;
    stat.fresidue_total=0;
    stat.fsparsity_total=0;
    stat.flaplacian_total = 0;
    stat.fmmd_total = 0;
    stat.var_tot=0;
    stat.svar_tot=0;
       
    % learn coefficients (conjugate gradient)
    if t ==1
        S= learn_coding(B, X, alpha, mu, pars.lambda/pars.sigma*pars.noise_var, L, M);
    else
        S= learn_coding(B, X, alpha, mu, pars.lambda/pars.sigma*pars.noise_var, L, M, S);
    end
    S(isnan(S))=0;
    Ss = S(:,1:ns);
    St = S(:,ns+1:ns+nt);

     % get objective
    [fobj, fresidue, fsparsity, flaplacian, fmmd] = getObjective(B, S, X, alpha, L, mu, M, pars.noise_var, pars.lambda, pars.sigma);

    stat.fobj_total      = stat.fobj_total + fobj;
    stat.flaplacian_total = stat.flaplacian_total + flaplacian;
    stat.fresidue_total  = stat.fresidue_total + fresidue;
    stat.fsparsity_total = stat.fsparsity_total + fsparsity;
    stat.fmmd_total = stat.fmmd_total + fmmd;
    stat.var_tot         = stat.var_tot + sum(sum(S.^2,1))/size(S,1);

    % update basis
    B = learn_dictionary(X, S, pars.VAR_basis);
    
    % get statistics
    stat.fobj_avg(t)      = stat.fobj_total / pars.nSmp;
    stat.fresidue_avg(t)  = stat.fresidue_total / pars.nSmp;
    stat.fsparsity_avg(t) = stat.fsparsity_total / pars.nSmp;
    stat.flaplacian_avg(t) = stat.flaplacian_total / pars.nSmp;
    stat.fmmd_avg(t) = stat.fmmd_total / pars.nSmp;
    stat.var_avg(t)       = stat.var_tot / pars.nSmp;
    stat.svar_avg(t)      = stat.svar_tot / pars.nSmp;
    stat.elapsed_time(t)  = toc(start_time);
    
    
    if t>199
        if(stat.fobj_avg(t-1) - stat.fobj_avg(t)<diff)
            return;
        end
    end
    
    
    fprintf(['epoch= %d  fobj= %f  fresidue= %f  flaplacian= %f  fmmd= %f  fsparsity= %f  elapsed_time= %0.2f ' ...
             'seconds\n'], t, stat.fobj_avg(t), stat.fresidue_avg(t), stat .flaplacian_avg(t),...
            stat.fmmd_avg(t), stat.fsparsity_avg(t), stat.elapsed_time(t));
    
end

end

function [fobj, fresidue, fsparsity, flaplacian, fmmd] = getObjective(A, S, X, alpha, L, mu, M, noise_var, lambda, sigma)
E = A*S - X;
fresidue  = 0.5/noise_var*sum(sum(E.^2));
flaplacian = 0.5*alpha.*trace(S*L*S');
fmmd = 0.5*mu.*trace(S*M*S');
fsparsity = lambda*sum(sum(abs(S/sigma)));
fobj = fresidue + fsparsity + fmmd + flaplacian;
end