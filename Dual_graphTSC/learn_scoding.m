function Sout = learn_scoding(B, X, alpha, mu,gama, gamma, L, M,Ld, Sinit)
% The feature-sign search algorithm
% L1-regularized least squares problem solver
%
% This code solves the following problem:
% 
%   minimize_s 0.5*(||xi-B*si||^2 + alpha*si'si + 2*alpha*si(sigma(Lijsj))
%   + mu*si'si + 2*mu*si(sigma(Mijsj)) + gamma*||si||_1) 
% X : data matrix
% B : basis matrix
% Sinit : initial coefficient matrix
%
% Modified from Honglak Lee <hllee@cs.stanford.edu>
% Copyright 2007 by Honglak Lee, Alexis Battle, Rajat Raina, and Andrew Y. Ng
%
% Reference:
% M. Long, G. Ding, J. Wang, J. Sun, P.S. Yu.
% Transfer Sparse Coding for Robust Image Representation.
% IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013.

warning('off', 'MATLAB:divideByZero');

use_Sinit= false;
if exist('Sinit', 'var')
    use_Sinit= true;
end

Sout= zeros(size(B,2), size(X,2));
BtB = B'*B;
BtX = B'*X;

% rankB = rank(BtB);
rankB = min(size(B,1)-10, size(B,2)-10);

for i=1:size(X,2)
%     if mod(i, 100)==0, fprintf('.'); end %fprintf(1, 'l1ls_featuresign: %d/%d\r', i, size(X,2)); end
    if use_Sinit
        idx1 = find(Sinit(:,i)~=0);
        maxn = min(length(idx1), rankB);
        sinit = zeros(size(Sinit(:,i)));
        sinit(idx1(1:maxn)) =  Sinit(idx1(1:maxn), i);
        S = [Sout(:,1:i),Sinit(:,(i+1):size(X,2))];
        Sout(:,i) = ls_featuresign_sub (B, S, X(:,i), BtB, BtX(:,i), L, M,Ld, i, alpha, mu,gama, gamma, sinit);
    else
        Sout(:,i) = ls_featuresign_sub (B, Sout, X(:,i), BtB, BtX(:,i), L, M,Ld, i, alpha, mu,gama, gamma);
    end
end

% fprintf(1, '\n');

warning('on', 'MATLAB:divideByZero');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [s, fobj] = ls_featuresign_sub (B, S, x, BtB, Btx, L, M,Ld, i, alpha, mu,gama, gamma, sinit)

[n,m] = size(B);

% rankB = rank(BtB);
rankB = min(size(B,1)-10, size(B,2)-10);

% Step 1: Initialize
usesinit = false;
if ~exist('sinit', 'var') || isempty(sinit)
    sinit= [];
    s= sparse(zeros(m,1));
    theta= sparse(zeros(m,1));
    act= sparse(zeros(m,1));
    allowZero = false;
else
    % sinit = [];
    s= sparse(sinit);
    theta= sparse(sign(s));
    act= sparse(abs(theta));
    usesinit = true;
    allowZero = true;
end

L_new = L(:,i);
L_new(i) = 0;
P = S*L_new;
M_new = M(:,i);
M_new(i) = 0;
Q = S*M_new;
Ld_new=Ld(:,i);
Ld_new(i)=0;
C = S*Ld_new;

fobj = 0;
% fobj = fobj_featuresign(s, B, x, BtB, Btx, P, L, Q, M, i, alpha, mu, gamma);

ITERMAX=1000;
optimality1=false;
for iter=1:ITERMAX
    % check optimality0
    act_indx0 = find(act == 0);
    
    grad = BtB*sparse(s)+(alpha*L(i,i)+mu*M(i,i)+gama*Ld(i,i))*sparse(s)+alpha*P+mu*Q+gama*C - Btx;
    theta = sign(s);

    optimality0= false;
    % Step 2
    [mx,indx] = max (abs(grad(act_indx0)));

    if ~isempty(mx) && (mx >= gamma) && (iter>1 || ~usesinit)
        act(act_indx0(indx)) = 1;
        theta(act_indx0(indx)) = -sign(grad(act_indx0(indx)));
        usesinit= false;
    else
        optimality0= true;
        if optimality1
            break;
        end
    end
    act_indx1 = find(act == 1);

    if length(act_indx1)>rankB
        %warning('sparsity penalty is too small: too many coefficients are activated');
        return;
    end

    if isempty(act_indx1) %length(act_indx1)==0
        % if ~assert(max(abs(s))==0), save(fname_debug, 'B', 'x', 'gamma', 'sinit'); error('error'); end
        if allowZero, allowZero= false; continue, end
        return;
    end

    % if ~assert(length(act_indx1) == length(find(act==1))), save(fname_debug, 'B', 'x', 'gamma', 'sinit'); error('error'); end
    kk=0;
    while 1
        kk=kk+1;

        if kk>ITERMAX
            warning('Maximum number of iteration reached. The solution may not be optimal');
            % save(fname_debug, 'B', 'x', 'gamma', 'sinit');
            return;
        end

        if isempty(act_indx1) % length(act_indx1)==0
            % if ~assert(max(abs(s))==0), save(fname_debug, 'B', 'x', 'gamma', 'sinit'); error('error'); end
            if allowZero, allowZero= false; break, end
            return;
        end

        % Step 3: feature-sign step
        [s, theta, act, act_indx1, optimality1, lsearch, fobj] = compute_FS_step (s, B, x, BtB, Btx, P, L, Q, M,C,Ld, i, theta, act, act_indx1, alpha, mu,gama, gamma);

        % Step 4: check optimality condition 1
        if optimality1 break; end;
        if lsearch >0 continue; end;

    end
end

if iter >= ITERMAX
    warning('Maximum number of iteration reached. The solution may not be optimal');
    % save(fname_debug, 'B', 'x', 'gamma', 'sinit');
end

if 0  % check if optimality
    act_indx1 = find(act==1);
    grad =  BtB*sparse(s)+(alpha*L(i,i)+mu*M(i,i)+gama*Ld(i,i))*sparse(s)+alpha*P+mu*Q+gama*C - Btx;
    norm(grad(act_indx1) + gamma.*sign(s(act_indx1)),'inf')
    find(abs(grad(setdiff(1:m, act_indx1)))>gamma)
end

fobj = 0;
% fobj = fobj_featuresign(s, B, x, BtB, Btx, P, L, Q, M, i, alpha, mu, gamma);

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [s, theta, act, act_indx1, optimality1, lsearch, fobj] = compute_FS_step (s, B, x, BtB, Btx, P, L, Q, M,C,Ld, idx, theta, act, act_indx1, alpha, mu,gama, gamma)

s2 = s(act_indx1);
% B2 = B(:, act_indx1);
BtB2 = BtB(act_indx1, act_indx1);
theta2 = theta(act_indx1);


a = (alpha*L(idx,idx)+mu*M(idx,idx)+gama*Ld(idx,idx))*ones(size(act_indx1));
%a = L(idx,idx)*alpha*eye(size(act_indx1,1));
% call matlab optimization solver..
s_new = (BtB2+diag(a)) \ ( Btx(act_indx1) - gamma.*theta2 - alpha*P(act_indx1) - mu*Q(act_indx1)-gama*C(act_indx1)); % RR
% opts.POSDEF=true; opts.SYM=true; % RR
% s_new = linsolve(BtB2, ( Btx(act_indx1) - gamma.*theta2 ), opts); % RR
optimality1= false;
if (sign(s_new) == sign(s2)) 
    optimality1= true;
    s(act_indx1) = s_new;
    fobj = 0; 
    %fobj = fobj_featuresign(s, B, x, BtB, Btx, P, L, Q, M, i, alpha, mu, gamma);
    lsearch = 1;
    return; 
end

% do line search: s -> s_new
progress = (0 - s2)./(s_new - s2);
lsearch=0;

a= 0.5*sum((B(:,act_indx1)*(s_new-s2)).^2)+0.5*(alpha*L(idx,idx)+mu*M(idx,idx)+gama*Ld(idx,idx))*(s_new-s2)'*(s_new-s2);
b= (s_new-s2)'*(alpha*P(act_indx1)+mu*Q(act_indx1)+gama*C(act_indx1)-Btx(act_indx1)) + (s2'*BtB2+ (alpha*L(idx,idx)+mu*M(idx,idx)+gama*Ld(idx,idx))*s2')*(s_new-s2);

fobj_lsearch = gamma*sum(abs(s2));
[sort_lsearch, ix_lsearch] = sort([progress',1]);
remove_idx=[];
for i = 1:length(sort_lsearch)
    t = sort_lsearch(i); if t<=0 || t>1 continue; end
    s_temp= s2+ (s_new- s2).*t;
    fobj_temp = a*t^2 + b*t + gamma*sum(abs(s_temp));
    if fobj_temp < fobj_lsearch
        fobj_lsearch = fobj_temp;
        lsearch = t;
        if t<1  remove_idx = [remove_idx ix_lsearch(i)]; end % remove_idx can be more than two..
    elseif fobj_temp > fobj_lsearch
        break;
    else
        if (sum(s2==0)) == 0
            lsearch = t;
            fobj_lsearch = fobj_temp;
            if t<1  remove_idx = [remove_idx ix_lsearch(i)]; end % remove_idx can be more than two..
        end
    end
end

% if ~assert(lsearch >=0 && lsearch <=1), save(fname_debug, 'B', 'x', 'gamma', 'sinit'); error('error'); end

if lsearch >0
    % update s
    s_new = s2 + (s_new - s2).*lsearch;
    s(act_indx1) = s_new;
    theta(act_indx1) = sign(s_new);  % this is not clear...
end

% if s encounters zero along the line search, then remove it from
% active set
if lsearch<1 && lsearch>0
    %remove_idx = find(s(act_indx1)==0);
    remove_idx = find(abs(s(act_indx1)) < eps);
    s(act_indx1(remove_idx))=0;

    theta(act_indx1(remove_idx))=0;
    act(act_indx1(remove_idx))=0;
    act_indx1(remove_idx)=[];
end
fobj_new = 0; 
% fobj_new = fobj_featuresign(s, B, x, BtB, Btx, P, L, Q, M, i, alpha, mu, gamma);

fobj = fobj_new;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [f, g] = fobj_featuresign(s, B, x, BtB, Btx, P, L, Q, M,C,Ld, i, alpha, mu,gama, gamma)

f= 0.5*(norm(x-B*s)^2+alpha*L(i,i).*(s'*(s+2*P))+mu*M(i,i).*(s'*(s+2*Q))+gama*Ld(i,i).*(s'*(s+2*C)));
f= f+ gamma*norm(s,1);

if nargout >1
    g = BtB*s+alpha*L(i,i)*s+alpha*P+mu*M(i,i)*s+mu*Q+gama*Ld(i,i)*s+gama*C - Btx;
    g= g+ gamma*sign(s);
end

end

%%%%%%%%%%%%%%%%%%%%%