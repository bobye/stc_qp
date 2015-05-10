function [gamma, lambda, y] = stc_qp(H, q, M, D, x, C)
%% Sparse Tangent Coding with Quadratic Prior 
% copyright 2015 - Jianbo Ye
%
% Input
%   H : nxn positive semi-definite matrix
%   M : nxn positive semi-definite matrix    
%   q : nx1 vector
%   x : nx1 vector
%   D : nxm dictionary matrix
%
% Output
%   gamma : scalar cost
%   lambda : mx1 sparse dual lagrangian
%       
    n = size(H,1);
    m = size(D,2);
    assert(n == size(H,2) & n == size(M,1) & n == size(M,2) & n == length(q) & n == length(x) & n == size(D,1));
    
    H = sparse(H); 
    M = sparse(M);    
    xx = x / sqrt(x'*H*x);
    q = sparse(q); 

    % prepare mosek LP
    addpath('/gpfs/work/j/jxy198/software/mosek/7/toolbox/r2013a/');
    %addpath('/Users/bobye/mosek/7/toolbox/r2012a');
    prob.c = [1; zeros(2*m,1)];
    prob.subi = reshape([1:2*m; 1:2*m], 4*m,1);
    prob.subj = reshape(repmat([2:m+1; (2+m):(2*m+1)], 1, 2), 4*m, 1); 
    prob.valij = [ones(2*m,1); reshape([ones(1,m); -ones(1,m)], 2*m,1)];
    prob.a = sparse(prob.subi, prob.subj, prob.valij);
    prob.a = [prob.a; 0, -ones(1,m), sparse(1,m)];
    prob.blc = [zeros(2*m, 1); -C];
    prob.buc = [];
    prob.blx = [-inf; zeros(m,1); -inf(m,1);];
    prob.bux = [];    
    param.MSK_IPAR_OPTIMIZER = 'MSK_OPTIMIZER_PRIMAL_SIMPLEX';     
    
    % pre-computation
    HH = [H, -q; -q', - xx' * H * xx];
    MM = [M, M*xx; xx'*M', xx'* M * xx];
    DD = cell(m,1);
    for i=1:m
        DD{i} = [sparse(n, n), D(:,i); D(:,i)', 2 * xx' * D(:,i)];
    end

    % initialization
    gamma = 0;
    lambda = zeros(m,1);

    % main iterations (cutting plane methods)
    maxIters = 500;
    fprintf('iter\tres\t\tobj\t\tgamma\t\tnz\tseconds\n');
    eig_opt.tol = 1E-8; % set large error tolerance for leading eigenvector
    for iter = 1:maxIters
        tic;
        CC = HH + gamma * MM;
        for i=1:m            
            CC = CC + lambda(i) * DD{i};
        end
        CC = (CC+CC')/2.;
        [v, d] = eigs(CC, 1, 'SA', eig_opt); % R2013a  
        
        if (d >= -2E-3) % SDP feasibility tolerance
            break;
        end
        eig_opt.v0 = v; % warm-start eigen
        
        hh = v'*HH*v;
        mm = v'*MM*v;
        dd = arrayfun(@(ind) v'*DD{ind}*v, 1:m);
        
        % add one more constraint: hh + mm * gamma + dd' * lambda >= 0
        prob.a = [prob.a; mm, sparse(1,m), dd];
        prob.blc = [prob.blc; -hh];
        
        % warm-start LP
        if (false) 
            bas = res.sol.bas;
            bas.skc = [bas.skc; 'BS'];
            bas.xc = [bas.xc; mm*gamma + dd * lambda];
            bas.y = [bas.y; 0.0];
            bas.slc = [bas.slc; 0.0];
            bas.suc = [bas.suc; 0.0];
            prob.sol.bas = bas;
        end
        
        [rcode, res] = mosekopt('minimize echo(0)', prob, param);
        try 
            sol = res.sol.bas.xx;
            gamma = sol(1);
            lambda = sol((m+2):(2*m+1));
        catch
            fprintf('MSKERROR: Could not get solution');
        end
        
        elapsedTime = toc;
        obj = prob.c' * sol;
        nz = sum(abs(lambda)>1E-6);
        fprintf('%d\t%.6f\t%.3e\t%.3e\t%d\t%.1f\n', iter, min(d), obj, gamma, nz, elapsedTime);

        
    end
    
    % compute recovery solution
    Ds = D(:,abs(lambda)>1E-6);
    [res] = mskqpopt( H, zeros(n,1), Ds', [Ds'*x], [Ds'*x], [], []);
    y = res.sol.itr.xx;
end

