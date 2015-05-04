function [gamma, lambda] = stc_qp(H, q, M, D, x, C)
%% Sparse Tangent Coding with Quadratic Prior 
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
%   lambda : mx1 dual lagrangian
%       
    n = size(H,1);
    m = size(D,2);
    
    assert(n == size(H,2) & n == size(M,1) & n == size(M,2) & n == length(q) & n == length(x) & n == size(D,1));
    
    H = sparse(H);
    M = sparse(M);
    q = sparse(q);
    x = sparse(x);

    % prepare mosek LP
    addpath('/gpfs/work/j/jxy198/software/mosek/7/toolbox/r2013a/');
    %addpath('/Users/bobye/mosek/7/toolbox/r2012a');
    
    prob.c = [1; C * ones(m,1); zeros(m,1)];
    prob.subi = reshape([1:2*m; 1:2*m], 4*m,1);
    prob.subj = reshape(repmat([2:m+1; (2+m):(2*m+1)], 1, 2), 4*m, 1); 
    prob.valij = [ones(2*m,1); reshape([ones(1,m); -ones(1,m)], 2*m,1)];
    prob.a = sparse(prob.subi, prob.subj, prob.valij);
    prob.blc = zeros(2*m, 1);
    prob.buc = [];
    prob.blx = [zeros(1+m,1); -inf(m,1)];
    prob.bux = [];    
    param.MSK_IPAR_OPTIMIZER = 'MSK_OPTIMIZER_PRIMAL_SIMPLEX';     
    
    % pre-computation
    H = [H, -q; -q', - x' * H * x];
    M = [M, M*x; x'*M', x'* M * x];
    DD = cell(m,1);
    for i=1:m
        DD{i} = [sparse(n, n), D(:,i); D(:,i)', 2 * x' * D(:,i)];
    end

    % initialization
    gamma = 0;
    lambda = zeros(m,1);

    % main iterations (cutting plane methods)
    maxIters = 100;
    fprintf('prim res\tobj\t\tgamma\tnz\n');
    for iter = 1:maxIters
        CC = H + gamma * M;
        for i=1:m            
            CC = CC + lambda(i) * DD{i};
        end
        [v, d] = eigs(CC, 1, 'SR'); % R2013a       
        
        hh = v' * H * v;
        mm = v' * M * v;        
        dd = arrayfun(@(ind) v'*DD{ind}*v, 1:m);
        
        % add one more constraint: hh + mm * gamma + dd' * lambda >= 0
        prob.a = [prob.a; mm, sparse(1,m), dd];
        prob.blc = [prob.blc; -hh];
        
        % warm start
        if (iter > 1) 
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
            lambda = sol((end-m+1): end);              
        catch
            fprintf('MSKERROR: Could not get solution');
        end
        
        
        fprintf('%f\t%f\t%f\t%d\n', d, prob.c' * sol, gamma, sum(lambda>1E-6));
        if (d >= -1E-5) 
            break;
        end
        
    end
end

