function [pfm_01, pfm_11] = solve_fp_follower(P, idx_m, idx_n, Pf, pfm_01, pfm_11, pfn_01, pfn_11, Qfm)

err = 1;
count = 0;

pi_m = P(idx_m);
pi_n = P(idx_n);
while err > 1e-4 && count <=1e3
    count = count+1;
    
    qf = zeros(2, 2, 2);
    qf1 = zeros(2, 2);
    
    for s = [0 1]
        for af_m = [0 1]
            for af_n = [0 1]
                qf(s+1, af_m+1, af_n+1) = interpn(P, P, Pf, Pf, Pf, Pf, squeeze(Qfm(s+1, af_m+1, af_n+1, :, :, :, :, :, :)), pi_m, pi_n, pfm_01, pfm_11, pfn_01, pfn_11);
            end
        end
    end
    
    % Averaging over the other user
    for s = [0 1]
        for af_m = [0 1]
            qf1(s+1, af_m+1) = ((1 - pi_n)*(1 - pfn_01) + pi_n*(1 - pfn_11))*qf(s+1, af_m+1, 1) + ((1 - pi_n)*pfn_01 + pi_n*pfn_11)*qf(s+1, af_m+1, 2); 
        end
    end
    
    % Update Policy
    pfm_01_n = update_policy(pfm_01, qf1(1,1), qf1(1,2));
    pfm_11_n = update_policy(pfm_11, qf1(2,1), qf1(2,2));
    err = norm([pfm_01 pfm_11] - [pfm_01_n pfm_11_n]);
    
    pfm_01 = pfm_01_n;
    pfm_11 = pfm_11_n;
    
end
end