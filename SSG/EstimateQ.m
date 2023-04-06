
function [Qf1, Qf2] = EstimateQ(P, Pf, Qf1, Qf2, q, Pf1_01, Pf1_11, Pf2_01, Pf2_11, t,  Rf1_0, Rf1_1, Rf2_0, Rf2_1, Delta)
%{
Qf_bs, Ql_bs   : The Q value estimate for each belief state
%}

for idx1 = 1:length(P)
    for idx2 = 1:length(P)
        
        pi1 = P(idx1);
        pi2 = P(idx2);
        
        Mf = length(Pf);
        
        Qf1_bs = zeros(2, 2, 2, Mf, Mf, Mf, Mf);
        Qf2_bs = zeros(2, 2, 2, Mf, Mf, Mf, Mf);
        
        
        for i = 1: Mf
            for j = 1:Mf
                for k = 1: Mf
                    for l = 1:Mf
                        
                        
                        % Equilibrium Policy
                        pf1_01 = Pf(i);
                        pf1_11 = Pf(j);
                        pf2_01 = Pf(k);
                        pf2_11 = Pf(l);
                        
                        % Next Belief state if we follow the equi policy
                        [F_f1(1), F_f1(2)] = Next_belief_state(pi1, pf1_01, pf1_11, q);
                        [F_f2(1), F_f2(2)] = Next_belief_state(pi2, pf2_01, pf2_11, q);
                        % Equilibrium Policy if at the new belief state
                        pf_f1 = zeros(2,2);
                        pf_f2 = zeros(2,2);
                        
                        for p = 1:2
                            pf_f1(p, 1) = interp2(P, P, Pf1_01, F_f1(p), F_f2(p));
                            pf_f1(p, 2) = interp2(P, P, Pf1_11, F_f1(p), F_f2(p));
                            pf_f2(p, 1) = interp2(P, P, Pf2_01, F_f1(p), F_f2(p));
                            pf_f2(p, 2) = interp2(P, P, Pf2_11, F_f1(p), F_f2(p));
                            
                        end
                        
                        
                        vf1 = value_function_follower(P,Pf,Qf1, F_f1, F_f2, pf_f1, pf_f2);
                        vf2 = value_function_follower(P,Pf,Qf2, F_f2, F_f1, pf_f2, pf_f1);
                        
                        
                        Qf1m = Qf1(:, :, :, idx1, idx2, i, j, k, l);
                        Qf2m = Qf2(:, :, :, idx1, idx2, i, j, k, l);
                        
                        B = 20;
                        
                        
                        % Follower 1
                        for b = 1:B
                            alpha = 1/((t-1)*B + b);
                            for af1 = [0 1]
                                for af2 = [0 1]
                                    for s = [0 1]
                                        % Reward for the Follower1
                                        rf = (~s)*Rf1_0(af1 +1, af2 + 1) + (s)*Rf1_1(af1 +1, af2 + 1);
                                        
                                        % Next State
                                        w =  binornd(1, q);
                                        sn = s*(~w) + (~s)*w;
                                        
                                        Qf1m(s+1, af1+1, af2+1) = Qf1m(s+1, af1+1, af2+1) + alpha*(rf + Delta*vf1(sn+1) - Qf1m(s+1, af1+1, af2+1) );
                                        
                                    end
                                end
                            end
                            
                            for af1 = [0 1]
                                for af2 = [0 1]
                                    for s = [0 1]
                                        
                                        % Reward for the Follower1
                                        rf = (~s)*Rf2_0(af1 +1, af2 + 1) + (s)*Rf2_1(af1 +1, af2 + 1);
                                        
                                        % Next State
                                        w =  binornd(1, q);
                                        sn = s*(~w) + (~s)*w;
                                        
                                        Qf2m(s+1, af1+1, af2+1) = Qf2m(s+1, af1+1, af2+1) + alpha*(rf + Delta*vf2(sn+1) - Qf2m(s+1, af1+1, af2+1) );
                                        
                                        
                                    end
                                end
                            end
                        end
                        
                        Qf1_bs(:, :, :, i, j, k, l) = Qf1m;
                        Qf2_bs(:, :, :, i, j, k, l) = Qf2m;
                    end
                end
            end
        end
        Qf1(:, :, :, idx1, idx2, :, :, :, :) = Qf1_bs;
        Qf2(:, :, :, idx1, idx2, :, :, :, :) = Qf2_bs;
        
    end
end