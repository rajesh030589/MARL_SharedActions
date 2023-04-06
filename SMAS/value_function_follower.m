function qf_m = value_function_follower(pi_n, p_01, p_11, R_m, V_m, p, q, d)

qf_m = zeros(2, 2);

pi_n_1 = pi_n;
pi_n_0 = 1 - pi_n;
p_n_00 = 1 - p_01;
p_n_01 = p_01;
p_n_10 = 1 - p_11;
p_n_11 = p_11;

% Averaged across belief state of user n
Q = Qvalue_function_follower(p, q, R_m, V_m, d);
Q = Qlearning_function_follower(p, q, R_m, V_m, d);

for s_m = [0 1]
    for af_m = [0 1]
        qf_m(s_m+1, af_m+1) = pi_n_0*p_n_00*Q(s_m+1, af_m+1, 1, 1) +...
            pi_n_0*p_n_01*Q(s_m+1, af_m+1, 1, 2) +...
            pi_n_1*p_n_10*Q(s_m+1, af_m+1, 2, 1) +...
            pi_n_1*p_n_11*Q(s_m+1, af_m+1, 2, 2);
    end
end



end