function [p_1_01, p_1_11, p_2_01, p_2_11, Vpi_1, Vpi_2] = solve_fp_follower(P, idx_1, idx_2, R, p_1_01, p_1_11, p_2_01, p_2_11, V_1, V_2, p, q, d)

err = 1;
count = 0;

pi_1 = P(idx_1);
pi_2 = P(idx_2);

while err > 1e-3 && count <=20
    count = count+1;
    
    [p_1_01_nxt, p_1_11_nxt] = getPolicy(P, pi_1, pi_2, R, V_1, p_1_01, p_1_11,  p_2_01, p_2_11, p, q, d);
    [p_2_01_nxt, p_2_11_nxt] = getPolicy(P, pi_2, pi_1, R, V_2, p_2_01, p_2_11,p_1_01, p_1_11, p, q, d);
    err = norm([p_1_01 p_1_11 p_2_01 p_1_11] - [p_1_01_nxt p_1_11_nxt p_2_01_nxt p_2_11_nxt]);
    
    p_1_01 = p_1_01_nxt;
    p_1_11 = p_1_11_nxt;
    p_2_01 = p_2_01_nxt;
    p_2_11 = p_2_11_nxt;
    
end
p_1_00 = 1 - p_1_01;
p_1_10 = 1 - p_1_11;
p_2_00 = 1 - p_2_01;
p_2_10 = 1 - p_2_11;

Q_1 = getValue(P, pi_1, pi_2, R, p_1_01, p_1_11, p_2_01, p_2_11, V_1, p, q, d);
Vpi_1(1) = p_1_00*Q_1(1,1) + p_1_01*Q_1(1,2); 
Vpi_1(2) = p_1_10*Q_1(2,1) + p_1_11*Q_1(2,2); 

Q_2 = getValue(P, pi_2, pi_1, R, p_2_01, p_2_11, p_1_01, p_1_11, V_2, p, q, d);
Vpi_2(1) = p_2_00*Q_2(1,1) + p_2_01*Q_2(1,2); 
Vpi_2(2) = p_2_10*Q_2(2,1) + p_2_11*Q_2(2,2); 
end

function [p_m_01_nxt, p_m_11_nxt] = getPolicy(P, pi_m, pi_n, R, V_m, p_m_01, p_m_11, p_n_01, p_n_11, p, q, d)
    
    Q = getValue(P, pi_m, pi_n, R, p_m_01, p_m_11, p_n_01, p_n_11, V_m, p, q, d);
    
    % Update Policy
    p_m_01_nxt = update_policy(Q(1, :), p_m_01);
    p_m_11_nxt = update_policy(Q(2, :), p_m_11);
end

function Q = getValue(P, pi_m, pi_n, R, p_m_01, p_m_11, p_n_01, p_n_11, V_m, p, q, d)
    % User 1
    [F_m_0, F_m_1] = Next_belief_state(pi_m, p_m_01, p_m_11, p, q);
    
    % User 2
    [F_n_0, F_n_1] = Next_belief_state(pi_n, p_n_01, p_n_11, p, q);
    
    V1_m(1, :, :) = Next_Value_function(P, F_m_0, F_m_1, F_n_0, F_n_1, squeeze(V_m(1,:,:))); 
    V1_m(2, :, :) = Next_Value_function(P, F_m_0, F_m_1, F_n_0, F_n_1, squeeze(V_m(2,:,:))); 
    
    Q = value_function_follower(pi_n, p_n_01, p_n_11, R, V1_m, p, q, d);
end