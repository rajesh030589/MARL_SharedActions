function  [F_0, F_1] = Next_belief_state(pi, p_01, p_11, p, q)

pi_0 = 1 - pi;
pi_1 = pi;

p_00 = 1 - p_01;
p_10 = 1 - p_11;

t_10 = p;
t_11 = 1-q;

% check for discontinuitites
if pi_0*p_00 + pi_1*p_10 == 0
    F_0 = pi_0*t_10 + pi_1*t_11;
else
    F_0 = (pi_0*p_00*t_10 + pi_1*p_10*t_11)/(pi_0*p_00 + pi_1*p_10);
end

if pi_0*p_01 + pi_1*p_11 == 0
    F_1 = pi_0*t_10 + pi_1*t_11;
else
    F_1 = (pi_0*p_01*t_10 + pi_1*p_11*t_11)/(pi_0*p_01 + pi_1*p_11);
end
end