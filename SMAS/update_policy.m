function p_new = update_policy(Q, p_old)
q0 = Q(1);
q1 = Q(2);
qs = (1-p_old) * q0 + p_old * q1;

d = 0.8;

phi0 = q0 - qs;
phi1 = q1 - qs;

p0 = d*max(0, phi0);
p1 = d*max(0, phi1);

p_new = (p_old + p1)/(1+p0 + p1);
end