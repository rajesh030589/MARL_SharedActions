function p_new = update_policy(p_old, q0, q1)

qs = (1-p_old) * q0 + p_old * q1;

phi0 = abs(q0 - qs);
phi1 = abs(q1 -  qs);
if q1 >=qs
    p_new = p_old + phi1/(1 + phi0 + phi1);
else
    p_new = p_old - phi0/(1 + phi0 + phi1);
end
end