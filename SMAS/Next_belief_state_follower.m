function  [F1_0, F1_1] = Next_belief_state_follower(pi1, p1_01, p1_11, p, q)
% check for discontinuitites
if (pi1*(1-p1_11)*(1-q) + (1-pi1)*(1-p1_01)*p) == 0
    F1_0 = pi1;
else
    F1_0 = pi1*(1-p1-11)*(1-q)/(pi1*(1-p1_11)*(1-q) + (1-pi1)*(1-p1_01)*p);
end

if (pi1*p1_11*(1-q) + (1-pi1)*p1_01*p) == 0
    F1_1 = pi1;
else
    F1_1 = pi1*p1_11*(1-q)/(pi1*p1_11*(1-q) + (1-pi1)*p1_01*p);
end
end