function  [F1_0, F1_1] = Next_belief_state_follower(pi1, p1_01, p1_11, q)

at = 0;
n2 = 0;
N = 0;
while N < 5000
    
    s = binornd(1, pi1);
    a = (~s)*binornd(1,p1_01) +  (s)*binornd(1,p1_11);
    if (a == at)
        w =  binornd(1, q);
        sn = (s*(~w) + (~s)*w);
        n2 = n2 + sn;
    end
    N= N+1;
    
end
F1_0 = n2/N;
F1_0 = F1_0/(pi1*(1-p1_11) + (1-pi1)*(1-p1_01));

if (pi1*(1-p1_11) + (1-pi1)*(1-p1_01)) == 0
    F1_0 = pi1;
end
at = 1;
n2 = 0;
N = 0;
while N < 5000
    
    s = binornd(1, pi1);
    a = (~s)*binornd(1,p1_01) +  (s)*binornd(1,p1_11);
    if (a == at)
        w =  binornd(1, q);
        sn = (s*(~w) + (~s)*w);
        n2 = n2 + sn;
    end
    N= N+1;
    
    
    
end
F1_1 = n2/N;
F1_1 = F1_1/(pi1*(p1_11) + (1-pi1)*(p1_01));
if pi1*(p1_11) + (1-pi1)*(p1_01) == 0
    F1_1 = pi1;
end
    
end