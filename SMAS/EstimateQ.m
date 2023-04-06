
function Q = EstimateQ(p, q, R_m, V_m, d)

Q = zeros(2, 2, 2, 2);

for s_m = [0 1]
    for af_m = [0 1]
        for s_n = [0 1]
            for af_n = [0 1]
                for b = 1:B
                    w1 = binornd(1,p);
                    w2 = binornd(1,q);
                    
                    Q(s_m+1, af_m +1, s_n+1, af_n+1) = Q(s_m+1, af_m +1, s_n+1, af_n+1) +...
                        R_m(s_m+1, s_n + 1, af_m+1, af_n+1) +...
                        d*(1-s_m)*(1-w1)*V_m(1, af_m+1 + af_n+1) +...
                        d*(1-s_m)*w1*V_m(2, af_m+1 + af_n+1)+...
                        d*s_m*w2*V_m(1, af_m+1 + af_n+1) +...
                        d*s_m*(1-w2)*V_m(2, af_m+1 + af_n+1);
                end
            end
        end
    end
end
Q = Q./B;
    
end