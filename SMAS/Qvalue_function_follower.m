function Q = Qvalue_function_follower(p, q, R_m, V_m, d)


t_10 = p;
t_00 = 1-p;
t_11 = 1-q;
t_01 = q;
% Averaged across belief state of user n

Q = zeros(2,2,2,2);

for s_m = [0 1]
    for af_m = [0 1]
        for s_n = [0 1]
            for af_n = [0 1]
                Q(s_m+1, af_m +1, s_n+1, af_n+1) = R_m(s_m+1, s_n + 1, af_m+1, af_n+1) +...
                    d*(s_m*t_01 +(1-s_m)*t_00)*V_m(1, af_m+1 + af_n+1) +...
                    d*(s_m*t_11 +(1-s_m)*t_10)*V_m(2, af_m+1 + af_n+1);
            end
        end
    end
end
end