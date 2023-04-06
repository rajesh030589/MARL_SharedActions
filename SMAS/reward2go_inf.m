function [ V00, V01, V10, V11 ] = reward2go_inf( pi1v,pi2v,F1_0,F1_1,F2_0,F2_1,u)

f = @(x,y) interp2(pi1v,pi2v,u,x,y) ;

V00 = f(F1_0,F2_0);
V01 = f(F1_0,F2_1);
V10 = f(F1_1,F2_0);
V11 = f(F1_1,F2_1);

end

