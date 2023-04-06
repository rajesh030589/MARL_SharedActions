function vf = value_function_follower(P,Pf,Qf,F1, F2, pf1, pf2)

qf = zeros(2, 2, 2);
vf1 = zeros(2, 2, 2);
vf = zeros(2, 2);

for s = [0 1]
    for af1 = [0 1]
        for af2 = [0 1]
            qf(s+1, af1+1, af2+1) = interpn(P, P, Pf, Pf, Pf, Pf, squeeze(Qf(s+1, af1+1, af2+1, :, :, :, :, :, :)), F1(1), F2(1), pf1(1, 2),  pf1(2, 2), pf1(1,2), pf2(2, 2));
        end
    end
end


for s = [0 1]
    for p = [0 1]
        vf(s+1, p+1) = (1 - pf1(1, s+1))*vf1(s+1, 1, 1) +  pf1(1, s+1)*vf1(s+1, 1, 2) ;
    end
end
end