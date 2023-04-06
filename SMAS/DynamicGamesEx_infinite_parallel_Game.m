
% clc
% clear
xL = 0.2;
xH = 1.2;

% Suppose state evolves from xL to xH with probability p and xH to xL wp q

p=0.1;
q=0.2;
T=30; % horizon

delD = 0.9; % discount factor for infinite horizon reward.
delI = 0.8; % discount factor for equilibrium update. It dampens the step size

N=10; %resolution for pi space
pi1v=(0:N)*(1/N);
pi2v=(0:N)*(1/N);

R1LL = [0 1; 1-xL 1-xL];
R1LH = [0 1; 1-xL 1-xL];
R1HL = [0 1; 1-xH 1-xH];
R1HH = [0 1; 1-xH 1-xH];


% R2LL = R1LL;
% R2LH = R1LH;
% R2HL = R1HL;
% R2HH = R1HH;

R2LL = [0 1-xL; 1 1-xL];
R2LH = [0 1-xH; 1 1-xH];
R2HL = [0 1-xL; 1 1-xL];
R2HH = [0 1-xH; 1 1-xH];

results=[];
p1Lm=0.5*ones(N+1,N+1); % they are symmetric
p1Hm=0.5*ones(N+1,N+1);
p2Lm=0.5*ones(N+1,N+1);
p2Hm=0.5*ones(N+1,N+1);
u1Lm=zeros(N+1,N+1);
u1Hm=zeros(N+1,N+1);
u2Lm=zeros(N+1,N+1);
u2Hm=zeros(N+1,N+1);

for t=1:T
    t % DISPLAY
    p1Lm_n=zeros(N+1,N+1);
    p1Hm_n=zeros(N+1,N+1);
    p2Lm_n=zeros(N+1,N+1);
    p2Hm_n=zeros(N+1,N+1);
    u1Lm_n=zeros(N+1,N+1);
    u1Hm_n=zeros(N+1,N+1);
    u2Lm_n=zeros(N+1,N+1);
    u2Hm_n=zeros(N+1,N+1);
    
    for i1=1:length(pi1v)
        
        p1Lv_n=zeros(1,N+1);
        p1Hv_n=zeros(1,N+1);
        p2Lv_n=zeros(1,N+1);
        p2Hv_n=zeros(1,N+1);
        u1Lv_n=zeros(1,N+1);
        u1Hv_n=zeros(1,N+1);
        u2Lv_n=zeros(1,N+1);
        u2Hv_n=zeros(1,N+1);
        for i2=1:length(pi2v)
            pi1=pi1v(i1);
            pi2=pi2v(i2);
            [pi1 pi2]; % DISPLAY
            
            % Initial equilibrium guess
            p1Li = p1Lm(i1,i2);
            p1Hi = p1Hm(i1,i2);
            p2Li = p2Lm(i1,i2);
            p2Hi = p2Hm(i1,i2);
            err = 1;
            count = 0;
            p1L=p1Li;
            p2L=p2Li;
            p1H=p1Hi;
            p2H=p2Hi;
            while err > 1e-3 && count <=20
                %while count <=0
                count = count+1;
                
                
                [F1_0, F1_1] = Next_belief_state(pi1, p1L, p1H, p, q);
                [F2_0, F2_1] = Next_belief_state(pi2, p2L, p2H, p, q);
                
                [V1L00, V1L01, V1L10, V1L11]=reward2go_inf(pi1v,pi2v,F1_0,F1_1,F2_0,F2_1,u1Lm);
                [V1H00, V1H01, V1H10, V1H11]=reward2go_inf(pi1v,pi2v,F1_0,F1_1,F2_0,F2_1,u1Hm);
                [V2L00, V2L01, V2L10, V2L11]=reward2go_inf(pi1v,pi2v,F1_0,F1_1,F2_0,F2_1,u2Lm);
                [V2H00, V2H01, V2H10, V2H11]=reward2go_inf(pi1v,pi2v,F1_0,F1_1,F2_0,F2_1,u2Hm);
                
                
                % User 1 state L
                u1L_0 = (1-pi2)*p2L*R1LL(1,2) + pi2*p2H*R1LH(1,2) + (1-pi2)*(1-p2L)*R1LL(1,1) + pi2*(1-p2H)*R1LH(1,1) + delD*(1-p)*[ (1-(1-pi2)*p2L - pi2*p2H)* V1L00 + ((1-pi2)*p2L + pi2*p2H)* V1L01 ]+ delD*p*[ (1-(1-pi2)*p2L - pi2*p2H)* V1H00 + ((1-pi2)*p2L + pi2*p2H)* V1H01 ];
                u1L_1 = (1-pi2)*p2L*R1LL(2,2) + pi2*p2H*R1LH(2,2) + (1-pi2)*(1-p2L)*R1LL(2,1) + pi2*(1-p2H)*R1LH(2,1) + delD*(1-p)*[ (1-(1-pi2)*p2L - pi2*p2H)* V1L10 + ((1-pi2)*p2L + pi2*p2H)* V1L11 ]+ delD*p*[ (1-(1-pi2)*p2L - pi2*p2H)* V1H10 + ((1-pi2)*p2L + pi2*p2H)* V1H11 ] ;
                u1L_s = (1-p1L) * u1L_0 + p1L * u1L_1;
                phi1L_0 = delI*max( 0, u1L_0 - u1L_s);
                phi1L_1 = delI*max( 0, u1L_1 - u1L_s);
                p1L_n = (p1L + phi1L_1)/ (1 + phi1L_0 + phi1L_1 );
                
                % User 2 state L
                u2L_0 = (1-pi1)*p1L*R2LL(2,1) + pi1*p1H*R2HL(2,1) + (1-pi1)*(1-p1L)*R2LL(1,1) + pi1*(1-p1H)*R2HL(1,1) + delD*(1-p)*[ (1-(1-pi1)*p1L - pi1*p1H)* V2L00 + ((1-pi1)*p1L + pi1*p1H)* V2L10 ]+ delD*p*[ (1-(1-pi1)*p1L - pi1*p1H)* V2H00 + ((1-pi1)*p1L + pi1*p1H)* V2H10 ] ;
                u2L_1 = (1-pi1)*p1L*R2LL(2,2) + pi1*p1H*R2HL(2,2) + (1-pi1)*(1-p1L)*R2LL(1,2) + pi1*(1-p1H)*R2HL(1,2) + delD*(1-p)*[ (1-(1-pi1)*p1L - pi1*p1H)* V2L01 + ((1-pi1)*p1L + pi1*p1H)* V2L11 ] + delD*p*[ (1-(1-pi1)*p1L - pi1*p1H)* V2H01 + ((1-pi1)*p1L + pi1*p1H)* V2H11 ] ;
                u2L_s = (1-p2L) * u2L_0 + p2L * u2L_1;
                phi2L_0 = delI*max( 0, u2L_0 - u2L_s);
                phi2L_1 = delI*max( 0, u2L_1 - u2L_s);
                p2L_n = (p2L + phi2L_1)/ (1 + phi2L_0 + phi2L_1 );
                %p2L_n = p1L_n; % force symmetric eq
                
                % User 1 , state H
                u1H_0 = (1-pi2)*p2L*R1HL(1,2) + pi2*p2H*R1HH(1,2) + (1-pi2)*(1-p2L)*R1HL(1,1) + pi2*(1-p2H)*R1HH(1,1) + delD*(1-q)*[ (1-(1-pi2)*p2L - pi2*p2H)* V1H00 + ((1-pi2)*p2L + pi2*p2H)* V1H01 ]+ delD*q*[ (1-(1-pi2)*p2L - pi2*p2H)* V1L00 + ((1-pi2)*p2L + pi2*p2H)* V1L01 ] ;
                u1H_1 = (1-pi2)*p2L*R1HL(2,2) + pi2*p2H*R1HH(2,2) + (1-pi2)*(1-p2L)*R1HL(2,1) + pi2*(1-p2H)*R1HH(2,1)+ delD*(1-q)*[ (1-(1-pi2)*p2L - pi2*p2H)* V1H10 + ((1-pi2)*p2L + pi2*p2H)* V1H11 ]+ delD*q*[ (1-(1-pi2)*p2L - pi2*p2H)* V1L10 + ((1-pi2)*p2L + pi2*p2H)* V1L11 ] ;
                u1H_s =  (1-p1H)* u1H_0 +  (p1H)* u1H_1;
                phi1H_0 = delI*max( 0, u1H_0 - u1H_s);
                phi1H_1 = delI*max( 0, u1H_1 - u1H_s);
                p1H_n = (p1H + phi1H_1)/ (1 + phi1H_0 + phi1H_1 );
                
                % User 2 state H
                u2H_0 = (1-pi1)*p1L*R2LH(2,1) + pi1*p1H*R2HH(2,1) + (1-pi1)*(1-p1L)*R2LL(1,1) + pi1*(1-p1H)*R2HL(1,1) + delD*(1-q)*[ (1-(1-pi1)*p1L - pi1*p1H)* V2H00 + ((1-pi1)*p1L + pi1*p1H)* V2H10 ] +delD*q*[ (1-(1-pi1)*p1L - pi1*p1H)* V2L00 + ((1-pi1)*p1L + pi1*p1H)* V2L10 ]  ;
                u2H_1 = (1-pi1)*p1L*R2LH(2,2) + pi1*p1H*R2HH(2,2) + (1-pi1)*(1-p1L)*R2LH(1,2) + pi1*(1-p1H)*R2HH(1,2)+ delD*(1-q)*[ (1-(1-pi1)*p1L - pi1*p1H)* V2H01 + ((1-pi1)*p1L + pi1*p1H)* V2H11 ] + delD*q*[ (1-(1-pi1)*p1L - pi1*p1H)* V2L01 + ((1-pi1)*p1L + pi1*p1H)* V2L11 ];
                u2H_s =  (1-p2H)* u2H_0 +  (p2H)* u2H_1;
                phi2H_0 = delI*max( 0, u2H_0 - u2H_s);
                phi2H_1 = delI*max( 0, u2H_1 - u2H_s);
                p2H_n = (p2H + phi2H_1)/ (1 + phi2H_0 + phi2H_1 );
                %p2H_n = p1H_n; % force symmetric eq
                
                %
                err = norm([p1L p2L p1H p2H ] - [p1L_n p2L_n p1H_n p2H_n ]);
                
                
                p1L = p1L_n;
                p2L = p2L_n;
                p1H = p1H_n;
                p2H = p2H_n;
            end
            
            [F1_0, F1_1] = Next_belief_state(pi1, p1L, p1H, p, q);
            [F2_0, F2_1] = Next_belief_state(pi2, p2L, p2H, p, q);
            
            [V1L00, V1L01, V1L10, V1L11]=reward2go_inf(pi1v,pi2v,F1_0,F1_1,F2_0,F2_1,u1Lm);
            [V1H00, V1H01, V1H10, V1H11]=reward2go_inf(pi1v,pi2v,F1_0,F1_1,F2_0,F2_1,u1Hm);
            [V2L00, V2L01, V2L10, V2L11]=reward2go_inf(pi1v,pi2v,F1_0,F1_1,F2_0,F2_1,u2Lm);
            [V2H00, V2H01, V2H10, V2H11]=reward2go_inf(pi1v,pi2v,F1_0,F1_1,F2_0,F2_1,u2Hm);
            
            
            % User 1 state L
            u1L_0 = (1-pi2)*p2L*R1LL(1,2) + pi2*p2H*R1LH(1,2) + (1-pi2)*(1-p2L)*R1LL(1,1) + pi2*(1-p2H)*R1LH(1,1) + delD*(1-p)*[ (1-(1-pi2)*p2L - pi2*p2H)* V1L00 + ((1-pi2)*p2L + pi2*p2H)* V1L01 ]+ delD*p*[ (1-(1-pi2)*p2L - pi2*p2H)* V1H00 + ((1-pi2)*p2L + pi2*p2H)* V1H01 ];
            u1L_1 = (1-pi2)*p2L*R1LL(2,2) + pi2*p2H*R1LH(2,2) + (1-pi2)*(1-p2L)*R1LL(2,1) + pi2*(1-p2H)*R1LH(2,1) + delD*(1-p)*[ (1-(1-pi2)*p2L - pi2*p2H)* V1L10 + ((1-pi2)*p2L + pi2*p2H)* V1L11 ]+ delD*p*[ (1-(1-pi2)*p2L - pi2*p2H)* V1H10 + ((1-pi2)*p2L + pi2*p2H)* V1H11 ] ;
            
            
            % User 2 state L
            u2L_0 = (1-pi1)*p1L*R2LL(2,1) + pi1*p1H*R2HL(2,1) + (1-pi1)*(1-p1L)*R2LL(1,1) + pi1*(1-p1H)*R2HL(1,1) + delD*(1-p)*[ (1-(1-pi1)*p1L - pi1*p1H)* V2L00 + ((1-pi1)*p1L + pi1*p1H)* V2L10 ]+ delD*p*[ (1-(1-pi1)*p1L - pi1*p1H)* V2H00 + ((1-pi1)*p1L + pi1*p1H)* V2H10 ] ;
            u2L_1 = (1-pi1)*p1L*R2LL(2,2) + pi1*p1H*R2HL(2,2) + (1-pi1)*(1-p1L)*R2LL(1,2) + pi1*(1-p1H)*R2HL(1,2) + delD*(1-p)*[ (1-(1-pi1)*p1L - pi1*p1H)* V2L01 + ((1-pi1)*p1L + pi1*p1H)* V2L11 ] + delD*p*[ (1-(1-pi1)*p1L - pi1*p1H)* V2H01 + ((1-pi1)*p1L + pi1*p1H)* V2H11 ] ;
            
            %p2L_n = p1L_n; % force symmetric eq
            
            % User 1 , state H
            u1H_0 = (1-pi2)*p2L*R1HL(1,2) + pi2*p2H*R1HH(1,2) + (1-pi2)*(1-p2L)*R1HL(1,1) + pi2*(1-p2H)*R1HH(1,1) + delD*(1-q)*[ (1-(1-pi2)*p2L - pi2*p2H)* V1H00 + ((1-pi2)*p2L + pi2*p2H)* V1H01 ]+ delD*q*[ (1-(1-pi2)*p2L - pi2*p2H)* V1L00 + ((1-pi2)*p2L + pi2*p2H)* V1L01 ] ;
            u1H_1 = (1-pi2)*p2L*R1HL(2,2) + pi2*p2H*R1HH(2,2) + (1-pi2)*(1-p2L)*R1HL(2,1) + pi2*(1-p2H)*R1HH(2,1)+ delD*(1-q)*[ (1-(1-pi2)*p2L - pi2*p2H)* V1H10 + ((1-pi2)*p2L + pi2*p2H)* V1H11 ]+ delD*q*[ (1-(1-pi2)*p2L - pi2*p2H)* V1L10 + ((1-pi2)*p2L + pi2*p2H)* V1L11 ] ;
            
            % User 2 state H
            u2H_0 = (1-pi1)*p1L*R2LH(2,1) + pi1*p1H*R2HH(2,1) + (1-pi1)*(1-p1L)*R2LL(1,1) + pi1*(1-p1H)*R2HL(1,1) + delD*(1-q)*[ (1-(1-pi1)*p1L - pi1*p1H)* V2H00 + ((1-pi1)*p1L + pi1*p1H)* V2H10 ] +delD*q*[ (1-(1-pi1)*p1L - pi1*p1H)* V2L00 + ((1-pi1)*p1L + pi1*p1H)* V2L10 ]  ;
            u2H_1 = (1-pi1)*p1L*R2LH(2,2) + pi1*p1H*R2HH(2,2) + (1-pi1)*(1-p1L)*R2LH(1,2) + pi1*(1-p1H)*R2HH(1,2)+ delD*(1-q)*[ (1-(1-pi1)*p1L - pi1*p1H)* V2H01 + ((1-pi1)*p1L + pi1*p1H)* V2H11 ] + delD*q*[ (1-(1-pi1)*p1L - pi1*p1H)* V2L01 + ((1-pi1)*p1L + pi1*p1H)* V2L11 ];
            
            u1L = (1-p1L) * u1L_0 + p1L * u1L_1;
            u1H = (1-p1H)* u1H_0 +  (p1H)* u1H_1;
            u2L = (1-p2L) * u2L_0 + p2L * u2L_1;
            u2H = (1-p2H)* u2H_0 +  (p2H)* u2H_1;
            
            p1Lv_n(i2)=p1L;
            p1Hv_n(i2)=p1H;
            p2Lv_n(i2)=p2L;
            p2Hv_n(i2)=p2H;
            
            u1Lv_n(i2)=u1L;
            u1Hv_n(i2)=u1H;
            u2Lv_n(i2)=u2L;
            u2Hv_n(i2)=u2H;
        end
        p1Lm_n(i1,:)=p1Lv_n;
        p1Hm_n(i1,:)=p1Hv_n;
        p2Lm_n(i1,:)=p2Lv_n;
        p2Hm_n(i1,:)=p2Hv_n;
        
        u1Lm_n(i1,:)=u1Lv_n;
        u1Hm_n(i1,:)=u1Hv_n;
        u2Lm_n(i1,:)=u2Lv_n;
        u2Hm_n(i1,:)=u2Hv_n;
    end
    err_p = norm([p1Lm p1Hm p2Lm p2Hm ] - [p1Lm_n p1Hm_n p2Lm_n p2Hm_n ]);
    err_u = norm([u1Lm u1Hm u2Lm u2Hm ] - [u1Lm_n u1Hm_n u2Lm_n u2Hm_n ]);
    [err_p err_u] % DISPLAY
    p1Lm=p1Lm_n;
    p1Hm=p1Hm_n;
    p2Lm=p2Lm_n;
    p2Hm=p2Hm_n;
    u1Lm=u1Lm_n;
    u1Hm=u1Hm_n;
    u2Lm=u2Lm_n;
    u2Hm=u2Hm_n;
    
    k=round(0.1*N);
    results=[results; [t p1Lm(k,k) p1Hm(k,k) p2Lm(k,k) p2Hm(k,k)]]; % DISPLAY
    
end

results
mesh(p1Lm)
figure
mesh(p1Hm)
figure
mesh(u1Lm)
figure
mesh(u2Hm)
