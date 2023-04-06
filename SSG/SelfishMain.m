clearvars;
close all;
clc;

%{
Author:           Rajesh Kumar Mishra
 Description:     RL Implementation
Status:           Failed
Information:      Bigcase letters are for arrays and small case for scalars
                  things that come after the underscore are more states and actions
                        
%}

eps = 1e-3;
T = 5;                 % Time Horizon
N = 10;                 % Resolution of Belief-space

% State transition
for q = [.1 .9]             % Probability of transition

    Delta = 0.6;            % Discount Factor

% Leaders' Payoffs
Rf1_0 = [2 4; 1 3];
Rf1_1 = [3 2; 0 1];

% Follower's Payoffs
Rf2_0 = [1 0; 0 2];
Rf2_1 = [2 0; 1 1];

% Belief
P = (0:N)*(1/N);

%Init Strategies
Pf1_01 = 0.5*ones(N+1, N+1);        % Follower
Pf1_11 = 0.5*zeros(N+1, N+1);
Pf2_01 = 0.5*ones(N+1, N+1);        % Follower
Pf2_11 = 0.5*zeros(N+1, N+1);

% Strategy space
Pf = 0:.25:1;

Mf = length(Pf);

%Value Function
Qf1 = zeros(2, 2, 2, N+1, N+1, Mf, Mf, Mf, Mf);
Qf2 = zeros(2, 2, 2, N+1, N+1, Mf, Mf, Mf, Mf);

BR_f1 = zeros(N+1, N+1, 2);
BR_f2 = zeros(N+1, N+1, 2);

Eq1 = zeros(N+1, N+1, 4);

B = 10;

t=0;
err = 1;

while t <= T
    t=t+1;
    t
    % Temp Strategy Values
    Pf1_01_n = zeros(N+1,N+1);
    Pf1_11_n = zeros(N+1,N+1);
    
    Pf2_01_n = zeros(N+1,N+1);
    Pf2_11_n = zeros(N+1,N+1);
    
    % Belief Follower 1
    
    for i1=1:length(P)
        
        % Belief Follower 2
        for i2 = 1:length(P)
            
            % Follower 1 gamma
            
            pf1_01 = Pf1_01(i1, i2);        % tgamma_tf
            pf1_11 = Pf1_11(i1, i2);        % tgamma_tf
            
            % Follower 2 gamma
            
            pf2_01 = Pf2_01(i1, i2);        % tgamma_tf
            pf2_11 = Pf2_11(i1, i2);        % tgamma_tf
            
            
            [pf1_01, pf1_11] = solve_fp_follower(P, i1, i2, Pf, pf1_01, pf1_11, pf2_01, pf2_11, Qf1);
            
            BR_f1(i1, i2, 1) = pf1_01;      % Final equilibrium strategy
            BR_f1(i1, i2, 2) = pf1_11;
            
            [pf2_01, pf2_11] = solve_fp_follower(P, i2, i1, Pf, pf2_01, pf2_11, pf1_01, pf1_11, Qf2);
            
            BR_f2(i1, i2, 1) = pf2_01;      % Final equilibrium strategy
            BR_f2(i1, i2, 2) = pf2_11;
            
            
        end
        
        Pf1_01_n(i1,i2) = BR_f1(i1, 1);
        Pf1_11_n(i1,i2) = BR_f1(i1, 2);
        
        Pf2_01_n(i1,i2) = BR_f2(i1, 1);
        Pf2_11_n(i1,i2) = BR_f2(i1, 2);
    end
    
    Pf1_01 = Pf1_01_n;
    Pf1_11 = Pf1_11_n;

    Pf2_01 = Pf2_01_n;
    Pf2_11 = Pf2_11_n;
    
    
    [Qf1, Qf2] = EstimateQ(P, Pf, Qf1, Qf2, q, Pf1_01, Pf1_11, Pf2_01, Pf2_11, t,  Rf1_0, Rf1_1, Rf2_0, Rf2_1, Delta);

    figure;
    plot(Pf_01)
    pause(.1)
end
end
