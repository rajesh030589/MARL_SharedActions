clearvars;
close all;
clc;

%{
Author:           Rajesh Kumar Mishra
Description:      MARL Strategic Implementation
%}

eps = 1e-3;
T = 5;                  % Time Horizon
N = 10;                 % Resolution of Belief-space

% State transition
p = 0.1;              % Probability of transition
q = 0.2;

d = 0.9;            % Discount Factor

xL = 0.2;
xH = 1.2;


% User1 Payoffs
R(1, 1, :, :) = [0 1; 1-xL 1-xL];
R(1, 2, :, :) = [0 1; 1-xL 1-xL];
R(2, 1, :, :) = [0 1; 1-xH 1-xH];
R(2, 1, :, :) = [0 1; 1-xH 1-xH];


% Belief
P = (0:N)*(1/N);

%Init Strategies
P_1_01 = 0.5*ones(N+1, N+1);        % Follower
P_1_11 = 0.5*ones(N+1, N+1);
P_2_01 = 0.5*ones(N+1, N+1);        % Follower
P_2_11 = 0.5*ones(N+1, N+1);

%Value Function
V_1 = zeros(2, N+1, N+1);
V_2 = zeros(2, N+1, N+1);
t=0;
while t <= T
    t=t+1;
    t
    % Temp Strategy Values user 1
    P_1_01_n = zeros(N+1,N+1);
    P_1_11_n = zeros(N+1,N+1);
    
    % Temp Strategy Values user 2
    P_2_01_n = zeros(N+1,N+1);
    P_2_11_n = zeros(N+1,N+1);
    
    V_1_n = zeros(2, N+1, N+1);
    V_2_n = zeros(2, N+1, N+1);
    % Belief Follower 1
    for i1=1:length(P)
        
        % Belief Follower 2
        for i2 = 1:length(P)
            
            % User 1 gamma
            p_1_01 = P_1_01(i1, i2);        % tgamma_tf
            p_1_11 = P_1_11(i1, i2);        % tgamma_tf
            
            % User 2 gamma
            p_2_01 = P_2_01(i1, i2);        % tgamma_tf
            p_2_11 = P_2_11(i1, i2);        % tgamma_tf
            
            [p_1_01, p_1_11, p_2_01, p_2_11, Vpi_1, Vpi_2] = solve_fp_follower(P, i1, i2, R, p_1_01, p_1_11, p_2_01, p_2_11, V_1, V_2, p, q, d);
            
            P_1_01_n(i1, i2) = p_1_01;
            P_1_11_n(i1, i2) = p_1_11;
            
            P_2_01_n(i1, i2) = p_2_01;
            P_2_11_n(i1, i2) = p_2_11;
            
            V_1_n(:, i1, i2) = Vpi_1;
            V_2_n(:, i2, i1) = Vpi_2;
            
        end
    end

    err_p = norm([P_1_01 P_1_11 P_2_01 P_2_11] - [P_1_01_n P_1_11_n P_2_01_n P_2_11_n ]);
    err_u = norm([V_1(1) V_1(2) V_2(1) V_2(2) ] - [V_1_n(1) V_1_n(2) V_2_n(2) V_2_n(2) ]);
    [err_p err_u]
    P_1_01 = P_1_01_n;
    P_1_11 = P_1_11_n;
    
    P_2_01 = P_2_01_n;
    P_2_11 = P_2_11_n;
    
    
    Qf = EstimateQ(P, Pf, q, P_1_01, Pf1_11, Pf2_01, Pf2_11, t,  Rf1_0, Rf1_1, Rf2_0, Rf2_1, Delta);
    
    
    V_1 = V_1_n;
    V_2 = V_2_n;
end
