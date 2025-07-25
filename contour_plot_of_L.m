%==========================================================================
% CONTOUR PLOTS OF L
%==========================================================================
%
% This script plots the contour plots of L, expressed in natural (eta)
% coordinates
%
% Authors: Adwait Datar
% Associated Paper: Convergence Properties of Natural Gradient Descent for 
% Minimizing KL Divergence
%==========================================================================
clear
clc
%% Symbolic definitions
syms eta1 eta2 etap1 etap2 real
eta = [eta1; eta2];
etap = [etap1;etap2];

% Define q(eta) and reference p(etap) on the simplex
q=[eta1;eta2;1-eta1-eta2];
p=[etap1;etap2;1-etap1-etap2];

% KL divergence D_KL(q || p)
KL = q' * log(q ./ p);
KL = q'*log(q./p);

% Compute the gradient of KL w.r.t. theta
gradKL = gradient(KL, eta);

% Compute the Hessian of KL w.r.t. theta
hessKL = hessian(KL, eta);

% Simplify the result
hessKL = simplify(hessKL);

% Display
disp('Hessian of KL divergence w.r.t. theta:')
disp(hessKL)

%% Contour plot of KL divergence

% Convert symbolic expression to MATLAB function
KL_func = matlabFunction(KL, 'Vars', [eta1 eta2 etap1 etap2]);

% Define grid for theta1 and theta2
[eta1_vals, eta2_vals] = meshgrid(linspace(0, 0.4, 100), linspace(0, 0.4, 100));

% Set fixed values for thetap1 and thetap2 (reference distribution)
etap1_val = 0.1;
etap2_val = 0.1;

% Evaluate KL divergence over the grid
KL_vals = KL_func(eta1_vals, eta2_vals, etap1_val, etap2_val);

% Plot the contour of KL divergence
figure;
contourf(eta1_vals, eta2_vals, KL_vals, 20); % 20 contour levels
colorbar;
xlabel('\eta_1');
ylabel('\eta_2');
title('Contour plot of KL divergence');
