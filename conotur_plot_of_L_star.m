%==========================================================================
% CONTOUR PLOTS OF L*
%==========================================================================
%
% This script plots the contour plots of L*, expressed in natural (theta)
% coordinates
%
% Authors: Adwait Datar
% Associated Paper: Convergence Properties of Natural Gradient Descent for 
% Minimizing KL Divergence
%==========================================================================
clear; clc
%% Symbolic definitions
syms theta1 theta2 real
theta = [theta1; theta2];

% Define distribution q(theta) on the simplex
psi = exp(theta1) + exp(theta2) + 1;
q=1/psi*[exp(theta1);exp(theta2);1];

% Define fixed reference distribution p at theta = (0,0)
% Set fixed values for thetap1 and thetap2 (reference distribution)
thetap1_val = 0;
thetap2_val = 0;
thetap = [thetap1_val;thetap2_val];
psi_p = exp(thetap1_val) + exp(thetap2_val) + 1;
p=1/psi_p*[exp(thetap1_val);exp(thetap2_val);1];

% KL divergence D_KL(q || p)
KL = q'*log(q./p);
% KL = p'*log(p./q);

% Compute the gradient of KL w.r.t. theta
gradKL = gradient(KL, theta);

% Compute the Hessian of KL w.r.t. theta
hessKL = hessian(KL, theta);

% Simplify the result
hessKL = simplify(hessKL);

% Display
disp('Hessian of KL divergence w.r.t. theta:')
disp(hessKL)

%% Contour plot of KL divergence
% Convert symbolic expression to MATLAB function
KL_func = matlabFunction(KL, 'Vars', [theta1 theta2]);

% Define grid for theta1 and theta2
theta_lim=4;
[theta1_vals, theta2_vals] = meshgrid(linspace(-theta_lim, theta_lim, 100), linspace(-theta_lim, theta_lim, 100));

% Evaluate KL divergence over the grid
KL_vals = KL_func(theta1_vals, theta2_vals);

% Plot the contour of KL divergence
figure;
map = (0.1:0.1:1)'*[1,1,1]; % grayscale colormap
colormap(map)
contourf(theta1_vals, theta2_vals, KL_vals, 20); % 20 contour levels
colorbar;
set(gca, 'FontSize', 15, 'FontName', 'Times')
xlabel('$\theta_1$', 'Interpreter', 'latex');
ylabel('$\theta_2$', 'Interpreter', 'latex');
% title('Contour plot of KL divergence (fix second argument)');
title('Contour plot of $\mathcal{L}_p^*(\theta)$', 'Interpreter', 'latex');

%% Cross-sectional plots along radial directions
% Define angles for cross-sections (in radians)
angles = linspace(0, pi, 6); % 0 to 180 degrees, 6 directions
% angles = linspace(pi/4, pi/4, 1); % 0 to 180 degrees, 6 directions

% Range of r (distance from the center point)
r_vals = linspace(-theta_lim, theta_lim, 500);

% Prepare figure
figure;
hold on;

% Colors for different lines
colors = lines(length(angles));

% Loop over angles
for idx = 1:length(angles)
    alpha = angles(idx);
    direction = [cos(alpha); sin(alpha)];
    
    % Points along the line
    theta1_line = thetap1_val + r_vals * direction(1);
    theta2_line = thetap2_val + r_vals * direction(2);

    % theta1_line = -1 + r_vals * direction(1);
    % theta2_line = -1 + r_vals * direction(2);
    
    % Evaluate KL along this line
    KL_line = KL_func(theta1_line, theta2_line);
    
    % Plot
    plot(r_vals, KL_line, 'Color', colors(idx, :), 'DisplayName', sprintf('\\alpha = %.1f°', rad2deg(alpha)));
end

% Final plot touches
xlabel('r (distance along direction)');
ylabel('KL divergence');
title('Cross-sections of KL divergence along different directions');
legend show;
grid on;
hold off;
