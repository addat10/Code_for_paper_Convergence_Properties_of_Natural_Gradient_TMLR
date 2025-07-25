%==========================================================================
% NATURAL GRADIENT VS STANDARD GRADIENT FLOWS
%==========================================================================
% This script simulates and compares the natural gradient flow with the
% usual gradient flow in primal (eta) and dual (theta) coordinates.
%
% The optimization variable is a probability distribution p over (n+1)
% categories. We study the trajectory of p under different gradient
% dynamics and visualize convergence (including KL-divergence decay).
%
% Authors: Adwait Datar
% Associated Paper: Convergence Properties of Natural Gradient Descent for 
% Minimizing KL Divergence
%==========================================================================
%% Setup
close all; clear all; clc;
rng(18) % Random seed (change for different runs)
addpath('lib\')

% Problem dimensionality (p lives in a simplex of size n+1)
n=2;

% Target distribution q
q=rand(n+1,1); 
q=1/sum(q)*q;

% Random initialization p0
p_0=rand(n+1,1);
p_0=1/sum(p_0)*p_0; 

%% Coordinate transforms (eta and theta coordinates)

% Representation in eta coordinates
g=@(p)p(1:size(p,1)-1,:);
g_inv=@(eta) [eta;ones(1,size(eta,2))-sum(eta)];

eta_q=g(q);% target distribution in eta coordinates
eta_p0=g(p_0);% initialized distribution in eta coordinates

% Representation in theta coordinates
h=@(p) log((1./p(end,:)).*p(1:size(p,1)-1,:));
h_inv=@(theta)(1./(1+sum(exp(theta)))).*[exp(theta);ones(1,size(theta,2))];

theta_q=h(q); % target distribution (theta coord)
theta_p0=h(p_0); % initial distribution (theta coord)

%% Loss functions (KL divergences in eta/theta coordinates)
% KL in eta and theta coordinates
L_eta=@(eta) sum(eta_q.*log(eta_q./eta))+q(end)*log(q(end)/(1-sum(eta)));
Z=@(theta)(1+sum(exp(theta)));
V_theta = @(theta) 1/Z(theta_q)*(exp(theta_q)'*(theta_q-theta))+log((1+sum(exp(theta)))/Z(theta_q)) ;

%% Gradient definitions
% Natural gradient in eta coordinates
neg_nat_grad=@(eta)-1*(eta-eta_q);

% Standard gradient dL/deta in eta coordinates
neg_grad_eta=@(eta)-1*(-eta_q./eta+(1-sum(eta_q))/(1-sum(eta)));

% Standard gradient DV/dtheta in theta coordinates
neg_grad_theta=@(theta)-1*(1/Z(theta)*exp(theta)-1/Z(theta_q)*exp(theta_q));

%% Gradient verification (optional)
verify_grads=1;
if n==2 & verify_grads==1
    % The following code slows down the execution. So use only for
    % verification.Does not work with ode45.
    [gradp_L,~,~,~]=get_grad_hess_symbolic();
    [gradp_V,~,~,~]=get_grad_hess_symbolic_dual();
    
    neg_grad_eta_auto=@(eta)-gradp_L(eta(1,1),eta(2,1),eta_q(1),eta_q(2));
    neg_grad_theta_auto=@(theta)-gradp_V(theta(1),theta(2),theta_q(1),theta_q(2));
    
    samples=100;
    
    error_eta=zeros(samples,1);
    error_theta=zeros(samples,1);
    
    for i=1:samples
        test_eta=rand(2,1);
        error_eta(i)=max(abs(double(neg_grad_eta(test_eta)-neg_grad_eta_auto(test_eta))));
        test_theta=rand(2,1);
        error_theta(i)=max(abs(double(neg_grad_theta(test_eta)-neg_grad_theta_auto(test_eta))));
    end
    max(error_eta)
    max(error_theta)
end
   

%% Simulate gradient flows
T=3;
delta_t=1e-5;
iters=T/delta_t;% number of iterations/updates

% Solve ODEs
[eta_p_nat_grad,KL_of_eta_natgrad]=solve_ode(eta_p0,T,delta_t,neg_nat_grad,L_eta);
[eta_p,KL_of_eta]=solve_ode(eta_p0,T,delta_t,neg_grad_eta,L_eta);
[theta_p,KL_of_theta]=solve_ode(theta_p0,T,delta_t,neg_grad_theta,V_theta);

%% Plot trajectories and KL divergence
t=0:delta_t:(T-delta_t);
sub_sample_freq=1000;

% Plot natural gradient trajectories
plot_trajectories(t,sub_sample_freq,eta_q,eta_p_nat_grad,'eta','eta nat grad trajectories')

% Plot eta-gradient flow trajectories
plot_trajectories(t,sub_sample_freq,eta_q,eta_p,'eta','eta trajectories')

% Plot theta-gradient flow trajectories
plot_trajectories(t,sub_sample_freq,theta_q,theta_p,'theta','theta trajectories')


% Plot KL along trajectories
flag_linear_fit=0;
plot_KL3(t,sub_sample_freq,KL_of_eta_natgrad,KL_of_eta,KL_of_theta,'KL','L(eta natgrad)','L(eta)','V(theta)',flag_linear_fit);

% Plot KL along trajectories in logscale
flag_linear_fit=1;
[coeffs_1,coeffs_2,coeffs_3]=plot_KL3(t,sub_sample_freq,log(KL_of_eta_natgrad),log(KL_of_eta),log(KL_of_theta),'log KL','log(L(eta natgrad))','log(L(eta))','log(V(theta))',flag_linear_fit);
coeffs_1(1)
coeffs_2(1)
coeffs_3(1)

%% Plot contour plots and trajectories for n=2
if n==2
    eta_p_from_theta=g(h_inv(theta_p));
    %L = @(x,y) (eta_q(1)*log(eta_q(1)/x)+eta_q(2)*log(eta_q(2)/y)+(1-eta_q(1)-eta_q(2))*log((1-eta_q(1)-eta_q(2))/(1-x-y)));
    f1=@(x,y) L_eta([x;y]) ;
    plot_contours(eta_q,eta_p,eta_p_nat_grad,eta_p_from_theta,f1,'eta','eta nat grad', '(g(h inv (theta)))','eta');
    
    theta_p_from_eta=h(g_inv(eta_p));
    theta_p_from_eta_natgrad=h(g_inv(eta_p_nat_grad));
    f2 = @(x,y) V_theta([x;y]);
    plot_contours(theta_q,theta_p_from_eta,theta_p_from_eta_natgrad,theta_p,f2,'h(g inv (eta))','h(g inv (eta nat grad))', 'theta','theta');
end

%% User-defined functions
function []=plot_trajectories(t,sub_sample_freq,target,traj,ylabel_string,title_string)
% This function plots the sub-sampled trajectories along with the target
% set point
iters=size(traj,2);
figure()
plot(t(1:sub_sample_freq:iters),target*ones(1,iters/sub_sample_freq),'--')
hold on
plot(t(1:sub_sample_freq:iters),traj(:,1:sub_sample_freq:iters))
ylabel(ylabel_string)
xlabel('time')
title(title_string)
end
function [coeffs_1,coeffs_2,coeffs_3]=plot_KL3(t,sub_sample_freq,KL1,KL2,KL3,ylabel_string,legend1,legend2,legend3,flag_linear_fit)
% This function plots 3 trajectories KL1, KL2 and KL3 along with a best
% linear fit trajectory if the flag_linear_fit is set to 1
iters=size(KL1,1);
ts=1;tf=0.8*iters; % time-span data to be used for the linear fit
coeffs_1=polyfit(t(ts:tf),KL1(ts:tf),1);
coeffs_2=polyfit(t(ts:tf),KL2(ts:tf),1);
coeffs_3=polyfit(t(ts:tf),KL3(ts:tf),1);

figure();
plot(t(1:sub_sample_freq:iters),KL1(1:sub_sample_freq:iters))
hold on
plot(t(1:sub_sample_freq:iters),KL2(1:sub_sample_freq:iters))
plot(t(1:sub_sample_freq:iters),KL3(1:sub_sample_freq:iters))

if flag_linear_fit==1
    % plot the best fit lines for log scale plots
    plot(t(1:sub_sample_freq:iters),coeffs_1(2)+coeffs_1(1)*t(1:sub_sample_freq:iters),'--')
    plot(t(1:sub_sample_freq:iters),coeffs_2(2)+coeffs_2(1)*t(1:sub_sample_freq:iters),'--')
    plot(t(1:sub_sample_freq:iters),coeffs_3(2)+coeffs_3(1)*t(1:sub_sample_freq:iters),'--')
end
ylabel(ylabel_string)
xlabel('time')
title('KL along gradient flows')
legend(legend1,legend2,legend3)
end
function []=plot_contours(eta_q,traj_1,traj_2,traj_3,f,legend1,legend2,legend3,label)
    % This function plots the trajectories in 2d along with the contour
    % plots specified by the function f
    sample_freq=10000;
    traj_1_coarse=downsample(traj_1',sample_freq)';
    traj_2_coarse=downsample(traj_2',sample_freq)';
    traj_3_coarse=downsample(traj_3',sample_freq)';
    figure()
     map = (0.1:0.1:1)'*[1,1,1];
    colormap(map)
    xlims=[min(traj_1(1,:))-0.1;max(traj_1(1,:))+0.1];
    ylims=[min(traj_1(2,:))-0.1;max(traj_1(2,:))+0.1];
    fcontour(f,[xlims(1) xlims(2) ylims(1) ylims(2)],'LevelList',[0:0.01:0.4],'LineWidth',1.2)
    hold on
    plot(eta_q(1),eta_q(2),'*')
    plot(traj_1(1,:),traj_1(2,:),'k','LineWidth',1.2)
    plot(traj_2(1,:),traj_2(2,:),'k','LineWidth',1.2)
    plot(traj_3(1,:),traj_3(2,:),'k','LineWidth',1.2)
    plot(traj_1_coarse(1,:),traj_1_coarse(2,:),'ko','MarkerSize', 8,'LineWidth',1.2)
    plot(traj_2_coarse(1,:),traj_2_coarse(2,:),'k*','MarkerSize', 8,'LineWidth',1.2)
    plot(traj_3_coarse(1,:),traj_3_coarse(2,:),'ko','MarkerFaceColor', 'k','MarkerSize', 8,'LineWidth',1.2)
    xlabel([label,'1'])
    ylabel([label,'2'])
    % legend('target distribution',legend1,legend2,legend3)
end