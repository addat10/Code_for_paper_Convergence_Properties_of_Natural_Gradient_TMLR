%==========================================================================
% EMPIRICAL CONVERGENCE RATES ESTIMATES FOR CONTINUOUS TIME GRADIENT FLOWS
%==========================================================================
%
% This script plots empirically observed convergence rates for natural
% gradient flows and gradient flows in eta and theta coordinates
%
% Authors: Adwait Datar
% Associated Paper: Convergence Properties of Natural Gradient Descent for 
% Minimizing KL Divergence
%==========================================================================
% This script measures the numerical convergence rates for the different
% gradient flows
close all
clear all
clc
rng(18) % 3,4,5,8 for n=2
addpath('lib\')
%%
n=10; % dimension of sample space-1
p_samples=10;
q_samples=1;
num_conv_rate_nat_grad=zeros(q_samples,p_samples);
num_conv_rate_eta_grad=zeros(q_samples,p_samples);
num_conv_rate_theta_grad=zeros(q_samples,p_samples);
conv_rate_est_eta_grad=zeros(q_samples,p_samples);
conv_rate_est_theta_grad=zeros(q_samples,p_samples);
% Set target distribution
q=rand(n+1,q_samples);%
q=q*inv(diag(sum(q)));
% Set initial distribution
p_0=rand(n+1,p_samples);
p_0=p_0*inv(diag(sum(p_0))); % Random initialization

% Representation in eta coordinates
g=@(p)p(1:size(p,1)-1,:);
g_inv=@(eta) [eta;ones(1,size(eta,2))-sum(eta)];

% Representation in theta coordinates
h=@(p) log((1./p(end,:)).*p(1:size(p,1)-1,:));
h_inv=@(theta)(1./(1+sum(exp(theta)))).*[exp(theta);ones(1,size(theta,2))];
% theta_q=h(q); % target distribution (theta coord)
% theta_p0=h(p_0); % initial distribution (theta coord)

% KL in eta and theta coordinates
L_eta=@(eta,eta_q) sum(eta_q.*log(eta_q./eta))+(1-sum(eta_q))*log((1-sum(eta_q))/(1-sum(eta)));
Z=@(theta)(1+sum(exp(theta)));
V_theta = @(theta,theta_q) 1/Z(theta_q)*(exp(theta_q)'*(theta_q-theta))+log((1+sum(exp(theta)))/Z(theta_q)) ;

% Natural gradient in eta coordinates
neg_nat_grad=@(eta,eta_q)-1*(eta-eta_q);
% dLdeta in eta coordinates
neg_grad_eta=@(eta,eta_q)-1*(-eta_q./eta+(1-sum(eta_q))/(1-sum(eta)));
% DVdtheta in theta coordinates
neg_grad_theta=@(theta,theta_q)-1*(1/Z(theta)*exp(theta)-1/Z(theta_q)*exp(theta_q));

%% Simulate trajectories for all combinations
delta_t=1e-4;
stop_tol=1e-3;
for i=1:q_samples    
    q_i=q(:,i);
    for j=1:p_samples
        sprintf('i=%d | j= %d',i,j)
        p0_i=p_0(:,j);
       % Representation in eta coordinates
        eta_q=g(q_i);% target distribution in eta coordinates
        eta_p0=g(p0_i);% initialized distribution in eta coordinates
        
        % Representation in theta coordinates
        theta_q=h(q_i); % target distribution (theta coord)
        theta_p0=h(p0_i); % initial distribution (theta coord)


        % time-stepping
        [eta_p_nat_grad,KL_of_eta_natgrad,t_eta_natgrad]=simulate(eta_q,eta_p0,delta_t,stop_tol,neg_nat_grad,L_eta);
        [eta_p,KL_of_eta,t_eta]=simulate(eta_q,eta_p0,delta_t,stop_tol,neg_grad_eta,L_eta);
        [theta_p,KL_of_theta,t_theta]=simulate(theta_q,theta_p0,delta_t,stop_tol,neg_grad_theta,V_theta);

        % numerical conv rates
        sub_sample_freq=500;
        plot_flag=0;
        [coeffs_1]=plot_KL(t_eta_natgrad,sub_sample_freq,log(KL_of_eta_natgrad),'log(L(eta natgrad))',plot_flag);
        %sprintf('Numerical conv rate (nat grad)=%f | Estimate based on spectrum of Hessian= %f',-coeffs_1(1),2)
        num_conv_rate_nat_grad(i,j)=real(-coeffs_1(1));

        [coeffs_2]=plot_KL(t_eta,sub_sample_freq,log(KL_of_eta),'log(L(eta))',plot_flag);
        %sprintf('Numerical conv rate (eta grad flow)=%f | Estimate based on spectrum of Hessian= %f',-coeffs_2(1),1/max(eta_q))
        num_conv_rate_eta_grad(i,j)=real(-coeffs_2(1));
        conv_rate_est_eta_grad(i,j)=2/max(eta_q);

        [coeffs_3]=plot_KL(t_theta,sub_sample_freq,log(KL_of_theta),'log(V(theta))',plot_flag);
        %sprintf('Numerical conv rate (theta grad flow)=%f | Estimate based on spectrum of Hessian= %f',-coeffs_3(1),max(eta_q))
        num_conv_rate_theta_grad(i,j)=real(-coeffs_3(1));
        conv_rate_est_theta_grad(i,j)=2*max(eta_q);
    end
end
save('numerical_convergence_rates_data.mat')
%% Plot the obtained numerical rates
num_conv_rate_nat_grad_flat=reshape(num_conv_rate_nat_grad,[],1);
num_conv_rate_eta_grad_flat=reshape(num_conv_rate_eta_grad,[],1);
num_conv_rate_theta_grad_flat=reshape(num_conv_rate_theta_grad,[],1);
conv_rate_est_eta_grad_flat=reshape(conv_rate_est_eta_grad,[],1);
conv_rate_est_theta_grad_flat=reshape(conv_rate_est_theta_grad,[],1);

figure()
plot(num_conv_rate_nat_grad_flat,'r.')
hold on
plot(conv_rate_est_eta_grad_flat,'b--')
plot(2*ones(size(num_conv_rate_nat_grad_flat,1),1),'r-')
plot(num_conv_rate_eta_grad_flat,'b.')
plot(num_conv_rate_theta_grad_flat,'g.')
plot(conv_rate_est_theta_grad_flat,'g--')
xlabel('data')
ylabel('numerically observed convergence rate')


% Plot the randomly chosed initial and target distributions
if n==2
    figure()
    plot(p_0(1,:),p_0(2,:),'.')
    hold on
    plot(q(1,:),q(2,:),'*')
    plot([0;1],[1;0])
    xlabel('eta p (1)')
    ylabel('eta p (2)')
    xlim([0,1])
    ylim([0,1])
end
%% User-defined functions
function [p,KL,t]=simulate(eta_q,p0,delta_t,stop_tol,f_RHS,KLf)
% This function solves the ode with explicit Euler method and returns the
% tranjectories along with the evaluation of KLf along trajectories
    n=size(p0,1);
    p=p0; % Initialization    
    KL=KLf(p0,eta_q);
    while KL(end)>stop_tol
        p_next=p(:,end)+delta_t*f_RHS(p(:,end),eta_q);
        p=[p,p_next];
        KL=[KL,KLf(p_next,eta_q)];
    end
    t=delta_t:delta_t:size(KL,2)*delta_t;
end
function [coeffs]=plot_KL(t,sub_sample_freq,KL,ylabel_string,plot_flag)
% This function plots 3 trajectories KL1, KL2 and KL3 along with a best
% linear fit trajectory if the flag_linear_fit is set to 1
iters=size(KL,2);
ts=1;tf=iters; % time-span data to be used for the linear fit
coeffs=polyfit(t(ts:tf),KL(ts:tf),1);
if plot_flag==1
    figure();
    plot(t(1:sub_sample_freq:iters),KL(1:sub_sample_freq:iters))
    hold on
    % plot the best fit lines for log scale plots
    plot(t(1:sub_sample_freq:iters),coeffs(2)+coeffs(1)*t(1:sub_sample_freq:iters),'--')
    
    ylabel(ylabel_string)
    xlabel('time')
end
end
