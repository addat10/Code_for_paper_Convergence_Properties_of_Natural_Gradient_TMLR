%==========================================================================
% DEPENDANCE OF CONVERGENCE RATES ON LEARNING RATE
%==========================================================================
% This script simulates natural gradient flow and usual gradient flow 
% in primal (eta) and dual (theta) coordinates and sweeps over different 
% learning rates to evaluate convergence speed and rise times.
%
% Authors: Adwait Datar
% Associated Paper: Convergence Properties of Natural Gradient Descent for 
% Minimizing KL Divergence
%==========================================================================
% Set random seed for reproducibility
rng(18);
% Clear workspace, close figures, clear command window
clear all; close all; clc;


% Parameters
n=10; % dimension of sample space-
num_samples = 1000; % Number of samples from target distribution
batch_size = num_samples;  % Choose based on your data size

% uncomment the following line if we want to simulate SGD
% batch_size = 100;  % Choose based on your data size

num_iterations=100;% number of iterations/updates
no_alphas=100;  % Number of learning rates to test

% Learning rates for different updates (eta, natural gradient, theta)
learning_rate_eta_list = linspace(0, 0.006, no_alphas);
learning_rate_eta_ng_list = linspace(0, 3, no_alphas);
learning_rate_theta_list = linspace(0, 20, no_alphas);

a=10000;% Parameter for learning rate decay (effectively unused here)

% Coordinate charts
g=@(p)p(1:size(p,1)-1,:); % eta coordinates
g_inv=@(eta) [eta;ones(1,size(eta,2))-sum(eta)];

h=@(p) log((1./p(end,:)).*p(1:size(p,1)-1,:)); % theta coordinates
h_inv=@(theta)(1./(1+sum(exp(theta)))).*[exp(theta);ones(1,size(theta,2))];

% Generate a random target distribution q on simplex
q=rand(n+1,1); q=1/sum(q)*q;% target distribution

% Convert target distribution to eta and theta coordinates
eta_q=g(q);% target distribution in eta coordinates
theta_q=h(q); % target distribution (theta coord)

% Fisher information matrix at target distribution and its condition number
G_opt=diag(eta_q)-eta_q*eta_q';
cond(G_opt)

% Number of simulations
num_sims = 100;


% Initialize starting distributions (randomly)

% p_0=q+1*(0.5+rand(n+1,num_sims));p_0=p_0./sum(p_0); % Random initialization
p_0=rand(n+1,num_sims);p_0=p_0./sum(p_0); % Random initialization
eta_p0=g(p_0);% initialized distribution in eta coordinates
theta_p0 = h(p_0);

% Following code allows to sample close to the target
% delta_0 = randn(n, num_sims); delta_0 = delta_0 ./ vecnorm(delta_0);
% eta_p0=eta_q + 0.1*delta_0;% initialized distribution in eta coordinates
% p_0 = g_inv(eta_p0);
% theta_p0=theta_q+0.1*delta_0;% initialized distribution in theta coordinates

% Loss function in eta coordinates (KL divergence from target)
L_eta=@(eta) sum(eta_q.*log(eta_q./eta))+q(end)*log(q(end)/(1-sum(eta)));

% Sample from the target distribution
samples = randsample(n+1, num_samples, true, q);

% Compute empirical distribution of samples
counts = histcounts(samples, 1:(n+2)); % Histogram of samples
q_sample = (counts/num_samples)';
eta_q_sample=g(q_sample);% target distribution in eta coordinates

% Compute minimal empirical loss (for convergence threshold)
min_loss = empirical_KL(q_sample,samples);

% Initialize eta coordinates
eta = zeros(n,num_iterations,num_sims);
eta(:,1,:) = eta_p0; % Reasonable starting point

% Initialize theta coordinates
theta = zeros(n,num_iterations,num_sims);
theta(:,1,:) = theta_p0; % Reasonable starting point
% eta_from_theta = zeros(n,num_iterations,num_sims);

% Initialize eta for natural gradient simulation
eta_ng = zeros(n,num_iterations,num_sims);
eta_ng(:,1,:) = eta_p0; % Reasonable starting point

% Storage for probability distributions for each iteration
model_p_eta = zeros(n+1,num_iterations,num_sims);
model_p_theta = zeros(n+1,num_iterations,num_sims);
model_p_eta_ng = zeros(n+1,num_iterations,num_sims);

% Storage for empirical loss and true KL risk over iterations
loss_history_eta = zeros(1, num_iterations,num_sims);
loss_history_theta = zeros(1, num_iterations,num_sims);
loss_history_eta_ng = zeros(1, num_iterations,num_sims);

risk_history_eta = zeros(1, num_iterations,num_sims);
risk_history_theta = zeros(1, num_iterations,num_sims);
risk_history_eta_ng = zeros(1, num_iterations,num_sims);

% Variables to track convergence times (rise time)
lr_size=size(learning_rate_eta_list,2);
rise_time_eta=num_iterations*ones(num_sims,lr_size);
rise_time_eta_ng=num_iterations*ones(num_sims,lr_size);
rise_time_theta=num_iterations*ones(num_sims,lr_size);

% Gradient descent loop over learning rates and simulations
for lr_id = 1:lr_size
    learning_rate_eta = learning_rate_eta_list(1,lr_id);
    learning_rate_eta_ng = learning_rate_eta_ng_list(1,lr_id);
    learning_rate_theta = learning_rate_theta_list(1,lr_id);
    for sim_id = 1: num_sims
        for iter = 1:(num_iterations-1)
            % Convert parameters to probability distributions
            model_p_eta(:,iter,sim_id) = eta_to_p(eta(:,iter,sim_id));
            model_p_theta(:,iter,sim_id) = theta_to_p(theta(:,iter,sim_id));
            model_p_eta_ng(:,iter,sim_id) = eta_to_p(eta_ng(:,iter,sim_id));
            
            % Compute empirical loss (using samples) for each parameterization
            loss_history_eta(1,iter,sim_id) = empirical_KL(model_p_eta(:,iter,sim_id),samples);
            loss_history_theta(1,iter,sim_id) = empirical_KL(model_p_theta(:,iter,sim_id),samples);
            loss_history_eta_ng(1,iter,sim_id) = empirical_KL(model_p_eta_ng(:,iter,sim_id),samples);
            
            % Compute true KL risk (using target distribution q)
            risk_history_eta(1,iter,sim_id) = KL(q,model_p_eta(:,iter,sim_id));
            risk_history_theta(1,iter,sim_id) = KL(q,model_p_theta(:,iter,sim_id));
            risk_history_eta_ng(1,iter,sim_id) = KL(q,model_p_eta_ng(:,iter,sim_id));
        
            
            % batch_indices = randi(num_samples, [1, batch_size]);
            % batch = samples(batch_indices);
            % counts = histcounts(batch, 1:(n+2));
            % emp_target_dist_batch = (counts/batch_size)';
            
            % Compute gradients of empirical KL loss wrt parameters
            grad_eta = grad_empirical_KL_eta(q_sample,model_p_eta(:,iter,sim_id));
            grad_theta = grad_empirical_KL_theta(q_sample,model_p_theta(:,iter,sim_id));
            natgrad_eta = natgrad_empirical_KL_eta(q_sample,model_p_eta_ng(:,iter,sim_id));
        
            % % Update eta using gradient descent
            % eta(iter+1,:) = eta(iter,:) - learning_rate * grad;
            tol_simplex = 1e-3;  
            b=1;%a/(iter+a); % learning rate scaling (can be decayed)
            
            % Gradient update in eta coordinates followed by projection onto simplex
            eta(:,iter+1,sim_id) = eta(:,iter,sim_id) - learning_rate_eta*b* grad_eta;
            eta(:,iter+1,sim_id)=project_simplex(eta(:,iter+1,sim_id),tol_simplex);
            
            % Gradient update in theta coordinates
            theta(:,iter+1,sim_id) = theta(:,iter,sim_id) - learning_rate_theta*b* grad_theta;
            
            % Natural gradient update in eta coordinates followed by projection onto simplex
            eta_ng(:,iter+1,sim_id) = eta_ng(:,iter,sim_id) - learning_rate_eta_ng*b* natgrad_eta;
            eta_ng(:,iter+1,sim_id)=project_simplex(eta_ng(:,iter+1,sim_id),tol_simplex);
            
            % Record rise time (first iteration where loss is close to min_loss)
            if loss_history_eta(1,iter,sim_id)<1.05*min_loss && rise_time_eta(sim_id,lr_id)==num_iterations
                rise_time_eta(sim_id,lr_id)=iter;
            elseif loss_history_eta(1,iter,sim_id)>=1.05*min_loss && rise_time_eta(sim_id,lr_id)<num_iterations
                rise_time_eta(sim_id,lr_id)=num_iterations;
            end
            if loss_history_eta_ng(1,iter,sim_id)<1.05*min_loss && rise_time_eta_ng(sim_id,lr_id)==num_iterations
                rise_time_eta_ng(sim_id,lr_id)=iter;
            elseif loss_history_eta_ng(1,iter,sim_id)>1.05*min_loss && rise_time_eta_ng(sim_id,lr_id)<num_iterations
                rise_time_eta_ng(sim_id,lr_id)=num_iterations;
            end
            if loss_history_theta(1,iter,sim_id)<1.05*min_loss && rise_time_theta(sim_id,lr_id)==num_iterations
                rise_time_theta(sim_id,lr_id)=iter;
            elseif loss_history_theta(1,iter,sim_id)>1.05*min_loss && rise_time_theta(sim_id,lr_id)<num_iterations
                rise_time_theta(sim_id,lr_id)=num_iterations;
            end
            

        end

        % eta_from_theta(:,:,sim_id)=g(h_inv(theta(:,:,sim_id)));
        % Store final model distributions after last iteration
        model_p_eta(:,end,sim_id) = eta_to_p(eta(:,end,sim_id));
        model_p_theta(:,end,sim_id) = theta_to_p(theta(:,end,sim_id));
        model_p_eta_ng(:,end,sim_id) = eta_to_p(eta_ng(:,end,sim_id));
    
        % Compute final empirical loss and risk
        loss_history_eta(1,end,sim_id) = empirical_KL(model_p_eta(:,end,sim_id),samples);
        loss_history_theta(1,end,sim_id) = empirical_KL(model_p_theta(:,end,sim_id),samples);
        loss_history_eta_ng(1,end,sim_id) = empirical_KL(model_p_eta_ng(:,end,sim_id),samples);
        
        risk_history_eta(1,end,sim_id) = KL(q,model_p_eta(:,end,sim_id));
        risk_history_theta(1,end,sim_id) = KL(q,model_p_theta(:,end,sim_id));
        risk_history_eta_ng(1,end,sim_id) = KL(q,model_p_eta_ng(:,end,sim_id));
    
        % Estimate convergence rates from early iterations via linear fit of log loss
        flag_linear_fit=1;
        plot_flag=0;
        coeffs_eta(:,sim_id)=polyfit(1:2,log(loss_history_eta(1,1:2,sim_id)),1);
        coeffs_eta_ng(:,sim_id)=polyfit(1:2,log(loss_history_eta_ng(1,1:2,sim_id)),1);
        coeffs_theta(:,sim_id)=polyfit(1:2,log(loss_history_theta(1,1:2,sim_id)),1);
    end
    % Track worst-case convergence rate and rise time across simulations
    worst_conv_rate_eta(lr_id)=max(coeffs_eta(1,:));
    worst_conv_rate_eta_ng(lr_id)=max(coeffs_eta_ng(1,:));
    worst_conv_rate_theta(lr_id)=max(coeffs_theta(1,:));

    worst_rise_time_eta(lr_id)=max(rise_time_eta(:,lr_id));
    worst_rise_time_eta_ng(lr_id)=max(rise_time_eta_ng(:,lr_id));
    worst_rise_time_theta(lr_id)=max(rise_time_theta(:,lr_id));

end

%%
% Plot rise times vs learning rates for different updates
figure()
plot(learning_rate_eta_list,worst_rise_time_eta)
xlabel('learning rates')
ylabel('empricial rise time (eta)')
figure()
plot(learning_rate_eta_ng_list,worst_rise_time_eta_ng)
xlabel('learning rates')
ylabel('empricial rise time (eta ng)')
figure()
plot(learning_rate_theta_list,worst_rise_time_theta)
xlabel('learning rates')
ylabel('empricial rise time (theta)')
%% User-defined functions
function p = eta_to_p(eta)
    p = [eta;1 - sum(eta)];
end
function eta = p_to_eta(p)
    eta = p(1:end-1,1);
end
function p = theta_to_p(theta)
    p = (1/(1+sum(exp(theta))))*[exp(theta); 1];
end
function empirical_kl = empirical_KL(p,samples)
    % Compute empirical KL divergence: E_q[log(q(x)/p(x))]
    log_model_probs = log(p(samples));
    empirical_kl = -mean(log_model_probs); % q(x) ~ empirical, so q(x) = 1/N
end
function KL = KL(q,p)
    % Compute empirical KL divergence: E_q[log(q(x)/p(x))]
    KL=q'*log(q./p);
end
function grad = grad_empirical_KL_eta(q,p)
    n = size(p,1)-1;
    grad = zeros(n,1);
    for i = 1:n
        grad(i,1) = -q(i,1)/p(i,1) + q(n+1,1)/p(n+1,1);
    end
end
function natgrad = natgrad_empirical_KL_eta(q,p)
    n = size(p,1)-1;
    grad = zeros(n,1);
    for i = 1:n
        grad(i,1) = -q(i,1)/p(i,1) + q(n+1,1)/p(n+1,1);
    end
    eta=p_to_eta(p);
    G=diag(eta)-eta*eta';
    natgrad = G*grad;
end
function grad = grad_empirical_KL_theta(q,p)
    n = size(p,1)-1;
    grad = zeros(n,1);
    for i = 1:n
        grad(i,1) = -q(i,1)+p(i,1);
    end
end
function []=plot_contours(eta_q,eta_q_sample,traj_1,traj_2,traj_3,f,legend1,legend2,legend3,label)
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
    plot(eta_q_sample(1),eta_q_sample(2),'o')
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
function [coeffs_1,coeffs_2,coeffs_3] = linear_fit(KL1,KL2,KL3)
    % This function plots 3 trajectories KL1, KL2 and KL3 along with a best
    % linear fit trajectory if the flag_linear_fit is set to 1
    iters=size(KL1,2);
    ts=1;tf=0.8*iters; % time-span data to be used for the linear fit
    coeffs_1=polyfit(ts:tf,KL1(ts:tf),1);
    coeffs_2=polyfit(ts:tf,KL2(ts:tf),1);
    coeffs_3=polyfit(ts:tf,KL3(ts:tf),1);
end
function [coeffs_1,coeffs_2,coeffs_3]=plot_KL3(t,sub_sample_freq,KL1,KL2,KL3,ylabel_string,legend1,legend2,legend3,flag_linear_fit)
% This function plots 3 trajectories KL1, KL2 and KL3 along with a best
% linear fit trajectory if the flag_linear_fit is set to 1
iters=size(KL1,2);
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
function eta = project_simplex(eta,tol_simplex)
    if min(eta) < tol_simplex
        indices = find(eta< tol_simplex);
        eta(indices)=tol_simplex;
    end
    if sum(eta) > 1-tol_simplex
        eta = eta/sum(eta+tol_simplex);
    end
end