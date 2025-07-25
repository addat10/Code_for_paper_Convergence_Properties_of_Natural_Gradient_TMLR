%==========================================================================
% LOCAL SECTION OF KL DIVERGENCE IN ETA AND THETA COORDINATES
%==========================================================================
% This script plots KL divergence values along random directional sections 
% near the target and superimposes it with a quadratic function.
%
% Authors: Adwait Datar
% Associated Paper: Convergence Properties of Natural Gradient Descent for 
% Minimizing KL Divergence
%==========================================================================
clear all
close all
clc
rng(18) % 3,4,5,8 for n=2
% addpath('lib\')
%%
% Dimension of the probability simplex: here n=2 means a 3-element distribution
n=2; % dimension of sample space-1
q=rand(n+1,1); q=1/sum(q)*q;% target distribution 
p_0=rand(n+1,1);p_0=1/sum(p_0)*p_0; % Random initialization

% Representation in eta coordinates
g=@(p)p(1:size(p,1)-1,:);
g_inv=@(eta) [eta;ones(1,size(eta,2))-sum(eta)];
eta_q=g(q);% target distribution in eta coordinates
% eta_p0=g(p_0);% initialized distribution in eta coordinates

% Representation in theta coordinates
h=@(p) log((1./p(end,:)).*p(1:size(p,1)-1,:));
h_inv=@(theta)(1./(1+sum(exp(theta)))).*[exp(theta);ones(1,size(theta,2))];
theta_q=h(q); % target distribution (theta coord)
% theta_p0=h(p_0); % initial distribution (theta coord)

% KL in eta and theta coordinates
L_eta=@(eta) sum(eta_q.*log(eta_q./eta))+q(end)*log(q(end)/(1-sum(eta)));
Z=@(theta)(1+sum(exp(theta)));
V_theta = @(theta) 1/Z(theta_q)*(exp(theta_q)'*(theta_q-theta))+log((1+sum(exp(theta)))/Z(theta_q)) ;

%% Get section of L in eta coordinates
section_samples=50;
s_eta=-0.1:0.01:0.1;
L_vec=get_section(section_samples,s_eta,L_eta,eta_q);
% Quadratic function for comparison
f_quadratic_eta=@(eta) norm(eta-eta_q,2)^2;
L_quad_eta=get_section(section_samples,s_eta,f_quadratic_eta,eta_q);

%% Get section of V in theta coordinates
s_theta=-1:0.01:1;
V_vec=get_section(section_samples,s_theta,V_theta,theta_q);
% Quadratic function for comparison
f_quadratic_theta=@(theta) norm(theta-theta_q,2)^2;
L_quad_theta=get_section(section_samples,s_theta,f_quadratic_theta,theta_q);
%% Plot sections
figure()
plot(s_eta,L_quad_eta,'g')
hold on
plot(s_eta,L_vec,'b')
xlabel('eta')
ylabel('L(eta)/ quadratic')

figure()
plot(s_theta,L_quad_theta,'g')
hold on
plot(s_theta,V_vec,'r')
xlabel('theta')
ylabel('V(theta)/ quadratic')

%% User-defined functions
function f_vec=get_section(section_samples,s,f,x0)
    
    f_vec=zeros(section_samples,size(s,2));
    for j=1:section_samples
        alpha=0+(j/section_samples)*2*pi*rand();
        x_vec=x0+[cos(alpha);sin(alpha)]*s;
        for i=1:size(s,2)
            f_vec(j,i)=f(x_vec(:,i));
        end
    end
end