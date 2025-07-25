function [p,L]=solve_ode(p0,T,delta_t,f_RHS,Loss)   
% This function solves the ode with explicit Euler method and returns the
% tranjectories along with the evaluation of Loss along trajectories
    iters=T/delta_t;
    n=size(p0,1);
    p=zeros(n,iters); p(:,1)=p0; % Initialization    
    L=zeros(iters,1);
    solver=1;
    switch solver
        case 1 % Solve via Explicit Euler                      
            for k=1:iters-1
                noise=eye(n)+0*(-1+2*rand(n,n));
                p(:,k+1)=p(:,k)+delta_t*noise*f_RHS(p(:,k));        
                L(k,1)=Loss(p(:,k));
            end
            L(end,1)=Loss(p(:,end));
        
        case 2 % Solve via MATLABs solver ode45
            t=0:delta_t:(T-delta_t);
            [~,p_transp]=ode45(@(t,p)f_RHS(p),t,p0);
            p=p_transp';
            for k=1:iters-1                
                L(k,1)=Loss(p(:,k));
            end

    end
end
