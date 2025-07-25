function [gradp_V,Hessp_V,gradq_V,Hessq_V]=get_grad_hess_symbolic_dual()
syms theta_p_1 theta_p_2 theta_q_1 theta_q_2
theta_p=[theta_p_1;theta_p_2];
theta_q=[theta_q_1;theta_q_2];
temp_theta_p=1+sum(exp(theta_p));
temp_theta_q=1+sum(exp(theta_q));
V=log(temp_theta_p/temp_theta_q)+(1/temp_theta_q)*sum(exp(theta_q).*(theta_q-theta_p));
dV_dp1=diff(V,theta_p_1);
dV_dp2=diff(V,theta_p_2);
dV_dq1=diff(V,theta_q_1);
dV_dq2=diff(V,theta_q_2);

d2V_dp1dp1=diff(dV_dp1,theta_p_1);
d2V_dp1dp2=diff(dV_dp1,theta_p_2);
d2V_dp2dp2=diff(dV_dp2,theta_p_2);
gradp_V(theta_p_1,theta_p_2,theta_q_1,theta_q_2)=[dV_dp1;dV_dp2];
Hessp_V(theta_p_1,theta_p_2,theta_q_1,theta_q_2)=[d2V_dp1dp1,d2V_dp1dp2;d2V_dp1dp2,d2V_dp2dp2];

d2V_dq1dq1=diff(dV_dq1,theta_q_1);
d2V_dq1dq2=diff(dV_dq1,theta_q_2);
d2V_dq2dq2=diff(dV_dq2,theta_q_2);
gradq_V(theta_p_1,theta_p_2,theta_q_1,theta_q_2)=[dV_dq1;dV_dq2];
Hessq_V(theta_p_1,theta_p_2,theta_q_1,theta_q_2)=[d2V_dq1dq1,d2V_dq1dq2;d2V_dq1dq2,d2V_dq2dq2];
end