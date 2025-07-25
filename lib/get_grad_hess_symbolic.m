function [gradp_L,Hessp_L,gradq_L,Hessq_L]=get_grad_hess_symbolic()
% This function returns the gradient and hessians of the KL divergence in
% eta coordinates

syms eta_p_1 eta_p_2 eta_q_1 eta_q_2 % Initialize symbolic variables for coordinates
eta_p=[eta_p_1;eta_p_2];
eta_q=[eta_q_1;eta_q_2];
eta_p_end=1-sum(eta_p);
eta_q_end=1-sum(eta_q);
L=eta_q_end*log(eta_q_end/eta_p_end)+sum(eta_q.*log(eta_q./eta_p));
dL_dp1=diff(L,eta_p_1);
dL_dp2=diff(L,eta_p_2);
dL_dq1=diff(L,eta_q_1);
dL_dq2=diff(L,eta_q_2);

d2L_dp1dp1=diff(dL_dp1,eta_p_1);
d2L_dp1dp2=diff(dL_dp1,eta_p_2);
d2L_dp2dp2=diff(dL_dp2,eta_p_2);

d2L_dq1dq1=diff(dL_dq1,eta_q_1);
d2L_dq1dq2=diff(dL_dq1,eta_q_2);
d2L_dq2dq2=diff(dL_dq2,eta_q_2);


gradp_L(eta_p_1,eta_p_2,eta_q_1,eta_q_2)=[dL_dp1;dL_dp2];
gradq_L(eta_p_1,eta_p_2,eta_q_1,eta_q_2)=[dL_dq1;dL_dq2];

Hessp_L(eta_p_1,eta_p_2,eta_q_1,eta_q_2)=[d2L_dp1dp1,d2L_dp1dp2;d2L_dp1dp2,d2L_dp2dp2];
Hessq_L(eta_p_1,eta_p_2,eta_q_1,eta_q_2)=[d2L_dq1dq1,d2L_dq1dq2;d2L_dq1dq2,d2L_dq2dq2];
end