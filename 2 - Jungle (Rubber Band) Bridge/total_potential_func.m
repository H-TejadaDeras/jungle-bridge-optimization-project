%% compute the total potential energy of the bridge
%INPUTS:
%coords: vector of vertex positions from i=1 to i=(n-1)
% [x_1;y_1;...;x_(n-1),y_(n-1)]
%param_struct: struct containing parameters of the bridge
% param_struct.r0 = [x_0;y_0]: coordinates of leftmost vertex
% param_struct.rn = [x_n;y_n]: coordinates of rightmost vertex
% param_struct.num_links: number of rubber bands in bridge
% param_struct.k_list = [k_1;...;k_n]: list of stiffnesses
% param_struct.l0_list = [l0_1;...;l0_n]: list of natural lengths
% param_struct.m_list = [m_1;...;m_(n-1)]: list of weight masses
% param_struct.g = 9.8 m/sec^2: gravitational acceleration
%OUTPUTS:
%U_total: total potential energy of the bridge
function U_total = total_potential_func(coords,param_struct)
    %compute the gravitational potential energy of the weights
    U_g_total = total_G_potential_func(coords, param_struct);
    %compute the spring potential energy of the rubber bands
    U_RB_total = total_RB_potential_func(coords, param_struct);
    %sum the two results
    U_total = U_g_total + U_RB_total;
end