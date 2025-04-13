%% compute the total gravitational potential energy
%of all weights in bridge
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
%U_g_total: total gravitational potential energy
function U_g_total = total_G_potential_func(coords,param_struct)
    %initialize total gravitational potential energy
    U_g_total = 0;

    %iterate through each weight
    for i = 1:(param_struct.num_links-1)
        %extract the coordinates of the ith vertex
        x = coords(2 * i - 1);
        y = coords(2 * i);

        %extract the ith mass and the gravitational acceleration
        m = param_struct.m_list(i);
        g = param_struct.g;

        %compute the gravitational potential energy of the ith mass
        U_g_i = single_G_potential_func(x,y,m,g);

        %add the ith potential to the total
        U_g_total = U_g_total + U_g_i;
    end
end