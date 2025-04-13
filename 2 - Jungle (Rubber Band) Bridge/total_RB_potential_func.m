%% compute the total potential energy of all rubber bands in bridge
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
%U_RB_total: total potential energy of rubber bands in bridge
function U_RB_total = total_RB_potential_func(coords,param_struct)
    %initialize total spring potential energy
    U_RB_total = 0;

    %add the first and last vertex positions to the coordinate list
    coords_full = [param_struct.r0;coords;param_struct.rn];

    %iterate through each rubber band link
    for i = 1:param_struct.num_links
        %extract the ith stiffness and natural length
        l0 = struct2array(param_struct.l0_list);
        k = struct2array(param_struct.k_list);

        %extract the coordinates of the rubber band ends
        xA = coords_full(2 * i - 1);
        yA = coords_full(2 * i);
        xB = coords_full(2 * i + 1);
        yB = coords_full(2 * 1 + 2);

        %compute the potential energy of the ith rubber band
        U_RB_i = single_RB_potential_func(xA, yA, xB, yB, k(i), l0(i));

        %add the ith potential to the total
        U_RB_total = U_RB_total + U_RB_i;
    end
end