%% compute the gravitational potential energy of a SINGLE weight
%INPUTS
%(x,y): coordinates of a the current vertex
%m: mass of the weight
%g: gravitational acceleration
%OUTPUTS:
%U_g_i: gravitational potential energy of weight
function U_g_i = single_G_potential_func(x,y,m,g)
    %compute gravitational potential energy of weight
    U_g_i = m * g * y;
end