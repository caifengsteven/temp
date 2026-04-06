% Author :
% Jakob Kisiala , June 2015
% Computes the scaled CVaR norm of a vector at a given alpha , using
% componentwise definition

% INPUT:
% x = n?by?1 vector of values
% alpha = scalar between 0 and 1
% OUTPUT:
% C S alpha = << x >>?S {alpha}

function C_S_alpha = Scaled_CVaR_Norm_Component(x , alpha )
C_S_alpha = 0;
% check i f alpha is admissible
if ( alpha < 0 || alpha > 1)
    display( ' Please put in an alpha such that 0 <= alpha <= 1 ? Scaled CVaR could not be calculated ' ) ;
    return
end

% check i f x is a vector
size_x = size (x) ;
dim_x = length ( size_x ) ;

if (dim_x > 2) % x has more than 2 dimensions
    display( ' Please only input vectors x ? Scaled CVaR could not be calculated ' ) ;
    return
end
if ( size_x (1) > 1 && size_x(2) > 1) % x is a matrix
    display( ' Please only input vectors x ? Scaled CVaR could not be calculated ' ) ;
    return
end

n = length(x) ;

% check four cases :
% 0: alpha = 0
% 1: alpha > (n?1)/n
% 2: alpha equal to some alpha j
% 3: alpha between alpha j and alpha { j+1}

% case 0: alpha = 0
if( alpha == 0)
    C_S_alpha = sum( abs (x) )/n;
	return
end

% for the remaining three cases additional vectors are needed :
alpha_j_vector = ( [ 0 : n-1]')/n;

% case 1: alpha > (n?1)/n
if ( alpha > alpha_j_vector(n) )
    C_S_alpha = max( abs (x) ) ;
    return
end

% sort vector x by magnitude of components
x_abs_sorted = sort ( abs (x) ) ;


epsilon = 1e-10;
temp_vector = alpha_j_vector - alpha ;

% case 2: alpha equal to some alpha j
if (any( abs ( temp_vector ) < epsilon ) )
    C_S_alpha = calculate_Norm_for_alpha_j( x_abs_sorted , alpha ) ;
    return
end

% case 3: alpha between alpha j and alpha { j+1}
% find alpha j
temp_index = temp_vector < 0;
alpha_j = max( alpha_j_vector ( temp_index ) ) ;
% find alpha { j+1}
temp_index = temp_vector > 0;
alpha_jPlus1 = min( alpha_j_vector ( temp_index ) ) ;

mu = (( alpha_jPlus1 - alpha ) *(1 - alpha_j ) ) / (( alpha_jPlus1 - alpha_j ) *(1 - alpha ) ) ;

C_aj = calculate_Norm_for_alpha_j( x_abs_sorted , alpha_j ) ;
C_ajPlus1 = calculate_Norm_for_alpha_j ( x_abs_sorted , alpha_jPlus1 ) ;

C_S_alpha = mu*C_aj + (1 - mu)*C_ajPlus1 ;
% function to calculate the C?S {alpha} for alpha j
function C_S_alpha1 = calculate_Norm_for_alpha_j ( vector , alpha_j )
	j = find ( abs ( alpha_j_vector - alpha_j ) < 1e-10) - 1;
	C_S_alpha1 = (1 / (n - j ) ) * sum( vector ( j +1:n) ) ;
end

end

