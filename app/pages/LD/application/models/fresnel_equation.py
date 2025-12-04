from numpy import sin, cos, arcsin


def fresnel_equation_calculate(n_i, n_t, theta_i):
    theta_t = arcsin(n_i * sin(theta_i) / n_t)

    r_s = ((n_i * cos(theta_i) - n_t * cos(theta_t)) / 
           (n_i * cos(theta_i) + n_t * cos(theta_t)))**2
    r_p = ((n_i * cos(theta_t) - n_t * cos(theta_i)) / 
           (n_i * cos(theta_t) + n_t * cos(theta_i)))**2

    return (r_s + r_p) / 2


if __name__ == '__main__':
    print(fresnel_equation_calculate(1.47, 1, arcsin(0.8)))
