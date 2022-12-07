\documentclass{article}
\begin{document}

The well known Pythagorean theorem \(x^2 + y^2 = z^2\) was 
proved to be invalid for other exponents. 
Meaning the next equation has no integer solutions:

$L_{VAE}^{trd}=E\[log_{p_\theta}(x_L|z_L)\]-\beta D_{KL}(q_\chi(z_L|x_L)||p_z)+E\[log_{p_\theta}(x_U|z_U)\]-\beta D_{KL}(q_\chi(z_U|x_U)||p_z)$
\end{document}

$L^adv_VAE=-E[log(D(q_\chi(z_L|x_L))]- E[log(D(q_\chi(z_U|x_U))]$

The training objective function is following:

$L_D=-E[log(D(q_\chi(z_L|x_L))]- E[log(1-D(q_\chi(z_U|x_U))]$

By combining the above 2 formula, we have the full objective function for VAE:

$L_{VAE}=\lambda_1L_{VAE}^{trd}+\lambda_2L_{VAE}^{adv}$




