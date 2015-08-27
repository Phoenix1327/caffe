# caffe
#caffe examples

#edition 1
#Add a new loss layer, 
#L1 norm: L = abs(x1) + abs(x2) + abs(x3) + ... + abs(xn), L = L / N
#L2 norm: L = x1^2 + x2^2 + x3^2 + ... + xn^2, L = L / N


#edition 2
#Add a new loss layer,
#L1 norm: 
#L_positive = (x1 + x2 + ... + xnp) / Np
#L_negative = (x1 + x2 + ... + xnn) / Nn
#L_weakly = (x1 + x2 + ... + xnw) / Nw
#L = alpha * L_positive + beta * L_negative + gamma * L_weakly

#L2 norm
#L_positive = (x1^2 + x2^2 + ... + xnp^2) / Np
#L_negative = (x1^2 + x2^2 + ... + xnn^2) / Nn
#L_weakly = (x1^2 + x2^2 + ... + xnw^2) / Nw
#L = alpha * L_positive + beta * L_negative + gamma * L_weakly