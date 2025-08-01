import numpy as np
import matplotlib.pyplot as plt

def calculate_flop_increase(tpp, b):

    tmp = 1 + 0.009 * np.log(np.sqrt(20.0 / tpp)) ** 2
    flop_increase = tmp ** (1 / b)

    return flop_increase # F' / F_CE

def calculate_parameter_increase(tpp, b):

    tmp = tpp * (1 + 0.009 * np.log(np.sqrt(20.0 / tpp)) ** 2) ** (-1.0 / b)
    param_increase = np.sqrt(20 / tmp)

    return param_increase # N' / N_CE


b = 0.096
a = 4740.352 ** (1.0 / b)



tpp_list = np.logspace(0, 3, 100)   # values between 10^0 and 10^2

flop_increase = [calculate_flop_increase(tpp, b) for tpp in tpp_list]
param_increase = [calculate_parameter_increase(tpp, b) for tpp in tpp_list]

# Plot
plt.figure(figsize=(5,4))
 
sc = plt.scatter(
    flop_increase,
    param_increase,
    c=tpp_list,
    cmap='viridis',
    norm=plt.matplotlib.colors.LogNorm()
)

plt.xlabel('F / $F_{CE}$')
plt.ylabel('N / $N_{CE}$')

cbar = plt.colorbar(sc)
cbar.set_label('TPP')

plt.tight_layout()

plt.savefig("testing.png")

"""
fitted equation: 
loss = 4740.352 * flops ^ -0.096 * (1 + 0.009 * (ln(sqrt(20 / TPP)))^2)

a = 4740 ^ (1/b)
b = 0.096

"""