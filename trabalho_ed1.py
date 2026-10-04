import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#constantes de decaimento
Sb135 = 6.795e-3
Te135 = 3.648e-2
I135 = 2.930e-5
Xe135 = 2.106e-5

#condição inicial

N0 = np.array([1.0e24, 0, 0, 0, 0])

t_final = 600
t_inicial = 0

#Passos Pedidos

h_euler = [1,5,10]

h_rk4 = [2,5,10]

#Sistema do problema
def sistemaEdo(t, N):
    dN1dt = -Sb135 * N[0]
    dN2dt = Sb135 * N[0] - Te135 * N[1]
    dN3dt = Te135 * N[1] - I135 * N[2]
    dN4dt = I135 * N[2] - Xe135 * N[3]
    dN5dt = Xe135 * N[3]

    return np.array([dN1dt, dN2dt, dN3dt, dN4dt, dN5dt])

#Função auxiliadora dos métodos 
def montarMatriz(t0, tf, h, N0):
    t = np.arange(t0, tf+h, h)
    N = np.zeros((len(t), len(N0)))
    N[0] = N0
    return N, t

#Métodos Pedidos

def metodoEuler(f, N0, t0, tf, h):
    N , t= montarMatriz(t0,tf,h,N0)

    for i in range (1, len(t)):
        N[i] = N[i-1] + h*f(t[i-1], N[i-1])

    return N, t

def Rk4(f, N0, t0, tf, h):
    N, t = montarMatriz(t0,tf,h,N0)
    for i in range(1, len(t)):
        k1 = f(t[i-1], N[i-1])
        k2 = f(t[i-1] + h/2, N[i-1] + (h/2) * k1)
        k3 = f(t[i-1] + h/2, N[i-1] + (h/2) * k2)
        k4 = f(t[i-1] + h, N[i-1] + h * k3)
        
        N[i] = N[i-1] + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
    return N,t

#Solução analítica
def solucao_analitica(t, N0_val):

    N1_a = N0_val * np.exp(-Sb135 * t)
    
    N2_a = N0_val * (Sb135 / (Te135 - Sb135)) * (np.exp(-Sb135 * t) - np.exp(-Te135 * t))
    
    c3_1 = np.exp(-Sb135 * t) / ((Te135 - Sb135) * (I135 - Sb135))
    c3_2 = np.exp(-Te135 * t) / ((Sb135 - Te135) * (I135 - Te135))
    c3_3 = np.exp(-I135 * t) / ((Sb135 - I135) * (Te135 - I135))
    N3_a = N0_val * Sb135 * Te135 * (c3_1 + c3_2 + c3_3)
    
    c4_1 = np.exp(-Sb135 * t) / ((Te135 - Sb135) * (I135 - Sb135) * (Xe135 - Sb135))
    c4_2 = np.exp(-Te135 * t) / ((Sb135 - Te135) * (I135 - Te135) * (Xe135 - Te135))
    c4_3 = np.exp(-I135 * t) / ((Sb135 - I135) * (Te135 - I135) * (Xe135 - I135))
    c4_4 = np.exp(-Xe135 * t) / ((Sb135 - Xe135) * (Te135 - Xe135) * (I135 - Xe135))
    N4_a = N0_val * Sb135 * Te135 * I135 * (c4_1 + c4_2 + c4_3 + c4_4)
    
    N5_a = N0_val - (N1_a + N2_a + N3_a + N4_a)
    
    return N1_a, N2_a, N3_a, N4_a, N5_a

t_analitico = np.linspace(0, 600, 1000)

# Obtenção dos dados
N1_exato, N2_exato, N3_exato, N4_exato, N5_exato = solucao_analitica(t_analitico, N0[0])



#Grafico e Resultados

ResultadoEuler = {}
for h in h_euler:
    N_e, T_e = metodoEuler(sistemaEdo, N0, t_inicial, t_final, h)
    ResultadoEuler[h] = (N_e, T_e)

ResultadoRk4 = {}
for h in h_rk4:
    N_rk4, T_rk4 = Rk4(sistemaEdo, N0, t_inicial, t_final, h)
    ResultadoRk4[h] = (N_rk4, T_rk4)


nuclideos = ['Sb-135', 'Te-135', 'I-135', 'Xe-135', 'Cs-135']
solucoes_exatas = [N1_exato, N2_exato, N3_exato, N4_exato, N5_exato]


for i in range(5):
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Evolução do {nuclideos[i]} ao longo do tempo', fontsize=16, fontweight='bold')
    
 
    for j in range(3):
        h_e = h_euler[j]
        h_r = h_rk4[j]
        
      
        N_e, T_e = ResultadoEuler[h_e]
        N_rk4, T_rk4 = ResultadoRk4[h_r]
        
      
        axs[j].plot(t_analitico, solucoes_exatas[i], 'k-', linewidth=2, label='Analítica')
        
     
        axs[j].plot(T_e, N_e[:, i], 'b--', marker='o', markersize=4, alpha=0.7, 
                    label=f'Euler h={h_e}s')
      
        axs[j].plot(T_rk4, N_rk4[:, i], 'r:', marker='s', markersize=4, alpha=0.7, 
                    label=f'RK4 h={h_r}s')
        
        
        axs[j].set_title(f'Comparação {j+1}: Euler ({h_e}s) vs RK4 ({h_r}s)')
        axs[j].set_xlabel('Tempo (s)')
        axs[j].set_ylabel('Quantidade de Núcleos')
        axs[j].legend(loc='best')
        axs[j].grid(True, linestyle=':', alpha=0.6)
        
        
        axs[j].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85) 
    plt.show()

#Calculo do desvio padrão
tempos_alvo = [0, 150, 300, 450, 600]


def calcular_erro(num, exato):
    if exato == 0.0:
        return 0.0 if num == 0.0 else float('inf')
    return abs((num - exato) / exato) * 100


resultados_erro = []

for t in tempos_alvo:
    
    exatos_t = solucao_analitica(t, N0[0])
    
   
    for h in h_euler:
        idx = int(t / h) 
        num_euler = ResultadoEuler[h][0][idx]
        
        linha = {'Tempo (s)': t, 'Método': 'Euler', 'Passo h (s)': h}
        for i in range(5):
            linha[f'Erro {nuclideos[i]} (%)'] = calcular_erro(num_euler[i], exatos_t[i])
        resultados_erro.append(linha)
        

    for h in h_rk4:
        idx = int(t / h)
        num_rk4 = ResultadoRk4[h][0][idx]
        
        linha = {'Tempo (s)': t, 'Método': 'RK4', 'Passo h (s)': h}
        for i in range(5):
            linha[f'Erro {nuclideos[i]} (%)'] = calcular_erro(num_rk4[i], exatos_t[i])
        resultados_erro.append(linha)


df_erros = pd.DataFrame(resultados_erro)


nome_arquivo = 'desvios_relativos.csv'
df_erros.to_csv(nome_arquivo)