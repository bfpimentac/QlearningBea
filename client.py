import connection as cn 
import random
import numpy as np

s = cn.connect(2037)

# escolha aleatória de ações
def conversionChoice(num):
    print(num)
    if num==0:
        return "left"
    elif num==1:
        return "right"
    elif num==2:
        return "jump"

# state = (plataforma, direcao)


# conversão da plataforma e da direção de binários para decimal
def conversion(state):
    platform = str(state)[2:7] 
    direction = str(state)[7:9]

    platform_inversed = ''.join(reversed(platform))
    direction_inversed = ''.join(reversed(direction))

    iterations = 0
    platform_dec  = 0
    direction_dec = 0

    for i in platform_inversed:
        if i == '1':    
            platform_dec += 2 ** iterations
        
        iterations +=  1 
    
    iterations = 0

    for j in direction_inversed:
        if j == '1':
            direction_dec += 2 ** iterations

        iterations += 1

    return (platform_dec,direction_dec)


# função para mapeamento de plataforma/direção -> estado
def get_state(platform,direction):
    state = (platform*4) + direction

    return state

# função para retorno da melhor ação para cada estado
def best_action(state_index,q_table):
    if (q_table[state_index,0] > q_table[state_index,1]) and (q_table[state_index,0] > q_table[state_index,2]):
        action_index = 0
    elif (q_table[state_index,1] > q_table[state_index,0]) and (q_table[state_index,1] > q_table[state_index,2]):
        action_index = 1
    else:
        action_index = 2

    return action_index


# uso da equação de atualização da q_table 

def q_update(q_table,state,action,next_state,rw,alpha,gamma):

    estimate_q = rw + gamma * max(q_table[next_state,0],q_table[next_state,1], q_table[next_state,2])

    q_value = q_table[state,action] + alpha*(estimate_q - q_table[state,action])

    return q_value



q_table = np.loadtxt('resultado.txt')

np.set_printoptions(precision=6)

state = (0,0)   # estado inicial 

alpha = 0.2     # taxa de aprenzidagem que diz o quão rápido o agente aprende
gamma = 0.7     # fator de desconto, diz o peso da recompensa futura em relação à imediata
epsilon = 0    # epsilon greedy strategy -> uma taxa que define se o agente irá tomar ações aleatórias ou embasadas

while(True):

    random_num =  random.randint(0,2)                       # gera um número aleatório

    random_action = conversionChoice(random_num)            # transforma para a ação (aleatória)

    state_index = get_state(state[0],state[1])              # recebe o index da linha do estado

    based_num = best_action(state_index,q_table)            # gera a melhor ação para o estado atual

    based_action = conversionChoice(based_num)              # transforma um número para a ação 

    random_float = random.uniform(0,1)                    # gera um número aleatório entre 0 e 1 - excluindo 1

    if (random_float >= epsilon):                           # caso o valor gerado seja maior que epsilon o agente 
        action_num = based_num                              # irá usar a q_table para tomar a ação, caso     
        action = based_action                               # contrário será aleatória sua tomada de decisão
    else:                                                   # epsilon = 0 -> ações sempre usando a q_table    
        action_num = random_num                             # epsilon = 1 -> ações sempre aleatórias
        action = random_action  

    next_state, rw = cn.get_state_reward(s, action)          # chamada da função para ativar a ação e receber o estado / recompensa

    print(f'action:{action}')
    print(f'state:{next_state}')
    print(f'bounty:{rw}') 

    next_state = conversion(next_state)                       # converte o estado binário dado em uma tupla com a plataforma e a direção em decimais

    next_state_index = get_state(next_state[0],next_state[1]) # recebe o index da linha do estado

    q_table[state_index, action_num] = q_update(q_table,state_index,action_num,next_state_index,rw,alpha,gamma)   # atualiza a q_table

    np.savetxt('resultado.txt', q_table, fmt="%f")             # escreve no resultado.txt


    state = next_state                                         # o estado atual era o antigo próximo estado
