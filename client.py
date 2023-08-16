import connection as cn 
import random
import numpy as np

s = cn.connect(2037)

# mapeamento de número para ação
def conversionChoice(num):
    print(num)
    if num==0:
        return "left"
    elif num==1:
        return "right"
    elif num==2:
        return "jump"

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


# uso da biblioteca numpy para ler o 'resultado.txt' como uma matriz
q_table = np.loadtxt('resultado.txt')

# setando as casas decimais dos valores da q_table
np.set_printoptions(precision=6)

# estado inicial 
state = (0,0)   

alpha = 0.1     # taxa de aprendizagem que diz o quão rápido o agente aprende
gamma = 0.8     # fator de desconto, diz o peso da recompensa futura em relação à imediata
epsilon = 0    # epsilon greedy strategy -> uma taxa que define se o agente irá tomar ações aleatórias ou embasadas

while(True):
    # gera um número aleatório
    random_num =  random.randint(0,2)                       
    # transforma para a ação (aleatória)
    random_action = conversionChoice(random_num)            
    # recebe o index da linha do estado atual
    state_index = get_state(state[0],state[1])              
    # gera a melhor ação para o estado atual
    based_num = best_action(state_index,q_table)            
    # transforma um número para a ação 
    based_action = conversionChoice(based_num)              
    # gera um número aleatório entre 0 e 1 - excluindo 1
    random_float = random.uniform(0,1)                    

    # caso o valor gerado seja maior que epsilon o agente irá usar a q_table para tomar a ação, caso contrário será aleatória sua tomada de decisão   
      
    if (random_float >= epsilon):                          
        action_num = based_num                              
        action = based_action                               
    else:                                                   # epsilon = 0 -> ações sempre usando a q_table    
        action_num = random_num                             # epsilon = 1 -> ações sempre aleatórias
        action = random_action  

    # chamada da função para ativar a ação e receber o estado / recompensa
    next_state, rw = cn.get_state_reward(s, action)          

    # prints a título apenas de ter uma ideia do funcionamento do robô enquanto ele opera
    print(f'action:{action}')
    print(f'state:{next_state}')
    print(f'bounty:{rw}') 

    # converte o estado binário dado em uma tupla com a plataforma e a direção em decimais
    next_state = conversion(next_state)                       
     # recebe o index da linha do estado
    next_state_index = get_state(next_state[0],next_state[1]) 

    # atualiza a q_table
    q_table[state_index, action_num] = q_update(q_table,state_index,action_num,next_state_index,rw,alpha,gamma)   

    # escreve no resultado.txt
    np.savetxt('resultado.txt', q_table, fmt="%f")             

    # o estado atual era o antigo próximo estado
    state = next_state                                         
