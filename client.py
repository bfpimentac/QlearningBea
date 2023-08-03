import connection as cn 
import random
import numpy as np

s = cn.connect(2037)

# escolha aleatória de ações
def randomChoice(num):
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


# uso da equação de atualização da q_table 

def q_update(q_table,state,action,next_state,rw,alpha,gamma):

    estimate_q = rw + gamma * max(q_table[next_state,0],q_table[next_state,1], q_table[next_state,2])

    q_value = q_table[state,action] + alpha*(estimate_q - q_table[state,action])

    return q_value


q_table = np.loadtxt('resultado.txt')

state = (0,0)

alpha = 0.2
gamma = 0.5

while(True):

    random_num =  random.randint(0,2)   # gera um número aleatório

    action = randomChoice(random_num) # Escolha da ação (aleatória)

    next_state, rw = cn.get_state_reward(s, action) # chamada da função para ativar a ação e receber estado / recompensa

    next_state = conversion(next_state)     # converte o estado binário dado em uma tupla com a plataforma e a direção em decimais

    next_state_index = get_state(next_state[0],next_state[1]) # recebe o index da linha do estado

    state_index = get_state(state[0],state[1])    # recebe o index da linha do estado

    q_table[state_index, random_num] = q_update(q_table,state_index,random_num,next_state_index,rw,alpha,gamma)   # atualiza a q_table

    np.savetxt('resultado.txt', q_table)    # escreve no resultado.txt

    print(f'action:{action}')
    print(f'state:{next_state}')
    print(f'bounty:{rw}') 

    state = next_state                    # o estado atual era o antigo próximo estado
