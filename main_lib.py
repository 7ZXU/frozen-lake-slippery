#-*- coding: utf-8 -*-
import numpy as np
import copy


def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    
    while True :
        delta = 0
        
        ## loop through states
        for s in range(env.nS): ## env.nS == S_n
            Vs = 0 # temp state value
            for a, action_prob in enumerate(policy[s]): ## polciy action probability
                for prob, next_state, reward in env.MDP[s][a]: ## mdp[s][a] = [transition probability, newstate, reward]
                    Vs += action_prob * prob * (reward + gamma * V[next_state]) 
            delta = max(delta, np.abs(V[s]-Vs)) ## abs : absolute value
            
            V[s] = Vs
            
        if delta < theta : ## if delta small than theta, break
            break
    return V

def q_value(env, V, s, gamma = 0.99):
    q = np.zeros(env.nA)
    
    for a in range(env.nA):
        for prob, next_state, reward in env.MDP[s][a]:
            q[a] += prob * (reward + gamma * V[next_state])
    
    return q  

def policy_improvement(env, V, gamma=0.99):
    ###########################policy_stable##################################
    policy = np.zeros([env.nS, env.nA]) / env.nA
    
    for s in range(env.nS):
        # old_action = policy[s]
        q = q_value(env, V, s, gamma)
        
        ## np.argwhere(q==np.max(q)) ## return index where is the max q value ## shape (1, 1)
        ## np.argwhere(q==np.max(q)).flatten() ## shape (1,)
        best_a = np.argwhere(q==np.max(q)).flatten() 
        
        ## 최대값 동일한 경우 action은 인덱스 작은 값 선택 
        if len(best_a) > 1 :
            best_a = np.array([best_a.min()])
        
        ## np.eye(env.nA) ## make identity matrix ## shape (env.nA, env.nA) 
        # policy[s] = np.sum([np.eye(env.nA)[i] for i in best_a], axis = 0) / len(best_a)
        policy[s] = np.sum([np.eye(env.nA)[i] for i in best_a], axis = 0) / len(best_a)

    return policy

# gamma : discount value 
# theta : 종료 조건

def policy_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS) ## initialize
    policy = np.ones([env.nS, env.nA]) / env.nA ## (16, 4) matrix full of 0.25 # action
    
    while True:
        V = policy_evaluation(env, policy, gamma, theta)
        old_policy = policy 
        new_policy = policy_improvement(env, V, gamma)
        # print('old', old_policy, 'new', new_policy)
        if (old_policy==new_policy).all():
            break;
        
        ########################################## policy stable == theta ##########################################
        ## 반복문 종료 조건 고치기
        # if np.max(abs(policy_evaluation(env, policy) - policy_evaluation(env, new_policy))) < theta * 1e2:
        #     break;   
        
        policy = copy.copy(new_policy)   
    
    return policy, V

def value_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        delta = 0
        
        for s in range(env.nS):
            v = V[s]
            V[s] = max(q_value(env, V, s, gamma))
            delta = max(delta, abs(V[s]-v))
            
        if delta < theta:
            break
        
        policy = policy_improvement(env, V, gamma)
        
    return policy, V