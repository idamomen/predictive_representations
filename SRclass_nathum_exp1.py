import numpy as np
import matplotlib.pyplot as plt
import random
from SR_no_action import SR_no_action
#from dyna_replay import dyna_replay

def SRclass_nathum_exp1(envstep, gamma, alpha, p_sample, verbose):

    ''' This class is an SR agent that takes the environment from Exp 1, 
        Nat Hum Beh paper and learns predictive representations with the 
        specified learning rate and scale. 

        Note: This is not SR dyna, nor SR-MB. 
        This agent only learns the SR.

        Inputs: 
        
        envstep:  generated with ida_envs.generate_nathumbeh_env1()        
        gamma: discount parameter, determines scale of predictive representations
        alpha: learning rate
        p_sample: prob sampling each of the two sequences 
        verbose: TRUE or FALSE 

        Outputs:

        M: SR matrix 
        W: value weights W
        memory: memory of episodes 
        episodies: # episodes it takes to reach convergence '''

    SR_agent = SR_no_action(gamma, alpha, p_sample, len(envstep))
    episodes = 0
    done = False
   
    while True:
        SR_agent.biggest_change = 0
        # sample a starting point [really determined by experiment]
        s = np.random.choice(range(2), p=SR_agent.p_sample)

        done = False                
        while not done: # go through trajectory till the end
            
            s_new, reward, done = envstep[s] #take action        
            SR_agent.step(s, s_new, reward)
            
            s = s_new
        
        if verbose:
            if episodes % verbose ==0:                    
                print(f'SR training episode #{episodes} Done.')       
        episodes += 1        

        if SR_agent.convergence:
            print (episodes,' training episodes/iterations done')
            break

    return SR_agent.M, SR_agent.W , SR_agent.memory, episodes


