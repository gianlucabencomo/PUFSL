import torch
import numpy as np

def BatchSampler(tasks, pos, neg, iterations):
    '''
    Description: Each iteration of every episode 
    consists of randomly sampling a task and
    curating a dataset of pos/neg examples. This
    sampler does exactly that.
    Inputs:
        tasks - (list of data objects) tasks to
                sample from
        pos - (int) number of positive examples
        neg - (int) number of negative examples
        iterations - (int) number of iterations
                     in episode
    Returns:
        (list of lists of single data objects)
    '''
    D = []
    for task in tasks:
        # create an array of indicies and shuffle
        samp = np.array(range(len(task)))
        np.random.shuffle(samp)
         
        i = 0
        while (i < len(samp)):
            pos_count = 0
            neg_count = 0
            data = []
            
            # gather pos and neg examples and put in data
            while (pos_count + neg_count < pos + neg):
                y = task[samp[i]].y.numpy().item()
                if (pos_count < pos and y == 1):
                    data.append(task[samp[i]])
                    pos_count += 1
                elif (neg_count < neg and y == 0):
                    data.append(task[samp[i]])
                    neg_count += 1
                i += 1
                # break if we reach end
                if (i == len(samp)):
                    break
            # since we don't have enough samples
            # throw the last sample away
            if (i == len(samp)):
                break
 
            # return list instead of list of lists if 1
            if len(data) == 1:
                    return data[0][:iterations]
           
            # shuffle examples so pos/neg are 
            # uniformly distributed.  they are not
            # uniformly distributed without.
            np.random.shuffle(data) 
            D.append(data)

    # shuffle arrangement of task draws
    np.random.shuffle(D)
    return D[:iterations]
