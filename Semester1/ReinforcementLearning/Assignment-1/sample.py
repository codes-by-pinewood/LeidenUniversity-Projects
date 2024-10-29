def print_func():
    from tqdm import tqdm
    import time
    import sys 

    epsilon = [1,5,10]
    n_repetitions = 5 
    n_timesteps = 10 
    
    # is happening over the course of systme
    total_steps = len(epsilon)*n_repetitions
    count = 1 
    progress = round(100.0 * count / total_steps, 2)
    for e in epsilon:
        for n in range(n_timesteps):
            print('foo', end='')
            print('\rbar', end='', flush=True)
            # sys.stdout.write("Download progress: %d%%   \r" % (progress) )
            # sys.stdout.flush()
            count +=1


if __name__ == "__main__":
    import time
    # print_func()
    for i in range(100):
        print("", end=f"\rPercentComplete: {i} %")
        time.sleep(0.2)