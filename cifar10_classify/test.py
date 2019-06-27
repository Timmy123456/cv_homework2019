import numpy as np

epoch = list(0. for i in range(20))

def change():
    #global EPOCH
    for i in range(20):
        epoch[i] = i

if __name__ == "__main__":
    change()
    print(epoch)
