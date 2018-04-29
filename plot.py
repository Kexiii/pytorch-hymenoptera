import matplotlib  
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pickle
import json

def main():
    config = json.load(open("config.json"))
    f = open(config['history'],'rb')
    history = pickle.load(f)
    
    #summarize history for accuracy 
    fig = plt.figure(1)
    plt.plot(history['train']['acc'])  
    plt.plot(history['val']['acc'])  
    plt.title("Model accuracy")  
    plt.ylabel("accuracy")  
    plt.xlabel("epoch")  
    plt.legend(["train","val"],loc="upper left")  
    plt.savefig("accuracy.jpg") 
    plt.close(1)
    
    #summarize history for loss
    fig = plt.figure(2)     
    plt.plot(history['train']['loss'])  
    plt.plot(history['val']['loss'])  
    plt.title("Model loss")  
    plt.ylabel("loss")  
    plt.xlabel("epoch")  
    plt.legend(["train","val"],loc="upper left")  
    plt.savefig("loss.jpg")
    plt.close(2)
    
    f.close()
    
    
if __name__ == '__main__':
    main()