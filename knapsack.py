import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
class Knapsack: 
    def __init__(self) -> None:
        pass
    
    def input(): 
        capacity = 0
        num_of_classes = 0
        weights = []
        values = []
        labels = []
        
        f = open("input.txt", "r")
        print(f.read())
        
def input(): 
    capacity = 0
    num_of_classes = 0
    weights = []
    values = []
    labels = []
    
    f = open("input.txt", "r")
    lines = f.readlines()
    print(lines)

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))
