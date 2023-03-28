import math
import os
from queue import PriorityQueue
import random
dir_path = os.path.dirname(os.path.realpath(__file__))
W = 4  # number of nodes taken after a turn of generation
runs = 50
recursiveCalls = 2
recursiveCount = 0
exploredSet = dict()
maxVal = 0
resState = []

class Knapsack:
    def __init__(self) -> None:
        self.capacity = self.input()["capacity"]
        self.num_of_classes = self.input()["num_of_classes"]
        self.weights = self.input()["weights"]
        self.values = self.input()["values"]
        self.labels = self.input()["labels"]

    def input(self):
        capacity = 0
        num_of_classes = 0
        weights = []
        values = []
        labels = []

        f = open("input.txt", "r")
        lines = f.readlines()
        capacity = int(lines[0])
        num_of_classes = int(lines[1])
        weights = [float(i) for i in lines[2].split(',')]
        values = [int(i) for i in lines[3].split(',')]
        labels = [int(i) for i in lines[4].split(',')]
        # Sort the input
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                if values[i] / weights[i] < values[j] / weights[j]:
                    temp = values[i]
                    values[i] = values[j]
                    values[j] = temp

                    temp = weights[i]
                    weights[i] = weights[j]
                    weights[j] = temp

                    temp = labels[i]
                    labels[i] = labels[j]
                    labels[j] = temp
        return {
            "capacity": capacity,
            "num_of_classes": num_of_classes,
            "weights": weights,
            "values": values,
            "labels": labels
        }
class BeamSearch:
    listState: []
    def __init__(self):
        self.listState = []

    def genInitState(self, problem: Knapsack):
        for n in range(W):
            state = []
            for i in range(len(problem.values)):
                value = random.randint(0, 1)
                state.append(value)
            self.listState.append(state)

    def fitness(self, problem: Knapsack, state: []):
        totalVal = 0
        totalWeight = 0
        classCheck = set()
        for i in range(len(state)):
            if state[i] == 1:
                totalVal += problem.values[i]
                totalWeight += problem.weights[i]
                classCheck.add(problem.labels[i])

        if totalWeight > problem.capacity or len(classCheck) != problem.num_of_classes:
            return 0
        return -totalVal
    def genSuccessors(self, problem: Knapsack, state: []):
        successorsList = []
        for i in range(len(state)):
            successor = state.copy()
            successor[i] = 1 - successor[i]
            successorsList.append(successor)
        return successorsList

    def run(self, problem: Knapsack):
        self.genInitState(problem)
        q = PriorityQueue()
        n = 0
        global maxVal
        global resState
        global exploredSet
        global recursiveCount
        global recursiveCalls

        if recursiveCount == recursiveCalls:
            return maxVal, resState
        else:
            recursiveCount += 1

        for state in self.listState:
            print(self.fitness(problem, state), state)
            q.put((self.fitness(problem, state), state))
            exploredSet[str(state)] = True
            if maxVal < -self.fitness(problem, state):
                maxVal = -self.fitness(problem, state)
                resState = state.copy()

        while n < runs and not q.empty():
            initStates = []
            for i in range(W):
                fit,state = q.get()
                initStates.append((fit,state))
            q = PriorityQueue()

            for item in initStates:
                fit,state = item
                successors = self.genSuccessors(problem, state)
                for sc in successors:
                    if str(sc) not in exploredSet:
                        f = self.fitness(problem,sc)
                        if maxVal < -f:
                            maxVal = -f
                            resState = sc.copy()
                        exploredSet[str(sc)]=True
                        if f <= fit:
                            q.put((f,sc))

            n += 1
        self.run(problem)

if __name__ == "__main__":
    problem = Knapsack()
    search = BeamSearch()
    print(search.run(problem))
