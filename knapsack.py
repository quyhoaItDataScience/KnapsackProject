import heapq
import os
import random
dir_path = os.path.dirname(os.path.realpath(__file__))

W = 5  # number of nodes taken after a turn of generation
runs = 200
exploredSet = dict()
maxVal = 0
resState = []
recursiveCount = 0
recursiveCalls = 10

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
        q = []
        heapq.heapify(q)
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
            heapq.heappush(q, (self.fitness(problem, state), state))
            exploredSet[str(state)] = True
            if maxVal < -self.fitness(problem, state):
                maxVal = -self.fitness(problem, state)
                resState = state.copy()

        while n < runs and not len(q) == 0:
            initStates = []
            for i in range(W):
                if len(q) == 0:
                    self.run(problem)
                else:
                    fit,state = heapq.heappop(q)
                initStates.append((fit,state))

            while not len(q) == 0:
                heapq.heappop(q)

            for item in initStates:
                fit = item[0]
                state=item[1]
                successors = self.genSuccessors(problem, state)
                for sc in successors:
                    f = self.fitness(problem, sc)
                    if str(sc) not in exploredSet and f <= fit:
                        if maxVal < -f:
                            maxVal = -f
                            resState = sc.copy()
                        exploredSet[str(sc)]=True
                        heapq.heappush(q,(f, sc))

            n += 1
        return maxVal, resState

problem = Knapsack()
search = BeamSearch()
print(search.run(problem))
