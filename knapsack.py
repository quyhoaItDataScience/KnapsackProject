import random
from typing import List
import heapq
import time
import psutil


# Get the real state as we sort the item
def getRealState(realPos: list, state: list):
    realState = [0] * len(state)
    for i in range(len(state)):
        if state[i] == 1:
            realState[realPos[i]] = 1
    return realState


# Class input the file and run brute force and branch and bound algorithms
class Knapsack:
    realPos = []
    testCase = ""

    def __init__(self, case: str) -> None:
        self.testCase = case
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

        # Open the file
        f = open("input_" + self.testCase + ".txt", "r")
        lines = f.readlines()
        capacity = int(lines[0])
        num_of_classes = int(lines[1])
        weights = [int(i) for i in lines[2].split(',')]
        values = [int(i) for i in lines[3].split(',')]
        labels = [int(i) for i in lines[4].split(',')]
        self.realPos = [int(i) for i in range(len(labels))]
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

                    temp = self.realPos[i]
                    self.realPos[i] = self.realPos[j]
                    self.realPos[j] = temp
        return {
            "capacity": capacity,
            "num_of_classes": num_of_classes,
            "weights": weights,
            "values": values,
            "labels": labels
        }

    # Brute Force algorithm
    def brute(self, fout):
        n = len(self.values)
        length_unique_labels = len(set(self.labels))
        res = 0
        curLabels = []
        arrBits = [0] * n
        resBits = [0] * n

        def run(capacity, curValue, i):
            nonlocal res, arrBits, resBits
            # If we have examined n items
            if i == n or capacity == 0:
                # Save the best value
                if curValue > res and len(set(curLabels)) == length_unique_labels:
                    res = curValue
                    resBits = arrBits.copy()
                return

            # include the i-th item
            if self.weights[i] <= capacity:
                curLabels.append(self.labels[i])
                arrBits[i] = 1
                run(capacity - self.weights[i], curValue + self.values[i], i + 1)
                arrBits[i] = 0
                curLabels.remove(self.labels[i])

            # exclude the i-th item
            run(capacity, curValue, i + 1)

        run(self.capacity, 0, 0)
        fout.write(str(res) + '\n')
        fout.write(', '.join(str(v) for v in getRealState(self.realPos, resBits)))

    # Branch and Bound algorithm
    def branch_and_bound(self, fout):
        n = len(self.values)
        length_unique_labels = len(set(self.labels))
        res = 0
        curLabels = []
        arrBits = [0] * n
        resBits = [0] * n

        # calculate the branch  to determine whether to prune or not
        # Assume we can divide big item into small item
        # Each small item has 1 weight and value[i]/weight[i]
        def calculateBranch(capacity: float, pos: int) -> int:
            listItem = []
            weight = self.weights.copy()
            value = self.values.copy()
            labels = self.labels.copy()
            #Best value[i]/weight[i] is considered first
            for i in range(len(weight)):
                listItem.append((weight[i], value[i], labels[i], value[i] / weight[i]))

            # Calculate the branch
            Sum = 0
            i = 0
            while capacity > 0 and i < n:
                # If current capacity has enough space for big item then get all
                if capacity > weight[i]:
                    Sum = Sum + listItem[i][3] * listItem[i][0]
                    capacity = capacity - listItem[i][0]
                # Else get enough capacity from that item
                else:
                    Sum = Sum + listItem[i][3] * capacity
                    break
                i = i + 1
            return Sum

        # Run the same as Brute Force
        def run(capacity, curValue, i):
            nonlocal res, arrBits, resBits
            if i == n or capacity == 0:
                if curValue > res and len(set(curLabels)) == length_unique_labels:
                    res = curValue
                    resBits = arrBits.copy()
                return

            if curValue + calculateBranch(capacity, i) < res:
                return
            if self.weights[i] <= capacity:
                curLabels.append(self.labels[i])
                arrBits[i] = 1
                run(capacity - self.weights[i], curValue + self.values[i], i + 1)
                arrBits[i] = 0
                curLabels.remove(self.labels[i])

            # exclude the i-th item
            run(capacity, curValue, i + 1)

        run(self.capacity, 0, 0)
        fout.write(str(res) + '\n')
        fout.write(', '.join(str(v) for v in getRealState(self.realPos, resBits)))


#Local Beam Search
BeamSize = 5  # number of nodes taken after a turn of generation
runs = 200
exploredSet = dict()
maxVal = 0
resState = []
recursiveCount = 0
recursiveCalls = 10
flag = False


class BeamSearch:
    listState: []

    def __init__(self):
        self.listState = []

    # Generate init state to search
    def genInitState(self, problem: Knapsack):
        for n in range(BeamSize):
            state = []
            for i in range(len(problem.values)):
                value = random.randint(0, 1)
                state.append(value)
            self.listState.append(state)

    # Calculate the state value
    def fitness(self, problem: Knapsack, state):
        totalVal = 0
        totalWeight = 0
        classCheck = set()
        for i in range(len(state)):
            if state[i] == 1:
                totalVal += problem.values[i]
                totalWeight += problem.weights[i]
                classCheck.add(problem.labels[i])

        # If oversize the bag or not enough class then return 0
        if totalWeight > problem.capacity or len(classCheck) != problem.num_of_classes:
            return 0
        # Else return the value
        # Inverse the value to push into heap
        return -totalVal

    # Generate successors for each state
    def genSuccessors(self, problem: Knapsack, state: []):
        successorsList = []
        # Flip each bit once
        for i in range(len(state)):
            successor = state.copy()
            successor[i] = 1 - successor[i]
            successorsList.append(successor)
        return successorsList

    def run(self, problem: Knapsack):
        # Prepare to run
        self.genInitState(problem)
        # Init priority queue
        q = []
        heapq.heapify(q)
        n = 0
        global maxVal
        global resState
        global exploredSet
        global recursiveCount
        global recursiveCalls
        global flag

        # If we have reach number of restart then return best value
        if recursiveCount == recursiveCalls:
            # if not flag:
            #     fout.write(str(maxVal) + '\n')
            #     fout.write(', '.join(str(v) for v in getRealState(problem.realPos, resState)))
            #     flag = True
            return
        else:
            recursiveCount += 1

        # Push init states into heap
        for state in self.listState:
            heapq.heappush(q, (self.fitness(problem, state), state))
            exploredSet[str(state)] = True
            if maxVal < -self.fitness(problem, state):
                maxVal = -self.fitness(problem, state)
                resState = state.copy()

        # Loop through each search
        # Run the same as hill climbing
        while n < runs and not len(q) == 0:
            initStates = []
            # Get BeamSize best state
            for i in range(BeamSize):
                if len(q) == 0:
                    self.run(problem)
                else:
                    fit, state = heapq.heappop(q)
                initStates.append((fit, state))

            # Empty the queue
            q = []
            # Generate successors from each current state
            for item in initStates:
                fit, state = item
                successors = self.genSuccessors(problem, state)
                for sc in successors:
                    f = self.fitness(problem, sc)
                    if str(sc) not in exploredSet and f <= fit:
                        if maxVal < -f:
                            maxVal = -f
                            resState = sc.copy()
                        exploredSet[str(sc)] = True
                        heapq.heappush(q, (f, sc))

            n += 1
        self.run(problem)

# Genetic Algorithm
MAX_GEN = 300
INIT_POPULATION = 500
CROSSOVER_RATE = 0.53
MUTATION_RATE = 0.013


class GeneticAlgorithm:
    population = []
    num_of_classes = 0
    capacity = 0
    weights = []
    values = []
    labels = []
    maxVal = 0
    maxRes = []
    realPos = []

    def __init__(self, case):
        ks = Knapsack(case)
        source = ks.input()
        self.weights = source["weights"]
        self.values = source["values"]
        self.labels = source["labels"]
        self.num_of_class = source["num_of_classes"]
        self.capacity = source["capacity"]
        self.realPos=ks.realPos

    # generate init population
    def generateInitalPopulation(self):
        count = 0
        # Choose random bits
        while count < INIT_POPULATION:
            count = count + 1
            bits = [
                random.choice([0, 1])
                for _ in self.weights
            ]
            self.population.append(bits)

    # Calculate the fitness like local beam search
    def calculateFitness(self, bits: list[int]):
        totalWeight = 0
        totalValue = 0
        classCheck = set()
        for i in range(len(self.weights)):
            
            totalWeight = totalWeight + bits[i] * self.weights[i]
            totalValue = totalValue + bits[i] * self.values[i]
            classCheck.add(self.labels[i])

        if len(classCheck) != self.num_of_class or totalWeight > self.capacity:
            return 0
        return totalValue

    # Selection using tournament to determine individual for next gen
    def selection(self):
        random.shuffle(self.population)
        parents = []
        i = 0
        while i < len(self.population):
            if i + 1 == len(self.population):
                break
            if self.calculateFitness(self.population[i]) > self.calculateFitness(self.population[i + 1]):
                parents.append(self.population[i])
            else:
                parents.append(self.population[i + 1])
            i = i + 2
        self.population = parents.copy()

    # Crossover exchange each other parts from the father and mother
    def crossover(self, father: list[int], mother: list[int]):
        n = len(father)
        pos = random.randint(2, n - 2)
        child1 = father[:pos] + mother[pos:]
        child2 = mother[:pos] + father[pos:]
        return child1, child2

    # Crossover this gen to generate next gen
    def crossoverMethod(self):
        children = []
        i = 0
        while i < len(self.population):
            if i + 1 == len(self.population):
                break
            child1, child2 = self.crossover(self.population[i], self.population[i + 1])
            if self.calculateFitness(child1) > self.maxVal:
                self.maxVal = self.calculateFitness(child1)
                self.maxRes = child1

            if self.calculateFitness(child2) > self.maxVal:
                self.maxVal = self.calculateFitness(child2)
                self.maxRes = child2

            children.append(child1)
            children.append(child2)
            i = i + 2
        for child in children:
            self.population.append(child)

    # When mutation happens, choose a random bit and flip
    def mutate(self, individual: list[int]) -> list[int]:
        pos = random.randint(0, len(individual) - 1)
        individual[pos] = 1 - individual[pos]
        return individual

    def mutatationMethod(self):
        for i in range(len(self.population)):
            if random.random() < MUTATION_RATE:
                self.population[i] = self.mutate(self.population[i])

    def run(self):
        self.generateInitalPopulation()
        # Loop through each generation
        for i in range(MAX_GEN):
            self.selection()
            self.crossoverMethod()
            self.mutatationMethod()

        for individual in self.population:
            self.maxVal = max(self.maxVal, self.calculateFitness(individual))
        # fout.write(str(self.maxVal) + '\n')
        # fout.write(', '.join(str(v) for v in getRealState(self.realPos, self.maxRes)))


class RunAlgorithm:
    def run(self):
        print("1. Brute Force")
        print("2. Branch and Bound")
        print("3. Local Beam Search")
        print("4. Genetic Algorithm")
        choice = int(input("Select an algorithm: "))
        testCase = input("Select a test case (choose from 1-10): ")
        ks = Knapsack(testCase)
        fout = open("output_" + testCase + ".txt", "w")
        if choice == 1:
            ks.brute(fout)
        elif choice == 2:
            ks.branch_and_bound(fout)
        elif choice == 3:
            beamSearch = BeamSearch()
            beamSearch.run(ks, fout)
        else:
            for i in range(1, 11): 
                try: 
                    Gen = GeneticAlgorithm(str(i))
                    Gen.run()
                except:
                    print("sth wrong") 
            # Gen = GeneticAlgorithm('2')
            # Gen.run()

# runAlgorithm = RunAlgorithm()
# runAlgorithm.run()

start = time.time()
ks = Knapsack('10')
Gen = BeamSearch()
Gen.run(ks)
end = time.time()
memory = psutil.Process().memory_info().rss / (1024 * 1024)
print(f"{end - start}s", memory)
    
