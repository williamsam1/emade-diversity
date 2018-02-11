import operator
import matplotlib.pyplot as plt
import pandas as pd
from deap import base
from deap import creator
from deap import gp
from sklearn.model_selection import train_test_split
from selection_methods import *


data = pd.read_csv('train.csv')
data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
data['Sex'] = data['Sex'].map(lambda x: hash(x))
data['Embarked'] = data['Embarked'].map(lambda x: hash(x))
data.fillna(value=-10, axis=1, inplace=True)
(train_data, test_data) = train_test_split(data, test_size=0.2, random_state=67854895)
x_train = train_data.drop(['PassengerId', 'Survived'], axis=1)
y_train = train_data['Survived']
x_test = test_data.drop(['PassengerId', 'Survived'], axis=1)
y_test = test_data['Survived']


def accuracy(classifier, x, y) -> float:
    """Function that returns the accuracy of a classifier"""
    num_correct = 0
    for i in range(len(x_test)):
        x_in = x.values[i]
        pred = 1 if classifier(x_in[0], x_in[1], x_in[2], x_in[3], x_in[4], x_in[5], x_in[6]) > 0 else 0
        correct = y.values[i]
        if pred == correct:
            num_correct += 1
    return 0 if len(x_test) == 0 else num_correct / len(x_test)


def false_rates_old(classifier, x, y) -> (float, float):
    """Returns the false positive and false negative rate of a classifier"""
    num_negative = 0
    incorrect_negative = 0
    num_positive = 0
    incorrect_positive = 0
    for i in range(len(x_test)):
        x_in = x.values[i]
        pred = 1 if classifier(x_in[0], x_in[1], x_in[2], x_in[3], x_in[4], x_in[5], x_in[6]) > 0 else 0
        correct = y.values[i]
        if correct == 0:
            num_negative += 1
            if pred != correct:
                incorrect_negative += 1
        else:
            num_positive += 1
            if pred != correct:
                incorrect_positive += 1
    fp = 0
    if num_negative != 0:
        fp = incorrect_negative / num_negative
    fn = 0
    if num_positive != 0:
        fn = incorrect_positive / num_positive
    return fp, fn


def false_rates(classifier, x, y) -> (float, float):
    """Returns the false positive and false negative rate of a classifier"""

    num_negative = y.values.tolist().count(0)
    num_positive = len(y.values) - num_negative

    pred = [1 if classifier(x_in[0], x_in[1], x_in[2], x_in[3], x_in[4], x_in[5], x_in[6]) > 0 else 0 for x_in in x.values]

    zipped = list(zip(pred, y.values))
    incorrect_negative = zipped.count((1, 0))
    incorrect_positive = zipped.count((0, 1))

    fp = 0 if num_negative == 0 else incorrect_negative / num_negative
    fn = 0 if num_positive == 0 else incorrect_positive / num_positive

    return fp, fn

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)


def activation(x):
    return np.divide(1, 1 + np.exp(np.negative(x)))


def if_then_else(x, y, z):
    try:
        return y if x > 0 else z
    except:
        pass
    return [y if x > 0 else z for (x, y, z) in zip(x, y, z)]


def heaviside(x, y):
    return 0 if x < 0 else (y if x == 0 else 1)

pset = gp.PrimitiveSet("MAIN", arity=7)
pset.addPrimitive(np.add, arity=2)
pset.addPrimitive(np.subtract, arity=2)
pset.addPrimitive(np.multiply, arity=2)
pset.addPrimitive(np.negative, arity=1)
pset.addPrimitive(np.sin, arity=1)
pset.addPrimitive(np.cos, arity=1)
pset.addPrimitive(np.tan, arity=1)
pset.addPrimitive(heaviside, arity=2)
pset.addPrimitive(activation, arity=1)
pset.addPrimitive(if_then_else, arity=3)


toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evalPerformance(individual, pset):
    func = gp.compile(expr=individual, pset=pset)
    return false_rates(func, x_train, y_train)

toolbox.register("evaluate", evalPerformance, pset=pset)


toolbox.register("select", mynsga)


toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("vary", varOr)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def pareto_dominance(ind1, ind2):
    not_equal = False
    for value_1, value_2 in zip(ind1.fitness.values, ind2.fitness.values):
        if value_1 > value_2:
            return False
        elif value_1 < value_2:
            not_equal = True
    return not_equal


NGEN = 100
MU = 250
LAMBDA = 250
CXPB = 0.5
MUTPB = 0.2


pop = toolbox.population(n=500)
hof = tools.ParetoFront()
stats = tools.Statistics(lambda ind: ind.fitness)
stats.register('avg_fitness', lambda fv: np.mean([f.values for f in fv], axis=0))
stats.register('avg_crowding_dist', lambda fv: np.mean([x for x in [f.crowding_dist if 'crowding_dist' in dir(f) else 0
                                                                    for f in fv] if x != float("inf")]))
stats.register('fitness_variance', lambda fv: np.linalg.norm(np.std([f.values for f in fv], axis=0)))

pop, logbook, geno_metric = evolve(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats, hof)


def classify(fun, x_in):
    return 1 if fun(x_in[0], x_in[1], x_in[2], x_in[3], x_in[4], x_in[5], x_in[6]) > 0 else 0

compiled = [gp.compile(expr=x, pset=pset) for x in hof]


test_data = pd.read_csv('test.csv')
test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data['Sex'] = test_data['Sex'].map(lambda x: hash(x))
test_data['Embarked'] = test_data['Embarked'].map(lambda x: hash(x))
test_data.fillna(value=-10, axis=1, inplace=True)

"""
output = []
for i in range(len(test_data)):
    row = {'PassengerId': 892 + i}
    if i % 100 == 0:
        print(i)
    for j in range(len(compiled)):
        row['Individual' + str(j + 1)] = classify(compiled[j], test_data.values[i])
    output.append(row)
pd.DataFrame(output).to_csv(path_or_buf='predictions.csv', index=False)
"""

gen, avg = logbook.select('gen', 'avg_fitness')
plt.plot(gen, [x[0] for x in avg], label='average false positive')
plt.plot(gen, [x[1] for x in avg], label='average false negative')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend(loc="upper left")
plt.show()

plt.plot(gen, geno_metric['add'], label='add primitive')
plt.plot(gen, geno_metric['heaviside'], label='heaviside primitive')
plt.plot(gen, geno_metric['subtract'], label='subtract primitive')
plt.plot(gen, geno_metric['activation'], label='activation primitive')
plt.xlabel("Generation")
plt.ylabel("Occurrence of Primitives")
plt.legend(loc="upper left")
plt.show()

plt.plot(gen, geno_metric['avg_length'])
plt.xlabel("Generation")
plt.ylabel("Average Length")
plt.legend(loc="upper left")
plt.show()

crowding = logbook.select('fitness_variance')
plt.plot(gen, crowding)
plt.xlabel("Generation")
plt.ylabel("Norm of Variance in Fitness")
plt.legend(loc="upper left")
plt.show()

crowding = logbook.select('avg_crowding_dist')
plt.plot(gen, crowding)
plt.xlabel("Generation")
plt.ylabel("Average Crowding Distance")
plt.legend(loc="upper left")
plt.show()

fitness_1 = [ind.fitness.values[0] for ind in hof]
fitness_2 = [ind.fitness.values[1] for ind in hof]
pop_1 = [ind.fitness.values[0] for ind in pop]
pop_2 = [ind.fitness.values[1] for ind in pop]


plt.scatter(pop_1, pop_2, color='b')
plt.scatter(fitness_1, fitness_2, color='r')
plt.plot(fitness_1, fitness_2, color='r', drawstyle='steps-post')
plt.xlabel("False Positive")
plt.ylabel("False Negative")
plt.title("Pareto Front")
plt.show()

f1 = np.array([1, 0] + fitness_1)
f2 = np.array([0, 1] + fitness_2)

print("Area Under Curve: %s" % (np.sum(np.abs(np.diff(f1))*f2[:-1])))

"""
NGEN = 1000
MU = 250
LAMBDA = 250
CXPB = 0.5
MUTPB = 0.2
start temp = 1e50
temp decrease = 2

MyNSGA:
0.03501516404742211

Annealed NSGA:
0.028535980148883373
NSGA:


"""

"""
    NSGA
    ----
    Gens: 1000
    AUC: 0.09098428453267163

    MyNSGA
    ------
    Gens: 1000
    AUC: 0.10931899641577061

    AnnealedNSGA
    ------------
    0.08312655086848636

    Gens: 400
    Temp: 0.001
    AUC: 0.06630824372759857

    Temp: 0.01
    AUC: 0.04576785221946512

    Temp: 0.05
    AUC: 0.0676867934932451

    Temp: 0.1
    AUC: 0.05100634132892198

    Temp: 1
    AUC: 0.05514199062586159

    Temp: 10
    AUC: 0.09746346843121036

    Gens: 1000
    Temp: 1.0e300
    AUC: 0.09911772814998622

    Temp: 1.0e180
    AUC: 0.03336090432864626

    Temp: 1.0e90
    AUC: 0.03818582850840915

    Temp: 1.0e45
    AUC: 0.029225255031706643

    Temp: 1.0e30
    AUC: 0.009374138406396471
    0.05845051006341329

    Temp: 1.0e20
    AUC: 0.012682657843948167

    Temp: 1.0e10
    AUC: 0.04769782189137028

    Temp: 1.0
    AUC: 0.01833471188309898

    Temp: 1.0e-10
    AUC: 0.02109181141439206
"""
