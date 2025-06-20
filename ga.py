import os
import glob
import time
import random
import numpy as np
import pandas as pd
import argparse

from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from deap import base, creator, tools, algorithms

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, default="./DATA_Updated")
parser.add_argument('--kernel', type=str, default="rbf")
parser.add_argument('--n_gen', type=int, default=10)
parser.add_argument('--n_pop', type=int, default=50)
parser.add_argument('--cxpb', type=float, default=0.5, help="CX rate at each tournement")
parser.add_argument('--mutpb', type=float, default=0.1, help="Mut rate at each tournement")
args = parser.parse_args(args=[])

# read/combine csv
def load_data_from_csv(folder_path):
    all_files = glob.glob(os.path.join(folder_path, "*.xls"))
    feature_list = []
    label_list = []
    label_encoder = LabelEncoder()

    ## Get label name from the first file only
    label_names = set()
    df = pd.read_csv(all_files[0])
    label_names.update(df.iloc[:, 1].unique())
    label_encoder.fit(sorted(label_names))
    label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_) + 1))
    print("Label mapping:", label_map)

    ## read all files
    for file in all_files:
        df = pd.read_csv(file).iloc[:, 1:]   ## remove the first colume (subject label)
        y = label_encoder.transform(df.iloc[:, 0]) + 1
        X = df.iloc[:, 1:].values
        feature_list.append(X)
        label_list.append(y)
        print(f"Loading {file}:  {df.shape}")

    X_all = np.vstack(feature_list)
    y_all = np.hstack(label_list)
    print(f"Loaded {len(all_files)} csv: X.shape = {X_all.shape}, y.shape = {y_all.shape} in total.")
    return X_all, y_all, label_encoder

def evaluate(individual):
    global X_global, y_global  # Use global variables for data access

    if sum(individual) == 0:
        return (0.0,)
    #selected = [i for i, bit in enumerate(individual) if bit == 1]
    selected = np.atleast_1d(individual).nonzero()[0]   ## np.where(x==1) 
    X_sel = X_global[:, selected]
    #clf = SVC(kernel='linear', C=1.0)
    clf = SVC(kernel=args.kernel, gamma='scale', C=1.0)
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    scores = cross_val_score(clf, X_sel, y_global, cv=cv, scoring='accuracy', n_jobs=-1)
    #print(scores)
    return (scores.mean(),)

#
def run_ga(X, y, n_gen, pop_size):
    global X_global, y_global
    X_global, y_global = X, y

    n_features = X.shape[1]
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)   #flag to select the feature
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features) ## arr[n]=0/1
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.90) # gene mutation rate if varAnd decide to mutate this individual
    toolbox.register("select", tools.selTournament, tournsize=4)

    pop = toolbox.population(n=pop_size)
    print(f"\nInit population with np.shape(pop) = {np.shape(pop)}")

    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)
    stats.register("std", np.std)

    print(f"GA start with {n_gen} generations and {pop_size} pop size ...")   ## start GA
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "avg", "std", "min", "max"]
    for gen in range(n_gen):
        print(f"  Gen {gen:2d} ... ", end='')

        t0 = time.time()
        offspring = algorithms.varAnd(pop, toolbox, cxpb=args.cxpb, mutpb=args.mutpb)
        invalid_ind  = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop = toolbox.select(offspring, k=len(pop))
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        t1=time.time()-t0
        print(f"  avg={record['avg']:.4f} std={record['std']:.4g} min={record['min']:.4f} max={record['max']:.4f} in {t1:.2f} secs")

    best = hof[0]
    selected_indices = np.atleast_1d(best).nonzero()[0]    # np.where(best == 1)
    print(f"GA done. Selected {len(selected_indices)} / {n_features} features\n")
    return selected_indices, logbook

#
def evaluate_selected_features(X, y, selected_indices, label_encoder):
    print("SVC evaluating selected feature ... ", end='')

    X_sel = X[:, selected_indices]
    X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.2)

    t0 = time.time()
    ##clf = SVC(kernel='linear', C=1.0)
    clf = SVC(kernel='rbf', gamma='scale', C=1.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"in {time.time()-t0:.2f} secs.")

    print(" Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=1))

    print(" Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(" Accuracy:          ", accuracy_score(y_test, y_pred))
    print(" Precision (macro): ", precision_score(y_test, y_pred, average='macro', zero_division=1))
    print(" Recall (macro):    ", recall_score(y_test, y_pred, average='macro'))
    print(" F1-score (macro):  ", f1_score(y_test, y_pred, average='macro'))

#
if __name__ == "__main__":

    X, y, label_encoder = load_data_from_csv(args.data_folder)
    #unique, counts = np.unique(y, return_counts=True)   ## check labels balanced.
    #print(unique, counts)

    selected_features, log = run_ga(X, y, n_gen=args.n_gen, pop_size=args.n_pop)
    evaluate_selected_features(X, y, selected_features, label_encoder)
