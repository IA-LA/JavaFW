/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka;

/**
 *
 * @author FJ
 */
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.core.FastVector;
import weka.core.Instances;

public class WekaTest {
    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;
        
        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }
        
        return inputReader;
    }
    
    public static Evaluation simpleClassify(Classifier model, Instances trainingSet, Instances testingSet) throws Exception {
        Evaluation validation = new Evaluation(trainingSet);
        
        model.buildClassifier(trainingSet);
        validation.evaluateModel(model, testingSet);
        
        return validation;
    }
        
    public static ClusterEvaluation simpleClusterify(Clusterer model, Instances trainingSet, Instances testingSet) throws Exception {
        
        ClusterEvaluation validation =  new ClusterEvaluation();
                
        model.buildClusterer(trainingSet);
        model.numberOfClusters();
        validation.evaluateClusterer(testingSet);
        
        return validation;
    }
    
    public static double calculateAccuracy(FastVector predictions) {
        double correct = 0;
        
        for (int i = 0; i < predictions.size(); i++) {
            NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
            if (np.predicted() == np.actual()) {
                correct++;
            }
        }
        
        return 100 * correct / predictions.size();
    }
    
    public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds) {
        Instances[][] split = new Instances[2][numberOfFolds];
        
        for (int i = 0; i < numberOfFolds; i++) {
            split[0][i] = data.trainCV(numberOfFolds, i);
            split[1][i] = data.testCV(numberOfFolds, i);
        }
        
        return split;
    }
    
    public static void main(String[] args) throws Exception {
        // I've commented the code as best I can, at the moment.
        // Comments are denoted by "//" at the beginning of the line.
        
        String file = "iris.arff";
        
        BufferedReader datafile = readDataFile(file);
        
        // Data Classifiers
        Instances data = new Instances(datafile);
        data.setClassIndex(data.numAttributes() - 1);
        
        // Choose a type of validation split
        Instances[][] split = crossValidationSplit(data, 10);
        
        // Separate split into training and testing arrays
        Instances[] trainingSplits = split[0];
        Instances[] testingSplits  = split[1];
        
        // Choose a set of classifiers
        Classifier[] models = {     new J48(),
                                    new weka.classifiers.bayes.BayesNet(),
                                    new weka.classifiers.bayes.NaiveBayes(),
                                    new weka.classifiers.functions.MultilayerPerceptron(),
                                    new PART(),
                                    new DecisionTable(),
                                    new OneR(),
                                    new DecisionStump() };
        
        // Run for each classifier model
        for(int j = 0; j < models.length; j++) {

            // Collect every group of predictions for current model in a FastVector
            FastVector predictions = new FastVector();
            
            // For each training-testing split pair, train and test the classifier
            for(int i = 0; i < trainingSplits.length; i++) {
                Evaluation validation = simpleClassify(models[j], trainingSplits[i], testingSplits[i]);
                predictions.appendElements(validation.predictions());
                
                // Uncomment to see the summary for each training-testing pair.
                // System.out.println(models[j].toString());
            }
            
            // Calculate overall accuracy of current classifier on all splits
            double accuracy = calculateAccuracy(predictions);
            
            // Print current classifier's name and accuracy in a complicated, but nice-looking way.
            System.out.println(models[j].getClass().getSimpleName() + ": " + String.format("%.2f%%", accuracy) + "\n============================");
        }
        

        BufferedReader datafileC = readDataFile(file);
                
        // Data Clusterers
        Instances dataC = new Instances(datafileC);
        dataC.deleteAttributeAt(dataC.numAttributes() - 1);
        
        // Choose a type of validation split
        Instances[][] splitC = crossValidationSplit(dataC, 10);
        
        // Separate split into training and testing arrays
        Instances[] trainingSplitsC = splitC[0];
        Instances[] testingSplitsC  = splitC[1];
        
        // Choose a set of clusterers
        Clusterer[] modelsC = {
                                    new weka.clusterers.SimpleKMeans(),
                                    new weka.clusterers.FarthestFirst(),
                                    new weka.clusterers.FilteredClusterer(),
                                    new weka.clusterers.HierarchicalClusterer(),
                                    new weka.clusterers.MakeDensityBasedClusterer()
                                };
        
        // Run for each clusterifier model
        for(int j = 0; j < modelsC.length; j++) {

            // Collect every group of predictions for current model in a FastVector
            String[] predictionsC = new String[modelsC.length*trainingSplits.length];
            
            // For each training-testing split pair, train and test the classifier
            for(int i = 0; i < trainingSplits.length; i++) {
                ClusterEvaluation validation = simpleClusterify(modelsC[j], trainingSplitsC[i], testingSplitsC[i]);
                //predictions.appendElements(validation.predictions());
                predictionsC[i] = validation.clusterResultsToString();
                // Uncomment to see the summary for each training-testing pair.
                // System.out.println(modelos[j].toString());
            }
            
            // Calculate overall accuracy of current classifier on all splits
            //double accuracy = calculateAccuracy(predictions);
            
            // Print current classifier's name and accuracy in a complicated, but nice-looking way.
            System.out.println(modelsC[j].getClass().getSimpleName() + ": " + predictionsC[j] + "\n=====================");
        }        
    }
}
