/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekapro;

import weka.classifiers.trees.J48;

/**
 *
 * @author Admin
 */
public class WekaPro {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        // TODO code application logic here
//        MyKnowledgeModel model = new MyKnowledgeModel("D:\\Doc\\Weka-3-8-6\\data\\iris.arff");
//        System.out.println(model);
//        model.saveData("D:\\Doc\\data\\iris.arff");
//        model.saveDataToCSV("D:\\Doc\\data\\iris_csv.csv");
//          MyAprioriModel model = new MyAprioriModel(
//             "D:\\Doc\\Weka-3-8-6\\data\\weather.numeric.arff",
//              "-N 10 -T 0 -C 0.9 -D 0.05 -U 1.0 -M 0.1 -S -1.0 -c -1",
//              "-R 2-3"    
//          );
//          model.mineAssociationRules();
//          System.out.println(model);
//        
//         MyDecisionTreeModel model = new MyDecisionTreeModel(
//                 "D:\\Doc\\Weka-3-8-6\\data\\iris.arff", "-C 0.25 -M 2", null);
//         model.buildDecisionTree();
//         model.evaluateDecisionTree();
//         System.out.println(model);
////         model.saveModel("D:\\Doc\\data\\model\\decisiontree.model", model.tree);
//         model.tree = (J48)model.loadModel("D:\\\\Doc\\\\data\\\\model\\\\decisiontree.model");
//         model.predictClassLabel(model.testset);
//        MyNaiveBayerModel model = new MyNaiveBayerModel();
//        model.buildNaveBayes("D:\\Doc\\data\\iris_train.arff");
//        model.evaluateNaivebayes("D:\\Doc\\data\\iris_test.arff");
//        model.predictClassLabel("D:\\Doc\\data\\iris_unlabel.arff", "D:\\Doc\\data\\iris_predict_nb.arff");
//        System.out.print(model);
//        MyKNNModel model = new MyKNNModel("",
//                "-K 3 -W 0 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -R first-last\\\"\"", null);
//        model.buildkNN("D:\\Doc\\data\\iris_train.arff");
//        model.evaluatekNN("D:\\Doc\\data\\iris_test.arff");
//        model.predictClassLabel("D:\\Doc\\data\\iris_unlabel.arff", "D:\\Doc\\data\\iris_predict_knn.arff");
//        System.out.print(model);
        MyKMeansModel model = new MyKMeansModel("", null, null);
        model.buildKMeansModel("D:\\Doc\\data\\iris_cluster_train.arff");
        model.predictCluster("D:\\Doc\\data\\iris_cluster_unlabel.arff");
        System.out.println("Finished");
    }
    
}
