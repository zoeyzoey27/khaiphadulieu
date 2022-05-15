/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekapro;

import java.io.BufferedWriter;
import java.io.FileWriter;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author Admin
 */
public class MyKNNModel extends MyKnowledgeModel {
    IBk knn;
    Evaluation eval;

    public MyKNNModel(String filename, String m_opts, String d_opts) throws Exception {
        super(filename, m_opts, d_opts);
    }
    
    public void buildkNN(String filename) throws Exception{
        //đọc train set vào bộ nhớ
        setTrainSet(filename);
        this.trainset.setClassIndex(this.trainset.numAttributes()-1);
        //huấn luyên mô hình knn
        this.knn = new IBk();
        knn.setOptions(model_options);
        knn.buildClassifier(this.trainset);
    }
    
    public void evaluatekNN(String filename) throws Exception{
        //đọc test set vào bộ nhớ
        setTestSet(filename);
        this.testset.setClassIndex(this.testset.numAttributes()-1);
        // đánh giá mô hình bằng 10-fold cross validation
        Random rnd = new Random(1);
        int folds = 10;
        eval = new Evaluation(this.trainset);
        eval.crossValidateModel(knn, this.testset, folds, rnd);
        System.out.println(eval.toSummaryString(
                "\nKết quả đánh giá mô hình 10-fold cross-validation", false));
    }
    
    public void predictClassLabel(String fileIn, String fileOut) throws Exception{
        //đọc dữ liệu cần dự đoán vào bộ nhớ
        DataSource ds = new DataSource(fileIn);
        Instances unlabel = ds.getDataSet();
        unlabel.setClassIndex(unlabel.numAttributes()-1);
        //dự đoán classLabel cho từng instance
        for (int i=0; i<unlabel.numInstances(); i++){
            double predict = knn.classifyInstance(unlabel.instance(i));
            unlabel.instance(i).setClassValue(predict);
        }
        //xuất kết quả ra fileout
        BufferedWriter outWriter = new BufferedWriter(new FileWriter(fileOut));
        outWriter.write(unlabel.toString());
        outWriter.newLine();
        outWriter.flush();
        outWriter.close();
    }

    @Override
    public String toString() {
        return this.knn.toString(); //To change body of generated methods, choose Tools | Templates.
    }
    
    
}
