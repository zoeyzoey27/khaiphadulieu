/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekapro;

import java.io.BufferedWriter;
import java.io.FileWriter;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Debug;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author Admin
 */
public class MyNaiveBayerModel extends MyKnowledgeModel {
    NaiveBayes nbayes;

    public MyNaiveBayerModel() {
    }

    public MyNaiveBayerModel(String filename, String m_opts, String d_opts) throws Exception {
        super(filename, m_opts, d_opts);
    }
    
    public void buildNaveBayes (String filename) throws Exception{
        // đọc train set vào bộ nhớ
        setTrainSet(filename);
        this.trainset.setClassIndex(this.trainset.numAttributes()-1);
        //huấn luyện mô hình naiveBayes
        this.nbayes = new NaiveBayes();
        nbayes.buildClassifier(this.trainset);
    }
    
    public void evaluateNaivebayes(String filename) throws Exception{
        //đọc test set vào bộ nhớ
        setTestSet(filename);
        this.testset.setClassIndex(this.testset.numAttributes()-1);
        //đánh giá mô hình bằng 10-fold cross-validation
        Random rnd = new Debug.Random(1);
        int folds = 10;
        Evaluation eval = new Evaluation(this.trainset);
        eval.crossValidateModel(nbayes, this.testset, folds, rnd);
        System.out.println(eval.toSummaryString(
                "\nKết quả đánh giá mô hình 10-fold cross-validation\n",false));
       
    }
    public void predictClassLabel(String fileIn, String fileOut) throws Exception{
        //đọc dữ liệu cần dự đoán vào bộ nhớ
        DataSource ds = new DataSource(fileIn);
        Instances unlabel = ds.getDataSet();
        unlabel.setClassIndex(unlabel.numAttributes()-1);
        //dự đoán classlabel cho từng instance
        for (int i=0; i<unlabel.numInstances(); i++){
            double predict = nbayes.classifyInstance(unlabel.instance(i));
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
        return this.nbayes.toString(); //To change body of generated methods, choose Tools | Templates.
    }
    
    
}
