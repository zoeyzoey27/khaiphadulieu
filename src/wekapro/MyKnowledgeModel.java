/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekapro;

import java.io.File;
import java.io.IOException;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.filters.unsupervised.instance.Resample;

/**
 *
 * @author Admin
 */
public class MyKnowledgeModel {
    DataSource source;
    Instances dataset;
    String[] model_options;
    String[] data_options;
    Instances trainset;
    Instances testset;

    public MyKnowledgeModel() {
    }
    
    public MyKnowledgeModel(String filename, String m_opts, String d_opts) throws Exception {
        if (!filename.isEmpty()){
            this.source = new DataSource(filename);
            this.dataset = source.getDataSet();
        }
        if (m_opts != null){
            this.model_options = weka.core.Utils.splitOptions(m_opts);
        }
        if (d_opts != null){
            this.data_options = weka.core.Utils.splitOptions(d_opts);
        }
    }
    
    public Instances removeData(Instances originalData) throws Exception{
        Remove remove = new Remove();
        remove.setOptions(data_options);
        remove.setInputFormat(originalData);
        return Filter.useFilter(originalData, remove);
    }
    
    public Instances convertData(Instances originalData) throws Exception{
        NumericToNominal n2n = new NumericToNominal();
        n2n.setOptions(data_options);
        n2n.setInputFormat(originalData);
        return Filter.useFilter(originalData, n2n);
        
    }
    //xuất dữ liệu ra file
    public void saveData(String filename) throws IOException{
        ArffSaver outData = new ArffSaver ();
        outData.setInstances(this.dataset);
        outData.setFile(new File(filename));
        outData.writeBatch();
        System.out.println("Finished");
    }
    //chuyển đổi sang định dạng csv
    public void saveDataToCSV (String filename) throws IOException{
        CSVSaver outData = new CSVSaver();
        outData.setInstances(this.dataset);
        outData.setFile(new File (filename));
        outData.writeBatch();
        System.out.println("Converted");
    }
    //chia train test
    public Instances divideTrainTest(Instances originalSet,
            double percent, boolean isTest) throws Exception{
        RemovePercentage rp = new RemovePercentage();
        rp.setPercentage(percent);
        rp.setInvertSelection(isTest);
        rp.setInputFormat(originalSet);
        return Filter.useFilter(originalSet, rp);
    }
    //chia train test bg resample
    public Instances divideTrainTestR(Instances originalSet,
            double percent, boolean isTest) throws Exception{
        Resample rs = new Resample();
        rs.setNoReplacement(true);
        rs.setSampleSizePercent(percent);
        rs.setInvertSelection(isTest);
        rs.setInputFormat(originalSet);
        return Filter.useFilter(originalSet, rs);
    }
    
    public void saveModel(String filename, Object model) throws Exception{
        weka.core.SerializationHelper.write(filename,model);
    }
    public Object loadModel(String filename) throws Exception{
        return weka.core.SerializationHelper.read(filename);
    }
    
    public void setTrainSet(String filename) throws Exception{
        DataSource trainSource = new DataSource(filename);
        this.trainset = trainSource.getDataSet();
    }
    public void setTestSet(String filename) throws Exception{
        DataSource testSource = new DataSource(filename);
        this.testset = testSource.getDataSet();
    }
 
    @Override
    public String toString() {
        return dataset.toSummaryString();
    }
    
}
