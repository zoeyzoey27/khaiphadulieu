/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekapro;

import weka.classifiers.Evaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author Admin
 */
public class MyKMeansModel extends MyKnowledgeModel {
    SimpleKMeans kmeans;
    Evaluation eval;

    public MyKMeansModel() {
    }

    public MyKMeansModel(String filename, String m_opts, String d_opts) throws Exception {
        super(filename, m_opts, d_opts);
    }
    
    public void buildKMeansModel(String filename) throws Exception{
        //đọc train set vào bộ nhớ
        setTrainSet(filename);
        //thiết lập mô hình kmeans
        kmeans = new SimpleKMeans();
        kmeans.setNumClusters(3);
        kmeans.setDistanceFunction(new EuclideanDistance());
        kmeans.buildClusterer(trainset);
        //xuất thông số của mô hình ra màn hình
        System.out.println(kmeans);
    
    }
    
    public void predictCluster(String filename) throws Exception{
        //đọc dữ liệu vào bộ nhớ
        DataSource ds = new DataSource(filename);
        Instances unlabel = ds.getDataSet();
        //dự đoán cluster
        for (int i=0; i<unlabel.numInstances(); i++){
            double predict = kmeans.clusterInstance(unlabel.instance(i));
            System.out.println("Instance "+i+" belongs to cluster" + predict);
        }
    }
}
