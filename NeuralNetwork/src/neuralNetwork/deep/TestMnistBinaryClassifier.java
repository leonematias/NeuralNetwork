/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralNetwork.deep;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Use Mnist data to train binary classifier to predict images of 0 and 1 digits
 * 
 * @author matias.leone
 */
public class TestMnistBinaryClassifier {
 
    private final static int MNIST_IMG_WIDTH = 20;
    private final static int MNIST_IMG_HEIGHT = 20;
    
    public static void main(String[] args) {
        new TestMnistBinaryClassifier().run();
    }
    
    private void run() {
        //Load data
        String xPath = "data/input_images.csv";
        String yPath = "data/input_classification.csv";
        List<ImageData> allImageData = loadImageData(xPath, yPath);
        
        //Pick 0 and 1 images (same amount from both)
        Map<Integer, List<ImageData>> imageMap = toMap(allImageData);
        List<ImageData> zeroImages = imageMap.get(0);
        List<ImageData> oneImages = imageMap.get(1);
        int minSampleSize = Math.min(zeroImages.size(), oneImages.size());
        List<ImageData> digitImages = new ArrayList<>(minSampleSize * 2);
        for (int i = 0; i < minSampleSize; i++) {
            digitImages.add(zeroImages.get(i));
            digitImages.add(oneImages.get(i));
        }
        
        //Split train and test set
        List<ImageData> trainSet = new ArrayList<>();
        List<ImageData> testSet = new ArrayList<>();
        splitDataSet(digitImages, 0.7f, trainSet, testSet);
        
        //Conver to X,Y matrices
        Matrix2 trainX = toX(trainSet);
        Matrix2 trainY = toY(trainSet);
        Matrix2 testX = toX(testSet);
        Matrix2 testY = toY(testSet);
        
        //Train binary classifier with layers [400, 25, 10, 1]
        DeepNeuralNetwork classifier = new DeepNeuralNetwork(new int[]{MNIST_IMG_WIDTH * MNIST_IMG_HEIGHT, 25, 10, 1}, 3000, 0.0075f);
        classifier.train(trainX, trainY, true);
        
        //Predict train and test set
        Matrix2 trainYpred = classifier.predict(trainX);
        Matrix2 testYpred = classifier.predict(testX);
        System.out.println("Train set accuracy: " + accuracy(trainY, trainYpred));
        System.out.println("Test set accuracy: " + accuracy(testY, testYpred));
        
        
    }
    
    private float accuracy(Matrix2 Y, Matrix2 Yhat) {
        int m = Y.cols();
        int tp = 0;
        for (int col = 0; col < m; col++) {
            float y = Y.get(0, col);
            float yhat = Yhat.get(0, col);
            if(y == yhat) {
                tp++;
            }
        }
        return (float)tp / m * 100f;
    }
    
    private PredictionStats predictionStats(Matrix2 Y, Matrix2 Yhat) {
        int m = Y.cols();
        PredictionStats stats = new PredictionStats();
        for (int col = 0; col < m; col++) {
            float y = Y.get(0, col);
            float yhat = Yhat.get(0, col);
            if(y == yhat) {
                stats.truePositives++;
            } else {
                stats.falsePositives++;
            }
        }
        stats.falsePositives = m - stats.truePositives;
        //stats.falseNegatives = 
        return stats;
    }
    
    private static class PredictionStats {
        public int truePositives;
        public int falsePositives;
        public int trueNegatives;
        public int falseNegatives;
        public float accuracy;
    }
    
    private void splitDataSet(List<ImageData> items, float splitPercentage, List<ImageData> out1, List<ImageData> out2) {
        for (ImageData item : items) {
            if(Math.random() < splitPercentage) {
                out1.add(item);
            } else {
                out2.add(item);
            }
        }
    }
    
    private int[] shuffleIndexArray(int n) {
        List<Integer> l = listOfIndexes(n);
        Collections.shuffle(l);
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = l.get(i);
        }
        return a;
    }
    
    private List<Integer> listOfIndexes(int n) {
        List<Integer> list = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            list.add(i, i);
        }
        return list;
    }
    
    private Matrix2 toX(List<ImageData> items) {
        List<Matrix2> list = new ArrayList<>(items.size());
        for (ImageData m : items) {
            list.add(toX(m));
        }
        return Matrix2.appendColumns(list);
    }
    
    private Matrix2 toY(List<ImageData> items) {
        List<Matrix2> list = new ArrayList<>(items.size());
        for (ImageData m : items) {
            list.add(toY(m));
        }
        return Matrix2.appendColumns(list);
    }
    
    private Matrix2 toX(ImageData item) {
        return new Matrix2(MNIST_IMG_WIDTH * MNIST_IMG_HEIGHT, 1, item.data);
    }
    
    private Matrix2 toY(ImageData item) {
        return new Matrix2(item.label);
    }
    
    private Map<Integer, List<ImageData>> toMap(List<ImageData> items) {
        Map<Integer, List<ImageData>> map = new HashMap<Integer, List<ImageData>>();
        for (ImageData item : items) {
            List<ImageData> list = map.get(item.label);
            if(list == null) {
                list = new ArrayList<>();
                map.put(item.label, list);
            }
            list.add(item);
        }
        return map;
    }
    
    private List<ImageData> loadImageData(String xPath, String yPath) {
        List<ImageData> list = new ArrayList<>();
        
        InputStream inputX = this.getClass().getClassLoader().getResourceAsStream(xPath);
        InputStream inputY = this.getClass().getClassLoader().getResourceAsStream(yPath);
        if(inputX == null || inputY == null) {
            return list;
        }
        
        BufferedReader rx = null;
        BufferedReader ry = null;
        try {
            rx = new BufferedReader(new InputStreamReader(inputX));
            ry = new BufferedReader(new InputStreamReader(inputY));
            String lineX, lineY;
            while((lineX = rx.readLine()) != null) {
                lineX = lineX.trim();
                if(lineX.isEmpty())
                    continue;
                lineY = ry.readLine();
                
                String[] values = lineX.split(",");
                int label = Integer.parseInt(lineY.trim());
                if(label == 10) {
                    label = 0;
                }
                float[] data = new float[values.length];
                for (int i = 0; i < values.length; i++) {
                    data[i] = Float.parseFloat(values[i]);
                }
                list.add(new ImageData(data, label));
            }
            
            return list;
            
        } catch (Exception e) {
            throw new RuntimeException("Error reading trainning data from files: " + xPath + " and " + yPath, e);
        } finally {
            if(rx != null) {
                try {
                    rx.close();
                } catch (Exception e) {
                }
            }
            if(ry != null) {
                try {
                    ry.close();
                } catch (Exception e) {
                }
            }
        }
    }
    
    
    private static class ImageData {
        public final float[] data;
        public final int label;
        public ImageData(float[] data, int label) {
            this.data = data;
            this.label = label;
        }
    }
    
}
