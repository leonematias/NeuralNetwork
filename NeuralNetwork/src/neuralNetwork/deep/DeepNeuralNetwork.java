/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralNetwork.deep;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Deep Neural Network with L hidden layers
 * 
 * @author Matias Leone
 */
public class DeepNeuralNetwork {
    
    private final int[] layerDims;
    
    public DeepNeuralNetwork(int[] layerDims) {
        this.layerDims = layerDims;
    }
    
    public void train() {
        
    }

    public void predict() {
        
    }
    
    private void doTrain(Matrix2 X, Matrix2 Y, int iterations, float learningRate, boolean printCost) {
        int m = X.cols();
        
        //Initialize parameters
        Map<String, Matrix2> parameters = initializeParameters(this.layerDims);
        
        //Gradient descent loop
        Map<String, Matrix2> cache = new HashMap<>();
        List<CacheItem> caches = new ArrayList<>();
        for (int i = 0; i < iterations; i++) {
            caches.clear();
            
            //Forward propagation
            Matrix2 AL = modelForward(X, Y, parameters, caches);
            
            
        }
        
    }
    
    private Map<String, Matrix2> initializeParameters(int[] layerDims) {
        Map<String, Matrix2> parameters = new HashMap<>(layerDims.length * 2);
        
        for (int i = 1; i < layerDims.length; i++) {
            String iStr = String.valueOf(i);
            int rows = layerDims[i];
            int cols = layerDims[i - 1];
            parameters.put("W" + iStr, Matrix2.random(rows, cols).mul(0.01f));
            parameters.put("b" + iStr, Matrix2.zeros(rows, 1));
        }
        
        return parameters;
    }
    
    private Matrix2 modelForward(Matrix2 X, Matrix2 Y, Map<String, Matrix2> parameters, List<CacheItem> caches) {
        return null;
    }
    
    private void linearActivationForward(Matrix2 A_prev, Matrix2 W, Matrix2 b, String activation) {
        
    }
    
    private Matrix2 linearForward(Matrix2 A, Matrix2 W, Matrix2 b, CacheItem outCache) {
        //Z = W * A + b;
        Matrix2 WxA = W.mul(A);
        Matrix2 Z = WxA.add(b.broadcastCol(WxA.cols()));
        
        
        //outCache.A = 
        
        return Z;
    }
    
    
    private static class CacheTuple {
        public final CacheItem linearCache;
        public final CacheItem activationCache;
        public CacheTuple(CacheItem linearCache, CacheItem activationCache) {
            this.linearCache = linearCache;
            this.activationCache = activationCache;
        }
    }
    
    private static class CacheItem {
        public Matrix2 A;
        public Matrix2 W;
        public Matrix2 b;
        public CacheItem(Matrix2 A, Matrix2 W, Matrix2 b) {
            this.A = A;
            this.W = W;
            this.b = b;
        }
        
    }
    
}
