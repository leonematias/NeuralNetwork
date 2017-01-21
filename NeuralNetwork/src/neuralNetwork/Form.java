/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralNetwork;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.ImageIcon;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;

/**
 * Main UI
 * 
 * @author Matias Leone
 */
public class Form extends javax.swing.JFrame {

    //Parameters
    private final static int TRAINNING_IMG_WIDTH = 20;
    private final static int TRAINNING_IMG_HEIGHT = 20;
    private final static int GRADIENT_DESCENT_ITER = 500;
    private final static float ALPHA = 0.01f;
    private final static float LAMBDA = 1;
    private final static int INPUT_LAYER_SIZE = TRAINNING_IMG_WIDTH * TRAINNING_IMG_HEIGHT;
    private final static int HIDDEN_LAYER_SIZE = 25;
    private final static int OUTPUT_LAYER_SIZE = 10;

    
    private DrawingPanel drawingPanel;
    private Map<String, JPanel> trainningPanels;
    private Map<String, List<Image>> trainningImages;
    private NeuralNetwork neuralNetwork;
    private LogisticRegression logisticRegression;
    private NeuralNetwork2 neuralNetwork2;
    private int trainningImagesCount;
    private BufferedImage tempImg;
    private Graphics2D tempImgG;
    
    /**
     * Creates new form Form
     */
    public Form() {
        initComponents();
        
        this.drawingPanel = new DrawingPanel();
        this.drawingPanelPlaceholder.setLayout(new BorderLayout());
        this.drawingPanelPlaceholder.add(this.drawingPanel, BorderLayout.CENTER);
        
        this.trainningPanelsContainer.setLayout(new BoxLayout(this.trainningPanelsContainer, BoxLayout.Y_AXIS));
        
        this.trainningPanels = new HashMap<>();
        this.trainningImages = new HashMap<>();
        this.trainningImagesCount = 0;
        
        this.neuralNetwork = new NeuralNetwork(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);
        this.logisticRegression = new LogisticRegression(INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE, ALPHA, GRADIENT_DESCENT_ITER);
        this.neuralNetwork2 = new NeuralNetwork2(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, ALPHA, GRADIENT_DESCENT_ITER);
        
        tempImg = new BufferedImage(TRAINNING_IMG_WIDTH, TRAINNING_IMG_HEIGHT, BufferedImage.TYPE_INT_ARGB);
        tempImgG = tempImg.createGraphics();
        
        loadTrainningDataPanelFromFile("data/trainning_data.txt");
    }
    
    private void trainNetwork() {
        
        List<TrainningImageData> items = getTrainningDataFromImages();
        List<float[]> input = new ArrayList<>(items.size());
        int[] inputClass = new int[items.size()];
        
        int i = 0;
        for (TrainningImageData item : items) {
            input.add(item.imgData);
            inputClass[i++] = item.label;
        }
        
        
        /*
        List<float[]> input = new ArrayList<>();
        int[] inputClass = new int[2];
        
        input.add(new float[]{1, 0, 0});
        inputClass[0] = 0;
        
        input.add(new float[]{0, 0, 1});
        inputClass[1] = 1;
        */
        
        //Train
        //this.neuralNetwork.train(input, inputClass, GRADIENT_DESCENT_ITER, ALPHA, LAMBDA);
        
        
        List<MathUtils.Input> inputData = new ArrayList<>(items.size());
        for (TrainningImageData item : items) {
            inputData.add(new MathUtils.Input(item.imgData, item.label));
        }
        //this.logisticRegression.train(inputData);
        this.neuralNetwork2.train(inputData);
        
        
        
        JOptionPane.showMessageDialog(this, "Trainning is done");
    }
    
    private void predict() {
        Image image = getImageFromDrawingPanel();
                
        float[] imgData = drawToTempImageAndGetArray(image);
        

        //float[] imgData = new float[]{1, 0, 0};
        
        
        /*
        NeuralNetwork.Result result = neuralNetwork.predict(imgData);
        int n = result.predictedClass;
        */
        
        //int n = logisticRegression.predict(imgData);
        int n = neuralNetwork2.predict(imgData);
        
        labelPredict.setText(String.valueOf(n));
        //labelConfidence.setText(String.valueOf(result.confidence));
        
    }
    
    
    
    
    
    
    
    

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        drawingPanelPlaceholder = new javax.swing.JPanel();
        buttonClear = new javax.swing.JButton();
        buttonAdd = new javax.swing.JButton();
        jScrollPane1 = new javax.swing.JScrollPane();
        listLabels = new javax.swing.JList();
        buttonTrain = new javax.swing.JButton();
        buttonPredict = new javax.swing.JButton();
        labelPredict = new javax.swing.JLabel();
        jScrollPane2 = new javax.swing.JScrollPane();
        trainningPanelsContainer = new javax.swing.JPanel();
        buttonSaveTrainningData = new javax.swing.JButton();
        labelConfidence = new javax.swing.JLabel();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("Neural Network");
        setMinimumSize(new java.awt.Dimension(1000, 600));

        drawingPanelPlaceholder.setBackground(new java.awt.Color(255, 255, 255));
        drawingPanelPlaceholder.setPreferredSize(new java.awt.Dimension(200, 200));
        drawingPanelPlaceholder.addMouseMotionListener(new java.awt.event.MouseMotionAdapter() {
            public void mouseMoved(java.awt.event.MouseEvent evt) {
                drawingPanelPlaceholderMouseMoved(evt);
            }
            public void mouseDragged(java.awt.event.MouseEvent evt) {
                drawingPanelPlaceholderMouseDragged(evt);
            }
        });
        drawingPanelPlaceholder.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mousePressed(java.awt.event.MouseEvent evt) {
                drawingPanelPlaceholderMousePressed(evt);
            }
            public void mouseReleased(java.awt.event.MouseEvent evt) {
                drawingPanelPlaceholderMouseReleased(evt);
            }
            public void mouseExited(java.awt.event.MouseEvent evt) {
                drawingPanelPlaceholderMouseExited(evt);
            }
        });

        javax.swing.GroupLayout drawingPanelPlaceholderLayout = new javax.swing.GroupLayout(drawingPanelPlaceholder);
        drawingPanelPlaceholder.setLayout(drawingPanelPlaceholderLayout);
        drawingPanelPlaceholderLayout.setHorizontalGroup(
            drawingPanelPlaceholderLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 200, Short.MAX_VALUE)
        );
        drawingPanelPlaceholderLayout.setVerticalGroup(
            drawingPanelPlaceholderLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 200, Short.MAX_VALUE)
        );

        buttonClear.setText("Clear");
        buttonClear.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                buttonClearActionPerformed(evt);
            }
        });

        buttonAdd.setText("Add");
        buttonAdd.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                buttonAddActionPerformed(evt);
            }
        });

        listLabels.setModel(new javax.swing.AbstractListModel() {
            String[] strings = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };
            public int getSize() { return strings.length; }
            public Object getElementAt(int i) { return strings[i]; }
        });
        listLabels.setSelectionMode(javax.swing.ListSelectionModel.SINGLE_SELECTION);
        jScrollPane1.setViewportView(listLabels);

        buttonTrain.setText("Train network");
        buttonTrain.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                buttonTrainActionPerformed(evt);
            }
        });

        buttonPredict.setText("Predict symbol");
        buttonPredict.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                buttonPredictActionPerformed(evt);
            }
        });

        labelPredict.setBackground(new java.awt.Color(255, 255, 255));
        labelPredict.setFont(new java.awt.Font("Tahoma", 1, 24)); // NOI18N
        labelPredict.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        labelPredict.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));

        jScrollPane2.setBorder(javax.swing.BorderFactory.createTitledBorder("Trainning data"));

        trainningPanelsContainer.setPreferredSize(new java.awt.Dimension(5000, 5000));
        trainningPanelsContainer.setRequestFocusEnabled(false);

        javax.swing.GroupLayout trainningPanelsContainerLayout = new javax.swing.GroupLayout(trainningPanelsContainer);
        trainningPanelsContainer.setLayout(trainningPanelsContainerLayout);
        trainningPanelsContainerLayout.setHorizontalGroup(
            trainningPanelsContainerLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 0, Short.MAX_VALUE)
        );
        trainningPanelsContainerLayout.setVerticalGroup(
            trainningPanelsContainerLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 5000, Short.MAX_VALUE)
        );

        jScrollPane2.setViewportView(trainningPanelsContainer);

        buttonSaveTrainningData.setText("Save trainning data");
        buttonSaveTrainningData.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                buttonSaveTrainningDataActionPerformed(evt);
            }
        });

        labelConfidence.setText("Confidence");

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(jScrollPane2, javax.swing.GroupLayout.PREFERRED_SIZE, 0, Short.MAX_VALUE)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(drawingPanelPlaceholder, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(jScrollPane1, javax.swing.GroupLayout.PREFERRED_SIZE, 76, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                            .addComponent(buttonAdd, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addComponent(buttonClear, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(buttonSaveTrainningData)
                            .addComponent(buttonTrain, javax.swing.GroupLayout.PREFERRED_SIZE, 127, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addGap(116, 116, 116)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                            .addComponent(labelPredict, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addComponent(buttonPredict, javax.swing.GroupLayout.DEFAULT_SIZE, 118, Short.MAX_VALUE)
                            .addComponent(labelConfidence, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(buttonAdd)
                            .addComponent(buttonSaveTrainningData))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(buttonTrain)
                            .addComponent(buttonPredict)
                            .addComponent(buttonClear))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(labelPredict, javax.swing.GroupLayout.PREFERRED_SIZE, 104, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(labelConfidence))
                    .addComponent(drawingPanelPlaceholder, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(jScrollPane1))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jScrollPane2, javax.swing.GroupLayout.DEFAULT_SIZE, 363, Short.MAX_VALUE)
                .addContainerGap())
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void buttonTrainActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_buttonTrainActionPerformed
        trainNetwork();
    }//GEN-LAST:event_buttonTrainActionPerformed

    private void drawingPanelPlaceholderMousePressed(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_drawingPanelPlaceholderMousePressed

    }//GEN-LAST:event_drawingPanelPlaceholderMousePressed

    private void drawingPanelPlaceholderMouseReleased(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_drawingPanelPlaceholderMouseReleased

    }//GEN-LAST:event_drawingPanelPlaceholderMouseReleased

    private void drawingPanelPlaceholderMouseMoved(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_drawingPanelPlaceholderMouseMoved
        
    }//GEN-LAST:event_drawingPanelPlaceholderMouseMoved

    private void buttonClearActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_buttonClearActionPerformed
        drawingPanel.clearDraw();
    }//GEN-LAST:event_buttonClearActionPerformed

    private void buttonAddActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_buttonAddActionPerformed
        int n = Integer.parseInt((String)listLabels.getSelectedValue());
        addImageToTrainningDataPanel(getImageFromDrawingPanel(), n);

        this.drawingPanel.clearDraw();
    }//GEN-LAST:event_buttonAddActionPerformed

    
    
    private void drawingPanelPlaceholderMouseExited(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_drawingPanelPlaceholderMouseExited

    }//GEN-LAST:event_drawingPanelPlaceholderMouseExited

    private void drawingPanelPlaceholderMouseDragged(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_drawingPanelPlaceholderMouseDragged

    }//GEN-LAST:event_drawingPanelPlaceholderMouseDragged

    private void buttonSaveTrainningDataActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_buttonSaveTrainningDataActionPerformed
        List<TrainningImageData> items = getTrainningDataFromImages();
        File dst = new File("trainning_data.txt");
        saveTrainningDataToFile(dst.getAbsolutePath(), items);
        JOptionPane.showMessageDialog(this, "Data saved to " + dst.getAbsolutePath(), "Data saved", JOptionPane.INFORMATION_MESSAGE);
    }//GEN-LAST:event_buttonSaveTrainningDataActionPerformed

    private void buttonPredictActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_buttonPredictActionPerformed
        predict();
    }//GEN-LAST:event_buttonPredictActionPerformed

    private Image getImageFromDrawingPanel() {
        BufferedImage drawingImage = drawingPanel.createImage();
        Image scaledImage = drawingImage.getScaledInstance(TRAINNING_IMG_WIDTH, TRAINNING_IMG_HEIGHT, Image.SCALE_FAST);
        return scaledImage;
    }
    
    private void addImageToTrainningDataPanel(Image image, int label) {
        String n = String.valueOf(label);
        JPanel trainningPanel = this.trainningPanels.get(n);
        if(trainningPanel == null) {
            trainningPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
            trainningPanel.setBorder(BorderFactory.createTitledBorder(String.valueOf(n)));
            trainningPanel.setMaximumSize(new Dimension(5000, TRAINNING_IMG_HEIGHT + 50));
            this.trainningPanelsContainer.add(trainningPanel);
            this.trainningPanels.put(n, trainningPanel);
            this.trainningImages.put(n, new ArrayList<Image>());
            this.trainningImagesCount++;
        }
        
        ImageIcon imageIcon = new ImageIcon(image);
        JLabel imageLabel = new JLabel(imageIcon);
        trainningPanel.add(imageLabel);
        this.trainningImages.get(n).add(image);
        
        this.trainningPanelsContainer.validate();
        this.trainningPanelsContainer.repaint();
    }
    
    private void saveTrainningDataToFile(String path, List<TrainningImageData> items) {
        BufferedWriter w = null;
        try {
            w = new BufferedWriter(new FileWriter(new File(path)));
            for (TrainningImageData item : items) {
                w.write(item.label + ":");
                for (int i = 0; i < item.imgData.length; i++) {
                    if(i > 0) {
                        w.write(",");
                    }
                    w.write(String.valueOf(item.imgData[i]));
                }
                w.newLine();
            }
            
        } catch (Exception e) {
            throw new RuntimeException("Error saving trainning data", e);
        } finally {
            if(w != null) {
                try {
                    w.close();
                } catch (Exception e) {
                }
            }
        }
    }
    
    private void loadTrainningDataPanelFromFile(String resourcePath) {
        List<TrainningImageData> items = loadTrainningDataFromFile(resourcePath);
        for (TrainningImageData item : items) {
            
            BufferedImage image = floatArrayToImage(item.imgData, TRAINNING_IMG_WIDTH, TRAINNING_IMG_HEIGHT);
            addImageToTrainningDataPanel(image, item.label);
        }
    }
    
    private List<TrainningImageData> loadTrainningDataFromFile(String resourcePath) {
        List<TrainningImageData> list = new ArrayList<>();
        
        InputStream input = this.getClass().getClassLoader().getResourceAsStream(resourcePath);
        if(input == null) {
            return list;
        }
        
        BufferedReader r = null;
        try {
            r = new BufferedReader(new InputStreamReader(input));
            String line;
            while((line = r.readLine()) != null) {
                line = line.trim();
                if(line.isEmpty())
                    continue;
                
                String[] parts = line.split(":");
                String[] values = parts[1].split(",");
                TrainningImageData item = new TrainningImageData();
                item.label = Integer.parseInt(parts[0]);
                item.imgData = new float[values.length];
                for (int i = 0; i < values.length; i++) {
                    item.imgData[i] = Float.parseFloat(values[i]);
                }
                list.add(item);
            }
            
            return list;
            
        } catch (Exception e) {
            throw new RuntimeException("Error reading trainning data from file: " + resourcePath, e);
        } finally {
            if(r != null) {
                try {
                    r.close();
                } catch (Exception e) {
                }
            }
        }
    }
    
    private List<TrainningImageData> getTrainningDataFromImages() {
        List<TrainningImageData> list = new ArrayList<>(this.trainningImagesCount);
         
        for (Map.Entry<String, List<Image>> entry : trainningImages.entrySet()) {
            int label = Integer.parseInt(entry.getKey());
            for (Image image : entry.getValue()) {
                TrainningImageData item = new TrainningImageData();
                item.label = label;
                item.imgData = drawToTempImageAndGetArray(image);
                list.add(item);
            }
        }
         
         return list;
    }
    
    private float[] drawToTempImageAndGetArray(Image image) {
        //Clear
        tempImgG.setPaint(Color.WHITE);
        tempImgG.fillRect(0, 0, TRAINNING_IMG_WIDTH, TRAINNING_IMG_HEIGHT);
        
        //Draw image
        tempImgG.drawImage(image, 0, 0, this);
        
        //Get array
        return imageToFloatArray(tempImg);
    }
    
    
    private float[] imageToFloatArray(BufferedImage image) {
        int[] pixels = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
        float[] outputPixels = new float[pixels.length];
        int width = image.getWidth();
        int height = image.getHeight();
        boolean hasAlphaChannel = image.getAlphaRaster() != null;
        int pixelLength = 3;
        if (hasAlphaChannel) {
            pixelLength = 4;
        }
        
        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            Color c = new Color(pixel, hasAlphaChannel);
            int intensity = c.getRed() + c.getGreen() + c.getGreen();
            outputPixels[i] = intensity == 0 ? 1 : 0;
        }
        
        return outputPixels;
    }
    
    private BufferedImage floatArrayToImage(float[] data, int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        int[] pixels = new int[data.length];
        for (int i = 0; i < data.length; i++) {
            float p = 1 - data[i];
            int r = (int)(p * 255);
            int g = (int)(p * 255);
            int b = (int)(p * 255);
            int a = (int)(1 * 255);
            int c = ((a & 0xFF) << 24) |((r & 0xFF) << 16) | ((g & 0xFF) << 8)  | ((b & 0xFF) << 0);
            pixels[i] = c;
        }
        image.setRGB(0, 0, width, height, pixels, 0, width);
        
        return image;
    }
    
    private class TrainningImageData {
        public float[] imgData;
        public int label;
    }
    
    
    
    
    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(Form.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(Form.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(Form.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(Form.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new Form().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton buttonAdd;
    private javax.swing.JButton buttonClear;
    private javax.swing.JButton buttonPredict;
    private javax.swing.JButton buttonSaveTrainningData;
    private javax.swing.JButton buttonTrain;
    private javax.swing.JPanel drawingPanelPlaceholder;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JScrollPane jScrollPane2;
    private javax.swing.JLabel labelConfidence;
    private javax.swing.JLabel labelPredict;
    private javax.swing.JList listLabels;
    private javax.swing.JPanel trainningPanelsContainer;
    // End of variables declaration//GEN-END:variables
}
