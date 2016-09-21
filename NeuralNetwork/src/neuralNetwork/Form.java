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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.ImageIcon;
import javax.swing.JLabel;
import javax.swing.JPanel;

/**
 *
 * @author Matias Leone
 */
public class Form extends javax.swing.JFrame {

    private final static int DRAWING_PANEL_WIDTH = 200;
    private final static int DRAWING_PANEL_HEIGHT = 200;
    
    private final static int TRAINNING_IMG_WIDTH = 20;
    private final static int TRAINNING_IMG_HEIGHT = 20;
    
    private DrawingPanel drawingPanel;
    private Map<String, JPanel> trainningPanels;
    private Map<String, List<Image>> trainningImages;
    private NeuralNetwork neuralNetwork;
    private int trainningImagesCount;
    
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
        
        this.neuralNetwork = new NeuralNetwork(400, 25, 10);
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
            String[] strings = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10" };
            public int getSize() { return strings.length; }
            public Object getElementAt(int i) { return strings[i]; }
        });
        listLabels.setSelectionMode(javax.swing.ListSelectionModel.SINGLE_SELECTION);
        listLabels.setSelectedIndex(0);
        jScrollPane1.setViewportView(listLabels);

        buttonTrain.setText("Train network");
        buttonTrain.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                buttonTrainActionPerformed(evt);
            }
        });

        buttonPredict.setText("Predict symbol");

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
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(buttonClear)
                                .addGap(108, 108, 108)
                                .addComponent(buttonTrain)
                                .addGap(26, 26, 26)
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                                    .addComponent(labelPredict, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                    .addComponent(buttonPredict, javax.swing.GroupLayout.PREFERRED_SIZE, 118, Short.MAX_VALUE))
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, 18, Short.MAX_VALUE))
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(buttonAdd, javax.swing.GroupLayout.PREFERRED_SIZE, 76, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addGap(0, 0, Short.MAX_VALUE)))))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(buttonAdd)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(buttonTrain)
                            .addComponent(buttonPredict)
                            .addComponent(buttonClear))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(labelPredict, javax.swing.GroupLayout.PREFERRED_SIZE, 104, javax.swing.GroupLayout.PREFERRED_SIZE))
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
        String n = (String)listLabels.getSelectedValue();
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
        
        BufferedImage drawingImage = drawingPanel.createImage();
        Image scaledImage = drawingImage.getScaledInstance(TRAINNING_IMG_WIDTH, TRAINNING_IMG_HEIGHT, Image.SCALE_FAST);
        ImageIcon imageIcon = new ImageIcon(scaledImage);
        JLabel imageLabel = new JLabel(imageIcon);
        trainningPanel.add(imageLabel);
        this.trainningImages.get(n).add(scaledImage);

        this.drawingPanel.clearDraw();
        
        this.trainningPanelsContainer.validate();
        this.trainningPanelsContainer.repaint();
 
    }//GEN-LAST:event_buttonAddActionPerformed

    private void drawingPanelPlaceholderMouseExited(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_drawingPanelPlaceholderMouseExited

    }//GEN-LAST:event_drawingPanelPlaceholderMouseExited

    private void drawingPanelPlaceholderMouseDragged(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_drawingPanelPlaceholderMouseDragged

    }//GEN-LAST:event_drawingPanelPlaceholderMouseDragged

    
    private void trainNetwork() {
        List<float[]> input = new ArrayList<>(this.trainningImagesCount);
        int[] inputClass = new int[this.trainningImagesCount];
        
        BufferedImage tempImg = new BufferedImage(TRAINNING_IMG_WIDTH, TRAINNING_IMG_HEIGHT, BufferedImage.TYPE_INT_ARGB);
        Graphics2D tempImgG = tempImg.createGraphics();
        int i = 0;
        for (Map.Entry<String, List<Image>> entry : trainningImages.entrySet()) {
            int n = Integer.parseInt(entry.getKey());
            for (Image image : entry.getValue()) {
                tempImgG.setPaint(Color.WHITE);
                tempImgG.fillRect(0, 0, TRAINNING_IMG_WIDTH, TRAINNING_IMG_HEIGHT);
                tempImgG.drawImage(image, 0, 0, this);
                
                float[] imgData = imageToFloatArray(tempImg);
                input.add(imgData);
                inputClass[i++] = n;
            }
        }
        
        this.neuralNetwork.train(input, inputClass, 10, 1, 1);
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
            Color c = new Color(pixel, true);
            outputPixels[i] = (c.getRed() > 0) ? 1 : 0;
        }
        
        return outputPixels;
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
    private javax.swing.JButton buttonTrain;
    private javax.swing.JPanel drawingPanelPlaceholder;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JScrollPane jScrollPane2;
    private javax.swing.JLabel labelPredict;
    private javax.swing.JList listLabels;
    private javax.swing.JPanel trainningPanelsContainer;
    // End of variables declaration//GEN-END:variables
}