package neuralNetwork;

import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Canvas;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.Stroke;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.WindowAdapter;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import javax.swing.BoxLayout;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSpinner;
import javax.swing.JTextArea;
import javax.swing.SpinnerNumberModel;
import javax.swing.UIManager;

/**
 *
 * @author matias.leone
 */
public class UI {

    private final static int WIN_WIDTH = 1200;
    private final static int WIN_HEIGHT = 720;

    private final static int POINTS_COUNT = 500;
    
    private JFrame frame;
    private JTextArea logArea;
    private NeuralNetwork neuralNetwork;
    private List<float[]> imageData;
    
    public static void main(String[] args) {
        new UI();
    }
    
    public UI() {
        neuralNetwork = new NeuralNetwork(400, 25, 10);
        
        imageData = loadImages("data/input_images.csv");
        
        
        frame = new JFrame("Neural Network - Image recognition");
        frame.setMinimumSize(new Dimension(WIN_WIDTH, WIN_HEIGHT));
        frame.setLayout(new BorderLayout());
        frame.setDefaultCloseOperation(frame.EXIT_ON_CLOSE);
        frame.setResizable(false);
        
        JPanel controlsPanel = new JPanel(new FlowLayout());
        JButton button = new JButton("Predict");
        button.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                predict();
            }
        });
        controlsPanel.add(button);
        frame.add(controlsPanel, BorderLayout.NORTH);
        
        
        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.Y_AXIS));
        for (int i = 0; i < 60; i++) {
            JPanel panel = new JPanel(new FlowLayout());
            for (int j = 0; j < 45; j++) {
                int idx = i * 60 + j;
                ImageIcon image = new ImageIcon(createGrayScaleImage(imageData.get(idx), 20, 20));
                panel.add(new JLabel(image));
            }
            mainPanel.add(panel);
        }
        JScrollPane imageScrollPane = new JScrollPane(mainPanel);
        frame.add(imageScrollPane, BorderLayout.CENTER);
        
        
        
        
        logArea = new JTextArea(4, 100);
        logArea.setEditable(false);
        JScrollPane scrollPane = new JScrollPane(logArea);
        scrollPane.setMinimumSize(new Dimension(-1, 200));
        frame.add(scrollPane, BorderLayout.SOUTH);
        
        Dimension screenDim = Toolkit.getDefaultToolkit().getScreenSize();
        frame.setLocation(screenDim.width / 2 - WIN_WIDTH / 2, screenDim.height / 2 - WIN_HEIGHT / 2);
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch(Exception e) {
            throw new RuntimeException(e);
        }
        frame.setVisible(true);

        
        
        
    }
    

    
    /**
     * Predict number
     */
    private void predict() {
        //neuralNetwork.predict(input)
    }
    
    
    
    private void clearLog() {
        logArea.setText("");
        logArea.setCaretPosition(0);
    }
    
    private void log(String txt) {
        logArea.append(txt);
        logArea.append("\n");
        logArea.setCaretPosition(logArea.getDocument().getLength());
    }
    
    private List<float[]> loadImages(String filePath) {
        BufferedReader r = null;
        try {
            InputStream in = this.getClass().getClassLoader().getResourceAsStream(filePath);
            r = new BufferedReader(new InputStreamReader(in));
            String line;
            List<float[]> images = new ArrayList<>(5000);
            while((line = r.readLine()) != null) {
                String[] columns = line.split(",");
                float[] pixels = new float[columns.length];
                for (int i = 0; i < columns.length; i++) {
                    pixels[i] = Float.parseFloat(columns[i]);
                }
                
                images.add(pixels);
            }
            
            return images;
        } catch (Exception e) {
            throw new RuntimeException("Error openning file: " + filePath);
        } finally {
            try {
                if(r != null)
                    r.close();
            } catch (Exception ex) {
            }
        }
        
    }
    
    
    private Image createGrayScaleImage(float[] data, int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        
        int[] pixels = new int[data.length];
        for (int i = 0; i < data.length; i++) {
            float p = data[i];
            //pixels[i] = (int)(data[i] * 0x00010101);

            int r = (int)(p * 255);
            int g = (int)(p * 255);
            int b = (int)(p * 255);
            int a = (int)(1 * 255);
            int c = ((a & 0xFF) << 24) |((r & 0xFF) << 16) | ((g & 0xFF) << 8)  | ((b & 0xFF) << 0);
            pixels[i] = c;
            //pixels[i] = Integer.MAX_VALUE;
            
            
        }
        
        
        image.setRGB(0, 0, width, height, pixels, 0, width);
        return image;
    }
    
}
