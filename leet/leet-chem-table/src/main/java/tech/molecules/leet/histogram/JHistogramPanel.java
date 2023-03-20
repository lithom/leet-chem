package tech.molecules.leet.histogram;


import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.Rectangle2D;
import java.util.Random;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.ValueMarker;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYBarRenderer;
import org.jfree.data.statistics.HistogramDataset;
import org.jfree.data.xy.IntervalXYDataset;

public class JHistogramPanel extends JPanel {

    //private static final long serialVersionUID = 1L;
    private double threshold = 0.5;
    private JFreeChart chart;
    private ChartPanel chartPanel;
    private JSlider slider;
    private ValueMarker marker;

    public JHistogramPanel() {
        //super("Threshold Line Demo");

        // Create the dataset
        IntervalXYDataset dataset = createDataset();

        // Create the chart
        chart = ChartFactory.createHistogram("Histogram", "X", "Y", dataset, PlotOrientation.VERTICAL, true, true, false);
        XYPlot plot = (XYPlot) chart.getPlot();
        XYBarRenderer renderer = (XYBarRenderer) plot.getRenderer();
        renderer.setDrawBarOutline(false);

        // Add the threshold line
        ValueAxis domainAxis = plot.getDomainAxis();
        marker = new ValueMarker(threshold);
        marker.setPaint(Color.RED);
        plot.addDomainMarker(marker);

        // Create the chart panel
        chartPanel = new ChartPanel(chart);
        chartPanel.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Rectangle2D dataArea = chartPanel.getScreenDataArea();
                threshold = domainAxis.java2DToValue(e.getX(), dataArea, plot.getDomainAxisEdge());
                marker.setValue(threshold);
                slider.setValue((int) threshold);
            }
        });

        // Create the slider
        slider = new JSlider(0, 100, (int) threshold);
        slider.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                threshold = slider.getValue();
                marker.setValue(threshold);
            }
        });

        this.setLayout(new BorderLayout());
        this.add(chartPanel,BorderLayout.CENTER);
        this.add(slider,BorderLayout.NORTH);
    }

    public IntervalXYDataset createDataset() {

        Random r = new Random();
        HistogramDataset hd = new HistogramDataset();

        double ri[] = new double[200];
        for(int zi = 0;zi<ri.length;zi++) { ri[zi] = r.nextGaussian(); }

        hd.addSeries("A",ri,20);
        return hd;
    }

    public static void main(String args[]) {
        JFrame fi = new JFrame();
        JHistogramPanel da = new JHistogramPanel();
        fi.getContentPane().setLayout(new BorderLayout());
        fi.add(da,BorderLayout.CENTER);
        fi.setSize(600,600);
        fi.setVisible(true);
    }

}