package tech.molecules.leet.datatable.chart.jfc;

import org.jfree.chart.JFreeChart;

import javax.swing.*;

public abstract class VisualizationConfigPanel extends JPanel {
    public abstract void applyConfiguration(JFreeChart chart);
}