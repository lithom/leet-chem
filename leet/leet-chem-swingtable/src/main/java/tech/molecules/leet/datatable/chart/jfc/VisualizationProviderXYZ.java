package tech.molecules.leet.datatable.chart.jfc;

import org.jfree.chart.JFreeChart;

public interface VisualizationProviderXYZ {
    public VisualizationConfigPanel getConfigPanel();
    public JFreeChart createChart(LeetXYZDataSet dataset);
}
