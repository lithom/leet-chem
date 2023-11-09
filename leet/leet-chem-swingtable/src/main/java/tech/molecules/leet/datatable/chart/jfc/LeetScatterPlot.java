package tech.molecules.leet.datatable.chart.jfc;

import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.DefaultXYItemRenderer;

public class LeetScatterPlot implements VisualizationProviderXYZ {
    public LeetScatterPlot() {

    }

    @Override
    public VisualizationConfigPanel getConfigPanel() {
        return new VisualizationConfigPanel() {
            @Override
            public void applyConfiguration(JFreeChart chart) {

            }
        };
    }

    @Override
    public JFreeChart createChart(LeetXYZDataSet dataset) {
        DefaultXYItemRenderer renderer = new DefaultXYItemRenderer();
        renderer.setDefaultLinesVisible(false);
        XYPlot plot = new XYPlot(dataset,new NumberAxis("x"),new NumberAxis("y"),renderer);
        JFreeChart chart = new JFreeChart(plot);
        return chart;
    }

    @Override
    public String toString() {
        return "Scatterplot XY";
    }
}
