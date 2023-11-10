package tech.molecules.leet.datatable.chart.jfc;

import javax.swing.*;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import tech.molecules.leet.datatable.DataTable;
import tech.molecules.leet.datatable.swing.XYZDataSourceController;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;

public class VisualizationComponent extends JPanel {

    //private LeetXYZDataSet dataset;


    // plot type combobox and datasource config
    private JPanel topPanel;
    // plot type config
    private JPanel rightPanel;

    private JComboBox<VisualizationProviderXYZ> visualizationComboBox;
    private XYZDataSourceController datasetController;

    private VisualizationConfigPanel configPanel;
    private ChartPanel chartPanel;

    public VisualizationComponent(DataTable table) {
        setLayout(new BorderLayout());

        List<VisualizationProviderXYZ> visualizationProviders = new ArrayList<>();
        visualizationProviders.add(new LeetScatterPlot());

        this.topPanel = new JPanel();
        this.topPanel.setLayout(new BorderLayout());
        this.rightPanel = new JPanel();
        this.rightPanel.setLayout(new BorderLayout());

        // Initialize components
        visualizationComboBox = new JComboBox<>(visualizationProviders.toArray(new VisualizationProviderXYZ[0]));
        datasetController = new XYZDataSourceController(table);
        this.topPanel.add(this.visualizationComboBox,BorderLayout.WEST);
        this.topPanel.add(this.datasetController,BorderLayout.CENTER);

        configPanel = new DefaultConfigPanel();
        this.rightPanel.add(configPanel);

        chartPanel = new ChartPanel(null);

        // Create the configuration panel
        //JPanel configurationPanel = new JPanel(new BorderLayout());
        //configurationPanel.add(visualizationComboBox, BorderLayout.NORTH);
        //configurationPanel.add(configPanel, BorderLayout.CENTER);

        // Add components to the main panel
        add(topPanel, BorderLayout.NORTH);
        add(chartPanel, BorderLayout.CENTER);
        add(rightPanel, BorderLayout.EAST);

        // Add action listener to the combo box
        visualizationComboBox.addActionListener(e -> updateVisualization());
        datasetController.addChangeListener(e -> updateVisualization());

        this.updateVisualization();
        this.revalidate();
        this.repaint();
    }

    private void updateVisualization() {
        //String selectedOption = (String) visualizationComboBox.getSelectedItem();
        //VisualizationProviderXYZ visualizationProvider = createVisualizationProvider(selectedOption);
        VisualizationProviderXYZ visualizationProvider = (VisualizationProviderXYZ) visualizationComboBox.getSelectedItem();

        if (visualizationProvider != null) {
            JFreeChart chart = visualizationProvider.createChart(this.datasetController.getDataset());
            chartPanel.setChart(chart);
            configPanel.applyConfiguration(chart);
        }
    }

    private static class DefaultConfigPanel extends VisualizationConfigPanel {
        @Override
        public void applyConfiguration(JFreeChart chart) {
            // Default implementation (no configuration options)
        }
    }
}
