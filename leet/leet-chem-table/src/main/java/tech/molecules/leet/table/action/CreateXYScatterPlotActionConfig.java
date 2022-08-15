package tech.molecules.leet.table.action;

import tech.molecules.leet.table.NDataProvider;
import tech.molecules.leet.table.NexusTableModel;
import tech.molecules.leet.table.NumericalDatasource;

import javax.swing.*;

public class CreateXYScatterPlotActionConfig {


    private NDataProvider dp_x;
    private NDataProvider dp_y;

    private NDataProvider dp_color;

    //numerical datasource strings
    private NumericalDatasource nd_x;
    private NumericalDatasource nd_y;

    private NumericalDatasource nd_color;

    private JPanel plotPanel;

    public CreateXYScatterPlotActionConfig(JPanel plotpanel, NDataProvider dp_x, NDataProvider dp_y, NumericalDatasource nd_x, NumericalDatasource nd_y) {
        this.plotPanel = plotpanel;
        this.dp_x = dp_x;
        this.dp_y = dp_y;
        this.nd_x = nd_x;
        this.nd_y = nd_y;
    }

    public CreateXYScatterPlotActionConfig(JPanel plotpanel, NDataProvider dp_x, NDataProvider dp_y, NDataProvider dp_col, NumericalDatasource nd_y, NumericalDatasource nd_x, NumericalDatasource nd_col) {
        this.plotPanel = plotpanel;
        this.dp_x = dp_x;
        this.dp_y = dp_y;
        this.dp_color = dp_col;
        this.nd_x = nd_x;
        this.nd_y = nd_y;
        this.nd_color = nd_col;
    }


    public NumericalDatasource getNd_x() {
        return nd_x;
    }

    public void setNd_x(NumericalDatasource nd_x) {
        this.nd_x = nd_x;
    }
    public NumericalDatasource getNd_y() {
        return nd_y;
    }

    public void setNd_y(NumericalDatasource nd_y) {
        this.nd_x = nd_y;
    }

    public NumericalDatasource getNd_color() {
        return this.nd_color;
    }

    public void setNd_color(NumericalDatasource nd_color) {
        this.nd_color = nd_color;
    }


    public NDataProvider getDp_x() {
        return dp_x;
    }

    public NDataProvider getDp_y() {
        return dp_y;
    }

    public NDataProvider getDp_color() {
        return dp_color;
    }

    public JPanel getPlotPanel() {
        return this.plotPanel;
    }

    public void setPlotPanel(JPanel plotpanel) {
        this.plotPanel = plotpanel;
    }
}
