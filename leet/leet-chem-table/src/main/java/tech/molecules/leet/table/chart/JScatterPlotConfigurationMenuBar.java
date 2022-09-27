package tech.molecules.leet.table.chart;

import tech.molecules.leet.table.NexusTableModel;
import tech.molecules.leet.table.gui.JNumericalDataSourceSelector;

import javax.swing.*;
import java.util.ArrayList;
import java.util.List;

public class JScatterPlotConfigurationMenuBar extends JMenuBar {

    private NexusTableModel ntm;

    private List<ScatterPlotModel> models = new ArrayList<>();

    public void addModel(ScatterPlotModel mi) {
        this.models.add(mi);
    }

    public boolean removeModel(ScatterPlotModel mi) {
        return this.models.remove(mi);
    }

    public JScatterPlotConfigurationMenuBar(NexusTableModel ntm) {
        this.ntm = ntm;
        init();
    }

    private void init() {

    }



    public static void initMenuBar(JMenuBar jmb,NexusTableModel ntm, List<ScatterPlotModel> scatterplots) {

            JMenu jmsize = new JMenu("Marker");
            jmsize.add(new ScatterPlotModel.SetPointSizeAction(scatterplots,2));
            jmsize.add(new ScatterPlotModel.SetPointSizeAction(scatterplots,4));
            jmsize.add(new ScatterPlotModel.SetPointSizeAction(scatterplots,8));
            jmsize.add(new ScatterPlotModel.SetPointSizeAction(scatterplots,16));
            jmb.add(jmsize);

            JMenu jmcolor = new JMenu("Color");
            JNumericalDataSourceSelector jndss = new JNumericalDataSourceSelector(new JNumericalDataSourceSelector.NumericalDataSourceSelectorModel(ntm), JNumericalDataSourceSelector.SELECTOR_MODE.OnlyJMenu);
            JMenu jnds = jndss.getMenu();
            jnds.setText("From Numerical Datasource");
            jmcolor.add(jnds);
            jndss.addSelectionListener(new JNumericalDataSourceSelector.SelectionListener() {
                @Override
                public void selectionChanged() {
                    // set coloring according to values of nds
                    for(ScatterPlotModel jf : scatterplots) {
                        jf.setColorValues(jndss.getModel().getSelectedDatasource());
                    }
                }
            });
            jmb.add(jmcolor);
    }



}
