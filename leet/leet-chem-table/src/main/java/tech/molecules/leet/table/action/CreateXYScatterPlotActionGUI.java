package tech.molecules.leet.table.action;

import com.actelion.research.gui.VerticalFlowLayout;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;
import tech.molecules.leet.table.NDataProvider;
import tech.molecules.leet.table.NexusTableModel;
import tech.molecules.leet.table.NumericalDatasource;
import tech.molecules.leet.table.gui.JNumericalDataSourceSelector;

import javax.swing.*;
import java.awt.*;
import java.util.function.Consumer;
import java.util.function.Supplier;

public class CreateXYScatterPlotActionGUI {

    private CreateXYScatterPlotActionModel model;

    //private CreateXYScatterPlotActionConfig conf;
    private Frame owner;
    private JPanel pi;
    private Triple<JPanel, Supplier<NumericalDatasource>, Consumer<NumericalDatasource>> jb_x;
    private Triple<JPanel, Supplier<NumericalDatasource>, Consumer<NumericalDatasource>> jb_y;
    private Triple<JPanel, Supplier<NumericalDatasource>, Consumer<NumericalDatasource>> jb_col;

    public CreateXYScatterPlotActionGUI(CreateXYScatterPlotActionModel model, Frame owner) {
        //this.conf = conf;
        this.model = model;
        this.owner = owner;
        init();
    }



    private void init() {
        pi = new JPanel();
        pi.setLayout(new VerticalFlowLayout());
        jb_x   = JNumericalDataSourceSelector.getSelectorPanel2(model.getNTM(), owner);
        jb_y   = JNumericalDataSourceSelector.getSelectorPanel2(model.getNTM(), owner);
        jb_col = JNumericalDataSourceSelector.getSelectorPanel2(model.getNTM(), owner);
        //JButton jb_ok = new JButton("Create Plot");
        pi.add(jb_x.getLeft());
        pi.add(jb_y.getLeft());
        pi.add(jb_col.getLeft());
    }

    public void setConfig(CreateXYScatterPlotActionConfig conf) {
        //this.conf = conf;
        // todo implement
        if(conf.getNd_x()!=null) {
            jb_x.getRight().accept(conf.getNd_x());
        }
        if(conf.getNd_y()!=null) {
            jb_y.getRight().accept(conf.getNd_y());
        }
        if(conf.getNd_color()!=null) {
            jb_col.getRight().accept(conf.getNd_x());
        }
    }

    public CreateXYScatterPlotActionConfig getConfig(NexusTableModel ntm, JPanel plotpanel) {
        NumericalDatasource ndx = jb_x.getMiddle().get();
        NumericalDatasource ndy = jb_y.getMiddle().get();

        if(jb_col.getMiddle().get()!=null) {
            NumericalDatasource ndcolor = jb_col.getMiddle().get();
            return new CreateXYScatterPlotActionConfig(plotpanel, (NDataProvider) ntm.getDatasetForColumn(ndx.getColumn()), (NDataProvider) ntm.getDatasetForColumn(ndy.getColumn()), (NDataProvider) ntm.getDatasetForColumn(ndcolor.getColumn()) , ndx, ndy, ndcolor);
        }
        else {
            return new CreateXYScatterPlotActionConfig(plotpanel, (NDataProvider) ntm.getDatasetForColumn(ndx.getColumn()), (NDataProvider) ntm.getDatasetForColumn(ndy.getColumn()), ndx, ndy);
        }
    }

    public JPanel getActionGUI() {
        //pi.add(jb_ok);
        return pi;
    }
}
