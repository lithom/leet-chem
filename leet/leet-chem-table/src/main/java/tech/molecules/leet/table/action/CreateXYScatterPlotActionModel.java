package tech.molecules.leet.table.action;

import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.table.NColumn;
import tech.molecules.leet.table.NDataProvider;
import tech.molecules.leet.table.NexusTableModel;
import tech.molecules.leet.table.NumericalDatasource;
import tech.molecules.leet.table.chart.JFreeChartScatterPlot;
import tech.molecules.leet.table.chart.JFreeChartScatterPlot2;
import tech.molecules.leet.table.chart.ScatterPlotModel;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class CreateXYScatterPlotActionModel {

    private NexusTableModel ntm;
    private CreateXYScatterPlotActionConfig conf;

    public CreateXYScatterPlotActionModel(NexusTableModel ntm, CreateXYScatterPlotActionConfig conf) {
        this.ntm  = ntm;
        this.conf = conf;
    }

    public void setConfig(CreateXYScatterPlotActionConfig conf) {
        this.conf = conf;
    }

    public CreateXYScatterPlotActionModel(NexusTableModel ntm, JPanel plotpanel) throws Exception {
        this.ntm = ntm;
        this.conf = this.inventConfig(plotpanel);
    }


    public NexusTableModel getNTM() {
        return this.ntm;
    }

    public CreateXYScatterPlotActionConfig getConfig() {
        return this.conf;
    }

    public void performAction() {
        //JLabel jla = new JLabel("TEST!!");
        this.conf.getPlotPanel().setLayout(new BorderLayout());
        //this.conf.getPlotPanel().add(jla);
        //JFreeChartScatterPlot jfcsp = new JFreeChartScatterPlot(this.ntm,this.conf.getDp_x(),this.conf.getDp_y(), this.conf.getNd_x(),this.conf.getNd_y());
        ScatterPlotModel spm = new ScatterPlotModel(this.ntm,this.conf.getDp_x(),this.conf.getDp_y(), this.conf.getNd_x(),this.conf.getNd_y());
        spm.setHighlightNNearestNeighbors(1);
        JFreeChartScatterPlot2 jfcsp = new JFreeChartScatterPlot2(spm);
        this.conf.getPlotPanel().add(jfcsp,BorderLayout.CENTER);
    }

    public CreateXYScatterPlotActionConfig inventConfig(JPanel plotpanel) throws Exception {
        Map<NColumn, Map<String,NumericalDatasource>> nds = ntm.collectNumericDataSources();
        // try to find two numds:
        List<NumericalDatasource> found = new ArrayList<>();
        for(NColumn ci : nds.keySet()) {
            found.addAll( ci.getNumericalDataSources().values() );
        }

        if (found.size() < 2) {
            throw new Exception("Not enough numerical datasources in dataset");
        }
        NumericalDatasource ndx = found.get(0);
        NColumn             ncx = found.get(0).getColumn();
        NumericalDatasource ndy = found.get(found.size()-1);
        NColumn             ncy = found.get(found.size()-1).getColumn();

        return new CreateXYScatterPlotActionConfig(plotpanel,(NDataProvider) ntm.getDatasetForColumn(ncx), (NDataProvider) ntm.getDatasetForColumn(ncy),ndx,ndy);
    }

}
