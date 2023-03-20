package tech.molecules.leet.table;

import tech.molecules.leet.table.gui.JExtendedSlimRangeSlider;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.util.BitSet;
import java.util.List;

public class DefaultNumericRangeFilter<X,T> implements NColumn.NexusRowFilter<T> {

    private NexusTableModel model;
    private NColumn<X,T> col;
    private NumericalDatasource<X> nds;

    private Double rangeRestricted[] = null;
    private double rangeValue[] = new double[]{0.0,1.0};


    public DefaultNumericRangeFilter(NColumn<X,T> col, String numericDataSource) {
        this.col = col;
        this.nds = this.col.getNumericalDataSources().get(numericDataSource);
    }

    @Override
    public String getFilterName() {
        return this.nds.getName();
    }

    @Override
    public BitSet filterNexusRows(T data, List<String> ids, BitSet filtered) {
        BitSet filtered2 = (BitSet) filtered.clone();
        for(int zi=0;zi<ids.size();zi++) {
            if(!filtered2.get(zi)){continue;}
            double vi = nds.getValue(ids.get(zi));
            if(rangeRestricted!=null) {
                if(rangeRestricted[0]!=null && vi < rangeRestricted[0] ) { filtered2.set(zi); }
                if(rangeRestricted[1]!=null && vi > rangeRestricted[1] ) { filtered2.set(zi); }
            }
        }
        return filtered2;
    }

    @Override
    public double getApproximateFilterSpeed() {
        return 0.95;
    }

    @Override
    public void setupFilter(NexusTableModel model, T dp) {
        this.model = model;
        // determine range:
        double lb = Double.POSITIVE_INFINITY;
        double ub = Double.NEGATIVE_INFINITY;
        for(int zi=0;zi<model.getAllRows().size();zi++) {
            double vi = this.nds.getValue(model.getAllRows().get(zi));
            if(Double.isFinite(vi)) {
                lb = Math.min(vi, lb);
                ub = Math.max(vi, ub);
            }
        }
        this.rangeValue = new double[]{lb,ub};
    }

    @Override
    public JPanel getFilterGUI() {
        JPanel pi = new JPanel();
        JExtendedSlimRangeSlider jsl = new JExtendedSlimRangeSlider(nds.getName(),this.rangeValue);
        pi.setLayout(new BorderLayout());
        pi.add(jsl,BorderLayout.CENTER);
        jsl.getRangeSlider().addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {

                model.updateFiltering();
            }
        });
        return pi;
    }

    @Override
    public boolean isReady() {
        return true;
    }
}
