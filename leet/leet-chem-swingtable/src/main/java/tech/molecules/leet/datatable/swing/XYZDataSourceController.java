package tech.molecules.leet.datatable.swing;

import tech.molecules.leet.datatable.DataTable;
import tech.molecules.leet.datatable.NumericDatasource;
import tech.molecules.leet.datatable.chart.jfc.LeetXYZDataSet;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * Can be used to configure the LeetXYZDataset
 */
public class XYZDataSourceController extends JPanel {

    private DataTable table;
    private LeetXYZDataSet dataset;

    private JComboBox<WrappedNumericDatasource> comboBoxX;
    private JComboBox<WrappedNumericDatasource> comboBoxY;
    private JComboBox<WrappedNumericDatasource> comboBoxZ;


    public XYZDataSourceController(DataTable table) {
        this.table = table;
        reinitDatasetDefault();
        reinit();
    }

    private List<NumericDatasource> getAllNumericDatasources() {
        List<NumericDatasource> nds = (List<NumericDatasource>) this.table.getDataColumns().stream().flatMap(xi -> xi.getNumericDatasources().stream()).collect(Collectors.toList());
        return nds;
    }

    private void reinitDatasetDefault() {
        this.dataset = new LeetXYZDataSet(table);
        List<NumericDatasource> nds = getAllNumericDatasources();
        if(nds.size()>0) {
            Random ri = new Random();
            NumericDatasource ndx = nds.get(ri.nextInt(nds.size()));
            NumericDatasource ndy = nds.get(ri.nextInt(nds.size()));
            NumericDatasource ndz = nds.get(ri.nextInt(nds.size()));
            this.dataset.setDataSources(ndx,ndy,ndz,null);
        }
    }

    private class WrappedNumericDatasource {
        public final NumericDatasource nds;
        public WrappedNumericDatasource(NumericDatasource nds) {
            this.nds = nds;
        }
        public String toString() {
            return "colX"+":"+nds.getName();
        }
        public int hashCode() {
            return this.nds.hashCode();
        }
        public boolean equals(Object o) {
            if(o instanceof WrappedNumericDatasource) {
                return this.nds == ((WrappedNumericDatasource) o).nds;
            }
            return false;
        }
    }

    private List<WrappedNumericDatasource> getWrappedDatasources() {
        return this.getAllNumericDatasources().stream().map(xi -> new WrappedNumericDatasource(xi)).collect(Collectors.toList());
    }

    private void reinit() {
        this.removeAll();
        this.setLayout(new BorderLayout());
        this.setLayout(new FlowLayout(FlowLayout.LEFT));

        this.comboBoxX = new JComboBox(getWrappedDatasources().toArray());
        this.comboBoxY = new JComboBox(getWrappedDatasources().toArray());
        this.comboBoxZ = new JComboBox(getWrappedDatasources().toArray());

        if(this.dataset.getDataX()!=null) {
            this.comboBoxX.setSelectedItem(new WrappedNumericDatasource(this.dataset.getDataX()));
        }
        if(this.dataset.getDataY()!=null) {
            this.comboBoxY.setSelectedItem(new WrappedNumericDatasource(this.dataset.getDataY()));
        }
        if(this.dataset.getDataZ()!=null) {
            this.comboBoxZ.setSelectedItem(new WrappedNumericDatasource(this.dataset.getDataZ()));
        }

        JPanel jp_dx = new JPanel();
        jp_dx.setLayout(new GridLayout(1,3));
        jp_dx.add(this.comboBoxX);
        jp_dx.add(this.comboBoxY);
        jp_dx.add(this.comboBoxZ);
        this.add(jp_dx);

        ActionListener ali = new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                dataset.setDataSources(((WrappedNumericDatasource)comboBoxX.getSelectedItem()).nds,
                        ((WrappedNumericDatasource)comboBoxY.getSelectedItem()).nds,
                        ((WrappedNumericDatasource)comboBoxZ.getSelectedItem()).nds,
                        null
                        );
                fireChangeEvent();
            }
        };
        this.comboBoxX.addActionListener(ali);
        this.comboBoxY.addActionListener(ali);
        this.comboBoxZ.addActionListener(ali);

        this.revalidate();
        this.repaint();
    }

    private List<ChangeListener> listeners = new ArrayList<>();

    public void addChangeListener(ChangeListener li) {
        this.listeners.add(li);
    }

    public boolean removeChangeListener(ChangeListener li) {
        return this.listeners.remove(li);
    }

    private void fireChangeEvent() {
        for(ChangeListener li : listeners) {
            li.stateChanged(new ChangeEvent(this));
        }
    }

    public LeetXYZDataSet getDataset() {
        return this.dataset;
    }

}
