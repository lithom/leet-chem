package tech.molecules.leet.table;

import javax.swing.*;
import javax.swing.table.TableCellEditor;
import javax.swing.table.TableCellRenderer;
import java.io.Serializable;
import java.util.BitSet;
import java.util.List;
import java.util.Map;

public interface NColumn<U,T> {


    public String getName();
    public T getData(U data, String rowid);
    //public TableCellRenderer getCellRenderer();
    public TableCellEditor getCellEditor();

    //public List<Pair<String,Class>> getAvailableDataSources();
    //public Object getDataSource(String name);
    public List<String> getNumericalDataSources();
    public double evaluateNumericalDataSource(U dp, String datasource, String rowid);

    public List<String> getRowFilterTypes();
    public NexusRowFilter<U> createRowFilter(NexusTableModel tableModel, String name);
    //public List<NexusRowFilter<U>> getRowFilters();
    //public void removeRowFilter(NexusRowFilter<U> filter);

    public interface NexusRowFilter<T> extends Serializable {
        public String getFilterName();
        public BitSet filterNexusRows(T data, List<String> ids, BitSet filtered);
        public double getApproximateFilterSpeed();
        public void setupFilter(NexusTableModel model);
        public JPanel getFilterGUI();
    }

    public void addColumnDataListener(ColumnDataListener cdl);
    public boolean removeColumnDataListener(ColumnDataListener cdl);

    public interface ColumnDataListener {
        public void needFiltering();
    }

}
