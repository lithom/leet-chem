package tech.molecules.leet.table;

import com.actelion.research.chem.StereoMolecule;

import javax.swing.*;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.TableCellEditor;
import java.util.*;

public class FixedNumericalDataColumn implements NColumn<NDataProvider.StructureDataProvider, Double> {

    private String name;
    private HashMap<String,Double> data = new HashMap<>();

    public FixedNumericalDataColumn(String name, Map<String,Double> data) {
        this.name = name;
        this.data = new HashMap<>(data);
    }

    @Override
    public String getName() {
        return name;
    }

    public HashMap<String,Double> getDataset() {
        return this.data;
    }

    @Override
    public void startAsyncInitialization(NexusTableModel model, NDataProvider.StructureDataProvider dataprovider) {

    }

    @Override
    public Double getData(NDataProvider.StructureDataProvider data, String rowid) {
        return this.data.get(rowid);
    }

    @Override
    public TableCellEditor getCellEditor() {
        return new DefaultNumericCellEditor(this);
    }

    @Override
    public Map<String,NumericalDatasource<NDataProvider.StructureDataProvider>> getNumericalDataSources() {
        //return Collections.singletonList(this.name);
        Map<String,NumericalDatasource<NDataProvider.StructureDataProvider>> dsm = new HashMap<>();
        FixedNumericalDataSource fnds = new FixedNumericalDataSource();
        dsm.put(fnds.getName(),fnds);
        return dsm;
    }

    public FixedNumericalDataColumn getThisColumn() {
        return this;
    }

    private class FixedNumericalDataSource implements NumericalDatasource<NDataProvider.StructureDataProvider> {
        @Override
        public String getName() {
            return name;
        }

        @Override
        public NColumn<NDataProvider.StructureDataProvider, ?> getColumn() {
            return getThisColumn();
        }

        @Override
        public boolean hasValue(NDataProvider.StructureDataProvider dp, String row) {
            return getThisColumn().data.containsKey(row);
        }

        @Override
        public double getValue(NDataProvider.StructureDataProvider dp, String row) {
            return getThisColumn().data.get(row);
        }
    }

    @Override
    public double evaluateNumericalDataSource(NDataProvider.StructureDataProvider dp, String datasource, String rowid) {
        Double di = this.data.get(rowid);
        if(di==null) {return Double.NaN;}
        return di;
    }

    @Override
    public List<String> getRowFilterTypes() {
        return new ArrayList<>();
    }

    @Override
    public NexusRowFilter<NDataProvider.StructureDataProvider> createRowFilter(NexusTableModel tableModel, String name) {
        return null;
    }

    @Override
    public void addColumnDataListener(ColumnDataListener cdl) {

    }

    @Override
    public boolean removeColumnDataListener(ColumnDataListener cdl) {
        return false;
    }
}
