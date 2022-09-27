package tech.molecules.leet.table;

import javax.swing.table.TableCellEditor;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class StringColumn implements NColumn<NDataProvider.NStringDataProvider,String> {
    private String name;

    private NDataProvider.NStringDataProvider dp;
    public StringColumn(String name){
        this.name = name;
    }

    @Override
    public void setDataProvider(NDataProvider.NStringDataProvider dataprovider) {
        this.dp = dataprovider;
    }

    @Override
    public NDataProvider.NStringDataProvider getDataProvider() {
        return this.dp;
    }



    @Override
    public String getName() {
        return this.name;
    }

    @Override
    public String getData(String rowid) {
        return dp.getStringData(rowid);
    }

    @Override
    public TableCellEditor getCellEditor() {
        return new DefaultStringCellEditor(this);
    }

    @Override
    public Map<String, NumericalDatasource<NDataProvider.NStringDataProvider>> getNumericalDataSources() {
        return new HashMap<>();
    }

    @Override
    public void startAsyncReinitialization(NexusTableModel model) {

    }

    @Override
    public List<String> getRowFilterTypes() {
        return null;
    }

    @Override
    public NexusRowFilter<NDataProvider.NStringDataProvider> createRowFilter(NexusTableModel tableModel, String name) {
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
