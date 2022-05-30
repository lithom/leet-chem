package tech.molecules.leet.table.chem;

import tech.molecules.leet.table.NColumn;
import tech.molecules.leet.table.NexusTableModel;

import javax.swing.table.TableCellEditor;
import java.util.List;

public class NSynthonWithContextColumn implements NColumn<NSynthonWithContextDataProvider,String> {

    public static enum SYNTHON_COLUMN_MODE {SHOW, SHOW_ASSEMBLED};

    private SYNTHON_COLUMN_MODE mode = SYNTHON_COLUMN_MODE.SHOW;

    @Override
    public String getName() {
        return "Synthon";
    }

    //public void setMode(SYNTHON_COLUMN_MODE mode, StereoMolecule )

    @Override
    public String getData(NSynthonWithContextDataProvider data, String rowid) {
        return null;
    }

    @Override
    public TableCellEditor getCellEditor() {
        return null;
    }

    @Override
    public List<String> getNumericalDataSources() {
        return null;
    }

    @Override
    public double evaluateNumericalDataSource(NSynthonWithContextDataProvider dp, String datasource, String rowid) {
        return 0;
    }

    @Override
    public List<String> getRowFilterTypes() {
        return null;
    }

    @Override
    public NexusRowFilter<NSynthonWithContextDataProvider> createRowFilter(NexusTableModel tableModel, String name) {
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
