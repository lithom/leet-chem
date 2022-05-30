package tech.molecules.leet.table;

import com.actelion.research.gui.table.ChemistryCellRenderer;

import javax.swing.*;
import javax.swing.table.TableCellEditor;
import javax.swing.table.TableCellRenderer;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;

public class StructureColumn implements NColumn<NStructureDataProvider, NStructureDataProvider.StructureWithID> {

    @Override
    public String getName() {
        return "Structure";
    }

    @Override
    public NStructureDataProvider.StructureWithID getData(NStructureDataProvider data, String rowid) {
        return data.getStructureData(rowid);
    }

    @Override
    public List<String> getNumericalDataSources() {
        return new ArrayList<>();
    }

    @Override
    public double evaluateNumericalDataSource(NStructureDataProvider dp, String datasource, String rowid) {
        return Double.NaN;
    }

    //@Override
    public TableCellRenderer getCellRenderer() {
        return new StructureCellRenderer();
    }

    @Override
    public TableCellEditor getCellEditor() {
        return new NexusTable.DefaultEditorFromRenderer(this.getCellRenderer());
    }

    private List<ColumnDataListener> listeners = new ArrayList<>();

    @Override
    public void addColumnDataListener(ColumnDataListener cdl) {
        listeners.add(cdl);
    }

    @Override
    public boolean removeColumnDataListener(ColumnDataListener cdl) {
        return this.listeners.remove(cdl);
    }


    @Override
    public List<String> getRowFilterTypes() {
        return new ArrayList<>();
    }

    private List<NStructureDataProvider> providers = new ArrayList<>();

    @Override
    public NexusRowFilter<NStructureDataProvider> createRowFilter(NexusTableModel tableModel, String name) {
        return null;
    }

    public static class StructureCellRenderer extends ChemistryCellRenderer {
        @Override
        public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus, int row, int col) {
            String idc[] = ((NStructureDataProvider.StructureWithID)value).structure;
            return super.getTableCellRendererComponent(table, idc[0]+" "+idc[1], isSelected, hasFocus, row, col);
        }
    }

}
