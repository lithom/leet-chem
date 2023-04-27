package tech.molecules.leet.datatable.swing;

import tech.molecules.leet.datatable.DataTable;

import javax.swing.table.AbstractTableModel;
import java.util.List;

public class DefaultSwingTableModel {

    private DataTable table;
    private SwingTableModel swingModel;

    public DefaultSwingTableModel(DataTable table) {
        this.table = table;
        this.swingModel = new SwingTableModel();

        table.addDataTableListener(new DataTable.DataTableListener() {
            @Override
            public void tableStructureChanged() {
                swingModel.fireTableStructureChanged();
            }
            @Override
            public void tableDataChanged() {
                swingModel.fireTableDataChanged();
            }
            @Override
            public void tableCellsChanged(List<int[]> cells) {
                int[] allRows = cells.stream().mapToInt(ci -> ci[0]).distinct().toArray();
                for(int ri : allRows) {
                    swingModel.fireTableRowsUpdated(ri,ri);
                }
            }
        });
    }

    public DataTable.CellState getCellState(int row, int col) {
        return this.table.getCellState(row,col);
    }


    public class SwingTableModel extends AbstractTableModel {
        @Override
        public int getRowCount() {
            return table.getVisibleKeys().size();
        }

        @Override
        public int getColumnCount() {
            return table.getDataColumns().size();
        }

        @Override
        public Object getValueAt(int rowIndex, int columnIndex) {
            return table.getDataColumns().get(columnIndex).getValue(table.getVisibleKeys().get(rowIndex));
        }
    }

}
